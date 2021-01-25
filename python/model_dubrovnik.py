#! /usr/bin/env python3

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta
import pymongo
import mysql.connector

##########################
#### log function ########
##########################
def logs(s):
  head = '{} [mod_du] '.format(datetime.now().strftime('%y%m%d %H:%M:%S'))
  return head + s

def log_print(s, logger = None):
  if logger:
    logger.info(logs(s))
  else:
    print(logs(s), flush=True)

#############################
#### model ferrara class ####
#############################
class model_dubrovnik():

  def __init__(self, config, logger = None):
    self.logger = logger
    self.got_data = False
    self.date_format = '%Y-%m-%d %H:%M:%S'
    self.time_format = '%H:%M:%S'
    self.rates_dt = 10 * 60
    self.config = config
    self.station_map = {}
    self.data = pd.DataFrame()

    if 'station_mapping' in config:
      self.station_map = config['station_mapping']

    with open(os.path.join(os.environ['WORKSPACE'], 'slides', 'vars', 'extra', 'dubrovnik_sniffer.json')) as sin:
      self.st_info = json.load(sin)

  def full_table(self, start, stop, tag, resampling=None):
    if len(self.station_map) == 0:
      raise Exception(f'No station to generate')

    log_print(f'Generating model FE for {tag}', self.logger)

    if len(self.data) == 0:
      self.count_raw(start, stop)

    # retrieve data
    data = self.data
    #print(data)
    if tag in data:
      data = data[[tag]]
      #print(data[[tag]])
    else:
      raise Exception(f'[mod_fe] No station match for source {tag}')

    # upsample
    if resampling != None and resampling < self.rates_dt:
      dtot = data.sum()
      data = data.resample(f'{resampling}s').interpolate(direction='both')
      data = data * dtot / data.sum()

      resampling_min = resampling // 60
      start_date = start.replace(
        minute=resampling_min*(start.minute//resampling_min),
        second=0
      )
      stop_date = stop - timedelta(seconds=1)
      stop_date = stop_date.replace(
        minute=resampling_min*(stop_date.minute//resampling_min),
        second=0
      )
      fullt = pd.date_range(start_date, stop_date, freq=f'{resampling}s' )
      data = data.reindex(fullt).interpolate(direction='both')

    data = data[ (data.index >= start) & (data.index < stop) ]
    return data

  def count_raw(self, start, stop):
    """
    Perform device id counting with fine temporal scale
    """
    #log_print(f'Counting raw data', self.logger)

    df = self.get_data(start, stop)

    fine_freq = f'{self.rates_dt}s'

    df['wday'] = [ t.strftime('%a') for t in df.index ]
    df['date'] = df.index.date
    df['time'] = df.index.time
    #print(df)

    tnow = datetime.now()
    cnts = pd.DataFrame(index=pd.date_range(start, stop, freq=fine_freq))
    cnts = cnts.tz_localize('Europe/Rome')

    for station, dfg in df.groupby(['station_id']):
      print(station, len(dfg))
      s = pd.Series(dfg['device_uid'], index=dfg.index)
      dfu = pd.DataFrame(s.groupby(pd.Grouper(freq=fine_freq)).value_counts())
      dfu.columns = [station]
      dfu = dfu.reset_index()
      dfu = dfu.set_index('date_time')
      dfu = dfu.groupby('date_time')[['device_uid']].count()
      dfu.columns = [station]
      print(dfu)
      cnts[station] = np.nan
      mrg = cnts[[station]]
      print(mrg)
      mrg = pd.merge(mrg, dfu, left_index=True, right_index=True, how='left', suffixes=('_cnts', ''))
      print('merge\n', mrg)
      cnts[station] = mrg[f'{station}']

    # fix null/empty/nan/missing values
    cnts[ cnts == 0 ] = np.nan
    cnts = cnts.reset_index().interpolate(limit=10000, limit_direction='both').set_index('index')
    cnts.index.name = 'time'
    tcounting = datetime.now() - tnow
    log_print(f'Counting done in {tcounting}', self.logger)

    # convert to source/attractions naming convention and apply station-to-source mapping
    smap = self.station_map
    data = pd.DataFrame(index=cnts.index)
    for sid, names in smap.items():
      for name in names:
        data[name] = cnts[sid] / len(names)
    #print(data)

    self.data = data.astype(int)

  def get_data(self, start, stop):
    try:
      config = self.config['mysql']
      db = mysql.connector.connect(
        host     = config['host'],
        port     = config['port'],
        user     = config['user'],
        passwd   = config['pwd'],
        database = config['db']
      )
      cursor = db.cursor()

      id_list = sum(self.station_map.values(), [])
      start_utc = start.strftime(self.date_format)
      stop_utc = stop.strftime(self.date_format)
      print('CONVERT TO UTC REMINDER !!!!!')
      query = f"""
        SELECT 
          de.eventOccurredAt as date_time,
          de.id_device as station_id,
          de.eventClientiId as device_uid
        FROM 
          DubrovnikPma.DevicesEvents de
        WHERE
          (de.eventOccurredAt >= '{start_utc}' AND de.eventOccurredAt < '{stop_utc}')
          AND (de.id_device IN {tuple(id_list)} )
      """
      #print(query)

      tquery = datetime.now()
      cursor.execute(query)
      result = cursor.fetchall()
      tquery = datetime.now() - tquery
      log_print(f'Received {len(result)} mysql data in {tquery}', self.logger)
      if len(result) == 0:
        raise Exception(f'[mod_fe] Empty mysql query result')

      df1 = pd.DataFrame(result)
      df1.columns =  cursor.column_names
      df1.index = pd.to_datetime(df1.date_time)
      df1 = df1.tz_localize('utc')
      df = df1

      return df
    except Exception as e:
      raise Exception(f'[mod_fe] Query failed : {e}')

  def get_station_metadata(self):
    # mysql
    config = self.config['mysql']
    db = mysql.connector.connect(
      host     = config['host'],
      port     = config['port'],
      user     = config['user'],
      passwd   = config['pwd'],
      database = config['db']
    )
    cursor = db.cursor()

    query = f"""
      SELECT
        ds.id as id,
        ds.name as name,
        ds.serial as serial,
        ds.lat as lat,
        ds.lng as lon,
        ds.status as status
      FROM
        Devices ds
    """
    #print(query)

    tquery = datetime.now()
    cursor.execute(query)
    result = cursor.fetchall()
    tquery = datetime.now() - tquery
    log_print(f'Received {len(result)} mysql data in {tquery}', self.logger)
    if len(result) == 0:
      raise Exception(f'[mod_fe] Empty mysql query result')

    df1 = pd.DataFrame(result)
    df1.columns =  cursor.column_names

    return df1

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--cfg', help='config file', required=True)
  args = parser.parse_args()
  base = args.cfg
  base = base[:base.rfind('.')]

  with open(args.cfg) as f:
    config = json.load(f)

  mdu = model_dubrovnik(config)

  if 0:
    df = mdu.get_station_metadata()
    dfj = json.loads(df.to_json(orient="records"))
    with open('stations_geodata.json', 'w') as sout:
      json.dump(dfj, sout, indent=2, ensure_ascii=False)
    print(df)
    map_center = df[['lat', 'lon']].mean().values

    import folium
    m = folium.Map(location=map_center, control_scale=True, zoom_start=9)
    df.apply(lambda row: folium.CircleMarker(
      location=[row.lat, row.lon], 
      radius=7, 
      fill_color='red',
      color='red',
      popup=folium.Popup(f'<p><b>SNIFFER</b></br>id <b>{row.id}</b></br>serial <b>{row.serial}</b></p>', show=False, sticky=True),
    ).add_to(m), axis=1)
    s, w = df[['lat', 'lon']].min()
    n, e = df[['lat', 'lon']].max()
    m.fit_bounds([ [s,w], [n,e] ])
    m.save(f'station_map.html')

  start_date = '2021-01-21 10:00:00'
  stop_date = '2021-01-21 12:00:00'
  start = datetime.strptime(start_date, mdu.date_format)
  stop = datetime.strptime(stop_date, mdu.date_format)
  if 0:
    data = mdu.get_data(start, stop)
    print(data)

  if 1:
    cnt = mdu.count_raw(start, stop)
    print(cnt)