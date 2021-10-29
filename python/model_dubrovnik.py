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
from collections import defaultdict

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
    self.cnt = pd.DataFrame()
    self.date_format = '%Y-%m-%d %H:%M:%S'
    self.time_format = '%H:%M:%S'
    self.rates_dt = 15 * 60
    self.config = config
    self.station_map = {}
    self.data = pd.DataFrame()

    with open(os.path.join(os.environ['WORKSPACE'], 'slides', 'vars', 'extra', 'dubrovnik_router.json')) as sin:
      self.st_info = json.load(sin)
    st_ser2id = { st['serial'] : st['id'] for st in self.st_info }

    self.station_map = config['station_mapping']
    self.station_mapid = { k : [ st_ser2id[si] for si in v ] for k,v in self.station_map.items() }
    #for k,v in self.station_map.items(): log_print(f'Source {k}\n{v}\n{self.station_mapid[k]}', self.logger)
    self.source_map = { i : k for k,v in self.station_map.items() for i in v }
    self.source_mapid = { i : k for k,v in self.station_mapid.items() for i in v }
    #for (k,v),(k1,v1) in zip(self.source_map.items(), self.source_mapid.items()): log_print(f'Sniffer {k} ({k1}) : {v} ({v==v1})', self.logger)

  def full_table(self, start, stop, tag, resampling=None):
    start = pd.to_datetime(start).tz_localize('Europe/Rome').tz_convert('utc')
    stop = pd.to_datetime(stop).tz_localize('Europe/Rome').tz_convert('utc')

    if not tag in self.station_map:
      raise Exception(f'Model DU tag {tag} not in sniffer-source map')

    if not tag in self.cnt.columns:
      data = self.count_raw(start, stop)
    else:
      data = self.cnt

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
    data = data.tz_convert('Europe/Rome').tz_localize(None)
    self.cnt = self.cnt.drop(columns=tag)
    return data[[tag]]

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
    df['source'] = df.station_id.apply(lambda x: self.source_mapid[x])
    #print(df)

    tnow = datetime.now()
    cnts = pd.DataFrame(index=pd.date_range(start, stop, freq=fine_freq))
    #cnts = cnts.tz_localize('Europe/Rome')

    for station, dfg in df.groupby(['source']):
      #print(station, len(dfg))
      s = pd.Series(dfg['device_uid'], index=dfg.index)
      dfu = pd.DataFrame(s.groupby(pd.Grouper(freq=fine_freq)).value_counts())
      dfu.columns = [station]
      dfu = dfu.reset_index()
      dfu = dfu.set_index('date_time')
      dfu = dfu.groupby('date_time')[['device_uid']].count()
      dfu.columns = [station]
      # print(dfu)
      cnts[station] = np.nan
      mrg = cnts[[station]]
      # print(mrg)
      mrg = pd.merge(mrg, dfu, left_index=True, right_index=True, how='left', suffixes=('_cnts', ''))
      # print('merge\n', mrg)
      cnts[station] = mrg[f'{station}']

    # fix null/empty/nan/missing values
    #cnts[ cnts == 0 ] = np.nan

    #cnts = cnts.interpolate(limit=10000, limit_direction='both')
    cnts = cnts.fillna(0)

    cnts.index.name = 'time'
    tcounting = datetime.now() - tnow
    log_print(f'Counting done in {tcounting}', self.logger)
    #print(cnts)

    self.cnt = cnts.astype(int)
    return self.cnt

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

      id_list = sum(self.station_mapid.values(), [])
      start_utc = start.strftime(self.date_format)
      stop_utc = stop.strftime(self.date_format)
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

  def map_station_to_source(self):
    stations = pd.DataFrame.from_dict(self.st_info)
    sourcemap = defaultdict(lambda: 'none')
    sourcemap.update(self.source_mapid)
    #stations = stations.set_index('serial').loc[ self.source_map.keys(), :].reset_index()
    stations['source'] = stations.id.apply(lambda x: sourcemap[x])
    stations['color'] = 'blue'
    stations.loc[stations.source == 'none', 'color'] = 'red'
    # stations['color'] = stations.serial.apply(lambda x: clustercol[clustermap[x]])
    print(stations)
    #stations = stations[stations.cluster != 'none']
    map_center = stations[['lat', 'lon']].mean().values

    simconf = os.path.join(os.environ['WORKSPACE'], 'slides', 'work_ws', 'output', 'wsconf_sim_dubrovnik.json')
    with open(simconf) as sin:
      sconf = json.load(sin)
    sources = pd.DataFrame.from_dict(sconf['sources']).transpose().dropna(subset=['source_location'])
    sources['name'] = sources.index.str.replace('_IN', '')
    sources.index = sources['name']
    sources['lat'] = sources.source_location.apply(lambda x: x['lat'])
    sources['lon'] = sources.source_location.apply(lambda x: x['lon'])
    sources['type'] = 'synth'
    sources.loc[ self.station_map.keys() , 'type'] = 'data'
    colors = { 'synth':'red', 'data':'blue'}
    sources['color'] = sources.type.apply(lambda t: colors[t])
    #sources[['lat', 'lon']] = sources.source_location.apply(lambda x: { 'lat':x['lat'], 'lon':x['lon'] })
    sources = sources[['lat', 'lon', 'name', 'color']]
    print(sources)
    print(sources.columns)

    import folium
    m = folium.Map(location=map_center, control_scale=True, zoom_start=9)
    stations.apply(lambda row: folium.CircleMarker(
      location=[row.lat, row.lon],
      radius=7,
      fill_color=f'{row.color}',
      color=f'{row.color}',
      popup=folium.Popup(f'<p><b>STATION</b></br>id <b>{row.id}</b></br>serial <b>{row.serial}</b></br>source <b>{row.source}</b></p>', show=False, sticky=True),
    ).add_to(m), axis=1)
    stations[ stations.source != 'none' ].apply(lambda row: folium.PolyLine(
      locations=[
        [ sources.loc[row.source, 'lat'], sources.loc[row.source, 'lon'] ],
        [ row.lat, row.lon ]
      ],
      color='black',
      weight=2,
    ).add_to(m), axis=1)
    sources.apply(lambda row: folium.CircleMarker(
      location=[row.lat, row.lon],
      radius=10,
      fill_color=f'{row.color}',
      color=f'{row.color}',
      fill_opacity=1.0,
      popup=folium.Popup(f'<p><b>SOURCE</b></br>id <b>{row.name}</b></p>', show=True, sticky=True),
    ).add_to(m), axis=1)
    s, w = stations.loc[ stations.lon > 0, ['lat', 'lon']].min()
    n, e = stations.loc[ stations.lon > 0, ['lat', 'lon']].max()
    m.fit_bounds([ [s,w], [n,e] ])
    m.save(f'map_sniffer2sources.html')

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
    m.save(f'map_router.html')


  start_date = '2021-01-22 00:00:00 CET'
  stop_date = '2021-01-23 00:00:00 CET'
  dt_fmt = '%Y-%m-%d %H:%M:%S %Z'
  start = pd.to_datetime(start_date, format=dt_fmt).tz_localize(None)
  stop = pd.to_datetime(stop_date, format=dt_fmt).tz_localize(None)
  if 0:
    data = mdu.get_data(start, stop)
    print(data)

  if 0:
    cnt = mdu.count_raw(start, stop)
    print(cnt)

  if 0:
    df = mdu.full_table(start, stop)
    print(df)

    w, h = 12, 10
    fig, ax = plt.subplots(1, 1, figsize=(w, h))
    plt.suptitle(f'Source timetables')

    for cid in df.columns:
      ax.plot(df.index, df[cid], '-', label=cid)

    ax.set_title(f'period {start} -> {stop}')
    ax.legend()
    ax.grid()
    ax.tick_params(labelrotation=45)

    plt.savefig(f'source_timetables.png')
    plt.close()

  if 1:
    mdu.map_station_to_source()