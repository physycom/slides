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
  head = '{} [mod_fe] '.format(datetime.now().strftime('%y%m%d %H:%M:%S'))
  return head + s

def log_print(s, logger = None):
  if logger:
    logger.info(logs(s))
  else:
    print(logs(s), flush=True)

#############################
#### model ferrara class ####
#############################
class model_ferrara():

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

    with open(os.path.join(os.environ['WORKSPACE'], 'slides', 'vars', 'extra', 'ferrara_sniffer.json')) as sin:
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

    df = self.get_data_mongo(start, stop)

    fine_freq = f'{self.rates_dt}s'

    df['wday'] = [ t.strftime('%a') for t in df.index ]
    df['date'] = df.index.date
    df['time'] = df.index.time
    df['station_id'] = df.station_name.str[-2:-1]
    #print(df)

    tnow = datetime.now()
    cnts = pd.DataFrame(index=pd.date_range(start, stop, freq=fine_freq))
    for station, dfg in df.groupby(['station_id']):
      s = pd.Series(dfg['mac_address'], index=dfg.index)
      dfu = pd.DataFrame(s.groupby(pd.Grouper(freq=fine_freq)).value_counts())
      dfu.columns = [station]
      dfu = dfu.reset_index()
      dfu = dfu.set_index('date_time')
      dfu = dfu.groupby('date_time')[['mac_address']].count()
      dfu.columns = [station]
      #print(dfu)
      cnts[station] = np.nan
      mrg = cnts[[station]]
      mrg = pd.merge(mrg, dfu, left_index=True, right_index=True, how='left', suffixes=('_cnts', ''))
      #print('merge\n', mrg)
      cnts[station] = mrg[station]

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

  def get_data_mongo(self, start, stop):
    try:
      config = self.config['mongo']
      station_list = self.station_map.keys()
      start_date = start.strftime(self.date_format)
      stop_date = stop.strftime(self.date_format)

      use_mongo = False
      if use_mongo:
        # mongo
        client = pymongo.MongoClient(
          host          = config['host'],
          port          = config['port'],
          username      = config['user'],
          password      = config['pwd'],
          authSource    = config['db'],
          authMechanism = config['aut']
        )
        #print(f'Authentication ok')

        tnow = datetime.now()
        db_filter = {
          'date_time' : {
            '$gte' : start_date,
            '$lt'  : stop_date
          },
          'kind' : 'wifi',
          'station_name' : { '$in' : [ self.st_info[sid]['station_name'] for sid in station_list ] }
        }
        db_fields = {
          'mac_address'  : 1,
          'date_time'    : 1,
          'station_name' : 1,
          '_id'          : 0
        }
        #print(json.dumps(db_filter, indent=2))
        cursor = client['symfony'].FerraraPma.find(db_filter, db_fields)
        df = pd.DataFrame(list(cursor))
        if len(df) == 0:
          raise Exception(f'[mod_fe] Empty mongo query result')

        df.index = pd.to_datetime(df.date_time)
        tquery = datetime.now() - tnow
        log_print(f'Received {len(df)} mongo data in {tquery}', self.logger)
        #print(df)
      else:
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

        # fetch mysql station id
        station_filter = ' OR '.join([ f"s.station_id = '{self.st_info[sid]['station_name']}'" for sid in station_list ])
        query = f"""
          SELECT
            s.id,
            s.station_id,
            s.address
          FROM
            Stations s
          WHERE
            {station_filter}
        """
        #print(query)
        cursor.execute(query)
        result = cursor.fetchall()
        #print(result)
        sidconv = { v[0] : v[1] for v in result }
        #print('sid', sidconv)

        query = f"""
          SELECT
            ds.date_time as date_time,
            ds.id_device as mac_address,
            ds.id_station as station_mysql_id
          FROM
            DevicesStations ds
          WHERE
            (ds.date_time >= '{start_date}' AND ds.date_time < '{stop_date}')
            AND
            (ds.id_station IN {tuple(station_list)} )
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
        df1 = df1.set_index('date_time')
        df1.index = pd.to_datetime(df1.index)
        df1['station_name'] = [ sidconv[n] for n in df1.station_mysql_id.values ]
        df1 = df1.drop(columns=['station_mysql_id'])
        df = df1

      return df
    except Exception as e:
      raise Exception(f'[mod_fe] Query failed : {e}')

  def map_station_to_source(self):
    stations = pd.DataFrame.from_dict(self.st_info).transpose()
    stations = stations.drop(columns=['cluster'])

    map_center = stations[['latitude', 'longitude']].mean().values
    sourcemap = defaultdict(lambda: 'none')
    sourcemap.update({ i : k for k, v in self.station_map.items() for i in v })
    # print(sourcemap)
    stationsmap = defaultdict(lambda: 'none')
    stationsmap.update({ v:k for k,v in sourcemap.items() })
    # print(stationsmap)
    stations['source'] = [ stationsmap[str(i)] for i in stations.index ]
    stations['color'] = [ 'blue' if i != 'none' else 'red' for i in stations.source ]
    # print(stations)

    simconf = os.path.join(os.environ['WORKSPACE'], 'slides', 'work_ws', 'output', 'wsconf_sim_ferrara.json')
    with open(simconf) as sin:
      sconf = json.load(sin)
    sources = pd.DataFrame.from_dict(sconf['sources']).transpose().dropna(subset=['source_location'])
    sources['name'] = sources.index.str.replace('_IN', '')
    sources.index = sources['name']
    sources['lat'] = sources.source_location.apply(lambda x: x['lat'])
    sources['lon'] = sources.source_location.apply(lambda x: x['lon'])
    sources['type'] = 'synth'
    sources.loc[ [ i for v in self.station_map.values() for i in v ], 'type'] = 'data'
    colors = { 'synth':'red', 'data':'blue'}
    sources['color'] = sources.type.apply(lambda t: colors[t])
    sources['station'] = sources.name.apply(lambda n: sourcemap[n] )
    #sources[['lat', 'lon']] = sources.source_location.apply(lambda x: { 'lat':x['lat'], 'lon':x['lon'] })
    sources = sources[['lat', 'lon', 'name', 'color', 'station']]
    # print(sources)
    # print(sources.columns)

    import folium
    m = folium.Map(location=map_center, control_scale=True, zoom_start=9)
    stations.apply(lambda row: folium.CircleMarker(
      location=[row.latitude, row.longitude],
      radius=7,
      fill_color=f'{row.color}',
      color=f'{row.color}',
      popup=folium.Popup(f'<p><b>STATION</b></br>id <b>{row.station_id}</b></br>name <b>{row.name}</b></br>source <b>{row.source}</b></p>', show=False, sticky=True),
    ).add_to(m), axis=1)
    sources[ sources.station != 'none' ].apply(lambda row: folium.PolyLine(
      locations=[
        [ stations.loc[row.station, 'latitude'], stations.loc[row.station, 'longitude'] ],
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
    s, w = stations[['latitude', 'longitude']].min()
    n, e = stations[['latitude', 'longitude']].max()
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

  mfe = model_ferrara(config)

  if 1:
    mfe.map_station_to_source()
