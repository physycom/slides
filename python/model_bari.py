#! /usr/bin/env python3

import os
import re
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
import logging
logger = logging.getLogger('mod_ba')

#############################
#### model bari class #######
#############################
class model_bari():

  def __init__(self, config):
    self.got_data = False
    self.date_format = '%Y-%m-%d %H:%M:%S'
    self.time_format = '%H:%M:%S'
    self.rates_dt = 15 * 60
    self.config = config
    self.camera_map = {}
    self.data = pd.DataFrame()

    if 'camera_mapping' in config:
      self.camera_map = config['camera_mapping']

    # df = pd.read_csv('sacro_match.csv', sep=';')
    # df = df.rename(columns={'Ai-Tech_Id': 'id', 'Latitude':'lat', 'Longitude':'lon'})
    # df['name'] = [ f'{i:02d}' for i in df.id ]
    # print(df)
    # with open('bari_cameras.json', 'w') as cout:
    #   json.dump(df.to_dict(orient='records'), cout, indent=2)

    with open(os.path.join(os.environ['WORKSPACE'], 'slides', 'vars', 'extra', 'bari_cameras.json')) as sin:
      self.cam_info = json.load(sin)

  def full_table(self, start, stop, tag, resampling=None):
    if len(self.camera_map) == 0:
      raise Exception(f'No station to generate')

    logger.info(f'Generating model BA for {tag}')

    if len(self.data) == 0:
      self.get_data_mysql(start, stop)

    # retrieve data
    alldata = self.data
    if tag in alldata:
      data = alldata[[tag]]
      #print(data[[tag]])
    else:
      raise Exception(f'No station match for source {tag}')

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

  def get_data_mysql(self, start, stop):
    try:
      camera_list = self.camera_map.keys()
      start_date = start.strftime(self.date_format)
      stop_date = stop.strftime(self.date_format)

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
      camera_list = sum(self.camera_map.values(), [])
      query = f"""
        SELECT
          bc.station_name as camera_id,
          bc.date_from as date_time,
          bc.sum_s2 + bc.sum_s3 as counter
        FROM
          DevicesStationsCollectedBarriers bc
        WHERE
          (bc.date_from >= '{start_date}' AND bc.date_from < '{stop_date}')
          AND
          (bc.station_name IN {tuple(camera_list)} )
      """
      #print(query)
      cursor.execute(query)
      result = cursor.fetchall()
      logger.info(f'Received {len(result)} mysql data')
      if len(result) == 0:
        raise Exception(f'empty mysql result')

      df1 = pd.DataFrame(result)
      df1.columns =  cursor.column_names
      df1.camera_id = df1.camera_id.interpolate().astype('int')
      df1.date_time = pd.to_datetime(df1.date_time)
      df1 = df1.set_index(['camera_id', 'date_time']).unstack(level=0)
      #print(df1)

      # merge cameras to sources
      data = pd.DataFrame(index=df1.index)
      for sid, names in self.camera_map.items():
        ilist = [ ('counter', n) for n in names ]
        data[sid] = df1[ilist].sum(axis=1)

      self.data = data
      return data
    except Exception as e:
      raise Exception(f'Query failed : {e}')

  def map_station_to_source(self):
    camera = pd.DataFrame.from_dict(self.cam_info).set_index('id', drop=False)
    #print(stations)

    map_center = camera[['lat', 'lon']].mean().values
    cameramap = defaultdict(lambda: 'none')
    cameramap.update({ i : k for k, v in self.camera_map.items() for i in v })
    #print(cameramap)
    camera['source'] = [ cameramap[i] for i in camera.id ]
    camera['color'] = [ 'blue' if i != 'none' else 'red' for i in camera.source ]
    #print(camera)

    #print(camera.groupby(['lat', 'lon']).agg({'name':'sum'}))

    simconf = os.path.join('conf_bari.json')
    with open(simconf) as sin:
      sconf = json.load(sin)
    sources = pd.DataFrame.from_dict(sconf['sources']).transpose().dropna(subset=['source_location'])
    sources['name'] = sources.index.str.replace('_IN', '')
    sources.index = sources['name']
    sources['lat'] = sources.source_location.apply(lambda x: x['lat'])
    sources['lon'] = sources.source_location.apply(lambda x: x['lon'])
    sources['type'] = 'synth'
    sources['color'] = 'blue'
    print(sources)

    import folium
    m = folium.Map(location=map_center, control_scale=True, zoom_start=9)
    camera.apply(lambda row: folium.CircleMarker(
      location=[row.lat, row.lon],
      radius=7,
      fill_color=f'{row.color}',
      color=f'{row.color}',
      popup=folium.Popup(f'<p><b>CAMERA</b></br>id <b>{row.id}</b></br>source <b>{row.source}</b></p>', min_width=300, max_width=300, show=False, sticky=True),
    ).add_to(m), axis=1)
    camera[ camera.source != 'none' ].apply(lambda row: folium.PolyLine(
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
      popup=folium.Popup(f'<p><b>SOURCE</b></br>id <b>{row.name}</b></p>', max_width=300, show=True, sticky=True),
    ).add_to(m), axis=1)
    s, w = camera[['lat', 'lon']].min()
    n, e = camera[['lat', 'lon']].max()
    m.fit_bounds([ [s,w], [n,e] ])
    m.save(f'map_cameras2sources.html')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--cfg', help='config file', required=True)
  args = parser.parse_args()
  base = args.cfg
  base = base[:base.rfind('.')]

  with open(args.cfg) as f:
    config = json.load(f)

  config = config['model_data']['params']['bari']
  mba = model_bari(config)

  if 1:
    mba.map_station_to_source()

  if 0:
    start = datetime.strptime(config['start_date'], mba.date_format)
    stop = datetime.strptime(config['stop_date'], mba.date_format)
    df = mba.full_table(start, stop, 'Stazione')
    print(df)

