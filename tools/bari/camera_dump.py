#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import pandas as pd
import mysql.connector
from datetime import datetime, timedelta

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--cfg', help='config file', required=True)
  parser.add_argument('-db', '--db', choices=['mongo', 'mysql'], default='mysql')
  parser.add_argument('-tc', '--tc', help='time [H] for each chunk', default=2)

  args = parser.parse_args()

  base_save = os.path.join(os.environ['WORKSPACE'], 'slides', 'work_lavoro', 'bari')
  if not os.path.exists(base_save): os.mkdir(base_save)

  bari_cameras_path = os.path.join(os.environ['WORKSPACE'], 'slides', 'vars', 'extra')
  bari_json_file = f'{bari_cameras_path}/bari_cameras.json'

  with open(bari_json_file, encoding='utf-8') as f:
    bari_json = json.load(f)

  camera_map = []
  location_map = dict()
  for cam in bari_json:
    camera_map.append(cam['id'])
    location_map[cam['id']] = 'Camera ' + cam['name']  

  with open(args.cfg, encoding='utf-8') as f:
    config = json.load(f)

  start_date = config['start_date']
  stop_date  = config['stop_date']
  data_start_label = start_date.replace(':', '').replace('-', '').replace(' ', '_')
  data_stop_label = stop_date.replace(':', '').replace('-', '').replace(' ', '_')

  if args.db == 'mysql':
    try:
      print(f'Using {args.db} to get data from {start_date} to {stop_date}')
      conf = config['model_data']['params']['bari']['mysql']
      db = mysql.connector.connect(
        host     = conf['host'],
        port     = conf['port'],
        user     = conf['user'],
        passwd   = conf['pwd'],
        database = conf['db']
      )
      cursor = db.cursor()
      camera_filter = ' OR '.join([ f"m.CAM_NAME = '{name}'" for name in camera_map ])

      query = f"""
        SELECT
          m.UID,
          m.CAM_NAME
        FROM
          barriers_meta m
        WHERE
          {camera_filter}
      """
      cursor.execute(query)
      result = cursor.fetchall()
      camconv = { v[0] : v[1] for v in result }

      time_chunk = int(args.tc)
      tnow = pd.to_datetime(start_date)
      df_list = []
      while tnow < pd.to_datetime(stop_date):
        try:
          trange = tnow + timedelta(hours=time_chunk)
          query = f"""
            SELECT
              c.DATETIME,
              c.BARRIER_UID,
              c.COUNTER
            FROM
              barriers_cnt c
            WHERE
              c.DATETIME > ('{tnow}') AND c.DATETIME < ('{trange}')
              AND
              (BARRIER_UID in {tuple(camconv.keys())} )
          """
          tquery = datetime.now()
          cursor.execute(query)
          result = cursor.fetchall()
          tquery = datetime.now() - tquery
          print(f'Received {len(result)} mysql data in {tquery} for sub-query from {tnow} to {trange}')
          df = pd.DataFrame(result)
          df.columns =  cursor.column_names
          df_list.append(df)
        except Exception as e:
          print('Connection error : {}'.format(e))

        tnow = trange
      df = pd.concat(df_list)
      df = df.set_index('DATETIME')
      df['LOC'] = df['BARRIER_UID'].map(camconv)
      df = df.drop(columns='BARRIER_UID')
      df.to_csv(f'{base_save}/conf_{data_start_label}_{data_stop_label}.csv', sep = ';', index = True)
    except Exception as e:
      print('Connection error : {}'.format(e)) 