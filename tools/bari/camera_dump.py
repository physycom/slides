#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import pandas as pd
import mysql.connector
from datetime import datetime, timedelta

dir_SC = { #indicates for each camera which barrier detects flow direction from station (S) to center (N) as in first letter is S2 and second is S3
  '1':'WE', #s2 W s3 E
  '2':'NS',
  '3':'SN',
  '4':'SN',
  '5':'SN',
  '6':'SN',
  '7':'NS',
  '8':'SN',
  '9':'NS',
  '10':'NS',
  '11':'NS',
  '12':'NS',
  '13':'SN',
  '14':'SN',
  '15':'WE', #s2 W s3 E
  '16':'NS',
  '17':'SN',
  '18':'NS',
  '19':'SN',
  '20':'SN',
  '21':'NS',
  '22':'NS',
  '23':'EW' #s2 E s3 W
}

def dir_to_type(row): #converts dir_SC to a useful dfm column
  if row['BARRIER_NAME']=='S1':
    return 'A'
  if row['BARRIER_NAME']=='S2':
    return dir_SC[row['CAM_NAME']][0]
  if row['BARRIER_NAME']=='S3':
    return dir_SC[row['CAM_NAME']][1]


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--cfg', help='config file', default='conf_bari_camera.json')
  parser.add_argument('-tc', '--tc', help='time [H] for each chunk', default=2)

  args = parser.parse_args()

  base_save = os.path.join(os.environ['WORKSPACE'], 'slides', 'tools', 'bari')
  if not os.path.exists(base_save): os.mkdir(base_save)

  bari_cameras_path = os.path.join(os.environ['WORKSPACE'], 'slides', 'tools', 'bari')
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

  start_date = pd.to_datetime(config['start_date'])
  stop_date  = pd.to_datetime(config['stop_date'])

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
        m.CAM_NAME,
        m.BARRIER_NAME
      FROM
        barriers_meta m
      WHERE
        {camera_filter}
    """
    cursor.execute(query)
    result = cursor.fetchall()
  except Exception as e:
    print('Connection error : {}'.format(e))

  dfm = pd.DataFrame(result)
  dfm.columns =  cursor.column_names
  dfm['bartype'] = dfm.apply(dir_to_type,axis=1)
  dfm = dfm.rename(columns={'UID':'BARRIER_UID'})

  try:
    query = f"""
      SELECT
        c.DATETIME,
        c.BARRIER_UID,
        c.COUNTER
      FROM
        barriers_cnt c
      WHERE
        c.DATETIME > ('{start_date}') AND c.DATETIME < ('{stop_date}')
    """

    tquery = datetime.now()
    cursor.execute(query)
    result = cursor.fetchall()
    tquery = datetime.now() - tquery
    print(f'Received {len(result)} mysql data in {tquery}')
  except Exception as e:
    print('Connection error : {}'.format(e))

  df = pd.DataFrame(result)
  df.columns =  cursor.column_names
  dft = df.merge(dfm).drop(['BARRIER_UID','BARRIER_NAME'],axis=1)
  dfa = dft[dft['bartype']=='A'] # area countings
  dfb = dft[dft['bartype']!='A'] # barrier countings

  dfa = dfa.groupby([ pd.Grouper(freq = '15min', key='DATETIME'), 'CAM_NAME']).mean().rename(columns={'COUNTER':'area mean count'}).reset_index()
  dfb = dfb.groupby([ pd.Grouper(freq = '15min', key='DATETIME'), 'CAM_NAME', 'bartype']).sum().rename(columns={'COUNTER':'barrier count'}).reset_index()

  dfa.to_csv('dfarea.csv',index=False)
  dfb.to_csv('dfbars.csv',index=False)
