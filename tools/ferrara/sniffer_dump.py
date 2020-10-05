#! /usr/bin/env python3

import pymongo
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import json
import mysql.connector
from datetime import datetime

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--cfg', help='config file', required=True)
  parser.add_argument('-d', '--dev', help='filter wrt device type', choices=['both', 'wifi', 'bt'], default='both')
  parser.add_argument('-db', '--db', choices=['mongo', 'mysql'], default='mysql')
  args = parser.parse_args()
  base = args.cfg
  base = base[:base.rfind('.')]

  with open(args.cfg) as f:
    config = json.load(f)

  start_date = config['start_date']
  stop_date  = config['stop_date']

  start_tag = start_date.replace('-', '').replace(':', '').replace(' ', '-')
  stop_tag = stop_date.replace('-', '').replace(':', '').replace(' ', '-')

  print(f'Using {args.db} for {start_date} - {stop_date}')

  try:
    if args.db == 'mongo':
      config = config['mongo']
      client = pymongo.MongoClient(
        host=          config['host'],
        port=          config['port'],
        username=      config['user'],
        password=      config['pwd'],
        authSource=    config['db'],
        authMechanism= config['aut']
      )
      print(f'Authentication ok')

      if args.dev == 'both':
        cursor = client['symfony'].FerraraPma.find({
          'date_time' : {
            '$gte' : start_date,
            '$lt'  : stop_date
          }
        })
      else:
        cursor = client['symfony'].FerraraPma.find({
          'date_time' : {
            '$gte' : start_date,
            '$lt'  : stop_date
          },
          'kind' : args.dev
        })

      df = pd.DataFrame(list(cursor))
      print(f'Received {len(df)} data')

    elif args.db == 'mysql':
      config = config['mysql']
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
          ds.date_time as date_time,
          ds.id_device as mac_address,
          s.station_name as station_name
        FROM
          DevicesStations ds
        JOIN
          Stations s
        ON
          ds.id_station = s.id
        WHERE
          (ds.date_time >= '{start_date}' AND ds.date_time < '{stop_date}')
      """
      print(query)

      tquery = datetime.now()
      cursor.execute(query)
      result = cursor.fetchall()
      tquery = datetime.now() - tquery
      print(f'Received {len(result)} mysql data in {tquery}')

      df = pd.DataFrame(result)
      df.columns =  cursor.column_names

    out = f'{base}_{start_tag}_{stop_tag}_{args.dev}.csv'
    df.to_csv(out, sep=';', header=True, index=True)

  except Exception as e:
    print('Connection error : {}'.format(e))
