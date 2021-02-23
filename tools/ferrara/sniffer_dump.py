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
      table = config['table']
      print(f'Authentication ok')

      tquery = datetime.now()
      cursor = client['symfony'][table].aggregate([
        { 
          '$match' : {
            'date_time' : {
              '$gte' : start_date,
              '$lt'  : stop_date
            }
          }
        },
        { 
          '$group' : { 
            '_id' : "$station_id" 
          } 
        } 
      ])
      station_list = list(cursor)
      print(f'Retrieved stations : {station_list}')

      df = pd.DataFrame()
      for s in station_list:
        stid = s['_id']
        print(f'Query for station {stid}')
        tchunk = datetime.now()
        cursor = client['symfony'][table].find({
          'date_time' : {
            '$gte' : start_date,
            '$lt'  : stop_date
          },
          'station_id' : stid
        })

        dfi = pd.DataFrame(list(cursor))
        tchunk = datetime.now() - tchunk
        print(f'Data for station {stid} len {len(dfi)} in : {tchunk}')
        #print(dfi)
        df = df.append(dfi)
      
      print(df)
      tquery = datetime.now() - tquery
      print(f'Received {len(df)} data in {tquery}')

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

      # fetch mysql station id
      station_list = [
        'Corso di p. reno /via ragno (4)',
        'Castello via martiri (1)'
      ]
      station_list = [
        'Ferrara-1',
        'Ferrara-2',
        'Ferrara-3',
        'Ferrara-4',
        'Ferrara-5',
        'Ferrara-6'
      ]
      station_filter = ' OR '.join([ f"s.station_name = '{name}'" for name in station_list ])
      query = f"""
        SELECT
          s.id,
          s.station_name
        FROM
          Stations s
        WHERE
          {station_filter}
      """
      print(query)
      cursor.execute(query)
      result = cursor.fetchall()
      #print(result)
      sidconv = { v[0] : v[1] for v in result }
      #print('sid', sidconv)

      query = f"""
        SELECT
          ds.date_time as date_time,
          ds.id_device as mac_address,
          ds.id_station as station_mysql_id,
          'wifi' as kind
        FROM
          DevicesStations ds
        WHERE
          (ds.date_time >= '{start_date}' AND ds.date_time < '{stop_date}')
          AND
          (ds.id_station IN {tuple(sidconv.keys())} )
      """
      #print(query)
      #exit(1)

      tquery = datetime.now()
      cursor.execute(query)
      result = cursor.fetchall()
      tquery = datetime.now() - tquery
      print(f'Received {len(result)} mysql data in {tquery}')

      df = pd.DataFrame(result)
      df.columns =  cursor.column_names
      df['station_name'] = [ sidconv[n] for n in df.station_mysql_id.values ]
      df = df.drop(columns=['station_mysql_id'])
      print(df)

    out = f'{base}_{start_tag}_{stop_tag}_{args.dev}.csv'
    df.to_csv(out, sep=';', header=True, index=False)

  except Exception as e:
    print('Connection error : {}'.format(e))
