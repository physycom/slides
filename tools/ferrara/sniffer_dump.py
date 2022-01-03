#! /usr/bin/env python3

from pandas.tseries.offsets import Hour
import pymongo
import requests
import json
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import json
import mysql.connector
from datetime import datetime, timedelta

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--cfg', help='config file', required=True)
  parser.add_argument('-d', '--dev', help='filter wrt device type', choices=['both', 'wifi', 'bt'], default='both')
  parser.add_argument('-db', '--db', choices=['mongo', 'mysql','api'], default='mysql')
  parser.add_argument('-tc', '--tc', help='time [H] for each chunk', default=2)

  args = parser.parse_args()
  base = args.cfg
  base = base[:base.rfind('.')]

  with open(args.cfg, encoding='utf-8') as f:
    config = json.load(f)

  start_date = config['start_date']
  stop_date  = config['stop_date']

  start_tag = start_date.replace('-', '').replace(':', '').replace(' ', '-')
  stop_tag = stop_date.replace('-', '').replace(':', '').replace(' ', '-')

  config = config['model_data']['params']['ferrara']

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
        authMechanism= config['aut'],
        #compressors=   'snappy',
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
      # print(query)
      cursor.execute(query)
      result = cursor.fetchall()
      #print(result)
      sidconv = { v[0] : v[1] for v in result }
      #print('sid', sidconv)

      time_chunk = int(args.tc)
      start = pd.to_datetime(start_date)
      stop = pd.to_datetime(stop_date)
      tnow = start
      df_list = []
      df_all = pd.DataFrame(columns=['date_time', 'mac_address', 'kind', 'station_name'])
      while tnow < stop:
        try:
          trange = tnow + timedelta(hours=time_chunk)
          query = f"""
            SELECT
              ds.date_time as date_time,
              ds.id_device as mac_address,
              ds.id_station as station_mysql_id,
              'wifi' as kind
            FROM
              DevicesStations ds
            WHERE
              (ds.date_time >= '{tnow}' AND ds.date_time < '{trange}')
              AND
              (ds.id_station IN {tuple(sidconv.keys())} )
          """
          # print(query)
          tquery = datetime.now()
          cursor.execute(query)
          result = cursor.fetchall()
          tquery = datetime.now() - tquery
          print(f'Received {len(result)} mysql data in {tquery} for sub-query from {tnow} to {trange}')
          df = pd.DataFrame(result)
          df.columns =  cursor.column_names
          df['station_name'] = [ sidconv[n] for n in df.station_mysql_id.values ]
          df = df.drop(columns=['station_mysql_id'])
          df_list.append(df)
        except Exception as e:
          print('Connection error : {}'.format(e))

        tnow = trange      
      df_all = pd.concat(df_list)
      out = f'{base}_{start_tag}_{stop_tag}_{args.dev}.csv'
      df_all.to_csv(out, sep=';', header=True, index=False)    

    elif args.db == 'api':
      import urllib3
      urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

      base = base.split('/')[-1]
      outdir = os.path.join(os.environ['WORKSPACE'], 'slides', 'work_serra')
      if not os.path.exists(outdir): os.mkdir(outdir)
      url='https://openplatcol.m-iotechnology.it/snifferplatform/api_v1/timeSeries/'
      dict_station = {
        'adc5794e-fc0d-4d3b-bbde-8ada05f88b5d' : 'Ferrara-1',
        '4f875711-c315-47cc-8f8e-0293044d790e' : 'Ferrara-2',
        'ef3232e8-b559-46f8-ade4-74855a8bf9e4' : 'Ferrara-3',
        '5c8ff276-f0b2-476d-8690-5174db4a1aec' : 'Ferrara-4',
        '053f4694-9c96-4d54-8059-693603a615dc' : 'Ferrara-5',
        '9227deaf-bc21-4133-811a-62d695df3327' : 'Ferrara-6'
      }
      dict_inv = {i:j for j,i in dict_station.items()}

      time_chunk = int(args.tc)
      start = pd.to_datetime(start_date)
      stop = pd.to_datetime(stop_date)
      tnow = start
      df_list = []
      df_all = pd.DataFrame(columns=["_id","date_time","mac-address","data_type","station_id"])
      while tnow < stop:
        try:
          trange = tnow + timedelta(hours=time_chunk)
          start_date = pd.to_datetime(tnow, format='%Y-%m-%d %H:%M:%S').strftime('%d/%m/%Y_%H:%M')
          stop_date = pd.to_datetime(trange, format='%Y-%m-%d %H:%M:%S').strftime('%d/%m/%Y_%H:%M')
  
          data = {
              "apikey": "F0T+w/RZrYHpKoXW/I+krQ==",
              # "station_id": f'{str(dict_inv["Ferrara-1"])}',
              "start_dt": f"{start_date}",
              "stop_dt": f"{stop_date}"
          }
          header={
              'Content-Type': 'application/json',
          }
          r=requests.post(url, data=json.dumps(data), verify=False, headers=header)
          
          print(f'Data received by sub-query: from {tnow} to {trange}')
          df = pd.DataFrame.from_dict(r.json())
          col_list = ["_id","date_time","mac-address","data_type","station_id"]
          df = df[col_list]
          df['station_name'] = df['station_id'].map(dict_station)
          df = df.drop(columns=['station_id', '_id'])
          df = df.rename(columns={"data_type": "kind"})
          df['date_time'] = pd.to_datetime(df.date_time, format='%d/%m/%Y_%H:%M')
          df = df.sort_values(by="date_time")
          df = df.set_index('date_time')
          df_list.append(df)
        except Exception as e:
          print('Connection error : {}'.format(e))

        tnow = trange      
      df_all = pd.concat(df_list)

      out = f'{outdir}/{base}_{start_tag}_{stop_tag}_{args.dev}.csv'
      df_all.to_csv(out, sep=';', header=True, index=True)      
      print(f'Data saved for station on => {out}')      

