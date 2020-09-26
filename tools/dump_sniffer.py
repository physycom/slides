#! /usr/bin/env python3

import pymongo
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import json

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--cfg', help='config file', required=True)
  parser.add_argument('-d', '--dev', help='filter wrt device type', choices=['both', 'wifi', 'bt'], default='both')
  args = parser.parse_args()
  base = args.cfg
  base = base[:base.rfind('.')]

  with open(args.cfg) as f:
    config = json.load(f)

  try:
    client = pymongo.MongoClient(
      host=          config['host'],
      port=          config['port'],
      username=      config['user'],
      password=      config['pwd'],
      authSource=    config['db'],
      authMechanism= config['aut']
    )
    print(f'Authenitcation ok')

    start_date = config['start_date']
    stop_date  = config['stop_date']

    start_tag = start_date.replace('-', '').replace(':', '').replace(' ', '-')
    stop_tag = stop_date.replace('-', '').replace(':', '').replace(' ', '-')

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
    out = f'{base}_{start_tag}_{stop_tag}_{args.dev}.csv'
    df.to_csv(out, sep=';', header=True, index=True)

  except Exception as e:
    print('Connection error : {}'.format(e))
