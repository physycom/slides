#! /usr/bin/env python3

import pymongo
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from bson import json_util, ObjectId
from pandas.io.json import json_normalize
import json

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--cfg', help='config file', required=True)
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

    start_date = config['start_date']
    stop_date  = config['stop_date']

    start_tag = start_date.replace('-', '').replace(':', '').replace(' ', '-')
    stop_tag = stop_date.replace('-', '').replace(':', '').replace(' ', '-')

    cursor = client['symfony'].FerraraPma.find({
      'date_time': {
        '$gte': start_date,
        '$lt': stop_date
      }
    })

    df = pd.DataFrame(list(cursor))
    print(df)
    out = f'{base}_{start_tag}_{stop_tag}.csv'
    df.to_csv(out, sep=';', header=True, index=True)

  except Exception as e:
    print('Connection error : {}'.format(e))
