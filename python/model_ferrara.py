#! /usr/bin/env python3

import os
import pymongo
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime

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
    self.station_map = config['station_mapping']
    self.data = pd.DataFrame()

    with open(os.path.join(os.environ['WORKSPACE'], 'slides', 'vars', 'extra', 'ferrara_sniffer.json')) as sin:
      self.st_info = json.load(sin)

  def full_table(self, start, stop, tag, resampling=None):
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

    # downsample
    if resampling != None and resampling < self.rates_dt:
      data = data.resample(f'{resampling}s').interpolate()

    return data

  def count_raw(self, start, stop):
    """
    Perform device id counting with fine temporal scale
    """
    log_print(f'Counting raw data', self.logger)

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
    log_print(f'Counting done in {tcounting}\n', self.logger)

    # convert to source/attractions naming convention
    smap = self.station_map
    data = pd.DataFrame(index=cnts.index)
    for sid, names in smap.items():
      for name in names:
        data[name] = cnts[sid] / len(names)
    #print(data)

    self.data = data.astype(int)

  def get_data_mongo(self, start, stop):
    try:
      config = self.config
      station_list = self.station_map.keys()

      client = pymongo.MongoClient(
        host          = config['host'],
        port          = config['port'],
        username      = config['user'],
        password      = config['pwd'],
        authSource    = config['db'],
        authMechanism = config['aut']
      )
      #print(f'Authentication ok')
      start_date = start.strftime(self.date_format)
      stop_date = stop.strftime(self.date_format)
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
      tquery = datetime.now() - tnow

      df = pd.DataFrame(list(cursor))
      df.index = pd.to_datetime(df.date_time)
      log_print(f'Received {len(df)} raw data in {tquery}', self.logger)

      return df
    except Exception as e:
      raise Exception(f'[mod_fe] Query failed : {e}')

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
    print(f'Authentication ok')

    start_date = config['start_date']
    stop_date  = config['stop_date']

    start_tag = start_date.replace('-', '').replace(':', '').replace(' ', '-')
    stop_tag = stop_date.replace('-', '').replace(':', '').replace(' ', '-')

    station_list = [ "1", "4" ]

    with open(os.path.join(os.environ['WORKSPACE'], 'slides', 'vars', 'extra', 'ferrara_sniffer.json')) as sin:
      stations_info = json.load(sin)

    db_filter = {
      'date_time' : {
        '$gte' : start_date,
        '$lt'  : stop_date
      },
      'kind' : 'wifi',
      'station_name' : { '$in' : [ stations_info[sid]['station_name'] for sid in station_list ] }
    }
    #print(json.dumps(db_filter, indent=2))
    cursor = client['symfony'].FerraraPma.find(db_filter)

    df = pd.DataFrame(list(cursor))
    print(f'Received {len(df)} data')
    out = f'{base}_{start_tag}_{stop_tag}.csv'
    df.to_csv(out, sep=';', header=True, index=True)

  except Exception as e:
    print('Connection error : {}'.format(e))
