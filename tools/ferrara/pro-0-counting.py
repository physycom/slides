#! /usr/bin/env python3

import os
import sys
import json
import numpy as np
import pandas as pd
import pymongo
from datetime import datetime, timedelta
from dateutil import tz

if __name__ == '__main__':
  import argparse
  import matplotlib.pyplot as plt

  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--input', type=str, required=True)
  parser.add_argument('-s', '--show', action='store_true')
  parser.add_argument('-fs', '--fine_sampling', type=int, default=600)
  parser.add_argument('-in', '--interpolation', choices=['lin', 'no'], default='lin')
  parser.add_argument('-a', '--aggr', choices=['rec', 'uniq'], default='uniq')

  args = parser.parse_args()
  fine_freq = f'{args.fine_sampling}s'
  filein = args.input
  base = filein[:filein.rfind('.')]
  if not os.path.exists(base): os.mkdir(base)
  tok = filein[:filein.rfind('.')].split('_')

  dt_fmt = '%Y%m%d-%H%M%S'
  try:
    start = datetime.strptime(tok[-2], dt_fmt)
    stop = datetime.strptime(tok[-1], dt_fmt)
  except:
    start = datetime.strptime(tok[-3], dt_fmt)
    stop = datetime.strptime(tok[-2], dt_fmt)

  try:
    df = pd.read_csv(filein, sep=';', usecols=['mac-address', 'date_time', 'station_name', 'kind'], parse_dates=['date_time'], index_col='date_time', engine='c')
    df = df.rename(columns={'mac-address':'mac_address'})
    old_format = True
  except:
    # new format support
    old_format = False
    df = pd.read_csv(filein, sep=';', usecols=['mac_address', 'date_time', 'station_name', 'kind'], parse_dates=['date_time'], index_col='date_time', engine='c')
  df['wday'] = [ t.strftime('%a') for t in df.index ]
  df['date'] = df.index.date
  df['time'] = df.index.time

  if 1 and old_format:
    print(f'**** WARNING FILTERING FERRARA STATIONS')
    df = df[ df.station_name.str.startswith('Ferrara-') ]

  try:
    df['station_id'] = df.station_name.str.split('-', expand=True)[1]
  except:
    #old format support
    df['station_id'] = df.station_name.str.extract(r'.*\((\d)\)')

  #print(df)
  print(df[['wday', 'date', 'station_name', 'station_id']])

  """
  Perform device id counting with fine temporal scale
  """
  tnow = datetime.now()
  stats = pd.DataFrame(index=pd.date_range("00:00", "23:59:59", freq=fine_freq).time)
  for (station, date), dfg in df.groupby(['station_id', 'date']):
    #print(station, date)

    if args.aggr == 'uniq':
      s = pd.Series(dfg['mac_address'], index=dfg.index)
      dfu = pd.DataFrame(s.groupby(pd.Grouper(freq=fine_freq)).value_counts())
      dfu.columns = ['repetitions_counter']
      dfu = dfu.reset_index()
      dfu = dfu.set_index('date_time')
      dfu = dfu.groupby('date_time')[['mac_address']].count()
      #dfu.columns = [f'{sid}_unique']
      if len(dfu) != len(stats):
        newidx = [ datetime(
          year=date.year,
          month=date.month,
          day=date.day,
          hour=t.hour,
          minute=t.minute,
          second=t.second
        ) for t in stats.index ]
        dfu = dfu.reindex(newidx)
      stats[(station, str(date))] = dfu.mac_address.values
      #print('dfu', dfu)
    elif args.aggr == 'rec':
      dfr = dfg[['mac_address']].resample(fine_freq).count() # convert to count_unique
      if len(dfr) != len(stats):
        newidx = [ datetime(
          year=date.year,
          month=date.month,
          day=date.day,
          hour=t.hour,
          minute=t.minute,
          second=t.second
        ) for t in stats.index ]
        dfr = dfr.reindex(newidx)
      #print(dfr)
      stats[(station, str(date))] = dfr.mac_address.values

  # fix null/empty/nan/missing values
  stats[ stats == 0 ] = np.nan
  #print(stats)
  stats = stats.reset_index()
  if args.interpolation == 'lin':
    stats = stats.interpolate(limit=10000, limit_direction='both')
  else:
    stats = stats.fillna(0)
  stats = stats.set_index('index')
  stats.index.name = 'time'
  tstats = datetime.now() - tnow
  print(f'Counting done {args.interpolation} stats in {tstats}\n', stats)
  stats.to_csv(f'{base}/counters_{fine_freq}_{args.interpolation}.csv', sep=';', index=True)

  """
  Perform total unique device id counting
  """
  tnow = datetime.now()
  tot_freq = '1800s'
  tots = pd.DataFrame(index=pd.date_range("00:00", "23:59:59", freq=tot_freq).time)
  n = 0
  """
  df = df[[ 'station_id', 'mac_address', 'date' ]].copy()
  print(df)
  print(df.columns)
  grp = df.groupby(['station_id', 'date', 'mac_address', pd.Grouper(freq='1800s')]).first()
  print(len(grp))
  #print(grp)
  #print(grp.groupby(['date_time']).count())
  df1 = pd.DataFrame(list(grp.index.values))
  print(df1)
  print(df1.index[0:3])
  """
  for (station, date), dfg in df.groupby(['station_id', 'date']):
    print(station, date)
    totv = [ dfg[ dfg.index.time < t ].groupby('mac_address').first().shape[0] for t in tots.index ]
    tots[(station, date.strftime('%Y-%m-%d'))] = totv
    n += 1
    #if n > 2: break
  #print(tots)
  tots.index.name = 'time'
  tots.to_csv(f'{base}/totals_{tot_freq}.csv', sep=';', index=True)
  ttot = datetime.now() - tnow
  print(f'Total counting done in {ttot}')
