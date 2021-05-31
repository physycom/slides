#! /usr/bin/env python3

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil import tz
from glob import glob

if __name__ == '__main__':
  import argparse
  import matplotlib.pyplot as plt

  parser = argparse.ArgumentParser()
  parser.add_argument('-d', '--data', help='counters data csv', required=True)
  parser.add_argument('-pc', '--plotconf', default='')

  args = parser.parse_args()
  filein = args.data
  base = filein[:filein.rfind('/')]
  tok = filein[:filein.find('/')].split('_')
  fname = filein[filein.find('/')+1:filein.rfind('.')].split('_')
  fine_freq = fname[-2]

  dt_fmt = '%Y%m%d-%H%M%S'
  try:
    start = datetime.strptime(tok[-2], dt_fmt)
    stop = datetime.strptime(tok[-1], dt_fmt)
  except:
    start = datetime.strptime(tok[-3], dt_fmt)
    stop = datetime.strptime(tok[-2], dt_fmt)

  tots = pd.read_csv(filein, sep=';', parse_dates=['time'], index_col='time')
  tots.index = tots.index.time
  #print('tots\n', tots)
  tuplecol = [ tuple(c.replace('\'', '').replace('(', '').replace(')','').replace(' ','').split(',')) for c in tots.columns ]
  tots.columns = tuplecol

  """
  Groupby station_id and compute per day (or other criteria) mean signal.
  Perform moving average to remove fluctuations.
  """
  tnow = datetime.now()
  ave = tots.copy()
  ave = ave.stack()
  ave.index = pd.MultiIndex.from_tuples([ (t, i[0], i[1]) for t, i in ave.index ], names=['time', 'station_id', 'date'])
  ave = ave.reset_index()
  cols = ave.columns.values
  cols[-1] = 'tot'
  ave.columns = cols
  print('ave\n', ave)
  ave.date = pd.to_datetime(ave.date)
  ave['wday'] = ave.date.dt.strftime('%a')
  print(ave)
  dfave = ave.groupby(['station_id', 'wday', 'time']).mean()
  #print(dfave)
  totaves = {}
  for sid, dfg in dfave.groupby(['station_id']):
    try:
      dfp = dfg.unstack(level=1)
      dfp.index = pd.Index([ v[1] for v in dfp.index ], name='time')
      dfp.columns = [ v[1] for v in dfp.columns ]

      valid_days = [ d for d in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'] if d in dfp.columns ]
      valid_weekdays = [ d for d in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'] if d in dfp.columns ]
      valid_holidays = [ d for d in ['Sat', 'Sun'] if d in dfp.columns ]
      dfp = dfp[valid_days]
      dfp['feriali'] = dfp[valid_weekdays].mean(axis=1)
      dfp['festivi'] = dfp[valid_holidays].mean(axis=1)
      dfp = dfp.astype(int)
      dfp.to_csv(f'{base}/{sid}_{fine_freq}_ave.csv', sep=';', index=True)

      ma_width = 3 # scaled for 15min records
      dfsmooth = dfp.rolling(ma_width, min_periods=1, center=True, closed='both').mean()
      dfsmooth.to_csv(f'{base}/{sid}_{fine_freq}_smooth.csv', sep=';', index=True)

    except Exception as e:
      print(f'Error with station {sid} : {e}')
      continue
    totaves[sid] = (dfp, dfsmooth)
  tave = datetime.now() - tnow
  print(f'Averaging done in {tave} for {totaves.keys()}')

  """
  Display scheduled result
  """
  datetime_fmt = '%Y-%m-%d %H:%M:%S'
  if args.plotconf == '':
    selection = {
      s : {
        'start' : start,
        'stop' : stop
      }
    for s in totaves.keys() }
    ptag = 'full'
  else:
    with open(args.plotconf) as pcin:
      selection = json.load(pcin)
    selection = {
      s : {
        'start' : datetime.strptime(v['start'], datetime_fmt),
        'stop' : datetime.strptime(v['stop'], datetime_fmt)
      }
    for s, v in selection.items() }
    ptag = args.plotconf.split('_')[1].split('.')[0]
  wdclass = {
    'feriali' : [ 'Mon', 'Tue', 'Wed', 'Thu', 'Fri' ],
    'festivi' : [ 'Sat', 'Sun' ]
  }
  wdcat = {
    'Mon' : 'feriali',
    'Tue' : 'feriali',
    'Wed' : 'feriali',
    'Thu' : 'feriali',
    'Fri' : 'feriali',
    'Sat' : 'festivi',
    'Sun' : 'festivi'
  }
  for s in selection:
    cols = [ c for c in tots.columns if c[0] == s ]
    dft = tots[cols].copy()
    dft = dft.stack()
    dft = dft.reset_index()
    dft.columns = ['time', 'date', 'cnt']
    dft['datetime'] = [ datetime.strptime(f'{d[1]} {t}', datetime_fmt) for t, d in dft[['time', 'date']].values ]
    dft = dft.sort_values(by=['datetime'])

    try:
      replicas = len(dft) // len(totaves[s])
    except:
      print(f'Plot: Station {s} not available')
      continue
    #print('Replicas', replicas)

    dfaverage = totaves[s][0]
    dfsmooth = totaves[s][1]
    tidx = pd.date_range(f'{start}', f'{stop}', freq=fine_freq)[:-1] # only for stop = Y M D 00:00:
    drange = pd.date_range(f'{start}', f'{stop}', freq='1d')[:-1] # only for stop = Y M D 00:00:00
    #print(drange)
    drange = [ d.strftime('%a') for d in drange ]
    drange = [ d for d in drange if d in dfaverage.columns ]

    ave_class = [ dfaverage[wdcat[d]].values for d in drange ]
    ave_day = [ dfaverage[d].values for d in drange ]
    smooth_class = [ dfsmooth[wdcat[d]].values for d in drange ]
    smooth_day = [ dfsmooth[d].values for d in drange ]
    #print(np.asarray(drange).shape)
    ave_cnt = np.concatenate(ave_class)
    ave_d_cnt = np.concatenate(ave_day)
    smooth_cnt = np.concatenate(smooth_class)
    smooth_d_cnt = np.concatenate(smooth_day)

    dfave = pd.DataFrame(ave_cnt, index=tidx, columns=['ave_cnt'])
    dfave['ave_day_cnt'] = ave_d_cnt
    dfave['smooth_cnt'] = smooth_cnt
    dfave['smooth_day_cnt'] = smooth_d_cnt

    dfs = dft[['datetime', 'cnt']].set_index('datetime')
    dft = dfave.merge(dfs, left_index=True, right_index=True)
    dft = dft[ (dft.index >= selection[s]['start']) & (dft.index < selection[s]['stop']) ]
    s_start = dft.index.min()
    s_stop = dft.index.max()

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(16, 10))
    ts = [ t.timestamp() for t in dft.index ]
    unders = 3600 // int(fine_freq[:-1])
    ts_ticks = ts[::unders]
    ts_lbl = [ t.strftime('%a %d %H:%M') for t in dft.index ]
    ts_lbl = ts_lbl[::unders]
    ts_lbl = [ t if i%3==0 else '' for i, t in enumerate(ts_lbl)]
    axes = axs
    axes.plot(ts, dft['cnt'].values, 'r-o', label='Data', markersize=4)
#    axes.plot(ts, dft['ave_cnt'].values, 'r--', label='ave')
#    axes.plot(ts, dft['ave_cnt'].values, 'r--', label='ave')
    axes.plot(ts, dft['smooth_day_cnt'].values, 'g-', label='smooth_d')
    axes.plot(ts, dft['ave_day_cnt'].values, 'b--', label='ave_d')
    axes.set_xticks(ts_ticks)
    axes.set_xticklabels(ts_lbl, rotation=45)
    axes.grid()
    axes.legend()
    axes.set_xlabel('Daytime Wday DD HH:MM')
    axes.set_ylabel('Total unique devices')

    plt.tight_layout()
    fig.subplots_adjust(top=0.95)
    plt.suptitle(f'Station {s} daily data vs average values {s_start} {s_stop}', y=0.98)
    plt.savefig(f'{base}/{s}_{fine_freq}_avecompare_{ptag}.png')