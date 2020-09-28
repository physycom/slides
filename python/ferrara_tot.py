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
  """
  parser.add_argument('-s', '--show', action='store_true')
  parser.add_argument('-fs', '--fine_sampling', type=int, default=10)
  parser.add_argument('-os', '--out_sampling', type=int, default=300)
  parser.add_argument('-in', '--interpolation', choices=['lin', 'no'], default='lin')
  parser.add_argument('-a', '--aggr', choices=['rec', 'uniq'], default='uniq')
  """
  args = parser.parse_args()
  filein = args.data
  base = filein[:filein.rfind('/')]
  tok = filein[:filein.find('/')].split('_')
  start_date = tok[-2]
  stop_date = tok[-1]
  fname = filein[filein.find('/')+1:filein.rfind('.')].split('_')
  fine_freq = fname[-1]

  dt_fmt = '%Y%m%d-%H%M%S'
  start = datetime.strptime(start_date, dt_fmt)
  stop = datetime.strptime(stop_date, dt_fmt)

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
  #print(ave)
  dfave = ave.groupby(['station_id', 'wday', 'time']).mean()
  #print(dfave)
  totaves = {}
  for sid, dfg in dfave.groupby(['station_id']):
    try:
      dfp = dfg.unstack(level=1)
      dfp.index = pd.Index([ v[1] for v in dfp.index ], name='time')
      dfp.columns = [ v[1] for v in dfp.columns ]
      dfp = dfp[['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']]
      dfp['feriali'] = dfp[['Mon', 'Tue', 'Wed', 'Thu', 'Fri']].mean(axis=1)
      dfp['festivi'] = dfp[['Sat', 'Sun']].mean(axis=1)
      dfp = dfp.astype(int)
      dfp.to_csv(f'{base}/{sid}_{fine_freq}_totave.csv', sep=';', index=True)
    except Exception as e:
      print(f'Error with station {sid} : {e}')
      continue
    totaves[sid] = dfp
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
    ptag = 'pc'
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
    replicas = len(dft) // len(totaves[s])
    #print('Replicas', replicas)

    drange = pd.date_range(start, stop, freq='1d')[:-1] # only for stop = Y M D 00:00:00
    #print(drange)
    drange = [ d.strftime('%a') for d in drange ]
    ave_class = [ totaves[s][wdcat[d]].values for d in drange ]
    ave_day = [ totaves[s][d].values for d in drange ]
    #print(np.asarray(drange).shape)
    ave_cnt = np.concatenate(ave_class)
    ave_d_cnt = np.concatenate(ave_day)
    tidx = pd.date_range(start, stop, freq=fine_freq)[:-1] # only for stop = Y M D 00:00:
    dfave = pd.DataFrame(ave_cnt, index=tidx, columns=['ave_cnt'])
    dfave['ave_day_cnt'] = ave_d_cnt
    dfs = dft[['datetime', 'cnt']].set_index('datetime')
    dft = dfave.merge(dfs, left_index=True, right_index=True)

    dft = dft[ (dft.index >= selection[s]['start']) & (dft.index < selection[s]['stop']) ]

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
    axes.plot(ts, dft['ave_day_cnt'].values, 'b--', label='ave_d')
    axes.set_xticks(ts_ticks)
    axes.set_xticklabels(ts_lbl, rotation=45)
    axes.grid()
    axes.legend()
    axes.set_xlabel('Daytime Wday DD HH:MM')
    axes.set_ylabel('Total unique devices')

    """
    axes = axs[1]
    dftemp = ldf[ ldf.station_id == s ]
    print(dftemp)
    ts = [ datetime(t.year, t.month, t.day).timestamp() for t in dftemp.date ]
    range_start = selection[s]['start'].timestamp()
    range_stop = (selection[s]['stop'] - timedelta(days=1)).timestamp()
    ts_ticks = ts
    ts_lbl = [ t.strftime('%y-%m-%d') for t in dftemp.date ]
    #ts_lbl = ts_lbl[::unders]
    #ts_lbl = [ t if i%3==0 else '' for i, t in enumerate(ts_lbl)]
#    axes.plot(ts, dftemp['l2_norm'].values, 'r-o', label=f'Station {sid} ave', markersize=4)
    axes.plot(ts, dftemp['l2_norm_d'].values, 'b-o', label=f'Station {s} ave_d', markersize=4)
    axes.axvspan(range_start, range_stop, color='gray', alpha=0.3)
    axes.set_xticks(ts_ticks)
    axes.set_xticklabels(ts_lbl, rotation=45)
    axes.set_ylim(bottom=0)
    axes.grid()
    axes.legend()
    axes.set_xlabel('Day YY-MM-DD')
    axes.set_ylabel('Anomaly coefficient [au]')

    """
    plt.tight_layout()
    fig.subplots_adjust(top=0.95)
    plt.suptitle(f'Station {s} totals', y=0.98)
    plt.savefig(f'{base}/{s}_{fine_freq}_totcompare_{ptag}.png')