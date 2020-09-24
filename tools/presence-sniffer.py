#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

plt.style.use('seaborn-dark-palette')

if __name__ == '__main__':
  # parse cli and config
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--input', type=str, required=True)
  parser.add_argument('-s', '--show', action='store_true')
  parser.add_argument('-b', '--bin', action='store_true')
  parser.add_argument('-t', '--dt', type=int, default=300)
  parser.add_argument('-r', '--range', type=str, default='')
  parser.add_argument('-d', '--dev', type=str, default='wifi')
  args = parser.parse_args()
  base = args.input[:args.input.find('_')]

  dt_fmt = '%Y%m%d-%H%M%S'
  freq = f'{args.dt}s'
  if args.range == '':
    tok = args.input[:args.input.rfind('.')].split('_')
    start = datetime.strptime(tok[-2], dt_fmt)
    stop = datetime.strptime(tok[-1], dt_fmt)
  else:
    start = datetime.strptime(args.range.split('|')[0], dt_fmt)
    stop = datetime.strptime(args.range.split('|')[1], dt_fmt)
  base = f'{base}_{start.strftime(dt_fmt)}_{stop.strftime(dt_fmt)}_{args.dev}'

  df = pd.read_csv(args.input, sep=';')
  df.date_time = pd.to_datetime(df.date_time)
  df = df[ (df.date_time >= start) & (df.date_time < stop) ]
  df = df[ df.kind == args.dev ]

  stats = pd.DataFrame()
  t_index = pd.date_range(start=start, end=stop, freq=freq)
  for sid, dfg in df.groupby(['station_name']):
    print(f'{args.dev}) {sid}  -> {dfg.shape}')

    dfg = dfg.set_index('date_time')
    dfg.index = pd.to_datetime(dfg.index)
    dfr = dfg[['mac_address']].resample(freq).count()
    dfr.columns = [f'{sid}']

    s = pd.Series(dfg['mac_address'], index=dfg.index)
    dfu = pd.DataFrame(s.groupby(pd.Grouper(freq=freq)).value_counts())
    dfu.columns = ['repetitions_counter']
    dfu = dfu.reset_index()
    dfu = dfu.set_index('date_time')
    dfu = dfu.groupby('date_time')[['mac_address']].count()
    dfu.columns = [f'{sid}_unique']
    #print('dfu', dfu)
    #idx = [ i for i in dfu.index ]
    #print('my',idx)
    #print(dfu.index)
    #dfu = pd.DataFrame(idx, columns=[f'{sid}_unique'], index=dfu.index)#, index=[ i for idfu.index.get_level_values(0)] )
    #print(dfu)

    if len(stats) == 0:
      stats = dfr
    else:
      stats = pd.concat([stats, dfr], axis=1)
    stats = pd.concat([stats, dfu], axis=1)

  stats = stats.fillna(0)

  # plot
  w, h, d = 12, 7, 150
  fig, ax = plt.subplots(1, 1, figsize=(w, h), dpi=d)
  plt.suptitle(f'Device type {args.dev}, unique device count')

  #print(stats.columns)
  for cid in stats.columns:
    ntag = cid[cid.rfind('('):cid.rfind('(')+3]
    lbl = cid[0:cid.rfind(')')+1]

    if not ntag in ['(1)', '(5)', '(6)']: continue
    if cid.endswith('_unique'):
      ax.plot(stats.index, stats[cid], '-', label=lbl)
    else:
      pass

  ax.set_title(f'period {start} -> {stop}, resampling {freq}')
  ax.legend()
  ax.grid()
  ax.tick_params(labelrotation=45)

  if args.show:
    plt.show()
  else:
    plt.savefig(f'{base}_presence_{freq}.png')
