#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

if __name__ == '__main__':
  # parse cli and config
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--input', type=str, required=True)
  parser.add_argument('-s', '--show', action='store_true')
  parser.add_argument('-b', '--bin', action='store_true')
  parser.add_argument('-t', '--dt', type=int, default=300)
  parser.add_argument('-e', '--thresh', type=int, default=100)
  parser.add_argument('-r', '--range', type=str, default='')
  parser.add_argument('-d', '--dev', type=str, default='wifi')
  parser.add_argument('-o', '--top', type=int, default=10)
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

  dft = df[['mac_address','date_time','station_name']]

  pid_l = []
  for pid, dfg in dft.groupby(['mac_address']):
    df_date = [ t.date() for t in dfg.date_time ]
    s = dfg.shape[0]
    days = len(np.unique(df_date))
    stat = len(np.unique(dfg.station_name))
    pid_l.append([pid,s,days,stat])

  pid_df = pd.DataFrame(pid_l, columns = ['pid','n','days','stations'])

  tsl = list(pid_df.n)
  tsl_2stations  = list(pid_df[pid_df.stations>1].n)
  tsl_2days = list(pid_df[pid_df.days>1].n)

  thresh = args.thresh
  tsl_t = [t for t in tsl if t > thresh]
  tsl_2dt = [t for t in tsl_2days if t > thresh]
  tsl_2st = [t for t in tsl_2stations if t > thresh]
  top_pids = top_pids = list(pid_df[pid_df.stations>1].sort_values(by='n',ascending=False)[:args.top].pid)

#%% plot 1
  w, h, d = 12, 7, 150
  fig, ax = plt.subplots(1, 1, figsize=(w, h), dpi=d)
  plt.yscale('log')
  plt.title(f'Device type {args.dev}, ts id count with at least {thresh} timestamps')
  plt.xlabel('Countings')
  plt.ylabel('Number of IDs')
  ax.hist(tsl_t, bins=30, alpha = 0.5, label = 'All data')
  ax.hist(tsl_2dt, bins=30, alpha = 0.5, color = 'r', range = [min(tsl_t),max(tsl_t)], label = 'Only 2 or more days presence')
  ax.hist(tsl_2st, bins=30, alpha = 0.3, color = 'b', range = [min(tsl_t),max(tsl_t)], label = 'Only 2 or more stations presence')
  plt.legend()
  # plt.plot(dftest.date_time,dftest.station_name, '.')

  if args.show:
    plt.show()
  else:
    plt.savefig(f'{base}_{thresh}_idcounts.png')

#%% plot 2
  w, h, d = 12, 7, 150
  fig, ax = plt.subplots(1, 1, figsize=(w, h), dpi=d)
  plt.title(f'Device type {args.dev}, top {args.top} ids with at least 2 stations presence movements')
  for p in top_pids:
    dft_p = dft[dft.mac_address==p]
    plt.plot(dft_p.date_time,dft_p.station_name,linewidth=1,alpha=0.6)
  plt.tight_layout()

  if args.show:
    plt.show()
  else:
    plt.savefig(f'{base}_{args.top}_movements.png')