#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import multiprocessing as mp
from multiprocessing import Pool
import matplotlib.pyplot as plt
from glob import glob

if __name__ == '__main__':
  # parse cli and config
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--csv', help='input average sniffer data regexp', required=True)
  parser.add_argument('-c1', '--csv1', help='second input average sniffer data file', default='')
  parser.add_argument('-s', '--show', help='display', action='store_true')
  parser.add_argument('-f', '--filter', nargs='+', help='list of filters', default=[])
  parser.add_argument('-tt', '--time_ticks', help='set time spacing between ticks', type=int, default=300)
  parser.add_argument('-tl', '--time_labels', help='set time spacing between ticks\' labels', type=int, default=3600)
  args = parser.parse_args()
  files = glob(args.csv)
  filters = args.filter

  if args.csv1 != '':
    df1 = pd.read_csv(args.csv1, sep=';', parse_dates=['time'], index_col='time')
  else:
    df1 = pd.DataFrame()

  for csvin in files:
    base = csvin[:csvin.rfind('.')]

    df = pd.read_csv(csvin, sep=';', parse_dates=['time'], index_col='time')
    df.index = df.index.time
    print(df)

    #print(df.index[1].strftime('%s'))
    #print(df.index[0].strftime('%s'))
    #print()
    freq = int(df.index[1].strftime('%s')) - int(df.index[0].strftime('%s'))
    dt_ticks = args.time_ticks
    if dt_ticks > freq:
      tus = dt_ticks // freq
    else:
      tus = 1
      dt_ticks = freq
    dt_lbls = args.time_labels
    if dt_lbls > dt_ticks:
      lus = dt_lbls // dt_ticks
    else:
      lus = 1
      dt_lbls = dt_ticks
    print(f'Data sampling {freq}. Ticks sampling {dt_ticks} u {tus}. Labels sampling {dt_lbls} u {lus}')

    if len(filters) == 0:
      tag = ''
      filters = df.columns
    else:
      tag = '_sel'

    # subplot grid auto-sizing
    totplot = len(filters)
    n = int(np.sqrt(totplot))
    rest = totplot - n**2
    row = n if rest == 0 else n + 1
    col = n if rest <= n else n + 1

    fig, axes = plt.subplots(nrows=row, ncols=col, figsize=(16, 12))
    axes = axes.flatten()
    for i, c in enumerate(filters):
      #print(c)
      now = datetime.now()
      ts = [ datetime(now.year, now.month, now.day, t.hour,t.minute,t.second).timestamp() for t in df.index ]

      unders = 6
      ts_ticks = ts[::tus]
      ts_lbl = [ t.strftime('%H:%M') for i, t in enumerate(df.index) ]
      ts_lbl = ts_lbl[::tus]
      ts_lbl = [ t if i%lus==0 else '' for i, t in enumerate(ts_lbl)]
      axes[i].plot(ts, df[c].values, 'b-o', label=c, markersize=4)

      if len(df1):
        ts1 = [ datetime(now.year, now.month, now.day, t.hour,t.minute,t.second).timestamp() for t in df1.index ]
        axes[i].plot(ts1, df1[c].values, 'r--', label=f'{c}_second', markersize=3)

      axes[i].set_xticks(ts_ticks)
      axes[i].set_xticklabels(ts_lbl, rotation=45)
      axes[i].grid()
      axes[i].legend()
      axes[i].set_xlabel(f'Times of day [HH:MM] Sampling {freq} s')
      axes[i].set_ylabel('Total Counter')

    plt.tight_layout()
    fig.subplots_adjust(top=0.97)
    plt.suptitle(f'Sniffer sensor {base}', y=0.99)
    if len(df1):
      outpng = f'{base}_second{tag}.png'
    else:
      outpng = f'{base}{tag}.png'
    plt.savefig(outpng)
    if args.show: plt.show()
