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
  args = parser.parse_args()
  base = args.input[:args.input.rfind('.')]
  tok = base.split('_')
  dt_fmt = '%Y%m%d-%H%M%S'
  start = datetime.strptime(tok[-2], dt_fmt)
  stop = datetime.strptime(tok[-1], dt_fmt)
  freq = f'{args.dt}s'

  # read raw data
  df = pd.read_csv(args.input, sep=';')
  print(df)

  stats = pd.DataFrame()
  t_index = pd.date_range(start=start, end=stop, freq=freq)
  for (sid, kind), dfg in df.groupby(['station_name', 'kind']):
    print(f'{kind}) {sid}  -> {dfg.shape}')

    dfg = dfg.set_index('date_time')
    dfg.index = pd.to_datetime(dfg.index)
    dfr = dfg[['_id']].resample(freq).count()
    dfr.columns = [f'{sid}_{kind}']

    if len(stats) == 0:
      stats = dfr
    else:
      stats = pd.concat([stats, dfr], axis=1).fillna(0)

  print('*** stats\n', stats)

  # plot
  w, h, d = 10, 10, 150
  plt.figure(figsize=(w, h), dpi=d)

  cnt = stats.values

  if args.bin:
    cnt[ cnt > 0 ] = 1
    color_map = plt.imshow(cnt.T, extent=[0, 1, 0, 1], vmin=0, vmax=1)
  else:
    color_map = plt.imshow(cnt.T, extent=[0, 1, 0, 1])

  #color_map = plt.imshow(cnt.T, extent=[0, 1, 0, 1], vmin = 0)
  color_map.set_cmap('plasma')

  plt.colorbar()
  plt.title(f'Sniffer data presence\nfor {start} - {stop} @ {args.dt} sec')

  dy = 1 / len(stats.columns)
  plt.gca().set_yticks([ (n + 0.5)*dy for n in range(len(stats.columns))])
  plt.gca().set_yticklabels(stats.columns)

  dx = 1 / len(stats.index)
  plt.gca().set_xticks([ (n + 0.5)*dx for n in range(len(stats.index))])
  lbls = [ t.strftime('%Y %b %d %H:%M') for t in stats.index]
  for i in range(len(lbls)):
    if i % 5 != 0:
      lbls[i] = ''
  plt.gca().set_xticklabels(lbls, rotation=45, ha='right')

  plt.tight_layout()
  if args.show:
    plt.show()
  else:
    ptype = 'bin' if args.bin else 'cmap'
    plt.savefig(f'{base}_{ptype}_validation.png')

