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
  parser.add_argument('-r', '--range', type=str, default='')
  args = parser.parse_args()
  base = args.input[:args.input.find('_')]
  dt_fmt = '%Y%m%d-%H%M%S'
  freq = f'{args.dt}s'
  if args.range == '':
    tok = args.input[:args.input.rfind('.')].split('_')
    try:
      start = datetime.strptime(tok[-2], dt_fmt)
      stop = datetime.strptime(tok[-1], dt_fmt)
    except:
      start = datetime.strptime(tok[-3], dt_fmt)
      stop = datetime.strptime(tok[-2], dt_fmt)
  else:
    start = datetime.strptime(args.range.split('|')[0], dt_fmt)
    stop = datetime.strptime(args.range.split('|')[1], dt_fmt)
    print(f'Setting time range {start} {stop}')
  base = f'{base}_{start.strftime(dt_fmt)}_{stop.strftime(dt_fmt)}'

  # read raw data
  df = pd.read_csv(args.input, sep=';')
  df.date_time = pd.to_datetime(df.date_time)
  df.date_time = df.date_time.dt.tz_localize(None)
  df = df[ (df.date_time >= start) & (df.date_time < stop) ]
  if 'kind' not in df.columns: df['kind'] = 'wifi'
  if '_id' not in df.columns: df['_id'] = 1
  print(df)

  """
  print(df[ df.station_name == 'Via del Podestà (3)'][['date_time', 'mac_address']].groupby('date_time').count().to_csv('3.csv'))
  print(df[ df.station_name == 'Piazza Stazione isola (6)'][['date_time', 'mac_address']].groupby('date_time').count().to_csv('6.csv'))
  print(df[ df.station_name == 'Via del Podestà (3)'][['date_time', 'kind', 'mac_address']].to_csv('3_raw.csv'))
  print(df[
    (df.station_name == 'Castello via martiri (1)') &
    (df.kind == 'wifi')
  ][['date_time', 'kind', 'mac_address']].to_csv('1_raw.csv'))
  """

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

  # plot
  w, h, d = 10, 10, 150
  plt.figure(figsize=(w, h), dpi=d)

  cnt = stats.values

  if args.bin:
    cnt[ cnt > 0 ] = 1
    color_map = plt.imshow(cnt.T, extent=[0, 1, 0, 1], interpolation='none', vmin=0, vmax=1)
  else:
    color_map = plt.imshow(cnt.T, extent=[0, 1, 0, 1], interpolation='none')

  #color_map = plt.imshow(cnt.T, extent=[0, 1, 0, 1], vmin = 0)
  color_map.set_cmap('plasma')

  plt.colorbar()
  plt.title(f'Sniffer data presence\nfor {start} - {stop} @ {args.dt} sec')

  dy = 1 / len(stats.columns)
  plt.gca().set_yticks([ (n + 0.5)*dy for n in range(len(stats.columns))])
  plt.gca().set_yticklabels(stats.columns[::-1])

  dx = 1 / len(stats.index)
  max_ticks = 20
  plt.gca().set_xticks([ (n + 0.5)*dx for n in range(len(stats.index))])
  lbls = [ t.strftime('%Y %b %d %H:%M') for t in stats.index]
  if len(lbls) > max_ticks:
    for i in range(len(lbls)):
      if i % (len(lbls) // max_ticks) != 0:
        lbls[i] = ''
  plt.gca().set_xticklabels(lbls, rotation=45, ha='right')

  plt.tight_layout()
  if args.show:
    plt.show()
  else:
    ptype = 'bin' if args.bin else 'cmap'
    plt.savefig(f'{base}_{ptype}_vali_{freq}.png')
  plt.close()

