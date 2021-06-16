#! /usr/bin/env python3

import os
import sys
import json
import numpy as np
import pandas as pd
from glob import glob
from enum import Enum
from dateutil import tz
from matplotlib import pyplot as plt
from datetime import datetime, timedelta

map_station = {
  "Ferrara-1":"Castello, Via Martiri", "Ferrara-2":"Hotel Carlton", "Ferrara-3":"Via del Podestà", "Ferrara-4":"Corso di P.Reno / Via Ragno" ,
  "Ferrara-5":"Piazza Trento Trieste", "Ferrara-6":"Piazza Stazione"
}

if __name__ == '__main__':
  import argparse
  import matplotlib.pyplot as plt

  parser = argparse.ArgumentParser()
  parser.add_argument('-sh', '--show', action='store_true')
  parser.add_argument('-r', '--range', help='range', default='', type=str)
  parser.add_argument('-d', '--data', help='counters data csv', required=True)
  parser.add_argument('-s', '--station_name', help='station name', default='Ferrara-6', type=str)
  parser.add_argument('-f', '--frequency', help='sub-df frequency in minutes', default='60min', type=str)

  args = parser.parse_args()

  filein = args.data

  base_save = os.path.join(os.environ['WORKSPACE'], 'slides', 'work_lavoro', 'ferrara', 'data', 'stack_time')
  if not os.path.exists(base_save): os.mkdir(base_save)

  df = pd.read_csv(filein, sep=';', parse_dates=['date_time'], index_col='date_time')

  dt_fmt = '%Y%m%d-%H%M%S'

  dfList = [group[1] for group in df.groupby(df.index.day)]

  for i in range(0,len(dfList)):
    df = dfList[i]
    if args.range == '':
      start = dfList[i].index[0]
      stop = dfList[i].index[-1]
    else:
      start = datetime.strptime(args.range.split('|')[0], dt_fmt)
      stop = datetime.strptime(args.range.split('|')[1], dt_fmt)

    diff_h = (stop - start)
    diff_min = diff_h.total_seconds() / 60
    diff_time = diff_min / int(args.frequency[:2])

    height = int(args.frequency[:2]) * (9/10)

    df = df.loc[ (df.index >= start) & (df.index < stop) & (df['station_name'] == args.station_name)]

    if df.empty:
      print(f'Data range not in {i+1}st-day df')
      continue

    media_list = []
    len_col = []
    time = []

    for uselesstag, dfq in df.groupby(pd.Grouper(freq=args.frequency)):
      time.append(uselesstag.strftime('%H:%M'))
      lista = []

      for mc, dfg in dfq.groupby('mac_address'):
        time_delta = dfg.index[-1] - dfg.index[0]
        if time_delta >= timedelta(seconds=10) :
          time_delta = time_delta.total_seconds()
          lista.append([mc, time_delta])

      dfl = pd.DataFrame(lista)
      dfl.columns = ['mac_address', 'time_delta']
      len_col.append(len(dfl))

      media = dfl.time_delta.mean()
      media_min = round(media / 60)
      media_list.append(media_min)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))

    bins = np.arange(0, diff_time, 1)
    width = (bins[1] - bins[0])/2

    ax.bar(bins, media_list, align='center', width=width)
    ax.set_xticks(bins)
    ax.set_xticklabels(time, rotation=45, ha='right')             # TODO: labels every 15min are too dense if one takes a whole day
    ax.set_xlabel('Daytime [HH:MM]')
    ax.set_ylabel('Time [min]', color='blue')

    ymin, ymax = 0, height
    ax.set_ylim(ymin, ymax)
    ax.xaxis.grid(linewidth=1.5, alpha=0.5, linestyle='--')

    ax2 = ax.twinx()
    ax2.set_ylabel('Number of id', color='red')
    ax2.scatter(bins, len_col, color='red')

    plt.title(f'Stack time distribution - {map_station[args.station_name]}\n{df.index.date[0]}')

    if args.show :
      plt.show()
    else:
      start_h = time[0].replace(':','')
      stop_h = time[-1].replace(':','')
      day = str(df.index.date[0]).replace('-','')
      plt.savefig(f'{base_save}/stack_time_{day}-{start_h}_{stop_h}_{map_station[args.station_name]}_{args.frequency}.png')
    plt.clf()
