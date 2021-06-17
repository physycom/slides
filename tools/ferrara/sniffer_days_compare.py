#! /usr/bin/env python3

import os
import sys
import json
import numpy as np
import pandas as pd
from glob import glob
from enum import Enum
from dateutil import tz
from datetime import datetime, timedelta

map_station = {
  1:"Castello, Via Martiri", 2:"Hotel Carlton", 3:"Via del Podestà", 4:"Corso di P.Reno / Via Ragno" ,
  5:"Piazza Trento Trieste", 6:"Piazza Stazione"
}

if __name__ == '__main__':
  import argparse
  import matplotlib.pyplot as plt

  parser = argparse.ArgumentParser()
  parser.add_argument('-sh', '--show', action='store_true')
  parser.add_argument('-c', '--cfg', help='config file', required=True)
  parser.add_argument('-d', '--data', help='counters data csv', required=True)
  parser.add_argument('-tt', '--time_ticks', help='set time spacing between ticks', type=int, default=300)
  parser.add_argument('-tl', '--time_labels', help='set time spacing between ticks\' labels', type=int, default=3600)
  args = parser.parse_args()

  filein = args.data

  base = filein[:filein.rfind('/')]

  base_save = os.path.join(os.environ['WORKSPACE'], 'slides', 'work_lavoro', 'ferrara', 'data', 'compare_presence')
  if not os.path.exists(base_save): os.mkdir(base_save)

  fname = filein[filein.find('/')+1:filein.rfind('.')].split('_')
  fine_freq = fname[-2]
  fine_freq_s = int(fine_freq[:-1])
  interp = fname[-1]

  dt_ticks = args.time_ticks
  if dt_ticks > fine_freq_s:
    tus = dt_ticks // fine_freq_s
  else:
    tus = 1
    dt_ticks = fine_freq_s
  dt_lbls = args.time_labels
  if dt_lbls > dt_ticks:
    lus = dt_lbls // dt_ticks
  else:
    lus = 1
    dt_lbls = dt_ticks
  print(f'Data sampling {fine_freq_s}. Ticks sampling {dt_ticks} u {tus}. Labels sampling {dt_lbls} u {lus}')

  with open(args.cfg) as f:
    config = json.load(f)

  base_start_date = config['base_start_date']
  base_stop_date = config['base_stop_date']
  first_start_date = config['first_start_date']
  first_stop_date = config['first_stop_date']
  second_start_date = config['second_start_date']
  second_stop_date = config['second_stop_date']
  build_df = config['build']

  conf_name = args.cfg
  conf_name = conf_name[:conf_name.rfind('.')]

  b_start_date = f'{conf_name}_{base_start_date}_{base_stop_date}_wifi'
  b_first_date = f'{conf_name}_{first_start_date}_{first_stop_date}_wifi'
  b_second_date = f'{conf_name}_{second_start_date}_{second_stop_date}_wifi'

  file_base_date = f'{b_start_date}/counters_{fine_freq}_lin.csv'
  file_first_date = f'{b_first_date}/counters_{fine_freq}_lin.csv'
  file_second_date = f'{b_second_date}/counters_{fine_freq}_lin.csv'


  def box_centered_kernel(tot_len, box_len):
    pad_len = tot_len - box_len
    kern = np.concatenate([
      np.zeros((pad_len // 2)),
      np.ones((box_len)) / box_len,
      np.zeros((pad_len - pad_len // 2))
    ])
    return kern

  def building(filein):

    base = filein[:filein.rfind('/')]

    stats = pd.read_csv(filein, sep=';', parse_dates=['time'], index_col='time')

    stats.index = stats.index.time
    tuplecol = [ tuple(c.replace('\'', '').replace('(', '').replace(')','').replace(' ','').split(',')) for c in stats.columns ]
    stats.columns = tuplecol

    stats = stats.stack()

    stats.index = pd.MultiIndex.from_tuples([ (t, i[0], i[1]) for t, i in stats.index ], names=['time', 'station_id', 'date'])
    stats = stats.reset_index()
    cols = stats.columns.values
    cols[-1] = 'cnt'
    stats.columns = cols
    stats.date = pd.to_datetime(stats.date)

    stats['datatime'] = stats.apply(lambda r : pd.datetime.combine(r['date'],r['time']),1)
    stats.datatime = pd.to_datetime(stats.datatime)
    stats.index = stats.datatime
    stats = stats.sort_index()
    stats = stats.drop(['time', 'date', 'datatime'], axis=1)

    for sid, dfg in stats.groupby(['station_id']):
      dfg_smooth = dfg.copy()

      ma_size = 5
      kern = box_centered_kernel(len(dfg_smooth), ma_size)
      smooth = pd.DataFrame([], columns=dfg_smooth.columns, index=dfg_smooth.index)
      smooth.station_id = dfg.station_id
      for c in dfg_smooth.columns:
        if c != 'station_id':
          conv = np.fft.fftshift(np.real(np.fft.ifft( np.fft.fft( dfg_smooth[c].values ) * np.fft.fft(kern) )))
          smooth[c] = conv

      dfg['cnt_smooth'] = smooth.cnt

      dfg.to_csv(f'{base}/{base}_station_{sid}_{fine_freq_s}_smooth.csv', sep=';', index=True)

  if build_df:

    print(f'Building: {b_start_date}_station_{fine_freq_s}_smooth.csv\n')
    building(file_base_date)

    print(f'Building: {b_first_date}_station_{fine_freq_s}_smooth.csv\n')
    building(file_first_date)

    print(f'Building: {b_second_date}_station_{fine_freq_s}_smooth.csv\n')
    building(file_second_date)

  else:
    if base == b_start_date:
      for sid in range(1, 7):
        try:
          dfg =  pd.read_csv(f'{b_start_date}/{b_start_date}_station_{sid}_{fine_freq_s}_smooth.csv', sep=';', parse_dates=['datatime'], index_col='datatime')
          df_firstDay = pd.read_csv(f'{b_first_date}/{b_first_date}_station_{sid}_{fine_freq_s}_smooth.csv', sep=';', parse_dates=['datatime'], index_col='datatime')
          df_secondDay = pd.read_csv(f'{b_second_date}/{b_second_date}_station_{sid}_{fine_freq_s}_smooth.csv', sep=';', parse_dates=['datatime'], index_col='datatime')
        except:
          print(f'Station {sid} not found')
          continue

        dfg_diff = dfg.copy()

        dfg_diff['cntFirst'] = df_firstDay.cnt_smooth.to_list()
        dfg_diff['cntSecond'] = df_secondDay.cnt_smooth.to_list()

        diffBF = (dfg_diff.cnt_smooth - dfg_diff.cntFirst)
        diffBS = (dfg_diff.cnt_smooth - dfg_diff.cntSecond)

        ma_size = 5
        kernA = box_centered_kernel(len(diffBF), ma_size)
        kernB = box_centered_kernel(len(diffBS), ma_size)

        l2dA_smooth = np.fft.fftshift(np.real(np.fft.ifft( np.fft.fft( diffBF ) * np.fft.fft(kernA) )))
        l2dB_smooth = np.fft.fftshift(np.real(np.fft.ifft( np.fft.fft( diffBS ) * np.fft.fft(kernB) )))

        ts = [ t.timestamp() for t in dfg.index ]
        ts_ticks = ts[::tus]
        ts_lbl = [ t.strftime('%H:%M') for t in dfg.index ]
        ts_lbl = ts_lbl[::tus]
        ts_lbl = [ t if i%lus==0 else '' for i, t in enumerate(ts_lbl)]

        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(16, 10), sharex=True)

        axes = axs[0]
        axes.set_xticks(ts_ticks)
        axes.set_xticklabels(ts_lbl, rotation=45, ha='right')
        axes.grid()
        axes.plot(ts, dfg.cnt_smooth, 'b', label=dfg.index[0].strftime('%Y %b %d'))
        axes.plot(ts, dfg_diff.cntFirst, 'r', label=df_firstDay.index[0].strftime('%Y %b %d'))
        axes.plot(ts, dfg_diff.cntSecond, 'k', label=df_secondDay.index[0].strftime('%Y %b %d'))

        axes.legend()
        axes.set_ylabel('Counter')

        axes = axs[1]
        axes.set_xticks(ts_ticks)
        axes.set_xticklabels(ts_lbl, rotation=45, ha='right')
        axes.grid()

        axes.plot(ts, diffBF, 'r', label=f"Diff: \'{dfg.index[0].strftime('%y %b')} - \'{df_firstDay.index[0].strftime('%y %b')}")
        axes.plot(ts, diffBS, 'k', label=f"Diff: \'{dfg.index[0].strftime('%y %b')} - \'{df_secondDay.index[0].strftime('%y %b')}")

        axes.legend()
        axes.set_xlabel(f"Daytime on {dfg.index[0].strftime('%A')}")
        axes.set_ylabel('Difference')

        plt.suptitle(f'Comparison presences\n Station number {sid}: {map_station[sid]}')
        if args.show:
          plt.show()
        else:
          plt.savefig(f"{base_save}/comparison_presences_{fine_freq_s}_{sid}_{dfg.index[0].strftime('%y%a')}.png")

        plt.tight_layout()
        fig.subplots_adjust(top=0.95)

        plt.clf()

