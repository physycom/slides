#! /usr/bin/env python3

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil import tz
from glob import glob
from enum import Enum

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

# congestion status
class cg():
  LOW  = 1
  AVE  = 2
  HIGH = 3

kpi_colors = {
  cg.LOW  : 'blue',
  #cg.AVE  : 'gray',
  cg.AVE  : 'white',
  cg.HIGH : 'red'
}

data_colors = {
  cg.LOW  : 'blue',
  #cg.AVE  : 'gray',
  cg.AVE  : 'white',
  cg.HIGH : 'red'
}

if __name__ == '__main__':
  import argparse
  import matplotlib.pyplot as plt

  parser = argparse.ArgumentParser()
  parser.add_argument('-d', '--data', help='counters data csv', required=True)
  parser.add_argument('-pc', '--plotconf', default='')
  parser.add_argument('-tt', '--time_ticks', help='set time spacing between ticks', type=int, default=300)
  parser.add_argument('-tl', '--time_labels', help='set time spacing between ticks\' labels', type=int, default=3600)
  args = parser.parse_args()

  filein = args.data
  base = filein[:filein.rfind('/')]
  tok = filein[:filein.find('/')].split('_')
  dt_fmt = '%Y%m%d-%H%M%S'
  try:
    start = datetime.strptime(tok[-2], dt_fmt)
    stop = datetime.strptime(tok[-1], dt_fmt)
  except:
    start = datetime.strptime(tok[-3], dt_fmt)
    stop = datetime.strptime(tok[-2], dt_fmt)

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

  stats = pd.read_csv(filein, sep=';', parse_dates=['time'], index_col='time')
  stats.index = stats.index.time
  print('stats\n', stats)
  tuplecol = [ tuple(c.replace('\'', '').replace('(', '').replace(')','').replace(' ','').split(',')) for c in stats.columns ]
  stats.columns = tuplecol

  """
  Groupby station_id and compute per day (or other criteria) mean signal.
  Perform moving average to remove fluctuations.
  """
  tnow = datetime.now()
  ave = stats.copy()
  ave = ave.stack()
  ave.index = pd.MultiIndex.from_tuples([ (t, i[0], i[1]) for t, i in ave.index ], names=['time', 'station_id', 'date'])
  ave = ave.reset_index()
  cols = ave.columns.values
  cols[-1] = 'cnt'
  ave.columns = cols
  #ave.station_id = ave.station_id.astype(int)
  print('ave\n', ave)
  ave.date = pd.to_datetime(ave.date)
  ave['wday'] = ave.date.dt.strftime('%a')
  #print(ave)
  dfave = ave.groupby(['station_id', 'wday', 'time']).mean()
  #print(dfave)
  smooths = {}
  for sid, dfg in dfave.groupby(['station_id']):
    try:
      dfp = dfg.unstack(level=1)
      dfp.index = pd.Index([ v[1] for v in dfp.index ], name='time')
      dfp.columns = [ v[1] for v in dfp.columns ]
      dfp = dfp[['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']]
      dfp['feriali'] = dfp[['Mon', 'Tue', 'Wed', 'Thu', 'Fri']].mean(axis=1)
      dfp['festivi'] = dfp[['Sat', 'Sun']].mean(axis=1)
      dfp = dfp.astype(int)
      dfp.to_csv(f'{base}/{sid}_{fine_freq}_{interp}.csv', sep=';', index=True)
    except Exception as e:
      print(f'Error with station {sid} : {e}')
      continue

    # convolve with normalized centered box
    def box_centered_kernel(tot_len, box_len):
      pad_len = tot_len - box_len
      kern = np.concatenate([
        np.zeros((pad_len // 2)),
        np.ones((box_len)) / box_len,
        np.zeros((pad_len - pad_len // 2))# for odd box_len
      ])
      return kern
    ma_size = 5 # running average idx interval from time in seconds
    kern = box_centered_kernel(len(dfp), ma_size)
    smooth = pd.DataFrame([], columns=dfp.columns, index=dfp.index)
    for c in dfp.columns:
      conv = np.fft.fftshift(np.real(np.fft.ifft( np.fft.fft( dfp[c].values ) * np.fft.fft(kern) )))
      smooth[c] = conv
    #print(smooth)
    smooth.index.name='time'
    smooth.to_csv(f'{base}/{sid}_{fine_freq}_{interp}_smooth.csv', sep=';', index=True)
    smooths[sid] = smooth
  tave = datetime.now() - tnow
  print(f'Averaging done in {tave} for {smooths.keys()}')

  """
  Rebuild full-time dataframe
  """
  datetime_fmt = '%Y-%m-%d %H:%M:%S'
  fullt = {}
  flustats = {}
  for s in smooths.keys():
    cols = [ c for c in stats.columns if c[0] == s ]
    dft = stats[cols].copy()
    dft = dft.stack()
    dft = dft.reset_index()
    dft.columns = ['time', 'date', 'cnt']
    dft['datetime'] = [ datetime.strptime(f'{d[1]} {t}', datetime_fmt) for t, d in dft[['time', 'date']].values ]
    dft = dft.sort_values(by=['datetime'])
    replicas = len(dft) // len(smooths[s])

    drange = pd.date_range(start, stop, freq='1d')[:-1] # only for stop = Y M D 00:00:00
    #print(drange)
    drange = [ d.strftime('%a') for d in drange ]
    ave_class = [ smooths[s][wdcat[d]].values for d in drange ]
    ave_day = [ smooths[s][d].values for d in drange ]
    #print(np.asarray(drange).shape)
    ave_cnt = np.concatenate(ave_class)
    ave_d_cnt = np.concatenate(ave_day)
    tidx = pd.date_range(start, stop, freq=fine_freq)[:-1] # only for stop = Y M D 00:00:
    dfave = pd.DataFrame(ave_cnt, index=tidx, columns=['ave_cnt'])
    dfave['ave_day_cnt'] = ave_d_cnt

    dfs = dft[['datetime', 'cnt']].set_index('datetime')
    dft = dfave.merge(dfs, left_index=True, right_index=True)

    kern = box_centered_kernel(len(dft), ma_size)
    dft['cnt_smooth'] = np.fft.fftshift(np.real(np.fft.ifft( np.fft.fft( dft.cnt ) * np.fft.fft(kern) )))

    # l2 diff
    l2diff = (dft.cnt - dft.ave_day_cnt)**2
    l2d_smooth = np.fft.fftshift(np.real(np.fft.ifft( np.fft.fft( l2diff ) * np.fft.fft(kern) )))
    l2d_ave = l2d_smooth.mean()
    l2d_std = l2d_smooth.std()
    l2d_thresh = l2d_ave + l2d_std
    l2d_cut = l2d_smooth.copy()
    l2d_cut[ l2d_cut < l2d_thresh ] = 0
    dft['l2_diff'] = l2diff
    dft['l2_diff_smooth'] = l2d_smooth
    dft['l2_diff_cut'] = l2d_cut

    # diff
    diff = dft.cnt - dft.ave_day_cnt
    diff_smooth = np.fft.fftshift(np.real(np.fft.ifft( np.fft.fft( diff ) * np.fft.fft(kern) )))
    l1d_ave = diff_smooth.mean()
    l1d_std = diff_smooth.std()
    l1d_thresh_up = l1d_ave + l1d_std
    l1d_thresh_down = l1d_ave - l1d_std
    #l1d_thresh = diff_smooth.copy()
    #l1d_thresh[ l1d_thresh < l1d_thresh | ] = 0
    dft['l1_diff'] = diff
    dft['l1_diff_smooth'] = diff_smooth
    #dft['l1_diff_thresh'] = l2d_thresh

    kpi = np.zeros(len(dft))
    kpi[ dft.l1_diff_smooth < l1d_thresh_down ] = cg.LOW
    kpi[ dft.l1_diff_smooth > l1d_thresh_down ] = cg.AVE
    kpi[ dft.l1_diff_smooth > l1d_thresh_up ]   = cg.HIGH
    dft['l1_kpi'] = kpi
    #print(kpi)

    fullt[s] = dft
    flustats[s] = {
      'l2_ave'      : l2d_ave,
      'l2_std'      : l2d_std,
      'l2_thr'      : l2d_thresh,
      'l1_ave'      : l1d_ave,
      'l1_std'      : l1d_std,
      'l1_thr_up'   : l1d_thresh_up,
      'l1_thr_down' : l1d_thresh_down,
    }
  #print(json.dumps(flustats, indent=2))

  """
  Fluctuations
  """
  for s, df in fullt.items():
    df = df[['l1_diff_smooth']].copy()
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(16, 10))
    for c in df.columns:
      count, bins = np.histogram(df[c], bins=50, density=True)
      bins = bins[:-1]
      w = bins[1] - bins[0]
      axes.bar(bins, count, width=w, label=c)
    plt.grid()
    plt.legend()
    axes.set_xlabel(f'Fluctuation from daily average [Number of devices]')
    axes.set_ylabel('Fraction')

    plt.tight_layout()
    fig.subplots_adjust(top=0.95)
    plt.suptitle(f'Statistics for station {s} fluctuations, data {base}, mean = {flustats[s]["l1_ave"]:.2f} stddev = {flustats[s]["l1_std"]:.2f}', y=0.98)
    plt.savefig(f'{base}/{s}_{fine_freq}_flustats.png')

  """
  Display scheduled result
  """
  if args.plotconf == '':
    selection = {
      s : {
        'start' : start,
        'stop' : stop
      }
    for s in smooths.keys() }
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
  for s in selection:
    dft = fullt[s]
    dft = dft[ (dft.index >= selection[s]['start']) & (dft.index < selection[s]['stop']) ]

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(16, 10))
    ts = [ t.timestamp() for t in dft.index ]
    ts_ticks = ts[::tus]
    ts_lbl = [ t.strftime('%a %d %H:%M') for t in dft.index ]
    ts_lbl = ts_lbl[::tus]
    ts_lbl = [ t if i%lus==0 else '' for i, t in enumerate(ts_lbl)]
    axes = axs[0]
#    axes.plot(ts, dft.cnt.values, '-o', label=s, markersize=4)
    axes.plot(ts, dft.cnt_smooth.values, 'y-o', label=f'Data', markersize=4)
#    axes.plot(ts, dft.ave_cnt.values, 'r--', label='ave')
    axes.plot(ts, dft.ave_day_cnt.values, 'b--', label='Daily average data')

    for t, kpi in zip(ts, dft['l1_kpi'].values): # special effects...SKADOUSH!!!
      axes.axvspan(t-0.5*fine_freq_s, t+0.5*fine_freq_s, facecolor=data_colors[kpi], alpha=0.3)

    axes.set_xticks(ts_ticks)
    axes.set_xticklabels(ts_lbl, rotation=45)
    axes.grid()
    axes.legend()
    axes.set_xlabel(f'Daytime Wday DD HH:MM (Ticks Sampling {dt_ticks} s)')
    axes.set_ylabel('Counter')

    axes = axs[1]
    l2d_thresh_up = flustats[s]['l1_thr_up']
    l2d_thresh_down = flustats[s]['l1_thr_down']
    #axes.plot(ts, dft.l2_diff.values, 'b-o', label=f'Station {s} l2_diff', markersize=4)
    axes.plot(ts, dft.l1_diff_smooth.values, 'g-o', label=f'Fluctuations', markersize=4)
    #axes.plot(ts, dft.l2_diff_thresh.values, 'g-o', label=f'Station {s} l2_diff', markersize=4)
    axes.axhspan(axes.get_ylim()[0], 1*l1d_thresh_down, facecolor=kpi_colors[cg.LOW] , alpha=0.3)
    axes.axhspan(l1d_thresh_down, l1d_thresh_up, facecolor=kpi_colors[cg.AVE] , alpha=0.3)
    axes.axhspan(l1d_thresh_up, axes.get_ylim()[1], facecolor=kpi_colors[cg.HIGH] , alpha=0.3)
    axes.set_xticks(ts_ticks)
    axes.set_xticklabels(ts_lbl, rotation=45)
    axes.grid()
    axes.legend()
    axes.set_xlabel('Day YY-MM-DD')
    axes.set_ylabel('Anomaly coefficient [au]')

    plt.tight_layout()
    fig.subplots_adjust(top=0.95)
    plt.suptitle(f'Station {s} localnorm analysis, data sampling {fine_freq}', y=0.98)
    plt.savefig(f'{base}/{s}_{fine_freq}_localnorm_{ptag}.png')