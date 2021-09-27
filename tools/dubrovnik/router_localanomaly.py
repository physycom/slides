#! /usr/bin/env python3

import os
import json
import argparse
import numpy as np
from numpy.core.numeric import full
import pandas as pd
import mysql.connector
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.stats import linregress


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

kpi_thresh = {
  cg.LOW  : 0.2,
  #cg.AVE  : 'gray',
  cg.AVE  : 0,
  cg.HIGH : 0.8
}

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


def localsnomalys(data):

  dubro = os.path.join(os.environ['WORKSPACE'], 'slides', 'work_lavoro', 'dubrovnik')
  if not os.path.exists(dubro): os.mkdir(dubro)
  base_save = f'{dubro}/router_validate'
  if not os.path.exists(base_save): os.mkdir(base_save)

  df = pd.read_csv(data, sep =';', parse_dates=True, index_col=[0])

  tok = data[:data.find('.')].split('_')
  start = tok[-4] + "-" + tok[-3]
  stop = tok[-2] + "-" + tok[-1]

  dt_fmt = '%Y%m%d-%H%M%S'
  try:
    start = datetime.strptime(start , dt_fmt)
    stop = datetime.strptime(stop, dt_fmt)
  except:
    print(f'Data format {dt_fmt} doesn\'t match')

  fine_freq_s = int(freq[:-1])

  df.index = pd.to_datetime(df.time).rename('date_time')
  df = df.drop(columns = 'time')
  data_i = start
  data_f = stop
  start_time = pd.to_datetime(data_i)
  end_time = pd.to_datetime(data_f)
  time_index = pd.date_range(start = start_time, end = end_time, freq = freq)

  stats = pd.DataFrame()
  df_list = [122, 124, 128, 130, 127, 140, 152, 151, 157, 148, 132, 135, 133, 131, 136, 137, 138]
  df = df.loc[df['device'].isin(df_list)]

  fine_freq = freq
  df['wday'] = [ t.strftime('%a') for t in df.index ]
  df['date'] = df.index.date
  df['time'] = df.index.time

  tnow = datetime.now()
  stats = pd.DataFrame(index=pd.date_range("00:00", "23:59:59", freq=fine_freq).time)

  for (station, date), dfg in df.groupby(['device', 'date']):
    try:
      s = pd.Series(dfg['mac_address'], index=dfg.index)
      dfu = pd.DataFrame(s.groupby(pd.Grouper(freq=fine_freq)).value_counts())
      dfu.columns = ['repetitions_counter']
      dfu = dfu.reset_index()
      dfu = dfu.set_index('date_time')
      dfu = dfu.groupby('date_time')[['mac_address']].count()

      if len(dfu) != len(stats):
        newidx = [ datetime(
          year=date.year,
          month=date.month,
          day=date.day,
          hour=t.hour,
          minute=t.minute,
          second=t.second
        ) for t in stats.index ]
        dfu = dfu.reindex(newidx)
      stats[(station, str(date))] = dfu.mac_address.values
    except:
      print(f"Error: router {station}, in {date}")
      continue

  stats[ stats == 0 ] = np.nan
  stats = stats.reset_index()
  stats = stats.interpolate(limit_direction='both')
  stats = stats.set_index('index')
  stats.index.name = 'time'
  tstats = datetime.now() - tnow
  stats.to_csv(f'counters_{fine_freq}_.csv', sep=';', index=True)

  tnow = datetime.now()
  ave = stats.copy()
  ave = ave.stack()
  ave.index = pd.MultiIndex.from_tuples([ (t, i[0], i[1]) for t, i in ave.index ], names=['time', 'station_id', 'date'])
  ave = ave.reset_index()
  cols = ave.columns.values
  cols[-1] = 'cnt'
  ave.columns = cols
  #ave.station_id = ave.station_id.astype(int)
  ave.date = pd.to_datetime(ave.date)
  ave['wday'] = ave.date.dt.strftime('%a')
  dfave = ave.groupby(['station_id', 'wday', 'time']).mean()

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
      # dfp.to_csv(f'{base}/{sid}_{fine_freq}_{interp}.csv', sep=';', index=True)
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
    smooth.index.name='time'
    # smooth.to_csv(f'{base}/{sid}_{fine_freq}_{interp}_smooth.csv', sep=';', index=True)
    smooths[sid] = smooth
  tave = datetime.now() - tnow
  print(f'Averaging done in {tave} for {smooths.keys()}')

  base = f'{dubro}/router_local_analysis'
  if not os.path.exists(base): os.mkdir(base)

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

    stop_time = end_time + timedelta(seconds=84599)
    stop_time = pd.to_datetime(stop_time)

    drange = pd.date_range(start_time, stop_time, freq='1d')[:-1] # only for stop = Y M D 00:00:00

    drange = [ d.strftime('%a') for d in drange ]
    ave_class = [ smooths[s][wdcat[d]].values for d in drange ]
    ave_day = [ smooths[s][d].values for d in drange ]
    #print(np.asarray(drange).shape)
    ave_cnt = np.concatenate(ave_class)
    ave_d_cnt = np.concatenate(ave_day)
    tidx = pd.date_range(start_time, end_time, freq=fine_freq)#[:-1] # only for stop = Y M D 00:00:

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
    # diff_smooth = np.fft.fftshift(np.real(np.fft.ifft( np.fft.fft( diff ) * np.fft.fft(kern) )))
    diff_smooth = dft.cnt_smooth - dft.ave_day_cnt
    l1d_ave = diff_smooth.mean()
    l1d_std = diff_smooth.std()

    l1d_thresh_up = diff_smooth.quantile(kpi_thresh[cg.HIGH])
    l1d_thresh_down = diff_smooth.quantile(kpi_thresh[cg.LOW])

    print(f'Station {s} : LOW {l1d_thresh_down:.2f}({kpi_thresh[cg.LOW]} perc) HIGH {l1d_thresh_up:.2f}({kpi_thresh[cg.HIGH]} perc)')
    dft['l1_diff'] = diff
    dft['l1_diff_smooth'] = diff_smooth

    dft['l1_kpi'] = cg.LOW
    dft.loc[ dft.l1_diff_smooth > l1d_thresh_down, 'l1_kpi'] = cg.AVE
    dft.loc[ dft.l1_diff_smooth > l1d_thresh_up, 'l1_kpi'] = cg.HIGH

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

    try:
      dft = fullt[s]
    except:
      print(f'Plot: station {s} not available')

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(16, 10), sharex=True)
    ts = [ t.timestamp() for t in dft.index ]
    tus = 24
    lus = 1
    fine_freq_s = int(freq[:-1])
    ts_ticks = ts[::tus]
    ts_lbl = [ t.strftime('%a %d %b %H %M') for t in dft.index ]
    ts_lbl = ts_lbl[::tus]
    ts_lbl = [ t if i%lus==0 else '' for i, t in enumerate(ts_lbl)]
    axes = axs[0]
    axes.plot(ts, dft.cnt_smooth.values, 'r-o', label=f'Data smooth', markersize=4)
    axes.plot(ts, dft.ave_day_cnt.values, 'b--', label='Daily average data smooth')

    for t, kpi in zip(ts, dft['l1_kpi'].values): # special effects...SKADOUSH!!!
      axes.axvspan(t-0.5*fine_freq_s, t+0.5*fine_freq_s, facecolor=data_colors[kpi], alpha=0.3)

    axes.set_xticks(ts_ticks)
    axes.set_xticklabels(ts_lbl, rotation=45, ha='right')
    axes.grid()
    axes.legend()
    axes.set_ylabel('Counter')

    axes = axs[1]
    thresh_up = flustats[s]['l1_thr_up']
    thresh_down = flustats[s]['l1_thr_down']
    #axes.plot(ts, dft.l2_diff.values, 'b-o', label=f'Station {s} l2_diff', markersize=4)
    axes.plot(ts, dft.l1_diff.values, '-o', color='purple', label=f'Fluctuations', markersize=4)
    axes.plot(ts, dft.l1_diff_smooth.values, 'g-o', label=f'Fluctuations smooth', markersize=4)
    #axes.plot(ts, dft.l2_diff_thresh.values, 'g-o', label=f'Station {s} l2_diff', markersize=4)
    axes.axhspan(axes.get_ylim()[0], thresh_down, facecolor=kpi_colors[cg.LOW] , alpha=0.3, label=f'LOW < {kpi_thresh[cg.LOW]} centile')
    axes.axhspan(thresh_down, thresh_up, facecolor=kpi_colors[cg.AVE] , alpha=0.3)
    axes.axhspan(thresh_up, axes.get_ylim()[1], facecolor=kpi_colors[cg.HIGH] , alpha=0.3, label=f'HIGH > {kpi_thresh[cg.HIGH]} centile')

    axes.set_xticks(ts_ticks)
    axes.set_xticklabels(ts_lbl, rotation=45, ha='right')
    axes.grid()
    axes.legend()
    axes.set_xlabel(f'Daytime [Wday D M] (Ticks Sampling @ {freq})')
    axes.set_ylabel('Anomaly coefficient [au]')

    plt.tight_layout()
    fig.subplots_adjust(top=0.95)
    plt.suptitle(f'{s}: localnorm analysis, data sampling {fine_freq}', y=0.98)
    plt.savefig(f'{base}/{s}_{fine_freq}_localnorm.png')
    # plt.show()
    plt.clf()

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(16, 10), sharex=True)

    ts = [ t.timestamp() for t in dft.index ]
    tus = 24
    ts_ticks = ts[::tus]
    ts_lbl = [ t.strftime('%a %d %b %H %M') for t in dft.index ]
    ts_lbl = ts_lbl[::tus]
    ts_lbl = [ t if i%lus==0 else '' for i, t in enumerate(ts_lbl)]

    # axes = axs[0]
    axes = axs
    axes.plot(ts, dft.cnt_smooth.values, 'r-o', label=f'Data smooth', markersize=4)
    axes.plot(ts, dft.ave_day_cnt.values, 'b--', label='Daily average data smooth')

    axes.set_xticks(ts_ticks)
    axes.set_xticklabels(ts_lbl, rotation=45, ha='right')
    axes.grid(which='major')
    axes.legend()
    axes.set_ylabel('Counter')
    axes.set_xlabel(f'Daytime [Wday D M ] (Ticks Sampling @ {freq})')

    plt.tight_layout()
    fig.subplots_adjust(top=0.95)
    plt.suptitle(f'{s}: plot comparison, data sampling @ {fine_freq}', y=0.98)
    # plt.show()
    plt.savefig(f'{base}/{s}_{fine_freq}_comparison.png')

    plt.clf()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-d', '--data', help='conf data csv', required=True)
  parser.add_argument('-f', '--freq', help='freq data', type=int, default = 3600)
  parser.add_argument('-tt', '--time_ticks', help='set time spacing between ticks', type=int, default=300)
  parser.add_argument('-tl', '--time_labels', help='set time spacing between ticks\' labels', type=int, default=3600)

  args = parser.parse_args()

  data = args.data
  freq = f'{args.freq}s'
  localsnomalys(data)