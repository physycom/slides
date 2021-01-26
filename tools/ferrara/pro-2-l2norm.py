#! /usr/bin/env python3

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil import tz
from glob import glob

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

def box_centered_kernel(tot_len, box_len):
  pad_len = tot_len - box_len
  kern = np.concatenate([
    np.zeros((pad_len // 2)),
    np.ones((box_len)) / box_len,
    np.zeros((pad_len - pad_len // 2))# for odd box_len
  ])
  return kern

if __name__ == '__main__':
  import argparse
  import matplotlib.pyplot as plt

  parser = argparse.ArgumentParser()
  parser.add_argument('-d', '--data', help='counters data csv', required=True)
  parser.add_argument('-pc', '--plotconf', default='')
  args = parser.parse_args()

  filein = args.data
  base = filein[:filein.rfind('/')]
  tok = filein[:filein.find('/')].split('_')
  fname = filein[filein.find('/')+1:filein.rfind('.')].split('_')
  fine_freq = fname[-2]
  interp = fname[-1]

  dt_fmt = '%Y%m%d-%H%M%S'
  try:
    start = datetime.strptime(tok[-2], dt_fmt)
    stop = datetime.strptime(tok[-1], dt_fmt)
  except:
    start = datetime.strptime(tok[-3], dt_fmt)
    stop = datetime.strptime(tok[-2], dt_fmt)

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

  ave = ave[ ave.station_id != '4' ] # fix for missing data in dataset *****************
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
  print(smooths.keys())
  

  """
  Evaluate functional distance wrt ave signals
  """
  stats_smooth = pd.DataFrame(index=stats.index)
  ma_size = 10
  for c in stats.columns:
    kern = box_centered_kernel(len(stats), ma_size)
    conv = np.fft.fftshift(np.real(np.fft.ifft( np.fft.fft( stats[c].values ) * np.fft.fft(kern) )))
    stats_smooth[c] = conv
  print(stats_smooth)
  print(stats_smooth.columns)

  tnow = datetime.now()
  ldata = []
  for c in stats.columns:
    #print(c)
    wday = datetime.strptime(c[1], '%Y-%m-%d').strftime('%a')
    #print(wday)
    v = stats_smooth[c].values
    if wday == 'Sun' or wday == 'Sat':
      a = smooth['festivi'].values
    else:
      a = smooth['feriali'].values
    ad = smooth[wday].values
    l2v = np.sqrt((v**2).sum())
    l2a = np.sqrt((a**2).sum())
    l2ad = np.sqrt((ad**2).sum())
    l2dist = np.sqrt(((v - a)**2).sum())
    l2dist_d = np.sqrt(((v - ad)**2).sum())
    ldata.append([c[0], c[1], wday, l2v, l2a, l2ad, l2dist, l2dist_d])
  ldf = pd.DataFrame(ldata, columns=[
    'station_id',
    'date',
    'wday',
    'l2_day',
    'l2_ave',
    'l2_ave_d',
    'l2_dist',
    'l2_dist_d'
  ])
  ldf.date = pd.to_datetime(ldf.date)
  ldf['l2_norm'] = ldf.l2_dist / ldf.l2_ave
  ldf['l2_norm_d'] = ldf.l2_dist_d / ldf.l2_ave_d
  #print(ldf)
  # ldf.to_csv(f'{base}/{fine_freq}_{interp}_l2norm.csv', sep=';', index=True)

  """
  Rebuild full-time dataframe
  """
  datetime_fmt = '%Y-%m-%d %H:%M:%S'
  fullt = {}
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

    fullt[s] = dft

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
    unders = 10
    ts_ticks = ts[::unders]
    ts_lbl = [ t.strftime('%a %d %H:%M') for t in dft.index ]
    ts_lbl = ts_lbl[::unders]
    ts_lbl = [ t if i%3==0 else '' for i, t in enumerate(ts_lbl)]
    axes = axs[0]
#    axes.plot(ts, dft['cnt'].values, '-o', label=s, markersize=4)
    axes.plot(ts, dft['cnt_smooth'].values, 'y-o', label=f'Station {s} data', markersize=4)
#    axes.plot(ts, dft['ave_cnt'].values, 'r--', label='ave')
    axes.plot(ts, dft['ave_day_cnt'].values, 'b--', label='ave_d')
    axes.set_xticks(ts_ticks)
    axes.set_xticklabels(ts_lbl, rotation=45)
    axes.grid()
    axes.legend()
    axes.set_xlabel('Daytime Wday DD HH:MM')
    axes.set_ylabel('Counter')

    axes = axs[1]
    dftemp = ldf[ ldf.station_id == s ]
    print(dftemp)
    ts = [ datetime(t.year, t.month, t.day).timestamp() for t in dftemp.date ]
    sstart = selection[s]['start']
    sstop = selection[s]['stop']
    range_start = sstart.timestamp()
    range_stop = sstop.timestamp()
    ts_ticks = ts
    ts_lbl = [ t.strftime('%y-%m-%d') for t in dftemp.date ]
    #ts_lbl = ts_lbl[::unders]
    #ts_lbl = [ t if i%3==0 else '' for i, t in enumerate(ts_lbl)]
    axes.plot(ts, dftemp['l2_norm'].values, 'r-o', label=f'Station {s} l2 diff (fer/fest)', markersize=4)
    axes.plot(ts, dftemp['l2_norm_d'].values, 'b-o', label=f'Station {s} l2 diff (weekday)', markersize=4)
    axes.axvspan(range_start, range_stop, color='gray', alpha=0.3)
    axes.set_xticks(ts_ticks)
    axes.set_xticklabels(ts_lbl, rotation=45)
    axes.set_ylim(bottom=0)
    axes.grid()
    axes.legend()
    axes.set_xlabel('Day YY-MM-DD')
    axes.set_ylabel('Anomaly coefficient [au]')

    plt.tight_layout()
    fig.subplots_adjust(top=0.95)
    plt.suptitle(f'Station {s} anomaly analysis, period {sstart} - {sstop}', y=0.98)
    plt.savefig(f'{base}/{s}_{fine_freq}_aveanomaly_{ptag}.png')
    plt.close()
