#! /usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime, timedelta

router_group = defaultdict(
  first_group = [122, 123, 126],
  second_group = [145, 146, 147]
)

inv_router_group = {val: k for k, v in router_group.items() for val in v}

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


def localanomaly(data, freq):
  dubro = os.path.join(os.environ['WORKSPACE'], 'slides', 'work_lavoro', 'dubrovnik')
  if not os.path.exists(dubro): os.mkdir(dubro)
  base = f'{dubro}/router_local_anomaly'
  if not os.path.exists(base): os.mkdir(base)

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

  df.index = df.index.rename('date_time')
  data_i = start
  data_f = stop
  start_time = pd.to_datetime(data_i)
  end_time = pd.to_datetime(data_f)
  stats = pd.DataFrame()

  df_list = [k for k in inv_router_group.keys()]
  # df_list = [122, 124, 128, 130, 127, 140, 152, 151, 157, 148, 132, 135, 133, 131, 136, 137, 138]
  df = df.loc[df['device'].isin(df_list)]

  df['wday'] = [ t.strftime('%a') for t in df.index ]
  df['date'] = df.index.date
  df['time'] = df.index.time

  tnow = datetime.now()
  stats = pd.DataFrame(index=pd.date_range("00:00", "23:59:59", freq=freq).time)

  for (router, date), dfg in df.groupby(['device', 'date']):
    try:
      s = pd.Series(dfg['mac_address'], index=dfg.index)
      dfu = pd.DataFrame(s.groupby(pd.Grouper(freq=freq)).value_counts())
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
      stats[(router, str(date))] = dfu.mac_address.values
    except:
      print(f"Error: router {router}, in {date}")
      continue

  stats[ stats == 0 ] = np.nan
  stats = stats.reset_index()
  stats = stats.interpolate(limit_direction='both')
  stats = stats.set_index('index')
  stats.index.name = 'time'

  tnow = datetime.now()
  ave = stats.copy()
  ave = ave.stack()
  ave.index = pd.MultiIndex.from_tuples([ (t, i[0], i[1]) for t, i in ave.index ], names=['time', 'router_id', 'date'])
  ave = ave.reset_index()
  cols = ave.columns.values
  cols[-1] = 'cnt'
  ave.columns = cols
  ave.date = pd.to_datetime(ave.date)
  ave['wday'] = ave.date.dt.strftime('%a')

  # select period to compute the mean counter
  start_ave = pd.to_datetime('2021-09-01')
  stop_ave = pd.to_datetime('2021-09-30')
  masq_ave = (ave.date >= start_ave) & (ave.date <= stop_ave)
  ave = ave.loc[masq_ave]
  dfave = ave.groupby(['router_id', 'wday', 'time']).mean()

  smooths = {}
  for sid, dfg in dfave.groupby(['router_id']):
    try:
      dfp = dfg.unstack(level=1)
      dfp.index = pd.Index([ v[1] for v in dfp.index ], name='time')
      dfp.columns = [ v[1] for v in dfp.columns ]
      dfp = dfp[['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']]
      dfp['feriali'] = dfp[['Mon', 'Tue', 'Wed', 'Thu', 'Fri']].mean(axis=1)
      dfp['festivi'] = dfp[['Sat', 'Sun']].mean(axis=1)
      dfp = dfp.astype(int)
    except Exception as e:
      print(f'Error with router {sid} : {e}')
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
    smooths[sid] = smooth
  tave = datetime.now() - tnow
  print(f'Averaging done in {tave} for {smooths.keys()}')

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

    stop_time = end_time + timedelta(seconds=84599)
    stop_time = pd.to_datetime(stop_time)
    drange = pd.date_range(start_time, stop_time, freq='1d')[:-1] # only for stop = Y M D 00:00:00

    drange = [ d.strftime('%a') for d in drange ]
    ave_class = [ smooths[s][wdcat[d]].values for d in drange ]
    ave_day = [ smooths[s][d].values for d in drange ]
    ave_cnt = np.concatenate(ave_class)
    ave_d_cnt = np.concatenate(ave_day)
    tidx = pd.date_range(start_time, end_time, freq=freq)#[:-1] # only for stop = Y M D 00:00:

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
    diff_smooth = dft.cnt_smooth - dft.ave_day_cnt
    l1d_ave = diff_smooth.mean()
    l1d_std = diff_smooth.std()

    l1d_thresh_up = diff_smooth.quantile(kpi_thresh[cg.HIGH])
    l1d_thresh_down = diff_smooth.quantile(kpi_thresh[cg.LOW])

    print(f'Router {s} : LOW {l1d_thresh_down:.2f}({kpi_thresh[cg.LOW]} perc) HIGH {l1d_thresh_up:.2f}({kpi_thresh[cg.HIGH]} perc)')
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
      print(f'Plot: router {s} not available')

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(16, 10), sharex=True)
    
    start_plt = pd.to_datetime('2021-09-20')
    stop_plt = pd.to_datetime('2021-09-30')
    masq_plt = (dft.index >= start_plt) & (dft.index <= stop_plt)
    dft = dft.loc[masq_plt]

    ts = [ t.timestamp() for t in dft.index ]
    tus = 24
    lus = 1
    ts_ticks = ts[::tus]
    ts_lbl = [ t.strftime('%a %d %b') for t in dft.index ]
    ts_lbl = ts_lbl[::tus]
    ts_lbl = [ t if i%lus==0 else '' for i, t in enumerate(ts_lbl)]

    axes = axs[0]
    axes.plot(ts, dft.cnt_smooth.values, 'b-o', label=f'Data smoothed', markersize=4)
    axes.plot(ts, dft.ave_day_cnt.values, 'r--', label='Daily average smoothed')

    for t, kpi in zip(ts, dft['l1_kpi'].values):
      axes.axvspan(t-0.5*fine_freq_s, t+0.5*fine_freq_s, facecolor=data_colors[kpi], alpha=0.3)

    axes.set_xticks(ts_ticks)
    axes.set_xticklabels(ts_lbl, rotation=45, ha='right')
    axes.grid()
    axes.legend()
    axes.set_ylabel('Counter')

    axes = axs[1]
    thresh_up = flustats[s]['l1_thr_up']
    thresh_down = flustats[s]['l1_thr_down']
    axes.plot(ts, dft.l1_diff.values, '-o', color='purple', label=f'Fluctuations', markersize=4)
    axes.plot(ts, dft.l1_diff_smooth.values, 'g-o', label=f'Fluctuations smoothed', markersize=4)
    axes.axhspan(axes.get_ylim()[0], thresh_down, facecolor=kpi_colors[cg.LOW] , alpha=0.3, label=f'LOW < {kpi_thresh[cg.LOW]} centile')
    axes.axhspan(thresh_down, thresh_up, facecolor=kpi_colors[cg.AVE] , alpha=0.3)
    axes.axhspan(thresh_up, axes.get_ylim()[1], facecolor=kpi_colors[cg.HIGH] , alpha=0.3, label=f'HIGH > {kpi_thresh[cg.HIGH]} centile')

    axes.set_xticks(ts_ticks)
    axes.set_xticklabels(ts_lbl, rotation=45, ha='right')
    axes.grid()
    axes.legend()
    axes.set_xlabel(f'Daytime [Wday D M]')
    axes.set_ylabel('Anomaly coefficient [au]')

    plt.tight_layout()
    plt.suptitle(f'Router {s}: localanomaly analysis\nAve from {start_ave.date()} to {stop_ave.date()}\n@{freq}', y=0.98)
    fig.subplots_adjust(top=0.90)
    plt.savefig(f'{base}/{s}_{freq}_localanomaly.png')
    # plt.show()
    plt.clf()

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(16, 10), sharex=True)
    axs.plot(ts, dft.cnt_smooth.values, 'b-o', label=f'Data smoothed', markersize=4)
    axs.plot(ts, dft.ave_day_cnt.values, 'r--', label='Daily average smoothed')
    axs.set_xticks(ts_ticks)
    axs.set_xticklabels(ts_lbl, rotation=45, ha='right')
    axs.grid(which='major')
    axs.legend()
    axs.set_ylabel('Counter')
    axs.set_xlabel(f'Daytime [Wday D M ]')

    plt.tight_layout()
    plt.suptitle(f'Router {s}: plot comparison\nAve from {start_ave.date()} to {stop_ave.date()}\n@{freq}', y=0.98)
    fig.subplots_adjust(top=0.90)
    # plt.show()
    plt.savefig(f'{base}/{s}_{freq}_comparison.png')
    plt.clf()

def delta_router_group(data, freq):
  dubro = os.path.join(os.environ['WORKSPACE'], 'slides', 'work_lavoro', 'dubrovnik')
  if not os.path.exists(dubro): os.mkdir(dubro)
  output = f'{dubro}/router_perc_diff'
  if not os.path.exists(output): os.mkdir(output)

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

  df.index = pd.to_datetime(df.index).rename('date_time')
  start_date = '2021-09-01 00:00:00'
  stop_date = '2021-09-30 23:59:59'
  start_time = pd.to_datetime(start_date)
  end_time = pd.to_datetime(stop_date)
  masquer = (df.index >= start_time) & (df.index <= end_time)
  df = df.loc[masquer]
  df['group'] = df.device.map(inv_router_group)
  df['data_time'] = df.index
  df = df[df['group'].notnull()]

  for group, dfg in df.groupby(['group']):
    s = pd.Series(dfg['mac_address'], index=dfg.index)
    df = pd.DataFrame(s.groupby(pd.Grouper(freq=freq)).value_counts())
    df.columns = ['repetitions_counter']
    df = df.reset_index()
    df = df.set_index('date_time')
    df = df.groupby('date_time')[['mac_address']].count()
    df = df.resample('1D').sum()
    df = df.rename(columns={"mac_address":"device_counter"})

    df['wday'] = [ t.strftime('%a') for t in df.index ]
    d_mean = dict()
    for day, df_d in df.groupby('wday'):
      d_mean[day] = df_d.device_counter.mean()

    df['day_mean'] = df.wday.map(d_mean)
    device_counter_list = df.device_counter.tolist()
    day_mean_list = df.day_mean.tolist()
    changes = []
    for x1, x2 in zip(day_mean_list, device_counter_list):
      try:
        pct = (x2 - x1) * 100 / x1
      except ZeroDivisionError:
        pct = None
      changes.append(pct)     
    df['percentage'] = changes
    start_plt = pd.to_datetime('2021-09-20')
    stop_plt = pd.to_datetime('2021-09-30')
    masq_plt = (df.index >= start_plt) & (df.index <= stop_plt)
    df = df.loc[masq_plt]

    ts = [ t.timestamp() for t in df.index ]
    tus = 1
    lus = 1
    ts_ticks = ts[::tus]
    ts_lbl = [ t.strftime('%a %b %d') for t in df.index ]
    ts_lbl = ts_lbl[::tus]
    ts_lbl = [ t if i%lus==0 else '' for i, t in enumerate(ts_lbl)]
    
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(16, 10), sharex=True)
    ax = axs[0]
    ax.plot(ts, df.device_counter, 'b-o', label= 'Data', markersize=4)
    ax.plot(ts, df.day_mean, label= f'Daily average', c = 'r', linestyle='dashdot', markersize=4)
    trend_lim = (1900,5000)
    ax.legend()
    ax.grid(which='major', linestyle='-')
    ax.grid(which='minor', linestyle='--')
    ax.set_ylabel('Number of People per Day')
    ax.set_ylim(trend_lim)

    ax = axs[1]
    # width = 18*3600
    width = 9*3600
    ax.bar(ts, df.percentage, width = width, label='test')
    perc_lim = (-25, 10)
    ax.grid(which='major', linestyle='-')
    ax.grid(which='minor', linestyle='--')
    ax.set_ylabel('Value [%]', color='b')
    ax.set_ylim(perc_lim)
    ax.set_title(f'Variation from the average')

    ax.set_xticks(ts_ticks)
    ax.set_xticklabels(ts_lbl, rotation=45, ha='right')
    ax.set_xlabel(f'Daytime [Wday M D]')
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    ptitle = f'Router group: {router_group[group]}\nAve from {start_time.date()} to {end_time.date()}\n@{freq}'
    fig.suptitle(ptitle, y=0.98)
    # plt.show()
    fig.savefig(f'{output}/perc_diff_{group}_{freq}.png')
    plt.close()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-d', '--data', help='conf data csv', required=True)
  parser.add_argument('-f', '--freq', help='freq data', type=int, default = 3600)

  args = parser.parse_args()

  data = args.data
  freq = f'{args.freq}s'
  localanomaly(data, freq)
  delta_router_group(data, freq)