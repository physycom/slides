#! /usr/bin/env python3

import os
import sys
import json
import numpy as np
import pandas as pd
import pymongo
from datetime import datetime, timedelta
from dateutil import tz

#######################
#### log function #####
#######################
def logs(s):
  head = '{} [m_ferrara] '.format(datetime.now().strftime('%y%m%d %H:%M:%S'))
  return head + s

def log_print(s, logger = None):
  if logger:
    logger.info(logs(s))
  else:
    print(logs(s), flush=True)

############################
#### model_slides class ####
############################
class model_slides_ferrara:

  def __init__(self, config, logger=None):
    self.logger = logger
    self.got_data = False
    self.date_format = '%Y-%m-%d %H:%M:%S'
    self.time_format = '%H:%M:%S'
    self.rates_dt = 5 * 60

    self.wdir = config['work_dir']
    if not os.path.exists(self.wdir): os.mkdir(self.wdir)

  def full_table(self, start, stop, tag):
    df = pd.DataFrame([], columns=['time'])
    return df

if __name__ == '__main__':
  import argparse
  import matplotlib.pyplot as plt

  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--input', type=str, required=True)
  parser.add_argument('-s', '--show', action='store_true')
  parser.add_argument('-fs', '--fine_sampling', type=int, default=10)
  parser.add_argument('-os', '--out_sampling', type=int, default=300)
  parser.add_argument('-in', '--interpolation', choices=['lin', 'no'], default='lin')
  parser.add_argument('-a', '--aggr', choices=['rec', 'uniq'], default='uniq')
  parser.add_argument('-pc', '--plotconf', default='')

  args = parser.parse_args()
  fine_freq = f'{args.fine_sampling}s'
  filein = args.input
  base = filein[:filein.rfind('.')]
  if not os.path.exists(base): os.mkdir(base)
  tok = filein[:filein.rfind('.')].split('_')

  dt_fmt = '%Y%m%d-%H%M%S'
  start = datetime.strptime(tok[-2], dt_fmt)
  stop = datetime.strptime(tok[-1], dt_fmt)

  df = pd.read_csv(filein, sep=';', usecols=['mac_address', 'date_time', 'station_name', 'kind'], parse_dates=['date_time'], index_col='date_time')
  df['wday'] = [ t.strftime('%a') for t in df.index ]
  df['date'] = df.index.date
  df['station_id'] = df.station_name.str[-2:-1]
  #print(df)
  #print(df[['wday', 'date', 'station_id']])

  """
  Perform device id counting with fine temporal scale
  """
  tnow = datetime.now()
  stats = pd.DataFrame(index=pd.date_range("00:00", "23:59:59", freq=fine_freq).time)
  for (station, date), dfg in df.groupby(['station_id', 'date']):
    #print(station, date)

    if args.aggr == 'uniq':
      s = pd.Series(dfg['mac_address'], index=dfg.index)
      dfu = pd.DataFrame(s.groupby(pd.Grouper(freq=fine_freq)).value_counts())
      dfu.columns = ['repetitions_counter']
      dfu = dfu.reset_index()
      dfu = dfu.set_index('date_time')
      dfu = dfu.groupby('date_time')[['mac_address']].count()
      #dfu.columns = [f'{sid}_unique']
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
      #print('dfu', dfu)
    elif args.aggr == 'rec':
      dfr = dfg[['mac_address']].resample(fine_freq).count() # convert to count_unique
      if len(dfr) != len(stats):
        newidx = [ datetime(
          year=date.year,
          month=date.month,
          day=date.day,
          hour=t.hour,
          minute=t.minute,
          second=t.second
        ) for t in stats.index ]
        dfr = dfr.reindex(newidx)
      #print(dfr)
      stats[(station, str(date))] = dfr.mac_address.values

  # fix null/empty/nan/missing values
  stats[ stats == 0 ] = np.nan
  #print(stats)
  stats = stats.reset_index()
  if args.interpolation == 'lin':
    stats = stats.interpolate(limit=10000, limit_direction='both')
  else:
    stats = stats.fillna(0)
  stats = stats.set_index('index')
  tstats = datetime.now() - tnow
  print(f'Counting done {args.interpolation} stats in {tstats}\n', stats)
  stats.to_csv(f'{base}/counters_{fine_freq}_{args.interpolation}.csv', sep=';', index=True)

  """
  Groupby station_id and compute per day (or other criteria) mean signal.
  Resample by averaging fine values at output frequency.
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
      dfp.to_csv(f'{base}/{sid}_{fine_freq}_{args.interpolation}_{args.aggr}.csv', sep=';', index=True)
    except Exception as e:
      #print(f'Error with station {sid} : {e}')
      continue

    freq = f'{args.out_sampling}s'
    #print(dfp)
    dfp.index = pd.to_datetime([ datetime.strptime(f'1970-01-01 {t}', '%Y-%m-%d %H:%M:%S') for t in dfp.index ])
    dfpr = dfp.resample(freq).mean()#.round(0).astype(int)
    dfpr.index = pd.Index([ t.time() for t in dfpr.index ], name='time')
    #print(dfp)
    dfpr.to_csv(f'{base}/{sid}_{fine_freq}_{freq}_{args.interpolation}_{args.aggr}.csv', sep=';', index=True)

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
    smooth.to_csv(f'{base}/{sid}_{fine_freq}_{args.interpolation}_smooth.csv', sep=';', index=True)
    smooths[sid] = smooth
  tave = datetime.now() - tnow
  print(f'Averaging done in {tave}')

  """
  Evaluate functional distance wrt ave signals
  """
  tnow = datetime.now()
  ldata = []
  for c in stats.columns:
    #print(c)
    wday = datetime.strptime(c[1], '%Y-%m-%d').strftime('%a')
    #print(wday)
    v = stats[c].values
    if wday == 'Sun' or wday == 'Sat':
      a = smooth['festivi'].values
    else:
      a = smooth['feriali'].values
    l2v = (v**2).sum()
    l2a = (a**2).sum()
    l2dist = ((v - a)**2).sum()
    ldata.append([c[0], c[1], wday, l2v, l2a, l2dist])
  ldf = pd.DataFrame(ldata, columns=[
    'station_id',
    'date',
    'wday',
    'l2_day',
    'l2_ave',
    'l2_dist'
  ])
  ldf.date = pd.to_datetime(ldf.date)
  ldf['l2_norm'] = ldf.l2_dist / ldf.l2_ave
  #print(ldf)
  #ldf.to_csv(f'{base}/{sid}_{fine_freq}_{args.interpolation}_smooth.csv', sep=';', index=True)

  # subplot grid auto-sizing
  grp = ldf.groupby('station_id')
  totplot = len(grp)
  n = int(np.sqrt(totplot))
  rest = totplot - n**2
  row = n if rest == 0 else n + 1
  col = n if rest <= n else n + 1

  fig, axes = plt.subplots(nrows=row, ncols=col, figsize=(16, 12))
  axes = axes.flatten()
  for i, (sid, dfg) in enumerate(grp):
    #print(i, sid)
    ts = [ datetime(t.year, t.month, t.day).timestamp() for t in dfg.date ]
    ts_ticks = ts
    ts_lbl = [ t.strftime('%y-%m-%d') for t in dfg.date ]
    #ts_lbl = ts_lbl[::unders]
    #ts_lbl = [ t if i%3==0 else '' for i, t in enumerate(ts_lbl)]
    axes[i].plot(ts, dfg['l2_norm'].values, 'r-o', label=sid, markersize=4)
    axes[i].set_xticks(ts_ticks)
    axes[i].set_xticklabels(ts_lbl, rotation=45)
    axes[i].grid()
    axes[i].legend()
    axes[i].set_xlabel('Day [%y-%m-%d]')
    axes[i].set_ylabel('l2 distance [au]')

  plt.tight_layout()
  fig.subplots_adjust(top=0.97)
  plt.suptitle(f'Day l2 norm {base}', y=0.99)
  outpng = f'{base}_l2norm.png'
  plt.savefig(outpng)
  if args.show: plt.show()
  tl2 = datetime.now() - tnow
  print(f'L2 analysis done in {tl2}')

  """
  Plot raw counters vs average signal.
  Use external json (cli flag -pc) to cut time windows per station.
  """
  plt.clf()
  datetime_fmt = '%Y-%m-%d %H:%M:%S'
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
    cols = [ c for c in stats.columns if c[0] == s ]
    #print(cols)
    dft = stats[cols].copy()
    #print(dft)
    dft = dft.stack()
    dft = dft.reset_index()
    dft.columns = ['time', 'date', 'cnt']
    dft['datetime'] = [ datetime.strptime(f'{d[1]} {t}', datetime_fmt) for t, d in dft[['time', 'date']].values ]
    dft = dft.sort_values(by=['datetime'])
    replicas = len(dft) // len(smooths[s])
    #print('Replicas', replicas)

    day_start = dft.datetime[0].date()
    day_stop = dft['datetime'].iloc[-1]
    print(f'{day_start} {day_stop}')
    drange = pd.date_range(day_start, day_stop, freq='1d')
    print(drange)
    exit(1)

    ferave = np.tile(smooths[s]['feriali'].values, replicas)
    fesave = np.tile(smooths[s]['festivi'].values, replicas)
    #print(len(dft), len(ferave), len(fesave))
    #print(dft)
    dft['feriali'] = ferave
    dft['festivi'] = fesave

    dft = dft[ (dft.datetime >= selection[s]['start']) & (dft.datetime < selection[s]['stop']) ]

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))
    ts = [ t.timestamp() for t in dft.datetime ]
    unders = 10
    ts_ticks = ts[::unders]
    ts_lbl = [ t.strftime('%a %d %H:%M') for t in dft.datetime ]
    ts_lbl = ts_lbl[::unders]
    ts_lbl = [ t if i%3==0 else '' for i, t in enumerate(ts_lbl)]
    axes.plot(ts, dft['cnt'].values, 'y-o', label=s, markersize=4)
    axes.plot(ts, dft['feriali'].values, 'r--', label='ma_feriali', markersize=4)
    axes.plot(ts, dft['festivi'].values, 'g--', label='ma_festivi', markersize=4)
    axes.set_xticks(ts_ticks)
    axes.set_xticklabels(ts_lbl, rotation=45)
    axes.grid()
    axes.legend()
    axes.set_xlabel('Day [%y-%m-%d]')
    axes.set_ylabel('Distance')

    plt.savefig(f'{base}_{s}_presence_{ptag}.png')
