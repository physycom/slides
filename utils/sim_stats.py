#! /usr/bin/env python3

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def sim_stats(statsin, outbase=''):
  df = pd.read_csv(statsin, delimiter=';')
  #print(df)

  dfid = df.groupby('id').last()
  dfid['idletime'] = dfid.lifetime - dfid.triptime
  dfid['start_date'] = pd.to_datetime(dfid.start_time, unit='s')
  dfid = dfid.set_index('start_date')
  dfid = dfid.tz_localize('UTC').tz_convert('Europe/Rome').tz_localize(None)
  start = dfid.index[0]
  stop = dfid.index[-1]
  #print(dfid)

  dfg = dfid[['tag']].resample('300s').count().rename(columns={'tag':'pawn_created'})
  #print(dfg)

  dfin = dfid[ dfid.event == 'IN' ]
  dfout = dfid[ dfid.event == 'OUT' ]

  tot_pawn = len(dfid)
  tot_pawn_dead = len(dfout)
  tot_pawn_alive = len(dfin)
  tot_check = (tot_pawn == tot_pawn_dead + tot_pawn_alive)
  #print(f'Total pawn created {tot_pawn}, dead {tot_pawn_dead}, alive {tot_pawn_alive}, check {tot_check}')

  """
  Dead pawn
  """
  #print(dfout)

  fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
  axs = axs.flatten()

  for i, c in enumerate(['triptime', 'idletime', 'lifetime']):
    axes = axs[i]
    lt = dfout[c].values
    binwidth = 600 # sec
    bins = range(0, max(lt) + binwidth, binwidth)
    lt_cnt, lt_bins = np.histogram(dfout.lifetime.values, bins=bins)
    lt_bins = np.asarray(lt_bins[:-1])
    axes.set_xticks(lt_bins, minor=True)
    axes.set_xticks(lt_bins[::5])
    axes.set_xticklabels([ timedelta(seconds=int(s)) for s in lt_bins ], rotation=45)
    axes.set_xlabel(f'Trip time [HH:MM:SS]')
    axes.set_ylabel('Counter')
    #plt.tight_layout()
    axes.grid(which='major', linestyle='-')
    axes.grid(which='minor', linestyle='--')
    axes.set_axisbelow(True)
    binw = binwidth / 2
    axes.bar(lt_bins, lt_cnt, width=binw, label=c)
    axes.legend()

  plt.tight_layout()
  fig.subplots_adjust(top=0.95)
  plt.suptitle('Trip Time Distribution', y=0.98)
  if outbase == '':
    plt.show()
  else:
    plt.savefig(f'{outbase}_timedist.png')
  plt.close()

  """
  # working mode
  if popin == None and confin != None:
    with open(confin) as cin:
      config = json.load(cin)

    start_date = config['start_date']
    stop_date = config['stop_date']
    datetime_format = '%Y-%m-%d %H:%M:%S'
    start = datetime.strptime(start_date, datetime_format)
    stop = datetime.strptime(start_date, datetime_format)
    midn_start = start.replace(hour=0, minute=0, second=0)

    sampling = 300
    fullt = pd.date_range(midn_start.strftime(datetime_format), (midn_start + timedelta(hours=24, seconds=-sampling)).strftime(datetime_format), freq='{}s'.format(sampling), tz='UTC')
    df = pd.DataFrame(index=fullt)

    if 'sources' in config:
      src = config['sources']
      for k,v in src.items():
        rates = v['creation_rate']
        #print(k, len(rates))
        df[k] = rates
    #print(df)
  elif popin != None and confin == None:
  else:
    raise Exception(f'[sim_plotter] invalid mode popin {popin} csvin {confin}')

  # autoscaling parameters
  maxlbl = 15
  maxtick = maxlbl * 4
  start = df.index[0]
  stop = df.index[-1]
  dt = (stop - start).total_seconds()
  ts = [ t.timestamp() for t in df.index ]

  # autoscale minor axis ticks
  tsn = len(ts)
  if tsn > maxtick:
    min_us = tsn // maxtick
  else:
    min_us = 1
  minor_ticks = ts[::min_us]
  #print('mnt', len(minor_ticks))
  minor_dt = dt / (len(minor_ticks) - 1)
  dt_td = timedelta(seconds=minor_dt)

  # autoscale major axis ticks and labels
  if maxtick > maxlbl:
    maj_us = maxtick // maxlbl
  else:
    maj_us = 1
  major_ticks = minor_ticks[::maj_us]
  #print('mjt', len(major_ticks))
  major_lbl = [ t.strftime('%b %-d %H:%M') for t in df.index ]
  major_lbl = major_lbl[::min_us][::maj_us]

  # plot
  fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
  axes.set_xticks(minor_ticks, minor=True)
  axes.set_xticks(major_ticks)
  axes.set_xticklabels(major_lbl, rotation=45)

  for c in df.columns:
    axes.plot(ts, df[c], '-o', label=c)

  axes.legend()
  plt.xlabel(f'Time of day [Month Day HH:MM], minor ticks every {dt_td}')
  plt.ylabel('Counter')
  plt.title('Population')
  plt.tight_layout()
  plt.grid(which='major', linestyle='-')
  plt.grid(which='minor', linestyle='--')

  if outpng == '':
    plt.show()
  else:
    plt.savefig(outpng)
  """

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('-s', '--statsin', help='stats csv input', default=None)
  parser.add_argument('-o', '--outpng', help='output png', default='')
  args = parser.parse_args()

  sim_stats(statsin=args.statsin, outbase=args.outpng)
