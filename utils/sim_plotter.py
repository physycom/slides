#! /usr/bin/env python3

import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def sim_plot(popin=None, confin=None, outpng='', city='N/A'):
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
    ptitle = f'Source timetables, city {city}'
  elif popin != None and confin == None:
    df = pd.read_csv(popin, delimiter=';', parse_dates=['datetime'], index_col='datetime')
    df = df.drop(columns=['timestamp', 'transport', 'awaiting_transport'])
    ptitle = f'Population, city {city}'
    #print(df)
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
  if 'LOCALS' in df.columns:
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
    axs = axs.flatten()

    axes = axs[0]
    axes.set_xticks(minor_ticks, minor=True)
    axes.set_xticks(major_ticks)
    axes.set_xticklabels(major_lbl, rotation=45)

    for c in df.columns:
      if c != 'LOCALS':
        axes.plot(ts, df[c], '-o', label=c)

    axes.legend()
    axes.set_xlabel(f'Time of day [Month Day HH:MM], minor ticks every {dt_td}')
    axes.set_ylabel('Counter')
    axes.grid(which='major', linestyle='-')
    axes.grid(which='minor', linestyle='--')

    axes = axs[1]
    axes.set_xticks(minor_ticks, minor=True)
    axes.set_xticks(major_ticks)
    axes.set_xticklabels(major_lbl, rotation=45)

    axes.plot(ts, df['LOCALS'], '-o', label='LOCALS')

    axes.legend()
    axes.set_xlabel(f'Time of day [Month Day HH:MM], minor ticks every {dt_td}')
    axes.set_ylabel('Counter')
    axes.grid(which='major', linestyle='-')
    axes.grid(which='minor', linestyle='--')
  else:
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
    axes.set_xticks(minor_ticks, minor=True)
    axes.set_xticks(major_ticks)
    axes.set_xticklabels(major_lbl, rotation=45)

    for c in df.columns:
      axes.plot(ts, df[c], '-o', label=c)

    axes.legend()
    axes.set_xlabel(f'Time of day [Month Day HH:MM], minor ticks every {dt_td}')
    axes.set_ylabel('Counter')
    axes.grid(which='major', linestyle='-')
    axes.grid(which='minor', linestyle='--')

  plt.tight_layout()
  fig.subplots_adjust(top=0.9)
  plt.suptitle(ptitle, y=0.98)
  if outpng == '':
    plt.show()
  else:
    plt.savefig(outpng)
  plt.close()

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('-p', '--popin', help='population csv input', default=None)
  parser.add_argument('-c', '--confin', help='config json input', default=None)
  parser.add_argument('-o', '--outpng', help='output png', default='')
  args = parser.parse_args()

  sim_plot(popin=args.popin, confin=args.confin, outpng=args.outpng)
