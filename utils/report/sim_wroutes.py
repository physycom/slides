#! /usr/bin/env python3

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
from dateutil import tz

HERE = tz.tzlocal()
UTC = tz.gettz('UTC')

def sim_wroutes(wrin, outbase='', city='N/A'):
  df = pd.read_csv(wrin, delimiter=';')
  #print(df)

  dfc = df[ df.event_type == 'created'].copy().drop(columns=['event_type'])
  dfc['attr_num'] = dfc.wr_tag.str.len() - 2
  wrmap = {
    tag : [ l, w]
    for tag, l, w in dfc[['wr_tag', 'length', 'weight']].values
  }
  #print(dfc)


  dfp = df[ df.event_type != 'created'].copy()
  dfp['length'] = [ wrmap[tag][0] for tag in dfp.wr_tag.values ]
  dfp['weight'] = [ wrmap[tag][1] for tag in dfp.wr_tag.values ]
  dfp['pawn_id'] = dfp.event_type.str.split('#', expand=True)[1]
  dfp['attr_num'] = dfp.wr_tag.str.len() - 2
  dfp['len_km'] = dfp.length * 1e-3
  #print(dfp)

  # plots
  fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
  axes = axs.flatten()

  # length
  binw_km = 1.5
  hbins = np.arange(0, dfp.len_km.max() + 2*binw_km, binw_km)

  ax = axes[0]
  cnt, bins = np.histogram(dfp.len_km, bins=hbins)
  totcnt = cnt.sum()
  bins = bins[:-1]
  binw = (bins[1] - bins[0]) / 2
  ax.bar(bins + 0.5*binw, cnt, label='all', width=binw )
  #ax.bar(bins + 0.5*binw, cnt / cnt.sum(), label='all', width=binw )
  ax.set_xticks(bins)
  ax.set_xticklabels(bins)
  ax.set_xlabel(f'Length [km]')
  ax.set_ylabel('Counter')
  ax.grid(which='major', linestyle='-')
  ax.grid(which='minor', linestyle='--')
  ax.set_axisbelow(True)
  ax.legend()
  ax.set_title(f'Agendas length distribution')

  # attr number
  ax = axes[1]
  binw_m = 1
  hist_num = 3
  hbins = np.arange(0, dfc.attr_num.max() + 3*binw_m, binw_m)

  df1 = dfp
  cnt, bins = np.histogram(df1.attr_num, bins=hbins)
  bins = bins[:-1]
  binw = (bins[1] - bins[0]) / (hist_num + 1)
  ax.bar(bins + 0.5*binw, cnt, label='all', width=binw )

  df1 = dfp[ (dfp.length >= 1500) & (dfp.length < 3000) ]
  cnt, bins = np.histogram(df1.attr_num, bins=hbins)
  bins = bins[:-1]
  binw = (bins[1] - bins[0]) / (hist_num + 1)
  ax.bar(bins + 1.5*binw, cnt, label='1.5 - 3.0 km', width=binw )

  df1 = dfp[ (dfp.length >= 3000) & (dfp.length < 4500) ]
  cnt, bins = np.histogram(df1.attr_num, bins=hbins)
  bins = bins[:-1]
  binw = (bins[1] - bins[0]) / (hist_num + 1)
  ax.bar(bins + 2.5*binw, cnt, label='3.0 - 4.5 km', width=binw )

  ax.set_yscale('log')
  ax.set_xticks(bins)
  ax.set_xticklabels(bins)
  ax.set_xlabel(f'Visited attractions')
  ax.set_ylabel('Counter')
  ax.grid(which='major', linestyle='-')
  ax.grid(which='minor', linestyle='--')
  ax.set_axisbelow(True)
  ax.legend()
  ax.set_title(f'Number of visited attractions distribution')

  plt.tight_layout()
  fig.subplots_adjust(top=0.9)
  plt.suptitle(f'Statistics for {totcnt} simulated agendas, city {city}', y=0.98)
  if outbase == '':
    plt.show()
  else:
    plt.savefig(f'{outbase}_hist_report.png')
  plt.close()

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('-w', '--wrin', help='wroutes csv input', required=True)
  parser.add_argument('-o', '--outpng', help='output png', default='')
  parser.add_argument('-c', '--city', help='png title city', default='N/A')
  args = parser.parse_args()

  sim_wroutes(wrin=args.wrin, outbase=args.outpng, city=args.city)
