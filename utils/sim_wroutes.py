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
  #print(dfp)

  # histo and plots
  fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
  axes = axs.flatten()

  # length
  ax = axes[0]
  binw_m = 1500
  hbins = np.arange(0, dfc.length.max() + 2*binw_m, binw_m)
  cnt, bins = np.histogram(dfc.length, bins=hbins)
  bins = bins[:-1]
  binw = (bins[1] - bins[0]) / 2
  ax.bar(bins + 0.5*binw, cnt, label='all', width=binw )
  ax.set_xticks(bins)
  ax.set_xticklabels(bins, rotation=45)
  ax.set_xlabel(f'Agenda bestpath distance [m]')
  ax.set_ylabel('Counter')
  ax.grid(which='major', linestyle='-')
  ax.grid(which='minor', linestyle='--')
  ax.set_axisbelow(True)
  ax.legend()
  ax.set_title(f'Static length distribution, tot {cnt.sum()}')

  ax = axes[3]
  cnt, bins = np.histogram(dfp.length, bins=hbins)
  bins = bins[:-1]
  binw = (bins[1] - bins[0]) / 2
  ax.bar(bins + 0.5*binw, cnt, label='all', width=binw )
  ax.set_xticks(bins)
  ax.set_xticklabels(bins, rotation=45)
  ax.set_xlabel(f'Agenda bestpath distance [m]')
  ax.set_ylabel('Counter')
  ax.grid(which='major', linestyle='-')
  ax.grid(which='minor', linestyle='--')
  ax.set_axisbelow(True)
  ax.legend()
  ax.set_title(f'Simulation length distribution, tot {cnt.sum()}')

  # attr number
  ax = axes[1]
  binw_m = 1
  hist_num = 3
  hbins = np.arange(0, dfc.attr_num.max() + 3*binw_m, binw_m)

  df1 = dfc
  cnt, bins = np.histogram(df1.attr_num, bins=hbins)
  bins = bins[:-1]
  binw = (bins[1] - bins[0]) / (hist_num + 1)
  ax.bar(bins + 0.5*binw, cnt, label='all', width=binw )

  df1 = dfc[ (dfc.length >= 1500) & (dfc.length < 3000) ]
  cnt, bins = np.histogram(df1.attr_num, bins=hbins)
  bins = bins[:-1]
  binw = (bins[1] - bins[0]) / (hist_num + 1)
  ax.bar(bins + 1.5*binw, cnt, label='1.5 - 3.0 km', width=binw )

  df1 = dfc[ (dfc.length >= 3000) & (dfc.length < 4500) ]
  cnt, bins = np.histogram(df1.attr_num, bins=hbins)
  bins = bins[:-1]
  binw = (bins[1] - bins[0]) / (hist_num + 1)
  ax.bar(bins + 2.5*binw, cnt, label='3.0 - 4.5 km', width=binw )

  ax.set_yscale('log')
  ax.set_xticks(bins)
  ax.set_xticklabels(bins)
  ax.set_xlabel(f'Attraction number')
  ax.set_ylabel('Counter')
  ax.grid(which='major', linestyle='-')
  ax.grid(which='minor', linestyle='--')
  ax.set_axisbelow(True)
  ax.legend()
  ax.set_title(f'Static attraction per wroute distribution')

  ax = axes[4]
  binw_m = 1
  hist_num = 3

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
  ax.set_xlabel(f'Attraction number')
  ax.set_ylabel('Counter')
  ax.grid(which='major', linestyle='-')
  ax.grid(which='minor', linestyle='--')
  ax.set_axisbelow(True)
  ax.legend()
  ax.set_title(f'Simulation attraction per wroute distribution')

  # weight
  ax = axes[2]
  binw_w = 0.025
  #hbins = np.arange(0, 15*binw_m, binw_m)
  hbins = np.arange(0, dfc.weight.max() + binw_w, binw_w)

  cnt, bins = np.histogram(dfc.weight, bins=hbins)
  bins = bins[:-1]
  binw = (bins[1] - bins[0]) / 2
  ax.bar(bins, cnt, label='all', width=binw )
  ax.set_yscale('log')
  ax.set_xticks(bins)
  ax.set_xticklabels([ f'{b:.3f}' for b in bins], rotation=45)
  ax.set_xlabel(f'Agenda weight [au]')
  ax.set_ylabel('Counter')
  ax.grid(which='major', linestyle='-')
  ax.grid(which='minor', linestyle='--')
  ax.set_axisbelow(True)
  ax.legend()
  ax.set_title(f'Static weight distribution, tot {cnt.sum()}')

  ax = axes[5]
  binw_m = 0.05
  cnt, bins = np.histogram(dfp.weight, bins=hbins)
  bins = bins[:-1]
  binw = (bins[1] - bins[0]) / 2
  ax.bar(bins, cnt, label='all', width=binw)
  ax.set_yscale('log')
  ax.set_xticks(bins)
  ax.set_xticklabels([ f'{b:.3f}' for b in bins], rotation=45)
  ax.set_xlabel(f'Agenda weight [au]')
  ax.set_ylabel('Counter')
  ax.grid(which='major', linestyle='-')
  ax.grid(which='minor', linestyle='--')
  ax.set_axisbelow(True)
  ax.legend()
  ax.set_title(f'Simulation weight distribution, tot {cnt.sum()}')

  plt.tight_layout()
  fig.subplots_adjust(top=0.9)
  plt.suptitle(f'Weighted routes distribution, city {city}', y=0.98)
  if outbase == '':
    plt.show()
  else:
    plt.savefig(f'{outbase}_hist.png')
  plt.close()

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('-w', '--wrin', help='wroutes csv input', required=True)
  parser.add_argument('-o', '--outpng', help='output png', default='')
  args = parser.parse_args()

  sim_wroutes(wrin=args.wrin, outbase=args.outpng)
