#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import to_hex
from collections import defaultdict 
  
plt.style.use('seaborn-dark-palette')

if __name__ == '__main__':
  # parse cli and config
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--input', type=str, required=True)
  parser.add_argument('-s', '--show', action='store_true')
  parser.add_argument('-b', '--bin', action='store_true')
  parser.add_argument('-t', '--dt', type=int, default=300)
  parser.add_argument('-r', '--range', type=str, default='')
  parser.add_argument('-a', '--aggr', type=str, default='')

  args = parser.parse_args()
  base = args.input[:args.input.find('_')]
  dt_fmt = '%Y-%m-%dT%H%M%S%z'
  short_fmt = '%Y%m%d-%H%M%S%Z'
  freq = f'{args.dt}s'
  if args.range == '':
    tok = args.input[:args.input.rfind('.')].split('_')
    start = pd.to_datetime(tok[-2], format=dt_fmt)
    stop = pd.to_datetime(tok[-1], format=dt_fmt)
  else:
    start = pd.to_datetime(args.range.split('|')[0], format=dt_fmt)
    stop = pd.to_datetime(args.range.split('|')[1], format=dt_fmt)
  print(f'Setting time range {start} {stop}')
  base = f'{base}_{start.strftime(short_fmt)}_{stop.strftime(short_fmt)}'

  # read raw data and adapt to ferrara naming convention
  df = pd.read_csv(args.input, sep=';')
  colfilter = {
    'occurredAt':'date_time',
    'clientId':'mac-address',
    'deviceSerial':'station_name',    
  }
  df = df[colfilter.keys()].rename(columns=colfilter)
  # print(df)
  # print(df.columns)
  df.date_time = pd.to_datetime(df.date_time)
  #df.date_time = df.date_time.dt.tz_localize(None)
  df = df[ (df.date_time >= start) & (df.date_time < stop) ]

  stats = pd.DataFrame()
  t_index = pd.date_range(start=start, end=stop, freq=freq)
  for sid, dfg in df.groupby(['station_name']):
    print(f'{sid}  -> {dfg.shape}')

    dfg = dfg.set_index('date_time')
    dfg.index = pd.to_datetime(dfg.index)
    dfr = dfg[['mac-address']].resample(freq).count()
    dfr.columns = [f'{sid}']

    s = pd.Series(dfg['mac-address'], index=dfg.index)
    dfu = pd.DataFrame(s.groupby(pd.Grouper(freq=freq)).value_counts())
    dfu.columns = ['repetitions_counter']
    dfu = dfu.reset_index()
    dfu = dfu.set_index('date_time')
    dfu = dfu.groupby('date_time')[['mac-address']].count()
    dfu.columns = [f'{sid}_unique']
    #print('dfu', dfu)
    #idx = [ i for i in dfu.index ]
    #print('my',idx)
    #print(dfu.index)
    #dfu = pd.DataFrame(idx, columns=[f'{sid}_unique'], index=dfu.index)#, index=[ i for idfu.index.get_level_values(0)] )
    #print(dfu)

    if len(stats) == 0:
      stats = dfr
    else:
      stats = pd.concat([stats, dfr], axis=1)
    stats = pd.concat([stats, dfu], axis=1)

  stats = stats.fillna(0)
  tot = stats.sum()
  stats['tot_unique'] = stats.sum(axis=1)
  tot = tot[[ c for c in tot.index if c.endswith('_unique')]].sort_values()
  tot_highest = tot.index[-2:]
  tot_lowest = tot.index[:4]
  #print(tot)

  if args.aggr != '':
    with open(args.aggr) as ain:
      aggr = json.load(ain)

    dfagg = df.copy()
    clustermap = defaultdict(lambda: 'none')
    clustermap.update({ i:k for k,v in aggr.items() for i in v })
    #print(clustermap)

    cmap = cm.get_cmap('viridis', len(aggr)+1)
    clustercol = { 'none' : to_hex(cmap.colors[len(aggr)]) }
    clustercol.update({ k : to_hex(cmap.colors[i]) for i, k in enumerate(aggr.keys()) })
    print(clustercol)

    dfagg['cluster'] = dfagg.station_name.apply(lambda x: clustermap[x])
    dfagg['cluster_col'] = dfagg.cluster.apply(lambda x: clustercol[x])
    print(dfagg)

    astats = pd.DataFrame()
    for cid, dfg in dfagg.groupby(['cluster']):
      print(f'{cid}  -> {dfg.shape}')

      dfg = dfg.set_index('date_time')
      dfg.index = pd.to_datetime(dfg.index)
      dfr = dfg[['mac-address']].resample(freq).count()
      dfr.columns = [f'{cid}']

      s = pd.Series(dfg['mac-address'], index=dfg.index)
      dfu = pd.DataFrame(s.groupby(pd.Grouper(freq=freq)).value_counts())
      dfu.columns = ['repetitions_counter']
      dfu = dfu.reset_index()
      dfu = dfu.set_index('date_time')
      dfu = dfu.groupby('date_time')[['mac-address']].count()
      dfu.columns = [f'{cid}_unique']

      if len(stats) == 0:
        astats = dfr
      else:
        astats = pd.concat([astats, dfr], axis=1)
      astats = pd.concat([astats, dfu], axis=1)

    astats = astats.fillna(0)
    print(astats)
    print(astats.columns)
  
    w, h, d = 12, 7, 150
    fig, ax = plt.subplots(1, 1, figsize=(w, h), dpi=d)
    plt.suptitle(f'Unique device count')

    filterlist = [c for c in astats.columns]
    clusterlist = [ 
      # 'cluster1', 
      # 'cluster2', 
      'cluster3', 
      'cluster4',
    ]
    filterlist = [ f'{c}_unique' for c in clusterlist ]
    for cid in astats.columns:
      if not cid in filterlist: continue
      if cid.endswith('_unique'):
        ax.plot(astats.index, astats[cid], '-', label=cid)
      else:
        continue

    ax.set_title(f'period {start} -> {stop}, resampling {freq}')
    ax.legend()
    ax.grid()
    ax.tick_params(labelrotation=45)

    if args.show:
      plt.show()
    else:
      plt.savefig(f'{base}_clusterpresence_{freq}.png')
    plt.close()

    import folium
    with open(os.path.join(os.environ['WORKSPACE'], 'slides', 'vars', 'extra', 'dubrovnik_sniffer.json')) as sin:
      geodata = json.load(sin)
    stations = pd.DataFrame.from_dict(geodata)
    stations['cluster'] = stations.serial.apply(lambda x: clustermap[x])
    stations['color'] = stations.serial.apply(lambda x: clustercol[clustermap[x]])
    print(stations)
    stations = stations[stations.cluster != 'none']
    map_center = stations[['lat', 'lon']].mean().values
    
    clusters = stations.groupby('cluster').agg({ 
      'color':'first',
      'lat':'mean',
      'lon':'mean'
    })
    print(clusters)

    m = folium.Map(location=map_center, control_scale=True, zoom_start=9)
    stations.apply(lambda row: folium.CircleMarker(
      location=[row.lat, row.lon], 
      radius=7, 
      fill_color=f'{row.color}',
      color=f'{row.color}',
      popup=folium.Popup(f'<p><b>ROUTER</b></br>id <b>{row.id}</b></br>serial <b>{row.serial}</b></br>serial <b>{row.cluster}</b></p>', show=False, sticky=True),
    ).add_to(m), axis=1)
    stations.apply(lambda row: folium.PolyLine(
      locations=[ 
        [ clusters.loc[row.cluster, 'lat'], clusters.loc[row.cluster, 'lon'] ],  
        [ row.lat, row.lon ]  
      ],
      color='black',
      weight=2,
    ).add_to(m), axis=1)
    clusters.apply(lambda row: folium.CircleMarker(
      location=[row.lat, row.lon], 
      radius=10, 
      fill_opacity=1.0,
      fill_color=f'{row.color}',
      color=f'{row.color}',
      popup=folium.Popup(f'<p><b>CLUSTER</b></br>id <b>{row.name}</b></p>', show=True, sticky=True),
    ).add_to(m), axis=1)
    s, w = stations[['lat', 'lon']].min()
    n, e = stations[['lat', 'lon']].max()
    m.fit_bounds([ [s,w], [n,e] ])
    m.save(f'map_clusters.html')


  # plot
  w, h, d = 12, 7, 150
  fig, ax = plt.subplots(1, 1, figsize=(w, h), dpi=d)
  plt.suptitle(f'Unique device count')
  #print(stats.columns)
  #print(stats)

  filterlist = tot_highest
  filterlist = tot_lowest
  filterlist = ['tot_unique']
  skiplist = ['tot']
  for cid in stats.columns:
    ntag = lbl = cid
    #if ntag in skiplist: continue
    if not ntag in filterlist: continue
    #print(cid)
    # print(stats[cid])
    if cid.endswith('_unique'):
      ax.plot(stats.index, stats[cid], '-', label=lbl)
    else:
      continue

  ax.set_title(f'period {start} -> {stop}, resampling {freq}')
  ax.legend()
  ax.grid()
  ax.tick_params(labelrotation=45)

  if args.show:
    plt.show()
  else:
    plt.savefig(f'{base}_presence_{freq}.png')
  plt.close()