#! /usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Ricostruisco gli spostamenti degli id rilevati dagli sniffer e li rappresentato su un grafo.
-start -> rappresenta la stazione da cui partono gli id da trackare
-step -> rappresenta il numero di stazioni che vengono visitate
'''

import json
import argparse
import numpy as np
import pandas as pd
import datetime as dt
import networkx as nx
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from random import randint
from datetime import datetime
from itertools import permutations, combinations

if __name__ == '__main__':
  # parse cli and config
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--input', type=str, required=True)
  parser.add_argument('-s', '--show', action='store_true')
  parser.add_argument('-b', '--bin', action='store_true')
  parser.add_argument('-t', '--dt', type=int, default=300)
  parser.add_argument('-e', '--thresh', type=int, default=100)
  parser.add_argument('-r', '--range', type=str, default='')
  parser.add_argument('-d', '--dev', type=str, default='wifi')
  parser.add_argument('-o', '--top', type=int, default=10)
  parser.add_argument('-g', '--graph', default=None)
  parser.add_argument('-step', '--step', type=int, default=3)
  parser.add_argument('-start', '--startStation', type=str, default='Piazza Stazione isola (6)')
  # parser.add_argument('-o', '--top', type=int, default=5000)

  args = parser.parse_args()
  base = args.input[:args.input.find('_')]


  dt_fmt = '%Y%m%d-%H%M%S'
  freq = f'{args.dt}s'
  if args.range == '':
    tok = args.input[:args.input.rfind('.')].split('_')
    try:
      start = datetime.strptime(tok[-2], dt_fmt)
      stop = datetime.strptime(tok[-1], dt_fmt)
    except:
      start = datetime.strptime(tok[-3], dt_fmt)
      stop = datetime.strptime(tok[-2], dt_fmt)
  else:
    start = datetime.strptime(args.range.split('|')[0], dt_fmt)
    stop = datetime.strptime(args.range.split('|')[1], dt_fmt)
  base = f'{base}_{start.strftime(dt_fmt)}_{stop.strftime(dt_fmt)}_{args.dev}'

  if args.graph is None:
    df = pd.read_csv(args.input, sep=';')
    df.date_time = pd.to_datetime(df.date_time)
    df.date_time = df.date_time.dt.tz_localize(None)
    df = df.sort_values('date_time')
    df = df[ (df.date_time >= start) & (df.date_time < stop) ]
    df = df[ df.kind == args.dev ]

    dft = df[['mac_address','date_time','station_name']]
    dft.to_csv(f'{base}_dft.csv', sep=';')
  else:
    parse_date = ['date_time']
    dft = pd.read_csv(args.graph, sep=';', parse_dates=parse_date)

  dict_od = {}

  list_stations = dft['station_name'].unique()
  step_num = args.step
  n_chiara = 0
  for i in list_stations:
    dict_i = {}
    for j in list_stations:
      dict_j = {}
      if i != j:
        for s in np.arange(0,step_num):
          dict_j[s]=0
          dict_i[j] = dict_j
          n_chiara += 1
    dict_od[i] = dict_i

  station_arrival_id = []
  dft_grp = dft.groupby(by='mac_address')
  for k, v in dft_grp:
    v = v.sort_values('date_time')
    if not v.empty and len(v) >= 2 and len(v['station_name'].unique())>1:
      n_step_w = 0
      row1 = v.iloc[0]
      if row1['station_name'] == args.startStation:
        station_arrival_id.append(k)
        v['next_station'] = v.station_name.shift(-1)
        if pd.isnull(v.next_station.iloc[-1]):
          v.next_station.iloc[-1] = v.next_station.iloc[-2]
        for r in v.iterrows():
          act_station = r[1]['station_name']
          next_station = r[1]['next_station']
          if act_station != next_station and n_step_w < step_num:
            dict_od[act_station][next_station][n_step_w] += 1
            n_step_w += 1

  col_name_od = ['station_name','next_station','step','n']
  df_od =pd.DataFrame(columns=col_name_od)
  for ik, iv in dict_od.items():
    for jk, jv in iv.items():
      for sk, sv in jv.items():
        if sv != 0:
          df_row = pd.DataFrame([[ik,jk,sk,sv]], columns=col_name_od)
          df_od = df_od.append(df_row, ignore_index=True)

  pid_l = []
  for pid, dfg in dft.groupby(['mac_address']):
    if pid in station_arrival_id:
      df_date = [ t.date() for t in dfg.date_time ]
      s = dfg.shape[0]
      days = len(np.unique(df_date))
      stat = len(np.unique(dfg.station_name))
      pid_l.append([pid,s,days,stat])
  pid_df = pd.DataFrame(pid_l, columns = ['pid','n','days','stations'])

  tsl = list(pid_df.n)
  tsl_2stations  = list(pid_df[pid_df.stations>1].n)
  tsl_2days = list(pid_df[pid_df.days>1].n)

  thresh = args.thresh
  tsl_t = [t for t in tsl if t > thresh]
  tsl_2dt = [t for t in tsl_2days if t > thresh]
  tsl_2st = [t for t in tsl_2stations if t > thresh]
  top_pids = top_pids = list(pid_df[(pid_df.stations>1)].sort_values(by='n',ascending=False)[:args.top].pid)

#%% plot 1
  w, h, d = 12, 7, 150
  fig, ax = plt.subplots(1, 1, figsize=(w, h), dpi=d)
  plt.yscale('log')
  plt.title(f'Device type {args.dev}, ts id count with at least {thresh} timestamps')
  plt.xlabel('Countings')
  plt.ylabel('Number of IDs')
  ax.hist(tsl_t, bins=30, alpha = 0.5, label = 'All data')
  ax.hist(tsl_2dt, bins=30, alpha = 0.5, color = 'r', range = [min(tsl_t),max(tsl_t)], label = 'Only 2 or more days presence')
  ax.hist(tsl_2st, bins=30, alpha = 0.3, color = 'b', range = [min(tsl_t),max(tsl_t)], label = 'Only 2 or more stations presence')
  plt.legend()
  # plt.plot(dftest.date_time,dftest.station_name, '.')

  if args.show:
    plt.show()
  else:
    plt.savefig(f'{base}_{thresh}_idcounts.png')
  plt.clf()

#%% plot 2
  w, h, d = 12, 7, 150
  fig, ax = plt.subplots(1, 1, figsize=(w, h), dpi=d)
  plt.title(f'Device type {args.dev}, top {args.top} ids with at least 2 stations presence movements')
  for p in top_pids:
    dft_p = dft[dft['mac_address']==p]
    plt.plot(dft_p.date_time, dft_p.station_name, linewidth=1, alpha=0.6)
  plt.tight_layout()

  if args.show:
    plt.show()
  else:
    plt.savefig(f'{base}_{args.top}_movements.png')
  plt.clf()

#%% plot 3
  totals = df_od.groupby(['step']).sum()['n']

  dict_total={}
  for (i,j) in zip(totals.index.to_list(), totals.values):
    dict_total[i]=j


  norm_list = []
  for i in df_od.iterrows():
    norm_list.append(i[1]['n']/dict_total[i[1]['step']])
  df_od['norm'] = norm_list

  decimals = 2
  df_od['norm'] = df_od['norm'].apply(lambda x: round(x, decimals))

  colors = []
  for i in np.arange(0, step_num):
    colors.append('#%06X' % randint(0, 0xFFFFFF))

  dict_colors = {}
  dict_color = {}
  for sn in np.arange(0, step_num):
    dict_color = {sn:colors[sn]}
    dict_colors.update(dict_color)

  df_od['color'] = [dict_colors[i] for i in df_od['step']]

  fixed_positions = {'Piazza Stazione isola (6)':(0,8), 'Hotel Carlton (2)':(2,4), 'Piazza Trento Trieste (5)':(4.5,3.5), 'Via del Podestà (3)':(3,3), 'Castello via martiri (1)':(4,5)}

  for step, df in df_od.groupby(['step']):

    G = nx.from_pandas_edgelist(df, source="station_name", target="next_station", create_using=nx.DiGraph(), edge_attr='norm')

    df = df.drop(columns=['step','n'])

    dict_edge={}
    for r in df.iterrows():
      tuple_od = (r[1]['station_name'], r[1]['next_station'])
      dict_edge[tuple_od] = str(int(r[1]['norm']*100)) +'%'

    station_found = []
    station_found = df.next_station.to_list() + df.station_name.to_list()

    fixed_positions_in = {}
    for key in fixed_positions.keys():
      if key in station_found:
        fixed_positions_in.update({key:fixed_positions[key]})

    fixed_nodes = fixed_positions_in.keys()
    pos = nx.spring_layout(G, pos=fixed_positions_in, fixed=fixed_nodes)

    plt.figure(figsize=(12,8))
    plt.title(f'{args.dev} tracker from {start} to {stop}\nStart from {args.startStation}\nStep {step + 1}', fontweight='bold')


    nx.draw(G, pos, with_labels=False, arrows=True, arrowsize=50, arrowstyle='fancy', edge_color =df['color'], node_size=60, font_size=12)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=dict_edge, label_pos=0.5, font_color='black' )

    for p in pos:
      pos[p][1] += 0.1
    nx.draw_networkx_labels(G, pos)

    plt.tight_layout()

    if args.show:
      plt.show()
    else:
      plt.savefig(f'{base}_{args.top}_step_{step + 1}_start_{args.startStation}_graph.png')
    plt.clf()
