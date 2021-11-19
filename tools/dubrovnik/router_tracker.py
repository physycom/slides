#! /usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from random import randint
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime, timedelta

router_group = defaultdict(
  first_group = [145, 146, 147],  # West Center
  second_group = [122, 123, 126], # Center
  third_group = [131, 138, 157]   # Lazareto
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

def tracker(data, freq):
  dubro = os.path.join(os.environ['WORKSPACE'], 'slides', 'work_lavoro', 'dubrovnik')
  if not os.path.exists(dubro): os.mkdir(dubro)
  base = f'{dubro}/router_tracker'
  if not os.path.exists(base): os.mkdir(base)

  df = pd.read_csv(data, sep =';', parse_dates=True, index_col=[0])

  tok = data[:data.find('.')].split('_')
  start = tok[-4] + "-" + tok[-3]
  stop = tok[-2] + "-" + tok[-1]

  dt_fmt = '%Y%m%d-%H%M%S'

  if args.range == '':
    tok = data[:data.find('.')].split('_')
    start = tok[-4] + "-" + tok[-3]
    stop = tok[-2] + "-" + tok[-1]
    try:
      start = datetime.strptime(start, dt_fmt)
      stop = datetime.strptime(stop, dt_fmt)
    except Exception as e:
      print(f'Exception: {e}')
  
  else:
    start = datetime.strptime(args.range.split('|')[0], dt_fmt)
    stop = datetime.strptime(args.range.split('|')[1], dt_fmt)

  df = df[ (df.index >= start) & (df.index < stop) ]
  df = df[['mac_address','device']]

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

  df['group'] = df.device.map(inv_router_group)

  dict_od = {}
  list_stations = df['group'].unique()
  step_num = 1
  n_step = 0
  for i in list_stations:
    dict_i = {}
    for j in list_stations:
      dict_j = {}
      if i != j:
        for s in np.arange(0,step_num):
          dict_j[s]=0
          dict_i[j] = dict_j
          n_step += 1
    dict_od[i] = dict_i

  group_arrival_id = []
  total_id = []
  for k, v in df.groupby(by='mac_address'):
    total_id.append(k)
    v = v.sort_values('date_time')
    if not v.empty and len(v) >= 2 and len(v['group'].unique())>1:
      n_step_w = 0
      row1 = v.iloc[0]
      for num, which_group in enumerate(list_stations):
        if row1['group'] == list_stations[num]:
          group_arrival_id.append(k)
          v['next_group'] = v.group.shift(-1)
          if pd.isnull(v.next_group.iloc[-1]):
            pd.options.mode.chained_assignment = None  # default='warn'
            v.next_group.iloc[-1] = v.next_group.iloc[-2]
          for r in v.iterrows():
            act_group = r[1]['group']
            next_group = r[1]['next_group']
            if act_group != next_group and n_step_w < step_num:
              dict_od[act_group][next_group][n_step_w] += 1
              n_step_w += 1

  col_name_od = ['group_name','next_group','step','n']
  df_od =pd.DataFrame(columns=col_name_od)
  for ik, iv in dict_od.items():
    for jk, jv in iv.items():
      for sk, sv in jv.items():
        if sv != 0:
          df_row = pd.DataFrame([[ik,jk,sk,sv]], columns=col_name_od)
          df_od = df_od.append(df_row, ignore_index=True)
    
  totals = df_od.groupby(['step']).sum()['n']

  dict_total={}
  for (i,j) in zip(totals.index.to_list(), totals.values):
    dict_total[i]=j

  norm_list = []
  for i in df_od.iterrows():
    norm_list.append(i[1]['n']/dict_total[i[1]['step']])
  df_od['norm'] = norm_list

  decimals = 4
  df_od['norm'] = df_od['norm'].apply(lambda x: round(x, decimals))

  colors = []
  for i in np.arange(0, step_num):
    # colors.append('#%06X' % randint(0, 0xFFFFFF))
    colors.append('#D850EC')

  dict_colors = {}
  dict_color = {}
  for sn in np.arange(0, step_num):
    dict_color = {sn:colors[sn]}
    dict_colors.update(dict_color)

  df_od['color'] = [dict_colors[i] for i in df_od['step']]

  fixed_positions = {'first_group':(0,4.2), 'second_group':(2,4), 'third_group':(3.5,5.5),}

  print(df_od)

  for step, df in df_od.groupby(['step']):
    G = nx.from_pandas_edgelist(df, source="group_name", target="next_group", create_using=nx.DiGraph(), edge_attr='norm')
    df = df.drop(columns=['step','n'])
    dict_edge={}
    for r in df.iterrows():
      tuple_od = (r[1]['group_name'], r[1]['next_group'])
      dict_edge[tuple_od] = str(round(r[1]['norm']*100, 2)) +'%'

    station_found = []
    station_found = df.next_group.to_list() + df.group_name.to_list()

    fixed_positions_in = {}
    for key in fixed_positions.keys():
      if key in station_found:
        fixed_positions_in.update({key:fixed_positions[key]})

    fixed_nodes = fixed_positions_in.keys()
    pos = nx.spring_layout(G, pos=fixed_positions_in, fixed=fixed_nodes)

    plt.figure(figsize=(12,8))
    plt.title(f'People tracker from {start} to {stop}\nStart from  \nStep {step + 1}', fontweight='bold')

    nx.draw(G, pos, with_labels=False, arrows=True, arrowsize=20, edge_color =df['color'], node_size=60, font_size=12)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=dict_edge, label_pos=0.3, font_color='black' )

    for p in pos:
      pos[p][1] += 0.1
    nx.draw_networkx_labels(G, pos)

    plt.tight_layout()
    plt.show()
    plt.savefig(f'{base}/test_step_{step + 1}_graph.png')
    plt.clf()
    plt.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-d', '--data', help='conf data csv', required=True)
  parser.add_argument('-r', '--range', type=str, default='')
  parser.add_argument('-f', '--freq', help='freq data', type=int, default = 3600)

  args = parser.parse_args()

  data = args.data
  freq = f'{args.freq}s'
  tracker(data, freq)