#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from random import sample
import numpy as np
import pandas as pd
import mysql.connector
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--cfg', type=str, required=True)
  parser.add_argument('-s', '--show', action='store_true')
  parser.add_argument('-t', '--dt', type=int, default=900)
  parser.add_argument('-b', '--bin', action='store_true')
  parser.add_argument('-db', '--db', choices=['mongo', 'mysql'], default='mysql')
  parser.add_argument('-tc', '--tc', help='time [H] for each chunk', default=2)
  parser.add_argument('-sr', '--sr', help='sample rate for network counter', default=5)

  args = parser.parse_args()

  dubro = os.path.join(os.environ['WORKSPACE'], 'slides', 'work_lavoro', 'dubrovnik')
  if not os.path.exists(dubro): os.mkdir(dubro)

  conf_name = args.cfg.split('.')[0]

  base_save = f'{dubro}/router_validate'
  if not os.path.exists(base_save): os.mkdir(base_save)

  freq = f'{args.dt}s'
  
  with open(args.cfg, encoding='utf-8') as f:
    config = json.load(f)	
  
  start_date = config['start_date']
  stop_date = config['stop_date']
  if args.db == 'mysql':
    conf = config['model_data']['params']['dubrovnik']['mysql']
    db = mysql.connector.connect(
        host     = conf['host'],
        port     = conf['port'],
        user     = conf['user'],
        passwd   = conf['pwd'],
        database = conf['db']
        )
    cursor = db.cursor()

    time_chunk = int(args.tc)
    start = pd.to_datetime(start_date)
    stop = pd.to_datetime(stop_date)
    data_inizio = start_date.replace(':', '').replace('-', '').replace(' ', '_')
    data_fine = stop_date.replace(':', '').replace('-', '').replace(' ', '_')
    tnow = start
    df_list = []
    while tnow < stop:
      try:
        trange = tnow + timedelta(hours=time_chunk)
        query = f"""
        SELECT
          de.eventOccurredAt AS time,
          de.id_device AS device,
          ds.name AS name,
          ds.serial AS serial,
          ds.networkId AS network,
          de.eventClientiId AS mac_address,
          COUNT(de.eventClientiId) as device_counter
        FROM
          DevicesEvents de
        JOIN
          Devices ds
        WHERE
         de.eventOccurredAt > ('{tnow}') AND de.eventOccurredAt < ('{trange}')
         AND de.eventClientiId != ''
         AND de.id_device = ds.id
        GROUP BY de.eventClientiId, de.id_device
        ORDER BY eventOccurredAt ASC
       """
        # print(query)
        tquery = datetime.now()
        cursor.execute(query)
        result = cursor.fetchall()
        tquery = datetime.now() - tquery
        print(f'Received {len(result)} mysql data in {tquery} for sub-query from {tnow} to {trange}')
        df = pd.DataFrame(result)
        df.columns =  cursor.column_names
        df_list.append(df)
      except Exception as e:
        print('Connection error : {}'.format(e))

      tnow = trange
    df = pd.concat(df_list)
    df.to_csv(f'{dubro}/{conf_name}_{data_inizio}_{data_fine}.csv', sep=';', index=True)
  time_index = pd.date_range(start = start, end = stop, freq = freq)
  stats = pd.DataFrame()
  df.index = df.time
  df_network = df.copy()
  df = df.drop(columns=['time', 'network'])
  print(df)

  sample_rate = (str(args.sr) + 'T')
  df_network = df_network.groupby(['network']).resample(sample_rate)['device_counter'].sum()
  df_network = df_network.to_frame().reset_index()
  df_network.index = df_network.time
  df_network = df_network.drop(columns='time')
  df_network.to_csv(f'{dubro}/{conf_name}_network_{data_inizio}-{data_fine}.csv', sep=';', index=True)

  for device, dfs in df.groupby(['device']):
    dfr = dfs.resample(freq).sum()
    dfr = (dfr.reindex(time_index, fill_value=0).reset_index().reindex(columns=['COUNTER'])).set_index(time_index)
    dfr.columns = [f'{device}']
    if len(stats) == 0:
      stats = dfr
    else:
      stats = pd.concat([stats, dfr], axis=1).fillna(0)

  # plot
  w, h, d = 10, 10, 150
  plt.figure(figsize=(w, h), dpi=d)

  cnt = stats.values
  if args.bin:
    cnt[ cnt > 0 ] = 1
    color_map = plt.imshow(cnt.T, extent=[0, 1, 0, 1], interpolation='none', vmin=0, vmax=1)
  else:
    color_map = plt.imshow(cnt.T, extent=[0, 1, 0, 1], interpolation='none')

  #color_map = plt.imshow(cnt.T, extent=[0, 1, 0, 1], vmin = 0)
  color_map.set_cmap('plasma')

  plt.colorbar()
  plt.title(f'Router data presence\nfrom {start_date} to {stop_date} @ {freq}')

  dy = 1 / len(stats.columns)
  plt.gca().set_yticks([ (n + 0.5)*dy for n in range(len(stats.columns))])
  plt.gca().set_yticklabels(stats.columns[::-1])

  dx = 1 / len(stats.index)
  max_ticks = 20
  plt.gca().set_xticks([ (n + 0.5)*dx for n in range(len(stats.index))])
  lbls = [ t.strftime('%Y %b %d %H:%M') for t in stats.index]
  if len(lbls) > max_ticks:
    for i in range(len(lbls)):
      if i % (len(lbls) // max_ticks) != 0:
        lbls[i] = ''
  plt.gca().set_xticklabels(lbls, rotation=45, ha='right')

  plt.tight_layout()
  if args.show:
    plt.show()
  else:
    ptype = 'bin' if args.bin else 'cmap'
    plt.savefig(f'{base_save}/router_{data_inizio}_{data_fine}_{ptype}_vali_{freq}.png')
  plt.close()
