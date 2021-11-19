#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import pandas as pd
import mysql.connector
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--cfg', help='conf file', type=str, required=True)
  parser.add_argument('-d', '--data', help='data file', type=str)
  parser.add_argument('-t', '--dt', help='freq', type=int, default=3600)
  parser.add_argument('-db', '--db', choices=['mongo', 'mysql'])
  parser.add_argument('-b', '--bin', action='store_true')
  parser.add_argument('-tc', '--tc', help='time [H] for each chunk', default=2)
  parser.add_argument('-s', '--show', action='store_true')

  args = parser.parse_args()

  dubro = os.path.join(os.environ['WORKSPACE'], 'slides', 'work_lavoro', 'dubrovnik')
  if not os.path.exists(dubro): os.mkdir(dubro)

  conf_name = args.cfg.split('.')[0]

  router_validate = f'{dubro}/router_validate'
  if not os.path.exists(router_validate): os.mkdir(router_validate)

  freq = f'{args.dt}s'
  
  with open(args.cfg, encoding='utf-8') as f:
    config = json.load(f)	
  
  start_date = config['start_date']
  stop_date = config['stop_date']
  start = pd.to_datetime(start_date)
  stop = pd.to_datetime(stop_date)
  data_start_label = start_date.replace(':', '').replace('-', '').replace(' ', '_')
  data_stop_label = stop_date.replace(':', '').replace('-', '').replace(' ', '_')
  time_index = pd.date_range(start = start, end = stop, freq = freq)
  stats = pd.DataFrame()

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
    df = df.set_index('time')
    df.to_csv(f'{dubro}/{conf_name}_{data_start_label}_{data_stop_label}.csv', sep=';', index=True)
    df = df.drop(columns=['network'])
  else:
    name_file = args.data
    df = pd.read_csv(name_file, sep =';', parse_dates=True, index_col=[0])
    df = df.drop(columns=['network'])

  for dev, dfs in df.groupby(['device']):
    dfr = dfs.resample(freq).device_counter.sum()
    dfr = (dfr.reindex(time_index, fill_value=0).reset_index().reindex(columns=['device_counter'])).set_index(time_index)
    dfr = dfr.rename(columns= {"device_counter": f'{dev}'})
    if len(stats) == 0:
      stats = dfr
    else:
      stats = pd.concat([stats, dfr], axis=1).fillna(0)

  # device_list = []
  device_list = ['131', '132', '133', '134', '135', '136', '137', '138']
  if len(device_list) != 0: stats = stats[device_list]

  # plot
  w, h, d = 10, 10, 150
  plt.figure(figsize=(w, h), dpi=d)

  cnt = stats.values
  cnt[ cnt > 0 ] = 1
  color_map = plt.imshow(cnt.T, extent=[0, 1, 0, 1], interpolation='none', vmin=0, vmax=1)

  color_map.set_cmap('plasma')

  plt.colorbar(fraction=0.046, pad=0.04)
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
    plt.savefig(f'{router_validate}/router_{data_start_label}_{data_stop_label}_{ptype}_vali_{freq}.png')
  plt.close()
