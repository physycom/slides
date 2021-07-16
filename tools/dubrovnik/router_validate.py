#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import pandas as pd
import mysql.connector
from datetime import datetime
import matplotlib.pyplot as plt

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--cfg', type=str, required=True)
  parser.add_argument('-s', '--show', action='store_true')
  parser.add_argument('-t', '--dt', type=int, default=900)
  parser.add_argument('-b', '--bin', action='store_true')
  parser.add_argument('-db', '--db', choices=['mongo', 'mysql'], default='mysql')
  args = parser.parse_args()

  dubro = os.path.join(os.environ['WORKSPACE'], 'slides', 'work_lavoro', 'dubrovnik')
  if not os.path.exists(dubro): os.mkdir(dubro)

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
    query = f"""
    SELECT
        ds.eventOccurredAt AS 'DATETIME',
        ds.id_device AS station,
        COUNT(ds.eventClientiId) as COUNTER
        #ds.eventClientiId
        #ds.*
    FROM
        DevicesEvents ds
    WHERE
        ds.eventOccurredAt > ('{start_date}') AND ds.eventOccurredAt < ('{stop_date}') 
        AND ds.eventClientiId != ''
        GROUP BY ds.eventClientiId, ds.id_device
        ORDER BY ds.eventOccurredAt ASC
    """    
    # print(query)
    tquery = datetime.now()
    cursor.execute(query)
    result = cursor.fetchall()
    # print(result)
    tquery = datetime.now() - tquery
    print(f'Received {len(result)} mysql data in {tquery}')
    df = pd.DataFrame(result)
    df.columns =  cursor.column_names
    df.index = df.DATETIME
    df = df.drop(columns='DATETIME')

  start_time = pd.to_datetime(start_date)
  end_time = pd.to_datetime(stop_date)

  time_index = pd.date_range(start = start_time, end = end_time, freq = freq)
  stats = pd.DataFrame()

  for station, dfs in df.groupby(['station']):
    dfr = dfs.resample(freq).sum()
    dfr = (dfr.reindex(time_index, fill_value=0).reset_index().reindex(columns=['COUNTER'])).set_index(time_index)
    dfr.columns = [f'{station}']
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
    s_date = start_date.replace('-','').replace(':','').replace(' ','_')
    e_date = stop_date.replace('-','').replace(':','').replace(' ','_')
    ptype = 'bin' if args.bin else 'cmap'
    plt.savefig(f'{base_save}/router_{s_date}_{e_date}_{ptype}_vali_{freq}.png')
  plt.close()
