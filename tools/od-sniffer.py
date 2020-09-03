#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

if __name__ == '__main__':
  # parse cli and config
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--input', type=str, required=True)
  parser.add_argument('-s', '--show', action='store_true')
  parser.add_argument('-b', '--bin', action='store_true')
  parser.add_argument('-t', '--dt', type=int, default=300)
  parser.add_argument('-r', '--range', type=str, default='')
  parser.add_argument('-d', '--dev', type=str, default='wifi')
  args = parser.parse_args()
  base = args.input[:args.input.find('_')]

  dt_fmt = '%Y%m%d-%H%M%S'
  freq = f'{args.dt}s'
  if args.range == '':
    tok = args.input[:args.input.rfind('.')].split('_')
    start = datetime.strptime(tok[-2], dt_fmt)
    stop = datetime.strptime(tok[-1], dt_fmt)
  else:
    start = datetime.strptime(args.range.split('|')[0], dt_fmt)
    stop = datetime.strptime(args.range.split('|')[1], dt_fmt)
  base = f'{base}_{start.strftime(dt_fmt)}_{stop.strftime(dt_fmt)}_{args.dev}'

  tnow = datetime.now()
  df = pd.read_csv(args.input, sep=';')
  df.date_time = pd.to_datetime(df.date_time)
  df = df[ (df.date_time >= start) & (df.date_time < stop) ]
  df = df[ df.kind == args.dev ]
  df['station_code'] = df.station_name.str[-2]
  df['date'] = [ t.date() for t in df.date_time ]
  #print(df)
  print(f'Data parse and prep {datetime.now() - tnow} s')


  tnow = datetime.now()
  station_dict = dict(df[['station_name', 'station_code']].groupby(['station_name', 'station_code']).count().index)
  station_dict = { int(v) : k for k, v in sorted(station_dict.items(), key=lambda i: i[1]) }
  nstation = len(station_dict)
  #print( station_dict )
  print(f'Groupby station_name {nstation} took {datetime.now() - tnow} s')

  tnow = datetime.now()
  ndate = len(df[['date']].groupby('date').count())
  #print( station_dict )
  print(f'Groupby date {ndate} took {datetime.now() - tnow} s')

  tnow = datetime.now()
  grp = df.groupby('mac_address')
  uniq_ids = len(grp)
  print(f'Groupby mac_address took {datetime.now() - tnow} s')

  tnow = datetime.now()
  grpd = df.groupby(['date', 'mac_address'])
  print(f'Groupby date+mac_address took {datetime.now() - tnow} s')

  tnow = datetime.now()
  dpath_cnt = 0
  ndactivity = len(grpd)
  dpaths = {}
  dsingle_cnt = 0
  drest_cnt = 0
  dmulti_cnt = 0
  for (d, id), dfg in grpd:
    if len(dfg) < 2:
      dsingle_cnt += 1
      continue
    dmulti_cnt += 1
    path = ''.join(dfg.station_code)
    short_path = re.sub(r'(\d)\1+', r'\1', path)
    if len(short_path) > 1:
      dpath_cnt += 1
      #print(f'{d}, {id}, {dfg.shape}')
      #print( f'path {path} ({len(path)})')
      #print( f'short {short_path} ({len(short_path)})')
    else:
      drest_cnt += 1
    dpaths.setdefault(id, []).append(dfg[['station_code', 'date_time']].values)
  print(f"""daily activity {ndactivity},
daily multi {dmulti_cnt} ({dmulti_cnt/ndactivity*100:.1f}%), daily single {dsingle_cnt} ({dsingle_cnt/ndactivity*100:.1f}%)
daily path {dpath_cnt} ({dpath_cnt/dmulti_cnt*100:.1f}%), daily rest {drest_cnt} ({drest_cnt/dmulti_cnt*100:.1f}%)
""")
  print(f'Path analysis took {datetime.now() - tnow} s')

  idstats = {}
  for id, pathv in dpaths.items():
    path = [ ''.join([ x[0] for x in p ]) for p in pathv ]
    short_path = [ re.sub(r'(\d)\1+', r'\1', p) for p in path ]
    idstats.setdefault(id, {}).update({
      'days_seen' : len(pathv),
      'path' : path,
      'short_path' : short_path
    })
    #if len(pathv) > 2: print(f'{id} -> ({len(pathv)}) {pathv[0:1]}')
  #print(json.dumps(idstats, indent=2))

  #exit(1)

  daycnt = { n+1 : 0 for n in range(100) }
  for id, par in idstats.items():
    daycnt[par['days_seen']] += 1
  daycnt = { d : v for d,v in daycnt.items() if d < 15 }
  print(len(idstats), sum(daycnt.values()))
  for ds, cnt in daycnt.items():
    print(f'Days seen {ds} : {cnt} ({cnt/uniq_ids*100:.1f}%)')

  cnt = {
    str(o) : {
      str(d) : 0
      for d in station_dict.keys()
    }
    for o in station_dict.keys()
  }

  for id, par in idstats.items():
    pathv = par['short_path']
    #print(f'{id} -> {pathv}')
    for p in pathv:
      if len(p) > 1:
        for o,d in zip(p[:-1], p[1:]):
          cnt[o][d] += 1

  #print(cnt)
  cnt = np.asarray([ list(i.values()) for i in cnt.values() ])
  cnt = cnt // ndate
  #print(cnt)

  fig = plt.figure(figsize=(16, 12))
  ax = fig.add_subplot(111)

  im = ax.imshow(cnt, extent=[0, nstation, 0, nstation], origin='lower', interpolation='None', cmap='plasma')
  #im = ax.imshow(cnt, origin='lower', interpolation='None', cmap='viridis')

  for (j,i),label in np.ndenumerate(cnt):
    ax.text(i + 0.5,j + 0.5,label,ha='center',va='center')

  ax.set_xticks([ (n + 0.5) for n in range(len(station_dict))])
  ax.set_yticks([ (n + 0.5) for n in range(len(station_dict))])
  ax.set_xticklabels(station_dict.values(), rotation=45)
  ax.set_yticklabels(station_dict.values())
  ax.set_xlabel('Destination')
  ax.set_ylabel('Origin')

  fig.colorbar(im)
  plt.title(f'O/D Matrix for {start} - {stop}, device {args.dev}, ')
  plt.tight_layout()

  if args.show:
    plt.show()
  else:
    plt.savefig(f'{base}_od.png')


