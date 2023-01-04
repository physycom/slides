#! /usr/bin/env python3

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import collections
from datetime import datetime, timedelta

def sim_heat(gridfile, datafile):
  grid = {}
  data = collections.defaultdict(dict)

  with open(gridfile) as f:
    geogrid = json.load(f)
  for i in geogrid['features']:
    grid[i['properties']['id']] = np.mean(i['geometry']['coordinates'][0], 0)

  df = pd.read_csv(datafile, sep=' ', index_col=None, header=None)
  df['gid'] = df[0].str.split('=', expand=True)[1]
  df['cnt'] = df[1].str.split('=', expand=True)[1]
  df['ts'] = df[2] * 1e-9
  df = df[['gid','ts','cnt']].astype({'ts':'int', 'gid':'int'})
  for row in df.values:
    data[row[0]][row[1]] = row[2]
  geojson = {
    'type': 'FeatureCollection',
    'features': [],
  }

  # sanity checks and various init
  geojson['times'] = list(map(lambda x: datetime.fromtimestamp(x).strftime("%Y%m%d_%H%M%S"), data[0].keys()))

  for k, v in data.items():
    feat = {
      'type': 'Feature',
      'properties': {
        'time_cnt' : []
      },
      'geometry': {
        'type': 'Point',
        'coordinates': []
      }
    }
    feat['properties']['time_cnt'] = list(map(int, v.values()))
    feat['geometry']['coordinates'] = list(grid[k])
    geojson['features'].append(feat)

  with open('check_geo.geojson', 'w') as gout:
    json.dump(geojson, gout, indent=2)

  heattemplfile = os.path.join(os.environ['WORKSPACE'], 'covid_hep', 'vars', 'templates', 'heatmap_template.html')
  with open(heattemplfile, 'r') as file:
    heattempl = file.read()
  
  base = datafile[ :datafile.rfind('.') ]
  with open(f'{base}_heatmap.html', 'w') as file:
    file.write(heattempl.replace('@@GEOJSON_CONTENT@@', json.dumps(geojson)))
  

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('-g', '--grid', help='grid geojson file', required=True)
  parser.add_argument('-d', '--data', help='grid data influx-ready file', required=True)
  args = parser.parse_args()

  sim_heat(gridfile=args.grid, datafile=args.data)

