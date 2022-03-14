#! /usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import argparse
import glob
import json
import os
import re
import sys
import warnings
import pytz
from datetime import datetime
from collections import defaultdict
import folium
from matplotlib import cm
from matplotlib.colors import to_hex, rgb2hex
from shapely.errors import ShapelyDeprecationWarning


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--cartogeo', help='carto geojson file', required=True)
  parser.add_argument('-d', '--data', help='input sim_response json', required=True)
  args = parser.parse_args()

  html_folder = os.path.join(os.environ['WORKSPACE'], 'slides', 'work_ws', 'output', 'statefile', 'html_folder')
  if not os.path.exists(html_folder): os.mkdir(html_folder)

  with open(args.data) as data:
    cnt_dict = json.load(data)

  sim_id = cnt_dict['sim_id']
  cnt_dict = cnt_dict['poly_cnt']
  cnt = pd.DataFrame.from_dict(cnt_dict).T
  cnt['datetime'] = pd.to_datetime(cnt.index, unit='s') + pd.Timedelta('2H')
  cnt.set_index('datetime', inplace=True)
  cnt = cnt.interpolate()

  ## Tot population
  cnt_pop = cnt.copy()
  cnt_pop['total'] = cnt_pop.sum(axis=1)
  cnt_tot = cnt_pop[['total']]
  cnt_tot.plot()
  # plt.show()

  cnt = cnt.sum().to_frame()
  cnt.index = cnt.index.astype(int)
  cnt = cnt.rename(columns={0:'cnt_sum'})
  cnt['poly_lid'] = cnt.index

  carto = gpd.read_file(args.cartogeo)
  carto = pd.merge(carto, cnt, on=['poly_lid'])
  import matplotlib.pyplot as plt
  fig, ax = plt.subplots()
  cnt['cnt_sum'].hist(ax=ax, bins=70, rwidth=0.8)

  quantiles_val = [0.0, 0.20, 0.40, 0.60, 0.80, 0.90, 1.0]
  quantiles = cnt['cnt_sum'].quantile(quantiles_val).to_list()

  bins = [int(q) for q in quantiles]
  cmap = cm.get_cmap('Wistia', len(bins)-1)

  rgb_scale = cmap(quantiles_val)
  color_scale = [rgb2hex(r) for r in rgb_scale]
  color_scale = color_scale[:-1]
  default_color = color_scale[1]
  map_col_quantil={}
  for i in np.arange(0, len(color_scale)):
    label = f'{quantiles_val[i]} < q <= {quantiles_val[i+1]}'
    map_col_quantil[color_scale[i]]=label
  ax.set_yscale('log')

  list_color = pd.cut(cnt['cnt_sum'], bins=bins, labels=color_scale)
  carto['color'] = list_color.to_list()
  carto.color.fillna(default_color, inplace=True)
  carto['quantile'] = [map_col_quantil[i] for i in carto.color]
  carto['cnt'] = cnt['cnt_sum'].to_list()
  # layers
  center = carto.to_crs(epsg=3003).centroid.to_crs(epsg=4326)
  center = [ center.y.mean(), center.x.mean() ]

  carto.sort_values(by='quantile', inplace=True)

  warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
  m = folium.Map(location=center, control_scale=True)
  layerlabel = '<span style="color: {col};">{txt}</span>'
  for qval, grp in carto.groupby('quantile'):
    col = grp.color.to_list()[0]
    flayer_sel = folium.FeatureGroup(name=layerlabel.format(col=col, txt=f'{qval}'))
    for i, row in grp.iterrows():
      start_coord, end_coord = row.geometry.boundary
      begin = (str(start_coord).split(' ',1)[1])[1:-1]
      begin = begin.split(' ')
      begin = '(' + ' '.join(reversed(begin)) + ')'

      end = (str(end_coord).split(' ',1)[1])[1:-1]
      end = end.split(' ')
      end = '(' + ' '.join(reversed(end)) + ')'

      pol = folium.PolyLine(
        locations=[ [lat,lon] for lon,lat in row.geometry.coords],
        color=col,
        # popup=folium.Popup(f'Poly lid: <b>{row.poly_lid}</b></br>Daily Counts: <b>{round(row.cnt)}</b></p>', show=False, sticky=True),
        popup=folium.Popup(f'Daily Counts: <b>{round(row.cnt)}</b></p>', show=False, sticky=True),
      )
      flayer_sel.add_child(pol)
    m.add_child(flayer_sel)

  folium.map.LayerControl(collapsed=False).add_to(m)
  e, s, w, n = carto.total_bounds
  m.fit_bounds([ [s,w], [n,e] ])
  text_save = f'{html_folder}/fluxes_map_{sim_id}.html' 
  m.save(text_save)
  print(f'* File saved in: {text_save}')
