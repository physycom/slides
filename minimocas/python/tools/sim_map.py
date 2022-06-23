#! /usr/bin/env python3

import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import geopandas as gpd
import folium

def sim_map(confin):
  with open(confin) as cin:
    confj = json.load(cin)

  dfa = pd.DataFrame.from_dict(confj['attractions']).transpose()
  dfa['attr_name'] = [ i for i in dfa.index ]
  dfa = dfa.reset_index(drop=True)
  dfa['id'] = dfa.index
  print(dfa)

  dfs = pd.DataFrame.from_dict(confj['sources']).transpose()
  dfs = dfs.drop(labels='LOCALS')
  dfs = dfs[['source_location']]
  dfs['lat'] = dfs.source_location.apply(lambda x: x['lat'])
  dfs['lon'] = dfs.source_location.apply(lambda x: x['lon'])
  dfs = dfs.drop(columns='source_location')
  dfs['src_name'] = dfs.index  
  dfs = dfs.reset_index(drop=True)
  dfs['id'] = dfs.index
  print(dfs)
  
  print(dfs.columns)

  gdfa = gpd.GeoDataFrame(dfa, geometry=gpd.points_from_xy(dfa.lon, dfa.lat))
  gdfa = gdfa.set_crs(epsg=4326)
  map_center = gdfa[['lat', 'lon']].mean().values
  
  m = folium.Map(location=map_center, control_scale=True, zoom_start=9)
  dfa.apply(lambda row: folium.CircleMarker(
    location=[row.lat, row.lon], 
    radius=7, 
    fill_color='blue',
    color='blue',
    popup=folium.Popup(f'<p><b>ATTRACTION</b></br>id <b>{row.id}</b></br>name <b>{row.attr_name}</b></p>', show=False, sticky=True),
  ).add_to(m), axis=1)
  dfs.apply(lambda row: folium.CircleMarker(
    location=[row.lat, row.lon], 
    radius=7, 
    fill_color='red',
    color='red',
    popup=folium.Popup(f'<p><b>SOURCE</b></br>id <b>{row.id}</b></br>name <b>{row.src_name}</b></p>', show=False, sticky=True),
  ).add_to(m), axis=1)
  base = confin[:confin.rfind('.')]
  e, s, w, n = gdfa.total_bounds
  m.fit_bounds([ [s,w], [n,e] ])
  m.save(f'{base}_map.html')


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--confin', help='config json input', default=None)
  args = parser.parse_args()

  sim_map(confin=args.confin)
