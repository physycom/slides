#! /usr/bin/env python3

# Refs
# https://github.com/python-visualization/folium/blob/master/folium/plugins/heat_map_withtime.py

import json
import geopandas as gpd
import pandas as pd
import folium
import branca.colormap as cm
import os
from folium.plugins import HeatMapWithTime, MeasureControl, Search


def sim_folium_heat(geogridfile, griddatafile):
  #if os.path.exists(geogridfile):
  #  print( "ok esiste" )
  geogrid = gpd.read_file(geogridfile).sort_index()
  geogrid.id = geogrid.id.astype(int)
  geogrid['centroid'] = geogrid.to_crs(epsg=3857).geometry.centroid.to_crs(epsg=4326)
  geogrid['cen_lat'] = [ p.y for p in geogrid['centroid'] ]
  geogrid['cen_lon'] = [ p.x for p in geogrid['centroid'] ]
  

  with open(griddatafile) as gin:
    griddata = json.load(gin)

  sim_id = griddata['sim_id']
  griddata = griddata['grid_cnt']

  griddata = [ {
      'timestamp' : ts,
      'cell_id' : int(gid),
      'cnt' : val,
    } for ts, cnt in griddata.items() for gid, val in cnt.items()
  ]
  gridcnt = pd.DataFrame.from_dict(griddata)
  #gridcnt['norm_cnt'] = (gridcnt.cnt - gridcnt.cnt.min()) / (gridcnt.cnt.max() - gridcnt.cnt.min())
  gridcnt['norm_cnt'] = gridcnt.cnt / gridcnt.cnt.max()

  #print(gridcnt)
  gridcnt = gridcnt.merge(geogrid[['id', 'cen_lat', 'cen_lon']], left_on='cell_id', right_on='id')

  time_label = []
  data = []
  for ts, dfg in gridcnt.groupby('timestamp'):
    #dt = f"{pd.to_datetime(ts, unit='s').tz_localize('utc').tz_convert('Europe/Rome')}"
    dt = str(pd.to_datetime(ts, unit='s').tz_localize('utc').tz_convert('Europe/Rome'))
    time_label.append(dt)
    data.append(dfg[['cen_lat', 'cen_lon', 'norm_cnt']].values.tolist())
    # print(dt, dfg.cnt.sum())

  #print(time_label)
  m = folium.Map(control_scale=True)
  _radius = 60
  
  if "ubrovnik" in str(geogridfile):
    _radius = 100
  HeatMapWithTime(
    data=data,
    index=time_label,
    radius=_radius,
    # gradient={'0':'Navy', '0.25':'Blue','0.5':'Green', '0.75':'Yellow', '0.85':'orange','1': 'Red'},
    gradient={'0':'gray', '0.2':'Blue','0.4':'Green', '0.6':'Yellow', '0.8':'orange','1': 'Red'},
    # gradient={0.0: 'blue', 0.1: 'orange', 0.5: 'red'},
    min_opacity=0.2,
    max_opacity=0.6,
    use_local_extrema=False,
    display_index=True,
    auto_play=True,
  ).add_to(m)
  MeasureControl().add_to(m)
  colormap = cm.LinearColormap( colors=['gray', 'Blue', 'Green', 'Yellow', 'orange', 'red'],
                                index=[0, 0.2, 0.4, 0.6, 0.8, 1],
                                # tick_labels=[0, 25, 50, 75, 85, 100],
                                caption='Density of population') 
  m.add_child(colormap)

  s, w = geogrid[['cen_lat', 'cen_lon']].min()
  n, e = geogrid[['cen_lat', 'cen_lon']].max()
  m.fit_bounds([ [s,w], [n,e] ])
  text_save = f'{html_folder}/heatmap_{sim_id}.html' 
  m.save(text_save)
  print(f'* File saved in: {text_save}')

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('-g', '--geogrid', required=True, help='Grid geojson file')
  parser.add_argument('-d', '--geodata', required=True, help='Data timeseries file')
  args = parser.parse_args()

  html_folder = os.path.join(os.environ['WORKSPACE'], 'slides', 'work_ws', 'output', 'statefile', 'html_folder')
  if not os.path.exists(html_folder): os.mkdir(html_folder)

  sim_folium_heat(args.geogrid, args.geodata)
