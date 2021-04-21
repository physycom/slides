#! /usr/bin/env python3

# Refs
# https://github.com/python-visualization/folium/blob/master/folium/plugins/heat_map_withtime.py

import json
import geopandas as gpd
import pandas as pd
import folium
from folium.plugins import HeatMapWithTime, MeasureControl, Search


def sim_folium_heat(geogridfile, griddatafile):
  geogrid = gpd.read_file(geogridfile).sort_index()
  geogrid.id = geogrid.id.astype(int)
  geogrid['centroid'] = geogrid.to_crs(epsg=3857).geometry.centroid.to_crs(epsg=4326)
  geogrid['cen_lat'] = geogrid['centroid'].y
  geogrid['cen_lon'] = geogrid['centroid'].x
  #print(geogrid)

  with open(griddatafile) as gin:
    griddata = json.load(gin)['grid_cnt']

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
  #print(gridcnt)

  time_label = []
  data = []
  for ts, dfg in gridcnt.groupby('timestamp'):
    dt = str(pd.to_datetime(ts, unit='s').tz_localize('utc').tz_convert('Europe/Rome'))
    time_label.append(dt)
    data.append(dfg[['cen_lat', 'cen_lon', 'norm_cnt']].values.tolist())
    #print(dt, dfg.cnt.sum())

  #print(time_label)
  #print(data)
  #exit()

  m = folium.Map(control_scale=True)
  HeatMapWithTime(
    data=data,
    index=time_label,
    radius=60,
    #gradient={0.0: 'blue', 0.1: 'lime', 0.3: 'orange', 0.4: 'red'},
    gradient={0.0: 'blue', 0.1: 'orange', 0.5: 'red'},
    min_opacity=0.2,
    max_opacity=0.6,
    use_local_extrema=False,
    display_index=True,
    auto_play=True,
  ).add_to(m)
  MeasureControl().add_to(m)
  s, w = geogrid[['cen_lat', 'cen_lon']].min()
  n, e = geogrid[['cen_lat', 'cen_lon']].max()
  m.fit_bounds([ [s,w], [n,e] ])
  m.save(griddatafile + '.html')
  # FloatImage

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('-g', '--geogrid', required=True, help='Grid geojson file')
  parser.add_argument('-d', '--geodata', required=True, help='Data timeseries file')
  args = parser.parse_args()

  sim_folium_heat(args.geogrid, args.geodata)
