#! /usr/bin/env python3

# Refs
# https://github.com/python-visualization/folium/blob/master/folium/plugins/heat_map_withtime.py

import pandas as pd
import folium
from folium.plugins import HeatMap, HeatMapWithTime

def generateBaseMap(default_location=[45.4366884, 12.3203368], default_zoom_start=12):
  base_map = folium.Map(location=default_location, control_scale=True, zoom_start=default_zoom_start)
  return base_map

df = pd.read_csv('/var/www/html/symfony4/Venezia/Streets/slides_venezia2.csv', sep=',')
print(df)
lat, lon = df[['lat', 'lon']].mean().values
print(lat, lon)
base_map = generateBaseMap(default_location=[lat, lon])

if 0:
  df = df[['lat', 'lon', 'time']].groupby(['lat', 'lon']).count().reset_index()
  HeatMap(
    data=df.values.tolist(),
    radius=8,
    max_zoom=13
  ).add_to(base_map)
  base_map.save('/var/www/html/symfony4/public/f/foliumheat_venezia.html')

if 1:
  df['date'] = pd.to_datetime(df.time, unit='s')
  df = df.set_index('date')
  print(df)
  dft = []
  ts = []
  for t, dfi in df.resample('60T'):
    #print(t, dfi)
    dfg = dfi.groupby(['lat', 'lon']).count().reset_index()
    ts.append(t.strftime('%Y/%m/%d %H:%M'))
    dft.append(dfg.values.tolist())
  #print(dft)
  #exit(1)
  HeatMapWithTime(
    data=dft,
    index=ts,
    radius=5,
    gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'},
    min_opacity=0.5,
    max_opacity=0.8,
    use_local_extrema=True,
    display_index=True
  ).add_to(base_map)
  base_map.save('/var/www/html/symfony4/public/f/foliumaniheat_venezia.html')

