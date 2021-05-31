#! /usr/bin/env python3

import requests
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import re
import folium
from folium.features import DivIcon
from matplotlib import cm
from matplotlib.colors import to_hex
from datetime import datetime

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--csvin', help='raw data csv', required=True)
args = parser.parse_args()

datetime_fmt = '%Y-%m-%dT%H%M%SZ'

if 1:
  filename = args.csvin
  basename = filename[ :filename.rfind('.')]
  _, start, stop = basename.split('_')
  tstart = datetime.strptime(start, datetime_fmt)
  tstop = datetime.strptime(stop, datetime_fmt)
  print(tstart, tstop)
  df = pd.read_csv(args.csvin, sep=';')
  # 'occurredAt', 'networkId', 'type', 'description', 'clientId',
  # 'clientDescription', 'deviceSerial', 'deviceName', 'ssidNumber',
  # 'ssidName', 'eventData'
  print(df.columns)
  #print(df)
  print(df)
  df = df.drop_duplicates()
  df.occurredAt = pd.to_datetime(df.occurredAt)
  df = df.set_index('occurredAt')
  print(df)
  print(df.groupby('networkId').count()['deviceSerial'])
  print(df.groupby('type').count()['deviceSerial'])
  print(df.groupby('deviceSerial').count()['networkId'])
  print(df.groupby('description').count()['networkId'])
  print(df.groupby('clientId').count()['networkId'])
  print(df.groupby(['clientDescription']).count()['networkId'])
  print(df.groupby(['clientId', 'clientDescription']).count()['networkId'])

  # for cid, dfg in df.groupby(['clientId', 'clientDescription']):
  #   print(cid)
  #   print(dfg[['deviceSerial', 'description', 'clientDescription']])
  sampling_dt_min = 15
  freq = f'{sampling_dt_min}T'
  tidx = pd.date_range(start, stop, freq=freq)

  timecnt = pd.DataFrame(index=tidx)
  for netid, dfi in df.groupby(['networkId']):
    for ser, dfg in dfi.groupby(['deviceSerial']):
      print(ser)
      dfc = dfg.copy()
      dfc['uniq_key'] = [ f'{x}_{y}' for x,y in dfc[['clientId', 'clientDescription']].values ]
      cnt = dfc[['clientId', 'clientDescription', 'uniq_key']].resample(f'{sampling_dt_min}T').nunique()
      cnt = cnt.reindex(tidx).fillna(0)
      #print(cnt)
      timecnt[ser] = cnt.uniq_key
    print(timecnt)
    timecnt.plot(style='-o', figsize=(12,8))
    plt.savefig(f'{basename}_{netid}.png')
    plt.close()

if 1:
  filetag = 'station_metadata'
  df = pd.read_csv(f'{filetag}.csv', sep=';')
  print(df)
  print(df.columns)
  print(df[[
    'name',
    'serial',
    'networkId',
    'lat',
    'lng',
    'status',
    #'address',
    #'mac',
    #'lanIp',
    #'url',
    #'model',
    #'switchProfileId',
    #'firmware',
    #'floorPlanId',
    #'tags'
  ]])
  df = df[ df.lng > 0 ]
  gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lng, df.lat))
  gdf = gdf.set_crs(epsg=4326)
  ax = gdf.plot(column='networkId')
  plt.savefig(f'{filetag}.png')
  if 0:
    ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
    plt.savefig(f'{filetag}_map.png')
  plt.close()

  centerlat = gdf.geometry.centroid.y.mean()
  centerlon = gdf.geometry.centroid.x.mean()

  netcolors = { net : i for i, (net, dfg) in enumerate(gdf.groupby('networkId')) }
  colorm = cm.get_cmap('viridis', len(netcolors))
  gdf['color'] = [ to_hex(colorm.colors[netcolors[net]]) for net in gdf.networkId.values ]
  print(gdf)

  m = folium.Map(location=[centerlat, centerlon], control_scale=True, zoom_start=15)
  for lat, lon, ser, col in gdf[['lat', 'lng', 'serial', 'color']].values:
    folium.CircleMarker(
      location=[lat, lon],
      popup=folium.Popup(f'serial {ser}', show=False, sticky=True),
      color=col,
      fill_color=col
    ).add_to(m)
  m.save(f'{filetag}.html')

  print(timecnt)
  tots = timecnt.sum().transpose().reset_index().rename(columns={'index':'serial', 0:'tot'})
  print(tots)
  gdf = gdf.merge(tots, left_on='serial', right_on='serial')
  gdf['scaled_tot'] = 50 * gdf.tot / gdf.tot.max()
  print(gdf)

  m = folium.Map(location=[centerlat, centerlon], control_scale=True, zoom_start=15)
  for lat, lon, ser, col, tot in gdf[['lat', 'lng', 'serial', 'color', 'scaled_tot']].values:
    folium.CircleMarker(
      location=[lat, lon],
      popup=folium.Popup(f'serial {ser}', show=False, sticky=True),
      color='red',
      fill_color='red',
      radius=tot
    ).add_to(m)
  m.save(f'{filetag}_circleradius.html')
