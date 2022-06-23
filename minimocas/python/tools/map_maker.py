#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import geopy
from geopy.geocoders import Nominatim
import argparse
import pandas as pd

import folium
from folium.features import DivIcon

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--csv', type=str, default='')
parser.add_argument('-t', '--tag', type=str, default='VENEZIA')
args = parser.parse_args()

base = args.csv[:args.csv.rfind('.')]
geofile = os.path.join(os.environ['WORKSPACE'], 'scr-data', 'vars', 'extra', 'venice_barriers.csv')

###
locator = Nominatim(user_agent='myGeocoder')
center_tag = '{}, Italia'.format(args.tag)
center = locator.geocode(center_tag)
center_coords = [center.latitude, center.longitude]
print('Location "{}" geocoded at {}'.format(center_tag, center_coords))

map = folium.Map(
  location=center_coords,
  tiles='cartodbpositron',
  zoom_start=12,
)

geodf = pd.read_csv(geofile, sep=';', engine='python')
geodf['color'] = 'blue'
print(geodf)

geodf = geodf[ geodf.Description.str.endswith('_IN') ]
geodf.apply(lambda row : folium.CircleMarker(
  location=[row['Lat'], row['Lon']],
  radius=5,
  popup='some text',
  color=row['color'],
  line_color=row['color'],
  fill_color=row['color']
).add_child(folium.Popup(row['Description'])).add_to(map), axis=1)

geodf.apply(lambda row : folium.Marker(
  location=[row['Lat'], row['Lon']],
  icon=folium.DivIcon(
    html='<div style="color: black; font-size: 20;">{}</div>'.format(row['Description'])
  )
).add_to(map), axis=1)


map_file = base + '_map.html'
map.save(map_file)
print('Map saved to : {}'.format(map_file))
