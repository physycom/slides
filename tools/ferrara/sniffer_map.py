#! /usr/bin/env python3

import os
import pymongo
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import to_hex
import json
import mysql.connector
from datetime import datetime
import folium

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--conf', help='database config file', default=None)
  args = parser.parse_args()

  # query for station metadata and dump in json format
  if 0:
    station_json = 'ferrara_stations.json'

    print(f'Querying for stations metadata file')
    if args.conf is None:
      print(f'No config provided, relaunch with -c config.json')
      exit()

    try:
      config = json.load(open(args.conf))
      config = config['mysql']
      db = mysql.connector.connect(
        host     = config['host'],
        port     = config['port'],
        user     = config['user'],
        passwd   = config['pwd'],
        database = config['db']
      )
      cursor = db.cursor()

      query = """
        SELECT
          s.id,
          s.station_id,
          s.station_oid,
          s.latitude,
          s.longitude,
          s.address,
          s.station_name
        FROM
          Stations s
        WHERE
          s.latitude != 0
      """
      tquery = datetime.now()
      cursor.execute(query)
      result = cursor.fetchall()
      tquery = datetime.now() - tquery
      print(f'Received {len(result)} mysql data in {tquery}')
    except Exception as e:
      print(f'Error in querying stations metadata {e}')
      exit()

    stationsmeta = pd.DataFrame(result)
    stationsmeta.columns =  cursor.column_names
    stationsmeta = stationsmeta.to_dict(orient='records')

    print(stationsmeta[0])
    with open(station_json, 'w') as fout:
      json.dump(stationsmeta, fout, indent=2)

  # make map
  if 1:
    map_file = 'ferrara_stations.html'
    station_json = os.path.join(os.environ['WORKSPACE'], 'slides', 'vars', 'extra', 'ferrara_sniffer.json')
    stationsmeta = pd.DataFrame.from_dict(json.load(open(station_json))).T
    cmap = cm.get_cmap('viridis', len(stationsmeta))
    stationsmeta['color'] = [ to_hex(c) for c in cmap.colors ]
    print(stationsmeta)
    map_center = stationsmeta[['latitude', 'longitude']].mean()

    m = folium.Map(location=map_center, control_scale=True)
    layerlabel = '<span style="color: {col};">{txt}</span>'
    for sid, data in stationsmeta.iterrows():
      layer_sel = folium.FeatureGroup(name=layerlabel.format(col=f'{data.color}', txt=f'Station {sid}'))
      pt = folium.CircleMarker(
        location=[data.latitude, data.longitude],
        radius=5,
        color=f'{data.color}',
        fill=True,
        fill_color=f'{data.color}',
        fill_opacity=1,
        popup=folium.Popup(f'<p>Station <b>{sid}</b></br>Name <b>{data.station_addr}</b></p>', show=True, sticky=True, max_width=300),
      )
      layer_sel.add_child(pt)
      m.add_child(layer_sel)
    folium.map.LayerControl(collapsed=False).add_to(m)
    s, w = stationsmeta[['latitude', 'longitude']].min()
    n, e = stationsmeta[['latitude', 'longitude']].max()
    m.fit_bounds([ [s,w], [n,e] ])
    m.save(f'ferrara_stations.html')
