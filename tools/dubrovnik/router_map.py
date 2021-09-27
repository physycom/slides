#! /usr/bin/env python3

import os
import json
import folium
import argparse
import pandas as pd
import mysql.connector
from matplotlib import cm
from matplotlib.colors import to_hex

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--cfg', type=str, required=True)

  args = parser.parse_args()

  with open(args.cfg, encoding='utf-8') as f:
    config = json.load(f)

  conf = config['model_data']['params']['dubrovnik']['mysql']
  db = mysql.connector.connect(
      host     = conf['host'],
      port     = conf['port'],
      user     = conf['user'],
      passwd   = conf['pwd'],
      database = conf['db']
      )
  cursor = db.cursor()
  try:
    query = f"""
    SELECT
      ds.id AS id,
      ds.name AS name,
      ds.serial AS serial,
      ds.lat AS lat,
      ds.lng AS lon,
      ds.networkId,
      ds.status as status
    FROM	
      Devices ds 
    """ 
    # print(query)
    cursor.execute(query)
    result = cursor.fetchall()
    print(f'Received {len(result)} mysql data in {query}')
    stationsmeta = pd.DataFrame(result)
    stationsmeta.columns =  cursor.column_names
  except Exception as e:
    print('Connection error : {}'.format(e))

  if 0 == 1: # To be activate if and when the longitude will be fixed from Meraki
    data = list(stationsmeta.T.to_dict().values())
    with open('dubrovnik_router.json', 'w') as out:
      json.dump(data, out, indent=2, ensure_ascii=False)
  else:
    station_json = os.path.join(os.environ['WORKSPACE'], 'slides', 'vars', 'extra', 'dubrovnik_router.json')
    stationsmeta = pd.DataFrame.from_dict(json.load(open(station_json)))

  map_file = 'dubrovnik_router.html'
  cmap = cm.get_cmap('viridis', len(stationsmeta))
  stationsmeta['color'] = [ to_hex(c) for c in cmap.colors ]
  stationsmeta.index = stationsmeta.id
  stationsmeta = stationsmeta.drop(columns='id')
  stationsmeta = stationsmeta[stationsmeta.lon > 0]
  print(len(stationsmeta))
  map_center = stationsmeta[['lat', 'lon']].mean()

  m = folium.Map(location=map_center, control_scale=True, tiles = 'Stamen Terrain')
  layerlabel = '<span style="color: {col};">{txt}</span>'
  for sid, data in stationsmeta.iterrows():
    layer_sel = folium.FeatureGroup(name=layerlabel.format(col=f'{data.color}', txt=f'Router {sid}'))
    pt = folium.CircleMarker(
      location=[data.lat, data.lon],
      radius=5,
      color=f'{data.color}',
      fill=True,
      fill_color=f'{data.color}',
      fill_opacity=1,
      popup=folium.Popup(f'<p>Router <b>{sid}</b></br> Name <b>{data[0]}</b></br> Serial <b>{data.serial}</b></br></p>', show=False, sticky=True, max_width=300),
    )
    layer_sel.add_child(pt)
    m.add_child(layer_sel)
    
  folium.map.LayerControl(collapsed=False).add_to(m)
  s, w = stationsmeta[['lat', 'lon']].min()
  n, e = stationsmeta[['lat', 'lon']].max()
  m.fit_bounds([ [s,w], [n,e] ])
  m.save(f'dubrovnik_router_map.html')