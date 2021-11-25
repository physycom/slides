#! /usr/bin/env python3

import os
import json
import folium
import pandas as pd
from matplotlib import cm
from matplotlib.colors import to_hex

if __name__ == '__main__':

  station_json = os.path.join(os.environ['WORKSPACE'], 'slides', 'vars', 'extra', 'bari_cameras.json')
  stationsmeta = pd.DataFrame.from_dict(json.load(open(station_json)))

  map_file = 'bari_cameras.html'
  cmap = cm.get_cmap('viridis', len(stationsmeta))
  stationsmeta['color'] = [ to_hex(c) for c in cmap.colors ]
  stationsmeta.index = stationsmeta.id
  stationsmeta = stationsmeta[['id', 'name', 'lat', 'lon', 'color']]
  stationsmeta = stationsmeta.drop(columns='id')
  stationsmeta = stationsmeta[stationsmeta.lon > 0]
  map_center = stationsmeta[['lat', 'lon']].mean()

  m = folium.Map(location=map_center, control_scale=True, tiles = 'Stamen Terrain')
  layerlabel = '<span style="color: {col};">{txt}</span>'
  for sid, data in stationsmeta.iterrows():
    layer_sel = folium.FeatureGroup(name=layerlabel.format(col=f'{data.color}', txt=f'Camera {sid}'))
    pt = folium.CircleMarker(
      location=[data.lat, data.lon],
      radius=5,
      color=f'{data.color}',
      fill=True,
      fill_color=f'{data.color}',
      fill_opacity=1,
      popup=folium.Popup(f'<p>Camera <b>{sid}</b></br> Name <b>{data[0]}</b></br></p>', show=False, sticky=True, max_width=300),
    )
    layer_sel.add_child(pt)
    m.add_child(layer_sel)
    
  folium.map.LayerControl(collapsed=False).add_to(m)
  s, w = stationsmeta[['lat', 'lon']].min()
  n, e = stationsmeta[['lat', 'lon']].max()
  m.fit_bounds([ [s,w], [n,e] ])
  m.save(f'bari_camera_map.html')