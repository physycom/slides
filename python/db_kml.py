#! /usr/bin/env python3

import json
import base64
import sys
import os
import zipfile
import io
import re
import xml.etree.ElementTree as ET
#import numpy as np
#from datetime import datetime, timedelta
#from dateutil import tz
import pandas as pd
from requests import get
from pykml import parser as xmlparser

#################################
# github PRIVATE file retriever #
#################################
def get_github_file(url):
  usr = 'vivesimulator'
  pwd = '8e71274e4d93a822d8fd981da01c0f816a833868'
  response = get(url, auth=(usr, pwd))
  content = base64.b64decode(response.json()['content'])
  return content

##########################
#### log function ########
##########################
def log_print(*args, **kwargs):
  print('[db_kml] ', end='', flush=True)
  print(*args, **kwargs, flush=True)

##########################
### walker config class ##
##########################
class db_kml:

  def __init__(self, config):
    self.wdir = config['work_dir']
    if not os.path.exists(self.wdir): os.mkdir(self.wdir)

    self.city_default_list = [
      'bari',
      'dubrovnik',
      'ferrara',
      'sybenik',
      'venezia'
    ]
    cities = {
      city : {
        'valid' : False
      } for city in self.city_default_list
    }
    for c, mid in config['mymaps_id'].items():
      cities[c].update({ 'mid' : mid })

    self.cities = cities



  """
def parse_attractions_kml(kmlin):
  tree = ET.parse(kmlin)
  root = tree.getroot()

  origin = {}
  poi = {}
  roi = {}
  counter = 1
  plc_cnt = 0
  for c in root.iter():
    if c.tag.endswith('Placemark'):
      print('[parse_kml] Found placemark #{}'.format(plc_cnt), end='')
      plc_cnt += 1
      for p in c.getchildren():
        if p.tag.endswith('name'):
          name = p.text
        elif p.tag.endswith('description'):
          info = p.text
        elif p.tag.endswith('Point'):
          type = 'Point'
          for q in p.getchildren():
            if q.tag.endswith('coordinates'):
              point = list(map(float, q.text.strip().split(',')[:-1]))
        elif p.tag.endswith('Polygon'):
          type = 'Polygon'
          for q in p.getchildren():
            for r in q.getchildren():
              for s in r.getchildren():
                if s.tag.endswith('coordinates'):
                  coords = [ float(t) for t in re.split('[ ]+|\n|,',s.text) if t ]
                  bbox = {
                    'lat_max' : np.max(coords[1::3]),
                    'lat_min' : np.min(coords[1::3]),
                    'lon_max' : np.max(coords[::3]),
                    'lon_min' : np.min(coords[::3])
                  }
      print(' type "{}" name "{}" info "{}"'.format(type, name, info))
      if 'ingresso' in info or 'uscita' in info or 'Entry' in info:
        origin[name] = {
          'Point' : point,
        }
      elif 'destinazione' in info or 'interesse' in info or 'Interest' in info or 'interest' in info:
        poi[name] = {
          'Point'  : point,
          'Weight' : 1 if 'A' in info else 0.5
        }
      elif 'roi' in info:
        roi[name] = bbox
      else:
        print('[parse_kml] WARNING Placemark "{}" not handled.'.format(name))
      counter += 1

  print('[parse_kml] Found {} attractions, {} origins, {} rois'.format(len(poi), len(origin), len(roi)))
  attractions = {}
  for i,(k,v) in enumerate(poi.items()):
    attractions.setdefault('attractions', {}).update({
      k : {
        'lat'        : v['Point'][1],
        'lon'        : v['Point'][0],
        'weight'     : v['Weight'],
        'timecap'    : [ 1000 ],
        'visit_time' : 300
      }
    })

  return attractions
  """

  def parse_kml(self, kmlfile, citytag):
    log_print('Parse {} info from local kml {}'.format(citytag, kmlfile))

    with open(kmlfile) as f:
      folder = xmlparser.parse(f).getroot().Document.Folder

    ### parse bari kml
    if citytag == 'bari':
      locations = {}
      for pm in folder.Placemark:

        if re.match('.*limite.*', pm.name.text) != None: continue

        point = [p for p in pm.getchildren() if p.tag.endswith('Point')]
        if point:
          lon, lat, z = point[0].coordinates.text.split(',')
          locations[pm.name.text] = {
            'type' : 'Point',
            'lat' : float(lat),
            'lon' : float(lon)
          }
    elif citytag == 'dubrovnik':
      locations = {}
      for pm in folder.Placemark:
        name = pm.name.text.strip()
        description = pm.description.text.strip()

        if re.match('.*[cC]amera [0-9].*', name) != None: continue
        if re.match('.*[dD]egree A.*', description) == None: continue

        point = [ p for p in pm.getchildren() if p.tag.endswith('Point') ]
        if point:
          #print(name, '---', description)
          lon, lat, z = point[0].coordinates.text.split(',')
          locations[name] = {
            'type' : 'Point',
            'lat' : float(lat),
            'lon' : float(lon)
          }
    elif citytag == 'ferrara':
      locations = {}
      for pm in folder.Placemark:
        name = pm.name.text.strip()
        point = [ p for p in pm.getchildren() if p.tag.endswith('Point') ]
        description = [ p for p in pm.getchildren() if p.tag.endswith('description') ]
        if point and description:
          description = pm.description.text.strip()
          if re.match('.*ingresso.*uscita.*', description) != None: continue
          if re.match('.*destinazione.*', description) == None: continue
          #print(name, '---', description)
          lon, lat, z = point[0].coordinates.text.split(',')
          locations[name] = {
            'type' : 'Point',
            'lat' : float(lat),
            'lon' : float(lon)
          }
    elif citytag == 'sybenik':
      # retrieve roi bbox
      try:
        content = get_github_file('https://api.github.com/repos/physycom/cartography-data/contents/data/{}/roi.json'.format('sebenico'))
        roi = json.loads(content)
        lat_min = roi['bbox']['lat_min']
        lat_max = roi['bbox']['lat_max']
        lon_min = roi['bbox']['lon_min']
        lon_max = roi['bbox']['lon_max']
      except Exception as e:
        raise Exception('[db_tides] get_stationsid roi retrival failed : {}'.format(e))

      # parse kml locations
      locations = {}
      for pm in folder.Placemark:
        name = pm.name.text.strip()
        point = [ p for p in pm.getchildren() if p.tag.endswith('Point') ]
        #description = [ p for p in pm.getchildren() if p.tag.endswith('description') ]
        if point:# and description:
          #description = pm.description.text.strip()
          #if re.match('.*ingresso.*uscita.*', description) != None: continue
          if re.match('.*Camera.*', name) != None: continue
          if re.match('.*Camera.*', name) != None: continue
          if re.match('.* - A.*', name) == None: continue

          if re.match('.*Port.*', name) != None: continue
          if re.match('.*City [eE]ntrance.*', name) != None: continue
          if re.match('.*Parking.*', name) != None: continue

          lon, lat, z = point[0].coordinates.text.split(',')
          lat = float(lat)
          lon = float(lon)
          if lat < lat_min and lat > lat_max and lon < lon_min and lon > lon_max: continue

          #print(name)
          locations[name.replace('- A', '').strip()] = {
            'type' : 'Point',
            'lat' : float(lat),
            'lon' : float(lon)
          }

      explore_kmeans = False
      if explore_kmeans:
        data = [ [ name, pro['lat'], pro['lon'] ] for name, pro in locations.items() ]
        df = pd.DataFrame(data, columns=['name', 'lat', 'lon'])
        print(df)

        import matplotlib.pyplot as plt
        from sklearn.cluster import KMeans

        K_clusters = range(1,10)
        kmeans = [ KMeans(n_clusters=i) for i in K_clusters]
        Y_axis = df[['lat']]
        X_axis = df[['lon']]
        score = [kmeans[i].fit(Y_axis).score(Y_axis) for i in range(len(kmeans))]
        plt.plot(K_clusters, score)
        #plt.xlabel('Number of Clusters')plt.ylabel('Score')plt.title('Elbow Curve')
        plt.show()

        kmeans = KMeans(n_clusters = 7, init ='k-means++')
        X = df[['lon', 'lat']]
        kmeans.fit(X) # Compute k-means
        df['cluster_label'] = kmeans.fit_predict(X)
        centers = kmeans.cluster_centers_
        print(df)
        print(df.groupby('cluster_label').count())

        df.plot.scatter(x = 'lon', y = 'lat', c=df.cluster_label, s=50, cmap='viridis')
        plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
        plt.show()

    elif citytag == 'venezia':
      raise Exception('[db_kml] kml parsing for {} coming soon'.format(citytag))

    log_print('Parsed {} locations for {}'.format(len(locations), citytag))
    #for k,v in locations.items():
    #  print(k, v)

    attr = {}
    for kl, kv in locations.items():
      attr[kl] = {
        'lat' : kv['lat'],
        'lon' : kv['lon'],
        'weight' : 0.5,
        'timecap' : [ 1000 ],
        'visit_time' : 300
      }

    self.cities[citytag]['attractions'] = attr
    self.cities[citytag]['ok'] = True

  def get_data(self, citytag):
    city = self.cities[citytag]
    if 'mid' not in city: raise Exception('mid not avalaible for {}'.format(citytag))

    kmlfile = self.wdir + '/attractions_{}.kml'.format(citytag)
    if not os.path.exists(kmlfile):
      try:
        log_print('Retrieving kml data for {}'.format(citytag))
        url = 'https://mapsengine.google.com/map/kml?mid={}'.format(city['mid'])
        data = get(url)
        #print(data.content)
        zf = zipfile.ZipFile(io.BytesIO(data.content), 'r')
        #print(zf.namelist())
        kmzfiles = [ f for f in zf.namelist() if re.match('.*doc.kml.*', f) ]
        if len(kmzfiles) != 1: print('warning mulitple content')
        if len(kmzfiles) == 0: raise Exception('[db_kml] no kml in kmz for {}'.format(citytag))
        kmzfiles = zf.namelist()[0]
        with open(kmlfile, 'wb') as f:
          f.write(zf.read(kmzfiles))
      except Exception as e:
        raise Exception('[db_kml] kml download failed for {} : {}'.format(citytag, e))

    self.parse_kml(kmlfile, citytag)
    #os.remove(kmlfile)

  def generate(self, citytag):
    if not self.cities[citytag]['valid']:
      self.get_data(citytag)
    return self.cities[citytag]['attractions']

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--cfg', help='prepare config file', required=True)
  args = parser.parse_args()

  with open(args.cfg) as cfgfile:
    config = json.loads(cfgfile.read())

  city = config['city']

  dbk = db_kml(config)

  attr = dbk.generate(city)

  with open('attr_{}.json'.format(city), 'w') as simout:
    json.dump(attr, simout, indent=2)
#  try:
#  except Exception as e:
#    print('main EXC : {}'.format(e))
