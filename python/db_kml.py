#! /usr/bin/env python3

import json
import base64
import sys
import os
import zipfile
import io
import re
#import numpy as np
from datetime import datetime, timedelta
#from dateutil import tz
import pandas as pd
from requests import get
from pykml import parser as xmlparser

##########################
#### log function ########
##########################
def logs(s):
  head = '{} [db_kml] '.format(datetime.now().strftime('%y%m%d %H:%M:%S'))
  return head + s

def log_print(s, logger = None):
  if logger:
    logger.info(logs(s))
  else:
    print(logs(s), flush=True)

##############
### KML db ###
##############
class db_kml:

  def __init__(self, config, logger = None):
    self.logger = logger
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

  def parse_kml(self, kmlfile, citytag):
    log_print('Parse {} info from local kml {}'.format(citytag, kmlfile), self.logger)

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

          #print(name)
          lon, lat, z = point[0].coordinates.text.split(',')
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

    log_print('Parsed {} locations for {}'.format(len(locations), citytag), self.logger)

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
        log_print('Retrieving kml data for {}'.format(citytag), self.logger)
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
