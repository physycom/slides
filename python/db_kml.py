#! /usr/bin/env python3

import json
import base64
import sys
import os
import zipfile
import io
import re
from datetime import datetime, timedelta
import pandas as pd
from requests import get
from pykml import parser as xmlparser

import random as r
r.seed(19)

##########################
#### log function ########
##########################
import logging
logger = logging.getLogger('db_kml')

##############
### KML db ###
##############
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

  def parse_kml(self, kmlfile, citytag):
    logger.info('Parse {} info from local kml {}'.format(citytag, kmlfile))

    with open(kmlfile, encoding="utf8") as f:
      folder = xmlparser.parse(f).getroot().Document.Folder

    ### parse bari kml
    locations = {}
    if citytag == 'bari':
      for pm in folder.Placemark:
        if pm.ExtendedData != None:
          name = pm.name.text.strip()
          if re.match('.*limite.*', pm.name.text) != None: continue

          # parse metadata
          for data in pm.ExtendedData.getchildren():
            if data.attrib['name'] == 'descrizione':
              description = data.value.text
            elif data.attrib['name'] == 'weight':
              weight = float(data.value.text)
            elif data.attrib['name'] == 'visit_time [h]':
              visit_t = int(float(data.value.text) * 3600)
            elif data.attrib['name'] == 'opening_timetables':
              open_tt = data.value.text

          role = 'none'
          point = [p for p in pm.getchildren() if p.tag.endswith('Point')]
          if point:
            if re.match('.*poi A.*', description) != None: role = 'attraction'
            elif re.match('.*accesso.*', description) != None: role = 'source'
            else: continue
            lon, lat, z = point[0].coordinates.text.split(',')
            locations[pm.name.text] = {
              'type'    : 'Point',
              'role'    : role,
              'lat'     : float(lat),
              'lon'     : float(lon),
              'weight'  : weight,
              'visit_t' : visit_t,
              'open_tt' : None,
              'actv_dt' : None,
              'close_d' : None,
              'cap'     : 1000,
            }
    elif citytag == 'dubrovnik':
      for pm in folder.Placemark:
        name = pm.name.text.strip()
        description = pm.description.text.strip()
        role = 'none'
        # parse metadata
        for data in pm.ExtendedData.getchildren():
          if data.attrib['name'] == 'descrizione':
            description = data.value.text
          elif data.attrib['name'] == 'weight':
            weight = float(data.value.text)
          elif data.attrib['name'] == 'visit_time [h]':
            visit_t = int(float(data.value.text) * 3600)

        point = [ p for p in pm.getchildren() if p.tag.endswith('Point') ]
        if point:
          if re.match('.*[cC]amera [0-9].*', name) != None: role = 'source'
          if re.match('.*[dD]egree A.*', description) != None: role = 'attraction'

          #print(name, '---', description)
          lon, lat, z = point[0].coordinates.text.split(',')
          locations[name] = {
            'type'    : 'Point',
            'role'    : role,
            'lat'     : float(lat),
            'lon'     : float(lon),
            'weight'  : weight,
            'visit_t' : visit_t,
            'open_tt' : None,
            'actv_dt' : None,
            'close_d' : None,
            'cap'     : 1000,
          }
    elif citytag == 'ferrara':
      for pm in folder.Placemark:
        if pm.ExtendedData != None:
          name = pm.name.text.strip()
          # parse metadata
          for data in pm.ExtendedData.getchildren():
            if data.attrib['name'] == 'description':
              description = data.value.text
            elif data.attrib['name'] == 'weight':
              weight = float(data.value.text)
            elif data.attrib['name'] == 'visit_time [h]':
              visit_t = int(float(data.value.text) * 3600)
            elif data.attrib['name'] == 'opening_timetables':
              open_tt = data.value.text
            elif data.attrib['name'] == 'active_dates':
              actv_dt = data.value.text
            elif data.attrib['name'] == 'closing_days':
              close_d = data.value.text
            elif data.attrib['name'] == 'capacity':
              cap = int(float(data.value.text))

          point = [ p for p in pm.getchildren() if p.tag.endswith('Point') ]

          # create location object
          if point and description:
            lon, lat, z = point[0].coordinates.text.split(',')
            role = 'none'
            if re.match('.*ingresso.*uscita.*', description) != None: role = 'source'
            if re.match('.*destinazione.*', description) != None: role = 'attraction'
            #print(name, '---', description)
            locations[name] = {
              'type'    : 'Point',
              'role'    : role,
              'lat'     : float(lat),
              'lon'     : float(lon),
              'weight'  : weight,
              'visit_t' : visit_t,
              'open_tt' : open_tt,
              'actv_dt' : actv_dt,
              'close_d' : close_d,
              'cap'     : cap,
            }
    elif citytag == 'sybenik':
      for pm in folder.Placemark:
        if pm.ExtendedData != None:
          name = pm.name.text.strip()

          # parse metadata
          for data in pm.ExtendedData.getchildren():
            if data.attrib['name'] == 'descrizione':
              description = data.value.text
            elif data.attrib['name'] == 'weight':
              weight = float(data.value.text)
            elif data.attrib['name'] == 'visit_time [h]':
              visit_t = int(float(data.value.text) * 3600)

          point = [ p for p in pm.getchildren() if p.tag.endswith('Point') ]
          if point:
            role = 'none'
            if re.match('.*source.*', description) != None: role = 'source'
            # if re.match('.*A.*', name) != None: role = 'attraction'
            if re.match('.*A.*', description) != None: role = 'attraction'


            lon, lat, z = point[0].coordinates.text.split(',')
            locations[name.replace('- A', '').strip()] = {
              'type'    : 'Point',
              'role'    : role,
              'lat'     : float(lat),
              'lon'     : float(lon),
              'weight'  : weight,
              'visit_t' : visit_t,
              'open_tt' : None,
              'actv_dt' : None,
              'close_d' : None,
              'cap'     : 1000,
            }

      #####################################################
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
      #####################################################
    elif citytag == 'venezia':
      for pm in folder.Placemark:
        if pm.ExtendedData != None:
          name = pm.name.text.strip()

          # parse metadata
          for data in pm.ExtendedData.getchildren():
            if data.attrib['name'] == 'description':
              description = data.value.text
            elif data.attrib['name'] == 'weight':
              weight = float(data.value.text)
            elif data.attrib['name'] == 'visit_time [h]':
              visit_t = int(float(data.value.text) * 3600)

          point = [ p for p in pm.getchildren() if p.tag.endswith('Point') ]

          # create location object
          if point and description:
            lon, lat, z = point[0].coordinates.text.split(',')
            role = 'none'
            if re.match('.*poi.*', description) != None: role = 'attraction'
            if re.match('.*accesso.*', description) != None: role = 'source'
            #print(name, '---', description)
            locations[name] = {
              'type'    : 'Point',
              'role'    : role,
              'lat'     : float(lat),
              'lon'     : float(lon),
              'weight'  : weight,
              'visit_t' : visit_t,
              'open_tt' : None,
              'actv_dt' : None,
              'close_d' : None,
              'cap'     : 1000,
            }
    logger.info(f'Parsed {len(locations)} locations for {citytag}')

    attr = {
      k.replace(' ', '_') : {
        'lat' : v['lat'],
        'lon' : v['lon'],
        'weight' : v['weight'],
        'capacity' : v['cap'],
        'visit_time' : v['visit_t'],
        'opening_timetable' : v['open_tt'],
        'closing_days' : v['close_d'],
        'active_dates' : v['actv_dt'],
      }
      for k,v in locations.items() if v['role'] == 'attraction'
    }
    src = {
      k.replace(' ', '_') : {
        'lat'    : v['lat'],
        'lon'    : v['lon'],
        'weight' : v['weight'],
      }
      for k,v in locations.items() if v['role'] == 'source'
    }
    logger.info(f'Parsed {len(attr)} attractions {len(src)} sources')

    self.cities[citytag]['attractions'] = attr
    self.cities[citytag]['sources'] = src
    self.cities[citytag]['valid'] = True

  def convert_attr(self, citytag, start):
    attr = self.cities[citytag]['attractions']
    df = pd.DataFrame.from_dict(attr).T
    df['name'] = df.index
    #print(df)

    df = df.fillna(value={
      'opening_timetable' : '00:00-23:59',
      'closing_days' : '',
      'active_dates' : 'all',
    })
    #print(df)

    today = start.strftime('%Y-%m-%d')
    #print(today)
    weekday = start.strftime('%a').lower()
    #print(weekday)
    attr_sampling_dt = 3600
    #print(attr_sampling_dt)

    attr = {}
    for i, row in df.iterrows():
      name = row['name']
      #print(i, name)
      lat = row['lat']
      lon = row['lon']
      weight = row['weight']
      vtime = row['visit_time']
      ott = [ range.split('-') for range in row['opening_timetable'].split('|') ]
      cds = row['closing_days'].split('|')
      open_close_dt = row['active_dates'].split('|')
      close_period = []
      open_period = []
      close_dt = []
      open_dt = []
      for i in open_close_dt:
        if ":" in i:
          if i.startswith('!'):
            start_clos, end_clos = i[1:].split(':')
            close_period.append([start_clos, end_clos])
          else:
            start_open, end_open = i.split(':')
            open_period.append([start_open, end_open])
        else:
          if i.startswith('!'):
            close_dt.append(i)
          else:
            open_dt.append(i)

      #print('open_dt', open_dt)
      #print('close_dt', close_dt)
      #print('open_period', open_period)
      #print('close_period', close_period)
      #print(ott)
      #print("cds: ",  cds)

      if len(open_dt) == 0:  #improve the elegance of all this logic
        open_dt.append('all')

      # skip closing days
      if 'all' in cds or weekday in cds:
        logger.warning(f'Attraction {name} is closed for all in weekday')
        continue

      # skip if today is in close date or period
      if today in close_dt:
        logger.warning(f'Attraction {name} is in close date')
        continue

      close_founded = False
      for cp in close_period:
        start = cp[0]
        stop = cp[1]
        if today >= start and today <= stop and not today in open_dt: #close in a period but open in a specific date
          close_founded = True

      if close_founded == True:
        logger.warning(f'Attraction {name} is in close period {start}:{stop} ')
        continue

      # skip non-active date if needed
      if not 'all' in open_dt:
        if not today in open_dt:
          logger.warning(f'Attraction {name} is not in active day all')
          continue

      capacity = row['capacity']
      #print(f'weekday {weekday} in {cds} = {closed}')

      # evaluate low level timetable
      self.time_format = '%H:%M'
      tstart = datetime.strptime('00:00', self.time_format)
      dt = timedelta(seconds=attr_sampling_dt)
      tt_len = int(24 * 60 * 60 / attr_sampling_dt )
      #print(f'{tstart} @ {dt} size {tt_len}')
      tt = {
        tstart + i * dt : 0
        for i in range(tt_len)
      }
      #for t, v in tt.items(): print(t,v)
      for ti, tf in ott:
        d_i = datetime.strptime(ti, self.time_format)
        d_f = datetime.strptime(tf, self.time_format)
        tt.update({ t : capacity for t in tt if d_i <= t < d_f })
        #for t, v in tt.items(): print(t,v)
      #print(tt)
      timecap = list(tt.values())

      #transform weight from double to vector
      weight_vector=[]
      for i in timecap:
        if i==0:
          weight_vector.append(0)
        else:
          weight_vector.append(weight)

      # write low level attraction json format
      attr.update({
        name : {
          'lat'        : lat,
          'lon'        : lon,
          'weight'     : weight_vector,
          'timecap'    : timecap,
          'visit_time' : vtime
        }
      })

    logger.info(f'Created {len(attr)} attractions')
    for i,k in enumerate(attr.keys()):
      logger.debug(f'Attraction #{i} tag {k}')

    #print(json.dumps(attr, indent=2))
    return attr

  def retrieve_data(self, citytag):
    city = self.cities[citytag]
    if 'mid' not in city: raise Exception('mid not avalaible for {}'.format(citytag))

    kmlfile = self.wdir + '/attractions_{}.kml'.format(citytag)
    if not os.path.exists(kmlfile) or True:
      try:
        logger.debug('Retrieving kml data for {}'.format(citytag))
        url = 'https://mapsengine.google.com/map/kml?mid={}'.format(city['mid'])
        response = get(url)
        if response.status_code != 200:
          raise Exception('[db_kml] map download failed with code {}'.format(response.status_code))
        #print(data.content)
        zf = zipfile.ZipFile(io.BytesIO(response.content), 'r')
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

  def get_attractions(self, citytag, start):
    if not self.cities[citytag]['valid']:
      self.retrieve_data(citytag)
    attr = self.convert_attr(citytag, start)
    return attr

  def get_sources(self, citytag):
    if not self.cities[citytag]['valid']:
      self.retrieve_data(citytag)
    return self.cities[citytag]['sources']

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--cfg', help='prepare config file', required=True)
  args = parser.parse_args()

  with open(args.cfg) as cfgfile:
    config = json.loads(cfgfile.read())

  city = config['city']

  dbk = db_kml(config)

  attr = dbk.get_attractions(city)
  with open('attr_{}.json'.format(city), 'w') as simout: json.dump(attr, simout, indent=2)

  src = dbk.get_sources(city)
  with open('src_{}.json'.format(city), 'w') as simout: json.dump(src, simout, indent=2)
#  try:
#  except Exception as e:
#    print('main EXC : {}'.format(e))
