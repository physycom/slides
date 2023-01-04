#! /usr/bin/env python3

import xml.etree.ElementTree as ET
import json
import re
import numpy as np

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

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--kmlin", help="map data KML file", required=True)
  args = parser.parse_args()

  attractions = parse_attractions_kml(args.kmlin)

  #print(json.dumps(attractions, indent=2))
  with open('attractions.json', 'w') as aout:
    json.dump(attractions, aout, indent=2)
