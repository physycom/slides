#! /usr/bin/env python3

from http.server import BaseHTTPRequestHandler
import os
import json
import re
from matplotlib import cm
from datetime import datetime
import pandas as pd
import collections
import numpy as np

datafile = ''
grid = {}
data = collections.defaultdict(dict)

class Server(BaseHTTPRequestHandler):

  def do_HEAD(self):
    return

  def do_POST(self):
    return

  def do_GET(self):
    self.respond()

  def serve_html(self, filename):
    f = open(os.path.dirname(os.path.realpath(__file__)) + '/web/' + filename)
    status = 200
    content_type = 'text/html; charset=ISO-8859-1'
    response_content = f.read()
    response_content = bytes(response_content, 'UTF-8')
    size = len(response_content)
    return status, content_type, response_content, size

  def serve_404(self):
    status = 404
    content_type = 'text/plain'
    response_content = '404 Url not found.'
    response_content = bytes(response_content, 'UTF-8')
    size = len(response_content)
    return status, content_type, response_content, size

  def serve_json(self, geojson):
    status = 200
    content_type = 'application/json; charset=ISO-8859-1'
    response_content = json.dumps(geojson)
    response_content = bytes(response_content, 'UTF-8')
    size = len(response_content)
    return status, content_type, response_content, size

  def handle_http(self):
    if self.path.endswith('.html'):
      try:
        htmlfile = self.path.split('/')[-1]
        status, content_type, response_content, size = self.serve_html(htmlfile)
      except:
        status, content_type, response_content, size = self.serve_404()
    elif self.path == '/':
      status, content_type, response_content, size = self.serve_html('index.html')
    elif self.path.startswith('/heat'):
      status, content_type, response_content, size = self.serve_html('heatmap.html')
    elif self.path.startswith('/json'):
      geojson = {
        'type': 'FeatureCollection',
        'features': [],
      }

      # sanity checks and various init
      geojson['times'] = list(map(lambda x: datetime.fromtimestamp(x).strftime("%Y%m%d_%H%M%S"), data[0].keys()))

      for k, v in data.items():
        feat = {
          'type': 'Feature',
          'properties': {
            'time_cnt' : []
          },
          'geometry': {
            'type': 'Point',
            'coordinates': []
          }
        }
        feat['properties']['time_cnt'] = list(map(int, v.values()))
        feat['geometry']['coordinates'] = list(grid[k])
        geojson['features'].append(feat)

#      print(geojson)

      status, content_type, response_content, size = self.serve_json(geojson)
    else:
      status, content_type, response_content, size = self.serve_404()

    self.send_response(status)
    self.send_header('Content-type', content_type)
    self.send_header('Content-length', size)
    self.end_headers()
    return response_content

  def respond(self):
    content = self.handle_http()
    self.wfile.write(content)


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('-a', '--server-address', help='http server address', default='localhost')
  parser.add_argument('-p', '--server-port', help='http server port', default=9999, type=int)
  parser.add_argument('-g', '--grid', help='grid geojson file', required=True)
  parser.add_argument('-d', '--data', help='grid data influx-ready file', required=True)
  args = parser.parse_args()
  gridfile = args.grid
  datafile = args.data

  with open(gridfile) as f:
    geogrid = json.load(f)
  for i in geogrid['features']:
    grid[i['properties']['id']] = np.mean(i['geometry']['coordinates'][0], 0)

  df = pd.read_csv(datafile, sep=' ', index_col=None, header=None)
  df['gid'] = df[0].str.split('=', expand=True)[1]
  df['cnt'] = df[1].str.split('=', expand=True)[1]
  df['ts'] = df[2] * 1e-9
  df = df[['gid','ts','cnt']].astype({'ts':'int', 'gid':'int'})
  for row in df.values:
    data[row[0]][row[1]] = row[2]

  import time
  from http.server import HTTPServer

  HOST_NAME = args.server_address
  PORT_NUMBER = args.server_port

  httpd = HTTPServer((HOST_NAME, PORT_NUMBER), Server)
  print(time.asctime(), 'Server UP - %s:%s' % (HOST_NAME, PORT_NUMBER))
  try:
    httpd.serve_forever()
  except KeyboardInterrupt:
    httpd.server_close()
    print(time.asctime(), 'Server DOWN - %s:%s' % (HOST_NAME, PORT_NUMBER))