import os
import sys
import json
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import numpy as np
import coloredlogs
import time

class detector:

  def __init__(self, configfile):
    with open(configfile) as cin:
      config = json.load(cin)

    self.wdir = config['workdir']
    self.datadir = f'{self.wdir}/detection_data'
    if not os.path.exists(self.datadir): os.mkdir(self.datadir)
    logfile = f'{self.wdir}/detection-engine.log'
    self.clock_dt = config['clock_dt']
    self.dump_dt = config['dump_dt']
    self.pending_detection = True

    console_formatter = coloredlogs.ColoredFormatter('%(asctime)s (%(name)s) [%(levelname)s] %(message)s', '%H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)

    file_formatter = logging.Formatter('%(asctime)s (%(name)s) [%(levelname)s] %(message)s', '%y-%m-%d %H:%M:%S')
    file_handler = TimedRotatingFileHandler(logfile, when='D', backupCount = 7)
    file_handler.setFormatter(file_formatter)

    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[
            file_handler,
            console_handler,
        ]
    )
    self.logger = logging.getLogger('detection')

    self.config = config

    camdatafile = os.path.join(os.environ['WORKSPACE'], 'slides', 'pvt', 'venezia', 'cam_data.json')
    with open(camdatafile) as cin:
      self.camcfg = json.load(cin)
    self.camdata = [c for c in self.camcfg if c['cam_cnt']]
    # Initialize model

    # Dump helper
    self.ns = len(self.camdata)
    self.dumper = {'cams':[[] for x in range(self.ns)], 'bars':[[] for x in range(self.ns)]}
    self.logger.info('*********** init engine done')
    self.barriers = [{} for x in range(self.ns)]

  def do_task(self):
    try:
      tnow = datetime.now()
      ts = int(tnow.timestamp())
      if int(ts) % self.clock_dt == 0:

        for j in range(self.ns):
          print(ts, len(self.dumper['cams'][j]))
          cname = self.camdata[j]['name']

          # create cam record
          cam_dump = {
            'cam_name'  : cname,
            'timestamp' : ts,
            'datetime'  : datetime.fromtimestamp(ts).strftime('%y%m%d %H%M%S'),
            'counter'   : {
              'MEAN' : 7,
              'MAX'  : 10,
              'MIN'  : 1,
            }
          }
          self.dumper['cams'][j].append(cam_dump)

          bar_dump = {
            'cam_name'  : cname,
            'timestamp' : ts,
            'datetime'  : datetime.fromtimestamp(ts).strftime('%y%m%d %H%M%S'),
            'counter'   : {
              'main': {
                'IN': 6,
                'OUT': 48
              }
            }
          }
          self.dumper['bars'][j].append(bar_dump)

          time.sleep(0.2) # fake sleeper

      if int(ts) % self.dump_dt == 0:
        for j in range(self.ns):
          cname = self.camdata[j]['name']

          self.logger.info(f'dumping for : {cname}')
          cdatafile = f'{self.datadir}/cam_{cname}_{ts}.json'
          with open(cdatafile, 'w') as cout:
            json.dump(self.dumper['cams'][j], cout, indent=2)
          self.dumper['cams'][j] = [] #reset

          if len(self.dumper['bars'][j])!=0:
            # write raw data
            bar_aggregate = {
              'cam_name'  : cname,
              'timestamp' : ts,
              'datetime'  : datetime.fromtimestamp(ts).strftime('%y%m%d %H%M%S'),
              'counter'   : {
                'main': {
                  'IN': 6,
                  'OUT': 48
                }
              }
            }
            bdatafile = f'{self.datadir}/bar_{cname}_{ts}.json'
            with open(bdatafile, 'w') as cout:
              json.dump([bar_aggregate], cout, indent=2)

            self.dumper['bars'][j] = [] #reset
    except Exception as e:
      print('runtime error: ',e)



def make_cam_map(filename):
  import pandas as pd
  import folium

  with open(filename) as jin:
    camdata = json.load(jin)

  df = pd.DataFrame.from_dict(camdata)
  df['lat'] = df.coords.apply(lambda x: x[0])
  df['lon'] = df.coords.apply(lambda x: x[1])
  df['fake_name'] = df.name
  df.loc[ df.name.str.startswith('Centro'), 'fake_name' ] = 'Centro 1 & Centro 2'
  df = df.drop(columns=['barriers', 'ip'])
  center = df[['lat', 'lon']].mean()
  print(df)
  m = folium.Map(location=center, control_scale=True, zoom_start=9)
  df.apply(lambda row: folium.CircleMarker(
    location=[row.lat, row.lon],
    radius=8,
    fill_color='purple',
    color='purple',
    fill_opacity=1.,
    popup=folium.Popup(f'camera {row["fake_name"]}', show=True, sticky=True),
  ).add_to(m), axis=1)
  s, w = df[['lat', 'lon']].min()
  n, e = df[['lat', 'lon']].max()
  m.fit_bounds([ [s,w], [n,e] ])
  m.save('map_camera.html')


if __name__ == '__main__':

  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--cfile', help='engine config json', required=True)
  args = parser.parse_args()

  if 1:
    det = detector(configfile=args.cfile)
    while True:
      det.do_task()

  if 0:
    make_cam_map(os.path.join(os.environ['WORKSPACE'], 'slides', 'tools', 'sybenik', 'cam_data.json'))
