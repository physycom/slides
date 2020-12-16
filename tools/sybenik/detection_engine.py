#! /usr/bin/env python3

import os
import json
import logging
from datetime import datetime
import numpy as np

def fake_cam_cnt(ts, maxval=100):
  t = ts % int(24 * 60 * 60)
  v = maxval * np.sin( t / (48 * 60 * 60) * 2 * np.pi )
  return int(v)

class detection:

  def __init__(self, configfile):
    with open(configfile) as cin:
      config = json.load(cin)
    self.wdir = config['workdir']
    self.datadir = f'{self.wdir}/detection_data'
    if not os.path.exists(self.datadir): os.mkdir(self.datadir)
    logfile = f'{self.wdir}/detection-engine.log'
    self.clock_dt = 60
    self.fine_dt = 15
    #self.fine_dt = self.clock_dt / 6
    self.rec_per_pack = int(self.clock_dt / self.fine_dt)

    self.pending_detection = True

    logging.basicConfig(
      filename=logfile,
      filemode='w',
      level=logging.DEBUG,
      format='%(asctime)s [%(levelname)s] %(message)s',
      datefmt='%y-%m-%d %H:%M:%S%z'
    )
    self.config = config

    camdatafile = os.path.join(os.environ['SYBENIK_WORKSPACE'], 'tools', 'sybenik', 'data', 'cam_data.json')
    with open(camdatafile) as cin:
      self.camdata = json.load(cin)

    logging.info('init engine')

  def do_task(self):
    try:
      tnow = datetime.now()
      ts = int(tnow.timestamp())
      if int(ts) % self.clock_dt == 0:
        if self.pending_detection:
          logging.info('performing detections')

          ############## super duper fake
          cid = 1
          for cd in self.camdata:
            cname = cd['name']
            logging.info(f'detection for : {cname}')
            cdata = {
              'cam_name'  : cname,
              'timestamp' : ts,
              'datetime'  : datetime.fromtimestamp(ts).strftime('%y%m%d %H%M%S'),
              'counter'   : fake_cam_cnt(ts, 1000*cid+10)
            }
            cdatafile = f'{self.datadir}/cam_{cname}_{ts}.json'
            with open(cdatafile, 'w') as cout:
              json.dump(cdata, cout, indent=2)

            bdata = []
            for btag in cd['barriers']:
              for n in range(self.rec_per_pack):
                tf = int(ts - (self.rec_per_pack - n)*self.fine_dt)
                bdata.append({
                  'cam_name'  : cname,
                  'timestamp' : tf,
                  'datetime'  : datetime.fromtimestamp(tf).strftime('%y%m%d %H%M%S'),
                  'counter'   : {
                    btag : { 'IN' : fake_cam_cnt(tf, 100*cid+10), 'OUT' : fake_cam_cnt(tf, 100*cid+60)}
                  }
                })
              bdatafile = f'{self.datadir}/bar_{cname}_{ts}.json'
              with open(bdatafile, 'w') as cout:
                json.dump(bdata, cout, indent=2)
            cid += 1
          ###############

          self.pending_detection = False
      else:
        if not self.pending_detection:
          self.pending_detection = True
    except Exception as e:
      print(e)

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--cfile', help='engine config json', required=True)
  args = parser.parse_args()

  det = detection(configfile=args.cfile)
  while True:
    det.do_task()
