#! /usr/bin/env python3

import json
import sys
import os
import numpy as np
from datetime import datetime, timedelta
from dateutil import tz

##########################
#### log function ########
##########################
def logs(s):
  head = '{} [conf] '.format(datetime.now().strftime('%y%m%d %H:%M:%S'))
  return head + s

def log_print(s, logger = None):
  if logger:
    logger.info(logs(s))
  else:
    print(logs(s), flush=True)

#####################
### config class ####
#####################
class conf:

  def __init__(self, config, logger = None):
    self.logger = logger
    self.date_format = '%Y-%m-%d %H:%M:%S'
    self.creation_dt = 30
    self.rates_dt = 5 * 60

    self.HERE = tz.tzlocal()
    self.UTC = tz.gettz('UTC')

    try:
      sys.path.append(os.path.join(os.environ['WORKSPACE'], 'slides', 'python'))
      from db_kml import db_kml
      from model_slides import model_slides
    except Exception as e:
      raise Exception('[conf] library load failed : {}'.format(e))

    self.config = config
    self.cparams = { k : os.path.join(os.environ['WORKSPACE'], 'slides', 'vars', 'templates', '{}_template.json'.format(k)) for k in config['cities'].keys() }
    #print(self.cparams)

    try:
      self.wdir = config['work_dir']
      if not os.path.exists(self.wdir): os.mkdir(self.wdir)

      self.ms = model_slides(config['model_data'], self.logger)

      self.dbk = db_kml(config['kml_data'], self.logger)

    except Exception as e:
      raise Exception('conf init failed : {}'.format(e)) from e

  def generate(self, start_date, stop_date, citytag):
    #print(citytag)
    if citytag not in self.cparams:
      raise Exception('[db_kml] generate citytag {} not found'.format(citytag))

    start = datetime.strptime(start_date, self.date_format)
    stop = datetime.strptime(stop_date, self.date_format)
    mid_start = datetime.strptime(start.strftime('%Y-%m-%d 00:00:00'), self.date_format)

    with open(self.cparams[citytag]) as tin:
      conf = json.load(tin)

    conf['start_date'] = start_date
    conf['stop_date'] = stop_date

    conf['state_basename'] = self.wdir + '/{}'.format(citytag)

    # attractions
    attr = self.dbk.get_attractions(citytag)
    if len(attr) > 6:
      log_print('*********** Temporary lowering of attractions number', self.logger)
      attr = { k : v for k, v in list(attr.items())[:6] }
    conf['attractions'] = attr

    # blank timetable
    rates_per_day = 24 * 60 * 60 // self.rates_dt
    ttrates = { t : 0 for t in [ mid_start + i*timedelta(seconds=self.rates_dt) for i in range(rates_per_day) ] }
    sources = {}

    # tourist sources
    src_list = self.dbk.get_sources(citytag)
    for tag, src in src_list.items():
      #print(tag, src)
      data = self.ms.full_table(start, stop, citytag, tag)
      #print('data\n', data)

      if 'weight' in src: # if ferrara
        # calcolo tot sniffer
        # tot m0 (start, stop) - tot sniffer (start, stop)
        # spalmo su src non sniffer
        data = data * src['weight']
      #print('rescaled ', data)

      tt = ttrates.copy()
      rates = { t.replace(
          year=mid_start.year,
          month=mid_start.month,
          day=mid_start.day
        ) : v
        for t, v in zip(data.index, data[tag].values)
      }
      tt.update(rates)

      beta_bp = 0.8
      speed_mps = 0.7

      sources.update({
        tag + '_IN' : {
          'creation_dt' : self.creation_dt,
          'creation_rate' : [ v for v in tt.values() ],
          'source_location' : {
            'lat' : src['lat'],
            'lon' : src['lon']
          },
          'pawns_from_weight': {
            'tourist' : {
              'beta_bp_miss' : beta_bp,
              'speed_mps'    : speed_mps
            }
          }
        }
      })

    """
    # control
    df = self.ms.rescaled_data(start, stop, max = 1000)
    #print(df)
    rates = { t.replace(
        year=mid_start.year,
        month=mid_start.month,
        day=mid_start.day
      ) : v
      for t,v in zip(df.index, df['data'].values)
    }
    tt = ttrates.copy()
    tt.update(rates)

    locals = {
      'is_control'    : True,
      'creation_dt'   : self.creation_dt ,
      'creation_rate' : [ int(v) for v in tt.values() ],
      'pawns' : {
        'locals' : {
          'beta_bp_miss'   : 0.5,
          'start_node_lid' : -1,
          'dest'           : -1
        }
      }
    }
    sources['LOCALS'] = locals
    """

    conf['sources'] = sources

    return conf

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--cfg', help='prepare config file', required=True)
  args = parser.parse_args()

  with open(args.cfg) as cfgfile:
    config = json.loads(cfgfile.read())

  try:
    cfg = conf(config)

    start_date = config['start_date']
    stop_date = config['stop_date']
    try:
      tag = config['city_tag']
    except:
      tag = 'ferrara'

    sim = cfg.generate(start_date, stop_date, tag)
    #sim['explore_node'] = [0]

    with open('conf_{}.json'.format(tag), 'w') as simout:
      json.dump(sim, simout, indent=2)
  except Exception as e:
    print('main EXC : {}'.format(e))
