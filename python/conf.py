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
      #sys.path.append(os.path.join(os.environ['WORKSPACE'], 'minimocas', 'python'))
      #from model0 import model0
      sys.path.append(os.path.join(os.environ['WORKSPACE'], 'slides', 'python'))
      from model import model0, model1

      sys.path.append(os.path.join(os.environ['WORKSPACE'], 'slides', 'python'))
      from db_kml import db_kml
    except Exception as e:
      raise Exception('[conf] library load failed : {}'.format(e))

    self.config = config
    self.cparams = { k : os.path.join(os.environ['WORKSPACE'], 'slides', 'vars', 'templates', '{}_template.json'.format(k)) for k in config['cities'].keys() }
    #print(self.cparams)

    try:
      self.wdir = config['work_dir']
      if not os.path.exists(self.wdir): os.mkdir(self.wdir)

      self.m0 = model0(config['model0'])

      #per alle, mancano tutti i cosi di sicurezz varia, questo è fatto un po' a cavolo :D
      self.enable_m1 = False
      if (config['cities'][config['city_tag']] != {}):
        self.enable_m1 = True
        self.m1 = model1(config['cities'][config['city_tag']]['model1'])

      self.dbk = db_kml(config['kml_data'], self.logger)

    except Exception as e:
      raise Exception('conf init failed : {}'.format(e)) from e

  def generate(self, start_date, stop_date, citytag):
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

    rates_per_day = 24 * 60 * 60 // self.rates_dt
    ttrates = { t : 0 for t in [ mid_start + i*timedelta(seconds=self.rates_dt) for i in range(rates_per_day) ] }

    # attractions
    attr = self.dbk.generate(citytag)
    if len(attr) > 6:
      log_print('*********** Temporary lowering of attractions number', self.logger)
      attr = { k : v for k, v in list(attr.items())[:6] }
    conf['attractions'] = attr

    sources = {}
    # sources
    # < insert code >

    # control
    df = self.m0.rescaled_data(start, stop, max = 1000)

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

    if(self.enable_m1):
      df = self.m1.rescaled_data(start,stop)

      rates = { t.replace(
          year=mid_start.year,
          month=mid_start.month,
          day=mid_start.day
        ) : v
        for t,v in zip(df.index, df['data'].values)
      }
      tt = { t : 0 for t in [ mid_start + i*timedelta(seconds=3600) for i in range(24) ] }
      tt.update(rates)

      locals = {
        'is_control'    : True,
        'creation_dt'   : self.creation_dt,
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
    sim['explore_node'] = [0]

    with open('conf_{}.json'.format(tag), 'w') as simout:
      json.dump(sim, simout, indent=2)
  except Exception as e:
    print('main EXC : {}'.format(e))
