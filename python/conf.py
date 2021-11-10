#! /usr/bin/env python3

import json
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil import tz

##########################
#### log function ########
##########################
import logging
logger = logging.getLogger('conf')

#####################
### config class ####
#####################
class conf:

  def __init__(self, config):
    self.date_format = '%Y-%m-%d %H:%M:%S'
    self.creation_dt = 30
    self.rates_dt = 15 * 60

    self.HERE = tz.tzlocal()
    self.UTC = tz.gettz('UTC')

    self.remove_local_output = config['remove_local_output']

    try:
      sys.path.append(os.path.join(os.environ['WORKSPACE'], 'slides', 'python'))
      from db_kml import db_kml
      from model_slides import model_slides
    except Exception as e:
      raise Exception('[conf] library load failed : {}'.format(e))

    self.cparams = { k : os.path.join(os.environ['WORKSPACE'], 'slides', 'vars', 'templates', '{}_template.json'.format(k)) for k in config['cities'].keys() }
    #print(self.cparams)

    try:
      self.wdir = config['work_dir']
      if not os.path.exists(self.wdir): os.mkdir(self.wdir)

      config['model_data']['work_dir'] = f'{self.wdir}/m_data'
      self.ms = model_slides(config['model_data'])

      config['kml_data']['work_dir'] = f'{self.wdir}/kml_data'
      self.dbk = db_kml(config['kml_data'])

    except Exception as e:
      raise Exception('conf init failed : {}'.format(e)) from e

    self.config = config


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
    attr = self.dbk.get_attractions(citytag, start)
    max_attr = 25
    if len(attr) > max_attr:
      logger.warning(f'*********** Lowering attractions number to {max_attr}')
      attr = { k : v for k, v in list(attr.items())[:max_attr] }
    conf['attractions'] = attr

    # blank timetable
    rates_per_day = 24 * 60 * 60 // self.rates_dt
    ttrates = { t : 0 for t in [ mid_start + i*timedelta(seconds=self.rates_dt) for i in range(rates_per_day) ] }
    sources = {}

    # sources timetable df generation
    src_list = self.dbk.get_sources(citytag)
    srcdata = pd.DataFrame()
    for tag, src in src_list.items():
      #print(tag, src)
      data = self.ms.full_table(start, stop, citytag, tag)
      # print('data\n', data)

      if len(srcdata) == 0:
        srcdata = data
      else:
        srcdata = srcdata.join(data)
    #print(srcdata)
    #print(srcdata.sum())

    # city-specific caveat
    if citytag == 'ferrara':
      snif_src = { src : None for src in src_list }
      params = self.ms.mod_fe.station_map
      snif_src.update({ src : snif for snif in params for src in params[snif] })
      src_num = len(snif_src)
      norm_src = [ n for n, v in snif_src.items() if v == None ]
      m0_num = len(norm_src)
      logger.debug(f'Caveat FE - src {src_num}, m0_src {m0_num}')
      norm_wei = np.asarray([ src_list[n]['weight'] for n in norm_src ])
      norm_wei /= ( norm_wei.sum() * src_num / m0_num )
      for s, c in zip(norm_src, norm_wei):
        #print(f'{s}, {srcdata[s].sum()}, {c}')
        srcdata[s] *= c
      synth_src = [ s for s in src_list if s not in norm_src ]
      for s in synth_src:
        #print(f"{s} {src_list[s]['weight']}")
        srcdata[s] *= src_list[s]['weight'] / src_num * ( src_num - m0_num )
    elif citytag == 'dubrovnik':
      snif_src = { src : None for src in src_list }
      params = self.ms.mod_du.source_map
      snif_src.update({ snif : 'data' for snif in params.values() })
      src_num = len(snif_src)
      norm_src = [ n for n, v in snif_src.items() if v == None ]
      m0_num = len(norm_src)
      logger.debug(f'Caveat DU - src {src_num}, m0_src {m0_num}')
      norm_wei = np.asarray([ src_list[n]['weight'] for n in norm_src ])
      norm_wei /= ( norm_wei.sum() * src_num / m0_num )
      for s, c in zip(norm_src, norm_wei):
        srcdata[s] *= c
    else:
      for s in src_list:
        srcdata[s] *= src_list[s]['weight']

    # log totals for debug
    for c, v in srcdata.sum().items():
      logger.info(f'Source {c} total pawn : {v:.2f}')
    logger.info(f'Simulation total pawn ; {srcdata.sum().sum():.2f}')

    # cast dataframe to timetable json format
    for tag in srcdata.columns:
      data = srcdata[[tag]].copy()


      # wrap around midnight
      tt = ttrates.copy()
      rates = { t.replace(
          year=mid_start.year,
          month=mid_start.month,
          day=mid_start.day
        ) : v
        for t, v in zip(data.index, data[tag].values)
      }
      tt.update(rates)

      beta_bp = 0.5
      speed_mps = 1.0

      sources.update({
        tag + '_IN' : {
          'creation_dt' : self.creation_dt,
          'creation_rate' : [ float(v) for v in tt.values() ],
          'source_location' : {
            'lat' : src_list[tag]['lat'],
            'lon' : src_list[tag]['lon']
          },
          'pawns_from_weight': {
            'tourist' : {
              'beta_bp_miss' : beta_bp,
              'speed_mps'    : speed_mps
            }
          }
        }
      })

    # control
    df = srcdata.copy()
    df = self.ms.locals(start, stop, citytag)
    #print(df)
    rates = { t.replace(
        year=mid_start.year,
        month=mid_start.month,
        day=mid_start.day
      ) : v
      for t,v in zip(df.index, df.values)
    }
    tt = ttrates.copy()
    tt.update(rates)

    locals = {
      'source_type'   : 'ctrl',
      'creation_dt'   : self.creation_dt,
      'creation_rate' : [ int(v) for v in tt.values() ],
      'pawns' : {
        'locals' : {
          'beta_bp_miss'   : 0,
          'start_node_lid' : -1,
          'dest'           : -1
        }
      }
    }
    #print(locals)
    sources['LOCALS'] = locals

    conf['sources'] = sources

    return conf

if __name__ == '__main__':
  import argparse
  import coloredlogs

  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--cfg', help='prepare config file', required=True)
  args = parser.parse_args()

  console_formatter = coloredlogs.ColoredFormatter('%(asctime)s [%(levelname)s] (%(name)s:%(funcName)s) %(message)s', "%H:%M:%S")
  console_handler = logging.StreamHandler()
  console_handler.setFormatter(console_formatter)
  logging.basicConfig(
    level=logging.DEBUG,
    handlers=[console_handler]
  )
  logging.getLogger('matplotlib').setLevel(logging.WARNING)

  with open(args.cfg, encoding="utf8") as cfgfile:
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
