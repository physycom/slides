#! /usr/bin/env python3

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil import tz

try:
  sys.path.append(os.path.join(os.environ['WORKSPACE'], 'slides', 'python'))
  from model_slides_ferrara import model_slides_ferrara
except Exception as e:
  raise Exception('[model_slides] library load failed : {}'.format(e))

##########################
#### log function ########
##########################
def logs(s):
  head = '{} [model_slides] '.format(datetime.now().strftime('%y%m%d %H:%M:%S'))
  return head + s

def log_print(s, logger = None):
  if logger:
    logger.info(logs(s))
  else:
    print(logs(s), flush=True)

#############################
### model_slides class  #####
#############################
class model_slides:

  def __init__(self, config, logger = None):
    self.logger = logger
    self.got_data = False
    self.date_format = '%Y-%m-%d %H:%M:%S'
    self.time_format = '%H:%M:%S'
    self.rates_dt = 5 * 60
    self.config = config

    self.wdir = config['work_dir']
    if not os.path.exists(self.wdir): os.mkdir(self.wdir)

    self.models = {}
    if 'model1_file_dir' in config:
      model1_file_dir = config['model1_file_dir']
      print(model1_file_dir)
      for dir in model1_file_dir:
        for root, subdirs, files in os.walk(dir):
          for f in files:
            try:
              city, tag, mod = f.split('.')[0].split('-')
              if mod != 'model1':
                log_print('Model type {} not supported, skipping'.format(mod))
                continue
              print(city, tag, mod)
              fname = os.path.join(root, *subdirs, f)
              print(fname)
              self.import_model1(city, tag, fname)
            except Exception as e:
              log_print('Problem parsing model1 file {} : {}'.format(f, e))
              continue

    params = {}
    for k, v in config['params'].items():
      params[k] = {
        'population' : v['population'] if 'population' in v else 1000,
        'daily_t'    : v['daily_tourist'] if 'daily_tourist' in v else 100
      }
    self.params = params

  def get_data(self, start, stop, city, tag):
    if not (city, tag) in self.models:
      self.create_model0(city, tag)

    mod = self.models[(city, tag)]
    data = self.full_table(start, stop, city, tag, mod)
    return data

  def full_table(self, start, stop, city, tag, model):
    if city == '_ferrara':
      msfconf = self.config['params']['ferrara']['sniffer_db']
      msfconf['work_dir'] = self.wdir + '/m_ferrara'
      if not os.path.exists(msfconf['work_dir']): os.mkdir(msfconf['work_dir'])
      msf = model_slides_ferrara(msfconf)
      data = msf.full_table(start, stop, tag)
    else:
      if 'm1' in model:
        m01 = 'm1'
      else:
        m01 = 'm0'
      log_print('Generating data for ({}, {}) from {}'.format(city, tag, model[m01]))
      mfile = self.models[(city,tag)][m01]
      data = pd.read_csv(mfile, sep=';')
    print(data)
    return data

  def import_model1(self, city, tag, file):
    log_print('importing model1 {}-{}'.format(city, tag), self.logger)
    m1filename = self.wdir + '/{city}-{tag}-model1.csv'.format(city=city, tag=tag)

    df = pd.read_csv(file, sep=';')
    df.to_csv(m1filename, sep=';', header=True, index=False)

    self.models[(city, tag)] = {
      'm1' : m1filename
    }

  def create_model0(self, city, tag):
    log_print('creating model0 {}-{}'.format(city, tag), self.logger)
    m0filename = self.wdir + '/{city}-{tag}-model0.csv'.format(city=city, tag=tag)

    if not os.path.exists(m0filename):
      generic_monday = '2020-05-04 12:00:00' # just because it's monday
      weekdays = [ t.strftime('%a').lower() for t in [ datetime.strptime(generic_monday, self.date_format) + timedelta(days=i) for i in range(7) ] ]

      midn = datetime.strptime('00:00:00', self.time_format)
      rates_per_day = 24 * 60 * 60 // self.rates_dt
      ttrates = { t : -1 for t in [ (midn + i*timedelta(seconds=self.rates_dt)).time() for i in range(rates_per_day) ] }

      ### fake data generation #############################
      tt = ttrates.copy()
      t1 = datetime.strptime('01:00:00', self.time_format).time()
      t2 = datetime.strptime('04:00:00', self.time_format).time()
      t3 = datetime.strptime('09:00:00', self.time_format).time()
      t4 = datetime.strptime('12:00:00', self.time_format).time()
      t5 = datetime.strptime('16:00:00', self.time_format).time()
      t6 = datetime.strptime('20:00:00', self.time_format).time()
      t7 = datetime.strptime('23:00:00', self.time_format).time()
      for t in tt:
        if t < t1 or t >= t7:
          tt[t] = 100
        elif t < t2:
          tt[t] = 100
        elif t < t3:
          tt[t] = 250
        elif t < t4:
          tt[t] = 500
        elif t < t5:
          tt[t] = 200
        elif t < t6:
          tt[t] = 400
        elif t < t7:
          tt[t] = 200
      self.tt_raw = tt

      runave_size = 5 * 60 * 60 // self.rates_dt # running average idx interval from time in seconds
      kern = np.concatenate([ np.ones((runave_size,))/runave_size, np.zeros((len(tt) - runave_size, )) ])
      conv = np.real(np.fft.ifft( np.fft.fft(list(tt.values())) * np.fft.fft(kern) ))
      tt = { t : v for t, v in zip( tt.keys(), conv ) }
      #############################################################

      df = pd.DataFrame([], index=tt.keys())
      for i, day in enumerate(weekdays):
        scale = ( len(weekdays) - i ) / len(weekdays)
        df[day] = [ scale * v for v in tt.values() ]
      df = df.astype('int')
      df.index = pd.to_datetime(df.index, format=self.time_format)
      df.index.name = 'time'
      df.to_csv(m0filename, sep=';', header=True, index=True)

    self.models[(city, tag)] = {
      'm0' : m0filename
    }

  def bla():
    #df_data = pd.DataFrame(result, columns=['datetime','counter','barrier'])
    df_data = pd.read_csv(m0_data_file, sep=';')
    df_data.time = pd.to_datetime(df_data.time, format=self.date_format)
    df_data.time = [ t.replace(
        year=start.year,
        month=start.month,
        day=start.day
      )
      for t in df_data.time
    ]
    self.df_data_day = df_data.set_index('time')

    mask = (df_data.time >= start) & (df_data.time < stop)
    df_data = df_data[mask].set_index('time')

    self.got_data = True
    self.df_data = df_data
    return df_data

  def rescale_data(self, start, stop, df, tot = None, max = None, min = None):
    df.time = pd.to_datetime(df.time, format=self.date_format)
    df.time = [ t.replace(
        year=start.year,
        month=start.month,
        day=start.day
      )
      for t in df.time
    ]
    df = df.set_index('time')

    wday = start.strftime('%a').lower() # fix for near midn simulation
    df = df[[wday]].copy()

    if tot != None:
      dftot = df[wday].sum()
      df[wday] = (df[wday] / dftot * tot).astype('float')

    if max != None and min != None:
      dfmax = df[wday].max()
      dfmin = df[wday].min()
      df[wday] = (df[wday] / (dfmax - dfmin) * (max - min) + min).astype('float')

    if max != None:
      dfmax = df[wday].max()
      df[wday] = (df[wday] / dfmax * max).astype('float')

    df.columns = ['data']
    mask = (df.index >= start) & (df.index < stop)
    return df[mask]

if __name__ == '__main__':
  import argparse
  import matplotlib.pyplot as plt

  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--cfg', help='prepare config file', required=True)
  args = parser.parse_args()

  with open(args.cfg) as cfgfile:
    config = json.load(cfgfile)

  date_format = '%Y-%m-%d %H:%M:%S'
  start = datetime.strptime(config['date_start'], date_format)
  stop = datetime.strptime(config['date_stop'], date_format)
  city = config['city']

  try:
    m0 = model_slides(config)

    """
    # save data
    data = m0.get_data(start, stop)
    data.plot()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid()
    plt.savefig(m0.data_dir + '/m0_data_{}_{}.png'.format(
      start.strftime('%y%m%d-%H%M'),
      stop.strftime('%y%m%d-%H%M')
    ))
    plt.clf()

    # save day data
    m0.df_data.plot()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid()
    plt.savefig(m0.data_dir + '/m0_day_{}_{}.png'.format(
      start.strftime('%y%m%d-%H%M'),
      stop.strftime('%y%m%d-%H%M')
    ))
    plt.clf()
    """

    # save rescaled tot
    data_tot = m0.rescaled_data(start, stop, tot = 1000)
    data_tot.plot()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid()
    plt.savefig(m0.data_dir + '/m0_tot_{}_{}.png'.format(
      start.strftime('%y%m%d-%H%M'),
      stop.strftime('%y%m%d-%H%M')
    ))
    plt.clf()

    # save rescaled max
    data_max = m0.rescaled_data(start, stop, max = 1000)
    data_max.plot()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid()
    plt.savefig(m0.data_dir + '/m0_max_{}_{}.png'.format(
      start.strftime('%y%m%d-%H%M'),
      stop.strftime('%y%m%d-%H%M')
    ))
    plt.clf()

    # save rescaled max
    data_mm = m0.rescaled_data(start, stop, max = 2000, min = 1000)
    data_mm.plot()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid()
    plt.savefig(m0.data_dir + '/m0_mm_{}_{}.png'.format(
      start.strftime('%y%m%d-%H%M'),
      stop.strftime('%y%m%d-%H%M')
    ))
    plt.clf()

  except Exception as e:
    log_print('EXC : {}'.format(e))
