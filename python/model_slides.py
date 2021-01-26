#! /usr/bin/env python3

import re
import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil import tz
from glob import glob

try:
  sys.path.append(os.path.join(os.environ['WORKSPACE'], 'slides', 'python'))
  from model_ferrara import model_ferrara
  from model_dubrovnik import model_dubrovnik
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

################
### utils  #####
################
def box_centered_kernel(tot_len, box_len):
  pad_len = tot_len - box_len
  kern = np.concatenate([
    np.zeros((pad_len // 2)),
    np.ones((box_len)) / box_len,
    np.zeros((pad_len - pad_len // 2))# for odd box_len
  ])
  return kern

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

    # init city-specific model
    self.mod_fe = model_ferrara(config['params']['ferrara'], self.logger)
    self.mod_du = model_dubrovnik(config['params']['dubrovnik'], self.logger)
    self.models = {}

    # collect model1 filenames
    if 'model1_file_dir' in config:
      model1_file_dir = config['model1_file_dir']
      #print(model1_file_dir)
      for dir in model1_file_dir:
        for root, subdirs, files in os.walk(dir):
          for f in files:
            try:
              city, tag, mod = f.split('.')[0].split('-')
              if mod != 'model1':
                log_print('Model type {} not supported, skipping'.format(mod))
                continue
              #print(city, tag, mod)
              fname = os.path.join(root, *subdirs, f)
              #print(fname)
              self.import_model1(city, tag, fname)
            except Exception as e:
              log_print('Problem parsing model1 file {} : {}'.format(f, e))
              continue

    # collect model0 filenames
    m0files = glob(f'{self.wdir}/*model0.csv')
    for m0f in m0files:
      tok = re.split('[/-]', m0f)
      city = tok[-3]
      tag = tok[-2]
      self.models[(city, tag)] = {
        'm0' : m0f
      }

    params = {}
    for k, v in config['params'].items():
      params[k] = {
        'population' : v['population'] if 'population' in v else 1000,
        'daily_t'    : v['daily_tourist'] if 'daily_tourist' in v else 100
      }
    self.params = params

  def full_table(self, start, stop, city, tag):
    #print((city, tag), self.models.keys())
    #print(self.models)
    #log_print(f'Creating data for {city} {tag}', self.logger)
    if not (city, tag) in self.models:
      self.create_model0(city, tag)

    model = self.models[(city, tag)]
    if city == 'ferrara':
      m01 = 'FE'
      try:
        data = self.mod_fe.full_table(start, stop, tag, resampling=self.rates_dt)
      except Exception as e:
        log_print(f'Model {m01} errors for {tag}, falling back to m0 : {e}', self.logger)
        data = pd.DataFrame()
    elif city == 'dubrovnik':
      m01 = 'DU'
      try:
        data = self.mod_du.full_table(start, stop, tag, resampling=self.rates_dt)
      except Exception as e:
        log_print(f'Model {m01} errors for {tag}, falling back to m0 : {e}', self.logger)
        data = pd.DataFrame()
    elif 'm1' in model:
      m01 = 'm1'
      mfile = self.models[(city,tag)][m01]
      data = pd.read_csv(mfile, sep=';')
    else:
      data = pd.DataFrame()

    #print(data)
    #print(f'tag {tag}')

    if not tag in data.columns:
      m01 = 'm0'
      mfile = self.models[(city,tag)][m01]
      data = pd.read_csv(mfile, sep=';')
      dtot = self.params[city]['daily_t']
      data = self.rescale_data(start, stop, data, tot=dtot).rename(columns={'data':tag})
      #print('rescaled\n', data)

    log_print(f'Data created for ({city}, {tag}) mode {m01}', self.logger)
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

    if not os.path.exists(m0filename) or True:
      generic_monday = '2020-05-04 12:00:00' # just because it's a monday
      weekdays = [ t.strftime('%a').lower() for t in [ datetime.strptime(generic_monday, self.date_format) + timedelta(days=i) for i in range(7) ] ]

      midn = datetime.strptime('00:00:00', self.time_format)
      rates_per_day = 24 * 60 * 60 // self.rates_dt
      ttrates = { t : -1 for t in [ (midn + i*timedelta(seconds=self.rates_dt)).time() for i in range(rates_per_day) ] }

      ### fake data generation #############################
      tt = ttrates.copy()
      for t in tt:
        if   t < datetime.strptime('06:00:00', self.time_format).time():
          tt[t] = 10
        elif t < datetime.strptime('07:00:00', self.time_format).time():
          tt[t] = 20
        elif t < datetime.strptime('09:00:00', self.time_format).time():
          tt[t] = 60
        elif t < datetime.strptime('11:00:00', self.time_format).time():
          tt[t] = 30
        elif t < datetime.strptime('14:00:00', self.time_format).time():
          tt[t] = 40
        elif t < datetime.strptime('19:00:00', self.time_format).time():
          tt[t] = 30
        else:
          tt[t] = 10
      self.tt_raw = tt

      runave_size = 2 * 60 * 60 // self.rates_dt # running average idx interval from time in seconds
      kern = box_centered_kernel(len(tt), runave_size)
      conv = np.fft.fftshift(np.real(np.fft.ifft( np.fft.fft(list(tt.values())) * np.fft.fft(kern) )))
      #conv = list(tt.values())
      tt = { t : v for t, v in zip( tt.keys(), conv ) }
      #############################################################

      df = pd.DataFrame([], index=tt.keys())
      #print(self.params[city]['daily_t'])
      for i, day in enumerate(weekdays):
        vals = np.asarray(list(tt.values()))
        df[day] = vals / vals.sum() * self.params[city]['daily_t']
        #print(day, df[day].sum())
      df = df.astype('int')
      df.index = pd.to_datetime(df.index, format=self.time_format)
      df.index.name = 'time'
      df.to_csv(m0filename, sep=';', header=True, index=True)

    self.models[(city, tag)] = {
      'm0' : m0filename
    }

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

    if max != None:
      dfmax = df[wday].max()
      df[wday] = (df[wday] / dfmax * max).astype('float')

    if max != None and min != None:
      dfmax = df[wday].max()
      dfmin = df[wday].min()
      df[wday] = (df[wday] / (dfmax - dfmin) * (max - min) + min).astype('float')

    df.columns = ['data']
    mask = (df.index >= start) & (df.index < stop)
    return df[mask]

  def locals(self, start, stop, city):
    ti = datetime.strptime(start.strftime('%Y-%m-%d 00:00:00'), self.date_format)
    tf = (ti + timedelta(hours=24, seconds=-self.rates_dt) )
    fullt = pd.date_range(ti, tf, freq='{}s'.format(self.rates_dt))

    df = pd.DataFrame(index=fullt)
    tt = []
    for t in df.index:
      if t < datetime.strptime(ti.strftime('%Y-%m-%d 01:00:00'), self.date_format):
        v = 0.
      elif t < datetime.strptime(ti.strftime('%Y-%m-%d 04:00:00'), self.date_format):
        v = 0.3
      elif t < datetime.strptime(ti.strftime('%Y-%m-%d 07:00:00'), self.date_format):
        v = 0.5
      elif t < datetime.strptime(ti.strftime('%Y-%m-%d 11:00:00'), self.date_format):
        v = 0.7
      elif t < datetime.strptime(ti.strftime('%Y-%m-%d 15:00:00'), self.date_format):
        v = 1.0
      elif t < datetime.strptime(ti.strftime('%Y-%m-%d 18:00:00'), self.date_format):
        v = 0.8
      elif t < datetime.strptime(ti.strftime('%Y-%m-%d 20:00:00'), self.date_format):
        v = 0.6
      elif t < datetime.strptime(ti.strftime('%Y-%m-%d 22:00:00'), self.date_format):
        v = 0.4
      else:
        v = 0.
      tt.append(v)
    df['tot'] = tt
    tt = np.array(tt)

    runave_size = 2 * 60 * 60 // self.rates_dt # running average idx interval from time in seconds
    kern = box_centered_kernel(len(tt), runave_size)
    conv = np.fft.fftshift(np.real(np.fft.ifft( np.fft.fft(tt) * np.fft.fft(kern) )))
    df['tot-smooth'] = conv

    max_pop = self.params[city]['population']
    min_pop = 0.1*max_pop
    norm = (conv - conv.min()) / (conv.max() - conv.min())
    #print(norm.max(), norm.min())
    df['tot-smooth'] = (max_pop - min_pop) * norm + min_pop
    #print(df['tot-smooth'].min(), df['tot-smooth'].max())

    df = df[['tot-smooth']].rename({'tot-smooth':'data'})
    df = df[ (df.index >= start) & (df.index < stop) ]
    #print(df)
    return df

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
