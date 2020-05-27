#! /usr/bin/env python3

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil import tz

##########################
#### log function ########
##########################
def log_print(*args, **kwargs):
  print('{} [model_slides] '.format(datetime.now()), end='', flush=True)
  print(*args, **kwargs, flush=True)

##########################
### model0 class  #####
##########################
class model_slides:

  def __init__(self, config):
    self.got_data = False
    self.date_format = '%Y-%m-%d %H:%M:%S'
    self.time_format = '%H:%M:%S'
    self.rates_dt = 5 * 60

    self.wdir = config['work_dir']
    if not os.path.exists(self.wdir): os.mkdir(self.wdir)

    models = {}
    if 'model1_file_dir' in config:
      model1_file_dir = config['model1_file_dir']
      print(model_file_dir)
      for dir in model_file_dir:
        for root, subdirs, files in os.walk(dir):
          print(files)
        try:
          pass
        except Exception as e:
          log_print('Errors parsing data dir {} : {}'.format(dir, e))
          continue
    self.models = models



  def get_data(self, start, stop, city, tag):
    if not city in self.models:
      self.create_model0(city, tag)
    else:
      mod = self.models[city]
      if not tag in mod:
        self.create_model0(city, tag)
      else:
        if 'm1' in mod[tag]:
          data = self.get_m1_data(start, stop, city, tag)
        elif 'm0' in mod[tag]:
          data = self.get_m0_data(start, stop, city, tag)
        else:
          self.create_model0(city, tag)
          data = self.get_m0_data(start, stop, city, tag)

  def get_m1_data(start, stop, city, tag):
    pass

  def get_m0_data(start, stop, city, tag):
    pass

  def create_model0(self, city, tag):
    log_print('creating model {}-{}'.format(city, tag))

  def buba(self):

    data = self.rescaled_data(start, stop, city, tag)
    if self.m0_data_file == None:
      m0_data_file = self.data_dir + '/model0_data.csv'
    else:
      m0_data_file = self.m0_data_file

    try: os.remove(m0_data_file)
    except: pass

    if not os.path.exists(m0_data_file):
      generic_monday = '2020-05-04 12:00:00' # just because it's monday
      weekdays = [ t.strftime('%a') for t in [ datetime.strptime(generic_monday, self.date_format) + timedelta(days=i) for i in range(7) ] ]

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
      df.to_csv(self.data_dir + '/prova.csv', sep=';', header=True, index=True)
      df.to_csv(m0_data_file, sep=';', header=True, index=True)

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

  def rescaled_data(self, start, stop, tot = None, max = None, min = None):
    # get raw data if needed
    if not self.got_data:
      self.get_data(start, stop)

    wday = start.strftime('%a') # fix for near midn simulation
    df = self.df_data[[wday]].copy()

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

  try:
    m0 = model_slides(config)

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
