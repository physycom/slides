#! /usr/bin/env python3

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil import tz
import calendar

##########################
#### log function ########
##########################
def log_print(*args, **kwargs):
  print('{} [model0] '.format(datetime.now()), end='', flush=True)
  print(*args, **kwargs, flush=True)

##########################
#### model0 class  #####
##########################
class model0:

  def __init__(self, config):
    self.got_data = False
    self.date_format = '%Y-%m-%d %H:%M:%S'
    self.time_format = '%H:%M:%S'
    self.rates_dt = 5 * 60
    self.m0_data_file = None

    self.data_dir = config['data_dir']
    if not os.path.exists(self.data_dir): os.mkdir(self.data_dir)

  def get_data(self, start, stop):
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
      #df.to_csv(self.data_dir + '/prova.csv', sep=';', header=True, index=True)
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

##########################
#### model1 class  #####
##########################
class model1:

  def __init__(self, config):
    self.got_data = False
    self.date_format = '%Y-%m-%d %H:%M:%S'
    self.time_format = '%H:%M:%S'
    self.rates_dt = 5 * 60
    self.m1_data_file = None

    self.data_dir  = config['data_dir']
    self.input_dir = config['input_dir']
    self.input_rescale_ita = config['input_rescale_ita']
    self.input_rescale_str = config['input_rescale_str']

    if not os.path.exists(self.data_dir): os.mkdir(self.data_dir)

  def get_data(self, start, stop):
    if self.m1_data_file == None:
      m1_data_file = self.data_dir + '/model1_data.csv'
    else:
      m1_data_file = self.m1_data_file

    try: os.remove(m1_data_file)
    except: pass

    if not os.path.exists(m1_data_file):
      df_collection=pd.DataFrame()
      for file_name in os.listdir(self.input_dir):
        with open(os.path.join(self.input_dir, file_name)) as input_file:
          #print(file_name)
          df = pd.read_csv(input_file, sep=';')
          df['weekday'] = [datetime.strptime(i, "%Y-%m-%d").weekday() for i in df.Timestamp]
          df_day = df.groupby(['weekday','start_hour'])
          df_mean = df_day.Veicoli.mean()
          df_mean = df_mean.to_frame()
          df_mean.reset_index(level=['start_hour', 'weekday'], inplace=True)
          if df_collection.empty:
            df_collection = df_mean
          else:
            df_collection['Veicoli'] = df_mean['Veicoli']+df_collection['Veicoli']

      df_grouped = df_collection.groupby('weekday')
      df_data = pd.DataFrame()
      weekdays_name = list(calendar.day_abbr)

      for name, group in df_grouped:
        if df_data.empty:
          df_data = group.copy()
          df_data.index = pd.to_datetime(df_data.start_hour, format = self.time_format)
          df_data.index.name = 'time'
          df_data.rename(columns={'Veicoli':weekdays_name[name]}, inplace=True)
          df_data = df_data.drop(['start_hour', 'weekday'], axis=1)
        else:
          df_w = group.copy()
          df_w.index = pd.to_datetime(df_w.start_hour, format = self.time_format)
          df_w.index.name = 'time'
          df_w.rename(columns={'Veicoli':weekdays_name[name]}, inplace=True)
          df_w = df_w.drop(['start_hour', 'weekday'], axis=1)
          df_data = pd.concat([df_data, df_w], axis=1)
      df_data = df_data.astype('int')
      df_data.to_csv(m1_data_file, sep=';', header=True, index=True)

    df_data = pd.read_csv(m1_data_file, sep=';')
    df_data.time = pd.to_datetime(df_data.time, format=self.date_format)
    df_data.time = [ t.replace(
        year=start.year,
        month=start.month,
        day=start.day
      )
      for t in df_data.time
    ]
    self.df_data_day = df_data.set_index('time')

    self.got_data = True
    self.df_data = df_data
    return df_data

  def rescaled_data(self, start, stop):
    # get raw data if needed
    if not self.got_data:
      self.get_data(start, stop)

    wday = start.strftime('%a') # fix for near midn simulation
    df = self.df_data[[wday]].copy()
    df.set_index(self.df_data.time, inplace=True)

    df_ita = pd.read_csv(self.input_rescale_ita, sep=';')
    df_str = pd.read_csv(self.input_rescale_str, sep=';')

    if start.year > 2018:
      year = '2018'
    else:
      year = start.year

    total_ita = df_ita[df_ita.Month == calendar.month_name[start.month]][year].get_values()[0]
    total_str = df_str[df_str.Month == calendar.month_name[start.month]][year].get_values()[0]
    days = calendar.monthrange(int(year), int(start.month))[1]
    total_day_ita = int(total_ita/days)
    total_day_str = int(total_str/days)

    tot = total_day_ita + total_day_str
    dftot = df[wday].sum()
    df[wday] = (df[wday] / dftot * tot).astype('float')

    df.columns = ['data']
    df = df.astype('int')

    mask = (df.index >= start) & (df.index < stop)
    df = df[mask]
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
    m0 = model0(config)

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
