#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
from numpy.core.numeric import full
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def box_centered_kernel(tot_len, box_len):
  pad_len = tot_len - box_len
  kern = np.concatenate([
    np.zeros((pad_len // 2)),
    np.ones((box_len)) / box_len,
    np.zeros((pad_len - pad_len // 2))# for odd box_len
  ])
  return kern

def data_analysis(df, output):
  analysis_folder = f'{output}/analysis_folder'
  if not os.path.exists(analysis_folder): os.mkdir(analysis_folder)

  ave = df.copy()

  if config['start_ave_date'] is not None and config['stop_ave_date'] is not None:
    start_ave_date = pd.to_datetime(config['start_ave_date'], format='%Y-%m-%d %H:%M:%S')
    stop_ave_date = pd.to_datetime(config['stop_ave_date'], format='%Y-%m-%d %H:%M:%S')

    mask = (ave.index >= start_ave_date) & (ave.index <= stop_ave_date)
    ave = ave.loc[mask]

    # typ = output.split('_')[-1]
    print(f'Period over which the average is calculated for data: {start_ave_date} to {stop_ave_date}')

  for locName, dfave in ave.groupby(['LOC']):
    ave = dfave.groupby(['DATETIME',  pd.Grouper(freq = freq)]).sum()
    ave = ave.reset_index(level=[0])
    ave = ave.drop(columns='DATETIME')
    ave = ave.groupby('DATETIME').sum()

    s_fill_date = pd.to_datetime(str(df.index.date[0]) + ' ' + '00:00:00')
    e_fill_date = pd.to_datetime(str(df.index.date[-1]) + ' ' + '23:59:59')        

    fullt = pd.date_range(start=s_fill_date,end= e_fill_date, freq = freq)

    ave = (ave.reindex(fullt, fill_value=0).reset_index().reindex(columns=['COUNTER'])).set_index(fullt)

    ave = ave.interpolate(limit_direction='both')
    ma_size = 5 # running average idx interval from time in seconds
    kern = box_centered_kernel(len(ave), ma_size)
    conv = np.fft.fftshift(np.real(np.fft.ifft( np.fft.fft( ave.COUNTER ) * np.fft.fft(kern) )))
    ave['SMOOTH'] = conv
    ave['WDAY'] = ave.index.strftime('%a')
    ave['TIME'] =  ave.index.time
    df_ave =  ave.groupby(['WDAY', 'TIME']).mean()
    df_ave = df_ave.sort_values(by=['WDAY', 'TIME'])
    df_ave.to_csv(f'{analysis_folder}/df_ave_{locName}.csv', sep = ';', index = True)

  dt_fmt = '%Y%m%d-%H%M%S'
  if args.range == '':
    try:
      start_analysis_date = pd.to_datetime('2021-11-08 00:00:00')
      stop_analysis_date = pd.to_datetime('2021-11-14 23:59:59')
    except Exception as e:
      print(f'Exception: {e}')
  else:
    start_analysis_date = datetime.strptime(args.range.split('|')[0], dt_fmt)
    stop_analysis_date = datetime.strptime(args.range.split('|')[1], dt_fmt)

  masquer = (df.index >= start_analysis_date) & (df.index <= stop_analysis_date)
  df = df.loc[masquer]

  fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(16, 10), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
  curves = []
  lista = []

  for locName, dfg in df.groupby(['LOC']):
    dfg = dfg.drop(columns='LOC')
    df =  dfg.groupby(['DATETIME', pd.Grouper(freq = freq)]).sum()
    df = df.reset_index(level=[0])
    df = df.drop(columns='DATETIME')
    df = df.interpolate(limit_direction='both')
    df = df.groupby(df.index).sum()

    ma_size = 5
    kern = box_centered_kernel(len(df), ma_size)
    conv = np.fft.fftshift(np.real(np.fft.ifft( np.fft.fft( df.COUNTER ) * np.fft.fft(kern) )))

    df['SMOOTH'] = conv
    df['DAY'] = df.index.strftime('%a')

    ts = [ t.timestamp() for t in df.index ]
    curve, = axs[0].plot(ts, df.SMOOTH, '-o', label=f'{location_map[locName]}', markersize=4)
    curves.append(curve)
    lista.append(df.SMOOTH)


    if (df.index[-1] - df.index[0]) > timedelta(days = 1):
      for wday, dfw in df.groupby(['DAY']):
        s_date = pd.to_datetime(str(dfw.last('1D').index.date[0]) + ' ' + '00:00:00')
        e_date = pd.to_datetime(str(dfw.last('1D').index.date[0]) + ' ' + '23:59:59')

        fullt = pd.date_range(start=s_date,end= e_date, freq = freq)

        df_ave = pd.read_csv(f'{analysis_folder}/df_ave_{locName}.csv', sep = ';', index_col=[0])
        df_ave = df_ave.loc[wday]

        ma_size = 5 
        kern = box_centered_kernel(len(df_ave), ma_size)
        conv = np.fft.fftshift(np.real(np.fft.ifft( np.fft.fft( df_ave.COUNTER ) * np.fft.fft(kern) )))
        df_ave['SMOOTH'] = conv    

        df_anal = dfw.last('1D')
        df_anal = df_anal.groupby('DATETIME').sum()
        df_anal = (df_anal.reindex(fullt, fill_value=0).reset_index().reindex(columns=['COUNTER', 'SMOOTH'])).set_index(fullt)
        # df_anal = (df_anal.reindex(fullt).reset_index().reindex(columns=['COUNTER', 'SMOOTH'])).set_index(fullt)
        df_anal = df_anal.interpolate(limit_direction='both')

        ma_size = 5 
        kern = box_centered_kernel(len(df_anal), ma_size)
        conv = np.fft.fftshift(np.real(np.fft.ifft( np.fft.fft( df_anal.COUNTER ) * np.fft.fft(kern) )))
        df_anal['SMOOTH'] = conv 

        fig2, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 10))

        ts = [ t.timestamp() for t in df_anal.index ]
        tus = 1
        lus = 4
        ts_ticks = ts[::tus]
        ts_lbl = [ t.strftime('%b %a %H:%M') for t in df_anal.index ]
        ts_lbl = ts_lbl[::tus]
        ts_lbl = [ t if i%lus==0 else '' for i, t in enumerate(ts_lbl)]

        ax.plot(ts, df_anal.SMOOTH, '-bo', label=f'Daily Data', markersize=4)
        ax.plot(ts, df_ave.SMOOTH, '-ro', label=f'{wday} average', markersize=4)

        ax.legend()
        ax.grid(which='major', linestyle='-')
        ax.grid(which='minor', linestyle='--')
        ax.set_xticks(ts_ticks)
        ax.set_xticklabels(ts_lbl, rotation=45, ha='right')
        ax.set_ylabel('Counter')
        if freq == '300s':
          ax.set_ylim(0, 170)  # Grouped by 5min
        elif freq == '900s':
          ax.set_ylim(0, 400)  # Grouped by 15min
        ax.set_xlabel('Daytime [Mon Week-Day HH:MM ]')

        plt.tight_layout()
        fig2.subplots_adjust(top=0.9)
        ptitle = f'Comparison in "{location_map[locName]}"\non {df_anal.index.date[0]}\nAve from {start_ave_date.strftime("%Y-%m-%d")} to {stop_ave_date.strftime("%Y-%m-%d")} @ {freq}'
        plt.suptitle(ptitle, y=0.98)

        ave_folder = f'{analysis_folder}/ave_{start_ave_date.strftime("%Y%m%d")}_{stop_ave_date.strftime("%Y%m%d")}'
        if not os.path.exists(ave_folder): os.mkdir(ave_folder)
        week_day_folder = f'{ave_folder}/{df_anal.index.date[0].strftime("%Y%m%d")}_{wday}'
        if not os.path.exists(week_day_folder): os.mkdir(week_day_folder)
        plt.savefig(f'{week_day_folder}/camera_{locName}_day_{df_anal.index.date[0].strftime("%Y%m%d")}_wday_{wday}_at_{freq}.png')

        plt.close()
        fig2.clf()

  df_smooth = pd.DataFrame(lista)
  df_smooth = df_smooth.T
  df_smooth['mean'] = df_smooth.mean(axis=1)
  df_smooth = df_smooth[['mean']]

  s_date = pd.to_datetime(str(df.index.date[0]))
  e_date = pd.to_datetime(str(df.index.date[-1]))
  e_date = e_date + timedelta(hours=23)
  fullt = pd.date_range(start=s_date,end= e_date, freq = freq)

  ts = [ t.timestamp() for t in fullt ]
  tus = 6
  lus = 1
  ts_ticks = ts[::tus]
  ts_lbl = [ t.strftime('%b %a %d %H:%M') for t in fullt ]
  ts_lbl = ts_lbl[::tus]
  ts_lbl = [ t if i%lus==0 else '' for i, t in enumerate(ts_lbl)]

  leg = axs[0].legend()
  leg.get_frame().set_alpha(0.4)
  axs[0].grid(which='major', linestyle='-')
  axs[0].grid(which='minor', linestyle='--')
  axs[0].set_xticks(ts_ticks)
  axs[0].set_xticklabels(ts_lbl, rotation=45, ha='right')
  axs[0].set_ylabel('Counter')

  axs[1].plot(ts, df_smooth, '-o', label=f'Camera\'s mean', markersize=4)
  axs[1].grid(which='major', linestyle='-')
  axs[1].grid(which='minor', linestyle='--')
  axs[1].legend()
  axs[1].set_xticks(ts_ticks)
  axs[1].set_xticklabels(ts_lbl, rotation=45, ha='right')
  axs[1].set_ylabel('Counter')

  axs[1].set_xlabel('Date Time [Month Week-Day Day Hour Min]')

  curved = dict()
  for legline, origline in zip(leg.get_lines(), curves):
    legline.set_picker(5)
    curved[legline] = origline

  def onpick(event):
    legline = event.artist
    origline = curved[legline]
    vis = not origline.get_visible()
    origline.set_visible(vis)

    if vis:
        legline.set_alpha(1.0)
    else:
        legline.set_alpha(0.2)
    fig.canvas.draw()

  fig.canvas.mpl_connect('pick_event', onpick)

  plt.tight_layout()
  fig.subplots_adjust(top=0.9)
  ptitle = f'Number of People @ {freq}'
  plt.suptitle(ptitle, y=0.98)
  if args.show:
    plt.show()
  else:
    plt.savefig(f'{analysis_folder}/cameras_{start_analysis_date.strftime("%Y%m%d_%H%M%S")}_{stop_analysis_date.strftime("%Y%m%d_%H%M%S")}_at_{freq}.png')

  plt.clf()
  plt.close()  


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--cfg', help='config file', required=True)
  parser.add_argument('-r', '--range', help='start and stop time of the analysis - format "YYYYMMDD-HHmmss|YYYYMMDD-HHmmss"', type=str, default='')
  parser.add_argument('-t', '--dt', type=int, default=900)
  parser.add_argument('-s', '--show', help='show cameras\'plot', action='store_true')

  args = parser.parse_args()

  freq = f'{args.dt}s'

  conf_save = os.path.join(os.environ['WORKSPACE'], 'slides', 'work_lavoro', 'bari')
  if not os.path.exists(conf_save): os.mkdir(conf_save)

  bari_cameras_path = os.path.join(os.environ['WORKSPACE'], 'slides', 'vars', 'extra')
  bari_json_file = f'{bari_cameras_path}/bari_cameras.json'

  with open(bari_json_file, encoding='utf-8') as f:
    bari_json = json.load(f)

  camera_map = []
  location_map = dict()
  for cam in bari_json:
    camera_map.append(cam['id'])
    location_map[cam['id']] = 'Camera ' + cam['name']  

  with open(args.cfg, encoding='utf-8') as f:
    config = json.load(f)

  start_date = config['start_date']
  stop_date  = config['stop_date']
  data_start_label = start_date.replace(':', '').replace('-', '').replace(' ', '_')
  data_stop_label = stop_date.replace(':', '').replace('-', '').replace(' ', '_')
  try:
    df = pd.read_csv(f'{conf_save}/conf_{data_start_label}_{data_stop_label}.csv', sep = ';', index_col=[0], parse_dates=True )
  except Exception as e:
    print(f'Exception: {e}')

  data_analysis(df, conf_save)