#! /usr/bin/env python3

import os
import json
import argparse
import numpy as np
from numpy.core.numeric import full
import pandas as pd
import mysql.connector
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import linregress

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--cfg', help='config file', required=True)
  parser.add_argument('-s', '--show', action='store_true')
  parser.add_argument('-db', '--db', choices=['mongo', 'mysql'], default='mysql')
  parser.add_argument('-q', '--query', action='store_true')

  args = parser.parse_args()

  save_path = os.path.join(os.environ['WORKSPACE'], 'slides', 'work_lavoro', 'dubrovnik')
  if not os.path.exists(save_path): os.mkdir(save_path)  

  output_raw = os.path.join(os.environ['WORKSPACE'], 'slides', 'work_lavoro', 'dubrovnik', 'data_analysis_raw')
  if not os.path.exists(output_raw): os.mkdir(output_raw)  

  output_agg = os.path.join(os.environ['WORKSPACE'], 'slides', 'work_lavoro', 'dubrovnik', 'data_analysis_agg')
  if not os.path.exists(output_agg): os.mkdir(output_agg)

  with open(args.cfg, encoding='utf-8') as f:
    config = json.load(f)

  start_date = config['start_date']
  stop_date  = config['stop_date']

  start_date_ave = config['startdate']
  stop_date_ave  = config['enddate']

  if pd.to_datetime(config['startdate'], format='%Y-%m-%d %H:%M:%S') < pd.to_datetime(config['start_date'], format='%Y-%m-%d %H:%M:%S'):
    data_inizio = config['startdate']
  else:
    data_inizio = config['start_date']

  if pd.to_datetime(config['enddate'], format='%Y-%m-%d %H:%M:%S') > pd.to_datetime(config['stop_date'], format='%Y-%m-%d %H:%M:%S'):
    data_fine = config['enddate']
  else:
    data_fine = config['stop_date']

  print(f'Using {args.db} to get data from {data_inizio} to {data_fine}')
  print(f'Using {args.db} for analysis from {start_date} to {stop_date}')

  def data_analysis(df, output):
    def box_centered_kernel(tot_len, box_len):
      pad_len = tot_len - box_len
      kern = np.concatenate([
      np.zeros((pad_len // 2)),
      np.ones((box_len)) / box_len,
      np.zeros((pad_len - pad_len // 2))# for odd box_len
      ])
      return kern

    ave = df.copy()

    if config['startdate'] is not None and config['enddate'] is not None:

      startdate = pd.to_datetime(config['startdate'], format='%Y-%m-%d %H:%M:%S')
      enddate = pd.to_datetime(config['enddate'], format='%Y-%m-%d %H:%M:%S')

      mask = (ave.index >= startdate) & (ave.index <= enddate)
      ave = ave.loc[mask]

      typ = output.split('_')[-1]
      print(f'Period over which the average is calculated for {typ} data: {startdate} to {enddate}')

    for locName, dfave in ave.groupby(['LOC']):
      # print(locName)
      # print(dfave)
      ave = dfave.groupby(['DATETIME',  pd.Grouper(freq='300s')]).sum()
      ave = ave.reset_index(level=[0])
      ave = ave.drop(columns='DATETIME')
      ave = ave.groupby('DATETIME').sum()

      s_fill_date = pd.to_datetime(str(df.index.date[0]) + ' ' + '00:00:00')
      e_fill_date = pd.to_datetime(str(df.index.date[-1]) + ' ' + '23:59:59')        

      fullt = pd.date_range(start=s_fill_date,end= e_fill_date, freq=f'5min')

      ave = (ave.reindex(fullt, fill_value=0).reset_index().reindex(columns=['COUNTER'])).set_index(fullt)
      # ave = (ave.reindex(fullt).reset_index().reindex(columns=['COUNTER'])).set_index(fullt)
      ave = ave.interpolate(limit_direction='both')

      ma_size = 5 # running average idx interval from time in seconds
      kern = box_centered_kernel(len(ave), ma_size)
      conv = np.fft.fftshift(np.real(np.fft.ifft( np.fft.fft( ave.COUNTER ) * np.fft.fft(kern) )))

      ave['SMOOTH'] = conv
      ave['WDAY'] = ave.index.strftime('%a')
      ave['TIME'] =  ave.index.time

      df_ave =  ave.groupby(['WDAY', 'TIME']).mean()
      df_ave = df_ave.sort_values(by=['WDAY', 'TIME'])
      df_ave.to_csv(f'{output}/df_ave_{locName}.csv', sep = ';', index = True)

    masquer = (df.index >= start_date) & (df.index <= stop_date)
    df = df.loc[masquer]

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(16, 10))

    curves = []

    for locName, dfg in df.groupby(['LOC']):
      df =  dfg.groupby(['DATETIME', pd.Grouper(freq='300s')]).sum()
      df = df.reset_index(level=[0])
      df = df.drop(columns='DATETIME')
      df = df.interpolate(limit_direction='both')

      ma_size = 5 # running average idx interval from time in seconds
      kern = box_centered_kernel(len(df), ma_size)
      conv = np.fft.fftshift(np.real(np.fft.ifft( np.fft.fft( df.COUNTER ) * np.fft.fft(kern) )))

      df['SMOOTH'] = conv
      df['DAY'] = df.index.strftime('%a')

      ts = [ t.timestamp() for t in df.index ]
      tus = 100
      lus = 4
      ts_ticks = ts[::tus]
      ts_lbl = [ t.strftime('%b %a %d') for t in df.index ]
      ts_lbl = ts_lbl[::tus]
      ts_lbl = [ t if i%lus==0 else '' for i, t in enumerate(ts_lbl)]

      curve, = axs.plot(ts, df.SMOOTH, '-o', label=f'{locName}', markersize=4)
      curves.append(curve)

      df_daily = df.resample('1D').sum()
      df_daily = df_daily.interpolate(limit_direction='both')
      df_daily = df_daily.sort_index()

      kern = box_centered_kernel(len(df_daily), ma_size)
      conv = np.fft.fftshift(np.real(np.fft.ifft( np.fft.fft( df_daily.COUNTER ) * np.fft.fft(kern) )))

      df_daily['SMOOTH'] = conv          

      tsd = [ t.timestamp() for t in df_daily.index ]
      tusd = 1
      lusd = 1
      tsd_ticks = tsd[::tusd]
      tsd_lbl = [ t.strftime('%a %d %b') for t in df_daily.index ]
      tsd_lbl = tsd_lbl[::tusd]
      tsd_lbl = [ t if i%lusd==0 else '' for i, t in enumerate(tsd_lbl)]

      df_daily['date_ordinal'] = pd.to_datetime(df_daily.index).map(datetime.toordinal)
      slope, intercept, r_value, p_value, std_err = linregress(df_daily['date_ordinal'], df_daily.SMOOTH)

      figg, axxs = plt.subplots(nrows=1, ncols=1, figsize=(16, 10))
      axxs.plot(tsd, df_daily.SMOOTH, label= 'Data', markersize=4)

      axxs.plot(np.unique(tsd), np.poly1d(np.polyfit(tsd, df_daily.SMOOTH, 1))(np.unique(tsd)), c = 'r', label = 'Trend')#, label = fr'Slope = {slope:.3f} $\pm$ {std_err:.3f} rad')

      axxs.legend()
      axxs.grid(which='major', linestyle='-')
      axxs.grid(which='minor', linestyle='--')
      axxs.set_xticks(tsd_ticks)
      axxs.set_xticklabels(tsd_lbl, rotation=45, ha='right')
      axxs.set_ylabel('Number of People per Day')
      # axxs.set_ylim(0, 70)
      plt.tight_layout()
      figg.subplots_adjust(top=0.9)
      ptitle = f'Trend in "{locName}"'
      figg.suptitle(ptitle, y=0.98)
      trend = f'{output}/trend'
      if not os.path.exists(trend): os.mkdir(trend)
      figg.savefig(f'{trend}/trend_{locName}.png')
      plt.close()

      for wday, dfw in df.groupby(['DAY']):

        s_date = pd.to_datetime(str(dfw.last('1D').index.date[0]) + ' ' + '00:00:00')
        e_date = pd.to_datetime(str(dfw.last('1D').index.date[0]) + ' ' + '23:59:59')

        fullt = pd.date_range(start=s_date,end= e_date, freq=f'5min')

        df_ave = pd.read_csv(f'{output}/df_ave_{locName}.csv', sep = ';', index_col=[0])
        df_ave = df_ave.loc[wday]
        # df_ave = df_ave.interpolate(direction='both')

        ma_size = 5 # running average idx interval from time in seconds
        kern = box_centered_kernel(len(df_ave), ma_size)
        conv = np.fft.fftshift(np.real(np.fft.ifft( np.fft.fft( df_ave.COUNTER ) * np.fft.fft(kern) )))
        df_ave['SMOOTH'] = conv     

        df_anal = dfw.last('1D')
        df_anal = df_anal.groupby('DATETIME').sum()
        # df_anal = (df_anal.reindex(fullt, fill_value=0).reset_index().reindex(columns=['COUNTER', 'SMOOTH'])).set_index(fullt)
        df_anal = (df_anal.reindex(fullt).reset_index().reindex(columns=['COUNTER', 'SMOOTH'])).set_index(fullt)
        df_anal = df_anal.interpolate(limit_direction='both')

        # if locName == 'Camera_1_Pile_Gate' and wday == 'Sun':
        #   print(locName)

        ma_size = 5 # running average idx interval from time in seconds
        kern = box_centered_kernel(len(df_anal), ma_size)
        conv = np.fft.fftshift(np.real(np.fft.ifft( np.fft.fft( df_anal.COUNTER ) * np.fft.fft(kern) )))
        df_anal['SMOOTH'] = conv 

        fig2, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 10))

        ts = [ t.timestamp() for t in df_anal.index ]
        tus = 3
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
        # ax.set_ylim(0, 50)

        plt.tight_layout()
        fig2.subplots_adjust(top=0.9)
        ptitle = f'Comparison in "{locName}"\non {df_anal.index.date[0]}'
        plt.suptitle(ptitle, y=0.98)

        comparison = f'{output}/comparison'
        if not os.path.exists(comparison): os.mkdir(comparison)
        plt.savefig(f'{comparison}/compare_{locName}_{df_anal.index.date[0].strftime("%Y%m%d")}_{wday}.png')

        plt.close()
        fig2.clf()

    s_date = pd.to_datetime(str(df.index.date[0]))
    e_date = pd.to_datetime(str(df.index.date[-1]))

    fullt = pd.date_range(start=s_date,end= e_date, freq=f'5min')

    ts = [ t.timestamp() for t in fullt ]
    tus = 100
    lus = 4
    ts_ticks = ts[::tus]
    ts_lbl = [ t.strftime('%b %a %d') for t in fullt ]
    ts_lbl = ts_lbl[::tus]
    ts_lbl = [ t if i%lus==0 else '' for i, t in enumerate(ts_lbl)]

    leg = axs.legend()
    leg.get_frame().set_alpha(0.4)
    axs.grid(which='major', linestyle='-')
    axs.grid(which='minor', linestyle='--')
    axs.set_xticks(ts_ticks)
    axs.set_xticklabels(ts_lbl, rotation=45, ha='right')
    axs.set_ylabel('Counter')

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
    ptitle = f'Number of People'
    plt.suptitle(ptitle, y=0.98)
    if args.show:
      plt.show()
    else:
      plt.savefig(f'{output}/compare.png')

    plt.clf()
    plt.close()

  try:
   if args.db == 'mysql':
      conf = config['model_data']['params']['dubrovnik']['mysql']
      db = mysql.connector.connect(
        host     = conf['host'],
        port     = conf['port'],
        user     = conf['user'],
        passwd   = conf['pwd'],
        database = conf['db']
      )
      cursor = db.cursor()

      router_map = config['model_data']['params']['dubrovnik']['router_mapping']
      router_serial = [serial for serial in router_map.values()]
      router_serial = [router_serial for sublist in router_serial for router_serial in sublist]

      router_filter = ' OR '.join([ f"d.serial = '{serial}'" for serial in router_serial ])
      
      print_sdate = data_inizio.replace(' ', '_').replace(':', '').replace('-', '')
      print_edate = data_fine.replace(' ', '_').replace(':', '').replace('-', '')      

      query = f"""
      SELECT
        d.id,
        d.name,
        d.serial
      FROM
        Devices d
      WHERE
        {router_filter}
      """
      cursor.execute(query)
      result = cursor.fetchall()
      routconv = { v[0] : v[1] for v in result }
      idserial = {v[0] : v[2] for v in result}
      # print('sid', routconv)

      if args.query:
        query = f"""
          SELECT
            ds.eventOccurredAt AS 'DATETIME',
            ds.id_device AS station,
            COUNT(ds.eventClientiId) as COUNTER
            #ds.eventClientiId
            #ds.*
          FROM	
            DevicesEvents ds
          WHERE
           ds.eventOccurredAt > ('{data_inizio}') AND ds.eventOccurredAt < ('{data_fine}') 
           AND ds.eventClientiId != ''
           AND (ds.id_device in {tuple(routconv.keys())} )
          GROUP BY ds.eventClientiId, ds.id_device
          ORDER BY ds.eventOccurredAt ASC
        """
        print(query)
        tquery = datetime.now()
        cursor.execute(query)
        result = cursor.fetchall()
        # print(result)
        tquery = datetime.now() - tquery
        print(f'Received {len(result)} mysql data in {tquery}')

        df = pd.DataFrame(result)
        df.columns =  cursor.column_names
        df.index = df.DATETIME
        df = df.drop(columns='DATETIME')
        df['LOC'] = df['station'].map(routconv)
        df.to_csv(f'{save_path}/df_{print_sdate}_{print_edate}.csv', sep = ';')
      else:
        df = pd.read_csv(f'{save_path}/df_{print_sdate}_{print_edate}.csv', sep = ';', header=[0], index_col=[0], parse_dates=True)

      df_grouped = df.copy()
      df = df.drop(columns='station')

      df_grouped['IDSERIAL'] = df_grouped['station'].map(idserial)
      df_grouped['GROUPCAMERA'] = [k for ids in df_grouped.IDSERIAL for k, v in router_map.items() if ids in v]
      df_grouped = df_grouped.drop(columns=['station', 'LOC', 'IDSERIAL'])
      df_grouped = df_grouped.rename(columns={'GROUPCAMERA':'LOC'})

      data_analysis(df, output_raw)
      data_analysis(df_grouped, output_agg)

  except Exception as e:
    print('Connection error : {}'.format(e))
