#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
from numpy.core.numeric import full
import pandas as pd
import datetime as dt
import mysql.connector
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import linregress

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--cfg', help='config file', required=True)
  parser.add_argument('-s', '--show', action='store_true')
  parser.add_argument('-db', '--db', choices=['mongo', 'mysql'], default='mysql')
  parser.add_argument('-wd', '--weekDay', default='Thu' , type=str)
  parser.add_argument('-wn', '--weekNumber', default=27 , type=int)
  parser.add_argument('-t', '--dt', type=int, default=300)

  args = parser.parse_args()

  freq = f'{args.dt}s'

  base_save = os.path.join(os.environ['WORKSPACE'], 'slides', 'work_lavoro', 'sybenik', 'data_analysis')
  if not os.path.exists(base_save): os.mkdir(base_save)

  with open(args.cfg, encoding='utf-8') as f:
    config = json.load(f)

  start_date = config['start_date']
  stop_date  = config['stop_date']

  location_map = {'Centro1':'Prolaz', 'Piazza':'TrgRH', 'Porto':'Riva', 'Centro2':'Poljana'}

  camera_map = ['Porto', 'Piazza', 'Centro1', 'Centro2']
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

  def box_centered_kernel(tot_len, box_len):
    pad_len = tot_len - box_len
    kern = np.concatenate([
    np.zeros((pad_len // 2)),
    np.ones((box_len)) / box_len,
    np.zeros((pad_len - pad_len // 2))# for odd box_len
    ])
    return kern
  
  def data_analysis(df, output):

    comparison = f'{output}/comparison'
    if not os.path.exists(comparison): os.mkdir(comparison)

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
      ave = dfave.groupby(['DATETIME',  pd.Grouper(freq = freq)]).sum()
      ave = ave.reset_index(level=[0])
      ave = ave.drop(columns='DATETIME')
      ave = ave.groupby('DATETIME').sum()

      s_fill_date = pd.to_datetime(str(df.index.date[0]) + ' ' + '00:00:00')
      e_fill_date = pd.to_datetime(str(df.index.date[-1]) + ' ' + '23:59:59')        

      fullt = pd.date_range(start=s_fill_date,end= e_fill_date, freq = freq)

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
      df =  dfg.groupby(['DATETIME', pd.Grouper(freq = freq)]).sum()
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

      axxs.plot(np.unique(tsd), np.poly1d(np.polyfit(tsd, df_daily.SMOOTH, 1))(np.unique(tsd)), c = 'r', label = 'Trend')#), label = fr'Slope = {slope:.3f} $\pm$ {std_err:.3f} rad')

      day_list = df_daily.SMOOTH.tolist()
      changes = []
      for x1, x2 in zip(day_list[:-1], day_list[1:]):
        try:
            x1 = day_list[0]
            pct = (x2 - x1) * 100 / x1
        except ZeroDivisionError:
            pct = None
        changes.append(pct)     
      changes.insert(0,0)

      df_daily['INCREMENT'] = changes

      axxxs = axxs.twinx()
      axxxs.plot(tsd, df_daily.INCREMENT)
      axxxs.set_ylabel('Percentage of Increment from Day 1', color='b')

      axxs.legend()
      axxs.grid(which='major', linestyle='-')
      axxs.grid(which='minor', linestyle='--')
      axxs.set_xticks(tsd_ticks)
      axxs.set_xticklabels(tsd_lbl, rotation=45, ha='right')
      axxs.set_ylabel('Number of People per Day')
      # axxs.set_ylim(0, 70)
      plt.tight_layout()
      figg.subplots_adjust(top=0.9)
      ptitle = f'Trend in "{location_map[locName]}"'
      figg.suptitle(ptitle, y=0.98)
      trend = f'{output}/trend'
      if not os.path.exists(trend): os.mkdir(trend)
      figg.savefig(f'{trend}/trend_{locName}.png')
      plt.close()

      for wday, dfw in df.groupby(['DAY']):

        s_date = pd.to_datetime(str(dfw.last('1D').index.date[0]) + ' ' + '00:00:00')
        e_date = pd.to_datetime(str(dfw.last('1D').index.date[0]) + ' ' + '23:59:59')

        fullt = pd.date_range(start=s_date,end= e_date, freq = freq)

        df_ave = pd.read_csv(f'{output}/df_ave_{locName}.csv', sep = ';', index_col=[0])
        df_ave = df_ave.loc[wday]
        # df_ave = df_ave.interpolate(direction='both')

        ma_size = 5 # running average idx interval from time in seconds
        kern = box_centered_kernel(len(df_ave), ma_size)
        conv = np.fft.fftshift(np.real(np.fft.ifft( np.fft.fft( df_ave.COUNTER ) * np.fft.fft(kern) )))
        df_ave['SMOOTH'] = conv     

        df_anal = dfw.last('1D')
        df_anal = df_anal.groupby('DATETIME').sum()
        df_anal = (df_anal.reindex(fullt, fill_value=0).reset_index().reindex(columns=['COUNTER', 'SMOOTH'])).set_index(fullt)
        # df_anal = (df_anal.reindex(fullt).reset_index().reindex(columns=['COUNTER', 'SMOOTH'])).set_index(fullt)
        df_anal = df_anal.interpolate(limit_direction='both')

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
        ax.set_xlabel('Daytime [Mon Week-Day HH:MM ]')

        plt.tight_layout()
        fig2.subplots_adjust(top=0.9)
        ptitle = f'Comparison in "{location_map[locName]}"\non {df_anal.index.date[0]}\nAve from {startdate.strftime("%Y-%m-%d")} to {enddate.strftime("%Y-%m-%d")} @ {freq}'
        plt.suptitle(ptitle, y=0.98)

        comparison_day = f'{comparison}/comparison_day'
        if not os.path.exists(comparison_day): os.mkdir(comparison_day)
        averanged = f'{comparison_day}/ave_{startdate.strftime("%Y%m%d")}_{enddate.strftime("%Y%m%d")}'
        if not os.path.exists(averanged): os.mkdir(averanged)
        plt.savefig(f'{averanged}/{locName}_{df_anal.index.date[0].strftime("%Y%m%d")}_{wday}_{freq}.png')

        plt.close()
        fig2.clf()

    s_date = pd.to_datetime(str(df.index.date[0]))
    e_date = pd.to_datetime(str(df.index.date[-1]))

    fullt = pd.date_range(start=s_date,end= e_date, freq = freq)

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
    ptitle = f'Number of People @ {freq}'
    plt.suptitle(ptitle, y=0.98)
    if args.show:
      plt.show()
    else:
      plt.savefig(f'{output}/compare_at_{freq}.png')

    plt.clf()
    plt.close()  

  def spec_analysis(df, output):
    # Dal 08 Luglio al 02 Settembre plottare tutti i giovedì dalle 17.00 alle 21.00
    # - rispetto a qualunque altro giorno della settimana nello stesso orario
    # - rispetto alla media della stessa settimana

    comparison = f'{output}/comparison'
    if not os.path.exists(comparison): os.mkdir(comparison)

    for locName, df_time in df.groupby(['LOC']):
      df_time['hours'] = df_time.index.hour

      start_hour = pd.to_datetime('17:00:00', format='%H:%M:%S')
      end_hour = pd.to_datetime('21:00:00', format='%H:%M:%S')

      mask = (df_time.hours >= start_hour.hour) & (df_time.hours < end_hour.hour)
      df_time = df_time.loc[mask]

      df_time = df_time.groupby(['DATETIME',  pd.Grouper(freq = freq)]).sum()
      df_time = df_time.reset_index(level=[0])
      df_time = df_time.drop(columns=['DATETIME', 'hours'])
      df_time = df_time.groupby('DATETIME').sum()  
      df_time['WEEK_NUMBER'] = df_time.index.isocalendar().week
      for week_num, df_week in df_time.groupby(['WEEK_NUMBER']):
        if week_num == args.weekNumber:
          df_week = df_week.drop(columns=['WEEK_NUMBER'])
          mean_list = []
          for time, df_hour in df_week.groupby([df_week.index.time]):
            mean_list.append(df_hour.COUNTER.mean())

          s_fake_date = pd.to_datetime('2021-01-01 17:00:00')
          e_fake_date = pd.to_datetime('2021-01-01 20:59:59')
          fullt = pd.date_range(start = s_fake_date,end= e_fake_date, freq = freq)
          mean_df = pd.DataFrame(index=fullt)
          mean_df['COUNTER'] = mean_list  
          ma_size = 5 # running average idx interval from time in seconds
          kern = box_centered_kernel(len(mean_df), ma_size)
          conv = np.fft.fftshift(np.real(np.fft.ifft( np.fft.fft( mean_df.COUNTER ) * np.fft.fft(kern) )))
          mean_df['SMOOTH'] = conv   

          fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(16, 10))
          fig2, axs2 = plt.subplots(nrows=2, ncols=1, figsize=(16, 10), sharex=True)
          curves = []          

          for data, df_data in df_week.groupby([df_week.index.date]):

            if len(df_data) != len(mean_df):
              s_fill_date = pd.to_datetime(str(df_data.index.date[0]) + ' ' +  '17:00:00')
              e_fill_date = pd.to_datetime(str(df_data.index.date[0]) + ' ' +  '20:59:59')
              fillt = pd.date_range(start = s_fill_date,end= e_fill_date, freq = freq)
              df_data = (df_data.reindex(fillt, fill_value=0).reset_index().reindex(columns=['COUNTER'])).set_index(fillt)

            ma_size = 5 # running average idx interval from time in seconds
            kern = box_centered_kernel(len(df_data), ma_size)
            conv = np.fft.fftshift(np.real(np.fft.ifft( np.fft.fft( df_data.COUNTER ) * np.fft.fft(kern) )))

            df_data['SMOOTH'] = conv
            df_data['TIME'] = df_data.index.time

            df_data['plot_ts'] = [t.replace(year=2021, month=1, day=1) for t in df_data.index]

            ts = [ t.timestamp() for t in df_data['plot_ts'] ]
            tus = 3
            lus = 1
            ts_ticks = ts[::tus]
            ts_lbl = [ t.strftime('%H:%M') for t in df_data['plot_ts'] ]
            ts_lbl = ts_lbl[::tus]
            ts_lbl = [ t if i%lus==0 else '' for i, t in enumerate(ts_lbl)]

            wday = df_data.index.date[0].strftime('%a')

            data_list = []
            medie_list = []
            diff_list = []
            if wday == args.weekDay:
              data_list = df_data['SMOOTH'].tolist()
              medie_list = mean_df['SMOOTH'].tolist() 

              zip_obj = zip(data_list, medie_list)
              for data_listi, medie_listi in zip_obj:
                diff_list.append(((data_listi - medie_listi)/medie_listi)*100)

              ax = axs2[0]
              ax.plot(ts, df_data.SMOOTH, '-o', label=f'{wday}, {df_data.index.date[0]}', markersize=4)
              ax.plot(ts, mean_df.SMOOTH, '-o', label=f'Week Mean', markersize=4)
              ax.legend()
              ax.grid(which='major', linestyle='-')
              ax.grid(which='minor', linestyle='--')
              ax.set_xticks(ts_ticks)
              ax.set_xticklabels(ts_lbl, rotation=45, ha='right')
              ax.set_ylabel('Number of People [sampled 5 min]')

              ax = axs2[1]
              ax.plot(ts, diff_list, '-o', label=f'Plot Difference ', markersize=4)

              ax.legend()
              ax.grid(which='major', linestyle='-')
              ax.grid(which='minor', linestyle='--')
              ax.set_xticks(ts_ticks)
              ax.set_xticklabels(ts_lbl, rotation=45, ha='right')
              ax.set_ylabel('Differences [%]')

              ax.set_xlabel('Time [HH:MM]')

              plt.tight_layout()
              fig2.subplots_adjust(top=0.9)
              ptitle = f'Comparison between {wday}day and weekly mean in {location_map[locName]}\nWeek: {week_num}'
              plt.suptitle(ptitle, y=0.98)
              if args.show:
                plt.show()
              else:
                thisday = df_data.index.date[0].strftime('%Y%m%d')
                compare_weekly = f'{output}/comparison/diff_week_{week_num}'
                if not os.path.exists(compare_weekly): os.mkdir(compare_weekly)
                plt.savefig(f'{compare_weekly}/diff_{locName}__week_num_{week_num}_{wday}_{thisday}_{freq}.png')

              plt.clf()
              plt.close()

            curve, = axs.plot(ts, df_data.SMOOTH, '-o', label=f'{wday}, {df_data.index.date[0]}', markersize=4)
            curves.append(curve)
  
            leg = axs.legend()
            leg.get_frame().set_alpha(0.4)
            axs.grid(which='major', linestyle='-')
            axs.grid(which='minor', linestyle='--')
            axs.set_xticks(ts_ticks)
            axs.set_xticklabels(ts_lbl, rotation=45, ha='right')
            axs.set_ylabel('Number of People [sampled 5 min]')
            axs.set_xlabel('Time [HH:MM]')
  
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
          ptitle = f'Compare\nPeople in {location_map[locName]}, Week Number {week_num}'
          plt.suptitle(ptitle, y=0.98)
          if args.show:
            plt.show()
          else:
            comparison = f'{output}/comparison'
            all_week_day = f'{comparison}/week_data_in week_{week_num}'
            if not os.path.exists(all_week_day): os.mkdir(all_week_day)
            plt.savefig(f'{all_week_day}/week_num_{week_num}_{locName}_{freq}.png')
    
          plt.clf()
          plt.close()

  try:
   if args.db == 'mysql':
      conf = config['model_data']['params']['sybenik']['mysql']
      db = mysql.connector.connect(
        host     = conf['host'],
        port     = conf['port'],
        user     = conf['user'],
        passwd   = conf['pwd'],
        database = conf['db']
      )
      cursor = db.cursor()

      camera_filter = ' OR '.join([ f"m.CAM_NAME = '{name}'" for name in camera_map ])

      query = f"""
        SELECT
          m.UID,
          m.CAM_NAME
        FROM
          barriers_meta m
        WHERE
          {camera_filter}
      """
      cursor.execute(query)
      result = cursor.fetchall()
      camconv = { v[0] : v[1] for v in result }
      # print('sid', camconv)

      query = f"""
        SELECT
          c.DATETIME,
          c.BARRIER_UID,
          c.COUNTER
        FROM
          barriers_cnt c
        WHERE
          c.DATETIME > ('{data_inizio}') AND c.DATETIME < ('{data_fine}')
          AND
          (BARRIER_UID in {tuple(camconv.keys())} )
      """
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
      df['LOC'] = df['BARRIER_UID'].map(camconv)
      df = df.drop(columns='BARRIER_UID')

      data_analysis(df, base_save)

      spec_analysis(df, base_save)

  except Exception as e:
    print('Connection error : {}'.format(e))
