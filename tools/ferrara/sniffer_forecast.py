#! /usr/bin/env python3

'''
Previsione per ogni stazione dell'andamento nei giorni successivi usando SARIMAX
'''

from glob import glob
from enum import Enum
from dateutil import tz
from itertools import product
from tqdm import tqdm_notebook
from pandas.plotting import lag_plot
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

wdclass = {
  'feriali' : [ 'Mon', 'Tue', 'Wed', 'Thu', 'Fri' ],
  'festivi' : [ 'Sat', 'Sun' ]
}

wdcat = {
  'Mon' : 'feriali',
  'Tue' : 'feriali',
  'Wed' : 'feriali',
  'Thu' : 'feriali',
  'Fri' : 'feriali',
  'Sat' : 'festivi',
  'Sun' : 'festivi'
}

map_hour = {0: '00', 1: '01', 2: '02', 3: '03', 4: '04',
            5: '05', 6: '06', 7: '07', 8: '08', 9: '09',
            10: '10', 11: '11', 12: '12', 13: '13', 14: '14',
            15: '15', 16: '16', 17: '17', 18: '18', 19: '19',
            20: '20', 21: '21', 22: '22', 23: '23'}

map_station = {
  "1":"Castello, Via Martiri", "2":"Hotel Carlton", "3":"Via del Podestà", "4":"Corso di P.Reno / Via Ragno" ,
  "5":"Piazza Trento Trieste", "6":"Piazza Stazione"
}

if __name__ == '__main__':
  import argparse
  import matplotlib.pyplot as plt

  def valid_date(s):
    try:
      return datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
      msg = "Not a valid date: '{0}'.".format(s)
      raise argparse.ArgumentTypeError(msg)


  def optimize_SARIMA(parameters_list, d, D, m, exog):
    """
    Return dataframe with parameters, corresponding AIC and SSE
    parameters_list - list with (p, q, P, Q) tuples
    d - integration order
    D - seasonal integration order
    m - length of season
    exog - the exogenous variable
    """
    results = []

    for param in tqdm_notebook(parameters_list):
      try:
        model = SARIMAX(exog, order=(param[0], d, param[1]), seasonal_order=(param[2], D, param[3], m)).fit(disp=-1)
      except:
        continue

      aic = model.aic
      results.append([param, aic])

    result_df = pd.DataFrame(results)
    result_df.columns = ['(p,q)x(P,Q)', 'AIC']
    #Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)

    return result_df

  parser = argparse.ArgumentParser()
  parser.add_argument('-d', '--data', help='counters data csv', required=True)
  parser.add_argument('-pc', '--plotconf', default='')
  parser.add_argument('-tt', '--time_ticks', help='set time spacing between ticks', type=int, default=300)
  parser.add_argument('-tl', '--time_labels', help='set time spacing between ticks\' labels', type=int, default=3600)
  parser.add_argument('-to', '--forecast_upto', help="The End Date for Forecasting - format YYYY-MM-DD", type=valid_date)
  args = parser.parse_args()

  # inputs and plot ticks manipulations
  filein = args.data
  base = filein[:filein.rfind('/')]
  baseSave = os.path.join(os.environ['WORKSPACE'], 'slides', 'work_lavoro', 'station_forecast')
  if not os.path.exists(baseSave): os.mkdir(baseSave)

  tok = filein[:filein.find('/')].split('_')
  dt_fmt = '%Y%m%d-%H%M%S'
  try:
    start = datetime.strptime(tok[-2], dt_fmt)
    stop = datetime.strptime(tok[-1], dt_fmt)
  except:
    start = datetime.strptime(tok[-3], dt_fmt)
    stop = datetime.strptime(tok[-2], dt_fmt)

  fname = filein[filein.find('/')+1:filein.rfind('.')].split('_')
  fine_freq = fname[-2]
  fine_freq_s = int(fine_freq[:-1])
  interp = fname[-1]

  dt_ticks = args.time_ticks
  if dt_ticks > fine_freq_s:
    tus = dt_ticks // fine_freq_s
  else:
    tus = 1
    dt_ticks = fine_freq_s
  dt_lbls = args.time_labels
  if dt_lbls > dt_ticks:
    lus = dt_lbls // dt_ticks
  else:
    lus = 1
    dt_lbls = dt_ticks
  print(f'Data sampling {fine_freq_s}. Ticks sampling {dt_ticks} u {tus}. Labels sampling {dt_lbls} u {lus}')

  # parsing input counters file
  stats = pd.read_csv(filein, sep=';', parse_dates=['time'], index_col='time')
  stats.index = stats.index.time
  tuplecol = [ tuple(c.replace('\'', '').replace('(', '').replace(')','').replace(' ','').split(',')) for c in stats.columns ]
  stats.columns = tuplecol

  """
  Perform moving average to remove fluctuations.
  Groupby station_id and compute per day (or other criteria) mean signal.
  """
  tnow = datetime.now()
  ave = stats.copy()
  ave = ave.stack()
  ave.index = pd.MultiIndex.from_tuples([ (t, i[0], i[1]) for t, i in ave.index ], names=['time', 'station_id', 'date'])
  ave = ave.reset_index()
  cols = ave.columns.values
  cols[-1] = 'cnt'
  ave.columns = cols
  #ave.station_id = ave.station_id.astype(int)
  ave.date = pd.to_datetime(ave.date)
  ave['wday'] = ave.date.dt.strftime('%a')
  dfave = ave.groupby(['station_id', 'wday', 'time']).mean()
  smooths = {}
  for sid, dfg in dfave.groupby(['station_id']):
    try:
      dfp = dfg.unstack(level=1)
      dfp.index = pd.Index([ v[1] for v in dfp.index ], name='time')
      dfp.columns = [ v[1] for v in dfp.columns ]
      dfp = dfp[['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']]
      dfp['feriali'] = dfp[['Mon', 'Tue', 'Wed', 'Thu', 'Fri']].mean(axis=1)
      dfp['festivi'] = dfp[['Sat', 'Sun']].mean(axis=1)
      dfp = dfp.astype(int)
      dfp.to_csv(f'{baseSave}/{sid}_{fine_freq}_{interp}.csv', sep=';', index=True)
    except Exception as e:
      print(f'Error with station {sid} : {e}')
      continue

    # convolve with normalized centered box
    def box_centered_kernel(tot_len, box_len):
      pad_len = tot_len - box_len
      kern = np.concatenate([
        np.zeros((pad_len // 2)),
        np.ones((box_len)) / box_len,
        np.zeros((pad_len - pad_len // 2))# for odd box_len
      ])
      return kern
    ma_size = 5 # running average idx interval from time in seconds
    kern = box_centered_kernel(len(dfp), ma_size)
    smooth = pd.DataFrame([], columns=dfp.columns, index=dfp.index)
    for c in dfp.columns:
      conv = np.fft.fftshift(np.real(np.fft.ifft( np.fft.fft( dfp[c].values ) * np.fft.fft(kern) )))
      smooth[c] = conv
    smooth.index.name='time'
    smooth.to_csv(f'{baseSave}/{sid}_{fine_freq}_{interp}_smooth.csv', sep=';', index=True)
    smooths[sid] = smooth
  tave = datetime.now() - tnow
  print(f'Averaging done in {tave} for {smooths.keys()}')
  """
  Evaluate several timeseries differences and compute stats to define
  data-driven thresholds for anomaly coefficient
  """
  datetime_fmt = '%Y-%m-%d %H:%M:%S'
  fullt = {}
  flustats = {}
  for s in smooths.keys():
    cols = [ c for c in stats.columns if c[0] == s ]
    dft = stats[cols].copy()
    dft = dft.stack()
    dft = dft.reset_index()
    dft.columns = ['time', 'date', 'cnt']
    dft['datetime'] = [ datetime.strptime(f'{d[1]} {t}', datetime_fmt) for t, d in dft[['time', 'date']].values ]
    dft = dft.sort_values(by=['datetime'])
    replicas = len(dft) // len(smooths[s])

    drange = pd.date_range(start, stop, freq='1d')[:-1] # only for stop = Y M D 00:00:00
    drange = [ d.strftime('%a') for d in drange ]
    ave_class = [ smooths[s][wdcat[d]].values for d in drange ]
    ave_day = [ smooths[s][d].values for d in drange ]
    #print(np.asarray(drange).shape)
    ave_cnt = np.concatenate(ave_class)
    ave_d_cnt = np.concatenate(ave_day)
    tidx = pd.date_range(start, stop, freq=fine_freq)[:-1] # only for stop = Y M D 00:00:
    dfave = pd.DataFrame(ave_cnt, index=tidx, columns=['ave_cnt'])
    dfave['ave_day_cnt'] = ave_d_cnt

    dfs = dft[['datetime', 'cnt']].set_index('datetime')
    dft = dfave.merge(dfs, left_index=True, right_index=True)

    kern = box_centered_kernel(len(dft), ma_size)
    dft['cnt_smooth'] = np.fft.fftshift(np.real(np.fft.ifft( np.fft.fft( dft.cnt ) * np.fft.fft(kern) )))

    dft_f = dft.copy()
    dft_f = dft_f[['cnt_smooth']]

    dft_f['date'] = dft_f.index.date
    dft_f['hour'] = dft_f.index.hour
    dft_f['map_hour'] = dft_f['hour'].map(map_hour)
    dft_f = dft_f.groupby(['date','map_hour']).sum()

    dft_f.index = [str(i)+' '+ str(j) for i,j in zip(dft_f.index.get_level_values(0), dft_f.index.get_level_values(1))]
    dft_f.index = pd.to_datetime(dft_f.index, format='%Y-%m-%d %H').strftime('%Y-%m-%d %H:%M:%S')
    dft_f.index = pd.to_datetime(dft_f.index)
    dft_f = dft_f.drop(columns='hour')

    dft_f.to_csv(f'{baseSave}/smoothed_to_compare_s_{s}.csv', sep=';', index=True)

    # l2 diff
    l2diff = (dft.cnt - dft.ave_day_cnt)**2
    l2d_smooth = np.fft.fftshift(np.real(np.fft.ifft( np.fft.fft( l2diff ) * np.fft.fft(kern) )))
    l2d_ave = l2d_smooth.mean()
    l2d_std = l2d_smooth.std()
    l2d_thresh = l2d_ave + l2d_std
    l2d_cut = l2d_smooth.copy()
    l2d_cut[ l2d_cut < l2d_thresh ] = 0
    dft['l2_diff'] = l2diff
    dft['l2_diff_smooth'] = l2d_smooth
    dft['l2_diff_cut'] = l2d_cut

    # diff
    diff = dft.cnt - dft.ave_day_cnt
    # diff_smooth = np.fft.fftshift(np.real(np.fft.ifft( np.fft.fft( diff ) * np.fft.fft(kern) )))
    diff_smooth = dft.cnt_smooth - dft.ave_day_cnt
    l1d_ave = diff_smooth.mean()
    l1d_std = diff_smooth.std()

    dft['l1_diff'] = diff
    dft['l1_diff_smooth'] = diff_smooth

    fullt[s] = dft
  #print(json.dumps(flustats, indent=2))

  if args.plotconf == '':
    selection = {
      s : {
        'start' : start,
        'stop' : stop
      }
    for s in smooths.keys() }
    ptag = 'full'
  else:
    with open(args.plotconf) as pcin:
      selection = json.load(pcin)
    selection = {
      s : {
        'start' : datetime.strptime(v['start'], datetime_fmt),
        'stop' : datetime.strptime(v['stop'], datetime_fmt)
      }
    for s, v in selection.items() }
    ptag = args.plotconf.split('_')[1].split('.')[0]

  for s in selection:
    try:
      dft = fullt[s]
    except:
      print(f'Plot: station {s} not available')
      continue
    dft = dft[ (dft.index < selection[s]['stop']) ]

    dft_f = dft.copy()
    dft_f = dft_f[['cnt_smooth']]
    dft_f['date'] = dft_f.index.date
    dft_f['hour'] = dft_f.index.hour
    dft_f['map_hour'] = dft_f['hour'].map(map_hour)
    dft_f = dft_f.groupby(['date','map_hour']).sum()

    dft_f.index = [str(i)+' '+ str(j) for i,j in zip(dft_f.index.get_level_values(0), dft_f.index.get_level_values(1))]
    dft_f.index = pd.to_datetime(dft_f.index, format='%Y-%m-%d %H').strftime('%Y-%m-%d %H:%M:%S')
    dft_f.index = pd.to_datetime(dft_f.index)
    dft_f = dft_f.drop(columns='hour')

    '''
    Compute Sarima optimization
    '''
    p = range(0, 2, 1)
    d = 1
    q = range(0, 2, 1)
    P = range(0, 2, 1)
    D = 1
    Q = range(0, 2, 1)
    m = 120
    parameters = product(p, q, P, Q)
    parameters_list = list(parameters)
    print(len(parameters_list))

    ## Uncomment to find the optimization parameters:
    result_df = optimize_SARIMA(parameters_list, 1, 1, m, dft_f['cnt_smooth'])
    print('\nSarima Optimization\n', result_df)
    best_param = result_df['(p,q)x(P,Q)'][0]
    (p_best, q_best, P_best, Q_best) = best_param

    # best_model = SARIMAX(dft_f['cnt_smooth'], order=(1, 1, 1), seasonal_order=(0, 1, 1, m)).fit(dis=-1)

    best_model = SARIMAX(dft_f['cnt_smooth'], order=(p_best, 1, q_best), seasonal_order=(P_best, 1, Q_best, m)).fit(dis=-1)
    print(best_model.summary())

    best_model.plot_diagnostics(figsize=(12,8))
    plt.suptitle(f'Diagnostic Best model')
    plt.savefig(f'{baseSave}/diagnostic_plot_station_{s}.png', dpi=150)
    plt.clf()
    '''
    END: Compute Sarima optimization
    '''
    end_endto = args.forecast_upto.date() + timedelta(days=-1)
    dft_all = pd.read_csv(f'{baseSave}/smoothed_to_compare_s_{s}.csv', header=[0], index_col=[0], sep=';', parse_dates=True)
    dft_all_upto = dft_all.loc[selection[s]['stop']:end_endto.strftime('%Y-%m-%d')]

    dft_f_from = dft_f.loc[selection[s]['start']:]
    pred_uc = best_model.get_forecast(steps=pd.to_datetime(args.forecast_upto.date().strftime('%Y-%m-%d')))
    pred_ci = pred_uc.conf_int()
    ax = dft_f_from['cnt_smooth'].plot(label='Observed Smoothed Data', c='r', figsize=(12, 8))
    dft_all_upto['cnt_smooth'].plot(label='Observed "Forecasted" Data', c='b', figsize=(12, 8))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecasted Data', c='k', linestyle='--')

    ax.set_xlabel('Date')
    ax.set_ylabel('Counter')
    ax.grid(which='major', linestyle='-')
    ax.grid(which='minor', linestyle='--')
    plt.legend()
    plt.title(f'Forecasted Prediction\n{map_station[s]} {args.forecast_upto.date()}')
    plt.savefig(f'{baseSave}/forecasted_prediction_station_{s}_to_{args.forecast_upto.date()}.png', dpi=150)
    plt.clf()
    os.remove(f'{baseSave}/smoothed_to_compare_s_{s}.csv')
