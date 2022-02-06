# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#%% VARS
start_ave_date = "2021-10-04 00:00:00"
stop_ave_date = "2021-10-31 23:59:59"

start_check_date = "2021-11-01 00:00:00"
stop_check_date = "2021-11-28 23:59:59"
dates_tocheck = ['2021-11-12','2021-11-13', '2021-11-14', '2021-11-19','2021-11-20', '2021-11-21']
weekDays = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

#%% ANALYSIS

# area countings
dfa = pd.read_csv('dfarea.csv')
dfa['DATETIME'] = pd.to_datetime(dfa['DATETIME'])
dfa['CAM_NAME'] = dfa['CAM_NAME'].astype(str)
dfa['wd'] = dfa['DATETIME'].dt.weekday # 0 = monday
dfa['timestamp'] = dfa['DATETIME'].dt.time.astype(str)
dfa['date'] = dfa['DATETIME'].dt.date.astype(str)

dfa_ave = dfa[(dfa['DATETIME'] >= start_ave_date) & (dfa['DATETIME'] <= stop_ave_date)]
dfa_ave = dfa_ave.groupby(['CAM_NAME','wd','timestamp']).mean().reset_index()
dfa_ave_tot = dfa_ave.groupby(['wd','timestamp']).sum()
dfa_ave_tot = dfa_ave_tot.reset_index().rename(columns={"area mean count":"area mean count weekday average"})

dfp = dfa[(dfa['DATETIME'] >= start_check_date) & (dfa['DATETIME'] <= stop_check_date)]
dfp_tot = dfp.groupby(['DATETIME','timestamp','wd']).sum().reset_index()
dfp_tot_pave = dfp_tot.merge(dfa_ave_tot, on=['wd','timestamp']).sort_values(by='DATETIME')

# barrier countings
dfb = pd.read_csv('dfbars.csv')
dfb['DATETIME'] = pd.to_datetime(dfb['DATETIME'])
dfb['CAM_NAME'] = dfb['CAM_NAME'].astype(str)
dfb['wd'] = dfb['DATETIME'].dt.weekday # 0 = monday
dfb = dfb.rename(columns={"bartype":"barrier direction"})
dfbp = dfb[dfb['barrier direction'].isin(['N','S'])].groupby(['DATETIME','barrier direction']).sum()



#%% PLOTS
# all area count cams
sns.lineplot(data=dfa, x='DATETIME', y='area mean count', hue='CAM_NAME')
# all area summed cams
sns.lineplot(data=dfa.groupby('DATETIME').sum(), x='DATETIME', y='area mean count')
# all barriers
sns.lineplot(data=dfb.groupby(['DATETIME','barrier direction']).sum(), x='DATETIME', y='barrier count', hue='barrier direction')
# some barriers
sns.lineplot(data=dfb[dfb['barrier direction'].isin(['N','S'])][dfb['CAM_NAME'].isin(['11','2','19'])].groupby(['DATETIME','barrier direction','CAM_NAME']).sum(), x='DATETIME', y='barrier count', hue='CAM_NAME', style='barrier direction')
#%% plot countings
fig, ax = plt.subplots(1, 1, figsize=(14,7))
sns.lineplot(data=dfp_tot_pave, x='DATETIME', y='area mean count', ax=ax, label='Month checked')
sns.lineplot(data=dfp_tot_pave, x='DATETIME', y='area mean count weekday average', ax=ax, label='Prior month weekdays mean')

# ticks = dfp_tot['timestamp'][::4]
# ax.set_xticks(ticks)
# ax.set_xticklabels(ticks, rotation=45)
ax.set(ylabel='date')
plt.legend()
plt.tight_layout()
plt.grid()

plt.savefig('bari_totcount_.png')
# plt.close()

#%% plot countings single day, only some cams
slice_cams = ['1','6','23']
dfa_ave_slice = dfa_ave[dfa_ave['CAM_NAME'].isin(slice_cams)].groupby(['wd','timestamp']).sum().reset_index()
for d in dates_tocheck:
  fig, ax = plt.subplots(1, 1, figsize=(10,7))
  dfp = dfa[dfa['date']==d]
  comp_day = dfp['wd'].iloc[0]
  dfp_comp = dfa_ave_slice[dfa_ave_slice['wd']==comp_day]
  dfp_tot = dfp[dfp['CAM_NAME'].isin(slice_cams)].groupby('timestamp').sum().reset_index()
  sns.lineplot(data=dfp_tot, x='timestamp', y='area mean count', ax=ax, label=d)
  sns.lineplot(data=dfp_comp, x='timestamp', y='area mean count', ax=ax, label=f'prior 5 {weekDays[comp_day]}s mean')
  ticks = dfp_tot['timestamp'][::4]
  ax.set_xticks(ticks)
  ax.set_xticklabels(ticks, rotation=45)
  plt.legend()
  plt.tight_layout()
  plt.savefig(f'bari_slice_{d}.png')
  plt.close()

#%% plot barriers
fig, ax = plt.subplots(1, 1, figsize=(14,7))
sns.lineplot(data=dfbp, x='DATETIME', y='barrier count', hue='barrier direction', ax=ax, alpha=0.7, linewidth=3)
plt.xlabel('date')

# """
#%% flussi sibenik
dfs = pd.read_csv('flux_ex_august.csv')
dfs['DATETIME'] = pd.to_datetime(dfs['DATETIME'])
dfsc = dfs[dfs['LOC']=='Centro1'].groupby(['BARRIER_UID','DATETIME']).sum().reset_index()
dfsc = dfsc.groupby([ pd.Grouper(freq = '15min', key='DATETIME'), 'BARRIER_UID']).sum().reset_index()

dirdic = {6:'IN', 7:'OUT'}
dfsc['barrier direction'] = dfsc['BARRIER_UID'].map(dirdic)

fig, ax = plt.subplots(1, 1, figsize=(14,7))
sns.lineplot(data=dfsc, x='DATETIME', y='COUNTER', hue='barrier direction', ax=ax, alpha=0.7, linewidth=2)
plt.xlabel('date')
plt.ylabel('barrier count')

maj_loc = mdates.DayLocator()
ax.xaxis.set_major_locator(maj_loc)
min_loc = mdates.HourLocator(interval=12)
ax.xaxis.set_minor_locator(min_loc)
ax.figure.autofmt_xdate(rotation=90, ha='center')
plt.grid(which='both')
# """
