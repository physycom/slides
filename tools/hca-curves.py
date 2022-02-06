#HCA analysis for curve shapes
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from scipy.signal import savgol_filter
import time

#%%
def norm(a):
  tot = a.sum()
  if tot!=0: return a/tot
  else:      return a

def dist(a,b,t='euclidean'):
  if t == 'euclidean':
    return(np.sqrt(np.sum((a-b)**2)))  # or   ssd.euclidean(a,b)
  elif t == 'correlation':
    return(ssd.correlation(a,b))
  
#%% INPUT

dffile = 'coils_ave_200201-000000_200331-234500.csv'
nan_thresh = 50 # droppa le spire con troppi nan
dist_type = 'euclidean' # norma L2
# dist_type = 'correlation' # distanza di correlazione
# hca_method = 'complete'
hca_method = 'ward'
sm_window = 21 # smoothing window, deve essere dispari
sm_order = 3 # smoothing polynomial order
use_smooth = True # usare i dati smooth o i dati raw

plt.switch_backend('Agg')
plt.ioff()

dffspire = 'coils-dir-mapping_template.csv' # df con i nomi delle spire utilizzate nel modello
dfspire = pd.read_csv(dffspire,sep=';')
dfspire['tag'] = dfspire['coil_id'].astype(str)+'_'+dfspire['dir_reg'].astype(str)
tags_sp = list(dfspire['tag'])


df = pd.read_csv(dffile, sep=';')
# trova le spire con troppi nan
drop_list = []
for c in df.columns:
    if df[c].isna().sum()>nan_thresh:
        drop_list.append(c)
# trova anche le spire nel verso opposto
for s in drop_list:
  lc = s[-1]
  if lc == '0': invs = s[:-1]+'1'
  else:         invs = s[:-1]+'0'
  if (invs not in drop_list) and (invs in df.columns) : drop_list.append(invs) 
  
df = df.drop(drop_list, axis=1).fillna(0)

tags = list(df.columns.drop(['wday','wday_n','time']))
tagn = len(tags)
week = np.unique(df['wday'])

#smoothing
dfsm = df.apply(lambda x: savgol_filter(x,sm_window,sm_order) if x.name in tags else x)
dfc = dfsm if use_smooth else df
  
#%% DISTANCE MATRIX

# dividi per giorno della settimana
for day in week:
  t_c = time.time()
  print(f'Elaborating {day}')
  dfs = dfc.loc[dfc['wday']==day]
  dist_mat = np.zeros((tagn,tagn))
  for j in range(0,tagn-1):
    for k in range(j+1,tagn):
      distance = dist(norm(dfs[tags[j]].values),norm(dfs[tags[k]].values),dist_type)
      dist_mat[j,k] = dist_mat[k,j] = distance
      
  t_c1 = time.time()
  print('  -distance matrix done in: ',t_c1-t_c)
  
  fig = plt.figure(figsize=(30, 15))
  ax = fig.add_subplot(111)
  sch.dendrogram(sch.linkage(ssd.squareform(dist_mat),method=hca_method), labels=tags)
  
  t_c2 = time.time()
  print('  -dendrogram done in: ',t_c2-t_c1)
  
  plt.title(f'HCA dendrogram day:{day}, method: {hca_method}, distance: {dist_type}')
  plt.tight_layout()
  for t in ax.xaxis.get_ticklabels():
    if t.get_text() in tags_sp: t.set_color('red') 
  plt.savefig(f'./hca/HCA_dendrogram_{day}_{hca_method}_{dist_type}.png')
  plt.close()
  
  t_c3 = time.time()
  print('  -graph done in : ',t_c3-t_c2)
        
