#HCA analysis for curve shapes
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy.spatial.distance as ssd
from scipy.signal import savgol_filter
import scipy.cluster.hierarchy as sch
import time
import branca.colormap as cm
import geopy
import geopy.distance
import folium
from folium.plugins import BeautifyIcon
import dtw

#%%
def norm(a):
  tot = a.sum()
  if tot!=0: return a/tot
  else:      return a

def dist(a,b,t='euclidean'):
  if t == 'euclidean':
    return np.sqrt(np.sum((a-b)**2)) # or   ssd.euclidean(a,b)
  elif t == 'correlation':
    return ssd.correlation(a,b)
  elif t == 'dtw':
    return dtw.dtw(a,b,distance_only=True).distance
    # with radius 8 and 15min data interval there is a 2 hour range for dtw

def inv_spire(s):
  lc = s[-1]
  if lc == '0': return s[:-1]+'1'
  else:         return s[:-1]+'0'

#%% INPUT

dffile = 'coils_table_200201-000000_200331-234500.csv' # febbraio-marzo 2020
# dffile = 'coils_table_210331-220000_210430-214500.csv' # aprile 2021
# dffile = 'coils_table_210430-220000_210531-214500.csv' # maggio 2021

dist_type = 'euclidean' # norma L2
# dist_type = 'correlation' # distanza di correlazione
# dist_type = 'dtw' # dynamic time warping distance

hca_method = 'ward'
# hca_method = 'complete'

nan_t = 5 # %. droppa le spire con troppi nan
sm_window = 9 # smoothing window, deve essere dispari
sm_order = 3 # smoothing polynomial order
use_smooth = True # usare i dati smooth o i dati raw
normalize = True # normalizzare i dati per giornata

# plt.ioff()

df = pd.read_csv(dffile, sep=';')
nan_thresh = len(df)*nan_t/100
tagn_orig = len(df.columns) -1
"""
# usa solo le spire utilizzate nel modello
dffspire = 'coils-dir-mapping_template.csv'
dfspire = pd.read_csv(dffspire,sep=';')
dfspire['tag'] = dfspire['coil_id'].astype(str)+'_'+dfspire['dir_reg'].astype(str)
tags_sp = list(dfspire['tag'])
df = df[['datetime']+tags_sp]
"""

# trova le spire con troppi nan
drop_list = []
for c in df.columns:
    if df[c].isna().sum()>nan_thresh:
        drop_list.append(c)
# trova anche le spire nel verso opposto
for s in drop_list:
  invs = inv_spire(s)
  if (invs not in drop_list) and (invs in df.columns) : drop_list.append(invs)
df = df.drop(drop_list, axis=1)

df = df.fillna(0)
df['datetime'] = pd.to_datetime(df['datetime'])

tags = list(df.columns.drop('datetime'))
tagn = len(tags)
print(f'Using {tagn} / {tagn_orig} spires')

#smoothing
if use_smooth:
  dfsm = df.apply(lambda x: savgol_filter(x,sm_window,sm_order) if x.name in tags else x)
  dfc = dfsm
  """
  # smooth plot example
  plt.title('Smoothing example for spire 636_2')
  plt.plot(df['datetime'],df['636_2'],'-r')
  plt.plot(dfc['datetime'],dfc['636_2'],'-b')
  plt.legend(['raw','smooth'])
  """
else:
  dfc = df

group = dfc.groupby(pd.Grouper(key='datetime', freq='D'))

#%% DISTANCE MATRIX

dist_mat = []
for day, dfg in group:
  t_mat = []
  print('Elaborating '+day.strftime('%d-%m-%Y'))
  for j in range(0,tagn-1):
    for k in range(j+1,tagn):
      if normalize:
        distance = dist(norm(dfg[tags[j]].values),norm(dfg[tags[k]].values),dist_type)
      else:
        distance = dist(dfg[tags[j]].values,dfg[tags[k]].values,dist_type)
      t_mat.append([distance,j,k])
  dist_mat.append(t_mat)
dist_mat = np.array(dist_mat) # pairwise long form distance matrix

dfdevs = pd.DataFrame()
dfdevs['tag1']= dist_mat[0,:,1].astype(int)
dfdevs['tag2']= dist_mat[0,:,2].astype(int)
dfdevs['spira1'] = [tags[int(a)] for a in dfdevs['tag1'].values]
dfdevs['spira2'] = [tags[int(a)] for a in dfdevs['tag2'].values]
dfdevs['std'] = np.std(dist_mat,axis=0)[:,0]
dfdevs['sum'] = np.sum(dist_mat,axis=0)[:,0]

# add all columns before sorting
dfstds = dfdevs.sort_values(by='std')
dfsums = dfdevs.sort_values(by='sum')

dfdevs.to_csv('df_spire_dist.csv')

#%% PLOTS
pos = 0 # Nth best pair
ordtype = 'sum' # sum or std?

titolo = f'Spire countings, ordered by curve distance {ordtype}, '

if   ordtype == 'sum': data = dfsums
elif ordtype == 'std': data = dfstds
if normalize: titolo+= 'normalized, '
if use_smooth: titolo+= 'smooth, '

spira1 = data['spira1'].iloc[pos]
spira2 = data['spira2'].iloc[pos]
titolo+= f'{pos+1}° best pair: {spira1} / {spira2}'

fig = plt.figure(figsize=(12, 7))
if normalize:
  plt.plot(dfc.datetime.values, norm(dfc[spira1]), 'r', alpha=0.8)
  plt.plot(dfc.datetime.values, norm(dfc[spira2]), 'b', alpha=0.8)
else:
  plt.plot(dfc.datetime.values, dfc[spira1], 'r', alpha=0.8)
  plt.plot(dfc.datetime.values, dfc[spira2], 'b', alpha=0.8)
plt.title(titolo)
plt.xlabel('Date')
plt.ylabel('Countings')
plt.legend([spira1,spira2])

#%% sum vs std plots

# plt.title('Spire pairs daily curve sum')
# plt.plot(dfsums['sum'].values,'-b', linewidth=1)
plt.title('Normalized daily curve distances sum vs std')
plt.plot(norm(dfsums['std'].values),',r')
plt.plot(norm(dfsums['sum'].values),'-b')
plt.legend(['std','sum'])
plt.xlabel('Spire pair ordered by sum')

#%% geoplot
# top N link and worst N link
topn = 100

# setup
coilsdf = pd.read_csv('er_barriers.csv', sep=';')
center_coords = coilsdf[['Lat', 'Lon']].mean().values

coord_mat = {}
for s in tags:
  coilid = int(s.split('_')[0])
  coords = list(coilsdf[coilsdf['ID']==coilid][['Lat','Lon']].values[0])
  coord_mat[s] = coords

maxv, minv = np.max(data[ordtype].values), np.min(data[ordtype].values)
colormap = cm.LinearColormap(colors=['blue','yellow','red'], index=[minv,maxv/2,maxv], vmin=minv, vmax=maxv)

# top N
dataslice = data[:topn].copy()
mappa = folium.Map(location=center_coords, tiles='cartodbpositron', control_scale=True, zoom_start=9)
for cid, row in dataslice.iterrows():
  inp,out = row['spira1'], row['spira2']
  folium.Marker(location=coord_mat[inp], popup=inp).add_to(mappa)
  folium.Marker(location=coord_mat[out], popup=out).add_to(mappa)
  folium.PolyLine([coord_mat[inp],coord_mat[out]],color=colormap(row[ordtype]),weight=2,popup='{}>{}:{}'.format(inp,out,row[ordtype])).add_to(mappa)

mappa.add_child(colormap)
mappa.save(f'spire_top_{topn}.html')

# worst N
mappa = folium.Map(location=center_coords, tiles='cartodbpositron', control_scale=True, zoom_start=9)
for cid, row in data[-topn:].iterrows():
  inp,out = row['spira1'], row['spira2']
  folium.Marker(location=coord_mat[inp], popup=inp).add_to(mappa)
  folium.Marker(location=coord_mat[out], popup=out).add_to(mappa)
  folium.PolyLine([coord_mat[inp],coord_mat[out]],color=colormap(row[ordtype]),weight=2,popup='{}>{}:{}'.format(inp,out,row[ordtype])).add_to(mappa)

mappa.add_child(colormap)
mappa.save(f'spire_worst_{topn}.html')

#%% check delle spire nella top che hanno link in tutte le variazioni possibli
unilist = []
for cid, row in dataslice.iterrows():
  inp,out = row['spira1'], row['spira2']
  unitag =  inp[:-2]+'-'+out[:-2]
  if unitag not in unilist:
    invi, invo = inv_spire(inp), inv_spire(out)
    con1 = (dataslice[['spira1','spira2']]==np.array([inp,invo])).all(1).any()
    con2 = (dataslice[['spira1','spira2']]==np.array([invi,out])).all(1).any()
    con3 = (dataslice[['spira1','spira2']]==np.array([invi,invo])).all(1).any()
    if con1 and con2 and con3:
      unilist.append(unitag)

print('List of all verse correlated spires: ', unilist)

mappa = folium.Map(location=center_coords,tiles='cartodbpositron',control_scale=True,zoom_start=9)
for pair in unilist:
  inp,out = pair.split('-')
  folium.Marker(location=coord_mat[inp+'_0'], popup=folium.Popup(inp,show=True)).add_to(mappa)
  folium.Marker(location=coord_mat[out+'_0'], popup=folium.Popup(out,show=True)).add_to(mappa)
  folium.PolyLine([coord_mat[inp+'_0'],coord_mat[out+'_0']],color=colormap(row[ordtype]),weight=2).add_to(mappa)

mappa.add_child(colormap)
mappa.save('spire_top_allverse.html')

#%% migliori/peggiori spire come correlazione in generale
titolo = 'Spire total distance from other spires, data '
if use_smooth: titolo+= 'smooth '
if normalize: titolo+= 'normalized '

square_mat = ssd.squareform(dfdevs[ordtype].values) # square symmetric form distance matrix
best_spire =[a.sum() for a in square_mat] # somma di tutte le differenze rispetto alle altre spire
df_best = pd.DataFrame({'spira':tags,'tot':best_spire})
df_best = df_best.sort_values(by='tot')
df_best.to_csv('df_spire_tot.csv')

plt.plot(df_best['spira'],df_best['tot'],'.b')
plt.tight_layout()
plt.title(titolo)
plt.xticks(fontsize=7,rotation=90)
plt.grid()

mappa = folium.Map(location=center_coords,tiles='cartodbpositron', control_scale=True,zoom_start=9)
for s in df_best.iloc[-10:]['spira']:
  folium.Marker(location=coord_mat[s], popup=folium.Popup(s,show=True)).add_to(mappa)
mappa.save('spire_worst_singles_10.html')

###############################################################################
#%% clustering hca
iconcolors = ['purple', 'orange', 'red', 'green', 'blue', 'pink', 'beige', 'darkred', 'darkpurple', 'lightblue', 'lightgreen', 'cadetblue', 'lightgray', 'gray', 'darkgreen', 'white', 'darkblue', 'lightred', 'black']
fig = plt.figure(figsize=(20, 12))
ax = fig.add_subplot(111)

linkg = sch.linkage(dfdevs[ordtype].values,method=hca_method)
sch.dendrogram(linkg, labels=tags)
plt.title(f'HCA dendrogram method: {hca_method}, distance: {dist_type}')
plt.xticks(fontsize=7)

#%%
cut_distance = 6 # distanza nel dendrogramma a cui tagliare per decidere il numero di cluster
clusterlist = sch.fcluster(linkg, cut_distance, criterion='distance')-1
df_cluster = pd.DataFrame({'spira':tags,'cluster':clusterlist})
nclusters = len(np.unique(clusterlist))

assert nclusters <= len(iconcolors) # troppi pochi colori per tutti i cluster altrimenti

m = folium.Map(location=center_coords,tiles='cartodbpositron', control_scale=True,zoom_start=9)
folium.TileLayer(tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', name='satellite', attr='1').add_to(m)
folium.TileLayer('openstreetmap').add_to(m)
folium.TileLayer('Stamen Terrain').add_to(m)
folium.TileLayer('Stamen Toner').add_to(m)
folium.TileLayer('Stamen Water Color').add_to(m)
folium.TileLayer('cartodbpositron').add_to(m)
folium.TileLayer('cartodbdark_matter').add_to(m)
folium.TileLayer(tiles='http://a.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png', name='osm2', attr='1').add_to(m)
folium.TileLayer(tiles='https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', name='topomap', attr='1').add_to(m)
folium.TileLayer(tiles='https://{s}.tile.thunderforest.com/transport-dark/{z}/{x}/{y}.png', name='transport dark', attr='1').add_to(m)
folium.TileLayer(tiles='https://stamen-tiles-{s}.a.ssl.fastly.net/terrain-background/{z}/{x}/{y}{r}.png', name='terrain background', attr='1').add_to(m)

layerlabel = '<span style="color: {col};">{txt}</span>'
flayer = [folium.FeatureGroup(name=layerlabel.format(col=iconcolors[c], txt=f'cluster {c+1}'), show=True) for c in range(nclusters)]

for cid, row in df_cluster.iterrows():
  s = row['spira']
  c = row['cluster']
  # i = BeautifyIcon(icon=f'{verse}', inner_icon_style=f'color:{iconcolors[c]};font-size:30px;', background_color='transparent', border_color='transparent')
  if s[-1]=='0':
    i = folium.DivIcon(html=(f'<svg height="50" width="50"> <text x="0" y="35" fill={iconcolors[c]}>0</text> </svg>'))
  else:
    i = folium.DivIcon(html=(f'<svg height="50" width="50"> <text x="6" y="35" fill={iconcolors[c]}>1</text> </svg>'))
  shp = folium.Marker(location=coord_mat[s], popup=s, icon=i)
  flayer[c].add_child(shp)

for l in flayer: m.add_child(l)
folium.map.LayerControl(collapsed=False).add_to(m)
m.save('spire_all.html')


#%% grafico di singola spira
spirap = '288_0'
plt.plot(dfc.datetime.values, df[spirap], 'b')
plt.title(f'Spire {spirap} counting')
plt.xlabel('Date')
plt.ylabel('Countings')
plt.grid()

#%% confronto con dati nuovi
df_old = pd.read_csv('df_spire_tot_old.csv')
df_new = pd.read_csv('df_spire_tot_new.csv')
idxoldl = []
idxnewl = []
for s in df_old.spira.values:
  idxoldl.append(df_old.index[df_old.spira==s][0]/len(df_old))
  if s in df_new.spira.values:   idxnewl.append(df_new.index[df_new.spira==s][0]/len(df_new))
  else: idxnewl.append(-1)

plt.plot(df_old.spira.values, idxoldl, '.b')
plt.plot(df_old.spira.values, idxnewl, '.r')
plt.xticks(fontsize=7,rotation=90)
plt.grid()
plt.title('Old vs new data normalized index in total distance order')

#%% locality map
spirap = '174_1' # spira centrale
kil_radius = 15 # semilato del box in km

p_latlon = coord_mat[spirap]
p_cen = geopy.Point(p_latlon)
p_dist = geopy.distance.distance(kilometers=kil_radius)
p_NE = p_dist.destination(point=p_cen, bearing=45) # 0 gradi = nord, senso orario
p_SW = p_dist.destination(point=p_cen, bearing=225)
box_latmin, box_latmax, box_lonmin, box_lonmax = p_SW.latitude, p_NE.latitude, p_SW.longitude, p_NE.longitude

box_spire = []
# check quali spire sono nel box
for spira in coord_mat:
  if (box_latmin < coord_mat[spira][0] < box_latmax) and (box_lonmin < coord_mat[spira][1] < box_lonmax):
    box_spire.append(spira)
if spirap in box_spire: box_spire.remove(spirap)

mappa = folium.Map(location=p_latlon,tiles='cartodbpositron',control_scale=True,zoom_start=9)
folium.Marker(location=p_latlon, popup=folium.Popup(spirap,show=True)).add_to(mappa)
for spira in box_spire:
  folium.Marker(location=coord_mat[spira], popup=folium.Popup(spira,show=True)).add_to(mappa)
  link = data[((data['spira2']==spirap) & (data['spira1']==spira)) | ((data['spira1']==spirap) & (data['spira2']==spira))][ordtype].values
  folium.PolyLine(
    [p_latlon,coord_mat[spira]],
    color=colormap(link),
    weight=2
  ).add_to(mappa)

mappa.add_child(colormap)
mappa.save(f'spire_locality_{spirap}.html')