# -*- coding: utf-8 -*-
import pandas as pd, numpy as np
df=pd.read_excel('共享单车数据十万条1.xls')
df['start']=pd.to_datetime(df['开始时间'],format='%m/%d/%Y %H:%M')
df['date']=df['start'].dt.date; df['hour']=df['start'].dt.hour; df['dow']=df['start'].dt.dayofweek
# 覆盖度
dep=df.groupby('开始站点id').size().sort_values(ascending=False)
cum=dep.cumsum()/dep.sum()
for p in [0.5,0.8,0.9]:
    print('覆盖 %d%% 流量需要站点数: %d / %d'%(p*100,(cum<=p).sum()+1,len(dep)))
print('日均出发>=10 的站点数:',(dep/85>=10).sum(),' >=5:',(dep/85>=5).sum(),' <1:',(dep/85<1).sum())
# 空间聚类
loc=df.groupby('开始站点id')[['开始经度','开始维度']].median().dropna()
from sklearn.cluster import KMeans
for k in [4,6,8]:
    km=KMeans(n_clusters=k,n_init=10,random_state=42).fit(loc.values)
    print('k=%d 簇大小:'%k, np.bincount(km.labels_))
km=KMeans(n_clusters=6,n_init=10,random_state=42).fit(loc.values)
lab=pd.Series(km.labels_,index=loc.index,name='cluster')
df['c']=df['开始站点id'].map(lab)
print('\n各簇 行程量占比:'); print((df['c'].value_counts(normalize=True).sort_index()*100).round(1).to_string())
print('\n各簇 工作日小时净流(到达-出发) 均值/天:')
w=df[df['dow']<5]; nd=w['date'].nunique()
piv=[]
for c,g in w.groupby('c'):
    d=g.groupby('hour').size()/nd; a=g.groupby('结束站点').size()  # placeholder
g_arr=df[df['dow']<5].assign(cc=df['结束站点'].map(lab))
tab=pd.DataFrame(index=range(24))
for c in sorted(df['c'].dropna().unique()):
    depc=w[w['c']==c].groupby('hour').size()/nd
    arrc=g_arr[g_arr['cc']==c].groupby('hour').size()/nd
    net=(arrc.reindex(range(24),fill_value=0)-depc.reindex(range(24),fill_value=0))
    tab[int(c)]=net.round(2)
print(tab.to_string())
print('\n簇空间中心:'); print(loc.groupby(lab).median().round(4).to_string())
print('\n=== 站点漂移量（85天累计净流，缺车信号代理）===')
arr=df.groupby('结束站点').size(); dp=df.groupby('开始站点id').size()
drift=(arr.reindex(dp.index.union(arr.index),fill_value=0)-dp.reindex(dp.index.union(arr.index),fill_value=0)).sort_values()
print('最严重漂移 top5 缺车:',drift.head(5).to_dict())
print('最严重漂移 top5 积压:',drift.tail(5).to_dict())
print('\n=== 单量最大站点 3005 的日曲线（探针）===')
s=df[(df['开始站点id']==3005)&(df['dow']<5)]
print((s.groupby('hour').size()/nd).round(1).to_string())
