# -*- coding: utf-8 -*-
import pandas as pd, numpy as np
df=pd.read_excel('共享单车数据十万条1.xls')
df['start']=pd.to_datetime(df['开始时间'],format='%m/%d/%Y %H:%M')
df['date']=df['start'].dt.date; df['hour']=df['start'].dt.hour; df['dow']=df['start'].dt.dayofweek
dep=df.groupby(['开始站点id','date','hour']).size().rename('dep')
arr=df.groupby(['结束站点','date','hour']).size().rename('arr')
idx=pd.MultiIndex.from_product([sorted(set(df['开始站点id'])|set(df['结束站点'])),sorted(df['date'].unique()),range(24)],names=['sid','date','hour'])
m=pd.concat([dep,arr],axis=1).reindex(idx).fillna(0)
print('站点-小时 面板: %d 行 (120站 x 85天 x 24h)'%len(m))
print('出发为0的格子占比: %.1f%%'%((m['dep']==0).mean()*100))
print('单格期望: %.2f'%m['dep'].mean())
print('每站每天总出发 分布:'); print(m.groupby(['sid','date'])['dep'].sum().describe().round(1).to_string())
print('\n=== 关键站点 工作日 小时净流(到达-出发)均值 ===')
m2=m.reset_index(); m2['dow']=pd.to_datetime(m2['date']).dt.dayofweek
w=m2[m2['dow']<5]
for sid in [3005,3042,3014,3068,3082,3000]:
    g=(w[w.sid==sid].groupby('hour')[['dep','arr']].sum())
    prof=(g['arr']-g['dep'])/w['date'].nunique()
    print(sid, '早7-9:%+.1f 午12-14:%+.1f 晚17-19:%+.1f'%(prof.loc[7:9].mean(),prof.loc[12:14].mean(),prof.loc[17:19].mean()))
print('\n=== 每日总量趋势 ===')
d=df.groupby('date').size()
print(d.head(3).to_dict(),'...',d.tail(3).to_dict())
import statistics
x=np.arange(len(d)); print('线性趋势斜率: %.2f 单/天, 首周均值 %.0f 末周均值 %.0f'%(np.polyfit(x,d.values,1)[0],d.values[:7].mean(),d.values[-7:].mean()))
print('周内均值(0=周一):'); print(d.groupby(pd.to_datetime(d.index).dayofweek).mean().round(1).to_string())
print('\n=== 站点空间聚类（kmeans, 经纬度）===')
try:
    from sklearn.cluster import KMeans
    loc=df.groupby('开始站点id')[['开始经度','开始维度']].median()  # 39=lat,116=lon
    km=KMeans(n_clusters=6,n_init=10,random_state=0).fit(loc.values)
    loc['c']=km.labels_
    print(loc['c'].value_counts().sort_index().to_string())
    print('簇中心:'); print(loc.groupby('c').median().round(4).to_string())
except Exception as e:
    print('sklearn不可用',e)
print('\n=== 时长特征（用于还车时间建模）===')
print(df['骑行时间'].quantile([.5,.75,.9,.95,.99]).to_dict())
print('Round Trip 平均时长 %.1f, One Way %.1f'%(df[df['骑行路线类别']=='Round Trip']['骑行时间'].mean(),df[df['骑行路线类别']=='One Way']['骑行时间'].mean()))
print('\n=== OD 集中度 ===')
od=df.groupby(['开始站点id','结束站点']).size().sort_values(ascending=False)
print('OD对总数:',len(od),' top10占比 %.1f%%'%(od.head(10).sum()/od.sum()*100))
print(od.head(8).to_string())
