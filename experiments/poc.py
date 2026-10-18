# -*- coding: utf-8 -*-
import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'; os.environ['OMP_NUM_THREADS']='1'
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
tf.random.set_seed(42); np.random.seed(42)

df=pd.read_excel('共享单车数据十万条1.xls')
df['start']=pd.to_datetime(df['开始时间'],format='%m/%d/%Y %H:%M')
df['date']=df['start'].dt.date; df['hour']=df['start'].dt.hour
df['ts']=df['start'].dt.floor('h')
loc=df.groupby('开始站点id')[['开始经度','开始维度']].median().dropna()
km=KMeans(n_clusters=6,n_init=10,random_state=42).fit(loc.values)
lab=pd.Series(km.labels_,index=loc.index)
df['c']=df['开始站点id'].map(lab); df['ce']=df['结束站点'].map(lab)
full=pd.date_range(df['ts'].min(),df['ts'].max(),freq='h')
res={}
def build(c):
    dep=df[df['c']==c].groupby('ts').size().reindex(full,fill_value=0)
    arr=df[df['ce']==c].groupby('ts').size().reindex(full,fill_value=0)
    return dep.values.astype('float32'), arr.values.astype('float32')

W=24
def feat_matrix(t, hour, dow, net):
    return np.stack([net,
                     np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24),
                     np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7),
                     (dow>=5).astype(float)],1)

def make(dep,arr):
    net=arr-dep
    X=[];Y=[];base_p=[];base_s=[];base_m=[]
    for i in range(len(net)-W):
        sl=slice(i,i+W)
        h=full[sl].hour.values; dw=full[sl].dayofweek.values
        X.append(feat_matrix(None,h,dw,net[sl]))
        Y.append(net[i+W])
        base_p.append(net[i+W-1]); base_s.append(net[i+W-24]); base_m.append(net[i+W-3:i+W].mean())
    return map(np.array,(X,Y,base_p,base_s,base_m))

for c in range(6):
    dep,arr=build(c)
    X,Y,bp,bs,bm=make(dep,arr)
    n=len(X); tr=int(n*0.7); va=int(n*0.85)
    sc=StandardScaler().fit(X[:tr].reshape(-1,X.shape[2]))
    f=lambda A: sc.transform(A.reshape(-1,A.shape[2])).reshape(A.shape)
    Xt,Xv,Xs=f(X[:tr]),f(X[tr:va]),f(X[va:])
    m=models.Sequential([layers.Input((W,6)),layers.LSTM(32),layers.Dropout(.2),layers.Dense(16,activation='relu'),layers.Dense(1)])
    m.compile('adam','mse')
    m.fit(Xt,Y[:tr],validation_data=(Xv,Y[tr:va]),epochs=40,batch_size=64,verbose=0,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=6,restore_best_weights=True)])
    p=m.predict(Xs,verbose=0).ravel()
    y=Y[va:]; 
    def met(a,b): 
        mae=np.abs(a-b).mean(); 
        sign=(np.sign(a)==np.sign(b)).mean()*100
        return mae, sign
    res[c]=dict(volume=float((dep+arr).mean()), lstm=met(p,y), persist=met(bp[va:],y),
                snaive=met(bs[va:],y), ma3=met(bm[va:],y))
print('%-4s %-8s | %-16s %-16s %-16s %-16s'%('簇','小时均量','LSTM MAE/方向','persist','seasonalNaive','MA3'))
for c,v in res.items():
    print('%-4d %-8.2f | %5.2f / %5.1f%%   %5.2f / %5.1f%%   %5.2f / %5.1f%%   %5.2f / %5.1f%%'%(
        c,v['volume'],v['lstm'][0],v['lstm'][1],v['persist'][0],v['persist'][1],
        v['snaive'][0],v['snaive'][1],v['ma3'][0],v['ma3'][1]))
imp=np.mean([1-res[c]['lstm'][0]/res[c]['persist'][0] for c in res])
print('\nLSTM 相对 persistence MAE 平均改善: %.1f%%'%(imp*100))
