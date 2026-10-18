# -*- coding: utf-8 -*-
import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'; os.environ['OMP_NUM_THREADS']='1'
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
tf.random.set_seed(7); np.random.seed(7)
df=pd.read_excel('共享单车数据十万条1.xls')
df['start']=pd.to_datetime(df['开始时间'],format='%m/%d/%Y %H:%M'); df['ts']=df['start'].dt.floor('h')
loc=df.groupby('开始站点id')[['开始经度','开始维度']].median().dropna()
lab=pd.Series(KMeans(6,n_init=10,random_state=42).fit(loc.values).labels_,index=loc.index)
df['c']=df['开始站点id'].map(lab); df['ce']=df['结束站点'].map(lab)
full=pd.date_range(df['ts'].min(),df['ts'].max(),freq='h'); W=24
H=full.hour.values; DW=full.dayofweek.values
def mk(dep,arr):
    n=len(full); net=arr-dep
    def feats(series):
        X=[];Y=[]
        for i in range(n-W):
            s=series[i:i+W]
            X.append(np.stack([s,np.sin(2*np.pi*H[i:i+W]/24),np.cos(2*np.pi*H[i:i+W]/24),
                               np.sin(2*np.pi*DW[i:i+W]/7),np.cos(2*np.pi*DW[i:i+W]/7),(DW[i:i+W]>=5).astype(float)],1))
            Y.append(series[i+W])
        return np.array(X),np.array(Y)
    return feats(dep),feats(arr),feats(net)
def fit(X,Y,tr,va,ep=45):
    sc=StandardScaler().fit(X[:tr].reshape(-1,X.shape[2]))
    f=lambda A: sc.transform(A.reshape(-1,A.shape[2])).reshape(A.shape)
    m=models.Sequential([layers.Input((W,6)),layers.LSTM(32),layers.Dropout(.2),layers.Dense(16,activation='relu'),layers.Dense(1)])
    m.compile('adam','mse'); m.fit(f(X[:tr]),Y[:tr],validation_data=(f(X[tr:va]),Y[tr:va]),epochs=ep,batch_size=64,verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=6,restore_best_weights=True)])
    return m.predict(f(X[va:]),verbose=0).ravel()
print('%-4s %-7s %-22s %-22s %-22s'%('簇','均值','A 直接预测净流','B 分别预测借/还再相减','C 滚动3h累计净流'))
for c in [0,3,4,1]:
    dep=df[df['c']==c].groupby('ts').size().reindex(full,fill_value=0).values.astype('float32')
    arr=df[df['ce']==c].groupby('ts').size().reindex(full,fill_value=0).values.astype('float32')
    (Xd,Yd),(Xa,Ya),(Xn,Yn)=mk(dep,arr)
    n=len(Xn); tr=int(n*.7); va=int(n*.85)
    pn=fit(Xn,Yn,tr,va)
    pd_=fit(Xd,Yd,tr,va); pa=fit(Xa,Ya,tr,va); pnet=pa-pd_
    true_net=Ya[va:]-Yd[va:]
    # C: 未来3小时累计净流
    net=arr-dep; cum=np.array([net[i+W:i+W+3].sum() for i in range(len(net)-W)])
    Xc,Yc=Xn,cum
    pc=fit(Xc,Yc,tr,va)
    def ev(p,t):
        mae=np.abs(p-t).mean(); 
        mask=np.abs(t)>0.5
        dr=(np.sign(p[mask])==np.sign(t[mask])).mean()*100 if mask.sum()>0 else float('nan')
        return 'MAE %5.2f 方向%5.1f%%'%(mae,dr)
    print('%-4d %-7.2f %-22s %-22s %-22s'%(c,(dep+arr).mean()/2,ev(pn,true_net),ev(pnet,true_net),ev(pc,cum[va:])))
