# -*- coding: utf-8 -*-
"""
静态份额分解：站点预测 = 片区预测 × 训练期同窗历史平均份额（不训练份额模型）。
对照 A（纯站点直接预测，同 split30 结果）。目的：验证最朴素的分解能否救小站。
"""
# --- 产物统一输出到项目根 outputs/ ---
from pathlib import Path as _Path
_OUT_DIR = _Path(__file__).resolve().parents[1] / "outputs"
_OUT_DIR.mkdir(exist_ok=True)
def _out(name):
    return str(_OUT_DIR / name)

import os, sys, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import lstm_dispatch_baseline as B

WIN_BLK, WINDOWS, SEED = 24, [6, 12, 18], 42


def r2(p, t):
    st = ((t - t.mean()) ** 2).sum()
    return float(1 - ((p - t) ** 2).sum() / st) if st > 1e-9 else float("nan")


df = B.load_clean(); df, lab = B.add_cluster(df)
clusters = sorted(lab.unique())
stations = sorted(set(df["开始站点id"].dropna()) | set(df["结束站点"].dropna()))
idx3 = pd.date_range(df["ts"].min().floor("3h"), df["ts"].max().floor("3h"), freq="3h")
nb = len(idx3)
days_all = pd.date_range(df["start"].dt.normalize().min(), df["start"].dt.normalize().max(), freq="D")
D = len(days_all); d_tr, d_va = int(D * .70), int(D * .85)
bh = idx3.hour.values
blk_day = (idx3.normalize() - idx3.normalize()[0]).days.values
bw = idx3.dayofweek.values
tz = {h: k for k, h in enumerate(WINDOWS)}

dep3 = (df.assign(b=df["ts"].dt.floor("3h")).groupby(["开始站点id", "b"]).size()
        .unstack(1).reindex(index=stations, columns=idx3).fillna(0))
arr3 = (df.assign(b=df["ts"].dt.floor("3h")).groupby(["结束站点", "b"]).size()
        .unstack(1).reindex(index=stations, columns=idx3).fillna(0))
depz = (df.assign(b=df["ts"].dt.floor("3h")).groupby(["c_start", "b"]).size()
        .unstack(1).reindex(index=clusters, columns=idx3).fillna(0))
arrz = (df.assign(b=df["ts"].dt.floor("3h")).groupby(["c_end", "b"]).size()
        .unstack(1).reindex(index=clusters, columns=idx3).fillna(0))


def seq_feats(d, a, sc, t):
    sl = slice(t - WIN_BLK, t)
    return np.stack([d[sl] / sc, a[sl] / sc,
                     np.sin(2 * np.pi * bh[sl] / 24), np.cos(2 * np.pi * bh[sl] / 24),
                     np.sin(2 * np.pi * bw[sl] / 7), np.cos(2 * np.pi * bw[sl] / 7),
                     (bw[sl] >= 5).astype("float32")], 1)


# --- 片区模型 ---
ZX, ZE, ZY, ZID, ZT = [], [], [], [], []
zsc = {}
for zi, c in enumerate(clusters):
    d = depz.loc[c].values.astype("float32"); a = arrz.loc[c].values.astype("float32")
    sc = max(np.percentile(np.concatenate([d, a]), 99), 1.0); zsc[c] = sc
    for t in range(WIN_BLK, nb):
        if bh[t] not in WINDOWS: continue
        ZX.append(seq_feats(d, a, sc, t)); ZY.append([d[t] / sc, a[t] / sc])
        ZID.append(zi); ZT.append(t)
ZX = np.array(ZX, "float32"); ZY = np.array(ZY, "float32"); ZID = np.array(ZID); ZT = np.array(ZT)
dw = bw[ZT].astype("float32")
ZE = np.concatenate([np.eye(len(clusters), dtype="float32")[ZID],
                     np.eye(3, dtype="float32")[[tz[h] for h in bh[ZT]]],
                     np.sin(2 * np.pi * dw / 7)[:, None], np.cos(2 * np.pi * dw / 7)[:, None],
                     (dw >= 5)[:, None].astype("float32")], 1)
zd = blk_day[ZT]
tf.random.set_seed(SEED); np.random.seed(SEED)
ix, ie = layers.Input((WIN_BLK, 7)), layers.Input((ZE.shape[1],))
h = layers.LSTM(48, return_sequences=True)(ix); h = layers.LSTM(32)(h)
h = layers.Concatenate()([h, ie]); h = layers.Dense(32, activation="relu")(h)
h = layers.Dropout(.2)(h)
zm = models.Model([ix, ie], layers.Dense(2)(h))
zym, zys = ZY[zd < d_tr].mean(0), ZY[zd < d_tr].std(0) + 1e-6
zm.compile("adam", loss=tf.keras.losses.Huber(delta=0.5))
zm.fit([ZX[zd < d_tr], ZE[zd < d_tr]], (ZY[zd < d_tr] - zym) / zys,
       validation_data=([ZX[(zd >= d_tr) & (zd < d_va)], ZE[(zd >= d_tr) & (zd < d_va)]],
                        (ZY[(zd >= d_tr) & (zd < d_va)] - zym) / zys),
       epochs=120, batch_size=64, verbose=0,
       callbacks=[tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)])
zpred = zm.predict([ZX, ZE], verbose=0) * zys + zym
zmap = {(clusters[ZID[k]], int(ZT[k])): (zpred[k, 0] * zsc[clusters[ZID[k]]], zpred[k, 1] * zsc[clusters[ZID[k]]])
        for k in np.where(zd >= d_va)[0]}

# --- 静态份额（训练期同窗均值）---
lvl = dep3.sum(1) / nb * 8
top = lvl.sort_values(ascending=False).index[:30].tolist()
st_z = lab.reindex(top).values
rows = []
for si, st in enumerate(top):
    z = st_z[si]
    d = dep3.loc[st].values; a = arr3.loc[st].values
    dz = depz.loc[z].values; az = arrz.loc[z].values
    mask_tr = blk_day < d_tr
    for w, h0 in enumerate(WINDOWS):
        m_ = mask_tr & (bh == h0)
        sh_d = d[m_].sum() / max(dz[m_].sum(), 1e-6)
        sh_a = a[m_].sum() / max(az[m_].sum(), 1e-6)
        m_te = (blk_day >= d_va) & (bh == h0)
        idx = np.where(m_te)[0]
        pd_, pa_ = [], []
        for t in idx:
            zpair = zmap.get((z, int(t)))
            if zpair is None: continue
            pd_.append(zpair[0] * sh_d); pa_.append(zpair[1] * sh_a)
        if not pd_: continue
        rows.append(dict(st=st, win=h0, pred_dep=np.array(pd_), pred_arr=np.array(pa_),
                         true_dep=d[idx].astype(float), true_arr=a[idx].astype(float)))

res = []
for st in top:
    sub = [r for r in rows if r["st"] == st]
    if not sub: continue
    pn = np.concatenate([r["pred_dep"] - r["pred_arr"] for r in sub])
    tn = np.concatenate([r["true_dep"] - r["true_arr"] for r in sub])
    res.append(dict(st=st, r2=r2(pn, tn), dir=float((np.sign(pn) == np.sign(tn)).mean()),
                    vol=float(np.abs(tn).mean()), mae=float(np.abs(pn - tn).mean())))
pr = pd.DataFrame(res).sort_values("vol", ascending=False)
print("\n===== 静态份额分解（片区预测 × 历史同窗份额）=====")
for k in (5, 10, 15, 30):
    s = pr.head(k)
    print(f"top{k}: 净流R2 {s.r2.mean():.3f} / 方向准确率 {s.dir.mean():.3f} / 净流MAE {s.mae.mean():.2f}")
pr.round(3).to_csv(_out("static_share_exp.csv"), index=False, encoding="utf-8-sig")
print("\nsaved -> outputs/static_share_exp.csv")
