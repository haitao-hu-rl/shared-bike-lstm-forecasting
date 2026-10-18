# -*- coding: utf-8 -*-
"""
份额分解版层次模型：站点预测 = 片区预测 × 站点份额预测。
对照组 A（纯站点直接预测），同切分同种子。
份额目标 = 站点 3h 借出 / 片区 3h 借出（归还同理），稳定且有界，适合稀疏小站。
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

dep3 = (df.assign(b=df["ts"].dt.floor("3h")).groupby(["开始站点id", "b"]).size()
        .unstack(1).reindex(index=stations, columns=idx3).fillna(0))
arr3 = (df.assign(b=df["ts"].dt.floor("3h")).groupby(["结束站点", "b"]).size()
        .unstack(1).reindex(index=stations, columns=idx3).fillna(0))
depz = (df.assign(b=df["ts"].dt.floor("3h")).groupby(["c_start", "b"]).size()
        .unstack(1).reindex(index=clusters, columns=idx3).fillna(0))
arrz = (df.assign(b=df["ts"].dt.floor("3h")).groupby(["c_end", "b"]).size()
        .unstack(1).reindex(index=clusters, columns=idx3).fillna(0))
tz = {h: k for k, h in enumerate(WINDOWS)}


def seq_feats(d, a, sc, t):
    sl = slice(t - WIN_BLK, t)
    return np.stack([d[sl] / sc, a[sl] / sc,
                     np.sin(2 * np.pi * bh[sl] / 24), np.cos(2 * np.pi * bh[sl] / 24),
                     np.sin(2 * np.pi * bw[sl] / 7), np.cos(2 * np.pi * bw[sl] / 7),
                     (bw[sl] >= 5).astype("float32")], 1)


def extra(ids, n, win, dwt):
    dw = dwt.astype("float32")
    return np.concatenate([np.eye(n, dtype="float32")[ids], np.eye(3, dtype="float32")[win],
                           np.sin(2 * np.pi * dw / 7)[:, None], np.cos(2 * np.pi * dw / 7)[:, None],
                           (dw >= 5)[:, None].astype("float32")], 1)


def train(X, E, Y, i_tr, i_va, seed=SEED):
    tf.random.set_seed(seed); np.random.seed(seed)
    ix, ie = layers.Input((WIN_BLK, 7)), layers.Input((E.shape[1],))
    h = layers.LSTM(48, return_sequences=True)(ix)
    h = layers.LSTM(32)(h)
    h = layers.Concatenate()([h, ie])
    h = layers.Dense(32, activation="relu")(h); h = layers.Dropout(.2)(h)
    m = models.Model([ix, ie], layers.Dense(2)(h))
    ym, ys = Y[i_tr].mean(0), Y[i_tr].std(0) + 1e-6
    m.compile("adam", loss=tf.keras.losses.Huber(delta=0.5))
    m.fit([X[i_tr], E[i_tr]], (Y[i_tr] - ym) / ys,
          validation_data=([X[i_va], E[i_va]], (Y[i_va] - ym) / ys),
          epochs=120, batch_size=64, verbose=0,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)])
    return m, ym, ys


# ---- 1) 片区模型（供份额法乘回去）----
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
ZE = extra(ZID, len(clusters), np.array([tz[bh[t]] for t in ZT]), bw[ZT])
zd = blk_day[ZT]
zm, zym, zys = train(ZX, ZE, ZY, np.where(zd < d_tr)[0], np.where((zd >= d_tr) & (zd < d_va))[0])
zpred = zm.predict([ZX, ZE], verbose=0) * zys + zym
zmap = {(clusters[ZID[k]], int(ZT[k])): (zpred[k, 0] * zsc[clusters[ZID[k]]], zpred[k, 1] * zsc[clusters[ZID[k]]])
        for k in np.where(zd >= d_va)[0]}

# ---- 2) 站点：A 直接 / S 份额 ----
lvl = dep3.sum(1) / nb * 8
top = lvl.sort_values(ascending=False).index[:30].tolist()
S = len(top); st_z = lab.reindex(top).values
sc_all = {st: max(np.percentile(np.concatenate([dep3.loc[st].values, arr3.loc[st].values]), 99), 1.0) for st in top}

Xa, Ea, Ya, Xs, Ys_, SIDs, WINs, Ts = [], [], [], [], [], [], [], []
for si, st in enumerate(top):
    d = dep3.loc[st].values.astype("float32"); a = arr3.loc[st].values.astype("float32")
    sc = sc_all[st]; z = st_z[si]
    dz = depz.loc[z].values.astype("float32"); az = arrz.loc[z].values.astype("float32")
    for t in range(WIN_BLK, nb):
        if bh[t] not in WINDOWS: continue
        Xa.append(seq_feats(d, a, sc, t)); Ya.append([d[t] / sc, a[t] / sc])
        sh = [d[t] / dz[t] if dz[t] > 0 else 0.0, a[t] / az[t] if az[t] > 0 else 0.0]
        Xs.append(seq_feats(d, a, sc, t)); Ys_.append(sh)
        SIDs.append(si); WINs.append(tz[bh[t]]); Ts.append(t)
Xa = np.array(Xa, "float32"); Ya = np.array(Ya, "float32")
Xs = np.array(Xs, "float32"); Ys_ = np.array(Ys_, "float32")
SIDs = np.array(SIDs); WINs = np.array(WINs); Ts = np.array(Ts)
E_ = extra(SIDs, S, WINs, bw[Ts])
d2 = blk_day[Ts]
i_tr = np.where(d2 < d_tr)[0]; i_va = np.where((d2 >= d_tr) & (d2 < d_va))[0]; i_te = np.where(d2 >= d_va)[0]

ma, ym_a, ys_a = train(Xa, E_, Ya, i_tr, i_va)
pa = (ma.predict([Xa[i_te], E_[i_te]], verbose=0) * ys_a + ym_a) * np.array([sc_all[top[s]] for s in SIDs[i_te]])[:, None]
ta = Ya[i_te] * np.array([sc_all[top[s]] for s in SIDs[i_te]])[:, None]

ms, ym_s, ys_s = train(Xs, E_, Ys_, i_tr, i_va)
sh_p = np.clip(ms.predict([Xs[i_te], E_[i_te]], verbose=0) * ys_s + ym_s, 0, 1)
zp = np.array([zmap.get((st_z[int(s)], int(t)), (0.0, 0.0)) for s, t in zip(SIDs[i_te], Ts[i_te])], dtype="float32")
ps = sh_p * zp                                        # 片区预测 × 份额预测

rows = []
for tag, p, t in (("A 纯站点直接预测", pa, ta), ("S 片区×份额", ps, ta)):
    pn, tn = p[:, 0] - p[:, 1], t[:, 0] - t[:, 1]
    per = []
    for si in range(S):
        m_ = SIDs[i_te] == si
        if m_.sum() < 10: continue
        per.append(dict(st=top[si], r2=r2(pn[m_], tn[m_]),
                        dir=float((np.sign(pn[m_]) == np.sign(tn[m_])).mean()), vol=np.abs(tn[m_]).mean(),
                        mae=float(np.abs(pn[m_] - tn[m_]).mean())))
    ps_ = pd.DataFrame(per).sort_values("vol", ascending=False)
    for k in (5, 10, 15, 30):
        sub = ps_.head(k)
        rows.append(dict(模型=tag, 子集=f"top{k}", 净流R2=round(sub.r2.mean(), 3),
                         方向准确率=round(sub.dir.mean(), 3), 净流MAE=round(sub.mae.mean(), 2)))
out = pd.DataFrame(rows)
print("\n===== 份额分解 vs 直接预测 =====")
print(out.pivot(index="子集", columns="模型", values=["净流R2", "方向准确率", "净流MAE"]).to_string())
out.to_csv(_out("share_decomp_exp.csv"), index=False, encoding="utf-8-sig")
print("\nsaved -> outputs/share_decomp_exp.csv")
