# -*- coding: utf-8 -*-
"""
层次模型：片区级 3h 预测 → 作为 top30 站点模型的输入特征。
对照 A（纯站点模型） vs C（站点 + 片区预测特征），同架构、同种子、同切分。
训练期站点模型使用真实片区值作特征（teacher forcing），测试期使用片区模型预测值。
评估：逐站 净流 R² / 方向准确率，分 top5/10/15/30 汇总。
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

WIN_BLK = 24
WINDOWS = [6, 12, 18]
SEED = 42


def r2(p, t):
    st = ((t - t.mean()) ** 2).sum()
    return float(1 - ((p - t) ** 2).sum() / st) if st > 1e-9 else float("nan")


df = B.load_clean()
df, lab = B.add_cluster(df)
clusters = sorted(lab.unique())
stations = sorted(set(df["开始站点id"].dropna()) | set(df["结束站点"].dropna()))
idx3 = pd.date_range(df["ts"].min().floor("3h"), df["ts"].max().floor("3h"), freq="3h")
nb = len(idx3)
days_all = pd.date_range(df["start"].dt.normalize().min(), df["start"].dt.normalize().max(), freq="D")
D = len(days_all)
d_tr, d_va = int(D * .70), int(D * .85)

dep3 = (df.assign(b=df["ts"].dt.floor("3h")).groupby(["开始站点id", "b"]).size()
        .unstack(1).reindex(index=stations, columns=idx3).fillna(0))
arr3 = (df.assign(b=df["ts"].dt.floor("3h")).groupby(["结束站点", "b"]).size()
        .unstack(1).reindex(index=stations, columns=idx3).fillna(0))
depz = (df.assign(b=df["ts"].dt.floor("3h")).groupby(["c_start", "b"]).size()
        .unstack(1).reindex(index=clusters, columns=idx3).fillna(0))
arrz = (df.assign(b=df["ts"].dt.floor("3h")).groupby(["c_end", "b"]).size()
        .unstack(1).reindex(index=clusters, columns=idx3).fillna(0))

blk_hour = idx3.hour.values; blk_dow = idx3.dayofweek.values
blk_day = (idx3.normalize() - idx3.normalize()[0]).days.values
tz = {h: k for k, h in enumerate(WINDOWS)}


def seq_feats(d, a, sc, t):
    sl = slice(t - WIN_BLK, t)
    return np.stack([d[sl] / sc, a[sl] / sc,
                     np.sin(2 * np.pi * blk_hour[sl] / 24), np.cos(2 * np.pi * blk_hour[sl] / 24),
                     np.sin(2 * np.pi * blk_dow[sl] / 7), np.cos(2 * np.pi * blk_dow[sl] / 7),
                     (blk_dow[sl] >= 5).astype("float32")], 1)


def extra(ids, grp_n, win, dow_t):
    oh = np.eye(grp_n, dtype="float32")[ids]
    ohw = np.eye(3, dtype="float32")[win]
    dw = dow_t.astype("float32")
    return np.concatenate([oh, ohw, np.sin(2 * np.pi * dw / 7)[:, None],
                           np.cos(2 * np.pi * dw / 7)[:, None], (dw >= 5)[:, None].astype("float32")], 1)


def train_model(X, E, Y, i_tr, i_va, hid=(48, 32), seed=SEED):
    tf.random.set_seed(seed); np.random.seed(seed)
    ix, ie = layers.Input((WIN_BLK, 7)), layers.Input((E.shape[1],))
    h = layers.LSTM(hid[0], return_sequences=True)(ix)
    h = layers.LSTM(hid[1])(h)
    h = layers.Concatenate()([h, ie])
    h = layers.Dense(32, activation="relu")(h)
    h = layers.Dropout(.2)(h)
    m = models.Model([ix, ie], layers.Dense(2)(h))
    ym, ys = Y[i_tr].mean(0), Y[i_tr].std(0) + 1e-6
    m.compile("adam", loss=tf.keras.losses.Huber(delta=0.5))
    m.fit([X[i_tr], E[i_tr]], (Y[i_tr] - ym) / ys,
          validation_data=([X[i_va], E[i_va]], (Y[i_va] - ym) / ys),
          epochs=120, batch_size=64, verbose=0,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)])
    return m, ym, ys


# ---------------- 1. 片区级 3h 模型 ----------------
ZX, ZE, ZY, ZID, ZWIN, ZT = [], [], [], [], [], []
zscale = {}
for zi, c in enumerate(clusters):
    d = depz.loc[c].values.astype("float32"); a = arrz.loc[c].values.astype("float32")
    sc = max(np.percentile(np.concatenate([d, a]), 99), 1.0); zscale[c] = sc
    for t in range(WIN_BLK, nb):
        if blk_hour[t] not in WINDOWS:
            continue
        ZX.append(seq_feats(d, a, sc, t)); ZY.append([d[t] / sc, a[t] / sc])
        ZID.append(zi); ZWIN.append(tz[blk_hour[t]]); ZT.append(t)
ZX = np.array(ZX, dtype="float32"); ZY = np.array(ZY, dtype="float32")
ZID = np.array(ZID); ZWIN = np.array(ZWIN); ZT = np.array(ZT)
ZE = extra(ZID, len(clusters), ZWIN, blk_dow[ZT])
zd = blk_day[ZT]
zi_tr = np.where(zd < d_tr)[0]; zi_va = np.where((zd >= d_tr) & (zd < d_va))[0]; zi_te = np.where(zd >= d_va)[0]
zm, zym, zys = train_model(ZX, ZE, ZY, zi_tr, zi_va)
zpred = zm.predict([ZX, ZE], verbose=0) * zys + zym

# pred_map[(zone, block)] = (dep, arr) 预测值（原始量纲）
pred_map = {}
for k in zi_te:
    pred_map[(clusters[ZID[k]], int(ZT[k]))] = (float(zpred[k, 0] * zscale[clusters[ZID[k]]]),
                                               float(zpred[k, 1] * zscale[clusters[ZID[k]]]))
zt = ZY[zi_te] * np.array([zscale[clusters[z]] for z in ZID[zi_te]])[:, None]
print(f"片区模型：训练 {len(zi_tr)}/验证 {len(zi_va)}/测试 {len(zi_te)}；"
      f"测试 dep R² {r2(zpred[zi_te, 0] * np.array([zscale[clusters[z]] for z in ZID[zi_te]]), zt[:, 0]):.2f}")

# ---------------- 2. 站点模型（A 无片区特征 / C 有） ----------------
lvl = dep3.sum(1) / nb * 8
top = lvl.sort_values(ascending=False).index[:30].tolist()
S = len(top)
st_zone = lab.reindex(top).values                      # 站点所属片区
sc_all = {st: max(np.percentile(np.concatenate([dep3.loc[st].values, arr3.loc[st].values]), 99), 1.0) for st in top}

X2, E2, Y2, SID2, WIN2, T2, ZF = [], [], [], [], [], [], []
for si, st in enumerate(top):
    d = dep3.loc[st].values.astype("float32"); a = arr3.loc[st].values.astype("float32")
    sc = sc_all[st]; z = st_zone[si]
    dz = depz.loc[z].values.astype("float32"); az = arrz.loc[z].values.astype("float32")
    zsc = zscale[z]
    for t in range(WIN_BLK, nb):
        if blk_hour[t] not in WINDOWS:
            continue
        X2.append(seq_feats(d, a, sc, t)); Y2.append([d[t] / sc, a[t] / sc])
        lvl_t = blk_day[t]
        if lvl_t >= d_va and (z, t) in pred_map:
            zf = pred_map[(z, t)]                       # 测试期：用片区预测值
        else:
            zf = (dz[t], az[t])                          # 训练/验证：用真实值（teacher forcing）
        ZF.append([zf[0] / zsc, zf[1] / zsc])
        SID2.append(si); WIN2.append(tz[blk_hour[t]]); T2.append(t)
X2 = np.array(X2, dtype="float32"); Y2 = np.array(Y2, dtype="float32")
SID2 = np.array(SID2); WIN2 = np.array(WIN2); T2 = np.array(T2); ZF = np.array(ZF, dtype="float32")
base_ext = extra(SID2, S, WIN2, blk_dow[T2])
d2 = blk_day[T2]
i_tr2 = np.where(d2 < d_tr)[0]; i_va2 = np.where((d2 >= d_tr) & (d2 < d_va))[0]; i_te2 = np.where(d2 >= d_va)[0]

res = {}
for tag, E_ in (("A 纯站点", base_ext), ("C 站点+片区特征", np.concatenate([base_ext, ZF], 1))):
    m_, ym_, ys_ = train_model(X2, E_, Y2, i_tr2, i_va2)
    p = m_.predict([X2[i_te2], E_[i_te2]], verbose=0) * ys_ + ym_
    sc_te = np.array([sc_all[st] for st in np.array(top)[SID2[i_te2]]])[:, None]
    res[tag] = dict(pred=p * sc_te, true=Y2[i_te2] * sc_te, sid=SID2[i_te2])

# ---------------- 3. 逐站评估 ----------------
rows = []
for tag, r_ in res.items():
    p, t, sid = r_["pred"], r_["true"], r_["sid"]
    pn, tn = p[:, 0] - p[:, 1], t[:, 0] - t[:, 1]
    per_st = []
    for si in range(S):
        m_ = sid == si
        if m_.sum() < 10: continue
        per_st.append(dict(st=top[si], r2=r2(pn[m_], tn[m_]),
                           dir=float((np.sign(pn[m_]) == np.sign(tn[m_])).mean()),
                           vol=np.abs(tn[m_]).mean()))
    ps = pd.DataFrame(per_st).sort_values("vol", ascending=False)
    for k in (5, 10, 15, 30):
        sub = ps.head(k)
        rows.append(dict(模型=tag, 子集=f"top{k}", 净流R2=round(sub.r2.mean(), 3),
                         方向准确率=round(sub.dir.mean(), 3), 每窗净流=round(sub.vol.mean(), 2)))
out = pd.DataFrame(rows)
print("\n===== 层次模型对照（测试期逐站汇总）=====")
print(out.pivot(index="子集", columns="模型", values=["净流R2", "方向准确率"]).to_string())
out.to_csv(_out("hierarchical_exp.csv"), index=False, encoding="utf-8-sig")
print("\nsaved -> outputs/hierarchical_exp.csv")
