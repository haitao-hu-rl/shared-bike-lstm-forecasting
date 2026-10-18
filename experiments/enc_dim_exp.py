# -*- coding: utf-8 -*-
"""
片区编码的“维度-效果”对比：one-hot 6 维 vs embedding 3 维 vs 地理坐标 2 维。
同一共享模型、同数据、同切分、同种子，只换片区编码方式。
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

W, H = B.WINDOW, B.HORIZON


def mae(p, t):  return float(np.abs(p - t).mean())
def r2(p, t):
    st = ((t - t.mean()) ** 2).sum()
    return float(1 - ((p - t) ** 2).sum() / st) if st > 0 else float("nan")


df = B.load_clean(); df, lab = B.add_cluster(df)
clusters = sorted(lab.unique())
K = len(clusters)
full = pd.date_range(df["ts"].min(), df["ts"].max(), freq="h")
dep_p, arr_p = B.build_panel(df, clusters, full)
L = len(full); n = L - W - H + 1
tr, va = int(n * .70), int(n * .85)

# 片区质心坐标（编码用）
cent = df.groupby("c_start")[["lat", "lon"]].median()
cent = (cent - cent.mean()) / cent.std()

hour, dow = full.hour.values, full.dayofweek.values
BASE = [np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7), (dow >= 5).astype(float)]

Xs, Ys, Gs, scales = [], [], [], {}
for ci, c in enumerate(clusters):
    dep = dep_p[c].values.astype("float32"); arr = arr_p[c].values.astype("float32")
    s = max(dep.max(), arr.max(), 1.0); scales[c] = s
    for i in range(n):
        sl = slice(i, i + W)
        Xs.append(np.stack([dep[sl] / s, arr[sl] / s] + [b[sl] for b in BASE], 1))
        Ys.append(np.concatenate([dep[i + W:i + W + H], arr[i + W:i + W + H]]) / s)
        Gs.append(ci)
Xs = np.array(Xs, dtype="float32"); Ys = np.array(Ys, dtype="float32"); Gs = np.array(Gs)
pos = np.arange(len(Gs)) % n
idx_tr = np.concatenate([np.where((Gs == k) & (pos < tr))[0] for k in range(K)])
idx_va = np.concatenate([np.where((Gs == k) & (pos >= va))[0] for k in range(K)])
g_va = Gs[idx_va]
scale_va = np.array([scales[clusters[k]] for k in g_va])[:, None]


def build_and_train(kind, seed=42):
    """kind: oh6 / emb3 / geo2"""
    tf.random.set_seed(seed); np.random.seed(seed)
    ser_in = layers.Input((W, 7))
    if kind == "oh6":
        oh = np.eye(K, dtype="float32")[Gs]                              # (N,K)
        Xf = np.concatenate([Xs, np.repeat(oh[:, None, :], W, 1)], 2)
        n_in = 7 + K
        x = layers.Input((W, n_in))
        inputs = [x]; feed = [Xf]
        flat = Xf.reshape(-1, n_in)
    elif kind == "ord1":
        ordv = (Gs / (K - 1)).astype("float32").reshape(-1, 1)            # (N,1) 0~1
        Xf = np.concatenate([Xs, np.repeat(ordv[:, None, :], W, 1)], 2)
        n_in = 8
        x = layers.Input((W, n_in))
        inputs = [x]; feed = [Xf]
        flat = Xf.reshape(-1, n_in)
    elif kind == "geo2":
        geo = cent.loc[clusters].values.astype("float32")[Gs]            # (N,2)
        Xf = np.concatenate([Xs, np.repeat(geo[:, None, :], W, 1)], 2)
        n_in = 9
        x = layers.Input((W, n_in))
        inputs = [x]; feed = [Xf]
        flat = Xf.reshape(-1, n_in)
    else:  # emb3：双输入，embedding 可训练
        cid_in = layers.Input((1,), dtype="int32")
        emb = layers.Embedding(K, 3)(cid_in)                             # (None,1,3)
        emb = layers.Flatten()(emb)                                      # (None,3)
        emb = layers.RepeatVector(W)(emb)                                # (None,24,3)
        x = layers.Concatenate(axis=-1)([ser_in, emb])
        n_in = 10
        inputs = [ser_in, cid_in]; feed = [Xs, Gs.astype("int32")]
        flat = None

    if flat is not None:
        mu, sd = flat[idx_tr].mean((0, 1)), flat[idx_tr].std((0, 1)) + 1e-6
        feed_n = [(feed[0] - mu) / sd]
    else:
        # emb3：只标准化序列部分
        flat7 = Xs.reshape(-1, 7)
        mu, sd = flat7[idx_tr].mean((0, 1)), flat7[idx_tr].std((0, 1)) + 1e-6
        feed_n = [(Xs - mu) / sd, feed[1]]

    h = layers.LSTM(48, return_sequences=True)(x)
    h = layers.LSTM(32)(h)
    h = layers.Dropout(.2)(h)
    h = layers.Dense(32, activation="relu")(h)
    out = layers.Dense(2 * H)(h)
    m = models.Model(inputs, out)
    ym, ys = Ys[idx_tr].mean(0), Ys[idx_tr].std(0) + 1e-6
    m.compile("adam", loss=tf.keras.losses.Huber(delta=1.0))
    m.fit([f[idx_tr] for f in feed_n], (Ys[idx_tr] - ym) / ys,
          validation_data=([f[idx_va] for f in feed_n], (Ys[idx_va] - ym) / ys),
          epochs=80, batch_size=64, verbose=0,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])

    pred_n = m.predict([f[idx_va] for f in feed_n], verbose=0) * ys + ym
    pred = pred_n * scale_va
    true = Ys[idx_va] * scale_va
    n_params = m.count_params()
    return pred, true, n_params


rows = []
for kind, name in (("oh6", "one-hot 6维"), ("ord1", "序数1维(0~1)"), ("geo2", "质心坐标 2维")):
    pred, true, n_params = build_and_train(kind)
    for k, c in enumerate(clusters):
        m_ = g_va == k
        pc, tc = pred[m_], true[m_]
        pnet, tnet = pc[:, H:] - pc[:, :H], tc[:, H:] - tc[:, :H]
        ns = len(pnet); T = np.arange(ns)
        per_k = np.stack([np.where(T - k_ >= 0, pnet[np.clip(T - k_, 0, ns - 1), k_], np.nan)
                          for k_ in range(H)])
        mg = np.nanmean(per_k, 0); v = ~np.isnan(mg)
        rows.append(dict(编码=name, 片区=c, 参数量=n_params,
                         净流MAE=round(mae(mg[v], tnet[:, 0][v]), 3),
                         净流R2=round(r2(mg[v], tnet[:, 0][v]), 3),
                         depR2=round(r2(pc[:, :H].mean(1), tc[:, :H].mean(1)), 3),
                         cum3R2=round(r2(pnet.sum(1), tnet.sum(1)), 3)))
    print(f"\n===== {name}（{n_params} 参数）=====")
    print(pd.DataFrame([r for r in rows if r["编码"] == name]).to_string(index=False))

out = pd.DataFrame(rows)
out.to_csv(_out("enc_dim_exp.csv"),
           index=False, encoding="utf-8-sig")
print("\nsaved -> outputs/enc_dim_exp.csv")
