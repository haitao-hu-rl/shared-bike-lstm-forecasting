# -*- coding: utf-8 -*-
"""
滚动时间切分（expanding-window 交叉验证）：
- 每片区共 n=1995 个样本（按时间排列）。
- 5 折，测试块互不重叠、按时间先后排列，每块 200 个样本，覆盖样本后 50%（约 1000 小时）。
- 每折：train = 测试块之前的全部样本（扣除验证段），val = 紧邻测试块前 150 个样本。
- 模型与 one-hot 共享架构、种子与 split_mode_exp 完全一致，便于横向对比。
- 另附 persistence 基线（用最后观测小时的净流作为下小时净流预测）。
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
N_FOLDS, TEST_BLK, VAL_BLK = 5, 200, 150


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

hour, dow = full.hour.values, full.dayofweek.values
BASE = [np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7), (dow >= 5).astype(float)]

Xs, Ys, Gs, scales, pers_net = [], [], [], {}, []   # pers_net: persistence 首小时净流(原始量纲)
for ci, c in enumerate(clusters):
    dep = dep_p[c].values.astype("float32"); arr = arr_p[c].values.astype("float32")
    s = max(dep.max(), arr.max(), 1.0); scales[c] = s
    for i in range(n):
        sl = slice(i, i + W)
        Xs.append(np.stack([dep[sl] / s, arr[sl] / s] + [b[sl] for b in BASE], 1))
        Ys.append(np.concatenate([dep[i + W:i + W + H], arr[i + W:i + W + H]]) / s)
        Gs.append(ci)
        pers_net.append(dep[i + W - 1] - arr[i + W - 1])
Xs = np.array(Xs, dtype="float32"); Ys = np.array(Ys, dtype="float32"); Gs = np.array(Gs)
pers_net = np.array(pers_net, dtype="float32")
N = len(Gs)
pos = np.arange(N) % n
oh = np.eye(K, dtype="float32")[Gs]
Xf = np.concatenate([Xs, np.repeat(oh[:, None, :], W, 1)], 2)
n_in = 7 + K

# 5 折的测试块起点（样本索引空间，要求 n >= 5*200 + 150 = 1150）
assert n >= N_FOLDS * TEST_BLK + VAL_BLK, f"n={n} 不够 5 折"
starts = [n - (N_FOLDS - f) * TEST_BLK for f in range(N_FOLDS)]

rows = []
for f, t0 in enumerate(starts):
    tr_end = t0 - VAL_BLK
    idx_tr = np.where(pos < tr_end)[0]
    idx_va = np.where((pos >= tr_end) & (pos < t0))[0]
    idx_te = np.where((pos >= t0) & (pos < t0 + TEST_BLK))[0]

    tf.random.set_seed(42); np.random.seed(42)
    flat = Xf.reshape(-1, n_in)
    mu, sd = flat[idx_tr].mean((0, 1)), flat[idx_tr].std((0, 1)) + 1e-6
    Xn = (Xf - mu) / sd

    x = layers.Input((W, n_in))
    h = layers.LSTM(48, return_sequences=True)(x)
    h = layers.LSTM(32)(h)
    h = layers.Dropout(.2)(h)
    h = layers.Dense(32, activation="relu")(h)
    out = layers.Dense(2 * H)(h)
    m = models.Model(x, out)
    ym, ys = Ys[idx_tr].mean(0), Ys[idx_tr].std(0) + 1e-6
    m.compile("adam", loss=tf.keras.losses.Huber(delta=1.0))
    m.fit(Xn[idx_tr], (Ys[idx_tr] - ym) / ys,
          validation_data=(Xn[idx_va], (Ys[idx_va] - ym) / ys),
          epochs=80, batch_size=64, verbose=0,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])

    pred = m.predict(Xn[idx_te], verbose=0) * ys + ym
    scale_te = np.array([scales[clusters[k]] for k in Gs[idx_te]])[:, None]
    pred = pred * scale_te
    true = Ys[idx_te] * scale_te
    g_te, p_te = Gs[idx_te], pos[idx_te]

    t_start_h = full[0] + pd.Timedelta(hours=int(t0 + W))   # 折的测试起始小时
    for k, c in enumerate(clusters):
        m_ = g_te == k
        pc, tc = pred[m_], true[m_]
        pnet, tnet = pc[:, H:] - pc[:, :H], tc[:, H:] - tc[:, :H]
        rows.append(dict(折=f + 1, 测试起始小时=str(t_start_h), 片区=c, 样本数=len(pc),
                         净流MAE=round(mae(pnet[:, 0], tnet[:, 0]), 3),
                         净流R2=round(r2(pnet[:, 0], tnet[:, 0]), 3),
                         depR2=round(r2(pc[:, :H].mean(1), tc[:, :H].mean(1)), 3),
                         arrR2=round(r2(pc[:, H:].mean(1), tc[:, H:].mean(1)), 3),
                         cum3R2=round(r2(pnet.sum(1), tnet.sum(1)), 3)))
    # 汇总：0+4 池化 + 方向准确率 + persistence 基线
    core = np.isin(g_te, [0, 4])
    pc, tc = pred[core], true[core]
    pnet, tnet = pc[:, H:] - pc[:, :H], tc[:, H:] - tc[:, :H]
    pers = pers_net[idx_te][core][:, 0] if pers_net[idx_te][core].ndim > 1 else pers_net[idx_te][core]
    d1 = float((np.sign(pnet[:, 0]) == np.sign(tnet[:, 0])).mean())
    d3 = float((np.sign(pnet.sum(1)) == np.sign(tnet.sum(1))).mean())
    rows.append(dict(折=f + 1, 测试起始小时=str(t_start_h), 片区="0+4池化", 样本数=len(pc),
                     净流MAE=round(mae(pnet[:, 0], tnet[:, 0]), 3),
                     净流R2=round(r2(pnet[:, 0], tnet[:, 0]), 3),
                     depR2=round(r2(pc[:, :H].mean(1), tc[:, :H].mean(1)), 3),
                     arrR2=round(r2(pc[:, H:].mean(1), tc[:, H:].mean(1)), 3),
                     cum3R2=round(r2(pnet.sum(1), tnet.sum(1)), 3),
                     dir1h=d1, dir3h=d3,
                     persMAE=round(mae(pers, tnet[:, 0]), 3),
                     persR2=round(r2(pers, tnet[:, 0]), 3)))
    print(f"\n===== 折 {f+1}（测试起始 {t_start_h}，train={len(idx_tr)} val={len(idx_va)} test={len(idx_te)}）=====")
    print(pd.DataFrame([r for r in rows if r["折"] == f + 1]).to_string(index=False))

out = pd.DataFrame(rows)
out.to_csv(_out("rolling_split_exp.csv"),
           index=False, encoding="utf-8-sig")

# ---- 5 折汇总（0+4 池化）----
agg = out[out["片区"] == "0+4池化"]
summary = agg[["净流MAE", "净流R2", "depR2", "arrR2", "cum3R2", "dir1h", "dir3h", "persMAE", "persR2"]] \
    .agg(["mean", "std"]).round(3)
print("\n===== 5 折汇总（0+4 池化，mean±std）=====")
print(summary.to_string())

# 小片区漂移观察：片区1 各折 depR2
z1 = out[out["片区"] == 1][["折", "测试起始小时", "depR2", "净流R2"]]
print("\n===== 片区1 各折（漂移观察）=====")
print(z1.to_string(index=False))
print("\nsaved -> outputs/rolling_split_exp.csv")
