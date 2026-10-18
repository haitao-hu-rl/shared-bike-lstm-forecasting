# -*- coding: utf-8 -*-
"""
切分方式对照实验：按时间顺序切分（现行） vs 打乱后随机切分（用户建议）。
同一 one-hot 6 维共享模型、同数据、同种子，只改 train/val/test 的划分方式。
目的：量化“随机打乱切分”对测试指标的影响（时间序列泄漏问题）。
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
N = len(Gs)

# ---------- 两种切分 ----------
# A. 按时间：每片区前 70% 训练 / 15% 验证 / 后 15% 测试（现行方案）
pos = np.arange(len(Gs)) % n
tr_t, va_t = int(n * .70), int(n * .85)
idx_tr_A = np.concatenate([np.where((Gs == k) & (pos < tr_t))[0] for k in range(K)])
idx_va_A = np.concatenate([np.where((Gs == k) & (tr_t <= pos) & (pos < va_t))[0] for k in range(K)])
idx_te_A = np.concatenate([np.where((Gs == k) & (pos >= va_t))[0] for k in range(K)])

# B. 打乱：全体样本随机洗牌后 70/15/15
rng = np.random.default_rng(42)
perm = rng.permutation(N)
n_tr, n_va = int(N * .70), int(N * .85)
idx_tr_B, idx_va_B, idx_te_B = perm[:n_tr], perm[n_tr:n_va], perm[n_va:]


def train_eval(idx_tr, idx_va, idx_te, seed=42):
    tf.random.set_seed(seed); np.random.seed(seed)
    oh = np.eye(K, dtype="float32")[Gs]
    Xf = np.concatenate([Xs, np.repeat(oh[:, None, :], W, 1)], 2)
    n_in = 7 + K
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
    return pred, true, m.count_params()


rows = []
for tag, (i_tr, i_va, i_te) in {
    "A 按时间顺序(现行)": (idx_tr_A, idx_va_A, idx_te_A),
    "B 打乱随机切分": (idx_tr_B, idx_va_B, idx_te_B),
}.items():
    pred, true, n_par = train_eval(i_tr, i_va, i_te)
    g_te = Gs[i_te]
    # 逐片区：净流(首目标小时) + dep/arr + 3h累计
    for k, c in enumerate(clusters):
        m_ = g_te == k
        pc, tc = pred[m_], true[m_]
        pnet, tnet = pc[:, H:] - pc[:, :H], tc[:, H:] - tc[:, :H]
        rows.append(dict(切分=tag, 片区=c, 样本数=len(pc),
                         净流MAE=round(mae(pnet[:, 0], tnet[:, 0]), 3),
                         净流R2=round(r2(pnet[:, 0], tnet[:, 0]), 3),
                         depR2=round(r2(pc[:, :H].mean(1), tc[:, :H].mean(1)), 3),
                         arrR2=round(r2(pc[:, H:].mean(1), tc[:, H:].mean(1)), 3),
                         cum3MAE=round(mae(pnet.sum(1), tnet.sum(1)), 3),
                         cum3R2=round(r2(pnet.sum(1), tnet.sum(1)), 3)))
    # 汇总：核心片区 0+4 池化 + 方向准确率（首小时净流 & 3h累计）
    core = np.isin(g_te, [0, 4])
    pc, tc = pred[core], true[core]
    pnet, tnet = pc[:, H:] - pc[:, :H], tc[:, H:] - tc[:, :H]
    d1 = float((np.sign(pnet[:, 0]) == np.sign(tnet[:, 0])).mean())
    d3 = float((np.sign(pnet.sum(1)) == np.sign(tnet.sum(1))).mean())
    rows.append(dict(切分=tag, 片区="0+4池化", 样本数=len(pc),
                     净流MAE=round(mae(pnet[:, 0], tnet[:, 0]), 3),
                     净流R2=round(r2(pnet[:, 0], tnet[:, 0]), 3),
                     depR2=round(r2(pc[:, :H].mean(1), tc[:, :H].mean(1)), 3),
                     arrR2=round(r2(pc[:, H:].mean(1), tc[:, H:].mean(1)), 3),
                     cum3MAE=round(mae(pnet.sum(1), tnet.sum(1)), 3),
                     cum3R2=round(r2(pnet.sum(1), tnet.sum(1)), 3),
                     dir1h=d1, dir3h=d3))
    print(f"\n===== 切分 {tag}（{n_par} 参数）=====")
    print(pd.DataFrame([r for r in rows if r["切分"] == tag]).to_string(index=False))

out = pd.DataFrame(rows)
out.to_csv(_out("split_mode_exp.csv"),
           index=False, encoding="utf-8-sig")
print("\nsaved -> outputs/split_mode_exp.csv")
