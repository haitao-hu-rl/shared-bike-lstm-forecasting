# -*- coding: utf-8 -*-
"""
实验：把「片区」编进输入能否提升预测质量？

三个方案（同数据、同时间切分、同结构）：
  A. 每片区独立模型（现状，结果来自 metrics_full.pkl）
  B. 全片区共享一个模型，输入不含片区信息（7 通道）
  C. 共享模型 + 片区 one-hot（7+6=13 通道，逐时间步恒定）

共享模型的 dep/arr 按各片区自身 scale 归一后再进网络，
输出乘回片区 scale。评估与 metrics_full 相同：验证集逐小时净流
（三次重叠预测取平均）+ 3h 累计净流。
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
def rmse(p, t): return float(np.sqrt(((p - t) ** 2).mean()))
def r2(p, t):
    st = ((t - t.mean()) ** 2).sum()
    return float(1 - ((p - t) ** 2).sum() / st) if st > 0 else float("nan")


df = B.load_clean(); df, lab = B.add_cluster(df)
clusters = sorted(lab.unique())
K = len(clusters)
full = pd.date_range(df["ts"].min(), df["ts"].max(), freq="h")
dep_p, arr_p = B.build_panel(df, clusters, full)
L = len(full)
n = L - W - H + 1
tr, va = int(n * .70), int(n * .85)

hour = full.hour.values; dow = full.dayofweek.values
BASE = [np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
        (dow >= 5).astype(float)]          # 每个都是 (L,)

Xs, Ys, Gs = [], [], []                     # 样本、目标、所属片区
scales = {}
for ci, c in enumerate(clusters):
    dep = dep_p[c].values.astype("float32")
    arr = arr_p[c].values.astype("float32")
    s = max(dep.max(), arr.max(), 1.0)
    scales[c] = s
    for i in range(n):
        sl = slice(i, i + W)
        Xs.append(np.stack([dep[sl] / s, arr[sl] / s] + [b[sl] for b in BASE], 1))
        Ys.append(np.concatenate([dep[i + W:i + W + H], arr[i + W:i + W + H]]) / s)
        Gs.append(ci)
Xs = np.array(Xs, dtype="float32")          # (6*1995, 24, 7)
Ys = np.array(Ys, dtype="float32")          # (6*1995, 6)
Gs = np.array(Gs)
n_per = n

# 时间切分（所有片区共用同一时间边界）
idx_tr = np.concatenate([np.where((Gs == k) & (np.arange(len(Gs)) % n_per < tr))[0] for k in range(K)])
idx_va = np.concatenate([np.where((Gs == k) & (np.arange(len(Gs)) % n_per >= va))[0] for k in range(K)])


def run_shared(use_cluster_id, seed=42):
    tf.random.set_seed(seed); np.random.seed(seed)
    if use_cluster_id:
        oh = np.eye(K, dtype="float32")[Gs]                     # (N, 6)
        Xf = np.concatenate([Xs, np.repeat(oh[:, None, :], W, 1)], 2)
    else:
        Xf = Xs
    mu, sd = Xf[idx_tr].mean((0, 1)), Xf[idx_tr].std((0, 1)) + 1e-6
    Xn = (Xf - mu) / sd
    ym, ys = Ys[idx_tr].mean(0), Ys[idx_tr].std(0) + 1e-6

    inp = layers.Input((W, Xf.shape[2]))
    h = layers.LSTM(48, return_sequences=True)(inp)
    h = layers.LSTM(32)(h)
    h = layers.Dropout(.2)(h)
    h = layers.Dense(32, activation="relu")(h)
    out = layers.Dense(2 * H)(h)
    m = models.Model(inp, out)
    m.compile("adam", loss=tf.keras.losses.Huber(delta=1.0))
    m.fit(Xn[idx_tr], (Ys[idx_tr] - ym) / ys,
          validation_data=(Xn[idx_va], (Ys[idx_va] - ym) / ys),
          epochs=80, batch_size=64, verbose=0,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])

    pred_n = m.predict(Xn[idx_va], verbose=0) * ys + ym         # 归一化空间
    # 还原为各片区原始单位
    g = Gs[idx_va]
    pred = pred_n * np.array([scales[clusters[k]] for k in g])[:, None]
    true = Ys[idx_va] * np.array([scales[clusters[k]] for k in g])[:, None]
    return pred, true, g


def eval_pred(pred, true, g, tag):
    """逐片区：逐小时净流（重叠平均）与 3h 累计。"""
    rows = []
    for k, c in enumerate(clusters):
        m_ = g == k
        pc, tc = pred[m_], true[m_]
        pnet = pc[:, H:] - pc[:, :H]
        tnet = tc[:, H:] - tc[:, :H]
        # 重叠平均（与 metrics_full 相同口径）
        n_s = len(pnet)
        T = np.arange(n_s)
        per_k = np.stack([np.where(T - k_ >= 0, pnet[np.clip(T - k_, 0, n_s - 1), k_], np.nan)
                          for k_ in range(H)])
        merged = np.nanmean(per_k, 0); v = ~np.isnan(merged)
        rows.append(dict(方案=tag, 片区=c,
                         MAE=round(mae(merged[v], tnet[:, 0][v]), 3),
                         RMSE=round(rmse(merged[v], tnet[:, 0][v]), 3),
                         R2=round(r2(merged[v], tnet[:, 0][v]), 3),
                         dep_R2=round(r2(pc[:, :H].mean(1), tc[:, :H].mean(1)), 3),
                         cum3_R2=round(r2(pnet.sum(1), tnet.sum(1)), 3)))
    return rows


all_rows = []
for tag, use_id in (("B 共享·无片区信息", False), ("C 共享·one-hot片区", True)):
    pred, true, g = run_shared(use_id)
    all_rows += eval_pred(pred, true, g, tag)
    pooled = g == g  # 全部
    pnet = pred[:, H:] - pred[:, :H]; tnet = true[:, H:] - true[:, :H]
    all_rows.append(dict(方案=tag, 片区="0+4池化",
                         MAE="-", RMSE="-", R2="-",
                         dep_R2=round(r2(pred[:, :3].mean(1), true[:, :3].mean(1)), 3),
                         cum3_R2="-"))
    print(f"\n===== {tag} =====")
    print(pd.DataFrame([r for r in all_rows if r["方案"] == tag]).to_string(index=False))

# 对照：A 方案（每片区独立，seed 固定）在同口径下的结果
print("\n===== A 每片区独立模型（对照，来自 metrics_full）=====")
res = pd.read_pickle(_out("metrics_full.pkl"))
for c in clusters:
    if c in res and "hourly" in res[c]:
        h = res[c]["hourly"].set_index("方案")
        if "A 合并(3次平均)" in h.index:
            d = res[c]["dep_arr"].set_index("方案")
            print(f"片区 {c}: 净流 MAE {h.loc['A 合并(3次平均)','MAE']:.3f} / R2 {h.loc['A 合并(3次平均)','R2']:+.3f}"
                  f" | dep R2 {d.loc['dep 借出(合并)','R2']:+.3f} | cum3 R2 "
                  f"{res[c]['cum3'].set_index('方案').loc['3h累计净流(LSTM)','R2']:+.3f}")

out = pd.DataFrame(all_rows)
out.to_csv(_out("cluster_id_exp.csv"),
           index=False, encoding="utf-8-sig")
print("\nsaved -> outputs/cluster_id_exp.csv")
