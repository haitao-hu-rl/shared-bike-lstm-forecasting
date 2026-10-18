# -*- coding: utf-8 -*-
"""
最优模型的完整指标表（最终汇报口径）。

口径定义
  · 核心片区 0+4 池化（两个片区的验证样本合并计算）
  · 时间外推切分：前 70% 训练 / 15% 验证 / 后 15% 测试（不打乱）
  · 「合并」= 同一目标小时由 3 个不同锚点各预测一次，取平均
  · 净流 net = arr − dep（正 = 净流入，负 = 净流出）
  · 方向准确率只统计 |真实值| > 0.5 辆的样本

模型行
  LSTM(合并) / LSTM(单次,提前1h) / persistence / seasonal-naive / 训练均值(常数)
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
import lstm_dispatch_baseline as B

W, H = B.WINDOW, B.HORIZON


def mae(p, t):  return float(np.abs(p - t).mean())
def rmse(p, t): return float(np.sqrt(((p - t) ** 2).mean()))
def r2(p, t):
    st = ((t - t.mean()) ** 2).sum()
    return float(1 - ((p - t) ** 2).sum() / st) if st > 0 else float("nan")
def diracc(p, t):
    m = np.abs(t) > 0.5
    return float((np.sign(p[m]) == np.sign(t[m])).mean()) if m.any() else float("nan")


rows = []
df = B.load_clean(); df, lab = B.add_cluster(df)
full = pd.date_range(df["ts"].min(), df["ts"].max(), freq="h")
dep_p, arr_p = B.build_panel(df, sorted(lab.unique()), full)

P = {k: [] for k in ("net_m", "net_l", "net_p", "net_s", "net_mu", "net_t",
                     "cum_m", "cum_p", "cum_s", "cum_mu", "cum_t",
                     "dep_m", "dep_p", "dep_s", "dep_mu", "dep_t",
                     "arr_m", "arr_p", "arr_s", "arr_mu", "arr_t",
                     "net_hour", "cum_hour")}

for c in (0, 4):
    dep = dep_p[c].values.astype("float32")
    arr = arr_p[c].values.astype("float32")
    net = arr - dep
    tf.random.set_seed(42 + c); np.random.seed(42 + c)
    pred, true, scale, model, norm, n_tr = B.train_one(dep, arr, full)

    n = len(pred)
    t_first = (len(dep) - W - H + 1) - n + W
    T = np.arange(t_first, t_first + n)
    pnet = pred[:, H:] - pred[:, :H]

    def merge(cols):
        st = []
        for k in range(H):
            v = np.full(n, np.nan)
            j = np.arange(n) - k
            ok = j >= 0
            v[ok] = cols[j[ok], k]
            st.append(v)
        return np.stack(st)

    per_net = merge(pnet)
    per_dep = merge(pred[:, :H])
    per_arr = merge(pred[:, H:])
    merged_net = np.nanmean(per_net, 0)
    merged_dep = np.nanmean(per_dep, 0)
    merged_arr = np.nanmean(per_arr, 0)

    # 训练段（用于均值基线）：训练样本覆盖到第 n_tr+W-1 个小时
    tr_end = n_tr + W
    mu_net = float(net[:tr_end].mean())
    mu_dep = float(dep[:tr_end].mean())
    mu_arr = float(arr[:tr_end].mean())

    # 小时口径：j 从 2 起（保证 3 次平均都齐）
    jh = np.arange(2, n)
    P["net_m"].append(merged_net[jh]);            P["net_t"].append(net[T[jh]])
    P["net_l"].append(per_net[0][jh]);            P["net_p"].append(net[T[jh] - 1])
    P["net_s"].append(net[T[jh] - 24]);           P["net_mu"].append(np.full(len(jh), mu_net))
    P["dep_m"].append(merged_dep[jh]);            P["dep_t"].append(dep[T[jh]])
    P["dep_p"].append(dep[T[jh] - 1]);            P["dep_s"].append(dep[T[jh] - 24])
    P["dep_mu"].append(np.full(len(jh), mu_dep))
    P["arr_m"].append(merged_arr[jh]);            P["arr_t"].append(arr[T[jh]])
    P["arr_p"].append(arr[T[jh] - 1]);            P["arr_s"].append(arr[T[jh] - 24])
    P["arr_mu"].append(np.full(len(jh), mu_arr))
    P["net_hour"].append(full[T[jh]].hour.values)

    # 3h 累计口径：j 从 2 到 n-3，用逐小时合并序列求和
    jc = np.arange(2, n - 2)
    cum_m = np.array([merged_net[j:j + 3].sum() for j in jc])
    cum_t = np.array([net[T[j]:T[j] + 3].sum() for j in jc])
    cum_p = np.array([net[T[j] - 3:T[j]].sum() for j in jc])
    cum_s = np.array([net[T[j] - 24:T[j] - 21].sum() for j in jc])
    cum_mu_tr = float(np.mean([net[j + W:j + W + 3].sum() for j in range(0, n_tr - 3, 3)]))
    P["cum_m"].append(cum_m); P["cum_t"].append(cum_t)
    P["cum_p"].append(cum_p); P["cum_s"].append(cum_s)
    P["cum_mu"].append(np.full(len(jc), cum_mu_tr))
    P["cum_hour"].append(full[T[jc]].hour.values)
    print(f"片区 {c} 完成：小时 n={len(jh)}，3h n={len(jc)}，训练均值 net={mu_net:.2f}", flush=True)

g = lambda k: np.concatenate(P[k])


def add(口径, 模型, pk, tk, direction=True, hk=None):
    p, t = g(pk), g(tk)
    d = dd = None
    if direction:
        m = np.abs(t) > 0.5
        d = round(float((np.sign(p[m]) == np.sign(t[m])).mean()), 3)
        if hk is not None:
            h = g(hk)
            m2 = m & (h >= 6) & (h <= 19)
            dd = round(float((np.sign(p[m2]) == np.sign(t[m2])).mean()), 3) if m2.any() else None
    rows.append(dict(口径=口径, 模型=模型, MAE=round(mae(p, t), 2), RMSE=round(rmse(p, t), 2),
                     R2=round(r2(p, t), 3), 方向准确率=d, 白天方向=dd, n=len(t)))


for nm, pk in [("LSTM（三次重叠平均）", "net_m"), ("LSTM（单次·提前1h）", "net_l"),
               ("persistence（复制上一小时）", "net_p"), ("seasonal-naive（复制昨天同时刻）", "net_s"),
               ("训练均值（常数基线）", "net_mu")]:
    add("逐小时净流", nm, pk, "net_t", hk="net_hour")

for nm, pk in [("LSTM（三次重叠平均后累计）", "cum_m"), ("persistence（上一 3h 窗口）", "cum_p"),
               ("seasonal-naive（昨天同一 3h 窗口）", "cum_s"), ("训练均值（常数基线）", "cum_mu")]:
    add("3h 累计净流", nm, pk, "cum_t", hk="cum_hour")

for nm, dk, ak in [("LSTM（三次重叠平均）", "dep_m", "arr_m"), ("persistence", "dep_p", "arr_p"),
                   ("seasonal-naive", "dep_s", "arr_s"), ("训练均值（常数基线）", "dep_mu", "arr_mu")]:
    add("借出 dep", nm, dk, "dep_t", direction=False)
    add("归还 arr", nm, ak, "arr_t", direction=False)

out = pd.DataFrame(rows)
pd.set_option("display.width", 200)
for q in ("逐小时净流", "3h 累计净流", "借出 dep", "归还 arr"):
    print(f"\n===== {q} =====")
    print(out[out["口径"] == q].drop(columns=["口径"]).to_string(index=False))

out.to_csv(_out("final_metrics_table.csv"),
           index=False, encoding="utf-8-sig")
print("\nsaved -> outputs/final_metrics_table.csv")
