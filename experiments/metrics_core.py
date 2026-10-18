# -*- coding: utf-8 -*-
"""核心片区 0+4（市中心，占 72% 流量）验证集池化指标：MAE / RMSE / R²。"""
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


df = B.load_clean(); df, lab = B.add_cluster(df)
full = pd.date_range(df["ts"].min(), df["ts"].max(), freq="h")
dep_p, arr_p = B.build_panel(df, sorted(lab.unique()), full)

pool = {k: [] for k in ("net_m", "net_l", "net_p", "net_s", "net_t",
                        "dep_m", "dep_t", "arr_m", "arr_t",
                        "cum_m", "cum_t", "cum_p", "cum_s")}

for c in (0, 4):
    dep = dep_p[c].values.astype("float32")
    arr = arr_p[c].values.astype("float32")
    net = arr - dep
    tf.random.set_seed(42 + c); np.random.seed(42 + c)
    pred, true, scale, *_ = B.train_one(dep, arr, full)

    n = len(pred)
    t_first = (len(dep) - W - H + 1) - n + W      # 第一个目标小时的绝对索引
    T = np.arange(t_first, t_first + n)

    pnet = pred[:, H:] - pred[:, :H]
    per_k = np.stack([np.where(T - t_first - k >= 0,
                               pnet[np.clip(T - t_first - k, 0, n - 1), k], np.nan)
                      for k in range(H)])
    merged = np.nanmean(per_k, 0)
    v = ~np.isnan(merged)

    dep_m = np.nanmean(np.stack([np.where(T - t_first - k >= 0,
                                          pred[np.clip(T - t_first - k, 0, n - 1), k], np.nan)
                                 for k in range(H)]), 0)
    arr_m = np.nanmean(np.stack([np.where(T - t_first - k >= 0,
                                          pred[np.clip(T - t_first - k, 0, n - 1), H + k], np.nan)
                                 for k in range(H)]), 0)

    pool["net_m"].append(merged[v]);  pool["net_l"].append(per_k[0][v])
    pool["net_t"].append(net[T][v])
    pool["net_p"].append(net[T - 1][v]); pool["net_s"].append(net[T - 24][v])
    pool["dep_m"].append(dep_m[v]);   pool["dep_t"].append(dep[T][v])
    pool["arr_m"].append(arr_m[v]);   pool["arr_t"].append(arr[T][v])
    pool["cum_m"].append(pnet.sum(1))
    pool["cum_t"].append(np.stack([net[T + k] for k in range(H)], 1).sum(1))
    pool["cum_p"].append(np.stack([net[T - H + k] for k in range(H)], 1).sum(1))
    pool["cum_s"].append(np.stack([net[T - 24 + k] for k in range(H)], 1).sum(1))

g = lambda k: np.concatenate(pool[k])
specs = [
    ("逐小时净流 · 合并(3次平均)", "net_m", "net_t"),
    ("逐小时净流 · 单次(提前1h)", "net_l", "net_t"),
    ("逐小时净流 · persistence", "net_p", "net_t"),
    ("逐小时净流 · seasonal-naive", "net_s", "net_t"),
    ("借出 dep · 合并", "dep_m", "dep_t"),
    ("归还 arr · 合并", "arr_m", "arr_t"),
    ("3h累计净流 · LSTM", "cum_m", "cum_t"),
    ("3h累计净流 · persistence", "cum_p", "cum_t"),
    ("3h累计净流 · seasonal-naive", "cum_s", "cum_t"),
]
out = []
for name, pk, tk in specs:
    p, t = g(pk), g(tk)
    out.append(dict(指标=name, MAE=round(mae(p, t), 3),
                    RMSE=round(rmse(p, t), 3), R2=round(r2(p, t), 3)))
out = pd.DataFrame(out)
print("核心片区 0+4 池化（验证集逐小时，n = %d）" % len(g("net_m")))
print(out.to_string(index=False))
out.to_csv(_out("metrics_core.csv"), index=False, encoding="utf-8-sig")
print("saved -> outputs/metrics_core.csv")
