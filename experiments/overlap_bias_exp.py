# -*- coding: utf-8 -*-
"""
非重叠 3h 口径评估。

背景：此前报告的「3h 累计净流」用的是滑动窗口（每个目标小时取一次未来 3 小时求和），
相邻两个窗口共享 2 个小时的数据。persistence（复制上一个窗口）因此白拿 2/3 的答案，
R² 被系统性抬高。

本脚本对比两种口径：
  S 滑动口径：窗口逐小时前进（与 metrics_full 一致）
  N 非重叠口径：窗口每 3 小时前进一次（预测 T..T+2 后跳过 3 小时）
同时给出小时级结果作为参照。
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


def r2(p, t):
    ss = ((t - t.mean()) ** 2).sum()
    return float(1 - ((p - t) ** 2).sum() / ss) if ss > 0 else float("nan")


def mae(p, t):  return float(np.abs(p - t).mean())
def rmse(p, t): return float(np.sqrt(((p - t) ** 2).mean()))
def diracc(p, t): return float((np.sign(p) == np.sign(t)).mean())


def block_sum(series, starts, k=3):
    return np.stack([series[starts + i] for i in range(k)], 1).sum(1)


rows = []
df = B.load_clean(); df, lab = B.add_cluster(df)
clusters = sorted(lab.unique())
full = pd.date_range(df["ts"].min(), df["ts"].max(), freq="h")
dep_panel, arr_panel = B.build_panel(df, clusters, full)

for c in [0, 4]:
    dep = dep_panel[c].values.astype("float32")
    arr = arr_panel[c].values.astype("float32")
    tf.random.set_seed(42 + c); np.random.seed(42 + c)
    pred, true, scale, model, norm, n_tr = B.train_one(dep, arr, full)

    n = len(pred)
    va = (len(dep) - W - H + 1) - n
    t_first = va + W
    T = np.arange(t_first, t_first + n)          # 每样本第 1 个目标小时
    net = arr - dep                              # 正 = 净流入

    # ---- 小时间口径 ----
    idx_h = np.arange(n)
    p_h = pred[idx_h, H] - pred[idx_h, 0]        # 第 1 步净流预测
    t_h = net[T[idx_h]]
    rows.append(dict(片区=c, 口径="小时级", 方案="LSTM", MAE=mae(p_h, t_h), RMSE=rmse(p_h, t_h),
                     R2=r2(p_h, t_h), 方向=diracc(p_h, t_h), sigma=t_h.std()))
    p_p = net[T[idx_h] - 1]
    rows.append(dict(片区=c, 口径="小时级", 方案="persistence", MAE=mae(p_p, t_h), RMSE=rmse(p_p, t_h),
                     R2=r2(p_p, t_h), 方向=diracc(p_p, t_h), sigma=t_h.std()))

    # ---- S 滑动 3h 口径（每隔 1 小时一个窗口）----
    idx_s = np.arange(n)
    cp = pred[idx_s, H:].sum(1) - pred[idx_s, :H].sum(1)
    ct = block_sum(net, T[idx_s])
    pp = block_sum(net, T[idx_s] - 3)            # 上一个窗口（重叠 2h）
    rows.append(dict(片区=c, 口径="3h滑动", 方案="LSTM", MAE=mae(cp, ct), RMSE=rmse(cp, ct),
                     R2=r2(cp, ct), 方向=diracc(cp, ct), sigma=ct.std()))
    rows.append(dict(片区=c, 口径="3h滑动", 方案="persistence(重叠2h)", MAE=mae(pp, ct), RMSE=rmse(pp, ct),
                     R2=r2(pp, ct), 方向=diracc(pp, ct), sigma=ct.std()))
    # 旧口径复现：T-1..T+1，其中 T、T+1 是未来，等于偷看答案
    pf = block_sum(net, T[idx_s] - 1)
    rows.append(dict(片区=c, 口径="3h滑动", 方案="persistence(含未来·旧口径)", MAE=mae(pf, ct), RMSE=rmse(pf, ct),
                     R2=r2(pf, ct), 方向=diracc(pf, ct), sigma=ct.std()))

    # ---- N 非重叠 3h 口径：窗口每 3 小时前进一次 ----
    idx_n = np.arange(0, n, 3)
    cp = pred[idx_n, H:].sum(1) - pred[idx_n, :H].sum(1)
    ct = block_sum(net, T[idx_n])
    pp = block_sum(net, T[idx_n] - 3)            # 上一个不重叠窗口
    rows.append(dict(片区=c, 口径="3h非重叠", 方案="LSTM", MAE=mae(cp, ct), RMSE=rmse(cp, ct),
                     R2=r2(cp, ct), 方向=diracc(cp, ct), sigma=ct.std()))
    rows.append(dict(片区=c, 口径="3h非重叠", 方案="persistence(不重叠)", MAE=mae(pp, ct), RMSE=rmse(pp, ct),
                     R2=r2(pp, ct), 方向=diracc(pp, ct), sigma=ct.std()))
    print(f"片区 {c} 完成（验证 {n} 小时，非重叠 {len(idx_n)} 个窗口）", flush=True)

res = pd.DataFrame(rows)
res["MAE"] = res["MAE"].round(2); res["RMSE"] = res["RMSE"].round(2)
res["R2"] = res["R2"].round(3); res["方向"] = res["方向"].round(3); res["sigma"] = res["sigma"].round(2)
print("\n===== 逐片区明细 =====")
print(res.to_string(index=False))

print("\n===== 片区 0/4 平均 =====")
avg = res.groupby(["口径", "方案"], sort=False)[["MAE", "RMSE", "R2", "方向", "sigma"]].mean().round(3)
print(avg.to_string())

res.to_csv(_out("overlap_bias_exp.csv"), index=False, encoding="utf-8-sig")
print("\nsaved -> outputs/overlap_bias_exp.csv")
