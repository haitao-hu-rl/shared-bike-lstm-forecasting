# -*- coding: utf-8 -*-
"""
逐小时锚点 + 三次重叠预测取平均 的正式评估实验。

指标：MAE、RMSE、R²（验证集，时间外推，不允许 shuffle）。
对比对象：
  A. 合并预测（本方案）：同一目标小时的 3 次重叠预测取平均
  B. 单次预测（只取提前 1h，即无重叠合并的对照组）
  C. persistence：净流(T-1)
  D. seasonal-naive：净流(T-24)
评估口径：
  1) 逐小时净流 net = arr - dep（每目标小时）
  2) 借出 dep / 归还 arr（每目标小时）
  3) 3h 累计净流（每个锚点样本的 3 步求和，与此前报告口径一致）
固定随机种子，保证可复现。
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


def mae(p, t):   return float(np.abs(p - t).mean())
def rmse(p, t):  return float(np.sqrt(((p - t) ** 2).mean()))
def r2(p, t):
    ss_res = ((p - t) ** 2).sum()
    ss_tot = ((t - t.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def row(name, p, t):
    return dict(方案=name, MAE=round(mae(p, t), 3), RMSE=round(rmse(p, t), 3),
                R2=round(r2(p, t), 3), n=len(t))


results = {}   # cluster -> {"hourly": DataFrame, "cum3": DataFrame, "dep_arr": DataFrame}

df = B.load_clean(); df, lab = B.add_cluster(df)
clusters = sorted(lab.unique())
full = pd.date_range(df["ts"].min(), df["ts"].max(), freq="h")
dep_panel, arr_panel = B.build_panel(df, clusters, full)

for c in clusters:
    dep = dep_panel[c].values.astype("float32")
    arr = arr_panel[c].values.astype("float32")
    if dep.mean() + arr.mean() < 1.0:      # 近零片区跳过（无信息量）
        continue
    tf.random.set_seed(42 + c); np.random.seed(42 + c)
    pred, true, scale, model, norm, n_tr = B.train_one(dep, arr, full)

    n_samples = len(pred)                  # 验证集样本数
    va = (len(dep) - W - H + 1) - n_samples  # 第一条验证样本在样本空间的下标
    t_first = va + W                       # 第一个目标小时的绝对索引
    T = np.arange(t_first, t_first + n_samples)   # 每条样本第 1 步的目标小时

    pnet_s = pred[:, H:] - pred[:, :H]     # 每样本逐步净流预测 (n, 3)
    net_true_full = arr - dep              # 全序列真实净流

    # ---- 每个目标小时收集 3 次重叠预测（提前 1/2/3 小时）----
    per_k = []
    for k in range(H):
        j = T - t_first - k                # 对应验证样本下标
        v = np.full(n_samples, np.nan)
        ok = (j >= 0) & (j < n_samples)
        v[ok] = pnet_s[j[ok], k]
        per_k.append(v)
    per_k = np.stack(per_k)                # (3, n)
    merged = np.nanmean(per_k, axis=0)
    latest = per_k[0]                      # 只取提前 1h 的对照组

    true_net_h = net_true_full[T]
    valid = ~np.isnan(merged)

    rows_h = [row("A 合并(3次平均)", merged[valid], true_net_h[valid]),
              row("B 单次(提前1h)", latest[valid], true_net_h[valid]),
              row("C persistence", net_true_full[T - 1], true_net_h),
              row("D seasonal-naive", net_true_full[T - 24], true_net_h)]
    results.setdefault(c, {})["hourly"] = pd.DataFrame(rows_h)

    # ---- 借出 / 归还 逐小时（同样合并）----
    dep_m = np.nanmean(np.stack(
        [np.where((T - t_first - k >= 0) & (T - t_first - k < n_samples),
                  pred[np.clip(T - t_first - k, 0, n_samples - 1), k], np.nan) for k in range(H)]), 0)
    arr_m = np.nanmean(np.stack(
        [np.where((T - t_first - k >= 0) & (T - t_first - k < n_samples),
                  pred[np.clip(T - t_first - k, 0, n_samples - 1), H + k], np.nan) for k in range(H)]), 0)
    results[c]["dep_arr"] = pd.DataFrame([
        row("dep 借出(合并)", dep_m[valid], dep[T][valid]),
        row("arr 归还(合并)", arr_m[valid], arr[T][valid])])

    # ---- 3h 累计净流（锚点口径）----
    cum_pred = pnet_s.sum(1)
    cum_true = np.stack([net_true_full[T + k] for k in range(H)], 1).sum(1)
    results[c]["cum3"] = pd.DataFrame([
        row("3h累计净流(LSTM)", cum_pred, cum_true),
        row("3h累计 persistence", np.stack([net_true_full[T - H + k] for k in range(H)], 1).sum(1), cum_true),
        row("3h累计 seasonal-naive", np.stack([net_true_full[T - 24 + k] for k in range(H)], 1).sum(1), cum_true)])

    print(f"\n===== 片区 {c}（小时均量 {dep.mean() + arr.mean():.1f} 单，验证集 {n_samples} 小时）=====")
    for tag in ("hourly", "dep_arr", "cum3"):
        print(tag, "：")
        print(results[c][tag].to_string(index=False))

# ---- 汇总：活跃片区的平均指标 ----
print("\n\n########## 活跃片区汇总（各片区指标取平均）##########")
active = [c for c in results if dep_panel[c].mean() + arr_panel[c].mean() >= 3]
for tag, label in (("hourly", "逐小时净流"), ("cum3", "3h 累计净流")):
    frames = [results[c][tag].set_index("方案") for c in active]
    avg = pd.concat(frames).groupby(level=0).mean().round(3)
    print(f"\n— {label}（片区 {active} 平均）—")
    print(avg.to_string())

pd.to_pickle(results, _out("metrics_full.pkl"))
print("\nsaved -> outputs/metrics_full.pkl")
