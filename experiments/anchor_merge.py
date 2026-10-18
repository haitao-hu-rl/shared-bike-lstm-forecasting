# -*- coding: utf-8 -*-
"""逐小时锚点下，同一目标小时会被 3 个锚点各预测一次（提前 1/2/3 小时）。
本实验在验证集上比较合并策略：取最新一次 or 平均，看方向准确率与 MAE。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import lstm_dispatch_baseline as B

df = B.load_clean(); df, lab = B.add_cluster(df)
clusters = sorted(lab.unique())
full = pd.date_range(df["ts"].min(), df["ts"].max(), freq="h")
dep, arr = B.build_panel(df, clusters, full)

STRAT = ["提前1h", "提前2h", "提前3h", "平均(1+2)", "平均(1+2+3)"]
res = {s: {"d": [], "m": []} for s in STRAT}
for c in clusters:
    if dep[c].mean() + arr[c].mean() < 3:
        continue
    pred, true, scale, *_ = B.train_one(dep[c].values.astype("float32"),
                                        arr[c].values.astype("float32"), full)
    pnet = pred[:, 3:] - pred[:, :3]          # (n,3)：样本 j 对 3 个目标小时的净流预测
    tnet = true[:, 3:] - true[:, :3]
    n = len(pnet)

    # 对齐关系：样本 j 的第 k 步预测的是「绝对目标小时 t0+j+k」；
    # 反过来，目标小时 t0+j 的「提前 k 小时」估计 = pnet[j-k, k]。
    per_k = {}
    for k in range(3):
        v = np.full(n, np.nan)
        idx = np.arange(n) - k
        m = idx >= 0
        v[m] = pnet[idx[m], k]
        per_k[k] = v
    # 目标小时的真净流 = 每条样本第 1 步的真实净流
    true_T = tnet[:, 0]

    avg12 = np.nanmean(np.stack([per_k[0], per_k[1]]), 0)
    avg123 = np.nanmean(np.stack([per_k[0], per_k[1], per_k[2]]), 0)
    for name, p in zip(STRAT, [per_k[0], per_k[1], per_k[2], avg12, avg123]):
        m = ~np.isnan(p) & (np.abs(true_T) > 0.5)
        res[name]["d"].append(float((np.sign(p[m]) == np.sign(true_T[m])).mean() * 100))
        res[name]["m"].append(float(np.abs(p[m] - true_T[m]).mean()))

print("%-12s %14s %12s" % ("合并策略", "方向准确率", "MAE(辆)"))
for s in STRAT:
    print("%-12s %12.1f%%  %10.2f" % (s, np.mean(res[s]["d"]), np.mean(res[s]["m"])))
