# -*- coding: utf-8 -*-
"""同一模型，比较 1h / 2h / 3h 净流的方向准确率（验证集）"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import lstm_dispatch_baseline as B

df = B.load_clean(); df, lab = B.add_cluster(df)
clusters = sorted(lab.unique())
full = pd.date_range(df["ts"].min(), df["ts"].max(), freq="h")
dep, arr = B.build_panel(df, clusters, full)

print("%-6s %8s | %22s | %22s | %22s" % ("片区", "小时均量",
      "1h 净流 MAE/方向", "2h 累计 MAE/方向", "3h 累计 MAE/方向"))
agg = {k: [] for k in ("m1", "m3", "d1", "d3")}
for c in clusters:
    if dep[c].mean() + arr[c].mean() < 3:   # 跳过近零郊区片区
        continue
    pred, true, scale, model, norm, n_tr = B.train_one(
        dep[c].values.astype("float32"), arr[c].values.astype("float32"), full)
    pnet = pred[:, 3:] - pred[:, :3]        # 每小时净流 (n,3)
    tnet = true[:, 3:] - true[:, :3]

    def dacc(p, t):
        m = np.abs(t) > 0.5
        return float((np.sign(p[m]) == np.sign(t[m])).mean() * 100), int(m.sum())

    d1, n1 = dacc(pnet[:, 0], tnet[:, 0])                       # 只看未来第 1 小时
    d2, n2 = dacc(pnet[:, :2].sum(1), tnet[:, :2].sum(1))       # 前 2 小时累计
    d3, n3 = dacc(pnet.sum(1), tnet.sum(1))                     # 3 小时累计
    m1 = float(np.abs(pnet[:, 0] - tnet[:, 0]).mean())
    m2 = float(np.abs(pnet[:, :2].sum(1) - tnet[:, :2].sum(1)).mean())
    m3 = float(np.abs(pnet.sum(1) - tnet.sum(1)).mean())
    print("%-6d %8.1f | %8.2f / %5.1f%%  | %8.2f / %5.1f%%  | %8.2f / %5.1f%%  (n=%d)"
          % (c, dep[c].mean() + arr[c].mean(), m1, d1, m2, d2, m3, d3, n3))
    for k, v in zip(("m1", "m3", "d1", "d3"), (m1, m3, d1, d3)):
        agg[k].append(v)

print("\n活跃片区平均：1h 方向 %.1f%%  →  3h 累计方向 %.1f%%  （提升 %+.1f 个百分点）"
      % (np.mean(agg["d1"]), np.mean(agg["d3"]), np.mean(agg["d3"]) - np.mean(agg["d1"])))

# 单小时逐 step 的方向（说明不是"第 2、3 小时更准"，是"累计"在起作用）
c = 0
pred, true, scale, model, norm, n_tr = B.train_one(
    dep[c].values.astype("float32"), arr[c].values.astype("float32"), full)
pnet = pred[:, 3:] - pred[:, :3]
tnet = true[:, 3:] - true[:, :3]
print("\n片区 0 逐小时 step（未累计）方向准确率：")
for k in range(3):
    m = np.abs(tnet[:, k]) > 0.5
    print("  第 %d 小时: %.1f%% (n=%d)" % (k + 1,
          (np.sign(pnet[m, k]) == np.sign(tnet[m, k])).mean() * 100, m.sum()))
