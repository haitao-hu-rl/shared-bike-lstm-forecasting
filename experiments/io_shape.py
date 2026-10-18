# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import lstm_dispatch_baseline as B

df = B.load_clean(); df, lab = B.add_cluster(df)
clusters = sorted(lab.unique())
full = pd.date_range(df["ts"].min(), df["ts"].max(), freq="h")
dep, arr = B.build_panel(df, clusters, full)
C = 0
dep_s = dep[C].values.astype("float32"); arr_s = arr[C].values.astype("float32")
pred, true, scale, model, norm, n_tr = B.train_one(dep_s, arr_s, full)
mu, sd, ym, ys = norm

print("scale = %.1f  （dep/arr 除以它压到 0-1）" % scale)
print("mu  =", np.round(mu, 3).tolist())
print("sd  =", np.round(sd, 3).tolist())
print("ym  =", np.round(ym, 3).tolist(), "  ← 6 个目标各自的均值（分 horizon 标准化）")
print("ys  =", np.round(ys, 3).tolist())

anchor = pd.Timestamp("2020-09-22 05:00"); ie = int(full.get_loc(anchor))
sl = slice(ie - B.WINDOW + 1, ie + 1)
h, dw = full.hour.values[sl], full.dayofweek.values[sl]
CH = ["dep", "arr", "sin(h)", "cos(h)", "sin(dow)", "cos(dow)", "is_weekend"]
X = np.stack([dep_s[sl] / scale, arr_s[sl] / scale,
              np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24),
              np.sin(2*np.pi*dw/7), np.cos(2*np.pi*dw/7), (dw >= 5).astype(float)], 1)
print("\n单条样本 X 形状：", X.shape, " = (时间步 24) × (通道 7)")
print("各通道统计：")
for i, c in enumerate(CH):
    print(f"   [{i}] {c:<11} min {X[:,i].min():+.3f}  max {X[:,i].max():+.3f}  mean {X[:,i].mean():+.3f}")
print("\n最后 3 个时间步的原始张量（标准化前）：")
for t in range(21, 24):
    print(f"   {full[sl][t]:%m-%d %H:%M}  " + "  ".join(f"{c}={X[t,i]:+.3f}" for i, c in enumerate(CH)))

Xn = (X - mu) / sd
print("\n标准化后：mean≈0、std≈1，形状不变", Xn.shape)
print("\n各层输出形状：")
for l in model.layers:
    try:
        print(f"   {l.__class__.__name__:<12} {l.name:<12} → {l.output_shape}")
    except Exception:
        pass

y = model.predict(Xn[None], verbose=0)[0]
print("\n单条样本输出 y 形状：", y.shape)
print("   标准化空间：", np.round(y, 3).tolist())
raw = y * ys + ym
print("   反标准化后（辆/小时）：", np.round(raw, 2).tolist())
print("   目标时刻：06:00 07:00 08:00 | 06:00 07:00 08:00")

# 负值检查
val = pred
neg_d = (val[:, :3] < 0).mean() * 100
neg_a = (val[:, 3:] < 0).mean() * 100
print("\n验证集预测出现负值的比例：借出 %.1f%%  归还 %.1f%%" % (neg_d, neg_a))
print("   最小值：借出 %.2f  归还 %.2f" % (val[:, :3].min(), val[:, 3:].min()))
