# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd, numpy as np
from sklearn.cluster import KMeans
import lstm_dispatch_baseline as B

df = B.load_clean()
df, lab = B.add_cluster(df)
loc = df.groupby("开始站点id")[["lat", "lon"]].median().dropna()
loc["c"] = lab
df["hour"] = df["start"].dt.hour
df["dow"] = df["start"].dt.dayofweek

print("片区数 %d，覆盖 %d 站（原始 120 站，1 站坐标全缺被剔除）" % (lab.nunique(), len(lab)))
print("\n各片区：站点数 / 坐标范围 / 中心 / 日均出发单量")
for c, g in loc.groupby("c"):
    dep = df[df["开始站点id"].isin(g.index)]
    print(f"  片区 {c}: {len(g):>2} 站 | lat {g.lat.min():.4f}-{g.lat.max():.4f}  lon {g.lon.min():.4f}-{g.lon.max():.4f} "
          f"| 中心 ({g.lat.median():.4f},{g.lon.median():.4f}) | 日均 {len(dep)/85:.0f} 单/天 "
          f"({len(dep)/df.shape[0]*100:.1f}%)")

print("\n各片区代表站点（按出发量 top3）")
for c, g in loc.groupby("c"):
    dep = df[df["开始站点id"].isin(g.index)].groupby("开始站点id").size().nlargest(3)
    print(f"  片区 {c}: " + ", ".join(f"{i}({v}单)" for i, v in dep.items()))

print("\n地理方位（按中心经纬度）")
cc = loc.groupby("c")[["lat","lon"]].median()
cc["方位"] = np.where(cc.lat > cc.lat.median(), "北", "南") + np.where(cc.lon > cc.lon.median(), "东", "西")
print(cc.round(4).to_string())

print("\n各片区 工作日 早(7-9)/晚(17-19) 每小时净流 均值（到达-出发）")
w = df[df.dow < 5]; nd = w.date.nunique() if "date" in w else w["start"].dt.date.nunique()
for c in sorted(lab.unique()):
    d1 = w[w.c_start == c].groupby("hour").size().reindex(range(24), fill_value=0).sum() / nd
    a1 = w[w.c_end == c].groupby("hour").size().reindex(range(24), fill_value=0).sum() / nd
    de = w[(w.c_start == c) & (w.hour.between(7, 9))].shape[0] / nd / 3
    ae = w[(w.c_end == c) & (w.hour.between(7, 9))].shape[0] / nd / 3
    dn = w[(w.c_start == c) & (w.hour.between(17, 19))].shape[0] / nd / 3
    an = w[(w.c_end == c) & (w.hour.between(17, 19))].shape[0] / nd / 3
    print(f"  片区 {c}: 早 {ae-de:+5.1f} 辆/时 | 晚 {an-dn:+5.1f} 辆/时")
