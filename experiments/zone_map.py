# -*- coding: utf-8 -*-
# --- 产物统一输出到项目根 outputs/ ---
from pathlib import Path as _Path
_OUT_DIR = _Path(__file__).resolve().parents[1] / "outputs"
_OUT_DIR.mkdir(exist_ok=True)
def _out(name):
    return str(_OUT_DIR / name)

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from scipy.spatial import ConvexHull
import lstm_dispatch_baseline as B

df = B.load_clean(); df, lab = B.add_cluster(df)
loc = df.groupby("开始站点id")[["lat", "lon"]].median().dropna(); loc["c"] = lab
vol = df.groupby("开始站点id").size()
loc["v"] = vol.reindex(loc.index).fillna(0)

W, H = 680, 520
PL, PR, PT, PB = 26, 190, 30, 44
lat0, lat1 = loc.lat.min() - .01, loc.lat.max() + .01
lon0, lon1 = loc.lon.min() - .02, loc.lon.max() + .02
coslat = np.cos(np.deg2rad(loc.lat.mean()))
# 等比例投影：经度乘 cos(lat) 后按同一 px/deg 缩放
sc = min((H - PT - PB) / (lat1 - lat0), (W - PL - PR) / ((lon1 - lon0) * coslat))
cx = PL + ((W - PL - PR) - (lon1 - lon0) * coslat * sc) / 2
X = lambda lon: cx + (lon - lon0) * coslat * sc
Y = lambda lat: PT + (lat1 - lat) * sc

COL = {0: "#2563eb", 1: "#ea580c", 2: "#94a3b8", 3: "#7c3aed", 4: "#0d9488", 5: "#c026d3"}
NAME = {0: "市中心偏西", 1: "东南", 2: "正南远郊", 3: "西北", 4: "市中心偏东", 5: "正南"}
tot = loc.v.sum()
g = []
g.append(f'<text x="{PL}" y="20" font-size="12.5" fill="#6f7887">站点经纬度散点 · 颜色 = KMeans(k=6) 划分的片区 · 点越大出发量越高</text>')

for c in sorted(loc.c.unique()):
    p = loc[loc.c == c]
    if len(p) >= 3:
        hull = ConvexHull(p[["lon", "lat"]].values)
        pts = " ".join(f"{X(p.lon.values[i]):.1f},{Y(p.lat.values[i]):.1f}" for i in hull.vertices)
        g.append(f'<polygon points="{pts}" fill="{COL[int(c)]}" opacity="0.10" stroke="{COL[int(c)]}" '
                 f'stroke-opacity="0.55" stroke-width="1.2" stroke-dasharray="4 3"/>')

vmax = loc.v.max()
for r in loc.itertuples():
    rad = 2.0 + 3.4 * (r.v / vmax) ** 0.55
    g.append(f'<circle cx="{X(r.lon):.1f}" cy="{Y(r.lat):.1f}" r="{rad:.1f}" fill="{COL[int(r.c)]}" '
             f'fill-opacity="0.55" stroke="{COL[int(r.c)]}" stroke-width="0.8"/>')

order = [0, 4, 3, 1, 2, 5]
for i, c in enumerate(order):
    p = loc[loc.c == c]
    ly = PT + 8 + i * 26
    lx = W - PR + 14
    g.append(f'<rect x="{lx}" y="{ly-8}" width="9" height="9" rx="2" fill="{COL[c]}"/>')
    g.append(f'<text x="{lx+15}" y="{ly}" font-size="11.5" fill="#2a313d" font-weight="700">片区 {c} · {NAME[c]}</text>')
    g.append(f'<text x="{lx+15}" y="{ly+13}" font-size="11" fill="#77808f">{len(p)} 站 · {p.v.sum()/tot*100:.1f}% 流量</text>')

# 标注 0 与 4 的镜像潮汐
p0, p4 = loc[loc.c == 0], loc[loc.c == 4]
mid = ((X(p0.lon.median()) + X(p4.lon.median())) / 2, (Y(p0.lat.median()) + Y(p4.lat.median())) / 2)
g.append(f'<line x1="{X(p0.lon.median()):.1f}" y1="{Y(p0.lat.median()):.1f}" x2="{X(p4.lon.median()):.1f}" '
         f'y2="{Y(p4.lat.median()):.1f}" stroke="#c026d3" stroke-width="1" stroke-dasharray="3 3" opacity="0.5"/>')
g.append(f'<text x="{mid[0]:.1f}" y="{mid[1]-10:.1f}" text-anchor="middle" font-size="11" fill="#c026d3">'
         f'相距约 2 km · 早高峰 −3.1 / +2.8 辆·时</text>')

# 比例尺 2km
km = 2 / 111.0 * sc
bx, by = PL + 8, H - 20
g.append(f'<line x1="{bx}" y1="{by}" x2="{bx+km:.1f}" y2="{by}" stroke="#77808f" stroke-width="2"/>')
g.append(f'<text x="{bx}" y="{by-6}" font-size="11" fill="#77808f">2 km</text>')
g.append(f'<text x="{PL}" y="{H-42}" font-size="11" fill="#9aa1ad">lat {lat0:.2f}–{lat1:.2f}°N · lon {lon0:.2f}–{lon1:.2f}°E（北京城区，非行政边界）</text>')

svg = (f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" xmlns="http://www.w3.org/2000/svg">'
       f'<title>119 个站点按经纬度聚成 6 个片区</title>'
       f'<desc>散点图：颜色代表 KMeans(k=6) 划分的片区，点大小代表出发量。市中心两个片区相距约 2 公里。</desc>'
       f'{"".join(g)}</svg>')
open(_out("zones.svg"), "w", encoding="utf-8").write(svg)
print("ok", len(svg), "站点", len(loc))
print(loc.groupby("c").agg(n=("v", "size"), lat=("lat", "median"), lon=("lon", "median"), v=("v", "sum")).round(4).to_string())
