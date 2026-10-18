# -*- coding: utf-8 -*-
"""
一个完整的真实算例：
  24 小时原始流水 → (24,7) 输入矩阵 → 6 个输出 → 站点清单
运行：D:\\miniconda3\\envs\\bike\\python.exe experiments/io_example.py
"""
# --- 产物统一输出到项目根 outputs/ ---
from pathlib import Path as _Path
_OUT_DIR = _Path(__file__).resolve().parents[1] / "outputs"
_OUT_DIR.mkdir(exist_ok=True)
def _out(name):
    return str(_OUT_DIR / name)

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import lstm_dispatch_baseline as B

pd.set_option("display.width", 200)

ANCHOR = pd.Timestamp("2020-09-22 05:00")
C = 0

df = B.load_clean()
df, lab = B.add_cluster(df)
clusters = sorted(lab.unique())
full = pd.date_range(df["ts"].min(), df["ts"].max(), freq="h")
dep, arr = B.build_panel(df, clusters, full)
dep_s = {c: dep[c].values.astype("float32") for c in clusters}
arr_s = {c: arr[c].values.astype("float32") for c in clusters}

pred, true, scale, model, norm, n_tr = B.train_one(dep_s[C], arr_s[C], full)
mu, sd, ym, ys = norm
ie = int(full.get_loc(ANCHOR))

# ---------- 第 0 步：原始流水 ----------
print("=" * 96)
print(f"【第 0 步】原始流水：片区 {C} 在 {ANCHOR:%Y-%m-%d %H:%M} 之前 24 小时，每个小时借出/归还多少辆")
print("=" * 96)
sl = slice(ie - B.WINDOW + 1, ie + 1)
raw_dep = dep_s[C][sl]
raw_arr = arr_s[C][sl]
hh = full[sl]

print(f"{'时刻':<16}{'借出(单)':>10}{'归还(单)':>10}{'净流(辆)':>12}")
for t in range(24):
    print(f"{hh[t]:%m-%d %H:%M}{'':<4}{int(raw_dep[t]):>10}{int(raw_arr[t]):>10}"
          f"{int(raw_arr[t]) - int(raw_dep[t]):>+12}")
print(f"{'24h 合计':<16}{int(raw_dep.sum()):>10}{int(raw_arr.sum()):>10}"
      f"{int(raw_arr.sum() - raw_dep.sum()):>+12}")

# ---------- 第 1 步：构造矩阵 ----------
print("\n" + "=" * 96)
print(f"【第 1 步】变成输入矩阵 X，形状 (24, 7)；缩放系数 scale = {scale:.1f}（取片区历史最大值）")
print("=" * 96)
h = hh.hour.values
dw = hh.dayofweek.values
CH = ["dep", "arr", "sin(h)", "cos(h)", "sin(dow)", "cos(dow)", "is_we"]
X = np.stack([raw_dep / scale, raw_arr / scale,
              np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
              np.sin(2 * np.pi * dw / 7), np.cos(2 * np.pi * dw / 7),
              (dw >= 5).astype(float)], 1)

print(f"{'t':<4}{'时刻':<14}" + "".join(f"{c:>10}" for c in CH))
for t in list(range(0, 3)) + list(range(19, 24)):
    print(f"{t:<4}{hh[t]:%m-%d %H:%M}  " + "".join(f"{X[t, i]:>10.3f}" for i in range(7)))
    if t == 2:
        print(f"{'':<4}{'...（省略 16 行）':<14}")

print(f"\n矩阵形状 {X.shape}  →  时间步 24 × 通道 7")
print(f"再标准化 (X-mu)/sd，mu/sd 只用训练段算：")
print("   mu =", np.round(mu, 3).tolist())
print("   sd =", np.round(sd, 3).tolist())

# ---------- 第 2 步：前向 ----------
print("\n" + "=" * 96)
print("【第 2 步】喂进网络，一次前向吐出 6 个数（形状 (6,)，单位：辆/小时）")
print("=" * 96)
y_norm = model.predict(((X - mu) / sd)[None], verbose=0)[0]
y = y_norm * ys + ym
y = np.clip(y, 0, None)
print("   标准化空间 y_norm =", np.round(y_norm, 3).tolist())
print("   反标准化（× ys + ym）=", np.round(y, 2).tolist())
print("\n   y[0:3] = 借出   06点 / 07点 / 08点")
print("   y[3:6] = 归还   06点 / 07点 / 08点")
print(f"\n   净流 = {y[3:].sum():.2f} - {y[:3].sum():.2f} = {y[3:].sum() - y[:3].sum():+.2f} 辆")
print("   当天实际：", end="")
act_d = raw_dep[1:4] if False else dep_s[C][ie + 1:ie + 4]
act_a = arr_s[C][ie + 1:ie + 4]
print(f"借出 {act_d.astype(int).tolist()} 归还 {act_a.astype(int).tolist()} → 净流 "
      f"{int(act_a.sum() - act_d.sum()):+d} 辆")

# ---------- 第 3 步：站点份额 ----------
print("\n" + "=" * 96)
print("【第 3 步】片区 6 个数 × 站点份额 → 每站每小时净流（模型并不知道站点存在）")
print("=" * 96)
df["is_we"] = (df["start"].dt.dayofweek >= 5).astype(int)
df["hour"] = df["start"].dt.hour


def build_share(frame, ckey, sid_col, hcol, alpha=2.0):
    keys = [ckey, "is_we", hcol]
    cnt = frame.groupby(keys + [sid_col]).size().reset_index(name="n")
    tot = frame.groupby(keys).size().reset_index(name="N")
    nun = frame.groupby(keys)[sid_col].nunique().reset_index(name="k")
    s = cnt.merge(tot, on=keys).merge(nun, on=keys)
    s["share"] = (s["n"] + alpha / s["k"]) / (s["N"] + alpha)
    return s.set_index(keys + [sid_col])["share"]


sd_ = build_share(df, "c_start", "开始站点id", "hour")
de = df.assign(sid=df["结束站点"], hour=df["end"].dt.hour)
sa_ = build_share(de, "c_end", "sid", "hour")

rows = []
for k in range(3):
    hr = (ANCHOR.hour + 1 + k) % 24
    try:
        wd = sd_.loc[(C, 0, hr)]
        wa = sa_.loc[(C, 0, hr)]
    except KeyError:
        continue
    if wd is None:
        continue
    for sid in lab[lab == C].index:
        rows.append(dict(station=sid, hour=hr,
                         net=y[3 + k] * wa.get(sid, 0.0) - y[k] * wd.get(sid, 0.0)))
tab = pd.DataFrame(rows)

# 逐小时宽表 + 实际值
wide = tab.pivot(index="station", columns="hour", values="net").fillna(0)
wide["3h"] = wide.sum(1)
print(f"份额之和 借出 {wd.sum():.4f} / 归还 {wa.sum():.4f} （≈1.0000）")
print(f"全部站点净流相加 = {wide['3h'].sum():+.2f}  ←  必须等于片区净流 "
      f"{y[3:].sum() - y[:3].sum():+.2f}")

# 实际站点净流（用于对照）
a0 = ANCHOR + pd.Timedelta(hours=1)
a1 = ANCHOR + pd.Timedelta(hours=3)
sub = df[(df["start"] >= a0) & (df["start"] <= a1)]
reald = sub[sub.c_start == C].groupby("开始站点id").size()
suba = df[(df["end"] >= a0) & (df["end"] <= a1)]
reala = suba[suba.c_end == C].groupby("结束站点").size()
wide["实际3h"] = (reala.reindex(wide.index).fillna(0) - reald.reindex(wide.index).fillna(0))

print(f"\n预测最缺车的 8 个站（{a0:%H:%M}–{a1:%H:%M}）：")
cols = [c for c in wide.columns if isinstance(c, int)]
show = wide.sort_values("3h").head(8).copy()
show.columns = [f"{c:02d}点" if isinstance(c, int) else c for c in show.columns]
show["动作"] = np.where(show["3h"] < 0, "补车", "清运")
show["需调辆"] = show["3h"].abs().round(0).astype(int)
print(show.round(2).to_string())

print(f"\n预测最积压的 4 个站：")
show2 = wide.sort_values("3h").tail(4).copy()
show2.columns = [f"{c:02d}点" if isinstance(c, int) else c for c in show2.columns]
show2["动作"] = np.where(show2["3h"] < 0, "补车", "清运")
show2["需调辆"] = show2["3h"].abs().round(0).astype(int)
print(show2.round(2).to_string())

# ---------- 单站完整算式 ----------
print("\n" + "=" * 96)
print("【第 4 步】任取一站的完整算式")
print("=" * 96)
sid = int(show.index[0])
hr = (ANCHOR.hour + 1) % 24
wd = sd_.loc[(C, 0, hr)]
wa = sa_.loc[(C, 0, hr)]
print(f"站 {sid}（片区 {C}）· {hr:02d} 点")
print(f"   借出 = 片区借出 {y[0]:6.2f} × 该站借出份额 {wd.get(sid, 0):.4f} = {y[0] * wd.get(sid, 0):6.2f} 辆")
print(f"   归还 = 片区归还 {y[3]:6.2f} × 该站归还份额 {wa.get(sid, 0):.4f} = {y[3] * wa.get(sid, 0):6.2f} 辆")
n1 = y[3] * wa.get(sid, 0) - y[0] * wd.get(sid, 0)
print(f"   净流 = {y[3] * wa.get(sid, 0):6.2f} − {y[0] * wd.get(sid, 0):6.2f} = {n1:+.2f} 辆 "
      f"→ {'补车' if n1 < 0 else '清运'} {abs(n1):.0f} 辆")

json.dump(dict(anchor=str(ANCHOR), cluster=C, scale=float(scale),
               raw_dep=[int(v) for v in raw_dep], raw_arr=[int(v) for v in raw_arr],
               times=[str(v) for v in hh],
               y=[round(float(v), 2) for v in y],
               actual=[int(act_d.sum()), int(act_a.sum())]),
          open(_out("example.json"), "w", encoding="utf-8"),
          ensure_ascii=False)
print("\n已写入 outputs/example.json")
