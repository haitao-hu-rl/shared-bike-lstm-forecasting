# -*- coding: utf-8 -*-
"""把"片区预测 → 站点份额"这条链路用真实数字走一遍，并导出作图数据。"""
# --- 产物统一输出到项目根 outputs/ ---
from pathlib import Path as _Path
_OUT_DIR = _Path(__file__).resolve().parents[1] / "outputs"
_OUT_DIR.mkdir(exist_ok=True)
def _out(name):
    return str(_OUT_DIR / name)

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import lstm_dispatch_baseline as B

df = B.load_clean(); df, lab = B.add_cluster(df)
clusters = sorted(lab.unique())
full = pd.date_range(df["ts"].min(), df["ts"].max(), freq="h")
dep, arr = B.build_panel(df, clusters, full)

C = 0
dep_s = dep[C].values.astype("float32"); arr_s = arr[C].values.astype("float32")
anchor = pd.Timestamp("2020-09-22 05:00"); ie = int(full.get_loc(anchor))
pred, true, scale, model, norm, n_tr = B.train_one(dep_s, arr_s, full)
mu, sd, ym, ys = norm
sl = slice(ie - B.WINDOW + 1, ie + 1)
h, dw = full.hour.values[sl], full.dayofweek.values[sl]
X = np.stack([dep_s[sl] / scale, arr_s[sl] / scale,
              np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24),
              np.sin(2*np.pi*dw/7), np.cos(2*np.pi*dw/7), (dw >= 5).astype(float)], 1)[None]
y = model.predict((X - mu)/sd, verbose=0)[0]*ys + ym
pH, aH = y[:3], y[3:]

print("【第 1 层】片区 0 · 锚点 09-22 05:00（辆/小时）")
print("   小时      预测借出  预测归还  预测净流 | 实际借出 实际归还 实际净流")
for k in range(3):
    i = ie + k + 1
    print(f"   {full[i].hour:02d}:00     {pH[k]:6.2f}   {aH[k]:6.2f}   {aH[k]-pH[k]:+6.2f} | "
          f"{dep_s[i]:6.0f}   {arr_s[i]:6.0f}   {arr_s[i]-dep_s[i]:+6.0f}")

df["is_we"] = (df["start"].dt.dayofweek >= 5).astype(int)
df["hour"] = df["start"].dt.hour
cnt = df.groupby(["c_start", "is_we", "hour", "开始站点id"]).size().reset_index(name="n")
tot = df.groupby(["c_start", "is_we", "hour"]).size().reset_index(name="N")
nun = df.groupby(["c_start", "is_we", "hour"])["开始站点id"].nunique().reset_index(name="k")
D = cnt.merge(tot, on=["c_start", "is_we", "hour"]).merge(nun, on=["c_start", "is_we", "hour"])
D["w"] = (D.n + 2/D.k) / (D.N + 2)

de = df.assign(sid=df["结束站点"], hour=df["end"].dt.hour)
cnt2 = de.groupby(["c_end", "is_we", "hour", "sid"]).size().reset_index(name="n")
tot2 = de.groupby(["c_end", "is_we", "hour"]).size().reset_index(name="N")
nun2 = de.groupby(["c_end", "is_we", "hour"])["sid"].nunique().reset_index(name="k")
A = cnt2.merge(tot2, on=["c_end", "is_we", "hour"]).merge(nun2, on=["c_end", "is_we", "hour"])
A["w"] = (A.n + 2/A.k) / (A.N + 2)

hr = 7
wd = D[(D.c_start == C) & (D.is_we == 0) & (D.hour == hr)].set_index("开始站点id")["w"]
wa = A[(A.c_end == C) & (A.is_we == 0) & (A.hour == hr)].set_index("sid")["w"]
print(f"\n【第 2 层】份额（工作日 {hr:02d} 点 · 片区 0）：借出合计 {wd.sum():.4f}，归还合计 {wa.sum():.4f}")

tab = pd.DataFrame({"wd": wd, "wa": wa}).fillna(0.0)
tab["dep"] = pH[1] * tab.wd
tab["arr"] = aH[1] * tab.wa
tab["net"] = tab.arr - tab.dep
tab = tab.sort_values("net")
print("   净流最负的 6 站：")
print(tab.head(6).round(3).to_string())
print(f"   29 站净流合计 {tab['net'].sum():+.2f}  vs  片区净流 {aH[1]-pH[1]:+.2f}")

i7 = ie + 2
act = df[(df["end"].dt.floor("h") == full[i7]) & df["结束站点"].isin(tab.index)].groupby("结束站点").size()
actd = df[(df["start"].dt.floor("h") == full[i7]) & df["开始站点id"].isin(tab.index)].groupby("开始站点id").size()
print("\n【对照】07 点真实进出（同一批站）：")
cmp = pd.DataFrame({"预测净流": tab["net"].round(2),
                    "实际出": actd.reindex(tab.index).fillna(0).astype(int),
                    "实际进": act.reindex(tab.index).fillna(0).astype(int)})
cmp["实际净流"] = cmp["实际进"] - cmp["实际出"]
print(cmp.head(6).to_string())

json.dump(dict(pred_dep=pH.tolist(), pred_arr=aH.tolist(),
               top={str(k): {"wd": float(v.wd), "wa": float(v.wa), "net": float(v.net),
                             "dep": float(v.dep), "arr": float(v.arr),
                             "act_dep": int(actd.get(k, 0)), "act_arr": int(act.get(k, 0))}
                    for k, v in tab.head(6).iterrows()},
               rest_wd=float(1 - tab.head(6).wd.sum()), rest_wa=float(1 - tab.head(6).wa.sum())),
          open(_out("split.json"), "w"), ensure_ascii=False)
print("\n已导出 split.json")
