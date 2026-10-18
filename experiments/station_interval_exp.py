# -*- coding: utf-8 -*-
"""
站点 × 调度窗（3h）级预测：top30 活跃站，3 个调度窗（06-09 / 12-15 / 18-21）。
输入：过去 3 天（24 个 3h 块）该站借/还序列 + 时间编码 + 站点静态（经纬度/日均量）
输出：目标窗 3 小时借出、归还总量（同一窗口内不递归）
对照：persistence（昨日同窗）、seasonal-naive（上周同日同窗）
产出：station_interval_pred.csv（测试期逐站逐窗预测 + 供调度用的缺口清单）
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
from tensorflow.keras import layers, models
import lstm_dispatch_baseline as B

WIN_BLK = 24                 # 回看 24 个 3h 块 = 3 天
WINDOWS = [6, 12, 18]        # 调度窗起始小时（3h）
SEED = 42


def mae(p, t):  return float(np.abs(p - t).mean())
def rmse(p, t): return float(np.sqrt(((p - t) ** 2).mean()))
def r2(p, t):
    st = ((p - t.mean()) ** 2).sum()
    return float(1 - ((p - t) ** 2).sum() / st) if st > 1e-9 else float("nan")


df = B.load_clean()
stations = sorted(set(df["开始站点id"].dropna()) | set(df["结束站点"].dropna()))
idx3 = pd.date_range(df["ts"].min().floor("3h"), df["ts"].max().floor("3h"), freq="3h")
nb = len(idx3)
days_all = pd.date_range(df["start"].dt.normalize().min(), df["start"].dt.normalize().max(), freq="D")

dep3 = (df.assign(b=df["ts"].dt.floor("3h")).groupby(["开始站点id", "b"]).size()
        .unstack(1).reindex(index=stations, columns=idx3).fillna(0))
arr3 = (df.assign(b=df["ts"].dt.floor("3h")).groupby(["结束站点", "b"]).size()
        .unstack(1).reindex(index=stations, columns=idx3).fillna(0))

# top30 按日均借出（全期，仅用于选站，不参与建模）
lvl_all = dep3.sum(1) / dep3.shape[1] * 8
top = lvl_all.sort_values(ascending=False).index[:30].tolist()
S = len(top)
print(f"3h 块 {nb} 个（{idx3[0]} ~ {idx3[-1]}），选站 {S} 个，日均借出 {lvl_all[top].mean():.1f} 辆")

blk_hour = idx3.hour.values
blk_dow = idx3.dayofweek.values
blk_dayidx = (idx3.normalize() - idx3.normalize()[0]).days.values

Xs, OTH, Ys, SID, WINID, TGT_T = [], [], [], [], [], []
for si, st in enumerate(top):
    d = dep3.loc[st].values.astype("float32")
    a = arr3.loc[st].values.astype("float32")
    sc = max(np.percentile(np.concatenate([d, a]), 99), 1.0)
    for t in range(WIN_BLK, nb):
        if blk_hour[t] not in WINDOWS:
            continue
        sl = slice(t - WIN_BLK, t)
        Xs.append(np.stack([d[sl] / sc, a[sl] / sc,
                            np.sin(2 * np.pi * blk_hour[sl] / 24), np.cos(2 * np.pi * blk_hour[sl] / 24),
                            np.sin(2 * np.pi * blk_dow[sl] / 7), np.cos(2 * np.pi * blk_dow[sl] / 7),
                            (blk_dow[sl] >= 5).astype("float32")], 1))
        Ys.append([d[t] / sc, a[t] / sc])
        SID.append(si); WINID.append(WINDOWS.index(blk_hour[t]))
        OTH.append(blk_dow[t]); TGT_T.append(t)
Xs = np.array(Xs, dtype="float32"); Ys = np.array(Ys, dtype="float32")
SID = np.array(SID); WINID = np.array(WINID); TGT_T = np.array(TGT_T)
N = len(SID)
print(f"样本 {N} 条（每站约 {N // S} 条）")

# 额外特征：站点 one-hot、窗口 one-hot、目标日 dow sin/cos + weekend
oh_s = np.eye(S, dtype="float32")[SID]
oh_w = np.eye(3, dtype="float32")[WINID]
dow = np.array(OTH, dtype="float32")
EXT = np.concatenate([oh_s, oh_w, np.sin(2 * np.pi * dow / 7)[:, None],
                      np.cos(2 * np.pi * dow / 7)[:, None], (dow >= 5)[:, None].astype("float32")], 1)
tgt_day = blk_dayidx[TGT_T]
d_tr, d_va = int(len(days_all) * .70), int(len(days_all) * .85)
i_tr = np.where(tgt_day < d_tr)[0]; i_va = np.where((tgt_day >= d_tr) & (tgt_day < d_va))[0]
i_te = np.where(tgt_day >= d_va)[0]
print(f"训练 {len(i_tr)} / 验证 {len(i_va)} / 测试 {len(i_te)}（测试目标日 {days_all[d_va].date()} 起）")

tf.random.set_seed(SEED); np.random.seed(SEED)
inp_x, inp_e = layers.Input((WIN_BLK, 7)), layers.Input((S + 3 + 3,))
h = layers.LSTM(48, return_sequences=True)(inp_x)
h = layers.LSTM(32)(h)
h = layers.Concatenate()([h, inp_e])
h = layers.Dense(32, activation="relu")(h)
h = layers.Dropout(.2)(h)
out = layers.Dense(2)(h)
model = models.Model([inp_x, inp_e], out)
ym, ys = Ys[i_tr].mean(0), Ys[i_tr].std(0) + 1e-6
model.compile("adam", loss=tf.keras.losses.Huber(delta=0.5))
model.fit([Xs[i_tr], EXT[i_tr]], (Ys[i_tr] - ym) / ys,
          validation_data=([Xs[i_va], EXT[i_va]], (Ys[i_va] - ym) / ys),
          epochs=120, batch_size=64, verbose=0,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)])

sc_all = np.array([max(np.percentile(np.concatenate([dep3.loc[st].values, arr3.loc[st].values]), 99), 1.0) for st in top])
sc_te = sc_all[SID[i_te]][:, None]
pred = model.predict([Xs[i_te], EXT[i_te]], verbose=0) * ys + ym
pred *= sc_te
true = Ys[i_te] * sc_te

# 基线：昨日同窗 / 上周同日同窗（目标块 t-8、t-56）
def lag_vals(lag):
    out = np.full_like(true, np.nan)
    for i, (si, t) in enumerate(zip(SID[i_te], TGT_T[i_te])):
        tt = t - lag
        if tt >= 0:
            out[i] = [dep3.iloc[si, tt], arr3.iloc[si, tt]]
    return out

pers, snv = lag_vals(8), lag_vals(56)


def metric_block(tag, p, t):
    rows = []
    for j, nm in enumerate(["借出 dep", "归还 arr"]):
        rows.append(dict(口径=tag, 目标=nm, MAE=round(mae(p[:, j], t[:, j]), 2),
                         RMSE=round(rmse(p[:, j], t[:, j]), 2), R2=round(r2(p[:, j], t[:, j]), 3)))
    pn, tn = p[:, 0] - p[:, 1], t[:, 0] - t[:, 1]
    rows.append(dict(口径=tag, 目标="窗口净流", MAE=round(mae(pn, tn), 2),
                     RMSE=round(rmse(pn, tn), 2), R2=round(r2(pn, tn), 3)))
    rows.append(dict(口径=tag, 目标="方向准确率", MAE=round(float((np.sign(pn) == np.sign(tn)).mean()), 3),
                     RMSE=np.nan, R2=np.nan))
    return rows


res = pd.DataFrame(metric_block("站点×窗口 LSTM", pred, true)
                   + metric_block("persistence", pers, true)
                   + metric_block("seasonal-naive", snv, true))
print("\n===== top30 站点 × 3h 调度窗（测试集）=====")
print(res.to_string(index=False))
print("注：方向准确率写在 MAE 列（0~1）。")

# 分窗口细看
rows = []
for k, (h0) in enumerate(WINDOWS):
    m_ = WINID[i_te] == k
    pn, tn = pred[m_, 0] - pred[m_, 1], true[m_, 0] - true[m_, 1]
    rows.append(dict(调度窗=f"{h0:02d}-{h0+3:02d}", 样本=int(m_.sum()),
                     MAE=round(mae(pn, tn), 2), R2=round(r2(pn, tn), 3),
                     方向准确率=round(float((np.sign(pn) == np.sign(tn)).mean()), 3)))
win_tab = pd.DataFrame(rows)
print("\n===== 分调度窗 =====")
print(win_tab.to_string(index=False))

# 调度清单（测试期，按预测缺口排序）
order = np.argsort(pred[:, 0] - pred[:, 1])           # 净流出最多在前
out = pd.DataFrame({
    "日期": idx3[TGT_T[i_te]].normalize().values,
    "调度窗": [f"{h:02d}-{h+3:02d}" for h in (idx3[TGT_T[i_te]].hour.values)],
    "站点": [top[s] for s in SID[i_te]],
    "预测借出": pred[:, 0].round(1), "预测归还": pred[:, 1].round(1),
    "预测净流": (pred[:, 0] - pred[:, 1]).round(1),
    "实际净流": (true[:, 0] - true[:, 1]).round(1),
}).iloc[order]
out["建议"] = np.where(out["预测净流"] <= -5, "补车", np.where(out["预测净流"] >= 5, "收车", "不动"))
pth = _out("station_interval_pred.csv")
out.to_csv(pth, index=False, encoding="utf-8-sig")

# 预警命中率：预测补车(|净流|>=5) 中实际确实缺车的比例
alert = out[out["预测净流"] <= -5]
hit = float((alert["实际净流"] <= -3).mean()) if len(alert) else float("nan")
print(f"\n测试期共 {len(out)} 条站-窗记录；预测补车预警 {len(alert)} 条（占 {len(alert)/len(out):.1%}），"
      f"其中实际缺车(≤-3 辆) {hit:.0%}")

print("\n前 10 条调度建议：")
print(out.head(10).to_string(index=False))
print(f"\nsaved -> {pth}")
