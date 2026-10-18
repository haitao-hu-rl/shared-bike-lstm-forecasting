# -*- coding: utf-8 -*-
"""
执行方案对照：空间聚合（片区-小时） vs 时间聚合（站点-日）
1) 零值率普查：站点/片区 × 小时/3小时/日
2) 站点-日模型：未来日借出/归还双输出，对比 persistence / seasonal-naive
3) 高活跃站点子集（按训练期日均单量 top-30）单独评估
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

WIN = 7          # 用过去 7 天预测下一天
SEED = 42


def mae(p, t):  return float(np.abs(p - t).mean())
def rmse(p, t): return float(np.sqrt(((p - t) ** 2).mean()))
def r2(p, t):
    st = ((t - t.mean()) ** 2).sum()
    return float(1 - ((p - t) ** 2).sum() / st) if st > 1e-9 else float("nan")


df = B.load_clean()
stations = sorted(set(df["开始站点id"].dropna()) | set(df["结束站点"].dropna()))
days = pd.date_range(df["start"].dt.normalize().min(), df["start"].dt.normalize().max(), freq="D")
S, D = len(stations), len(days)
print(f"站点 {S} 个，日期 {D} 天（{days[0].date()} ~ {days[-1].date()}），订单 {len(df)} 条")

# ---------- 1. 零值率普查 ----------
dep_h = df.groupby(["开始站点id", df["ts"]]).size().reset_index(name="v")
arr_h = df.groupby(["结束站点", df["ts"]]).size().reset_index(name="v")
hours = pd.date_range(df["ts"].min(), df["ts"].max(), freq="h")
n_hour_slots = S * len(hours)

dep_d = (df.groupby(["开始站点id", df["start"].dt.normalize()]).size()
         .unstack(1).reindex(index=stations, columns=days).fillna(0))
arr_d = (df.groupby(["结束站点", df["start"].dt.normalize()]).size()
         .unstack(1).reindex(index=stations, columns=days).fillna(0))

# 站点 3 小时块（非重叠）
dep_3h = dep_h.assign(b=dep_h["ts"].dt.floor("3h")).groupby(["开始站点id", "b"]).size()
arr_3h = arr_h.assign(b=arr_h["ts"].dt.floor("3h")).groupby(["结束站点", "b"]).size()
n_3h_slots = S * len(pd.date_range(df["ts"].min().floor("3h"), df["ts"].max(), freq="3h"))

# 片区级
dfc, lab = B.add_cluster(df)
clusters = sorted(lab.unique())
dep_z = B.build_panel(dfc, clusters, hours)[0]
dep_z3 = dep_z.resample("3h").sum()
dep_zd = dep_z.resample("D").sum()

print("\n===== 零值率普查 =====")
print(f"站点-小时    : 空槽 {1 - len(dep_h) / n_hour_slots:.1%}   （{n_hour_slots} 个槽）")
print(f"站点-3小时   : 空槽 {1 - len(dep_3h) / n_3h_slots:.1%}")
print(f"站点-日      : 空槽 {(dep_d.values == 0).mean():.1%}   （借出+归还合并 {(pd.concat([dep_d.stack(), arr_d.stack()]) == 0).mean():.1%}）")
print(f"片区-小时    : 空槽 {(dep_z.values == 0).mean():.1%}")
print(f"片区-3小时   : 空槽 {(dep_z3.values == 0).mean():.1%}")
print(f"片区-日      : 空槽 {(dep_zd.values == 0).mean():.1%}")
print(f"站点日均借出 : 均值 {dep_d.values.mean():.2f} 辆/天，中位数 {np.median(dep_d.values):.0f}，"
      f"≥10 辆/天的站点 {(dep_d.mean(1) >= 10).sum()} 个")

# ---------- 2. 站点-日模型 ----------
n_samp = D - WIN                      # 每站点可造 78 条样本
dow_all = days.dayofweek.values
Xs, Ys, SID, SCALE = [], [], [], np.zeros((S, 2), dtype="float32")
for si, st in enumerate(stations):
    d = dep_d.loc[st].values.astype("float32"); a = arr_d.loc[st].values.astype("float32")
    sd, sa = max(d.mean(), 1.0), max(a.mean(), 1.0)          # 训练前不做泄漏：全期均值只用于归一化尺度
    SCALE[si] = [sd, sa]
    for i in range(n_samp):
        sl = slice(i, i + WIN)
        Xs.append(np.stack([d[sl] / sd, a[sl] / sa,
                            np.sin(2 * np.pi * dow_all[sl] / 7), np.cos(2 * np.pi * dow_all[sl] / 7),
                            (dow_all[sl] >= 5).astype("float32")], 1))
        Ys.append([d[i + WIN] / sd, a[i + WIN] / sa])
        SID.append(si)
Xs = np.array(Xs, dtype="float32"); Ys = np.array(Ys, dtype="float32")
SID = np.array(SID)
pos = np.arange(len(SID)) % n_samp

# 站点静态特征：经纬度 + 日均单量（用前 70% 天算，避免测试泄漏）
tr_day_end = int(D * .70)
st_meta = df.groupby("开始站点id")[["lat", "lon"]].median().reindex(stations)
lvl = dep_d.iloc[:, :tr_day_end].mean(1).reindex(stations).values
meta = np.stack([st_meta["lat"].values, st_meta["lon"].values, np.log1p(lvl)], 1)
meta = (meta - np.nanmean(meta, 0)) / (np.nanstd(meta, 0) + 1e-6)
meta = np.nan_to_num(meta)
META = meta[SID]                                               # (N,3)

# 时间切分：目标日 <= 58 训练 / 59~71 验证 / >=72 测试
tgt_day = pos + WIN
td_tr, td_va = int(D * .70), int(D * .85)
i_tr = np.where(tgt_day < td_tr)[0]; i_va = np.where((tgt_day >= td_tr) & (tgt_day < td_va))[0]
i_te = np.where(tgt_day >= td_va)[0]

tf.random.set_seed(SEED); np.random.seed(SEED)
inp_x, inp_m = layers.Input((WIN, 5)), layers.Input((3,))
h = layers.LSTM(32, return_sequences=True)(inp_x)
h = layers.LSTM(24)(h)
h = layers.Concatenate()([h, inp_m])
h = layers.Dense(24, activation="relu")(h)
h = layers.Dropout(.15)(h)
out = layers.Dense(2)(h)
model = models.Model([inp_x, inp_m], out)
ym = Ys[i_tr].mean(0); ys = Ys[i_tr].std(0) + 1e-6
model.compile("adam", loss=tf.keras.losses.Huber(delta=0.5))
model.fit([Xs[i_tr], META[i_tr]], (Ys[i_tr] - ym) / ys,
          validation_data=([Xs[i_va], META[i_va]], (Ys[i_va] - ym) / ys),
          epochs=120, batch_size=64, verbose=0,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)])

pred = model.predict([Xs[i_te], META[i_te]], verbose=0) * ys + ym
sc_te = SCALE[SID[i_te]]
pred *= sc_te
true = Ys[i_te] * sc_te
sid_te = SID[i_te]
td_te = tgt_day[i_te]

# 基线：persistence（今天=明天）、seasonal-naive（上周同星期）
pers = np.stack([Xs[i_te][:, -1, 0] * sc_te[:, 0], Xs[i_te][:, -1, 1] * sc_te[:, 1]], 1)
snaive_idx = i_te - WIN + 0
ok = snaive_idx >= 0
snv = np.full_like(true, np.nan)
snv[ok] = Ys[snaive_idx[ok]] * sc_te[ok]


def report(tag, mask=None):
    m_ = np.ones(len(true), bool) if mask is None else mask
    rows = []
    for j, name in enumerate(["借出 dep", "归还 arr"]):
        p, t = pred[m_, j], true[m_, j]
        pk = pers[m_, j]
        sk = snv[m_, j]
        ok2 = ~np.isnan(sk)
        rows.append(dict(口径=tag, 目标=name, 样本=len(p),
                         MAE=round(mae(p, t), 2), RMSE=round(rmse(p, t), 2), R2=round(r2(p, t), 3),
                         persMAE=round(mae(pk, t), 2), persR2=round(r2(pk, t), 3),
                         snvMAE=round(mae(sk[ok2], t[ok2]), 2), snvR2=round(r2(sk[ok2], t[ok2]), 3)))
    pn, tn = pred[m_, 0] - pred[m_, 1], true[m_, 0] - true[m_, 1]
    rows.append(dict(口径=tag, 目标="净流 dep-arr", 样本=int(m_.sum()),
                     MAE=round(mae(pn, tn), 2), RMSE=round(rmse(pn, tn), 2), R2=round(r2(pn, tn), 3),
                     persMAE=round(mae(pers[m_, 0] - pers[m_, 1], tn), 2),
                     persR2=round(r2(pers[m_, 0] - pers[m_, 1], tn), 3),
                     snvMAE=np.nan, snvR2=np.nan))
    rows.append(dict(口径=tag, 目标="方向准确率", 样本=int(m_.sum()),
                     MAE=np.nan, RMSE=np.nan, R2=np.nan,
                     persMAE=np.nan, persR2=round(float((np.sign(pers[m_, 0] - pers[m_, 1]) == np.sign(tn)).mean()), 3),
                     snvMAE=np.nan, snvR2=np.nan))
    rows[-1]["MAE"] = round(float((np.sign(pn) == np.sign(tn)).mean()), 3)
    return rows


all_rows = report("全部站点") + report("top30 活跃站", mask=np.isin(sid_te, np.argsort(-lvl)[:30]))
res = pd.DataFrame(all_rows)
print("\n===== 站点-日模型（未来 24h 借出/归还）=====")
print(res.to_string(index=False))
print("\n注：方向准确率列写在 MAE 列（0~1）；净流的 seasonal-naive 未计算。")

res.to_csv(_out("station_day_exp.csv"), index=False, encoding="utf-8-sig")
print("saved -> outputs/station_day_exp.csv")
