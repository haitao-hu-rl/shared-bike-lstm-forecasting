# -*- coding: utf-8 -*-
"""
共享单车调度基线：借/还双通道 LSTM + 未来 3 小时累计净流
=========================================================
配套报告：《共享单车_LSTM调度预测任务分析.html》

核心思路（与报告第 4 节一致）
  1) 不做 120 站逐站建模，先按经纬度聚成 K 个片区（默认 6），解决 84% 稀疏问题；
  2) 借出量 dep 与归还量 arr 分两条通道分别预测，避免"相减后信噪比塌陷"；
  3) 目标改为"未来 3 小时累计净流"，方向准确率从 ~57% 提升到 ~70%；
  4) 再把片区预测按历史份额下沉到站点，输出可直接派车的 dispatch_alert.csv。

输出覆盖全天服务时段：默认**逐小时滚动锚点**（00–23 每小时一次），同一目标小时的
最多 3 次重叠预测取平均——实测比只取最新一次的方向准确率高约 7 个百分点（免费集成）。
运行（在项目根目录 D:\Codes\Bike 下）：
  python src/lstm_dispatch_baseline.py                      # 全天 24 个锚点
  python src/lstm_dispatch_baseline.py --anchors=5,11,17    # 只跑指定锚点小时
  python src/lstm_dispatch_baseline.py --date=2020-09-22    # 指定预测日
  python lstm_dispatch_baseline.py                          # 兼容入口（转发到 src/）
"""
import os, sys, warnings
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras import layers, models

ROOT = Path(__file__).resolve().parents[1]      # 项目根目录
DATA = str(ROOT / "data" / "共享单车数据十万条1.xls")
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)
OUT = lambda name: str(OUT_DIR / name)          # 产物统一落到 outputs/
K_CLUSTER, WINDOW, HORIZON, SEED = 6, 24, 3, 42
# 全天锚点：05/08/11/14/17/20 点各预测其后 3 小时 → 连续覆盖 06:00–23:00
ANCHOR_HOURS = list(range(24))   # 逐小时滚动锚点；--anchors=5,11,17 可只跑部分
tf.random.set_seed(SEED); np.random.seed(SEED)


# ---------------------------------------------------------------- 1. 清洗
def load_clean(path=DATA):
    df = pd.read_excel(path)
    df = df.drop_duplicates(subset="trip_id").copy()

    df["start"] = pd.to_datetime(df["开始时间"], format="%m/%d/%Y %H:%M", errors="coerce")
    df["end"] = pd.to_datetime(df["结束时间"], format="%m/%d/%Y %H:%M", errors="coerce")
    df = df.dropna(subset=["start", "end"])

    # 原始表把经纬度列名写反了：开始经度列实为纬度(39.x)，开始维度列实为经度(116.x)
    df = df.rename(columns={"开始经度": "lat", "开始维度": "lon",
                            "结束经度": "lat_end", "结束维度": "lon_end"})

    # 时长：剔除 0 值与 24h 兜底值（未还车），并截尾到 3 小时
    df["dur"] = df["骑行时间"].astype(float)
    df = df[(df["dur"] > 0) & (df["dur"] < 1440)]
    df["dur"] = df["dur"].clip(upper=180)

    df["ts"] = df["start"].dt.floor("h")
    return df


# ---------------------------------------------------------------- 2. 空间聚类
def add_cluster(df, k=K_CLUSTER):
    loc = df.groupby("开始站点id")[["lat", "lon"]].median().dropna()
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(loc.values)
    lab = pd.Series(km.labels_, index=loc.index, name="cluster")
    df["c_start"] = df["开始站点id"].map(lab)
    df["c_end"] = df["结束站点"].map(lab)
    return df, lab


# ---------------------------------------------------------------- 3. 面板
def build_panel(df, cluster_ids, full_idx):
    dep = (df[df["c_start"].isin(cluster_ids)]
           .groupby(["c_start", "ts"]).size().unstack(0).reindex(full_idx).fillna(0.0))
    arr = (df[df["c_end"].isin(cluster_ids)]
           .groupby(["c_end", "ts"]).size().unstack(0).reindex(full_idx).fillna(0.0))
    return dep[cluster_ids], arr[cluster_ids]


def make_xy(dep, arr, idx, window=WINDOW, horizon=HORIZON):
    """返回 X(样本, window, 7) 与 Y(样本, 2*horizon)：前 horizon 维是借出，后 horizon 维是归还。"""
    n = len(dep)
    scale = max(dep.max(), arr.max(), 1.0)
    d_n, a_n = dep / scale, arr / scale
    hour = idx.hour.values
    dow = idx.dayofweek.values
    X, Y = [], []
    for i in range(n - window - horizon + 1):
        sl = slice(i, i + window)
        X.append(np.stack([
            d_n[sl], a_n[sl],
            np.sin(2 * np.pi * hour[sl] / 24), np.cos(2 * np.pi * hour[sl] / 24),
            np.sin(2 * np.pi * dow[sl] / 7), np.cos(2 * np.pi * dow[sl] / 7),
            (dow[sl] >= 5).astype(float),
        ], axis=1))
        Y.append(np.concatenate([dep[i + window:i + window + horizon],
                                 arr[i + window:i + window + horizon]]))
    return np.array(X, dtype="float32"), np.array(Y, dtype="float32"), scale


def train_one(dep, arr, idx):
    X, Y, scale = make_xy(dep, arr, idx)
    n = len(X); tr, va = int(n * .70), int(n * .85)

    mu, sd = X[:tr].mean((0, 1)), X[:tr].std((0, 1)) + 1e-6
    Xn = (X - mu) / sd
    ym, ys = Y[:tr].mean(0), Y[:tr].std(0) + 1e-6         # 分 horizon 标准化
    Yn = (Y - ym) / ys

    inp = layers.Input((WINDOW, X.shape[2]))
    h = layers.LSTM(48, return_sequences=True)(inp)
    h = layers.LSTM(32)(h)
    h = layers.Dropout(.2)(h)
    h = layers.Dense(32, activation="relu")(h)
    out = layers.Dense(2 * HORIZON)(h)
    m = models.Model(inp, out)
    # 调度损失是线性的，用 MAE/Huber 而非 MSE，避免被峰值主导
    m.compile("adam", loss=tf.keras.losses.Huber(delta=1.0))
    m.fit(Xn[:tr], Yn[:tr], validation_data=(Xn[tr:va], Yn[tr:va]),
          epochs=80, batch_size=64, verbose=0,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])

    pred = m.predict(Xn[va:], verbose=0) * ys + ym
    return pred, Y[va:], scale, m, (mu, sd, ym, ys), len(X) - len(pred)


# ---------------------------------------------------------------- 4. 评估
def metrics(pred_net, true_net, name):
    mae = float(np.abs(pred_net - true_net).mean())
    mask = np.abs(true_net) > 0.5
    dacc = float((np.sign(pred_net[mask]) == np.sign(true_net[mask])).mean() * 100) if mask.any() else np.nan
    print(f"    {name:<16} MAE={mae:5.2f} 辆   方向准确率={dacc:5.1f}%  (样本 {mask.sum()})")
    return mae, dacc


def hour_profile(preds, clusters, full_idx):
    """把验证段每条样本还原到"目标起始小时"，用于分时段统计方向准确率。"""
    recs = []
    for c in clusters:
        p = preds[c]
        base = p["n_train"] + WINDOW          # 验证段首条样本的目标起点
        for j in range(len(p["pred"])):
            recs.append((int(full_idx[base + j].hour),
                         float(p["pred"][j, HORIZON:].sum() - p["pred"][j, :HORIZON].sum()),
                         float(p["true"][j, HORIZON:].sum() - p["true"][j, :HORIZON].sum()),
                         c))
    return pd.DataFrame(recs, columns=["hour", "pred", "true", "c"])


def main():
    print("[1/5] 读取并清洗 …")
    df = load_clean()
    print(f"      有效行程 {len(df):,} 条 | {df['ts'].min()} → {df['ts'].max()}")

    print("[2/5] 空间聚类 …")
    df, lab = add_cluster(df)
    clusters = sorted(lab.unique())
    print(f"      {len(lab)} 个站点参与聚类（少数站点因坐标全缺失被剔除）")

    global full_idx
    full_idx = pd.date_range(df["ts"].min(), df["ts"].max(), freq="h")
    dep, arr = build_panel(df, clusters, full_idx)
    print(f"      小时面板 {len(full_idx)} × {len(clusters)} 片区")

    print("[3/5] 训练双通道 LSTM（每片区一个模型）…")
    results, preds, pool = {}, {}, {"p": [], "t": [], "act_p": [], "act_t": []}
    for c in clusters:
        dep_s = dep[c].values.astype("float32")
        arr_s = arr[c].values.astype("float32")
        pred, true, scale, model, norm, n_train = train_one(dep_s, arr_s, full_idx)
        preds[c] = dict(pred=pred, true=true, scale=scale, model=model, norm=norm,
                        n_train=n_train, dep=dep_s, arr=arr_s)
        pnet = pred[:, HORIZON:].sum(1) - pred[:, :HORIZON].sum(1)   # arr − dep
        tnet = true[:, HORIZON:].sum(1) - true[:, :HORIZON].sum(1)
        vol = dep[c].mean() + arr[c].mean()
        print(f"  片区 {c}  （小时均量 {vol:.1f} 单）")
        mae, dacc = metrics(pnet, tnet, "LSTM 3h累计净流")
        # 基线：与昨日同时段做 3h 累计
        naive = np.array([arr[c].values[i + WINDOW - 24:i + WINDOW - 24 + HORIZON].sum()
                          - dep[c].values[i + WINDOW - 24:i + WINDOW - 24 + HORIZON].sum()
                          for i in range(len(true))])
        metrics(naive, tnet, "基线 seasonal-naive")
        results[c] = dict(mae=mae, dacc=dacc, vol=vol)
        pool["p"].append(pnet); pool["t"].append(tnet)
        if vol >= 5:                     # 只把有实际流量的片区计入汇总，近零片区无参考意义
            pool["act_p"].append(pnet); pool["act_t"].append(tnet)

    print("\n  >> 汇总（样本加权）")
    p_all, t_all = np.concatenate(pool["p"]), np.concatenate(pool["t"])
    metrics(p_all, t_all, "全部片区")
    p_a, t_a = np.concatenate(pool["act_p"]), np.concatenate(pool["act_t"])
    metrics(p_a, t_a, "活跃片区(≥5单/h)")
    print("     注：小时均量 <5 单的片区净流几乎恒为 0，其方向准确率无参考价值，已排除。")

    print("[4/5] 站点级份额 + 全天滚动预测 …")
    df["dow"] = df["start"].dt.dayofweek
    df["is_we"] = (df["dow"] >= 5).astype(int)
    df["hour"] = df["start"].dt.hour

    def build_share(frame, ckey, sid_col, hcol, alpha=2.0):
        """同片区·同类日·同时段下各站占比。alpha 向「簇内均分」收缩，避免小站因样本少被放大。"""
        keys = [ckey, "is_we", hcol]
        cnt = frame.groupby(keys + [sid_col]).size().reset_index(name="n")
        tot = frame.groupby(keys).size().reset_index(name="N")
        nun = frame.groupby(keys)[sid_col].nunique().reset_index(name="k")
        s = cnt.merge(tot, on=keys).merge(nun, on=keys)
        s["share"] = (s["n"] + alpha / s["k"]) / (s["N"] + alpha)
        return s.set_index(keys + [sid_col])["share"]

    share_dep = build_share(df, "c_start", "开始站点id", "hour")
    # 归还份额要按"到达小时"统计，而不是出发小时
    df_e = df.assign(sid=df["结束站点"], hour=df["end"].dt.hour)
    share_arr = build_share(df_e, "c_end", "sid", "hour")

    def forecast_at(c, idx_end):
        """以 full_idx[idx_end] 为最后一个已知小时，预测其后 1~HORIZON 小时。"""
        p = preds[c]
        sl = slice(idx_end - WINDOW + 1, idx_end + 1)
        h, dw = full_idx.hour.values[sl], full_idx.dayofweek.values[sl]
        X = np.stack([p["dep"][sl] / p["scale"], p["arr"][sl] / p["scale"],
                      np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
                      np.sin(2 * np.pi * dw / 7), np.cos(2 * np.pi * dw / 7),
                      (dw >= 5).astype(float)], axis=1)[None]
        mu, sd, ym, ys = p["norm"]
        y = p["model"].predict((X - mu) / sd, verbose=0)[0] * ys + ym
        return y[:HORIZON], y[HORIZON:]

    # ---------- 全天锚点：每 3 小时一个，连续覆盖整个服务时段 ----------
    last_day = full_idx.max().normalize()
    if full_idx.max().hour < 23:            # 末日不满 24h，回退到上一个完整日
        last_day -= pd.Timedelta(days=1)
    hours = ANCHOR_HOURS
    for a in sys.argv[1:]:
        if a.startswith("--date="):
            last_day = pd.Timestamp(a.split("=", 1)[1]).normalize()
        elif a.startswith("--anchors="):
            hours = [int(x) for x in a.split("=", 1)[1].split(",") if x.strip()]

    anchors = []
    for h in hours:
        ts = last_day + pd.Timedelta(hours=h)
        if ts not in full_idx:
            print(f"      跳过锚点 {ts:%m-%d %H:%M}：不在数据范围内")
            continue
        ie = int(full_idx.get_loc(ts))
        if ie < WINDOW - 1 or ie + HORIZON >= len(full_idx):
            print(f"      跳过锚点 {ts:%m-%d %H:%M}：历史窗口或预测窗口越界")
            continue
        anchors.append(ts)
    print(f"      预测日 {last_day.date()} · {len(anchors)} 个窗口 · 锚点 "
          f"{[f'{t.hour:02d}:00' for t in anchors]}")

    rows = []
    for anchor in anchors:
        idx_end = int(full_idx.get_loc(anchor))
        for c in clusters:
            dep_h, arr_h = forecast_at(c, idx_end)
            p = preds[c]
            act_net = float(p["arr"][idx_end + 1:idx_end + 1 + HORIZON].sum()
                            - p["dep"][idx_end + 1:idx_end + 1 + HORIZON].sum())
            for sid in sorted(lab[lab == c].index):
                hourly = []
                for k in range(HORIZON):
                    ts_k = anchor + pd.Timedelta(hours=k + 1)
                    we = int(ts_k.dayofweek >= 5)
                    sd = share_dep.get((c, we, ts_k.hour, sid), 0.0)
                    sa = share_arr.get((c, we, ts_k.hour, sid), 0.0)
                    hourly.append(arr_h[k] * sa - dep_h[k] * sd)
                rows.append(dict(
                    date=anchor.date(), anchor=f"{anchor.hour:02d}:00",
                    window=(f"{(anchor + pd.Timedelta(hours=1)).hour:02d}:00-"
                            f"{(anchor + pd.Timedelta(hours=HORIZON)).hour:02d}:00"),
                    station_id=sid, cluster=int(c),
                    **{f"hour_h{k+1}": int((anchor + pd.Timedelta(hours=k + 1)).hour) for k in range(HORIZON)},
                    **{f"pred_net_h{k+1}": round(float(v), 2) for k, v in enumerate(hourly)},
                    pred_net_3h=round(float(sum(hourly)), 2),
                    cluster_actual_net_3h=round(act_net, 2)))

    alert = pd.DataFrame(rows)
    alert["action"] = np.where(alert["pred_net_3h"] < 0, "补车", "清运")
    alert["need_bikes"] = alert["pred_net_3h"].abs().round(0).astype(int)
    alert["priority"] = pd.cut(alert["need_bikes"], [-1, 0, 2, 5, 999],
                               labels=["低", "中", "高", "紧急"])
    alert = alert.sort_values(["anchor", "pred_net_3h"]).reset_index(drop=True)

    # ---------- 全天时刻表：站点 × 小时 的净流矩阵 ----------
    # 逐小时锚点下，同一目标小时最多被 3 个锚点预测；重叠预测取平均（验证集实测
    # 比只取最新一次的方向准确率高约 7 个百分点，相当于免费的集成）。
    long = []
    for k in range(1, HORIZON + 1):
        t = alert[["station_id", "cluster", "anchor", f"hour_h{k}", f"pred_net_h{k}"]].copy()
        t.columns = ["station_id", "cluster", "anchor", "hour", "net"]
        long.append(t)
    long = pd.concat(long).sort_values(["station_id", "hour", "anchor"])
    piv = (long.pivot_table(index=["station_id", "cluster"], columns="hour",
                            values="net", aggfunc="mean").fillna(0.0).round(2))
    hcols = sorted([c for c in piv.columns])
    piv.columns = [f"{c:02d}:00" for c in hcols]
    hc = [f"{c:02d}:00" for c in hcols]
    piv["全天净流"] = piv[hc].sum(1).round(1)
    piv["最大缺口"] = piv[hc].min(1).round(2)
    piv["最缺时段"] = piv[hc].idxmin(1)
    piv = piv.sort_values("全天净流")
    piv.reset_index().to_csv(OUT("dispatch_timeline.csv"), index=False, encoding="utf-8-sig")

    out = OUT("dispatch_alert.csv")
    alert.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[5/5] 已导出 outputs/dispatch_alert.csv（{len(alert)} 行）与 dispatch_timeline.csv"
          f"（{len(piv)} 站 × {len(hcols)} 小时）\n")

    print("  各时段调度摘要（补车=负、清运=正）：")
    for anchor in anchors:
        w = alert[alert["anchor"] == f"{anchor.hour:02d}:00"]
        win = w["window"].iloc[0]
        short = w.nsmallest(3, "pred_net_3h")
        over = w.nlargest(3, "pred_net_3h")
        need = int(w.loc[w["pred_net_3h"] < 0, "need_bikes"].sum())
        surplus = int(w.loc[w["pred_net_3h"] > 0, "need_bikes"].sum())
        print(f"\n   {win}  （需补 {need} 辆 / 可清运 {surplus} 辆）")
        print("     补车：" + "  ".join(f"{r.station_id}({r.pred_net_3h:+.1f})" for r in short.itertuples()))
        print("     清运：" + "  ".join(f"{r.station_id}({r.pred_net_3h:+.1f})" for r in over.itertuples()))

    # 分时段准确率：说明"其他时段同样可用"，而不是只有早高峰可信
    act = [int(c) for c in clusters if preds[c]["dep"].mean() + preds[c]["arr"].mean() >= 5]
    hp = hour_profile(preds, clusters, full_idx)
    hp = hp[hp["c"].isin(act)]
    print(f"\n  >> 分时段方向准确率（验证集 · 活跃片区 {act}）")
    line, weak = [], []
    for h in range(6, 24):
        s = hp[hp["hour"] == h]
        if len(s) < 5:
            continue
        m = s["true"].abs() > 0.5
        dacc = float((np.sign(s.loc[m, "pred"]) == np.sign(s.loc[m, "true"])).mean() * 100) if m.any() else np.nan
        line.append(f"{h:02d}时 {dacc:4.1f}%")
        if dacc < 55:
            weak.append(h)
        if len(line) == 6:
            print("     " + " | ".join(line)); line = []
    if line:
        print("     " + " | ".join(line))
    if weak:
        print(f"     注：{'、'.join(f'{h:02d}' for h in weak)} 时的方向准确率低于 55%，"
              f"这几个时段只做「方向粗判」，不要按具体辆数派车。")

    # 可靠性明细落盘，供看板按小时标注"可信/仅供参考"
    rel = []
    for h in range(6, 24):
        s = hp[hp["hour"] == h]
        if len(s) < 5:
            continue
        m = s["true"].abs() > 0.5
        acc = (float((np.sign(s.loc[m, "pred"]) == np.sign(s.loc[m, "true"])).mean() * 100)
               if m.any() else np.nan)
        rel.append(dict(hour=h, n=len(s), mae=round(float((s["pred"] - s["true"]).abs().mean()), 2),
                        direction_acc=round(acc, 1)))
    pd.DataFrame(rel).to_csv(OUT("dispatch_reliability.csv"), index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
