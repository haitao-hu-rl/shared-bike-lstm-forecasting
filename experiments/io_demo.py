# -*- coding: utf-8 -*-
"""打印 P0 模型的真实输入/输出，用于规格文档中的算例。
运行：D:\\miniconda3\\envs\\bike\\python.exe experiments/io_demo.py
"""
# --- 产物统一输出到项目根 outputs/ ---
from pathlib import Path as _Path
_OUT_DIR = _Path(__file__).resolve().parents[1] / "outputs"
_OUT_DIR.mkdir(exist_ok=True)
def _out(name):
    return str(_OUT_DIR / name)

import os, sys, json, warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from lstm_dispatch_baseline import (load_clean, add_cluster, build_panel,
                                    train_one, WINDOW, HORIZON)

df = load_clean()
df, lab = add_cluster(df)
full_idx = pd.date_range(df["ts"].min(), df["ts"].max(), freq="h")
dep, arr = build_panel(df, sorted(lab.unique()), full_idx)

C = 4
dep_s = dep[C].values.astype("float32"); arr_s = arr[C].values.astype("float32")
pred, true, scale, model, norm, _ = train_one(dep_s, arr_s, full_idx)

anchor = pd.Timestamp("2020-09-22 06:00")
i = int(full_idx.get_loc(anchor))
sl = slice(i - WINDOW + 1, i + 1)
h, dw = full_idx.hour.values[sl], full_idx.dayofweek.values[sl]
X = np.stack([dep_s[sl] / scale, arr_s[sl] / scale,
              np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
              np.sin(2 * np.pi * dw / 7), np.cos(2 * np.pi * dw / 7),
              (dw >= 5).astype(float)], axis=1)[None]
mu, sd, ym, ys = norm
y = model.predict((X - mu) / sd, verbose=0)[0] * ys + ym
dep_h, arr_h = y[:HORIZON], y[HORIZON:]
net = float(arr_h.sum() - dep_h.sum())

out = dict(
    cluster=C, scale=float(scale), shape_in=list(X.shape), shape_out=[2 * HORIZON],
    anchor=str(anchor), window_start=str(full_idx[i - WINDOW + 1]),
    norm_mu=[float(v) for v in mu], norm_sd=[float(v) for v in sd],
    last12=[[str(full_idx[j]), int(full_idx[j].hour), int(full_idx[j].dayofweek),
             float(dep_s[j]), float(arr_s[j]),
             round(float(dep_s[j] / scale), 3), round(float(arr_s[j] / scale), 3)]
            for j in range(i - 11, i + 1)],
    pred_dep=[round(float(v), 2) for v in dep_h],
    pred_arr=[round(float(v), 2) for v in arr_h],
    hours=[str((anchor + pd.Timedelta(hours=k + 1)).strftime("%H:%M")) for k in range(HORIZON)],
    actual_dep=[float(v) for v in dep_s[i + 1:i + 1 + HORIZON]],
    actual_arr=[float(v) for v in arr_s[i + 1:i + 1 + HORIZON]],
    net_3h=round(net, 2),
    actual_net=round(float(arr_s[i + 1:i + 1 + HORIZON].sum() - dep_s[i + 1:i + 1 + HORIZON].sum()), 2),
)
print(json.dumps(out, ensure_ascii=False, indent=1))
open(_out("io_example.json"), "w", encoding="utf-8").write(
    json.dumps(out, ensure_ascii=False, indent=1))
