# -*- coding: utf-8 -*-
"""生成 P0 模型的输入/输出规格文档。"""
import json, os
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(BASE)
OUT = os.path.join(_ROOT, "docs")          # 报告 HTML 输出
OUTPUTS = os.path.join(_ROOT, "outputs")   # 实验产物
os.makedirs(OUT, exist_ok=True)
ex = json.load(open(os.path.join(OUTPUTS, "io_example.json"), encoding="utf-8"))
alert = pd.read_csv(os.path.join(OUTPUTS, "dispatch_alert.csv"))

# ---------- 图：数据流 ----------
def flow():
    W, H = 880, 206
    bw, bh = 176, 56
    xs = [8, 228, 448, 668]
    top, bot = 16, 116
    boxes = [
        (xs[0], top, "行程流水表", "65,535 条 × 15 列"),
        (xs[1], top, "片区小时面板", "2,021 小时 × 6 片区"),
        (xs[2], top, "输入窗口 X", "(24, 7) 浮点矩阵"),
        (xs[3], top, "LSTM 双通道", "48 → 32 → Dense(6)"),
        (xs[3], bot, "输出 Y", "6 个数：借 3 + 还 3"),
        (xs[2], bot, "合成净流", "Σ 归还 − Σ 借出 = ±辆"),
        (xs[1], bot, "站点份额拆分", "同簇·同时段历史占比"),
        (xs[0], bot, "调度表", "119 站 × 12 列"),
    ]
    g = []
    for x, y, t, s in boxes:
        g.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="10" fill="#fff" stroke="#dfe3ea"/>')
        g.append(f'<text x="{x+bw/2}" y="{y+24}" text-anchor="middle" font-size="12.5" font-weight="700" fill="#151a23">{t}</text>')
        g.append(f'<text x="{x+bw/2}" y="{y+42}" text-anchor="middle" font-size="11" fill="#77808f">{s}</text>')
    def arrow(x1, y1, x2, y2):
        g.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#b9c0cc" stroke-width="1.4"/>')
        if x2 > x1:
            g.append(f'<polygon points="{x2},{y2} {x2-7},{y2-4} {x2-7},{y2+4}" fill="#b9c0cc"/>')
        elif x2 < x1:
            g.append(f'<polygon points="{x2},{y2} {x2+7},{y2-4} {x2+7},{y2+4}" fill="#b9c0cc"/>')
        else:
            g.append(f'<polygon points="{x2},{y2} {x2-4},{y2-7} {x2+4},{y2-7}" fill="#b9c0cc"/>')
    for i in range(3):
        arrow(xs[i] + bw + 4, top + bh / 2, xs[i + 1] - 4, top + bh / 2)
    arrow(xs[3] + bw / 2, top + bh + 4, xs[3] + bw / 2, bot - 4)
    for i in [3, 2, 1]:
        arrow(xs[i] - 4, bot + bh / 2, xs[i - 1] + bw + 4, bot + bh / 2)
    for i, lb in enumerate(["① 聚合", "② 滑窗", "③ 建模"]):
        g.append(f'<text x="{xs[i]+bw+22}" y="{top+bh/2-9}" text-anchor="middle" font-size="10" fill="#a6aebb">{lb}</text>')
    for i, lb in enumerate(["⑥ 派车", "⑤ 拆站", "④ 合成"]):
        g.append(f'<text x="{xs[3-i]+bw/2}" y="{bot+bh+16}" text-anchor="middle" font-size="10" fill="#a6aebb">{lb}</text>')
    return f'<svg viewBox="0 0 {W} {H}" width="100%" role="img">{"".join(g)}</svg>'

# ---------- 表：输入窗口尾部 ----------
rows = ""
for r in ex["last12"]:
    ts, hh, dw, d, a, dn, an = r
    rows += (f'<tr><td><code>{ts[5:16]}</code></td><td class="num">{hh}</td><td class="num">{dw}</td>'
             f'<td class="num">{d:.0f}</td><td class="num">{a:.0f}</td>'
             f'<td class="num" style="color:#2563eb">{dn:.3f}</td><td class="num" style="color:#0d9488">{an:.3f}</td></tr>')

# ---------- 表：真实输出 vs 实际 ----------
orows = ""
for k in range(3):
    orows += (f'<tr><td><b>y{k+1}</b> / <b>y{k+4}</b></td><td>{ex["hours"][k]}</td>'
              f'<td class="num">{ex["pred_dep"][k]:.2f}</td><td class="num" style="color:#77808f">{ex["actual_dep"][k]:.0f}</td>'
              f'<td class="num">{ex["pred_arr"][k]:.2f}</td><td class="num" style="color:#77808f">{ex["actual_arr"][k]:.0f}</td></tr>')

# ---------- 表：调度表样例 ----------
pick = pd.concat([alert.nsmallest(3, "pred_net_3h"), alert.nlargest(2, "pred_net_3h")])
trows = ""
for _, r in pick.iterrows():
    col = "#c02626" if r["pred_net_3h"] < 0 else "#0b7f72"
    trows += (f'<tr><td>{r["date"]}</td><td><b>{r["station_id"]}</b></td><td class="num">{r["cluster"]}</td>'
              f'<td>{r["window"]}</td><td class="num">{r["pred_net_h1"]:+.2f}</td>'
              f'<td class="num">{r["pred_net_h2"]:+.2f}</td><td class="num">{r["pred_net_h3"]:+.2f}</td>'
              f'<td class="num" style="color:{col};font-weight:700">{r["pred_net_3h"]:+.2f}</td>'
              f'<td class="num">{r["need_bikes"]}</td><td>{r["action"]}</td><td>{r["priority"]}</td></tr>')

HEAD = open(os.path.join(BASE, "spec_head.html"), encoding="utf-8").read()
HTML = (HEAD
        .replace("__FLOW__", flow())
        .replace("__INROWS__", rows)
        .replace("__OUTROWS__", orows)
        .replace("__TROWS__", trows)
        .replace("__SCALE__", f'{ex["scale"]:.0f}')
        .replace("__NET__", f'{ex["net_3h"]:+.2f}')
        .replace("__ACTNET__", f'{ex["actual_net"]:+.2f}')
        .replace("__NST__", str(len(alert))))
p = os.path.join(OUT, "P0模型_输入输出规格.html")
open(p, "w", encoding="utf-8").write(HTML)
print("written", p, len(HTML))
