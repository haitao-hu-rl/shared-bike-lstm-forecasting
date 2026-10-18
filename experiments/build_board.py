# -*- coding: utf-8 -*-
"""把 dispatch_*.csv 组装成一张全天调度看板（单文件 HTML）。"""
import os
from pathlib import Path
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
OUT = str(_ROOT / "docs")        # 报告 HTML 输出
SRC = str(_ROOT / "outputs")     # 读取主流程产物
a = pd.read_csv(os.path.join(SRC, "dispatch_alert.csv"))
t = pd.read_csv(os.path.join(SRC, "dispatch_timeline.csv"), dtype={"station_id": str})
rel = pd.read_csv(os.path.join(SRC, "dispatch_reliability.csv"))

HOURS = [f"{h:02d}:00" for h in range(6, 24)]
weak_hours = set(f"{int(r.hour):02d}:00" for r in rel.itertuples() if r.direction_acc < 55)
date = str(a["date"].iloc[0])

# ---------- 时段卡片 ----------
cards = []
for anchor, w in a.groupby("anchor", sort=True):
    win = w["window"].iloc[0]
    short = w.nsmallest(3, "pred_net_3h")
    over = w.nlargest(3, "pred_net_3h")
    need = int(w.loc[w.pred_net_3h < 0, "need_bikes"].sum())
    sur = int(w.loc[w.pred_net_3h > 0, "need_bikes"].sum())
    chips_s = "".join(
        f'<span class="c red">{r.station_id}<i>{r.pred_net_3h:+.1f}</i></span>' for r in short.itertuples())
    chips_o = "".join(
        f'<span class="c teal">{r.station_id}<i>{r.pred_net_3h:+.1f}</i></span>' for r in over.itertuples())
    cards.append(f"""
    <div class="wcard">
      <div class="wt">{win}</div>
      <div class="wnum"><b class="red">{need}</b><span>辆需补</span><b class="teal">{sur}</b><span>辆可清</span></div>
      <div class="wlab">优先补车</div><div class="chips">{chips_s}</div>
      <div class="wlab">优先清运</div><div class="chips">{chips_o}</div>
    </div>""")

# ---------- 热力图 ----------
t["absmax"] = t[HOURS].abs().max(1)
sel = t.sort_values("absmax", ascending=False).head(26).copy()
sel = sel.sort_values("全天净流")
CAP = 2.2


def cell(v):
    if abs(v) < 0.05:
        return '<td class="z">·</td>'
    op = min(abs(v) / CAP, 1.0) * 0.78 + 0.08
    col = f"rgba(220,38,38,{op:.2f})" if v < 0 else f"rgba(13,148,136,{op:.2f})"
    txt = "#fff" if abs(v) / CAP > 0.55 else ("#8f1d1d" if v < 0 else "#0a5f57")
    return f'<td style="background:{col};color:{txt}">{v:+.1f}</td>'


hdr = "".join(
    f'<th class="{"weak" if h in weak_hours else ""}">{h[:2]}'
    f'<span>{"?" if h in weak_hours else ""}</span></th>'
    for h in HOURS)
rows = []
for _, r in sel.iterrows():
    tds = "".join(cell(float(r[h])) for h in HOURS)
    tot, gap, hot = float(r["全天净流"]), float(r["最大缺口"]), str(r["最缺时段"])
    cls = "neg" if tot < 0 else "pos"
    rows.append(f'<tr><td class="sid {cls}">{r["station_id"]}</td><td class="cl">{int(r["cluster"])}</td>'
                f'{tds}<td class="tot {cls}">{tot:+.1f}</td><td class="hot">{hot}</td></tr>')

# ---------- 可靠性条形 ----------
bars = []
W, H, pl, pb = 900, 150, 34, 26
for i, h in enumerate(HOURS):
    r = rel[rel["hour"] == int(h[:2])]
    acc = float(r["direction_acc"].iloc[0]) if len(r) else 0
    x = pl + i * (W - pl - 10) / len(HOURS)
    bw = (W - pl - 10) / len(HOURS) * 0.62
    hh = (H - pb - 16) * min(acc, 100) / 100
    col = "#dc2626" if acc < 55 else ("#0d9488" if acc >= 70 else "#d97706")
    bars.append(f'<rect x="{x:.1f}" y="{H-pb-hh:.1f}" width="{bw:.1f}" height="{hh:.1f}" rx="2" fill="{col}" opacity="0.82"/>')
    bars.append(f'<text x="{x+bw/2:.1f}" y="{H-pb+14:.1f}" text-anchor="middle" font-size="9.5" fill="#77808f">{h[:2]}</text>')
yl = H - pb - (H - pb - 16) * 0.55
bars.append(f'<line x1="{pl-6}" y1="{yl:.1f}" x2="{W-6}" y2="{yl:.1f}" stroke="#b3bac6" stroke-dasharray="4 3"/>')
bars.append(f'<text x="{pl-8}" y="{yl+4:.1f}" text-anchor="end" font-size="9.5" fill="#77808f">55%</text>')
bars.append(f'<text x="{pl-8}" y="{H-pb-2:.1f}" text-anchor="end" font-size="9.5" fill="#77808f">0</text>')
bars.append(f'<text x="{pl-8}" y="{16}" text-anchor="end" font-size="9.5" fill="#77808f">100%</text>')

CSS = """
:root{--ink:#151a23;--mut:#6f7887;--line:#e6e8ee;--bg:#f7f8fa;--red:#dc2626;--teal:#0d9488;--soft:#f4f6f9}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-size:14px;line-height:1.7;
font-family:"Microsoft YaHei","PingFang SC",system-ui,sans-serif;-webkit-font-smoothing:antialiased}
.wrap{max-width:1120px;margin:0 auto;padding:44px 26px 70px}
.kicker{font-size:12px;letter-spacing:.14em;color:var(--red);font-weight:700}
h1{font-size:29px;margin:10px 0 12px;letter-spacing:-.3px}
.sub{color:var(--mut);font-size:13.5px}
.chips-top{display:flex;gap:8px;flex-wrap:wrap;margin-top:16px}
.chips-top span{background:#fff;border:1px solid var(--line);border-radius:999px;padding:4px 12px;font-size:12.5px;color:#48505d}
.chips-top b{color:var(--ink)}
h2{font-size:18px;margin:44px 0 4px;font-weight:800}
h2 em{font-style:normal;color:#b3bac6;font-size:12.5px;margin-right:8px;font-weight:700}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:18px}
.wcard{background:#fff;border:1px solid var(--line);border-radius:12px;padding:15px 16px;box-shadow:0 1px 2px rgba(20,25,35,.04)}
.wt{font-size:14.5px;font-weight:800;font-variant-numeric:tabular-nums}
.wnum{display:flex;align-items:baseline;gap:5px;margin:6px 0 9px;font-variant-numeric:tabular-nums}
.wnum b{font-size:20px}.wnum span{font-size:11.5px;color:var(--mut);margin-right:8px}
.wlab{font-size:11.5px;color:var(--mut);margin-top:7px}
.c{display:inline-flex;align-items:center;gap:5px;border-radius:6px;padding:1px 7px;margin:3px 4px 0 0;
font-size:12px;font-variant-numeric:tabular-nums;background:var(--soft);color:#3a4250}
.c i{font-style:normal;font-weight:700;font-size:11.5px}
.c.red i{color:var(--red)}.c.teal i{color:var(--teal)}
.red{color:var(--red)}.teal{color:var(--teal)}
table{width:100%;border-collapse:collapse;font-size:12.3px;font-variant-numeric:tabular-nums;margin-top:14px;background:#fff}
th{font-weight:700;color:#3d4553;padding:8px 4px;font-size:11.5px;border-bottom:1.5px solid #d8dce4;text-align:center;background:#fbfcfd}
th.weak{color:#9aa1ad}
th.weak span{color:var(--red);font-weight:800}
td{padding:5px 3px;text-align:center;border-bottom:1px solid #f0f2f6;color:#3a4250}
td.sid{text-align:left;font-weight:700;padding-left:8px;width:64px}
td.sid.neg{color:#b91c1c}td.sid.pos{color:#0f766e}
td.cl{color:#9aa1ad;width:36px}
td.tot{font-weight:700;width:56px}
td.hot{color:var(--mut);width:56px;font-size:11.5px}
td.z{color:#cdd2db}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:#5a6270;margin-top:10px;align-items:center}
.sw{width:26px;height:11px;border-radius:3px;display:inline-block;vertical-align:-1px;margin-right:5px}
.notes{background:#fff;border:1px solid var(--line);border-left:3px solid var(--teal);border-radius:4px 12px 12px 4px;
padding:16px 20px;margin-top:20px;font-size:13px;color:#3d4553}
.notes b{color:var(--ink)}
footer{margin-top:36px;padding-top:14px;border-top:1px solid var(--line);color:var(--mut);font-size:12px}
@media(max-width:860px){.grid{grid-template-columns:1fr}.wrap{padding:28px 14px 50px}h1{font-size:23px}}
"""

HTML = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>全天调度看板 {date}</title><style>{CSS}</style></head><body><div class="wrap">
<div class="kicker">DISPATCH BOARD · 全天滚动预测</div>
<h1>今天哪个站、几点、缺几辆</h1>
<div class="sub">预测日 <b>{date}</b> · 借还双通道 LSTM · 逐小时滚动预测覆盖 00:00–23:00 · 119 个站点</div>
<div class="chips-top">
<span>需补合计 <b class="red">{int(a.loc[a.pred_net_3h<0,'need_bikes'].sum())}</b> 辆</span>
<span>可清运合计 <b class="teal">{int(a.loc[a.pred_net_3h>0,'need_bikes'].sum())}</b> 辆</span>
<span>最大单站缺口 <b>{sel['最大缺口'].min():+.1f}</b> 辆（{sel.loc[sel['最大缺口'].idxmin(),'station_id']} 站）</span>
<span>窗口 3 小时 / 锚点逐小时滚动 · 同一小时取 3 次重叠预测平均</span>
</div>

<h2><em>01</em>六个窗口的派车要点</h2>
<div class="grid">{''.join(cards)}</div>

<h2><em>02</em>站点 × 小时 净流矩阵</h2>
<div class="sub" style="margin-bottom:2px">取预测波动最大的 26 个站。负值＝净流出（要补车），正值＝净流入（要清运）。带 <b class="red">?</b> 的小时方向准确率低于 55%，只看方向、不按辆数派车。</div>
<table>
<tr><th style="text-align:left;padding-left:8px">站点</th><th>片区</th>{hdr}<th>全天</th><th>最缺时段</th></tr>
{''.join(rows)}
</table>
<div class="legend">
<span><i class="sw" style="background:rgba(220,38,38,.75)"></i>缺车（需补）</span>
<span><i class="sw" style="background:rgba(13,148,136,.75)"></i>积压（需清运）</span>
<span><i class="sw" style="background:#f4f6f9;border:1px solid #e6e8ee"></i>基本平衡</span>
<span>色深 ∝ |净流|，封顶 2.2 辆/小时</span>
</div>

<h2><em>03</em>每个小时到底能不能信</h2>
<div class="sub">验证集按目标小时统计的方向准确率（活跃片区）。这是决定"今晚要不要按这张表派车"的依据。</div>
<svg viewBox="0 0 {W} {H}" width="100%" role="img">{''.join(bars)}</svg>

<div class="notes">
<b>怎么用：</b>06:00 与 14:00 出两版清单。跨窗口看同一站点——<b>3014 / 3042 早上缺、下午积压，3005 / 3030 正好相反</b>，说明这批车一天能跑两趟，不需要增加车队规模。
<br><b>边界：</b>输出是「净流量（变化量）」，不是「库存」。要升级成真正的缺车报警，必须接入站点桩位容量与分时库存快照。
</div>
<footer>数据：{date} 预测输出 · 模型：单层 LSTM(48)+LSTM(32)，24h 输入窗口，Huber 损失 · 方向准确率见第 3 节，20:00 后 <55% 仅供参考。</footer>
</div></body></html>"""

p = os.path.join(OUT, "全天调度看板.html")
open(p, "w", encoding="utf-8").write(HTML)
print("written", p, len(HTML))
