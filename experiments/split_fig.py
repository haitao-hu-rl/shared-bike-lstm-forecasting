# -*- coding: utf-8 -*-
# --- 产物统一输出到项目根 outputs/ ---
from pathlib import Path as _Path
_OUT_DIR = _Path(__file__).resolve().parents[1] / "outputs"
_OUT_DIR.mkdir(exist_ok=True)
def _out(name):
    return str(_OUT_DIR / name)

import json, os
d = json.load(open(_out("split.json"), encoding="utf-8"))
top = d["top"]
order = sorted(top, key=lambda k: -top[k]["wd"])
COL = {"3014": "#0d9488", "3022": "#2563eb", "3082": "#7c3aed",
       "3066": "#ea580c", "3042": "#c026d3", "3068": "#b45309"}
GREY = "#dde1e8"
W, H = 680, 330
X0, BW = 40, 392

dep_txt = sum(top[k]["dep"] for k in top) / (1 - d["rest_wd"])
arr_txt = sum(top[k]["arr"] for k in top) / (1 - d["rest_wa"])

def bar(y, key, total, unit):
    g, x = [], X0
    for k in order:
        w = top[k][key] * BW
        g.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="32" fill="{COL[k]}" opacity="0.88"/>')
        if w > 44:
            g.append(f'<text x="{x+w/2:.1f}" y="{y+20}" text-anchor="middle" font-size="11.5" '
                     f'fill="#fff" font-weight="700">{k}</text>')
        x += w
        # 段间白线
        g.append(f'<line x1="{x:.1f}" y1="{y}" x2="{x:.1f}" y2="{y+32}" stroke="#fff" stroke-width="1.5"/>')
    w = d["rest_" + key] * BW
    g.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="32" fill="{GREY}"/>')
    g.append(f'<text x="{x+w/2:.1f}" y="{y+20}" text-anchor="middle" font-size="11.5" fill="#5a6270">'
             f'其他 {len(top and [])+23} 站 {d["rest_"+key]*100:.0f}%</text>')
    g.append(f'<text x="{X0}" y="{y-8}" font-size="12" fill="#2a313d" font-weight="700">'
             f'{unit} = {total:.2f} 辆　→ 按 29 个站的份额切开</text>')
    return "".join(g)

rows = []
for i, k in enumerate(order):
    y = 78 + i * 21
    rows.append(f'<rect x="452" y="{y-9}" width="8" height="8" rx="2" fill="{COL[k]}"/>')
    rows.append(f'<text x="466" y="{y}" font-size="11.5" fill="#2a313d" font-weight="700">{k}</text>')
    rows.append(f'<text x="534" y="{y}" font-size="11.5" fill="#6f7887" text-anchor="end">{top[k]["wd"]*100:.1f}%</text>')
    rows.append(f'<text x="602" y="{y}" font-size="11.5" fill="#6f7887" text-anchor="end">{top[k]["wa"]*100:.1f}%</text>')
    net = top[k]["net"]
    rows.append(f'<text x="662" y="{y}" font-size="11.5" text-anchor="end" font-weight="700" '
                f'fill="{"#dc2626" if net < 0 else "#0d9488"}">{net:+.2f}</text>')

head = (f'<text x="452" y="62" font-size="11.5" fill="#9aa1ad">站</text>'
        f'<text x="534" y="62" font-size="11.5" fill="#9aa1ad" text-anchor="end">借出占比</text>'
        f'<text x="602" y="62" font-size="11.5" fill="#9aa1ad" text-anchor="end">归还占比</text>'
        f'<text x="662" y="62" font-size="11.5" fill="#9aa1ad" text-anchor="end">净流</text>')

s = []
s.append(f'<text x="{X0}" y="24" font-size="12.5" fill="#6f7887">'
         f'片区 0 · 09-22 07:00 · 同一个总量，按「借出份额」和「归还份额」各切一次</text>')
s.append(bar(52, "wd", dep_txt, "借出 dep"))
s.append(bar(120, "wa", arr_txt, "归还 arr"))
s.append(head); s.append("".join(rows))
s.append(f'<line x1="{X0}" y1="222" x2="640" y2="222" stroke="#e6e8ee"/>')
s.append(f'<text x="{X0}" y="248" font-size="12" fill="#2a313d">'
         f'3014 净流 = {arr_txt:.2f}×15.7% − {dep_txt:.2f}×19.6% = 1.33 − 2.43 = '
         f'<tspan fill="#dc2626" font-weight="700">−1.10 辆</tspan></text>')
s.append(f'<text x="{X0}" y="272" font-size="11.5" fill="#6f7787">'
         f'两套份额各自求和都等于 100%，所以 29 个站的净流相加 = 片区净流（−3.94），一个数都不会多</text>')
s.append(f'<text x="{X0}" y="294" font-size="11.5" fill="#9aa1ad">'
         f'注意 3022：借出占 10.6%、归还只占 2.3% —— 缺口不是「车多车少」，是「借还结构不对称」</text>')

svg = (f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" xmlns="http://www.w3.org/2000/svg">'
       f'<title>片区总量按站点份额拆分</title>'
       f'<desc>两条 100% 堆叠条分别是借出与归还的份额，逐站相减得到净流。</desc>'
       f'{"".join(s)}</svg>')
p = _out("split.svg")
open(p, "w", encoding="utf-8").write(svg)
print("ok", len(svg))
