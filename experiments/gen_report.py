# -*- coding: utf-8 -*-
import pandas as pd, numpy as np, os, json
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
OUT = str(_ROOT / "docs")            # 报告 HTML 与配图输出
os.makedirs(os.path.join(OUT, "fig"), exist_ok=True)
df = pd.read_excel(str(_ROOT / "data" / "共享单车数据十万条1.xls"))
df['start'] = pd.to_datetime(df['开始时间'], format='%m/%d/%Y %H:%M')
df['date'] = df['start'].dt.date; df['hour'] = df['start'].dt.hour; df['dow'] = df['start'].dt.dayofweek

# ---------- 数据 ----------
daily = df.groupby('date').size().astype(float)
dvals = daily.values; ddates = [d.strftime('%m/%d') for d in daily.index]
ma7 = pd.Series(dvals).rolling(7).mean().values
hourly = df.groupby('hour').size(); hpct = (hourly/hourly.sum()*100)
wd = df[df['dow']<5].groupby('hour').size(); wdpct = wd/wd.sum()*100
we = df[df['dow']>=5].groupby('hour').size(); wepct = we/we.sum()*100
dep = df.groupby('开始站点id').size().sort_values(ascending=False)
cum = (dep.cumsum()/dep.sum()*100).values

from sklearn.cluster import KMeans
loc = df.groupby('开始站点id')[['开始经度','开始维度']].median().dropna()
lab = pd.Series(KMeans(6, n_init=10, random_state=42).fit(loc.values).labels_, index=loc.index)
df['c'] = df['开始站点id'].map(lab); df['ce'] = df['结束站点'].map(lab)
w = df[df['dow']<5]; nd = w['date'].nunique()
net = {}
for c in [0,4]:
    d_ = w[w['c']==c].groupby('hour').size().reindex(range(24),fill_value=0)/nd
    a_ = w[w['ce']==c].groupby('hour').size().reindex(range(24),fill_value=0)/nd
    net[c] = (a_-d_).values

# ---------- SVG 工具 ----------
C = dict(ink='#151a23', mut='#77808f', line='#e6e8ee', accent='#2563eb', teal='#0d9488',
         orange='#ea580c', red='#dc2626', soft='#f4f6f9')
def esc(s): return str(s)

def daily_chart():
    W,H,pl,pr,pt,pb = 880,240,52,16,18,34
    iw, ih = W-pl-pr, H-pt-pb
    ymax = 1000
    X = lambda i: pl + iw*i/(len(dvals)-1)
    Y = lambda v: pt + ih*(1-v/ymax)
    g = []
    for v in [0,250,500,750,1000]:
        g.append(f'<line x1="{pl}" y1="{Y(v):.1f}" x2="{W-pr}" y2="{Y(v):.1f}" stroke="{C["line"]}"/>')
        g.append(f'<text x="{pl-8}" y="{Y(v)+4:.1f}" text-anchor="end" font-size="10" fill="{C["mut"]}">{v}</text>')
    area = f'{pl},{Y(dvals[0]):.1f} ' + ' '.join(f'{X(i):.1f},{Y(v):.1f}' for i,v in enumerate(dvals)) + f' {X(len(dvals)-1):.1f},{Y(0):.1f} {pl},{Y(0):.1f}'
    g.append(f'<polygon points="{area}" fill="{C["accent"]}" opacity="0.08"/>')
    g.append('<polyline points="' + ' '.join(f'{X(i):.1f},{Y(v):.1f}' for i,v in enumerate(dvals)) + f'" fill="none" stroke="{C["accent"]}" stroke-width="1.6" opacity="0.55"/>')
    g.append('<polyline points="' + ' '.join(f'{X(i):.1f},{Y(v):.1f}' for i,v in enumerate(ma7) if not np.isnan(v)) + f'" fill="none" stroke="{C["teal"]}" stroke-width="2.4"/>')
    for i,t in enumerate(ddates):
        if i % 14 == 0:
            g.append(f'<text x="{X(i):.1f}" y="{H-pb+16}" text-anchor="middle" font-size="10" fill="{C["mut"]}">{t}</text>')
    # 标注趋势
    g.append(f'<line x1="{X(0):.1f}" y1="{Y(544):.1f}" x2="{X(len(dvals)-8):.1f}" y2="{Y(768):.1f}" stroke="{C["orange"]}" stroke-width="1.4" stroke-dasharray="5 4"/>')
    g.append(f'<text x="{pl+10}" y="{Y(980):.1f}" font-size="11" fill="{C["orange"]}">趋势 +1.92 单/天（首周 544 → 末周 768）</text>')
    g.append(f'<text x="{W-pr-4}" y="{pt+10}" text-anchor="end" font-size="11" fill="{C["teal"]}">7 日均线</text>')
    return f'<svg viewBox="0 0 {W} {H}" width="100%" role="img">{"".join(g)}</svg>'

def hourly_chart():
    W,H,pl,pr,pt,pb = 880,250,46,18,20,32
    iw,ih = W-pl-pr,H-pt-pb; ymax=10
    X = lambda h: pl + iw*h/23
    Y = lambda v: pt + ih*(1-v/ymax)
    g=[]
    for v in [0,2.5,5,7.5,10]:
        g.append(f'<line x1="{pl}" y1="{Y(v):.1f}" x2="{W-pr}" y2="{Y(v):.1f}" stroke="{C["line"]}"/>')
        g.append(f'<text x="{pl-8}" y="{Y(v)+4:.1f}" text-anchor="end" font-size="10" fill="{C["mut"]}">{v}%</text>')
    for h in range(0,24,3):
        g.append(f'<text x="{X(h):.1f}" y="{H-pb+16}" text-anchor="middle" font-size="10" fill="{C["mut"]}">{h}:00</text>')
    # 峰值高亮
    g.append(f'<rect x="{X(11.5):.1f}" y="{pt}" width="{X(13.5)-X(11.5):.1f}" height="{ih}" fill="{C["teal"]}" opacity="0.06"/>')
    g.append(f'<rect x="{X(16.5):.1f}" y="{pt}" width="{X(19.5)-X(16.5):.1f}" height="{ih}" fill="{C["orange"]}" opacity="0.07"/>')
    g.append('<polyline points="' + ' '.join(f'{X(h):.1f},{Y(wdpct[h]):.1f}' for h in range(24)) + f'" fill="none" stroke="{C["accent"]}" stroke-width="2.4"/>')
    g.append('<polyline points="' + ' '.join(f'{X(h):.1f},{Y(wepct[h]):.1f}' for h in range(24)) + f'" fill="none" stroke="{C["teal"]}" stroke-width="2.4" stroke-dasharray="6 3"/>')
    for h in range(24):
        g.append(f'<circle cx="{X(h):.1f}" cy="{Y(wdpct[h]):.1f}" r="2.2" fill="{C["accent"]}"/>')
    g.append(f'<text x="{X(12.5):.1f}" y="{pt+13}" text-anchor="middle" font-size="10.5" fill="{C["teal"]}">午峰</text>')
    g.append(f'<text x="{X(18):.1f}" y="{pt+13}" text-anchor="middle" font-size="10.5" fill="{C["orange"]}">晚峰</text>')
    g.append(f'<text x="{pl+8}" y="{pt+13}" font-size="11" fill="{C["accent"]}">—— 工作日</text>')
    g.append(f'<text x="{pl+8}" y="{pt+29}" font-size="11" fill="{C["teal"]}">---- 周末</text>')
    return f'<svg viewBox="0 0 {W} {H}" width="100%" role="img">{"".join(g)}</svg>'

def pareto_chart():
    W,H,pl,pr,pt,pb = 880,230,48,18,18,34
    iw,ih = W-pl-pr,H-pt-pb
    n=len(cum); ymax=100
    X = lambda i: pl + iw*i/(n-1)
    Y = lambda v: pt + ih*(1-v/ymax)
    g=[]
    for v in [0,25,50,75,100]:
        g.append(f'<line x1="{pl}" y1="{Y(v):.1f}" x2="{W-pr}" y2="{Y(v):.1f}" stroke="{C["line"]}"/>')
        g.append(f'<text x="{pl-8}" y="{Y(v)+4:.1f}" text-anchor="end" font-size="10" fill="{C["mut"]}">{v}%</text>')
    # 柱
    bw = iw/n*0.72
    vmax=dep.max()
    for i,v in enumerate(dep.values):
        hgt = ih*0.55*(v/vmax)
        g.append(f'<rect x="{X(i)-bw/2:.1f}" y="{pt+ih-hgt:.1f}" width="{bw:.1f}" height="{hgt:.1f}" fill="{C["mut"]}" opacity="0.22"/>')
    g.append('<polyline points="' + ' '.join(f'{X(i):.1f},{Y(v):.1f}' for i,v in enumerate(cum)) + f'" fill="none" stroke="{C["accent"]}" stroke-width="2.6"/>')
    for r,lab_,col in [(24,'25 站 = 50% 流量',C["orange"]),(57,'58 站 = 80%',C["teal"])]:
        g.append(f'<line x1="{X(r):.1f}" y1="{Y(0):.1f}" x2="{X(r):.1f}" y2="{Y(cum[r]):.1f}" stroke="{col}" stroke-width="1.4" stroke-dasharray="4 3"/>')
        g.append(f'<circle cx="{X(r):.1f}" cy="{Y(cum[r]):.1f}" r="4" fill="{col}"/>')
    g.append(f'<text x="{X(24)+10:.1f}" y="{Y(44):.1f}" font-size="11" fill="{C["orange"]}">25 站承担 50% 出发量</text>')
    g.append(f'<text x="{X(57)+10:.1f}" y="{Y(66):.1f}" font-size="11" fill="{C["teal"]}">58 站承担 80%</text>')
    g.append(f'<text x="{pl}" y="{H-pb+18}" font-size="10" fill="{C["mut"]}">站点按出发量降序（共 120 站）</text>')
    return f'<svg viewBox="0 0 {W} {H}" width="100%" role="img">{"".join(g)}</svg>'

def mirror_chart():
    W,H,pl,pr,pt,pb = 880,270,52,18,22,36
    iw,ih = W-pl-pr,H-pt-pb
    vals = np.concatenate([net[0],net[4]]); vmax = 4.4
    mid = pt + ih/2
    X = lambda h: pl + iw*(h+0.5)/24
    Y = lambda v: mid - (ih/2)*(v/vmax)
    g=[]
    g.append(f'<line x1="{pl}" y1="{mid}" x2="{W-pr}" y2="{mid}" stroke="#c9ced8" stroke-width="1.2"/>')
    for v in [-4,-2,0,2,4]:
        g.append(f'<line x1="{pl}" y1="{Y(v):.1f}" x2="{W-pr}" y2="{Y(v):.1f}" stroke="{C["line"]}" stroke-dasharray="3 4"/>')
        g.append(f'<text x="{pl-8}" y="{Y(v)+4:.1f}" text-anchor="end" font-size="10" fill="{C["mut"]}">{v:+.0f}</text>')
    for h in range(0,24,3):
        g.append(f'<text x="{X(h):.1f}" y="{H-pb+18}" text-anchor="middle" font-size="10" fill="{C["mut"]}">{h}:00</text>')
    bw = iw/24*0.3
    for h in range(24):
        for series,col,off in [(net[0],C["accent"],-bw/2-1),(net[4],C["orange"],bw/2+1)]:
            v = series[h]; y0=mid; y1=Y(v)
            g.append(f'<rect x="{X(h)+off-bw/2:.1f}" y="{min(y0,y1):.1f}" width="{bw:.1f}" height="{abs(y1-y0):.1f}" fill="{col}" opacity="0.85" rx="1"/>')
    g.append(f'<text x="{W-pr}" y="{pt-6}" text-anchor="end" font-size="11" fill="{C["mut"]}">纵轴：片区·单小时净流（到达 − 出发，整片合计，单位：辆）</text>')
    g.append(f'<text x="{X(12):.1f}" y="{pt+12}" text-anchor="middle" font-size="11" fill="{C["accent"]}">■ 簇 0（西部片区）</text>')
    g.append(f'<text x="{X(12):.1f}" y="{pt+28}" text-anchor="middle" font-size="11" fill="{C["orange"]}">■ 簇 4（东部片区）</text>')
    return f'<svg viewBox="0 0 {W} {H}" width="100%" role="img">{"".join(g)}</svg>'

svg_daily, svg_hour, svg_pareto, svg_mirror = daily_chart(), hourly_chart(), pareto_chart(), mirror_chart()
open(os.path.join(OUT,'fig','c1.svg'),'w',encoding='utf-8').write(svg_daily)
open(os.path.join(OUT,'fig','c2.svg'),'w',encoding='utf-8').write(svg_hour)
open(os.path.join(OUT,'fig','c3.svg'),'w',encoding='utf-8').write(svg_pareto)
open(os.path.join(OUT,'fig','c4.svg'),'w',encoding='utf-8').write(svg_mirror)

json.dump(dict(daily=len(daily), trips=int(len(df))), open(os.path.join(OUT,'fig','meta.json'),'w'))
print('charts ok', len(svg_daily), len(svg_hour))
