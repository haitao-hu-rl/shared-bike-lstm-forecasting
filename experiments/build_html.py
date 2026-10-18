# -*- coding: utf-8 -*-
import os
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
OUT = str(_ROOT / "docs")            # 报告 HTML 输出
FIG = str(_ROOT / "docs" / "fig")    # 配图目录（由 gen_report.py 生成）
rd = lambda f: open(os.path.join(FIG,f),encoding='utf-8').read()
C1,C2,C3,C4 = rd('c1.svg'), rd('c2.svg'), rd('c3.svg'), rd('c4.svg')

HTML = r"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>共享单车数据诊断与 LSTM 预测任务设计</title>
<style>
:root{--ink:#151a23;--mut:#6f7887;--line:#e6e8ee;--bg:#f7f8fa;--card:#fff;
--accent:#2563eb;--teal:#0d9488;--orange:#ea580c;--red:#dc2626;--soft:#f4f6f9}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font-family:"Microsoft YaHei","PingFang SC","Hiragino Sans GB",system-ui,-apple-system,sans-serif;
line-height:1.75;font-size:15px;-webkit-font-smoothing:antialiased}
.wrap{max-width:1040px;margin:0 auto;padding:48px 28px 80px}
header{margin-bottom:34px}
.kicker{font-size:12.5px;letter-spacing:.14em;color:var(--accent);font-weight:700;text-transform:uppercase}
h1{font-size:33px;line-height:1.32;margin:12px 0 14px;letter-spacing:-.4px;font-weight:800}
.sub{color:var(--mut);font-size:14.5px}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:20px}
.chip{background:#fff;border:1px solid var(--line);color:#48505d;border-radius:999px;
padding:5px 13px;font-size:12.5px;font-variant-numeric:tabular-nums}
.chip b{color:var(--ink);font-weight:700}
.lead{background:#fff;border:1px solid var(--line);border-left:3px solid var(--teal);
border-radius:4px 14px 14px 4px;padding:22px 26px;margin:30px 0 12px;
box-shadow:0 1px 2px rgba(20,25,35,.04)}
.lead .lab{font-size:12px;font-weight:700;color:var(--teal);letter-spacing:.08em;margin-bottom:8px}
.lead p{margin:0;font-size:15.5px;color:#2a313d}
.lead b{background:linear-gradient(transparent 62%,rgba(37,99,235,.16) 62%);font-weight:700}
h2{font-size:21px;margin:52px 0 6px;font-weight:800;letter-spacing:-.2px;
display:flex;align-items:baseline;gap:10px}
h2 span.n{font-size:12.5px;color:#b3bac6;font-weight:700;font-variant-numeric:tabular-nums}
h3{font-size:16px;margin:30px 0 8px;font-weight:700}
p{margin:10px 0;color:#333b48}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:22px 24px;
margin:18px 0;box-shadow:0 1px 2px rgba(20,25,35,.04)}
figure{margin:20px 0 6px}
figcaption{font-size:12.5px;color:var(--mut);margin-top:8px;line-height:1.6}
.fig-note{background:var(--soft);border-radius:8px;padding:11px 14px;font-size:13px;
color:#4a5260;margin-top:10px}
table{width:100%;border-collapse:collapse;font-size:13.4px;margin:14px 0;
font-variant-numeric:tabular-nums}
th{text-align:left;font-weight:700;color:#3d4553;border-bottom:1.5px solid #d8dce4;
padding:10px 10px;font-size:12.8px;letter-spacing:.02em;background:#fbfcfd}
td{padding:10px;border-bottom:1px solid var(--line);vertical-align:top;color:#3a4250}
tr:last-child td{border-bottom:none}
td.num,th.num{text-align:right}
.tag{display:inline-block;padding:1px 7px;border-radius:5px;font-size:11.5px;font-weight:700;
white-space:nowrap;line-height:1.6}
.p0{background:#eaf1ff;color:#1d4ed8}.p1{background:#e6f6f3;color:#0b7f72}
.p2{background:#f2f4f7;color:#5b6472}
.ok{color:#0b7f72;font-weight:700}.warn{color:#c2570b;font-weight:700}.bad{color:#c02626;font-weight:700}
.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:760px){.two{grid-template-columns:1fr}.wrap{padding:32px 16px 60px}h1{font-size:26px}}
.mini{background:var(--soft);border-radius:10px;padding:14px 16px}
.mini .t{font-size:12px;color:var(--mut);margin-bottom:4px}
.mini .v{font-size:22px;font-weight:800;letter-spacing:-.5px}
.mini .d{font-size:12.5px;color:#5a6270;margin-top:2px}
ul,ol{margin:10px 0;padding-left:22px}
li{margin:6px 0;color:#333b48}
code{background:#f0f2f6;border-radius:4px;padding:1px 5px;font-size:12.8px;
font-family:"JetBrains Mono",Consolas,monospace;color:#2c3441}
.step{display:flex;gap:14px;padding:14px 0;border-bottom:1px dashed var(--line)}
.step:last-child{border-bottom:none}
.step .k{flex:0 0 62px;font-size:12px;font-weight:800;color:var(--accent);
font-variant-numeric:tabular-nums;padding-top:2px}
.step .b{flex:1}.step .b b{font-size:14.5px}
.step .b p{margin:3px 0 0;font-size:13.4px;color:#4a5260}
.timeline{font-size:13.5px}
footer{margin-top:60px;padding-top:20px;border-top:1px solid var(--line);
color:var(--mut);font-size:12.5px}
.q{color:var(--accent);font-weight:700}
.hl{background:#fff8ed;border:1px solid #fde3c2;border-radius:10px;padding:14px 16px;font-size:13.5px;color:#7a4a12}
</style></head><body><div class="wrap">

<header>
<div class="kicker">Data Diagnosis · Demand Forecasting</div>
<h1>这份共享单车数据，能做哪些 LSTM 预测任务<br>来支撑调度决策？</h1>
<div class="sub">基于 <code>共享单车数据十万条1.xls</code> 的全量探查、时空结构分析与 3 组 LSTM 对照实验</div>
<div class="chips">
<span class="chip"><b>65,535</b> 条行程</span>
<span class="chip"><b>85</b> 天 · 2020/07/01–09/23</span>
<span class="chip"><b>120</b> 个站点</span>
<span class="chip"><b>1,297</b> 辆车</span>
<span class="chip"><b>2,021</b> 个小时</span>
</div>
</header>

<div class="lead">
<div class="lab">结论先行</div>
<p>这份数据最值得做的是 <b>「未来 1–6 小时、按片区／重点站分别预测借出量与归还量，再合成调拨缺口」</b>。
不建议把净流量直接当回归目标（实测方向准确率只有 <b>57%</b>，几乎等于抛硬币），也不建议对 120 个站逐站做小时级 LSTM——
<b>84.2%</b> 的「站点 × 小时」格子是 0 单。数据完全支撑「需求预测」，但在补齐库存快照与桩位容量之前，<b>支撑不了严格的库存预测</b>。</p>
</div>

<h2><span class="n">01</span>数据体检：它能回答什么，不能回答什么</h2>
<p>先看清数据口径。这张表是行程级流水（每行一次骑行），15 个字段，覆盖 2020 年 7 月 1 日到 9 月 23 日共 85 天，坐标落在北京（约 39.05°N, 116.24°E）。</p>

<div class="card">
<h3 style="margin-top:0">对建模最有用的 6 个字段</h3>
<table>
<tr><th style="width:19%">字段</th><th style="width:26%">含义</th><th>建模价值</th></tr>
<tr><td><code>开始时间</code> / <code>结束时间</code></td><td>精确到分钟</td><td>唯一的时间轴，可聚合到 15 分钟／小时／天；结束时间决定车辆"落地时刻"</td></tr>
<tr><td><code>开始站点id</code> / <code>结束站点</code></td><td>120 个站点</td><td>调度的空间单元，也是做「借／还」两条序列的骨架</td></tr>
<tr><td><code>骑行时间</code></td><td>分钟</td><td>推算在途车辆、预计归还时刻；中位数 11 分钟，P95 = 69 分钟</td></tr>
<tr><td><code>会员类型</code></td><td>月卡 66.9% / 单次 29.0% / 灵活卡 4.1%</td><td>月卡偏通勤（刚性、可预测），单次偏休闲（弹性、受天气影响大），分开建模能显著降噪</td></tr>
<tr><td><code>骑行路线类别</code></td><td>单程 85.8% / 往返 14.2%</td><td>往返骑行的车会回到原站，不产生空间失衡——直接决定净流量</td></tr>
<tr><td><code>车辆id</code></td><td>1,297 辆</td><td>可还原车辆轨迹链，估算站点库存与周转</td></tr>
</table>
</div>

<h3>7 个必须先处理的数据问题</h3>
<div class="card">
<table>
<tr><th style="width:26%">问题</th><th class="num" style="width:14%">规模</th><th>影响与处理建议</th></tr>
<tr><td>行数被 Excel 上限截断</td><td class="num warn">65,535 行</td><td>文件名叫"十万条"，但 <code>.xls</code> 单表上限 65,536 行，数据已被截断。请改用 <code>.csv</code> / <code>.xlsx</code> 交付全量，否则趋势与季节性判断会偏</td></tr>
<tr><td>经纬度列名互换</td><td class="num bad">全表</td><td><code>开始经度</code> 实为纬度（39.0x），<code>开始维度</code> 实为经度（116.2x）。直接拿去画图会把站点画到非洲，聚类/距离特征全部失效</td></tr>
<tr><td>坐标缺失</td><td class="num warn">58 / 731</td><td>起点缺 58 条、终点缺 731 条（约占 1.1%），空间特征需按站点 id 回填</td></tr>
<tr><td>重复记录</td><td class="num">4 个 trip_id</td><td>另有 13 条「同时段同站点同车」的疑似重复，建议按 trip_id 去重</td></tr>
<tr><td>站点口径不一致</td><td class="num">120 vs 122</td><td>终点出现 122 个站点，其中 2 个从未作为起点出现，建面板时需对齐成同一套站点索引</td></tr>
<tr><td>时长截断与异常</td><td class="num">1,291 条 &gt;3h</td><td>最大 1,440 分钟（=24h 上限，明显是未还车/计费兜底）。建模时截尾或剔除</td></tr>
<tr><td>缺少 4 类关键外部维度</td><td class="num bad">0 字段</td><td>无 <b>桩位容量</b>、无 <b>实时库存快照</b>、无 <b>天气</b>、无 <b>节假日</b>。这是本项目最大的能力边界，详见第 5 节</td></tr>
</table>
<div class="hl"><b>一个意外收获：</b>按车辆 id 串联行程后发现，<b>15.2%</b> 的行程起点 ≠ 该车上一程的终点。这部分「凭空位移」正是人工调拨（或记录缺失）的痕迹——虽然零星，但已足够作为「历史调拨发生点」的弱标签，用来校验预测出的缺口是否与真实调度行为吻合。</div>
</div>

<h2><span class="n">02</span>三个直接决定任务设计的事实</h2>

<h3>事实一：总量在持续爬坡，序列非平稳</h3>
<figure>__C1__
<figcaption>日均单量从首周 544 单升至末周 768 单，线性趋势 <b>+1.92 单/天</b>（7 月受疫情影响偏低，8–9 月恢复）。</figcaption>
<div class="fig-note"><b>建模含义：</b>直接喂原始计数，模型会把"7 月低、9 月高"当成规律。必须用差分、趋势项或按周同比归一化，否则上线后系统性低估。同时只用 85 天数据，跨月泛化能力有限，需留出最后 2–3 周做时间外推验证。</div>
</figure>

<h3>事实二：一天有两个峰，且工作日/周末形态不同</h3>
<figure>__C2__
<figcaption>工作日呈现"午峰（12–13 点）+ 晚峰（17–19 点）"的双峰，早高峰（8 点）只占 5.5%；周末峰值整体后移到 17–19 点。小时序列自相关 lag-24h = <b>0.857</b>，日周期极强。</figcaption>
<div class="fig-note"><b>建模含义：</b>这是典型的「商业区／休闲+通勤混合」形态，不是纯通勤城市。窗口长度取 <b>24 小时</b>即可覆盖一个完整周期；<code>hour</code> 的周期编码（sin/cos）比 one-hot 更省参数且能外推。周末必须单独建一层特征，不能与工作日混训。</div>
</figure>

<h3>事实三：流量高度集中在少数站点，且早晚方向恰好相反</h3>
<figure>__C3__
<figcaption>120 个站点中，<b>25 个站承担 50%</b> 的出发量、58 个承担 80%；另有 13 个站日均出发不足 1 单。</figcaption>
</figure>
<figure>__C4__
<figcaption>按经纬度聚成 6 个片区后，簇 0 与簇 4 的净流量呈明显<b>镜像关系</b>：早高峰西部片区（簇 0）每小时净流出约 3.9 辆、东部片区（簇 4）净流入 3.6 辆；晚高峰完全反过来。</figcaption>
<div class="fig-note"><b>调度含义（也是最值钱的一条）：</b>这不是"随机缺车"，而是<b>可预测的潮汐</b>。同一批车早上从簇 0 流向簇 4，晚上再流回来——意味着调度不需要创造新供给，只需要<b>预判方向并提前把车摆到正确的一侧</b>。站点级也一样：3005 站早高峰 +1.7 辆/小时（积压），3042 与 3014 各 −1.2 辆/小时（缺车），晚高峰三者角色互换。</div>
</figure>

<h2><span class="n">03</span>推荐的 LSTM 任务清单</h2>
<p>按「对调度的价值 ÷ 数据支持度」排序。P0 是立刻能做且直接产出调拨动作的，P1 是补齐数据后价值很高的，P2 建议先放着。</p>

<div class="card">
<table>
<tr><th style="width:6%">优先</th><th style="width:19%">预测任务</th><th style="width:23%">预测目标（模型输出）</th><th style="width:13%">粒度</th><th style="width:25%">调度员怎么用</th><th style="width:9%">数据<br>支持度</th></tr>

<tr>
<td><span class="tag p0">P0</span></td>
<td><b>借还双通道需求预测</b><br><span style="color:#6f7887;font-size:12.5px">Dual-channel demand</span></td>
<td>每个片区／重点站，未来 1–6 小时<br>借出量 <code>dep</code> 与归还量 <code>arr</code> 两条序列，分别预测</td>
<td>1 小时 ×<br>6 片区 +<br>Top 25 站</td>
<td>生成「今日潮汐图」，是后面所有调拨决策的底座</td>
<td class="ok">强</td>
</tr>

<tr>
<td><span class="tag p0">P0</span></td>
<td><b>未来 3 小时累计净流（缺车量）</b><br><span style="color:#6f7887;font-size:12.5px">Net-flow / imbalance</span></td>
<td>站点级带符号的净流入量：<br><code>+Y</code> 需清运、<code>−X</code> 需补车</td>
<td>3 小时滚动<br>× 站点</td>
<td><b>直接换算成工单</b>："8 点前给 3042 补 6 辆"，不需要再二次解释</td>
<td class="ok">强</td>
</tr>

<tr>
<td><span class="tag p1">P1</span></td>
<td><b>缺车／满桩风险概率</b><br><span style="color:#6f7887;font-size:12.5px">Risk classification</span></td>
<td>未来 1–2 小时该站发生"无车可借／无桩可还"的概率（0–1）</td>
<td>1 小时<br>× 站点</td>
<td>给看板做红黄绿灯，调度员只处理高概率的 top-20，避免信息过载</td>
<td class="warn">中<br><span style="font-weight:400;font-size:11.5px">需库存快照</span></td>
</tr>

<tr>
<td><span class="tag p1">P1</span></td>
<td><b>区域／系统总量预测</b><br><span style="color:#6f7887;font-size:12.5px">Aggregate demand</span></td>
<td>未来 24 小时整城／分区的总单量曲线</td>
<td>1 小时<br>× 系统</td>
<td>排班与车辆检修计划、决定今天要不要加派一辆卡车</td>
<td class="ok">强<br><span style="font-weight:400;font-size:11.5px">已有 R²=0.705 基线</span></td>
</tr>

<tr>
<td><span class="tag p1">P1</span></td>
<td><b>骑行时长／还车时刻预测</b><br><span style="color:#6f7887;font-size:12.5px">Trip duration</span></td>
<td>单次行程的时长，进而推算"未来 30–60 分钟将有多少车落地、落在哪"</td>
<td>单次行程</td>
<td>把"在途车辆"变成可计算的库存增量，让补车量算得更准</td>
<td class="ok">中强</td>
</tr>

<tr>
<td><span class="tag p2">P2</span></td>
<td><b>异常需求检测</b><br><span style="color:#6f7887;font-size:12.5px">Anomaly detection</span></td>
<td>LSTM-AE 重构误差 → 异常分数</td>
<td>1 小时<br>× 站点</td>
<td>识别突发聚集（活动/演唱会）、故障站点、极端天气，触发临时调度</td>
<td class="warn">中</td>
</tr>

<tr>
<td><span class="tag p2">P2</span></td>
<td><b>OD 流向矩阵预测</b><br><span style="color:#6f7887;font-size:12.5px">OD / seq2seq</span></td>
<td>站点对的流量矩阵</td>
<td>OD 对</td>
<td>理论上能做精细路径规划，但见第 5 节：这份数据的 OD 太分散</td>
<td class="bad">弱</td>
</tr>
</table>
</div>

<div class="lead" style="border-left-color:var(--orange);margin-top:22px">
<div class="lab" style="color:var(--orange)">如果只做一件事</div>
<p>做 <b>P0 的两条</b>：先把「片区／重点站的借出、归还双通道预测」跑通，再把它合成「未来 3 小时累计净流」。这两步产出的就是调度员每天早上最需要的那张表——<b>哪个站、几点、缺几辆</b>。</p>
</div>

<h2><span class="n">04</span>关键设计：为什么不能直接把"净流量"当回归目标</h2>
<p>直觉上，调度关心的是「到达 − 出发」这个净流量，那就直接预测它。我按这个思路做了对照实验，结论是：<b>不行</b>。</p>

<div class="card">
<h3 style="margin-top:0">实验设置</h3>
<p style="font-size:13.5px;color:#4a5260">6 个空间片区（KMeans）· 小时粒度 · 输入窗口 24 小时 · 单层 LSTM(32) + Dropout · 按时间 70/15/15 切分 · 共 2,021 个小时。
「方向准确率」= 预测的净流正负号与实际一致的比例，这是调度最关心的指标（补车还是清运，方向错了一切都错）。</p>

<table>
<tr><th style="width:34%">建模方式</th><th class="num" style="width:16%">MAE（辆）</th><th class="num" style="width:18%">方向准确率</th><th>评价</th></tr>
<tr>
<td><b>A. 直接回归单小时净流</b><br><span style="color:#6f7887;font-size:12px">目标 = arr − dep</span></td>
<td class="num">2.13</td><td class="num warn">58.6%</td>
<td>MAE 看着不大，但<b>方向几乎靠猜</b>。因为两个各自有噪声的序列相减，信噪比被放大了一倍，而净流均值又贴近 0</td>
</tr>
<tr>
<td><b>B. 分别预测借出／归还，再相减</b><br><span style="color:#6f7887;font-size:12px">dep、arr 各一个模型</span></td>
<td class="num">2.36</td><td class="num">61.9%</td>
<td>方向略微改善，两条序列都有稳定的日周期，模型更容易学；代价是两次误差叠加</td>
</tr>
<tr style="background:#f7fbfa">
<td><b>C. 预测"未来 3 小时累计净流"</b><br><span style="color:#6f7887;font-size:12px">目标 = Σ( arr − dep )，滚动窗口</span></td>
<td class="num">3.58</td><td class="num ok">70.7%</td>
<td><b>推荐方案。</b>3 小时累计把随机波动平滑掉，方向准确率从 57% 拉到 71%，而"未来 3 小时缺几辆"正好是调度真正需要的决策单位</td>
</tr>
</table>
<p style="font-size:12.5px;color:#77808f;margin:8px 0 0">注：同一实验在数量级相近的簇 4 上复现——A 57.0% / B 61.2% / C 66.7%，结论一致；表中 MAE 为簇 0（小时均量约 10.5 单）的结果。受随机初始化影响，方向准确率在多次运行间有 ±2 个百分点波动。</p>
</div>

<h3>由此推导出的 5 条实现要点</h3>
<ol>
<li><b>目标是"带符号的车辆数"，不是"单量"。</b>模型输出直接写成「−6 辆（需补）」「+9 辆（需清）」，中间不要再转译。</li>
<li><b>先分解、再合成。</b>借出与归还分别建模，各自捕捉自己的周期；净流只在最后一步做减法。反过来做会丢掉信息。</li>
<li><b>用 3 小时累计而不是 1 小时瞬时。</b>这既是精度问题（71% vs 57%），也是业务问题——卡车调度本来就是按 2–3 小时一趟来排的。</li>
<li><b>损失函数用 MAE / Huber，不用 MSE。</b>调度损失是线性的（少一辆就少一辆），MSE 会被高峰大值主导，模型会为了拟合峰值牺牲平峰时段的准确性。</li>
<li><b>稀疏站点不要硬建模。</b>84.2% 的站点-小时是 0，逐站小时级 LSTM 学到的只会是"预测 0"。要么聚合到片区，要么只对 Top 25 站建模，要么把粒度放宽到 2–3 小时。</li>
</ol>

<h2><span class="n">05</span>不建议做的三件事</h2>
<div class="card">
<h3 style="margin-top:0;color:#c02626">✕ 对 120 个站逐站做小时级 LSTM</h3>
<p style="font-size:13.5px">站点-小时面板共 248,880 个格子，其中 <b>84.2% 是 0 单</b>，单格期望只有 0.26 单；中位数站点一天出发 4 单，13 个站日均不到 1 单。在这么稀疏的序列上，LSTM 只能学会输出 0，验证集 MAE 看起来"极低"但毫无信息量（实验中近零流片区的方向准确率 95% 就是这么来的——因为真实值几乎恒为 0）。<b>替代方案：</b>聚合成 6 片区，或只取日均 ≥10 单的 25 个站。</p>

<h3 style="color:#c02626">✕ 端到端 OD 矩阵 / 站点对预测</h3>
<p style="font-size:13.5px">数据里有 <b>4,312 个 OD 对</b>，而 top 10 的 OD 对只占 <b>4.4%</b> 的流量——分布极其扁平。除往返骑行（如 4153→4153 共 536 单）外，绝大多数站点对在 85 天里只有个位数样本。<b>替代方案：</b>先做片区级净流，等缺口算出来之后，再用一趟多站路径规划（VRP）解决"从哪搬到哪"，不必让 LSTM 学 OD。</p>

<h3 style="color:#c02626">✕ 在缺库存数据的前提下承诺"库存预测"</h3>
<p style="font-size:13.5px">这份数据只有行程流水，没有<b>实时库存快照</b>，也没有<b>桩位容量</b>。没有这两个量，"缺车"就无法被定义——净流出 −10 辆，对一个容量 50 的站可能完全没事，对容量 15 的站早就空了。<b>替代方案：</b>现阶段用"累计净流漂移"作为缺车风险的代理指标（如 3068 站 85 天累计净流出 1,117 辆，是长期净流失站），并把补齐这两类数据列为第一优先。</p>
</div>

<h2><span class="n">06</span>落地路线：从数据到一张调度工单</h2>
<div class="card">
<div class="step"><div class="k">STEP 0</div><div class="b"><b>补数据（2 周，最高优先）</b>
<p>① 换成 <code>.csv</code> 全量导出，修掉经纬度列名互换；② 接入<b>站点桩位容量</b>与<b>分时库存快照</b>（哪怕是每 15 分钟一次的抓取）；③ 接入<b>天气</b>与<b>节假日日历</b>；④ 导出历史<b>调拨工单</b>，这是最理想的监督信号与效果评估基准。</p></div></div>

<div class="step"><div class="k">STEP 1</div><div class="b"><b>片区级双通道基线（2–3 周）</b>
<p>6 片区 × 1 小时，LSTM 分别预测 <code>dep</code> / <code>arr</code>，合成 3 小时累计净流。目标：方向准确率 ≥ 70%，并做 7 天滚动回测（walk-forward），而不是一次性留出。</p></div></div>

<div class="step"><div class="k">STEP 2</div><div class="b"><b>下沉到站点 + 风险分级（3–4 周）</b>
<p>用片区预测做总量约束，站点级做分配（top-down 分解，避免 120 个独立模型）。结合库存快照转成「缺车概率」，输出红黄绿灯。</p></div></div>

<div class="step"><div class="k">STEP 3</div><div class="b"><b>接进调度流程 + A/B（持续）</b>
<p>每天 06:30 与 14:30 各出一版缺口清单，接入现有派车系统。用「缺车时长」「订单满足率」「调拨里程」三个指标对比人工排班，而不是看 R²。</p></div></div>
</div>

<h3>一天怎么用：把模型输出翻译成调度动作</h3>
<div class="card" style="background:#fbfcfe">
<table>
<tr><th style="width:14%">时间</th><th style="width:30%">模型输出</th><th>调度动作</th></tr>
<tr><td><b>06:30</b></td><td>3042 站 7–10 点预计净流出 <b>−1.2 辆/小时</b>（合计 −3.6）；3014 同向 −1.2/h；3005 站净流入 <b>+1.7/h</b>（将积压 5 辆）</td><td>8 点前从 3005、3030 调 6 辆车至 3042、3014。<b>注意这不是凭空补车，而是把早高峰"多出来"的车搬到"缺"的那一侧</b></td></tr>
<tr><td><b>14:30</b></td><td>方向反转预警：3005 站 17–19 点转为净流出，3042 转净流入</td><td>16:30 前反向回补 3005，避免晚高峰无车可借。同一批车一天跑两趟，无需增加车队规模</td></tr>
</table>
<p style="font-size:13px;color:#6f7887;margin-bottom:0">这正是第 2 节"镜像潮汐"的直接应用：<b>预测的价值不在于知道总共有多少单，而在于知道车会往哪边跑、什么时候跑</b>。</p>
</div>

<h3>评估指标：别用 R²</h3>
<div class="two">
<div class="mini"><div class="t">模型层</div><div class="v" style="font-size:15px;line-height:1.9">MAE（辆）· 方向准确率 · 3h 累计误差</div><div class="d">方向准确率是硬指标，低于 65% 的模型不该上线</div></div>
<div class="mini"><div class="t">业务层</div><div class="v" style="font-size:15px;line-height:1.9">缺车时长 · 满足率 · 调拨里程/趟次</div><div class="d">同样的 MAE，可能对应完全不同的调度成本</div></div>
</div>
<p style="font-size:13px;color:#6f7887">你现有 notebook 里系统级模型的 <b>R² = 0.705 / RMSE = 14.92</b> 是一个合理的起点，但 R² 对调度没有直接含义——它只说明曲线形状拟合得不错。建议把它降级为"排班辅助"，把主战场换到站点级的方向准确率上。</p>

<h2><span class="n">07</span>随报告交付的代码</h2>
<div class="card">
<p style="margin-top:0"><code>D:/Codes/Bike/lstm_dispatch_baseline.py</code>：把 P0 方案落成可直接运行的基线，包含</p>
<ul style="margin-bottom:0">
<li>数据清洗（去重、经纬度换列修正、时长截尾、时间解析）</li>
<li>空间聚类 + 站点-小时面板构建（含缺失补 0）</li>
<li>借／还双通道 LSTM + 3 小时累计净流合成</li>
<li>与 persistence、seasonal-naive、滑动均值三个基线的对比评估</li>
<li>导出 <code>dispatch_alert.csv</code>：按站点列出未来 3 小时缺车/积压量与优先级</li>
</ul>

<h3>实跑输出（锚点 2020-09-22 06:00，预测 07:00–10:00）</h3>
<p style="font-size:13.5px;color:#4a5260;margin-top:0">为验证模型，用 9 月 22 日早晨做了一次真实预测。结果与第 2 节的潮汐结论独立吻合：<b>3042、3014 需要补车，3005、3030、3007 需要清运</b>——完全是早高峰应有的方向。</p>
<table>
<tr><th style="width:14%">站点</th><th style="width:10%">片区</th><th class="num" style="width:16%">预测净流 3h</th><th class="num" style="width:12%">需调车辆</th><th style="width:12%">优先级</th><th>动作</th></tr>
<tr><td><b>3042</b></td><td>0</td><td class="num" style="color:#c02626">−4.27</td><td class="num">4</td><td>高</td><td>早高峰前补车</td></tr>
<tr><td><b>3014</b></td><td>0</td><td class="num" style="color:#c02626">−4.00</td><td class="num">4</td><td>高</td><td>早高峰前补车</td></tr>
<tr><td><b>3055</b></td><td>4</td><td class="num" style="color:#c02626">−2.79</td><td class="num">3</td><td>高</td><td>早高峰前补车</td></tr>
<tr><td>3067</td><td>4</td><td class="num" style="color:#c02626">−2.51</td><td class="num">3</td><td>高</td><td>早高峰前补车</td></tr>
<tr style="border-top:1.5px solid #e6e8ee"><td><b>3005</b></td><td>4</td><td class="num" style="color:#0b7f72">+6.93</td><td class="num">7</td><td><span class="bad">紧急</span></td><td>尽快清运至 3042/3014</td></tr>
<tr><td>3030</td><td>0</td><td class="num" style="color:#0b7f72">+3.47</td><td class="num">3</td><td>高</td><td>清运</td></tr>
<tr><td>3007</td><td>4</td><td class="num" style="color:#0b7f72">+2.88</td><td class="num">3</td><td>高</td><td>清运</td></tr>
</table>
<p style="font-size:12.8px;color:#6f7887;margin-bottom:0">脚本实测：活跃片区（小时均量 ≥5 单）MAE <b>2.68 辆</b>、方向准确率 <b>67.8%</b>；同期 seasonal-naive 基线在片区 0 上方向准确率仅 45.7%。</p>
</div>

<footer>
数据源：<code>共享单车数据十万条1.xls</code>（65,535 条行程，2020-07-01 至 2020-09-23）·
实验环境：TensorFlow 2.21 / 单层 LSTM(32) · 所有指标为时间外推验证集结果<br>
本报告由数据分析自动生成，图表中的数值均来自对该文件的真实聚合计算。
</footer>
</div></body></html>
"""

HTML = HTML.replace('__C1__',C1).replace('__C2__',C2).replace('__C3__',C3).replace('__C4__',C4)
p = os.path.join(OUT,'共享单车_LSTM调度预测任务分析.html')
open(p,'w',encoding='utf-8').write(HTML)
print('written', p, len(HTML))
