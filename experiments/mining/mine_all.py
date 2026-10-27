"""
MVP 10-run 数据的二次挖掘（纯只读，不训练、不重新前向）

数据源：
  1. runs/slice/*/trajectory.jsonl   （25 个 eval 点 x 15 跳 + 该点 loss）
  2. runs/slice/*/train.log          （每 200 步 loss/lr/H_max/sps）
  3. experiments/u_shape_probe/*/logit_lens.npz（2000 样本 x 33 层 x 16 节点）

分析块：
  T1  U 形折叠检验：acc 是否只由 h_eff = min(H, 16-H) 决定
  T2  跨 run 每跳方差
  T3  loss 与能力（frontier）的关系
  T4  课程 H_max 到达时刻 vs frontier 跃升时刻
  T5  各跳首次达标所需步数
  T6  各 H 的答案 token 形态分布（纯数据生成，非训练）→ 首位红利覆盖率的理论上限
  T7  对称跳对（h, 16-h）的逐 run 差异
  T8  lens 重挖：ambiguous 字段、逐层最大概率、中间层预测漂移

用法：.venv/bin/python experiments/mining/mine_all.py
"""
import glob
import json
import math
import os
import re
import statistics as st
from collections import defaultdict

HOP = list(range(1, 16))


def eff(h):
    """有效组合深度：正向 h 步 或 反向 16-h 步，取小"""
    return min(h, 16 - h)


def load_traj(name):
    p = f"runs/slice/{name}/trajectory.jsonl"
    rows = [json.loads(l) for l in open(p)]
    seen = {}
    for r in rows:          # s42 有 step=68000 重复行，按 step 去重取后者
        seen[r["step"]] = r
    return [seen[s] for s in sorted(seen)]


def load_log(name):
    p = f"runs/slice/{name}/train.log"
    steps, evals = [], []
    for l in open(p, errors="ignore"):
        m = re.match(r"\s*step=\s*(\d+)\s+loss=([\d.]+)\s+lr=([\d.e+-]+)\s+H_max=\s*(\d+)\s+sps=([\d.]+)\s+elapsed=([\d.]+)min", l)
        if m:
            steps.append(dict(step=int(m.group(1)), loss=float(m.group(2)),
                              lr=float(m.group(3)), hmax=int(m.group(4)),
                              sps=float(m.group(5)), el=float(m.group(6))))
            continue
        m = re.match(r"\s*EVAL step=(\d+)\s+H=1: ([\d.]+)\s+H=7: ([\d.]+)\s+H=15: ([\d.]+)\s+H_sol@0\.8=(\d+)", l)
        if m:
            evals.append(dict(step=int(m.group(1)), h1=float(m.group(2)),
                              h7=float(m.group(3)), h15=float(m.group(4)),
                              hsol=int(m.group(5))))
    return steps, evals


def frontier(p):
    f = 0
    for h in map(str, HOP):
        if p[h] >= 0.8:
            f = int(h)
        else:
            break
    return f


RUNS = sorted(os.path.basename(d.rstrip("/")) for d in glob.glob("runs/slice/*/"))
# 只保留有 trajectory 的（s45 是中止的空目录）
RUNS = [n for n in RUNS if os.path.exists(f"runs/slice/{n}/trajectory.jsonl")]
LOOPED = [n for n in RUNS if n.startswith("ouro")]
UNTIED = [n for n in RUNS if n.startswith("deep")]

out = []
def P(*a):
    s = " ".join(str(x) for x in a)
    out.append(s)
    print(s)


# ---------------------------------------------------------------- T1 U 形折叠
P("=" * 78)
P("T1  U 形折叠检验：acc 是否只由 h_eff = min(H,16-H) 决定")
P("=" * 78)
P("思路：若 pi^15 = pi^-1，则 H 与 16-H 是同一道题，难度应只由 h_eff 决定。")
P("对每个 run 的末点，把 15 个 hop 按 h_eff 归组，看组内（不同 H）差异 vs 组间差异。")
P("")
P(f"{'run':16s} {'组内std均值':>10s} {'组间std':>8s} {'比值':>6s}   单调违例率")
for n in RUNS:
    rows = load_traj(n)
    acc = {h: rows[-1]["per_hop_acc"][str(h)] for h in HOP}
    buckets = defaultdict(list)
    for h in HOP:
        buckets[eff(h)].append(acc[h])
    # 组内 std：只有一个成员的组（h_eff=8 时 H=8 单成员；h_eff=1..7 各 2 成员）
    within = [st.pstdev(v) for v in buckets.values() if len(v) > 1]
    means = [st.mean(v) for v in buckets.values()]
    between = st.pstdev(means)
    # 单调违例：eff 更小却更差
    viol = tot = 0
    for i in HOP:
        for j in HOP:
            if eff(i) < eff(j):
                tot += 1
                if acc[i] < acc[j] - 1e-9:
                    viol += 1
    ratio = (st.mean(within) / between) if between > 1e-9 else float("nan")
    P(f"{n:16s} {st.mean(within):10.3f} {between:8.3f} {ratio:6.2f}   {viol}/{tot} = {viol/tot:.2f}")

P("")
P("--- 汇总：全部 run 全部 eval 点（10x25=250 个时刻）按 h_eff 池化 ---")
pool = defaultdict(list)
for n in RUNS:
    for r in load_traj(n):
        for h in HOP:
            pool[eff(h)].append(r["per_hop_acc"][str(h)])
P(f"{'h_eff':>5s} {'n':>5s} {'mean':>7s} {'std':>6s} {'min':>6s} {'max':>6s}   成员H")
for k in sorted(pool):
    v = pool[k]
    members = [h for h in HOP if eff(h) == k]
    P(f"{k:5d} {len(v):5d} {st.mean(v):7.3f} {st.pstdev(v):6.3f} {min(v):6.3f} {max(v):6.3f}   {members}")
P("")
P("（若 U 形成立：同一 h_eff 的两个 H 应落在同一水平；不同 h_eff 之间应有明显梯度）")


# ------------------------------------------------------------ T2 跨 run 方差
P("")
P("=" * 78)
P("T2  跨 run 每跳方差（末点）")
P("=" * 78)
P(f"{'hop':>4s} {'h_eff':>6s} " + " ".join(f"{n.split('_')[0][:9]:>10s}" for n in RUNS) + f" {'mean':>6s} {'std':>6s}")
for h in HOP:
    vals = [load_traj(n)[-1]["per_hop_acc"][str(h)] for n in RUNS]
    P(f"{h:4d} {eff(h):6d} " + " ".join(f"{v:10.3f}" for v in vals) + f" {st.mean(vals):6.3f} {st.pstdev(vals):6.3f}")


# ------------------------------------------------- T3 loss 与能力 / T7 对称对
P("")
P("=" * 78)
P("T3  loss 与能力的关系：末点 loss vs frontier；以及 达成 frontier 时刻的 loss")
P("=" * 78)
P(f"{'run':16s} {'frontier':>8s} {'末loss':>8s} {'loss@首达15':>12s} {'末loss-最低loss':>15s}")
for n in RUNS:
    rows = load_traj(n)
    fr = frontier(rows[-1]["per_hop_acc"])
    hit = next((r["step"] for r in rows if frontier(r["per_hop_acc"]) == 15), None)
    loss_hit = next((r["loss"] for r in rows if r["step"] == hit), None) if hit else None
    losses = [r["loss"] for r in rows]
    P(f"{n:16s} {fr:8d} {rows[-1]['loss']:8.4f} "
      f"{(f'{loss_hit:.4f}' if loss_hit is not None else '-'):>12s} "
      f"{rows[-1]['loss'] - min(losses):15.4f}")
P("")
P("--- train.log 的 loss 曲线采样（每 10k 步）---")
P(f"{'step':>7s} " + " ".join(f"{n.split('_')[0][:9]:>10s}" for n in RUNS))
LOGS = {n: load_log(n)[0] for n in RUNS}
for target in range(10000, 100001, 10000):
    line = []
    for n in RUNS:
        s = [x for x in LOGS[n] if x["step"] <= target]
        line.append(f"{s[-1]['loss']:10.4f}" if s else f"{'-':>10s}")
    P(f"{target:7d} " + " ".join(line))


P("")
P("=" * 78)
P("T7  对称跳对 acc[h] vs acc[16-h]（末点）——检验是否真的只是反方向同一题")
P("=" * 78)
P(f"{'run':16s} " + " ".join(f"{'H%d/H%d' % (h, 16-h):>12s}" for h in range(2, 8)))
for n in RUNS:
    p = load_traj(n)[-1]["per_hop_acc"]
    cells = []
    for h in range(2, 8):
        a, b = p[str(h)], p[str(16 - h)]
        cells.append(f"{a:.2f}/{b:.2f}")
    P(f"{n:16s} " + " ".join(f"{c:>12s}" for c in cells))


# ---------------------------------------------------- T4 课程 H_max 与跃升时刻
P("")
P("=" * 78)
P("T4  课程 H_max 到达各值的 step，vs frontier 跃升时刻")
P("=" * 78)
P(f"{'run':16s} " + " ".join(f"{'Hmax='+str(k):>9s}" for k in (1, 3, 7, 11, 15)))
for n in RUNS:
    steps = LOGS[n]
    cells = []
    for k in (1, 3, 7, 11, 15):
        hit = next((x["step"] for x in steps if x["hmax"] >= k), None)
        cells.append(f"{(str(hit) if hit else '-'):>9s}")
    P(f"{n:16s} " + " ".join(cells))
P("")
P("--- 每个 run：H_max 首次到 15 的 step，与 frontier 首次达 15 的 step ---")
P(f"{'run':16s} {'Hmax@15':>10s} {'frontier@15':>12s} {'后置步数':>9s}")
for n in RUNS:
    h15 = next((x["step"] for x in LOGS[n] if x["hmax"] >= 15), None)
    f15 = next((r["step"] for r in load_traj(n) if frontier(r["per_hop_acc"]) == 15), None)
    P(f"{n:16s} {(str(h15) if h15 else '-'):>10s} {(str(f15) if f15 else '-'):>12s} "
      f"{(str(f15 - h15) if (h15 and f15) else '-'):>9s}")


# ------------------------------------------------------- T5 各跳首达 0.8 步数
P("")
P("=" * 78)
P("T5  每个 hop 首次 >=0.8 的 step（末点未达则空）")
P("=" * 78)
P(f"{'run':16s} " + " ".join(f"{'H'+str(h):>7s}" for h in HOP))
FIRST = {}
for n in RUNS:
    rows = load_traj(n)
    cells, firsts = [], {}
    for h in HOP:
        hit = next((r["step"] for r in rows if r["per_hop_acc"][str(h)] >= 0.8), None)
        firsts[h] = hit
        cells.append(f"{(str(hit // 1000) + 'k' if hit else '-'):>7s}")
    FIRST[n] = firsts
    P(f"{n:16s} " + " ".join(cells))
P("")
P("--- looped 成功 run：首达步数 vs h_eff（若逐级计算，应随 h_eff 单调增）---")
for n in LOOPED:
    pairs = [(eff(h), FIRST[n][h]) for h in HOP if FIRST[n][h]]
    if pairs:
        pairs.sort()
        P(f"{n:16s} " + " ".join(f"{e}:{s//1000}k" for e, s in pairs))

with open("experiments/mining/mine_report.txt", "w") as f:
    f.write("\n".join(out) + "\n")
P("")
P("[已写入 experiments/mining/mine_report.txt]")
