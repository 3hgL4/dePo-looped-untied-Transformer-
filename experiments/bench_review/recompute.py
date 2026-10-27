#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
B 层改进的量化验证（零新增训练、零 GPU）。
只读 trajectory.jsonl 与 git HEAD 里的官方数据。

目的：把报告 §12.4 的 B1-B5 从"批评"变成"改与不改，结论变不变"的对照。
特别地，用来证伪我自己在 B1 里提出的"改报双向 frontier"这个改法。
"""
import json, glob, os, subprocess, re, statistics

THR = 0.8
HOPS = list(range(1, 16))

def frontier_continuous(seq):
    """从 h_eff=1 起连续满足 >=THR 的最大档数；遇首个不满足即停。"""
    f = 0
    for i, a in enumerate(seq, start=1):
        if a >= THR:
            f = i
        else:
            break
    return f

def four_frontiers(p, thr=THR):
    """p: dict str(hop)->acc（1..15）。返回 4 种 frontier 定义。"""
    global THR
    old = THR
    THR = thr
    try:
        fwd = frontier_continuous([p[str(h)] for h in range(1, 9)])              # 原口径：H=1..8 正向
        rev = frontier_continuous([p[str(16 - h)] for h in range(1, 9)])         # 反向：H=15..8
        both = frontier_continuous([min(p[str(h)], p[str(16 - h)]) for h in range(1, 9)])  # 两侧都要过
        mx = max(fwd, rev)                                                       # 取较优一侧
        return dict(fwd=fwd, rev=rev, both=both, maximum=mx)
    finally:
        THR = old

def heff_vector(p):
    """8 维连续向量：h_eff=1..8 上，fwd 与 rev 的均值。"""
    v = []
    for h in range(1, 9):
        a, b = p[str(h)], p[str(16 - h)]
        v.append((a + b) / 2.0)
    return v

def load_local():
    out = {}
    for d in sorted(glob.glob('runs/slice/*/')):
        n = os.path.basename(d.rstrip('/'))
        f = os.path.join(d, 'trajectory.jsonl')
        if not os.path.exists(f):
            continue
        rows = [json.loads(l) for l in open(f)]
        seen = {}
        for r in rows:
            seen[r['step']] = r            # 去重取后者（修 ouro-4x8_s42 的 68000 重复）
        out[n] = [seen[s] for s in sorted(seen)]
    return out

def load_official():
    """从 git HEAD 读官方 100 run（不恢复工作区）。"""
    names = subprocess.run(['git', 'ls-tree', '--name-only', 'HEAD',
                            'experiments/h1_clean_sweep/runs/'],
                           capture_output=True, text=True).stdout.split()
    out = {}
    for full in names:
        run = full.split('/')[-1]
        raw = subprocess.run(['git', 'show', 'HEAD:' + full + '/trajectory.jsonl'],
                             capture_output=True, text=True).stdout.strip()
        if not raw:
            continue
        rows = [json.loads(l) for l in raw.split('\n')]
        seen = {}
        for r in rows:
            seen[r['step']] = r
        out[run] = [seen[s] for s in sorted(seen)]
    return out

def w(s=''):
    print(s)

# ---------------------------------------------------------------- A 表
def table_A(local):
    w('=' * 100)
    w('A  本机 10 run：4 种 frontier 定义 + 8 维 h_eff 连续向量')
    w('=' * 100)
    w(f'{"run":16s} {"fwd":>4s} {"rev":>4s} {"both":>5s} {"max":>4s}   heff=1..8 连续向量')
    for n, rows in local.items():
        p = rows[-1]['per_hop_acc']
        f = four_frontiers(p)
        v = heff_vector(p)
        w(f'{n:16s} {f["fwd"]:4d} {f["rev"]:4d} {f["both"]:5d} {f["maximum"]:4d}   '
          + ' '.join(f'{x:.2f}' for x in v))
    w()

# ---------------------------------------------------------------- B 表
def table_B(official):
    w('=' * 100)
    w('B  官方 100 run（git HEAD，20 配置 × 5 seeds）：同样 4 种定义，按配置聚合')
    w('=' * 100)
    agg = {}
    for run, rows in official.items():
        cfg = re.sub(r'_s\d+$', '', run)
        p = rows[-1]['per_hop_acc']
        agg.setdefault(cfg, []).append(four_frontiers(p))
    def key(c):
        fam, rest = c.split('-')
        if 'x' in rest:
            a, b = rest.split('x'); return (0 if fam == 'ouro' else 1, int(a), int(b))
        return (1, 0, int(rest))
    w(f'{"config":12s} {"n":>2s}  {"fwd":>16s} {"rev":>16s} {"both":>16s} {"max":>16s}')
    for cfg in sorted(agg, key=key):
        fs = agg[cfg]
        row = lambda k: ','.join(str(x[k]) for x in fs)
        w(f'{cfg:12s} {len(fs):2d}  {row("fwd"):>16s} {row("rev"):>16s} {row("both"):>16s} {row("maximum"):>16s}')
    w()

# ---------------------------------------------------------------- C 表
def table_C(local, official):
    w('=' * 100)
    w('C  阈值敏感性：untied 臂在不同阈值下的 max frontier（检验 B1 的真实性质）')
    w('=' * 100)
    thr_list = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
    untied = [n for n in local if n.startswith('deep-')]
    w(f'{"run":16s} ' + ' '.join(f'{t:>5.2f}' for t in thr_list))
    for n in untied:
        p = local[n][-1]['per_hop_acc']
        vals = [four_frontiers(p, t)['maximum'] for t in thr_list]
        w(f'{n:16s} ' + ' '.join(f'{v:5d}' for v in vals))
    w()
    w('--- 官方 untied 配置（每配置 5 seed 的 max frontier 分布）---')
    for cfg in ['deep-2', 'deep-4', 'deep-8', 'deep-16', 'deep-32']:
        if cfg not in set(re.sub(r'_s\d+$', '', r) for r in official):
            continue
        w(f'  {cfg:8s}')
        for t in thr_list:
            vals = []
            for run, rows in official.items():
                if re.sub(r'_s\d+$', '', run) != cfg:
                    continue
                vals.append(four_frontiers(rows[-1]['per_hop_acc'], t)['maximum'])
            w(f'    thr={t:.2f}  ' + ','.join(map(str, vals)))
    w()

# ---------------------------------------------------------------- D 表
def table_D(local):
    w('=' * 100)
    w('D  B4 的收益：只评 h_eff=1..8 能省多少（用已实测的单次 eval 耗时外推）')
    w('=' * 100)
    # 单次 eval 实测（报告 §5 反推）：deep-32 7.0-7.2min，looped 0.2-0.3min
    cost = {}
    for n in local:
        cost[n] = 7.1 if n.startswith('deep-') else 0.25
    w(f'{"run":16s} {"15跳eval(min)":>14s} {"8跳eval(min)":>14s} {"节省":>8s}')
    tot15 = tot8 = 0.0
    for n in local:
        c = cost[n] * 25            # 25 个 eval 点
        c8 = c * 8 / 15
        tot15 += c; tot8 += c8
        w(f'{n:16s} {c:14.1f} {c8:14.1f} {(c-c8)/c*100:7.1f}%')
    w(f'{"合计":16s} {tot15:14.1f} {tot8:14.1f} {(tot15-tot8)/tot15*100:7.1f}%')
    w()
    w('注：15 跳里有 7 对是同一道题（H 与 16-H），h_eff=8 自配对，故 8 档覆盖全部信息。')
    w()

# ---------------------------------------------------------------- E 表
def table_E(local):
    w('=' * 100)
    w('E  B2 的量化：H_max=15 造成的难度折叠与训练数据冗余')
    w('=' * 100)
    pairs = [(h, 16 - h) for h in range(1, 9)]
    w('  15 个 H 的等价类（h_eff = min(H,16-H)）：')
    for h_eff, (a, b) in enumerate(pairs, start=1):
        tag = '同一题（自配对）' if a == b else f'H={a} 与 H={b} 等价'
        w(f'    h_eff={h_eff}  {tag}')
    w()
    w('  → 8 个独立难度档，被展开成 15 个评测项：评测冗余 15/8 = 1.88x')
    w('  → 若 H_max=8：8 个 H 恰好 8 个独立档，冗余 1.00x，且难度单调')
    w()
    # 三个 looped 成功 run 在 H 与 16-H 上的 acc 差（同一难度、两个方向）
    w('  同一 h_eff 下 fwd 与 rev 的实测差（looped 成功 run，末点）：')
    w(f'  {"run":16s} ' + ' '.join(f'{"h=%d"%h:>7s}' for h in range(1, 9)))
    for n in [x for x in local if x.startswith('ouro-')]:
        p = local[n][-1]['per_hop_acc']
        w(f'  {n:16s} ' + ' '.join(f'{p[str(h)]-p[str(16-h)]:+7.3f}' for h in range(1, 9)))
    w('  注：若方向完全无差别，这行应全为 0。偏差即为"方向效应"，是 U 形之外的另一维。')
    w()

# ---------------------------------------------------------------- F 表
def table_F(official):
    w('=' * 100)
    w('F  阈值不稳定性：有多少 run 的 frontier 在 0.75-0.85 之间跳变 >=3 档')
    w('=' * 100)
    thr_list = [0.70, 0.75, 0.80, 0.85]
    jumps = []
    w(f'{"run":16s} ' + ' '.join(f'{"fwd@%.2f" % t:>10s}' for t in thr_list) + '   跨度')
    for run in sorted(official):
        p = official[run][-1]['per_hop_acc']
        vals = [four_frontiers(p, t)['fwd'] for t in thr_list]
        span = max(vals) - min(vals)
        if span >= 3:
            jumps.append((run, vals, span))
            w(f'{run:16s} ' + ' '.join(f'{v:10d}' for v in vals) + f'   {span:4d}')
    w()
    w(f'→ 100 个官方 run 中有 {len(jumps)} 个 run 的 fwd frontier 在 0.70-0.85 阈值区间内跳变 >=3 档')
    w('  含义：这些 run 的逐跳曲线整段挤在阈值附近，frontier 这个标量对阈值高度敏感。')
    w()
    # 反向侧同样统计
    rev_jumps = 0
    for run in sorted(official):
        p = official[run][-1]['per_hop_acc']
        vals = [four_frontiers(p, t)['rev'] for t in thr_list]
        if max(vals) - min(vals) >= 3:
            rev_jumps += 1
    w(f'  同口径下 rev frontier 跳变 >=3 档的 run 数：{rev_jumps}')
    w()

if __name__ == '__main__':
    local = load_local()
    official = load_official()
    w(f'本机 run 数 = {len(local)}，官方 run 数 = {len(official)}，阈值 THR = {THR}')
    w()
    table_A(local)
    table_B(official)
    table_C(local, official)
    table_D(local)
    table_E(local)
    table_F(official)
