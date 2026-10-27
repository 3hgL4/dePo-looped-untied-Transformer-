"""
挖掘 T6 + T8（纯 CPU / 纯只读）

T6  各 H 的答案分布：如果模型完全不算、恒输出某个固定答案，在每个 H 上能拿多少分？
    → 给出「frontier 读数被 tokenizer + 答案分布污染」的定量上限
T8  lens 重挖：逐层 top1 概率、以及逐层 argmax 落在正确链条的哪个位置
    → 区分「算到第 k 步就停」与「根本没算」；这是对已撤回的『零计算』结论的定量替代
"""
import glob
import json
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.getcwd())
out = []
def P(*a):
    s = " ".join(str(x) for x in a)
    out.append(s)
    print(s)

# ------------------------------------------------------------------ T6
P("=" * 78)
P("T6  答案分布与「恒答常数」的期望得分（纯数据生成，N=5000/H，seed 固定）")
P("=" * 78)
try:
    from src.datasets.pointer_chasing import generate_pointer_chasing_sample
    N_GRAPH, M = 16, 5
    NS = 5000
    P("H :   P(ans=1)  P(ans in {1,10..15})  最大单点频率(哪个节点)  分布熵")
    BEST = {}
    for H in range(1, 16):
        cnt = Counter()
        for k in range(NS):
            rng = random.Random(H * 100003 + k)
            _text, _ans, meta = generate_pointer_chasing_sample(rng, N_GRAPH, H, M)
            a = meta["queries"][-1]["answer"] if "answer" in meta["queries"][-1] else meta.get("answer")
            cnt[int(a)] += 1
        tot = sum(cnt.values())
        p1 = cnt[1] / tot
        pgrp = sum(cnt[v] for v in [1, 10, 11, 12, 13, 14, 15]) / tot
        node, c = cnt.most_common(1)[0]
        ent = -sum((c / tot) * np.log2(c / tot) for c in cnt.values())
        BEST[H] = (p1, pgrp, node, c / tot)
        P(f"{H:2d} :   {p1:8.3f}  {pgrp:18.3f}  {c/tot:14.3f} (node {node:2d})   {ent:5.2f}")
    P("")
    P("解读：若某 H 上「最大单点频率」很高，则该 H 上的准确率读数有很大一块可由常量回答拿到。")
    P(f"理论随机基线（16 类均匀）= 0.0625；首 token 口径随机基线 = 见上表 pgrp 列（这是『恒答 1』的上限）")
except Exception as e:
    P(f"[T6 失败] {type(e).__name__}: {e}")

# ------------------------------------------------------------------ T8
P("")
P("=" * 78)
P("T8  lens 逐层重挖：top1 概率 / argmax 落在正确链的哪个位置")
P("=" * 78)
P("对每个 query 的每一层：pred = node_probs.argmax；pos = 该节点在 chain 中的下标（-1 = 不在链上）")
P("若模型算到第 k 步就停，pos 应集中在 k 附近；若没算，pos 应大量为 -1 或固定 0。")
P("")
tags = sorted(os.path.basename(d) for d in glob.glob("experiments/u_shape_probe/*/logit_lens.npz"))
tags = [t for t in tags if t != "logit_lens.npz"]
for d in glob.glob("experiments/u_shape_probe/*/logit_lens.npz"):
    tag = os.path.basename(os.path.dirname(d))
    z = np.load(d)
    probs, chains, solved = z["probs"], z["chains"], z["solved"]
    amb = z["ambiguous"]
    Q, L, N = probs.shape
    H = chains.shape[1] - 1
    if tag.endswith("H1") or "sanity" in tag:
        pass
    pred = probs.argmax(-1)                      # (Q, L)
    top1 = probs.max(-1)                         # (Q, L)
    # 每个 query 的 chain 位置索引
    posmap = np.full((Q, L), -1, dtype=np.int64)
    for q in range(Q):
        ch = chains[q].tolist()
        idx = {v: i for i, v in enumerate(ch)}
        row = pred[q]
        posmap[q] = [idx.get(int(v), -1) for v in row]
    # 统计：末层 pos 分布、中间层（去掉最后一层）pos 分布
    def dist(arr):
        c = Counter(arr.tolist())
        return c
    lastpos = dist(posmap[:, -1])
    midpos = dist(posmap[:, :-1].ravel())
    def fmt(c, total):
        top = sorted(c.items(), key=lambda kv: -kv[1])[:5]
        return "  ".join(f"{'off' if k < 0 else 'c'+str(k)}:{v/total:.2f}" for k, v in top)
    P(f"--- {tag}  (H={H}, Q={Q}, L={L}, solved={solved.mean():.3f}, ambiguous={amb.tolist()}) ---")
    P(f"    末层 top1 prob 均值 = {top1[:, -1].mean():.3f}   末层 pos 分布: {fmt(lastpos, Q)}")
    P(f"    非末层 top1 prob 均值 = {top1[:, :-1].mean():.3f}   非末层 pos 分布: {fmt(midpos, midpos.sum() if False else posmap[:, :-1].size)}")
    # 逐层：正确位置的累积比例（pos == H 即答对位置）
    P("    逐层：top1概率 | hit_final(末层位置正确) | on_chain(落在链上任一点)")
    step = max(1, L // 11)
    cells = []
    for li in range(0, L, step):
        on_chain = (posmap[:, li] >= 0).mean()
        hitf = (posmap[:, li] == H).mean()
        cells.append(f"l{li}:{top1[:,li].mean():.2f}/{hitf:.2f}/{on_chain:.2f}")
    P("      " + "  ".join(cells))
    P("")

with open("experiments/mining/mine2_report.txt", "w") as f:
    f.write("\n".join(out) + "\n")
P("[已写入 experiments/mining/mine2_report.txt]")
