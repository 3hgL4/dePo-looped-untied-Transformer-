"""
挖掘 T9：干净指标重算（修正 mine2 里 on_chain / hit_final 的两个缺陷）

缺陷 1：on_chain 的基线 = (H+1)/16，H=14/15 时基线就有 0.94/1.00 → 该指标无信息
缺陷 2：hit_final 用的是「预测节点 == 答案节点」，但 7 个歧义节点共享首 token，
        argmax 恒取最小索引 → 上限被压到 ~0.62

改用 token-group 级指标：
  group(v) = 'G'  if v in {1,10,11,12,13,14,15}   （共享首 token '1'）
           = str(v) otherwise                      （9 个独立类，共 10 类）
  hit = (group(pred_argmax) == group(answer))
  基线 = sum_g p_g^2  （该 eval 集上答案组的实际频率）
  归一化超额 = (hit - base) / (1 - base)

另附：
  M1  untied 臂的能力矩阵（h_eff x 方向）
  M2  s42 vs s44 在同 H=8 上的逐层曲线对比（量化「在做计算但不够」）
"""
import glob
import os
import json
from collections import Counter

import numpy as np

AMB = {1, 10, 11, 12, 13, 14, 15}
def grp(v):
    return "G" if int(v) in AMB else str(int(v))

out = []
def P(*a):
    s = " ".join(str(x) for x in a)
    out.append(s)
    print(s)

P("=" * 78)
P("T9  组级逐层命中率（消除 7 个歧义节点的污染）")
P("=" * 78)
P("hit = group(argmax probs) == group(answer)；base = sum_g p_g^2；超额 = (hit-base)/(1-base)")
P("超额 ≈ 0 表示该层对该题无信息；1.0 表示全对。")
P("")

rows = []
for d in sorted(glob.glob("experiments/u_shape_probe/*/logit_lens.npz")):
    tag = os.path.basename(os.path.dirname(d))
    if not any(tag.endswith(f"H{h}") for h in range(1, 16)):
        continue
    z = np.load(d)
    probs, chains, solved = z["probs"], z["chains"], z["solved"]
    Q, L, N = probs.shape
    H = chains.shape[1] - 1
    pred = probs.argmax(-1)
    ansgrp = np.array([grp(c) for c in chains[:, -1]])
    freq = Counter(ansgrp.tolist())
    base = sum((c / Q) ** 2 for c in freq.values())
    # 逐层
    curve = []
    for li in range(L):
        pg = np.array([grp(v) for v in pred[:, li]])
        hit = (pg == ansgrp).mean()
        curve.append(hit)
    ex = [(h - base) / (1 - base) for h in curve]
    rows.append(dict(tag=tag, H=H, L=L, base=base, solved=solved.mean(),
                     hit=curve, ex=ex))
    P(f"--- {tag}  H={H:2d} L={L:2d} Q={Q} base={base:.3f} solved={solved.mean():.3f} ---")
    P("    hit : " + " ".join(f"{v:.2f}" for v in curve))
    P("    超额: " + " ".join(f"{v:+.2f}" for v in ex))

P("")
P("=" * 78)
P("M1  untied 臂（deep-*）能力矩阵：按 h_eff 与方向展开（末点）")
P("=" * 78)
P("正向 = H=h（h<=8）；反向 = H=16-h（即 h_eff=h 的另一端）")
P("")
runs = sorted(os.path.basename(x.rstrip("/")) for x in glob.glob("runs/slice/*/"))
runs = [r for r in runs if os.path.exists(f"runs/slice/{r}/trajectory.jsonl")]
P(f"{'run':14s} " + " ".join(f"{'heff'+str(k):>14s}" for k in range(1, 8)))
P(f"{'':14s} " + " ".join(f"{'fwd/rev':>14s}" for k in range(1, 8)))
for n in runs:
    traj = [json.loads(l) for l in open(f"runs/slice/{n}/trajectory.jsonl")]
    seen = {}
    for r in traj:
        seen[r["step"]] = r
    last = seen[max(seen)]["per_hop_acc"]
    cells = []
    for k in range(1, 8):
        fwd = last[str(k)]
        rev = last[str(16 - k)] if k != 8 else float("nan")
        cells.append(f"{fwd:.2f}/{rev:.2f}")
    P(f"{n:14s} " + " ".join(f"{c:>14s}" for c in cells))

P("")
P("=" * 78)
P("M2  同 H=8 对比：ouro-4x8_s42（解出）vs s44（失败）逐层超额")
P("=" * 78)
byt = {r["tag"]: r for r in rows}
for tag in ["ouro-4x8_s42_H8", "ouro-4x8_s44_H8", "deep-32_s42_H2", "deep-32_s43_H2"]:
    if tag in byt:
        r = byt[tag]
        P(f"{tag:20s} H={r['H']:2d} solved={r['solved']:.3f} base={r['base']:.3f}")
        P("    超额: " + " ".join(f"{v:+.2f}" for v in r["ex"]))
P("")
P("解读要点：looped 的超额应从第 2-4 层起单调爬升到 ~1.0；")
P("          untied 若全层 ≈ 0、仅末层跳，则中间层对该题确实无信息。")

with open("experiments/mining/mine3_report.txt", "w") as f:
    f.write("\n".join(out) + "\n")
P("[已写入 experiments/mining/mine3_report.txt]")
