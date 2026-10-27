"""
挖掘 M3：looped 模型「达到 0.9 需要几次迭代」
若 s44 只是 K_max 不够，则 h_eff 小的题目所需迭代次数应与 s42 相同，只有难档更高。
若 s44 是「每次迭代更弱」，则所有 h_eff 上都比 s42 多要 1-2 次。
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

def curve(npz_path):
    z = np.load(npz_path)
    probs, chains, solved = z["probs"], z["chains"], z["solved"]
    Q, L, N = probs.shape
    H = chains.shape[1] - 1
    pred = probs.argmax(-1)
    ansgrp = np.array([grp(c) for c in chains[:, -1]])
    freq = Counter(ansgrp.tolist())
    base = sum((c / Q) ** 2 for c in freq.values())
    hits = []
    for li in range(L):
        pg = np.array([grp(v) for v in pred[:, li]])
        hits.append(float((pg == ansgrp).mean()))
    return H, hits, base, float(solved.mean())

P("=" * 78)
P("M3  looped：组级 hit 达到 0.90 所需「层号」= 所需迭代次数（层0=prelude 输出）")
P("=" * 78)
lut = {}
for d in sorted(glob.glob("experiments/u_shape_probe/ouro-4x8_s4*/logit_lens.npz")):
    run = os.path.basename(os.path.dirname(d)).rsplit("_H", 1)[0]
    H, hits, base, sol = curve(d)
    lut.setdefault(run, {})[H] = (hits, base, sol)

for run in sorted(lut):
    P(f"--- {run} ---")
    P(f"{'H':>3s} {'h_eff':>6s} {'末层hit':>8s} {'需迭代层':>9s}  逐层hit")
    for H in sorted(lut[run]):
        hits, base, sol = lut[run][H]
        need = next((i for i, v in enumerate(hits) if v >= 0.90), None)
        P(f"{H:3d} {min(H,16-H):6d} {hits[-1]:8.3f} {(str(need) if need is not None else '-'):>9s}  "
          + " ".join(f"{v:.2f}" for v in hits))
    P("")

P("=" * 78)
P("M3b  s42 vs s44 在相同 H 上的逐层差（s44 - s42）")
P("=" * 78)
common = sorted(set(lut.get("ouro-4x8_s42", {})) & set(lut.get("ouro-4x8_s44", {})))
P(f"共同 H: {common}")
for H in common:
    a = lut["ouro-4x8_s42"][H][0]
    b = lut["ouro-4x8_s44"][H][0]
    n = min(len(a), len(b))
    P(f"H={H}: " + " ".join(f"{b[i]-a[i]:+.2f}" for i in range(n)) + f"   (末层差 {b[-1]-a[-1]:+.3f})")

with open("experiments/mining/mine4_report.txt", "w") as f:
    f.write("\n".join(out) + "\n")
P("[已写入 experiments/mining/mine4_report.txt]")
