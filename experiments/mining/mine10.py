"""
T10  把 pi^15 = pi^-1 从 3 个 H 扩展到全部 15 个 H（纯数据，不训练）

对每个 H 生成 NS 个样本，从题干边表解析出置换 pi，验证：
    正向走 H 步  ==  反向走 (16-H) 步
若全部成立，则 H 与 16-H 是同一道题，难度只由 h_eff = min(H, 16-H) 决定。
这是 U 形的完整数学验证（此前只验了 H=1/14/15 三个点）。
"""
import os
import random
import sys

sys.path.insert(0, os.getcwd())
from src.datasets.pointer_chasing import generate_pointer_chasing_sample

N, M = 16, 5
NS = 5000
out = []
def P(*a):
    s = " ".join(str(x) for x in a)
    out.append(s)
    print(s)

P("=" * 78)
P(f"T10  正向 H 步 == 反向 (16-H) 步？ N={N}, 每个 H 取 {NS} 个样本")
P("=" * 78)
P("H    h_eff   正向链==反向链   答案==正向末端   答案==反向末端   反例数")
tot_bad = 0
for H in range(1, 16):
    bad_chain = bad_fwd = bad_rev = 0
    for k in range(NS):
        rng = random.Random(H * 7919 + k)
        text, ans, meta = generate_pointer_chasing_sample(rng, N, H, M)
        # 解析边表
        head = text.split(" | ")[0]
        pi = {}
        for part in head.split(" , "):
            u, v = part.split(" > ")
            pi[int(u)] = int(v)
        start = int(meta["queries"][-1]["start"])
        a = int(meta.get("answer", meta["queries"][-1].get("answer")))
        # 正向 H 步
        x = start
        for _ in range(H):
            x = pi[x]
        # 反向 (16-H) 步
        inv = {v: u for u, v in pi.items()}
        y = start
        for _ in range(N - H):
            y = inv[y]
        if x != y:
            bad_chain += 1
        if x != a:
            bad_fwd += 1
        if y != a:
            bad_rev += 1
    tot_bad += bad_chain + bad_fwd + bad_rev
    P(f"{H:2d}   {min(H, N-H):5d}   {NS-bad_chain:6d}/{NS}       {NS-bad_fwd:6d}/{NS}        "
      f"{NS-bad_rev:6d}/{NS}      {bad_chain}/{bad_fwd}/{bad_rev}")
P("")
P(f"全部 15 个 H × {NS} 样本 = {15*NS} 次检验，总反例数 = {tot_bad}")

with open("experiments/mining/mine10_report.txt", "w") as f:
    f.write("\n".join(out) + "\n")
P("[已写入 experiments/mining/mine10_report.txt]")
