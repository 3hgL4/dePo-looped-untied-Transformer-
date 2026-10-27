"""分析 logit_lens 输出：深层 vs 浅层在哪一层"算出答案"，失败时退化成什么。

用法: python /tmp/analyze_lens.py <probe_root>
"""
import sys, json, glob, os
import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "experiments/u_shape_probe"


def equiv_classes(probs):
    """按概率向量分组：同组节点共享同一个首 token（多位数节点 10-15）。"""
    Q, L, N = probs.shape
    sigs = {}
    for n in range(N):
        key = tuple(np.round(probs[:, -1, n], 6))
        sigs.setdefault(key, []).append(n)
    cls = np.zeros(N, dtype=int)
    groups = list(sigs.values())
    for gi, g in enumerate(groups):
        for n in g:
            cls[n] = gi
    return cls, groups


def analyze(path):
    d = np.load(os.path.join(path, "logit_lens.npz"))
    probs, chains, solved = d["probs"], d["chains"], d["solved"]
    Q, L, N = probs.shape
    cls, groups = equiv_classes(probs)
    ans = chains[:, -1]
    start = chains[:, 0]
    hop1 = chains[:, 1] if chains.shape[1] > 1 else start
    pred_node = probs.argmax(-1)                 # (Q, L)
    pred_cls = cls[pred_node]
    acc = (pred_cls == cls[ans][:, None]).mean(0)          # 首 token 级准确率，逐层
    acc_full_last = float(solved.mean())                   # 完整答案（数字串）级
    # 退化度：每层最常见预测类占比
    mode_share = []
    for l in range(L):
        vals, cnt = np.unique(pred_cls[:, l], return_counts=True)
        mode_share.append(vals[np.argmax(cnt)]), mode_share.__len__()
    mode_share = []
    mode_cls = []
    mode_val = []
    for l in range(L):
        vals, cnt = np.unique(pred_cls[:, l], return_counts=True)
        i = int(np.argmax(cnt))
        mode_cls.append(int(vals[i]))
        mode_share.append(float(cnt[i]) / Q)
        mode_val.append(float(cnt[i]))
    # 预测 start / hop1 的比例（逐层）
    frac_start = (pred_cls == cls[start][:, None]).mean(0)
    frac_hop1 = (pred_cls == cls[hop1][:, None]).mean(0)
    p_ans = probs[np.arange(Q)[:, None], np.arange(L)[None, :], ans[:, None]].mean(0)
    return dict(Q=Q, L=L, groups=groups, acc=acc, solved=acc_full_last,
                mode_share=np.array(mode_share), frac_start=frac_start,
                frac_hop1=frac_hop1, p_ans=p_ans, chains=chains)


def first(acc, thr=0.8):
    hit = np.where(acc >= thr)[0]
    return int(hit[0]) if len(hit) else None


def last(acc, thr=0.8):
    ok = acc >= thr
    # 连续从某层开始一直达标
    for l in range(len(acc)):
        if ok[l:].all():
            return l
    return None


rows = []
for path in sorted(glob.glob(os.path.join(root, "*_H*"))):
    name = os.path.basename(path)
    if name.startswith("sanity"):
        continue
    r = analyze(path)
    rows.append((name, r))

print(f"{'run_H':18s} {'Q':>4s} {'levels':>6s} {'solved%':>7s} {'首token acc 逐层':s}")
for name, r in rows:
    acc = r["acc"]
    print(f"{name:18s} {r['Q']:4d} {r['L']:6d} {r['solved']*100:6.1f}%")
    print("   acc   ", " ".join(f"{v:.2f}" for v in acc))
    print("   mode占比", " ".join(f"{v:.2f}" for v in r["mode_share"]))
    print("   =start  ", " ".join(f"{v:.2f}" for v in r["frac_start"]))
    print("   =hop1   ", " ".join(f"{v:.2f}" for v in r["frac_hop1"]))
    print(f"   首次>=0.8 @ level {first(acc)}；最后一次跳上0.8 @ {last(acc)}")

print()
print("=== 结论行 ===")
for name, r in rows:
    acc = r["acc"]
    f = first(acc)
    print(f"{name:18s} solved={r['solved']:.3f}  首token首次达标level={f}  "
          f"末层acc={acc[-1]:.3f}  末层最常见预测类占比={r['mode_share'][-1]:.2f}")
