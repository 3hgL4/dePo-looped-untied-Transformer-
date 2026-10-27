# Measurement audit of the dePo pointer-chasing benchmark

This branch adds a **measurement audit layer** on top of the MIT-licensed release of
*Inside the Loop* (Magnus Sesodia, Juil Sock, Philip Torr, Christian Schroeder de Witt).
The upstream code and its shipped runs are untouched; everything below is additive:
three new `src/` modules, five analysis tools, five run scripts, thirty tests, and the
raw summaries behind every number quoted here.

**中文摘要**：上游是论文的官方 release。我在它之上做的是**基准的测量审计**——先按预注册判据
跑 10 个 run 的 MVP，判据没达到；随后不是改结论，而是检查"是不是量具本身有问题"，据此改了
四处测量缺陷，并用五项对照实验（重训、逐层探针、无折叠网格、激活 patching + matched 控制、
格式替换）把每条结论压到它能承受的强度。下面每个数字都能在 `experiments/` 或
`intervention_outputs/` 里找到对应文件。

## What I did not do

This is **not** a reproduction of the paper's results, and it does not confirm or refute its
central claim. Task grid, model sizes, seeds, tokenizer format and evaluation protocol all
differ from the paper's. The object of study here is the *benchmark's measurement
behaviour* on one 16-node ring, on one 12 GB laptop GPU.

## 1. The pre-registered criterion was not met

The design doc froze, on 2026-10-21 and before any GPU hour: *looped (ouro-N×K) reaches
composition depth H ≤ 15 on 3/3 seeds*. The 10-run MVP (5681 min ≈ 94.7 GPU-h) did not
satisfy it: `ouro-4x8` scored frontier **15 / 15 / 6**, so 2 of 3 seeds. The failing seed was
not late — it sat at frontier 6 for all 19 evaluations between step 24000 and 100000. The
honest sequence was therefore not "report a negative result" but *check whether the
instrument was mis-measuring before believing the number*.

## 2. Four measurement defects, all in the harness

| defect | why it matters | evidence |
|---|---|---|
| `frontier` was computed on the raw-H axis only, so a run that solves H = 15 prints `H_sol@0.8 = 8` (the `h_eff = min(H, N−H)` cap) | a full solve is indistinguishable from a half solve in the log line | `src/metrics.py::frontier_three`; C1 run prints 8 while every hop is 1.000 |
| the logit-lens readout had no de-biasing: plain `"10".."15"` tokenise to `[▁,"1","0"]`, so 7 of 16 nodes share the first token and the chance floor is **0.2266**, not 1/16 | any per-node readout near the floor is uninterpretable | `src/node_tokens.py` audits the floor per format; `src/lens_metrics.py` implements a group-debiased estimator |
| `intervention` scoring used the whole trial set, so positions where the patched prefix leaves one or two candidates dominated the tail of the curve | the last j of every cell was degenerate (values 0.45–0.99 that no model produces) | `unambiguous_counts` is 0 for j ≥ H−1 |
| gradient accumulation was absent, so the memory knobs were all-or-nothing | 11.8 GB / 12.28 GB at batch 120 pinned the GPU at 43 W for 13 min per step | `--grad_accum_steps` (default 1 = bit-identical old behaviour) |

## 3. Five controlled experiments

**C1 — retrain the failing seed at K_max = 10** (10.82 h). Frontier moves 6 → 15, all fifteen
hop accuracies 1.000 from step 56000, stable under a threshold sweep 0.70–0.99. The cost is
*shape*, not ceiling: the phase transition arrives later than at K = 8 (40000 vs 24000) and in
shallower steps.

**C2 — per-layer linear probes, both seeds at H_eval = 6** (25 min total). The "s44 is two
layers later than s42" hypothesis is **false as stated**: half-rise layers are
`3,7,11,15,19,23,24` for s42 (Δ = 4 layers = one iteration, exactly) versus
`4,10,16,20,24,28,32` for s44 (Δ = +3,+5,+5,+5,+5,+8). Head commitment is late by **one**
iteration, not two — a ramp (0.696 → 0.983) in s42 versus a cliff (0.449 → 1.000) in s44.

**P3 — rerun on a fold-free grid** (`H_max = 8`, 11.02 h). At H ≤ 8 the reverse shortcut
π^H = π^-1 does not exist, so depth 8 cannot be faked by inverting one step. Result:
frontier `1 → 2 → 3 → 7 → 8`, then 17 consecutive evaluations at 8 with every hop ≥ 0.997.
This is what makes C1's `h_eff = 8` a real measurement rather than a label artefact.

**C4 — activation patching, with the matched control that was missing** (24 min, 14 cells).
Composition transfer is at chance in 6 of 7 cells. The seventh (`k8_s42_H8`) shows a small,
control-passing effect: **+2.6 pp [+1.9, +3.2], P(Δ≤0) < 0.001**, monotone across j. The
sensitivity anchor is the identity window, where the same pipeline recovers 0.167–0.630 —
2.7× to 10× the 1/16 floor — so the null in the composition window is not a blind ruler.

**P5 — a tokenizer-format swap that fails cleanly** (2 negative runs + a ladder diagnostic).
Replacing `"10".."15"` with single-token `▁A..▁P` lowers the readout floor to 1/16, and the
model then never learns 2 hops: h2 sits at 0.0659 ± 0.0061 over nine evaluations after 9000
extra steps from a checkpoint that already solves h1. A 5000-observation ladder diagnostic
shows the failure is not "stops after one hop" (k = 1 rate 0.061 ≈ 1/16) but **collapse to a
near-constant answer for hop ≥ 2** (one node takes 0.588 of the mass), with on-chain rate
0.129 *below* what random guessing over the chain would give. Reported as a single-seed
behavioural observation, not a property of the format.

## 4. Three readout traps found while running the control

- **j = 0 injection is a no-op.** Two provably different injections give bit-identical
  predictions at j = 0 (agreement 1.000/1.000/1.000/1.000/0.987/0.985), so that column
  measures nothing. Pooled windows start at j = 1.
- **The last column of `K = H` cells is phantom.** `pred_by_j` only fills 0..K−1; j = H was
  never executed but printed 0.0. Summaries truncate at `J = min(H+1, K)`.
- **`--random_init` is uninterpretable on this metric.** It ignores the injection entirely
  (real == control at every j) and its written-node decoding floor is 0, so its four zeros are
  the decoder, not the mechanism. The null used here is the analytic 1/16 plus the cross
  columns (0.024–0.068).

## 5. Reproduction

```bash
# C1 + C2  (probe at H_eval = 6, then the K_max = 10 retrain)
DRY_RUN=1 bash run_c1_c2.sh && bash run_c1_c2.sh
# P3 / P5 / C4 in one serial queue
DRY_RUN=1 bash run_p3_p5_c4.sh && bash run_p3_p5_c4.sh
# C4 with the matched control (includes a bit-exact regression of the published cells)
bash run_c4_control.sh
# summaries
python tools/frontier_report.py runs/slice/ouro-4x10_s44
python tools/c2_crossing.py --pair s42:probe_outputs/c2/H6_s42 --pair s44:probe_outputs/c2/H6_s44
python tools/h2_ladder.py --checkpoint <ckpt> --node_format letters --H_eval 1,2,3,4
python tools/c4_control_report.py --root intervention_outputs/p3p5c4_control --draws 1000
```

Tests: `pytest tests -q` → **99 passed** (CPU only, about 80 s). `test_tables.py` reads the
upstream shipped run directories, so it fails only if those are deleted from a working tree;
a clean clone is green.

## 6. What the numbers do not support

Not a reproduction. Not evidence that looped and untied Transformers differ in
composability in general — two seeds, one ring size, one GPU. No claim about *why* the
letters format collapses: no attention analysis, no probe, no intervention was run against
it, and the proposed explanation is explicitly left as a hypothesis. C4's single surviving
cell is one seed without multiple-comparison correction.

## Licence and attribution

Upstream code and data: MIT, © 2026 Magnus Sesodia, Juil Sock, Philip Torr,
Christian Schroeder de Witt. See `LICENSE` and `CITATION.cff`. The audit layer in this
branch is offered under the same licence.
