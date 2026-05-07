# Inside the Loop

Code release for **_Inside the Loop: A Mechanistic Study of Weight-Tied Transformers on Depth-Bound Algorithmic Tasks_** (anonymous submission).

This repository contains the training, probing, and figure code that produced
every numbered figure in the paper. There is one task (the directed
permutation traversal, dePo / pointer chasing), one trainer (fixed-K with
uniform-H sampling), one probe pipeline (per-iteration linear probes plus
logit lens), and the plotting scripts that turn their outputs into the §4 and
§5 panels.

| Paper figure | Script | Output filename used by `main.tex` |
|---|---|---|
| Fig. 2 — saturation gradient | `scripts/plot_h1_saturation.py` | `fig_saturation_gradient.pdf` |
| Fig. 3 — matched-D_eff hero | `scripts/plot_h1_saturation.py` | `fig_matched_deff32.pdf` |
| Fig. 4 — full-chain probe staircase | `scripts/plot_h2_section5.py` | `fig_h2_full_chain_hero.pdf` |
| Fig. 5 — logit-lens hero | `scripts/plot_h2_section5.py` | `fig_h2_lens_hero.pdf` |
| Fig. 6 — commitment deferral | `scripts/plot_h2_section5.py` | `fig_h2_commit_deferral.pdf` |
| Fig. A — per-seed staircase (appendix) | `scripts/plot_h2_section5.py` | `fig_h2_full_chain_appendix.pdf` |

## Installation

This project is `uv`-managed. With [uv](https://docs.astral.sh/uv/) installed:

```bash
git clone <repo-url>
cd insidetheloop
uv sync
```

`pyproject.toml` pins the CUDA 12.1 PyTorch wheels on Linux. On macOS / CPU /
a different CUDA, comment out the `[[tool.uv.index]]` and `[tool.uv.sources]`
blocks at the bottom of `pyproject.toml` and let uv resolve `torch` normally.

Verify:

```bash
uv run python -c "from looped_lm.model import LoopedTransformer; print('ok')"
```

## Reproducing the paper

The full pipeline is: **train** a sweep of looped and unlooped models with
the fixed-K uniform-H trainer, **probe** each saturated checkpoint, **plot**.

### 1. Train

The paper's main sweep is the matched-D_eff = 32 cell trio
(`ouro-2x16`, `ouro-4x8`, `deep-32`) plus the surrounding ladder used in
Fig. 2. Each cell is trained for `total_steps = 200000` over five seeds:

```bash
for cell in ouro-2x16 ouro-4x8 ouro-8x4 ouro-2x4 ouro-4x4 deep-4 deep-8 deep-16 deep-32; do
  for seed in 42 43 44 45 46; do
    uv run python -m looped_lm.train_uniform_h \
        --config_name $cell \
        --N 16 --M 5 --H_min 1 --H_max 15 \
        --total_steps 200000 \
        --use_nope \
        --save_dir runs/clean_sweep/${cell}_s${seed} \
        --seed $seed \
        --device cuda
  done
done
```

Each run writes `args.json`, `trajectory.jsonl`, and `checkpoint.pt` under
`--save_dir`. `trajectory.jsonl` records per-hop accuracy at every
`--eval_interval` step.

### 2. Probe and lens

Probe and lens both consume trained checkpoints. The probe orchestrator
launches `looped_lm.probe_h2` per (cell, seed); call the lens directly the
same way. The §5 plotting script expects the four output trees shown below.

```bash
# Probes at H_eval = 4 (used by Fig. 4 and Fig. A).
uv run python scripts/run_probes.py \
    --runs_root runs/clean_sweep \
    --out_root  results/probes/probe_h4 \
    --models ouro-2x16_s42 ouro-2x16_s44 ouro-2x16_s46 \
             ouro-4x8_s42 ouro-4x8_s43 ouro-4x8_s44 ouro-4x8_s45 \
             deep-32_s42 \
    --k_sweep   # also runs the K-sweep used by Fig. 6

# Per-mMLRD probes (used by the hero panel of Fig. 4).
uv run python scripts/run_probes.py \
    --runs_root runs/clean_sweep \
    --out_root  results/probes/probe_permlrd \
    --models ouro-2x16_s42 ouro-4x8_s42

# Lens at H_eval = 4 (used by Fig. 5, 6).
for target in ouro-2x16_s42 ouro-2x16_s44 ouro-2x16_s46 \
              ouro-4x8_s42 ouro-4x8_s43 ouro-4x8_s44 ouro-4x8_s45 \
              deep-32_s42; do
  cell=${target%_s*}
  uv run python -m looped_lm.logit_lens_h2 \
      --checkpoint runs/clean_sweep/$target/checkpoint.pt \
      --config_name $cell \
      --out_dir results/probes/lens_h4/$target \
      --H_eval 4 --n_samples 5000
done

# Per-mMLRD lens (used by hero panel of Fig. 5).
for target in ouro-2x16_s42 ouro-4x8_s42; do
  cell=${target%_s*}
  uv run python -m looped_lm.logit_lens_h2 \
      --checkpoint runs/clean_sweep/$target/checkpoint.pt \
      --config_name $cell \
      --out_dir results/probes/lens_permlrd/$target \
      --H_eval 12 --n_samples 5000
done
```

### 3. Plot

```bash
# §4 figures (Fig. 2 and Fig. 3).
uv run python scripts/plot_h1_saturation.py \
    --runs_dir runs/clean_sweep \
    --outdir   figures/section4

# §5 figures (Fig. 4, 5, 6, A).
uv run python scripts/plot_h2_section5.py \
    --results_root results/probes \
    --out_dir      figures/section5
```

The output filenames produced by these two scripts are exactly the source
filenames embedded by `main.tex`.

## Repository layout

```
insidetheloop/
  pyproject.toml          uv-managed project (package: looped_lm)
  src/looped_lm/
    model.py              Transformer + LoopedTransformer (RoPE / NoPE, SwiGLU,
                          RMSNorm, weight-shared loop, zero-init projections,
                          learned per-iteration timestep embeddings)
    model_configs.py      architecture registry: ouro-LxK, deep-N, wide-L, etd-P_Lr_C
    train_uniform_h.py    fixed-K trainer with uniform-H sampling
    evaluate.py           per-hop accuracy, K-sweep, depth extrapolation
    probe_h2.py           per-iteration linear probes + Hewitt-Liang shuffled control
    logit_lens_h2.py      logit lens read through the model's own LM head
    utils.py              tokenizers, seeding, LR schedules
    figstyle.py           shared matplotlib style (CM serif, palette, markers)
    datasets/
      base.py             OnlineSyntheticDataset abstraction
      pointer_chasing.py  dePo task generator (random N-permutation, H-hop chain)
  scripts/
    plot_h1_saturation.py Fig. 2 + Fig. 3
    plot_h2_section5.py   Fig. 4, 5, 6, A
    run_probes.py         Probe orchestrator (one process per (cell, seed))
```

## Architectures

Selected by `--config_name` (full list in `looped_lm/model_configs.py`):

- `ouro-LxK` — looped transformer, `L` recurrent layers iterated `K` times. Effective depth `L*K`, parameter count scales with `L`.
- `deep-N` — unlooped baseline of `N` layers.
- `wide-L` — width-matched control.
- `etd-P_Lr_C` — prelude / loop / coda (Encoder-Through-Decoder) variant.

## Citation

```bibtex
@article{insidetheloop2026,
  title  = {Inside the Loop: A Mechanistic Study of Weight-Tied Transformers on Depth-Bound Algorithmic Tasks},
  author = {Anonymous},
  year   = {2026},
  note   = {Anonymous preprint}
}
```

## License

Apache-2.0 (see [LICENSE](LICENSE)).
