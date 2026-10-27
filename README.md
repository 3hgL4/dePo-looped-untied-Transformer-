> **Note — audit fork (2026-10).** This repo is a copy of the MIT-licensed *Inside the Loop*
> release (NeurIPS 2026) plus a measurement-audit layer added on top. Start at
> **[AUDIT.md](AUDIT.md)**: four measurement defects found in the harness, five control
> experiments, three read-out traps, and the commands that reproduce every number.
> Upstream code and its shipped runs are unmodified; the original README follows.

# Inside the Loop

Code and results for **Inside the Loop: A Mechanistic Study of Weight-Tied Transformers on
Depth-Bound Algorithmic Tasks** (NeurIPS 2026).

Magnus Sesodia (University of Oxford), Juil Sock (BBC R&D), Philip Torr (University of Oxford),
Christian Schroeder de Witt (University of Oxford)

arXiv: to be added

The paper studies directed permutation traversal, in which a model follows H hops through a random
permutation of N nodes from a start node and reports the node it reaches, so every intermediate hop
has a known ground truth. On this task a minimum effective depth is necessary, and beyond it the
degree of weight tying decides the outcome: a looped transformer with a four-layer block iterated
eight times solves the task in every seed, while an identically trained untied transformer of the
same effective depth never does, and succeeds only with a lower learning rate or with supervision of
its intermediate layers. Probes and interventions on the residual stream show the looped model
advancing through the intermediate nodes gradually, in a staircase that adapts to its iteration
budget, while the untied model at the lower rate exposes them only in its final layers. This
repository holds what is needed to train the models, run the probes and rebuild every figure and
table in the paper.

## Installation

The project is managed with [uv](https://docs.astral.sh/uv/) (Python 3.10 to 3.12).

```bash
git clone https://github.com/magnus-sesodia/insidetheloop
cd insidetheloop
uv sync
uv run pytest -q          # CPU unit tests, under a minute
```

PyTorch is pinned to 2.5.1, the version the paper's runs used: newer releases reject the model's
attention call (a padding mask together with `is_causal=True`). On Linux, `pyproject.toml` installs
the build for CUDA 12.1 (driver 530 or newer); for another CUDA version, change the index URL at the
bottom of `pyproject.toml`. On macOS the default PyPI wheel is used and everything runs on CPU.
matplotlib is pinned to 3.10.8, the version that rendered the paper's figures.

The trainer and the probes download the tokenizer `hf-internal-testing/llama-tokenizer` from the
Hugging Face Hub on first use and cache it. On a machine without network access, copy the cache
entry (`~/.cache/huggingface/hub/models--hf-internal-testing--llama-tokenizer`) over from a machine
that has it and set `HF_HUB_OFFLINE=1`. The figure and table recipes do not need it.

## Quick start

```bash
bash recipes/smoke_test.sh      # every code path on a few training steps, CPU, about 5 min
bash recipes/make_figures.sh    # every figure, from the shipped results, CPU, about 10 s
bash recipes/make_tables.sh     # every table, from the shipped results, CPU, about 1 s
```

The smoke test trains tiny looped, untied, auxiliary-loss, no-tau_t and N=32 runs for a few steps,
resumes one of them, and runs both probe protocols, the full-answer scorer, the full-node logit lens,
activation patching and the forward-walk probe on the result. It writes to a fresh temporary
directory (or to `SMOKE_DIR`), and nothing it produces is a result.

`make_figures.sh` writes every builder's output to `figures/build/` and copies the paper's figures to
`figures/figure_<n>.pdf` and `.png` with the paper's numbering. With the pinned matplotlib the
rebuilt figures are pixel-identical to the figures in the paper. `make_tables.sh` rewrites the
Markdown and LaTeX files in `tables/`, which are committed, so `git status` stays clean after it.

## Figures and tables

In the paper's numbering. Builders marked h1 are in `experiments/h1_clean_sweep/scripts/` and h2 in
`experiments/h2_probing/scripts/`; inputs are relative to `experiments/`. Every builder takes
`--out_dir` and fails with `FileNotFoundError` naming the file when an input is missing.

| Release file | Paper | Builder | Inputs |
|---|---|---|---|
| `figures/figure_1` | Figure 1, probe maps of looped 4x8 and untied 32-layer at H=8, lr 3e-4 | h2 `build_fig1_teaser.py` | `h2_probing/results/matched_lr3e4_20260915` |
| `figures/figure_2` | Figure 2, the task | h2 `plot_task_schematic.py` | none |
| `figures/figure_3` | Figure 3, capability grid at lr 1e-3 (counts in `figure_3_counts.csv`) | h1 `capability_grid.py` | `h1_clean_sweep/runs`, `h1_clean_sweep/runs_gapfill_20261001` |
| `figures/figure_4` | Figure 4, probe maps at H=4 | h2 `build_fig4_layouts.py` | `h2_probing/results/geometry_lr1e3_20260920`, `clean_sweep_5seed_h4`, `matched_lr3e4_20260915`, `aux_rerun_h4`, `complete_answer_20261009` |
| `figures/figure_5` | Figure 5, how the loop spends its iterations at H=8 | h2 `build_fig5_dynamics.py` | `h2_probing/results/geometry_lr1e3_20260920`, `fig5_h8_lr1e3_20261009` |
| `figures/figure_6` | Figure 6, logit lens and activation patching | h2 `build_fig6_use.py` (panel c from `plot_patching_schematic.py`) | `h2_probing/results/full_node_lens_20261010`, `patching_fullanswer_digit_20261010` |
| `tables/capability_table.md` | Table 1, and the capability numbers in the text | `tables/capability_table.py` | every `h1_clean_sweep/runs*/*/trajectory.jsonl` |
| `figures/figure_A1` | Figure A1, training loss at matched effective depth | h1 `undertraining_figure.py` | `h1_clean_sweep/runs/*/train.log` |
| `figures/figure_A2` | Figure A2, first decodable layer for every seed (rows in `figure_A2_onsets.json`) | h2 `build_appendix_probe_figures.py` | `h2_probing/results/clean_sweep_5seed_h4`, `geometry_lr1e3_20260920`, `matched_lr3e4_20260915`, `aux_rerun_h4`, `complete_answer_20261009` |
| `figures/figure_A3` | Figure A3, forward-walk probes | h2 `build_appendix_probe_figures.py` | `h2_probing/results/rebuttal/dirprobe_o2x16_h12` |
| `figures/figure_A4` | Figure A4, longer chains at N=32 | h1 `n32_replication_figure.py` | `h1_clean_sweep/runs_n32`, `h1_clean_sweep/runs_n32_rerun_20261005` |
| `figures/figure_A5` | Figure A5, a looped seed with weak middle hops | h2 `build_figA5_anomalous_seed.py` | `h2_probing/results/rebuttal/probe_o2x16_s*`, `full_node_lens_20261010` |
| `figures/figure_A6` | Figure A6, activation patching, three seeds | h2 `build_fig6_use.py` | `h2_probing/results/patching_fullanswer_digit_20261010` |
| `tables/every_seed_table.md`, `.tex` | Table A1 | `tables/every_seed_table.py` | the rows of Figure A2 |
| `tables/retention.md` | the retention paragraph of the probing appendix | `tables/retention.py` | `h2_probing/results/matched_lr3e4_20260915`, `geometry_lr1e3_20260920`, `clean_sweep_5seed` |

The hyperparameters in the appendix are not computed: they are the settings in `recipes/common.sh`,
the model definitions in `src/model_configs.py` and the `args.json` of each run.

## Full reproduction

Reproduction runs in three stages: train, analyse, then plot. Each stage is a shell recipe in
`recipes/` with the exact commands, seeds and flags of the paper's runs (shared settings in
`recipes/common.sh`). A recipe runs its jobs one after another on one GPU; `DRY_RUN=1` prints the
commands instead, so they can be handed to any launcher, one job per GPU.

```bash
RUNS=/big/disk/runs GPU=0 bash recipes/train_grid.sh
DRY_RUN=1 bash recipes/train_rate_controls.sh      # print the commands
uv run python tools/check_recipes.py               # recipes against the shipped run records
```

`RUNS` (default `./runs`) receives checkpoints, `PROBES` (default `./probe_outputs`) receives the
analysis outputs, and `STEPS` (default 100000) sets the training budget. Every training run keeps one
rolling `checkpoint.pt` (about 40 MB for looped 4x8, 300 MB for untied 32-layer) beside its
`args.json`, `trajectory.jsonl` and `train.log`. The analyses read the `args.json` next to a
checkpoint, so keep them together and use the same `RUNS` for training and analysis. The shipped
`train.log` files of the three matched-depth models at lr 1e-3 (looped 4x8 and 2x16, untied
32-layer, 15 runs) record 329 to 331 minutes per 100k-step run on one NVIDIA H200. The shipped files
do not record the time of runs on other GPUs.

### 1. Training

| Recipe | Runs | What |
|---|---|---|
| `train_grid.sh` | 120 | capability grid at lr 1e-3: untied 2, 4, 8, 16 and 32 layers; looped 2x{2,4,5,6,7,8,10,12,14,16}, 4x{2,3,4,6,8}, 8x{2,3,4} and 16x2; seeds 42 to 46 |
| `train_rate_controls.sh` | 25 | looped 4x8 and untied 32-layer at lr 3e-4 and 1e-4, untied 16-layer at 3e-4; seeds 42 to 46 |
| `train_aux.sh` | 5 | untied 32-layer at lr 1e-3 with the auxiliary loss at layers 8, 16 and 24; seeds 42 to 46 |
| `train_notau.sh` | 4 | looped 4x8 at lr 1e-3 without the iteration embedding tau_t; seeds 42 to 45 |
| `train_n32.sh` | 10 | N=32 at lr 3e-4: looped 4x8 and untied 32-layer (seeds 42 to 44), untied 4-layer (42, 43, 46, 47) |

The trainer is `src/train_uniform_h.py`. Looped runs zero-initialise their output projections and
untied runs do not; the recipes pass `--zero_init` or `--no_zero_init` explicitly.

### 2. Analysis

| Recipe | Outputs | What |
|---|---|---|
| `probe_prompt_disjoint.sh` | 37 | prompt-disjoint linear probes after every layer, with full-answer scoring |
| `probe_rowsplit.sh` | 26 | the earlier probes split by query row, with an untrained control |
| `eval_full_answer.sh` | 18 | full-answer accuracy at H=4 of the checkpoints probed with the row split |
| `lens.sh` | 28 | full-node logit lens, with untrained controls |
| `patching.sh` | 22 | activation patching (redirect and compose), with untrained controls |
| `directional.sh` | 1 | forward-walk probe of looped 2x16 at H=12 |

Each recipe's header lists the runs it reads. `tools/check_recipes.py` dry-runs every recipe,
parses each command with its script's own argument parser and compares the settings with the
shipped records: 164 training runs and 132 analysis outputs, with 0 mismatches.

### 3. Figures and tables from your own runs

The builders read the shipped records under `experiments/`. To plot your own runs, copy each output
over the shipped file at the same relative path: a run's `args.json`, `trajectory.jsonl` and
`train.log` into the folder of the same name under `experiments/h1_clean_sweep/runs*/`, and the
contents of `$PROBES/<name>/` into `experiments/h2_probing/results/<name>/`. Then rerun
`make_figures.sh` and `make_tables.sh`.

## Model names

| Config or run name | Paper |
|---|---|
| `ouro-LxT` | looped L x T: one block of L weight-shared layers iterated T times, effective depth D_eff = L T |
| `deep-N` | untied N-layer model, each layer applied once |
| `-lr3e4`, `-lr1e4` | learning rate 3e-4 or 1e-4 (no suffix: 1e-3) |
| `-notau` | looped model without the iteration embedding tau_t (`--no_timestep_embedding`) |
| `-aux` | untied model with the auxiliary next-token loss at layers 8, 16 and 24, total weight 0.3 (`--aux_deep_supervision`) |
| `_n32` | the task at N=32 (the default is N=16) |
| `_s42` to `_s47` | seed |

`uv run python -m src.model_configs` prints every configuration with its effective depth and
parameter count.

## Accuracy metrics

Two accuracies appear in the paper and in the shipped files.

- **Full-answer accuracy on the first query** is reported for trained models (probes, logit lens,
  patching, and the accuracies beside the probe maps). Each prompt holds five queries. The first is
  scored, and an answer counts as correct only if every digit and the delimiter that follows it are
  correct under teacher forcing. Always writing the same node scores 1/16.
- **The training-time evaluation** (`src/evaluate.py`, logged in `trajectory.jsonl` every 4k steps
  on 8,000 prompts, about 533 per hop length) scores the digits of the last query's answer. The
  queries of a prompt share H and can repeat a start node, so a model that copies earlier answers
  scores about 0.36 at N=16. A seed **solves** the task when its final evaluation reaches 0.8 on every
  hop length from 1 to 15, and its solved-chain depth H* is the largest H such that every hop length
  up to H reaches 0.8. Figure 3, Table 1 and Figure A4 use this evaluation.

## Determinism

Training is deterministic on a given GPU type: the same seed and recipe give bit-identical
trajectories. The paper's runs were trained on NVIDIA H200 and A40 GPUs. The same seed on the other
type diverges within the first few hundred steps and, for some looped seeds, ends at a different
solved-chain depth, so a rerun on different hardware is a new sample of that seed.

Eleven runs were resumed after an interruption by a version of the trainer that did not yet replay
the data stream: looped 4x8 at lr 3e-4 (seeds 42 to 44) and 1e-4 (42 to 44), and untied 32-layer at
lr 3e-4 (43, 44) and 1e-4 (42 to 44). Their `args.json` records `"resume": true`. The current trainer
replays the data stream on `--resume`, so an uninterrupted rerun of these runs sees different
batches after the resume step. `tools/check_recipes.py` lists them as notes.

## Shipped data

The repository ships the summaries behind every figure and table (about 6 MB) and no checkpoints:

- `experiments/h1_clean_sweep/runs*/`: `args.json`, `trajectory.jsonl` and `model_config.json` of
  every training run, and `train.log` of the matched-depth runs at lr 1e-3.
- `experiments/h2_probing/results/`: probe, full-answer, logit-lens, patching and forward-walk
  summaries, without per-query arrays.

Absolute paths in the shipped JSON files are replaced by `<storage>` or made relative to the
repository. `data/MANIFEST.sha256` lists the SHA-256 of every shipped data file:

```bash
shasum -a 256 -c data/MANIFEST.sha256    # or sha256sum -c on Linux
```

## Layout

```
src/                           models, task generator, trainer, probes, logit lens, patching
  model.py, model_configs.py   looped (ouro-LxT) and untied (deep-N) transformers
  datasets/pointer_chasing.py  directed permutation traversal
  train_uniform_h.py           the trainer (uniform-H sampling with the hop curriculum)
  evaluate.py                  the training-time evaluation
  probe_h2.py                  linear probes, both protocols
  full_node_lens.py            full-node logit lens
  intervention_h2.py           activation patching
  directional_probe_h2.py      forward-walk probe
  figstyle.py                  shared figure style
experiments/
  h1_clean_sweep/              training records; capability, loss and N=32 figure builders
  h2_probing/                  probe and intervention summaries; probe figure builders
recipes/                       one shell recipe per training cohort and analysis, plus
                               smoke_test.sh, make_figures.sh and make_tables.sh
tables/                        table builders and their committed outputs
data/MANIFEST.sha256           checksums of the shipped data
tests/                         CPU unit tests
tools/check_recipes.py         recipes against the shipped records
tools/check_release.py         scans the tree for machine names and absolute paths
```

The experiment folders keep the project's working names so that every path recorded in the shipped
results stays valid.

## Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{sesodia_inside_2026,
  title     = {Inside the Loop: A Mechanistic Study of Weight-Tied Transformers on Depth-Bound Algorithmic Tasks},
  author    = {Sesodia, Magnus and Sock, Juil and Torr, Philip and Schroeder de Witt, Christian},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

`CITATION.cff` holds the same reference for GitHub's "Cite this repository" button.

## License

MIT, see `LICENSE`.
