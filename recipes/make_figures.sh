#!/usr/bin/env bash
# Every figure of the paper from the shipped summaries (CPU, about a minute; no checkpoints needed).
# Each builder writes into figures/build/; the paper's figures are then copied to
#   figures/figure_{1..6}.{pdf,png} and figures/figure_A{1..6}.{pdf,png}
# with the numbering of the paper. Fig. 3 also writes figures/figure_3_counts.csv, and Fig. A2
# figures/figure_A2_onsets.json (the rows of Table A1). A builder whose input is missing fails.
#   OUT     output directory (default: ./figures)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="${OUT:-$ROOT/figures}"
BUILD="$OUT/build"
PY="uv run --no-sync python"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/.mplconfig}"
H1=experiments/h1_clean_sweep/scripts
H2=experiments/h2_probing/scripts

for builder in $H2/build_fig1_teaser.py $H2/plot_task_schematic.py $H1/capability_grid.py \
               $H2/build_fig4_layouts.py $H2/build_fig5_dynamics.py $H2/build_fig6_use.py \
               $H1/undertraining_figure.py $H2/build_appendix_probe_figures.py \
               $H1/n32_replication_figure.py $H2/build_figA5_anomalous_seed.py; do
    echo "+ $builder"
    $PY "$builder" --out_dir "$BUILD"
done

# paper figure <- builder output
while read -r name stem; do
    for ext in pdf png; do cp "$BUILD/$stem.$ext" "$OUT/$name.$ext"; done
done <<'MAP'
figure_1 fig1_teaser
figure_2 fig2_task
figure_3 capability_grid
figure_4 fig4_layouts
figure_5 fig5_dynamics
figure_6 fig6_use
figure_A1 fig_undertraining
figure_A2 figA2_every_seed_onsets
figure_A3 figA3_directional
figure_A4 fig_n32
figure_A5 figA5_anomalous_seed
figure_A6 figA6_patching_every_seed
MAP
cp "$BUILD/capability_grid_counts.csv" "$OUT/figure_3_counts.csv"
cp "$BUILD/figA2_every_seed_onsets.json" "$OUT/figure_A2_onsets.json"
echo "wrote $(ls "$OUT"/figure_*.pdf | wc -l | tr -d ' ') figures to $OUT"
