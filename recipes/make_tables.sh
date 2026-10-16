#!/usr/bin/env bash
# Every table and the capability and retention numbers in the text, from the shipped summaries (CPU,
# a few seconds; no checkpoints needed). Writes
#   tables/capability_table.md   Table 1, the first-solve steps, the tau_t ablation and every seed's H*
#   tables/every_seed_table.md   Table A1 (and its LaTeX rows, every_seed_table.tex)
#   tables/retention.md          the retention paragraph of App. "Probing details"
# A script whose input is missing fails.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="uv run --no-sync python"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/.mplconfig}"
for script in tables/capability_table.py tables/every_seed_table.py tables/retention.py; do
    echo "+ $script"
    $PY "$script" > /dev/null
done
echo "wrote tables/capability_table.md, tables/every_seed_table.{md,tex}, tables/retention.md"
