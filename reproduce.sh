#!/usr/bin/env bash
# Maps paper figures to the commands that produce them.
# Usage: ./reproduce.sh <target>    (run with no args to list targets)
# Setup:  python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}

case "${1:-list}" in
  # --- v5 paper (Residual_Stream_Component_Grokking.pdf) ---
  v5-figs)
    # Figures 1-5 (causal addition d256, causal multiplication, overlap summary,
    # layer sweep, cross-task summary) -> ./figures/*.png|pdf.
    # NOTE: plot_paper_figures.py plots numbers hardcoded from earlier runs;
    # it does not recompute them from checkpoints.
    $PY plot_paper_figures.py ;;
  dyck-30seed)
    # Dyck 30-seed contrastive phase cancellation (results/dyck_30seed/).
    $PY train_dyck_30seed.py
    $PY analyze_dyck_directions.py ;;
  joint4)
    # GPT-2 Dyck-in-Text joint-4 cancellation (v6 sec 4.4; feeds Fig 8).
    # Requires prior gpt2_multi_results/gpu*/seed_*/dirs.npy and 8 GPUs.
    (cd experiments/joint4 && $PY run_joint4.py) ;;
  # --- v6 paper (papers/Residual_Stream_Component_Grokking_v6.md) ---
  v6-fig5|v6-fig6|v6-fig7|v6-fig8|v6-fig9)
    # TODO: no plotting script for v6 Figures 5-9 is in this repo
    # (capacity scaling, dual-80, sequential GPT-2, single/pair/joint-4,
    # concentrated vs distributed). Source data: results/csv/*.csv and the
    # result archives listed in INDEX.md. The dual-interference, 256/512x4
    # capacity, and GPT-2 multi-direction training/sweep scripts are also not here.
    echo "TODO: $1 has no generating script in this repo (see papers/FIGURES.md)" >&2; exit 2 ;;
  v6-tables)
    # TODO: results/SUMMARY.md tables are not produced by any script here.
    echo "TODO: no script generates results/SUMMARY.md tables" >&2; exit 2 ;;
  list|*)
    echo "targets: v5-figs dyck-30seed joint4 v6-fig5..v6-fig9 (TODO) v6-tables (TODO)" ;;
esac
