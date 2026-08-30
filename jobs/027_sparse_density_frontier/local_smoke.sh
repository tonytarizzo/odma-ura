#!/bin/bash
set -euo pipefail

RESULT_ROOT=${1:-results/sparse_density_frontier_smoke}
export SPARSITY_D0_EPOCHS=1
export SPARSITY_D1_EPOCHS=1
export SPARSITY_BATCHES_PER_EPOCH=2
export SPARSITY_EVAL_BATCHES=1
export SPARSITY_DIAGNOSTIC_PAIRS=300
export SPARSITY_DIAGNOSTIC_ACTIVE_SAMPLES=16

for INDEX in 1 2 61 62 65 66 69 70; do
  bash jobs/run_sparsity_grid_row.sh jobs/027_sparse_density_frontier/manifest.tsv "$RESULT_ROOT" "$INDEX"
done
uv run python -m tests.framework_sparsity_sweep_merge \
  --result-root "$RESULT_ROOT" \
  --manifest jobs/027_sparse_density_frontier/manifest.tsv \
  --allow-incomplete --out-dir "$RESULT_ROOT/merged"
echo "Smoke outputs: $RESULT_ROOT/merged"
