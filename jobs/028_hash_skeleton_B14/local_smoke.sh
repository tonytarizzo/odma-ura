#!/bin/bash
set -euo pipefail

RESULT_ROOT=${1:-results/hash_skeleton_smoke}
uv run python -m tests.framework_hash_skeleton_test
for DECODER in d0 d1; do
  uv run python -m tests.framework_product_experiment \
    --encoder hash_linear_selected_fixed --decoder "$DECODER" -B 6 --n 16 --sparse-support 4 \
    --hash-search-candidates 8 --k-min 2 --k-max 4 --eval-k 2,4 --eval-ebn0=8,12 \
    --num-layers 2 --power-iters 3 --encoder-epochs 0 --decoder-epochs 1 --batches-per-epoch 2 \
    --batch-size 3 --eval-batches 1 --diagnostic-pairs 100 --diagnostic-active-samples 8 \
    --diagnostic-active-gram-samples 4 --diagnostic-sum-pairs 4 --seed 2810 \
    --train-seed 102810 --eval-seed 202810 --out-dir "$RESULT_ROOT/$DECODER"
done
echo "Smoke outputs: $RESULT_ROOT"
