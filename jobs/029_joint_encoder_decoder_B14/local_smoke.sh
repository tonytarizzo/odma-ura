#!/bin/bash
set -euo pipefail

RESULT_ROOT=${1:-results/joint_learning_smoke}
uv run python -m tests.framework_hash_skeleton_test
uv run python -m tests.framework_product_experiment \
  --encoder hash_linear_selected_fixed --decoder d0 -B 6 --n 16 --sparse-support 4 \
  --learn-encoder --joint-train --hash-search-candidates 8 --k-min 2 --k-max 4 --eval-k 2,4 --eval-ebn0=8,12 \
  --num-layers 2 --power-iters 3 --encoder-epochs 0 --decoder-epochs 1 --batches-per-epoch 2 \
  --batch-size 3 --eval-batches 1 --diagnostic-pairs 100 --diagnostic-active-samples 8 \
  --diagnostic-active-gram-samples 4 --diagnostic-sum-pairs 4 --diagnose-before-training --seed 2810 \
  --train-seed 102810 --eval-seed 202810 --out-dir "$RESULT_ROOT/hash_d0"
uv run python - "$RESULT_ROOT/hash_d0/summary.json" <<'PY'
import json, math, pathlib, sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text()); meta = payload["metadata"]
assert payload["progress"] and payload["progress"][0]["phase"] == "joint"
assert meta["codebook_sparsity_initial"]["support_size"] == meta["codebook_sparsity"]["support_size"] == 4
assert meta["codebook_sparsity"]["max_unit_energy_deviation"] < 1e-6
print("joint smoke: finite training, fixed support, and final unit energy passed")
PY
echo "Smoke output: $RESULT_ROOT"
