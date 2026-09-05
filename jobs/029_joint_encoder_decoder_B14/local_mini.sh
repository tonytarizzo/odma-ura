#!/bin/bash
set -euo pipefail

RESULT_ROOT=${1:-results/joint_learning_mini}
FAMILIES=(dense_fixed sparse_iid_fixed hash_linear_selected_fixed)
for FAMILY in "${FAMILIES[@]}"; do
  for DECODER in d0 d1; do
    EXTRA=(--Q 1)
    if [[ "$FAMILY" != dense_fixed ]]; then EXTRA+=(--sparse-support 8); fi
    uv run python -m tests.framework_product_experiment \
      --encoder "$FAMILY" --decoder "$DECODER" -B 8 --n 64 "${EXTRA[@]}" \
      --learn-encoder --joint-train --hash-search-candidates 16 --k-min 3 --k-max 10 --eval-k 3,6,10 --eval-ebn0=8,12 \
      --num-layers 4 --power-iters 5 --encoder-epochs 0 --decoder-epochs 3 --batches-per-epoch 5 \
      --batch-size 4 --eval-batches 2 --diagnostic-pairs 400 --diagnostic-active-samples 16 \
      --diagnostic-active-gram-samples 8 --diagnostic-sum-pairs 8 --diagnose-before-training --seed 2920 \
      --train-seed 102920 --eval-seed 202920 --out-dir "$RESULT_ROOT/${FAMILY}_${DECODER}"
  done
done
uv run python - "$RESULT_ROOT" <<'PY'
import json, math, pathlib, sys
root = pathlib.Path(sys.argv[1]); summaries = sorted(root.glob("*/summary.json"))
assert len(summaries) == 6, f"expected 6 summaries, found {len(summaries)}"
for path in summaries:
    payload = json.loads(path.read_text()); progress = payload["progress"]; meta = payload["metadata"]
    assert len(progress) == 3 and all(row["phase"] == "joint" and math.isfinite(float(row["total"])) for row in progress)
    assert meta["codebook_sparsity_initial"]["support_size"] == meta["codebook_sparsity"]["support_size"]
    assert float(meta["codebook_sparsity"]["max_unit_energy_deviation"]) < 1e-6
    high = [float(row["pupe"]) for row in payload["learned"]]
    assert high and all(0.0 <= value <= 1.0 for value in high)
    print(f"{path.parent.name:40s} loss {progress[0]['total']:.4f}->{progress[-1]['total']:.4f} PUPE={sum(high)/len(high):.4f}")
PY
echo "Mini outputs: $RESULT_ROOT"
