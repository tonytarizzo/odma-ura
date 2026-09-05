#!/bin/bash
set -euo pipefail

RESULT_ROOT=${1:-results/hash_skeleton_mini}
FAMILIES=(dense_fixed sparse_iid_fixed hash_table_random_fixed hash_linear_random_fixed hash_linear_selected_fixed)
for FAMILY in "${FAMILIES[@]}"; do
  for DECODER in d0 d1; do
    EXTRA=(--Q 1)
    if [[ "$FAMILY" != dense_fixed ]]; then EXTRA+=(--sparse-support 8); fi
    EPOCHS=2
    if [[ "$DECODER" == d0 ]]; then EPOCHS=3; fi
    uv run python -m tests.framework_product_experiment \
      --encoder "$FAMILY" --decoder "$DECODER" -B 8 --n 64 "${EXTRA[@]}" \
      --hash-search-candidates 16 --k-min 3 --k-max 10 --eval-k 3,6,10 --eval-ebn0=8,12 \
      --num-layers 4 --power-iters 5 --encoder-epochs 0 --decoder-epochs "$EPOCHS" --batches-per-epoch 5 \
      --batch-size 4 --eval-batches 2 --diagnostic-pairs 400 --diagnostic-active-samples 16 \
      --diagnostic-active-gram-samples 8 --diagnostic-sum-pairs 8 --seed 2820 \
      --train-seed 102820 --eval-seed 202820 --out-dir "$RESULT_ROOT/${FAMILY}_${DECODER}"
  done
done
uv run python - "$RESULT_ROOT" <<'PY'
import json, math, pathlib, sys
root = pathlib.Path(sys.argv[1]); summaries = sorted(root.glob("*/summary.json"))
assert len(summaries) == 10, f"expected 10 summaries, found {len(summaries)}"
for path in summaries:
    payload = json.loads(path.read_text()); progress = payload["progress"]
    assert progress and all(math.isfinite(float(row["total"])) for row in progress)
    high = [float(row["pupe"]) for row in payload["learned"] if float(row["ebn0_db"]) >= 8]
    assert high and all(0.0 <= value <= 1.0 for value in high)
    print(f"{path.parent.name:42s} loss {progress[0]['total']:.4f}->{progress[-1]['total']:.4f} high-SNR PUPE={sum(high)/len(high):.4f}")
PY
echo "Mini outputs: $RESULT_ROOT"
