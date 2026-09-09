#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${LOCAL_OUT:-$(mktemp -d /tmp/ura-prototype-mini.XXXXXX)}
cd "$ROOT"

for DECODER in d0 d1; do
  for J in 0 4 8; do
    for MODE in fixed learned; do
      COMMAND=(uv run python -m tests.framework_product_experiment --encoder hash_prototype --decoder "$DECODER" \
        -B 8 --n 64 --sparse-support 8 --amplitude-label-bits "$J" --amplitude-init gaussian \
        --hash-search-candidates 8 --num-layers 4 --power-iters 4 --k-min 3 --k-max 6 --eval-k 3,6 \
        --eval-ebn0 4,8 --decoder-epochs 20 --batches-per-epoch 10 --batch-size 8 \
        --validation-batches 4 --eval-batches 3 --early-stopping-patience 5 --seed 29 \
        --out-dir "$OUT/${DECODER}_J${J}_${MODE}")
      if [[ "$MODE" == learned ]]; then COMMAND+=(--learn-encoder --joint-train); fi
      "${COMMAND[@]}"
    done
  done
done

uv run python - "$OUT" <<'PY'
import json, math, pathlib, sys
root = pathlib.Path(sys.argv[1]); improved = 0; total = 0
for path in sorted(root.glob("*/summary.json")):
    payload = json.loads(path.read_text()); values = [row["validation_total"] for row in payload["progress"]]
    assert values and all(math.isfinite(value) for value in values)
    improved += min(values) < values[0] - 1e-5; total += 1
    assert payload["metadata"]["amplitude_prototypes"]["max_energy_deviation"] < 1e-5
assert improved >= total // 2, f"only {improved}/{total} mini runs improved validation loss"
print(f"prototype mini passed: validation loss improved in {improved}/{total} runs; outputs at {root}")
PY
