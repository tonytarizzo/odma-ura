#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${LOCAL_OUT:-$(mktemp -d /tmp/ura-prototype-smoke.XXXXXX)}
cd "$ROOT"

COMMON=(--encoder hash_prototype -B 6 --n 16 --sparse-support 4 --hash-search-candidates 4
  --num-layers 2 --power-iters 2 --k-min 2 --k-max 3 --eval-k 2 --eval-ebn0 4
  --decoder-epochs 2 --batches-per-epoch 2 --batch-size 2 --validation-batches 2 --eval-batches 1
  --early-stopping-patience 5 --diagnostic-pairs 8)
uv run python -m tests.framework_product_experiment "${COMMON[@]}" --decoder d0 --amplitude-label-bits 0 \
  --amplitude-init equal --seed 17 --out-dir "$OUT/fixed_equal_d0"
uv run python -m tests.framework_product_experiment "${COMMON[@]}" --decoder d1 --amplitude-label-bits 2 \
  --amplitude-init gaussian --learn-encoder --joint-train --seed 17 --out-dir "$OUT/learned_J2_d1"
uv run python - "$OUT" <<'PY'
import json, math, pathlib, sys
root = pathlib.Path(sys.argv[1])
for path in sorted(root.glob("*/summary.json")):
    payload = json.loads(path.read_text())
    assert payload["progress"] and all(math.isfinite(row["validation_total"]) for row in payload["progress"])
    assert payload["metadata"]["amplitude_prototypes"]["max_energy_deviation"] < 1e-5
print(f"prototype smoke passed: {root}")
PY
