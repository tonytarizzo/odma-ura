#!/bin/bash
set -euo pipefail

MANIFEST=$1
RESULT_ROOT=$2
INDEX=${PBS_ARRAY_INDEX:-${3:-1}}
ROW=$(awk -F '\t' -v i="$INDEX" 'NR == i + 1 {print; exit}' "$MANIFEST")
if [[ -z "$ROW" ]]; then
  echo "No manifest row for array index $INDEX in $MANIFEST" >&2
  exit 2
fi
IFS=$'\t' read -r NAME ENCODER DECODER B N SUPPORT SEED SEARCH_CANDIDATES <<< "$ROW"
OUT_DIR="$RESULT_ROOT/$NAME"
mkdir -p "$OUT_DIR"
COMMAND=(uv run --no-sync python -m tests.framework_product_experiment
  --encoder "$ENCODER" --decoder "$DECODER" -B "$B" --n "$N" --Q 1
  --learn-encoder --joint-train --num-antennas 1 --num-layers 8 --power-iters 12
  --encoder-epochs 0 --decoder-epochs "${JOINT_EPOCHS:-120}"
  --batches-per-epoch "${JOINT_BATCHES_PER_EPOCH:-100}" --batch-size "${JOINT_BATCH_SIZE:-8}"
  --train-ebn0-min -4 --train-ebn0-max 12 --eval-ebn0=-4,0,4,8,12
  --eval-batches "${JOINT_EVAL_BATCHES:-16}" --extrapolate-k
  --diagnostic-pairs "${JOINT_DIAGNOSTIC_PAIRS:-20000}"
  --diagnostic-active-samples "${JOINT_DIAGNOSTIC_ACTIVE_SAMPLES:-256}"
  --diagnostic-active-gram-samples "${JOINT_DIAGNOSTIC_GRAM_SAMPLES:-64}"
  --diagnostic-sum-pairs "${JOINT_DIAGNOSTIC_SUM_PAIRS:-64}" --diagnose-before-training
  --hash-search-candidates "$SEARCH_CANDIDATES"
  --seed "$SEED" --train-seed "$((SEED + 100000))" --eval-seed "$((SEED + 200000))" --out-dir "$OUT_DIR")
if [[ "$ENCODER" != dense_fixed ]]; then
  COMMAND+=(--sparse-support "$SUPPORT")
fi
"${COMMAND[@]}"
