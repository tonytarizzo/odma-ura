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
IFS=$'\t' read -r NAME DECODER B N SUPPORT J MODE INIT SEED SEARCH_CANDIDATES <<< "$ROW"
OUT_DIR="$RESULT_ROOT/$NAME"
mkdir -p "$OUT_DIR"
COMMAND=(uv run --no-sync python -m tests.framework_product_experiment
  --encoder hash_prototype --decoder "$DECODER" -B "$B" --n "$N" --Q 1
  --sparse-support "$SUPPORT" --amplitude-label-bits "$J" --amplitude-init "$INIT"
  --hash-search-candidates "$SEARCH_CANDIDATES" --num-antennas 1 --num-layers 8 --power-iters 12
  --encoder-epochs 0 --decoder-epochs "${PROTOTYPE_EPOCHS:-120}"
  --batches-per-epoch "${PROTOTYPE_BATCHES_PER_EPOCH:-100}" --batch-size "${PROTOTYPE_BATCH_SIZE:-8}"
  --validation-batches "${PROTOTYPE_VALIDATION_BATCHES:-8}"
  --early-stopping-patience "${PROTOTYPE_PATIENCE:-5}" --early-stopping-min-delta 0
  --train-ebn0-min -4 --train-ebn0-max 12 --eval-ebn0=-4,0,4,8,12
  --eval-batches "${PROTOTYPE_EVAL_BATCHES:-16}" --extrapolate-k
  --diagnostic-pairs "${PROTOTYPE_DIAGNOSTIC_PAIRS:-20000}"
  --diagnostic-active-samples "${PROTOTYPE_DIAGNOSTIC_ACTIVE_SAMPLES:-256}"
  --diagnostic-active-gram-samples "${PROTOTYPE_DIAGNOSTIC_GRAM_SAMPLES:-64}"
  --diagnostic-sum-pairs "${PROTOTYPE_DIAGNOSTIC_SUM_PAIRS:-64}" --diagnose-before-training
  --seed "$SEED" --train-seed "$((SEED + 100000))" --validation-seed "$((SEED + 200000))"
  --eval-seed "$((SEED + 300000))" --out-dir "$OUT_DIR")
if [[ "$MODE" == learned ]]; then
  COMMAND+=(--learn-encoder --joint-train)
elif [[ "$MODE" != fixed ]]; then
  echo "Unknown amplitude mode '$MODE'" >&2
  exit 2
fi
"${COMMAND[@]}"
