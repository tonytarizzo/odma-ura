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
IFS=$'\t' read -r NAME ENCODER DECODER B N SUPPORT Q SEED <<< "$ROW"
OUT_DIR="$RESULT_ROOT/$NAME"
mkdir -p "$OUT_DIR"
DECODER_EPOCHS=${SPARSITY_D1_EPOCHS:-20}
if [[ "$DECODER" == d0 ]]; then
  DECODER_EPOCHS=${SPARSITY_D0_EPOCHS:-80}
fi
COMMAND=(uv run python -m tests.framework_product_experiment
  --encoder "$ENCODER" --decoder "$DECODER" -B "$B" --n "$N" --Q "$Q"
  --num-antennas 1 --num-layers 8 --power-iters 12
  --encoder-epochs 0 --decoder-epochs "$DECODER_EPOCHS"
  --batches-per-epoch "${SPARSITY_BATCHES_PER_EPOCH:-100}" --batch-size "${SPARSITY_BATCH_SIZE:-8}"
  --train-ebn0-min -4 --train-ebn0-max 12 --eval-ebn0=-4,0,4,8,12
  --eval-batches "${SPARSITY_EVAL_BATCHES:-8}" --extrapolate-k
  --diagnostic-pairs "${SPARSITY_DIAGNOSTIC_PAIRS:-30000}"
  --diagnostic-active-samples "${SPARSITY_DIAGNOSTIC_ACTIVE_SAMPLES:-256}"
  --seed "$SEED" --train-seed "$((SEED + 100000))" --eval-seed "$((SEED + 200000))" --out-dir "$OUT_DIR")
if [[ "$ENCODER" == sparse_global_fixed ]]; then
  COMMAND+=(--sparse-support "$SUPPORT" --sparse-nested)
elif [[ "$ENCODER" == odma_fixed ]]; then
  COMMAND+=(--odma-d "$SUPPORT")
fi
"${COMMAND[@]}"
