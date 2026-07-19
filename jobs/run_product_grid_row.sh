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
IFS=$'\t' read -r NAME ENCODER DECODER B N Q SEED <<< "$ROW"
EXTRA=()
if [[ "$ENCODER" == odma_fixed ]]; then
  EXTRA+=(--odma-d "$((N / Q))")
elif [[ "$ENCODER" == sparse_global_fixed ]]; then
  EXTRA+=(--sparse-support "$((N / Q))")
fi

OUT_DIR="$RESULT_ROOT/$NAME"
mkdir -p "$OUT_DIR"
DECODER_EPOCHS=20
if [[ "$DECODER" == d0 ]]; then
  DECODER_EPOCHS=80
fi
uv run python -m tests.framework_product_experiment \
  --encoder "$ENCODER" --decoder "$DECODER" -B "$B" --n "$N" --Q "$Q" \
  --num-antennas 1 --num-layers 8 --power-iters 12 \
  --encoder-epochs 40 --decoder-epochs "$DECODER_EPOCHS" --batches-per-epoch 100 --batch-size 8 \
  --train-ebn0-min -4 --train-ebn0-max 12 --eval-ebn0=-4,0,4,8,12 --eval-batches 4 \
  --extrapolate-k --seed "$SEED" --out-dir "$OUT_DIR" "${EXTRA[@]}"
