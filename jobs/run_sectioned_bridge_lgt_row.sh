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
IFS=$'\t' read -r NAME OUTER LEARN B J N SEED <<< "$ROW"
LEARN_ARGS=(--no-learn-encoder)
if [[ "$LEARN" == 1 ]]; then
  LEARN_ARGS=(--learn-encoder)
fi

OUT_DIR="$RESULT_ROOT/$NAME"
uv run python -m tests.framework_sectioned_bridge \
  --bridge lgt --outer-code "$OUTER" "${LEARN_ARGS[@]}" -B "$B" -J "$J" --n "$N" \
  --k-min 9 --k-max 26 --eval-k 9,17,26,30 \
  --encoder-steps 4000 --steps 8000 --batch-size 8 --eval-batches 4 \
  --train-ebn0-min -4 --train-ebn0-max 12 --eval-ebn0=-4,0,4,8,12 \
  --d0-layers 8 --bp-iterations 4 --power-iters 12 --num-path-negatives 32 \
  --beam-width 512 --list-extra 32 --mixing-stages 8 --log-every 500 \
  --seed "$SEED" --out-dir "$OUT_DIR"
