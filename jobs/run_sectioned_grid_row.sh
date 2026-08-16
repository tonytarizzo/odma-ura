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
IFS=$'\t' read -r NAME B J N K OUTER PARITY DEGREE ENERGY BANK LEARN STEPS WARMUP BATCH EVAL_BATCHES EVAL_EBN0 CANDIDATE SEED <<< "$ROW"

ARGS=(--preset custom -B "$B" -J "$J" --n "$N" --num-active "$K" --outer-code "$OUTER"
      --energy-mode "$ENERGY" --bank-type "$BANK" --steps "$STEPS" --outer-warmup-steps "$WARMUP"
      --batch-size "$BATCH" --eval-batches "$EVAL_BATCHES" --eval-ebn0 "$EVAL_EBN0"
      --d0-layers 8 --bp-iterations 4 --power-iters 12 --num-path-negatives 32 --log-every 25
      --beam-width 512 --list-extra 32 --mixing-stages 8 --seed "$SEED" --no-assert-reasonable)
if [[ "$OUTER" == random_sparse ]]; then
  ARGS+=(--num-parity-sections "$PARITY" --check-degree "$DEGREE")
fi
if [[ "$LEARN" == 1 ]]; then
  ARGS+=(--learn-encoder)
else
  ARGS+=(--no-learn-encoder)
fi
if [[ "$CANDIDATE" != none ]]; then
  ARGS+=(--candidate-cap "$CANDIDATE")
fi

OUT_DIR="$RESULT_ROOT/$NAME"
uv run python -m tests.framework_sectioned_learning "${ARGS[@]}" --out-dir "$OUT_DIR"
