#!/bin/bash
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -N ura020_ccs_amp_B100
#PBS -o jobs/020_CCS_AMP_B100/020_CCS_AMP_B100.o
#PBS -e jobs/020_CCS_AMP_B100/020_CCS_AMP_B100.e

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

module load miniforge/3
cd "${PBS_O_WORKDIR:-$HOME/odma-ura}"

AUTHOR_DIR=".cache/CCS-AMP-Code"
AUTHOR_COMMIT="92080d85408d5d19a123d1d61ba76ec6f15451a5"
if [[ ! -d "$AUTHOR_DIR/.git" ]]; then
  git clone https://github.com/vamsi128/CCS-AMP-Code.git "$AUTHOR_DIR"
fi
git -C "$AUTHOR_DIR" checkout "$AUTHOR_COMMIT"

RESULT_ROOT="jobs/020_CCS_AMP_B100/results"
mkdir -p "$RESULT_ROOT"
pids=()
for K in 10 25 50 75 100 125 150 175; do
  uv run python -m tests.ccs_amp_author_curve \
    --preset adapted_b100 \
    --author-code-dir "$AUTHOR_DIR" \
    --K-values "$K" \
    --ebn0-grid 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 \
    --num-seeds 20 \
    --amp-iterations 10 \
    --bp-iterations 1 \
    --schemes enhanced original \
    --out-dir "$RESULT_ROOT/K$K" > "$RESULT_ROOT/K$K.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

uv run python -m tests.ccs_amp_merge --input-dir "$RESULT_ROOT" --out-dir "$RESULT_ROOT/merged" --preset adapted_b100
