#!/bin/bash
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=8:mem=64gb
#PBS -N geom015_B12_n512
#PBS -o jobs/015_geometry_B12_n512/015_geometry_B12_n512.o
#PBS -e jobs/015_geometry_B12_n512/015_geometry_B12_n512.e

# Decoder-free geometry optimisation plus paired decoder evaluation at B=12.
# This repeats the local B=8,n=128 geometry study in a larger alphabet regime:
# M=4096, n=512, K=8, ODMA d/n=1/4. The larger M keeps repeated-message
# collisions negligible, so before/after decoder changes are less likely to be
# finite-alphabet collision artefacts.

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

module load miniforge/3
cd "${PBS_O_WORKDIR:-$HOME/odma-ura}"

JOB_DIR="jobs/015_geometry_B12_n512"
OUT_DIR="$JOB_DIR/results"
mkdir -p "$OUT_DIR"

COMMON_OPT_ARGS=(
  -B 12
  --n 512
  --active-k 8
  --steps 1000
  --batch-supports 32
)

COMMON_ODMA_ARGS=(
  --preset odma
  --d 128
  --num-blocks 4
)

COMMON_EVAL_ARGS=(
  --num-samples 5000
  --batch-size 100
  --ebn0-grid -4 -2 0 2 4 6 8
)

echo "=== ODMA AMP geometry optimisation ==="
uv run python -m tests.framework_geometry_optimisation \
  "${COMMON_ODMA_ARGS[@]}" \
  --objective amp \
  "${COMMON_OPT_ARGS[@]}" \
  --out-dir "$OUT_DIR/geometry_odma_amp_B12_n512"

echo "=== ODMA AMP paired decoder evaluation ==="
uv run python -m tests.framework_geometry_decoder_eval \
  --run-dir "$OUT_DIR/geometry_odma_amp_B12_n512" \
  "${COMMON_EVAL_ARGS[@]}"

echo "=== ODMA support-margin geometry optimisation ==="
uv run python -m tests.framework_geometry_optimisation \
  "${COMMON_ODMA_ARGS[@]}" \
  --objective support_margin \
  "${COMMON_OPT_ARGS[@]}" \
  --out-dir "$OUT_DIR/geometry_odma_margin_B12_n512"

echo "=== ODMA support-margin paired decoder evaluation ==="
uv run python -m tests.framework_geometry_decoder_eval \
  --run-dir "$OUT_DIR/geometry_odma_margin_B12_n512" \
  "${COMMON_EVAL_ARGS[@]}"

echo "=== ODMA VAMP geometry optimisation ==="
uv run python -m tests.framework_geometry_optimisation \
  "${COMMON_ODMA_ARGS[@]}" \
  --objective vamp \
  "${COMMON_OPT_ARGS[@]}" \
  --out-dir "$OUT_DIR/geometry_odma_vamp_B12_n512"

echo "=== ODMA VAMP paired decoder evaluation ==="
uv run python -m tests.framework_geometry_decoder_eval \
  --run-dir "$OUT_DIR/geometry_odma_vamp_B12_n512" \
  "${COMMON_EVAL_ARGS[@]}"

echo "=== Dense AMP geometry optimisation ==="
uv run python -m tests.framework_geometry_optimisation \
  --preset dense \
  --objective amp \
  "${COMMON_OPT_ARGS[@]}" \
  --out-dir "$OUT_DIR/geometry_dense_amp_B12_n512"

echo "=== Dense AMP paired decoder evaluation ==="
uv run python -m tests.framework_geometry_decoder_eval \
  --run-dir "$OUT_DIR/geometry_dense_amp_B12_n512" \
  "${COMMON_EVAL_ARGS[@]}"

echo "=== Geometry job complete ==="
