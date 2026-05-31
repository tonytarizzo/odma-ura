#!/bin/bash
#PBS -l walltime=36:00:00
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -N odma008_dense_B12_n2048
#PBS -o jobs/008_B12_n2048_scaledK/008_B12_n2048_scaledK_dense.o
#PBS -e jobs/008_B12_n2048_scaledK/008_B12_n2048_scaledK_dense.e

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

module load miniforge/3
cd "${PBS_O_WORKDIR:-$HOME/odma-ura}"

JOB_DIR="jobs/008_B12_n2048_scaledK"
OUT_DIR="$JOB_DIR/results_dense_only"
mkdir -p "$OUT_DIR"

uv run python -m tests.arrangement_sweep_threshold_test \
  -B 12 \
  --n 2048 \
  --arrangements dense:2048:1 \
  --num-antennas 2 \
  --decoder NNOMP-OracleK \
  --K-values 3 8 17 25 33 50 67 83 100 133 167 208 250 333 \
  --target 0.05 \
  --ebn0-min -4 \
  --ebn0-max 4 \
  --ebn0-tol 0.1 \
  --max-search-steps 16 \
  --num-seeds 50 \
  --seed-start 42 \
  --out-name 008_B12_n2048_scaledK_dense_only \
  --out-dir "$OUT_DIR"
