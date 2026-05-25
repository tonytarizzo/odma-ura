#!/bin/bash
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -N odma007_B6_n256
#PBS -o jobs/007_B6_n256/007_B6_n256.o
#PBS -e jobs/007_B6_n256/007_B6_n256.e

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

module load miniforge/3
cd "${PBS_O_WORKDIR:-$HOME/odma-ura}"

JOB_DIR="jobs/007_B6_n256"
OUT_DIR="$JOB_DIR/results"
mkdir -p "$OUT_DIR"

uv run python -m tests.arrangement_sweep_threshold_test \
  -B 6 \
  --n 256 \
  --arrangements d32_b16:32:16 d64_b8:64:8 d128_b2:128:2 dense:256:1 \
  --num-antennas 2 \
  --decoder NNOMP-OracleK \
  --K-values 2 4 6 8 12 17 21 25 33 42 52 62 83 \
  --target 0.05 \
  --ebn0-min -4 \
  --ebn0-max 4 \
  --ebn0-tol 0.1 \
  --max-search-steps 16 \
  --num-seeds 50 \
  --seed-start 42 \
  --out-name 007_B6_n256 \
  --out-dir "$OUT_DIR"
