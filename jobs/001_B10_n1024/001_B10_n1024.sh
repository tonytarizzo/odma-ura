#!/bin/bash
#PBS -l walltime=18:00:00
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -N odma001_B10_n1024
#PBS -o jobs/001_B10_n1024/001_B10_n1024.o
#PBS -e jobs/001_B10_n1024/001_B10_n1024.e

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

module load miniforge/3
cd "${PBS_O_WORKDIR:-$HOME/odma-ura}"

JOB_DIR="jobs/001_B10_n1024"
OUT_DIR="$JOB_DIR/results"
mkdir -p "$OUT_DIR"

uv run python -m tests.arrangement_sweep_threshold_test \
  -B 10 \
  --n 1024 \
  --arrangements d128_b16:128:16 d256_b8:256:8 d512_b2:512:2 dense:1024:1 \
  --num-antennas 2 \
  --decoder NNOMP-OracleK \
  --K-values 2 5 10 15 20 30 40 50 60 80 100 125 150 200 \
  --target 0.05 \
  --ebn0-min -4 \
  --ebn0-max 4 \
  --ebn0-tol 0.1 \
  --max-search-steps 16 \
  --num-seeds 50 \
  --seed-start 42 \
  --out-name 001_B10_n1024 \
  --out-dir "$OUT_DIR"
