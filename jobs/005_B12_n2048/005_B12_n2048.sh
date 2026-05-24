#!/bin/bash
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -N odma005_B12_n2048
#PBS -o jobs/005_B12_n2048/005_B12_n2048.o
#PBS -e jobs/005_B12_n2048/005_B12_n2048.e

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

module load miniforge/3
cd "${PBS_O_WORKDIR:-$HOME/odma-ura}"

JOB_DIR="jobs/005_B12_n2048"
OUT_DIR="$JOB_DIR/results"
mkdir -p "$OUT_DIR"

uv run python -m tests.arrangement_sweep_threshold_test \
  -B 12 \
  --n 2048 \
  --arrangements d256_b16:256:16 d512_b8:512:8 d1024_b2:1024:2 dense:2048:1 \
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
  --out-name 005_B12_n2048 \
  --out-dir "$OUT_DIR"
