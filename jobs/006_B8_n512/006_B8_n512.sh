#!/bin/bash
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -N odma006_B8_n512
#PBS -o jobs/006_B8_n512/006_B8_n512.o
#PBS -e jobs/006_B8_n512/006_B8_n512.e

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

module load miniforge/3
cd "${PBS_O_WORKDIR:-$HOME/odma-ura}"

JOB_DIR="jobs/006_B8_n512"
OUT_DIR="$JOB_DIR/results"
mkdir -p "$OUT_DIR"

uv run python -m tests.arrangement_sweep_threshold_test \
  -B 8 \
  --n 512 \
  --arrangements d64_b16:64:16 d128_b8:128:8 d256_b2:256:2 dense:512:1 \
  --num-antennas 2 \
  --decoder NNOMP-OracleK \
  --K-values 2 3 6 9 12 19 25 31 38 50 62 78 94 125 \
  --target 0.05 \
  --ebn0-min -4 \
  --ebn0-max 4 \
  --ebn0-tol 0.1 \
  --max-search-steps 16 \
  --num-seeds 50 \
  --seed-start 42 \
  --out-name 006_B8_n512 \
  --out-dir "$OUT_DIR"
