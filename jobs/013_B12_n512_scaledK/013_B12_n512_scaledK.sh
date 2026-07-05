#!/bin/bash
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -N odma013_B12_n512_scaledK
#PBS -o jobs/013_B12_n512_scaledK/013_B12_n512_scaledK.o
#PBS -e jobs/013_B12_n512_scaledK/013_B12_n512_scaledK.e

# Smaller-n stress test matched to the job-008 information-load grid.
# Keeps B=12 (M=4096) to reduce repeated-message collisions relative to B=10,
# while shrinking the ambient resource length to n=512. ODMA d/n ratios match
# job 008: 1/8 with 16 blocks, 1/4 with 8 blocks, and 1/2 with 2 blocks.

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

module load miniforge/3
cd "${PBS_O_WORKDIR:-$HOME/odma-ura}"

JOB_DIR="jobs/013_B12_n512_scaledK"
OUT_DIR="$JOB_DIR/results"
mkdir -p "$OUT_DIR"

uv run python -m tests.arrangement_sweep_threshold_test \
  -B 12 \
  --n 512 \
  --arrangements dense:512:1 d64_b16:64:16 d128_b8:128:8 d256_b2:256:2 \
  --num-antennas 2 \
  --decoder NNOMP-OracleK \
  --K-values 2 4 6 8 12 17 21 25 33 42 52 62 83 \
  --target 0.05 \
  --ebn0-min -4 \
  --ebn0-max 12 \
  --ebn0-tol 0.1 \
  --max-search-steps 16 \
  --num-seeds 50 \
  --seed-start 42 \
  --ci-bootstrap 500 \
  --out-name 013_B12_n512_scaledK \
  --out-dir "$OUT_DIR"
