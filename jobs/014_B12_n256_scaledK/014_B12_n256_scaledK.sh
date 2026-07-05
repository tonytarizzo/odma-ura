#!/bin/bash
#PBS -l walltime=16:00:00
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -N odma014_B12_n256_scaledK
#PBS -o jobs/014_B12_n256_scaledK/014_B12_n256_scaledK.o
#PBS -e jobs/014_B12_n256_scaledK/014_B12_n256_scaledK.e

# Smaller-n stress test matched to the job-008 information-load grid.
# Keeps B=12 (M=4096) and shrinks the ambient resource length to n=256. This is
# deliberately hard: it tests whether dense-vs-ODMA differences widen when the
# explicit global recovery problem is more compressed but repeated-message
# collisions are still moderate.

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

module load miniforge/3
cd "${PBS_O_WORKDIR:-$HOME/odma-ura}"

JOB_DIR="jobs/014_B12_n256_scaledK"
OUT_DIR="$JOB_DIR/results"
mkdir -p "$OUT_DIR"

uv run python -m tests.arrangement_sweep_threshold_test \
  -B 12 \
  --n 256 \
  --arrangements dense:256:1 d32_b16:32:16 d64_b8:64:8 d128_b2:128:2 \
  --num-antennas 2 \
  --decoder NNOMP-OracleK \
  --K-values 2 3 4 6 8 10 12 17 21 26 31 42 \
  --target 0.05 \
  --ebn0-min -4 \
  --ebn0-max 12 \
  --ebn0-tol 0.1 \
  --max-search-steps 16 \
  --num-seeds 50 \
  --seed-start 42 \
  --ci-bootstrap 500 \
  --out-name 014_B12_n256_scaledK \
  --out-dir "$OUT_DIR"
