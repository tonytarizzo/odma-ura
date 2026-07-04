#!/bin/bash
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -N odma008_B12_n2048_scaledK
#PBS -o jobs/008_B12_n2048_scaledK/008_B12_n2048_scaledK.o
#PBS -e jobs/008_B12_n2048_scaledK/008_B12_n2048_scaledK.e

# Full four-curve gap experiment (dense + three ODMA arrangements) at the matched
# B=12, n=2048, scaled-K scale. This supersedes the earlier dense-only patch run.
#
# The sweep now checkpoints after every (arrangement, K) point and resumes on
# restart, so if this job hits the 72h walltime just `qsub` it again and it
# continues from the last completed point (no data is lost, unlike the original
# 008 run which was recovered from stdout). Dense is listed first because it is
# the curve missing from the recovered partial results and the most expensive.

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

module load miniforge/3
cd "${PBS_O_WORKDIR:-$HOME/odma-ura}"

JOB_DIR="jobs/008_B12_n2048_scaledK"
OUT_DIR="$JOB_DIR/results_full"
mkdir -p "$OUT_DIR"

uv run python -m tests.arrangement_sweep_threshold_test \
  -B 12 \
  --n 2048 \
  --arrangements dense:2048:1 d256_b16:256:16 d512_b8:512:8 d1024_b2:1024:2 \
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
  --ci-bootstrap 500 \
  --out-name 008_B12_n2048_scaledK_full \
  --out-dir "$OUT_DIR"
