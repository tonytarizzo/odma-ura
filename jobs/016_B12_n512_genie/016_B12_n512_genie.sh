#!/bin/bash
#PBS -l walltime=8:00:00
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -N odma016_B12_n512_genie
#PBS -o jobs/016_B12_n512_genie/016_B12_n512_genie.o
#PBS -e jobs/016_B12_n512_genie/016_B12_n512_genie.e

# Genie-support reference for job 013. This matches B=12, n=512,
# arrangements, K grid, seeds, and Eb/N0 bracket, but hands the true support to
# the decoder. Compare against job 013 to separate oracle-support geometry from
# practical NNOMP support-search loss in the smaller-n regime.

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

module load miniforge/3
cd "${PBS_O_WORKDIR:-$HOME/odma-ura}"

JOB_DIR="jobs/016_B12_n512_genie"
OUT_DIR="$JOB_DIR/results"
mkdir -p "$OUT_DIR"

uv run python -m tests.arrangement_sweep_threshold_test \
  -B 12 \
  --n 512 \
  --arrangements dense:512:1 d64_b16:64:16 d128_b8:128:8 d256_b2:256:2 \
  --num-antennas 2 \
  --decoder Genie-OracleSupport \
  --K-values 2 4 6 8 12 17 21 25 33 42 52 62 83 \
  --target 0.05 \
  --ebn0-min -4 \
  --ebn0-max 12 \
  --ebn0-tol 0.1 \
  --max-search-steps 16 \
  --num-seeds 50 \
  --seed-start 42 \
  --ci-bootstrap 500 \
  --out-name 016_B12_n512_genie \
  --out-dir "$OUT_DIR"
