#!/bin/bash
#PBS -l walltime=8:00:00
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -N odma017_B12_n256_genie
#PBS -o jobs/017_B12_n256_genie/017_B12_n256_genie.o
#PBS -e jobs/017_B12_n256_genie/017_B12_n256_genie.e

# Genie-support reference for job 014. This matches B=12, n=256,
# arrangements, K grid, seeds, and Eb/N0 bracket, but hands the true support to
# the decoder. Compare against job 014 to separate oracle-support geometry from
# practical NNOMP support-search loss in the hardest smaller-n regime.

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

module load miniforge/3
cd "${PBS_O_WORKDIR:-$HOME/odma-ura}"

JOB_DIR="jobs/017_B12_n256_genie"
OUT_DIR="$JOB_DIR/results"
mkdir -p "$OUT_DIR"

uv run python -m tests.arrangement_sweep_threshold_test \
  -B 12 \
  --n 256 \
  --arrangements dense:256:1 d32_b16:32:16 d64_b8:64:8 d128_b2:128:2 \
  --num-antennas 2 \
  --decoder Genie-OracleSupport \
  --K-values 2 3 4 6 8 10 12 17 21 26 31 42 \
  --target 0.05 \
  --ebn0-min -4 \
  --ebn0-max 12 \
  --ebn0-tol 0.1 \
  --max-search-steps 16 \
  --num-seeds 50 \
  --seed-start 42 \
  --ci-bootstrap 500 \
  --out-name 017_B12_n256_genie \
  --out-dir "$OUT_DIR"
