#!/bin/bash
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -N fw_odma011_d512_b2
#PBS -o jobs/011_framework_odma_d512_b2/011_framework_odma_d512_b2.o
#PBS -e jobs/011_framework_odma_d512_b2/011_framework_odma_d512_b2.e

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

module load miniforge/3
cd "${PBS_O_WORKDIR:-$HOME/odma-ura}"

JOB_DIR="jobs/011_framework_odma_d512_b2"
OUT_DIR="$JOB_DIR/results"
mkdir -p "$OUT_DIR"

uv run --extra framework python -m tests.framework_odma_test \
  --preset odma \
  -B 10 \
  --n 1024 \
  --d 512 \
  --num-blocks 2 \
  --num-antennas 2 \
  --K-values 2 5 10 15 20 30 40 50 60 80 100 125 150 200 \
  --ebn0-grid 0 0.5 1 1.5 2 2.5 3 3.5 4 \
  --num-seeds 50 \
  --seed-start 42 \
  --out-dir "$OUT_DIR"
