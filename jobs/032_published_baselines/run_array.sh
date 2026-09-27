#!/bin/bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export UV_NO_SYNC=1
module load miniforge/3
mkdir -p jobs/032_published_baselines/logs
exec > >(tee "jobs/032_published_baselines/logs/${URA_PHASE}_${PBS_JOBID:-local}_${PBS_ARRAY_INDEX:-1}.log") 2>&1
uv run --no-sync python jobs/032_published_baselines/run_row.py
