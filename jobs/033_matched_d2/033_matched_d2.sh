#!/bin/bash
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=8:mem=64gb
#PBS -J 1-66
#PBS -N ura033_d2

set -euo pipefail
cd "${PBS_O_WORKDIR:?Submit from the repository root}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export UV_NO_SYNC=1
module load miniforge/3
mkdir -p jobs/033_matched_d2/logs
exec > >(tee "jobs/033_matched_d2/logs/${PBS_JOBID:-local}_${PBS_ARRAY_INDEX:-1}.log") 2>&1
uv run --no-sync python jobs/033_matched_d2/run_row.py
