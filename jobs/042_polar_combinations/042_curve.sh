#!/bin/bash
#PBS -N ura042_curve
#PBS -J 1-428
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=2:mem=12gb
#PBS -j oe
set -euo pipefail
cd "${PBS_O_WORKDIR:?Submit from the repository root}"
module load miniforge/3
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p results/dec42_paper_curve/logs
exec > "results/dec42_paper_curve/logs/${PBS_JOBID:?}.${PBS_ARRAY_INDEX:?}.log" 2>&1
uv run --no-sync python jobs/042_polar_combinations/paper_curve_extension.py --shard "$PBS_ARRAY_INDEX"
