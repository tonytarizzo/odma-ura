#!/bin/bash
#PBS -N ura039_analysis
#PBS -l walltime=02:00:00
#PBS -l select=1:ncpus=2:mem=16gb
#PBS -j oe
set -euo pipefail
cd "${PBS_O_WORKDIR:?Submit from the repository root}"
module load miniforge/3
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
study=jobs/039_polar_paper
mkdir -p "$study/results/logs"
exec > "$study/results/logs/${PBS_JOBID:?}.merge.log" 2>&1
uv run --no-sync python "$study/run_study.py" --merge
