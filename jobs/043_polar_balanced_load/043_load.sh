#!/bin/bash
#PBS -N ura043_load
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=1:mem=12gb
#PBS -j oe
set -euo pipefail
cd "${PBS_O_WORKDIR:?Submit from the repository root}"
module load miniforge/3
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
job=jobs/043_polar_balanced_load/load_curve.py
mkdir -p results/dec43_balanced_load/logs
exec > "results/dec43_balanced_load/logs/${PBS_JOBID:?}.${PBS_ARRAY_INDEX:-merge}.log" 2>&1
case "${LOAD_PHASE:?Submit through submit.sh}" in
  pilot)   uv run --no-sync python "$job" --pilot "${LOAD_ROUND:?}" --shard "${PBS_ARRAY_INDEX:?}" ;;
  confirm) uv run --no-sync python "$job" --confirm --shard "${PBS_ARRAY_INDEX:?}" ;;
  merge)
    uv run --no-sync python "$job" --merge
    uv run --no-sync python jobs/043_polar_balanced_load/grid_precision.py
    ;;
  *)       exit 2 ;;
esac
