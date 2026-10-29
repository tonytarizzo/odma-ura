#!/bin/bash
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=2:mem=16gb
#PBS -J 1-60%4
#PBS -N ura037_tractable

set -euo pipefail
cd "${PBS_O_WORKDIR:?Submit from the repository root}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export VECLIB_MAXIMUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export UV_NO_SYNC=1
module load miniforge/3
mkdir -p jobs/037_polar_tractable/logs_tractable
exec > >(tee "jobs/037_polar_tractable/logs_tractable/${PBS_JOBID:-local}_${PBS_ARRAY_INDEX:-1}.log") 2>&1
uv run --no-sync python jobs/037_polar_tractable/run_row.py --manifest "${POLAR_TRACTABLE_MANIFEST:-jobs/037_polar_tractable/manifest.jsonl}"
