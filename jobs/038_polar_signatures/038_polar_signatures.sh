#!/bin/bash
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=2:mem=16gb
#PBS -J 1-24
#PBS -N ura038_signatures

set -euo pipefail
cd "${PBS_O_WORKDIR:?Submit from the repository root}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export UV_NO_SYNC=1
module load miniforge/3
mkdir -p jobs/038_polar_signatures/logs
exec > >(tee "jobs/038_polar_signatures/logs/${PBS_JOBID:-local}_${PBS_ARRAY_INDEX:-1}.log") 2>&1
uv run --no-sync python jobs/038_polar_signatures/run_row.py \
  --baseline-results "${POLAR_REFERENCE_RESULTS:-jobs/035_polar_decoders/results_all_decoders}"
