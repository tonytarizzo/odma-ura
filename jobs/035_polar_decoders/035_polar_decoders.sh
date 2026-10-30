#!/bin/bash
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=2:mem=16gb
#PBS -J 1-90
#PBS -N ura035_decoders

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
mkdir -p jobs/035_polar_decoders/logs_all_decoders
exec > >(tee "jobs/035_polar_decoders/logs_all_decoders/${PBS_JOBID:-local}_${PBS_ARRAY_INDEX:-1}.log") 2>&1
uv run --no-sync python jobs/035_polar_decoders/run_row.py \
  --manifest "${POLAR_DECODER_MANIFEST:-jobs/035_polar_decoders/manifest.jsonl}" \
  --mapping "${POLAR_DECODER_MAPPING:-all}" \
  --out-root "${POLAR_DECODER_OUT_ROOT:-jobs/035_polar_decoders/results_all_decoders}"
