#!/bin/bash
#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=8:mem=64gb
#PBS -J 1-72
#PBS -N ura031_coordinate

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export MPLCONFIGDIR="${TMPDIR:-/tmp}/odma-mpl-${PBS_JOBID:-local}-${PBS_ARRAY_INDEX:-1}"
export UV_NO_SYNC=1

module load miniforge/3
cd "${PBS_O_WORKDIR:-/rds/general/user/at5424/home/odma-ura}"
mkdir -p jobs/031_coordinate_amplitude_frontier_B14/logs
exec > >(tee "jobs/031_coordinate_amplitude_frontier_B14/logs/${PBS_JOBID:-local}_array_${PBS_ARRAY_INDEX:-1}.log") 2>&1
uv run --no-sync python jobs/031_coordinate_amplitude_frontier_B14/run_row.py
