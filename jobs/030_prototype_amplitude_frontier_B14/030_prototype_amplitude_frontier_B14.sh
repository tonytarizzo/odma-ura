#!/bin/bash
#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=8:mem=64gb
#PBS -J 1-48
#PBS -N ura030_amp_frontier

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
mkdir -p jobs/030_prototype_amplitude_frontier_B14/logs
exec > >(tee "jobs/030_prototype_amplitude_frontier_B14/logs/${PBS_JOBID:-local}_array_${PBS_ARRAY_INDEX:-1}.log") 2>&1
bash jobs/run_prototype_amplitude_row.sh jobs/030_prototype_amplitude_frontier_B14/manifest.tsv \
  jobs/030_prototype_amplitude_frontier_B14/results
