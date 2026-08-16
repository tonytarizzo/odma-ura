#!/bin/bash
#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=8:mem=64gb
#PBS -J 1-8
#PBS -N ura024_sectioned_B128
#PBS -o jobs/024_sectioned_B128_scale/024_sectioned_B128_scale.o
#PBS -e jobs/024_sectioned_B128_scale/024_sectioned_B128_scale.e

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export MPLCONFIGDIR="${TMPDIR:-/tmp}/odma-mpl-${PBS_JOBID:-local}"

module load miniforge/3
cd "${PBS_O_WORKDIR:-$HOME/odma-ura}"
bash jobs/run_sectioned_grid_row.sh jobs/024_sectioned_B128_scale/manifest.tsv jobs/024_sectioned_B128_scale/results
