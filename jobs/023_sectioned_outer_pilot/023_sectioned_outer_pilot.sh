#!/bin/bash
#PBS -l walltime=06:00:00
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -J 1-26
#PBS -N ura023_sectioned_pilot
#PBS -o jobs/023_sectioned_outer_pilot/023_sectioned_outer_pilot.o
#PBS -e jobs/023_sectioned_outer_pilot/023_sectioned_outer_pilot.e

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export MPLCONFIGDIR="${TMPDIR:-/tmp}/odma-mpl-${PBS_JOBID:-local}"

module load miniforge/3
cd "${PBS_O_WORKDIR:-$HOME/odma-ura}"
bash jobs/run_sectioned_grid_row.sh jobs/023_sectioned_outer_pilot/manifest.tsv jobs/023_sectioned_outer_pilot/results
