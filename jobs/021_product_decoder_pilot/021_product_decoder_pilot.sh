#!/bin/bash
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -J 1-20
#PBS -N ura021_product_pilot
#PBS -o jobs/021_product_decoder_pilot/021_product_decoder_pilot.o
#PBS -e jobs/021_product_decoder_pilot/021_product_decoder_pilot.e

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export MPLCONFIGDIR="${TMPDIR:-/tmp}/odma-mpl-${PBS_JOBID:-local}"

module load miniforge/3
cd "${PBS_O_WORKDIR:-$HOME/odma-ura}"
bash jobs/run_product_grid_row.sh jobs/021_product_decoder_pilot/manifest.tsv jobs/021_product_decoder_pilot/results
