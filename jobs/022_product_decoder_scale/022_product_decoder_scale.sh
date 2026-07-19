#!/bin/bash
#PBS -l walltime=08:00:00
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -J 1-60
#PBS -N ura022_product_scale
#PBS -o jobs/022_product_decoder_scale/022_product_decoder_scale.o
#PBS -e jobs/022_product_decoder_scale/022_product_decoder_scale.e

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export MPLCONFIGDIR="${TMPDIR:-/tmp}/odma-mpl-${PBS_JOBID:-local}"

module load miniforge/3
cd "${PBS_O_WORKDIR:-$HOME/odma-ura}"
bash jobs/run_product_grid_row.sh jobs/022_product_decoder_scale/manifest.tsv jobs/022_product_decoder_scale/results
