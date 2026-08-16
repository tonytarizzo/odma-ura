#!/bin/bash
#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -J 1-6
#PBS -N ura026_sectioned_Lgt1
#PBS -o jobs/026_sectioned_Lgt1_bridge/026_sectioned_Lgt1_bridge.o
#PBS -e jobs/026_sectioned_Lgt1_bridge/026_sectioned_Lgt1_bridge.e

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export UV_NO_SYNC=1

module load miniforge/3
cd "${PBS_O_WORKDIR:-$HOME/odma-ura}"
bash jobs/run_sectioned_bridge_lgt_row.sh jobs/026_sectioned_Lgt1_bridge/manifest.tsv jobs/026_sectioned_Lgt1_bridge/results
