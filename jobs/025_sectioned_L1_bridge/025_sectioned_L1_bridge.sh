#!/bin/bash
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -J 1-6
#PBS -N ura025_sectioned_L1
#PBS -o jobs/025_sectioned_L1_bridge/025_sectioned_L1_bridge.o
#PBS -e jobs/025_sectioned_L1_bridge/025_sectioned_L1_bridge.e

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export UV_NO_SYNC=1

module load miniforge/3
cd "${PBS_O_WORKDIR:-$HOME/odma-ura}"
bash jobs/run_sectioned_bridge_l1_row.sh jobs/025_sectioned_L1_bridge/manifest.tsv jobs/025_sectioned_L1_bridge/results
