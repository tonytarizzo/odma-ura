#!/bin/bash
#PBS -l walltime=08:00:00
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -J 1-72
#PBS -N ura027_sparse_density

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export MPLCONFIGDIR="${TMPDIR:-/tmp}/odma-mpl-${PBS_JOBID:-local}-${PBS_ARRAY_INDEX:-1}"

cd "${PBS_O_WORKDIR:-/rds/general/user/at5424/home/odma-ura}"
mkdir -p jobs/027_sparse_density_frontier/logs
exec > >(tee "jobs/027_sparse_density_frontier/logs/${PBS_JOBID:-local}_array_${PBS_ARRAY_INDEX:-1}.log") 2>&1
bash jobs/run_sparsity_grid_row.sh \
  jobs/027_sparse_density_frontier/manifest.tsv \
  jobs/027_sparse_density_frontier/results
