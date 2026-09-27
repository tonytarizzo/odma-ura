#!/bin/bash
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=8:mem=64gb
#PBS -J 1-36
#PBS -N ura032_native

set -euo pipefail
export URA_PHASE=native
cd "${PBS_O_WORKDIR:?Submit from the repository root}"
exec bash jobs/032_published_baselines/run_array.sh
