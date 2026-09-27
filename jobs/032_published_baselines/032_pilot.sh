#!/bin/bash
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -J 1-80
#PBS -N ura032_pilot

set -euo pipefail
export URA_PHASE=pilot
cd "${PBS_O_WORKDIR:?Submit from the repository root}"
exec bash jobs/032_published_baselines/run_array.sh
