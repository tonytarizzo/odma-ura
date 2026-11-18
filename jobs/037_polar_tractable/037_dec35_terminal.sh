#!/bin/bash
#PBS -l walltime=08:00:00
#PBS -l select=1:ncpus=2:mem=16gb
#PBS -N ura037_terminal
#PBS -j oe

set -euo pipefail
cd "${PBS_O_WORKDIR:?Submit from the repository root}"
module load miniforge/3
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2
job=jobs/037_polar_tractable
out="$job/results_terminal"
mkdir "$out"
exec > "$out/job.log" 2>&1
printf 'PBS job %s\n' "${PBS_JOBID:?}"
git rev-parse HEAD
uv --version
uv run --no-sync python - <<'PY'
import platform, sys
import numpy, scipy, torch
if sys.version_info[:2] != (3, 12):
    raise SystemExit('Synchronize the checkout with uv sync --frozen --python 3.12 before submission')
print(platform.platform(), sys.version, numpy.__version__, scipy.__version__, torch.__version__)
numpy.show_config()
PY
uv run --no-sync python -m unittest tests.polar_folded_terminal_test tests.polar_terminal_query_test
uv run --no-sync python "$job/run_terminal_confirmation.py" --out "$out/main"
