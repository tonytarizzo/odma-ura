#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
uv run python -m tests.published_baselines_learning --preset mini --out-dir "${1:-jobs/032_published_baselines/results_mini}"
