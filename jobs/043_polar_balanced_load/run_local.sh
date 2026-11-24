#!/bin/bash
# Run the released six-load experiment locally, using the same resumable cells as PBS.
set -euo pipefail
workers=${1:-4}
[[ "$workers" =~ ^[1-9][0-9]*$ ]] || { echo 'Supply a positive local worker count.' >&2; exit 2; }
driver=jobs/043_polar_balanced_load/load_curve.py
test -f "$driver"
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for round in 0 1 2 3; do
    seq 1 48 | xargs -n 1 -P "$workers" uv run --no-sync python "$driver" --pilot "$round" --shard
done
seq 1 768 | xargs -n 1 -P "$workers" uv run --no-sync python "$driver" --confirm --shard
uv run --no-sync python "$driver" --merge
uv run --no-sync python jobs/043_polar_balanced_load/grid_precision.py
