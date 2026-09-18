#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
uv run python -m tests.framework_coordinate_amplitude_learning --mini "$@"
