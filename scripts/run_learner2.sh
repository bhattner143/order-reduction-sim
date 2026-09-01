#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/opt/anaconda3/envs/pydrake_sfil/bin/python}"
export KMP_DUPLICATE_LIB_OK=TRUE
export MPLBACKEND="${MPLBACKEND:-Agg}"
cd "$ROOT"
exec "$PYTHON" scripts/run_learner2.py "$@"
