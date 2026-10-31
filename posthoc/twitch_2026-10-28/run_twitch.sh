#!/usr/bin/env bash
# Post hoc Twitch run (28 Oct 2026). Writes results_posthoc_twitch/ only.
# Safe to stop and start again: finished jobs are skipped.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore
PY=""
for c in ".venv/bin/python" "../../GraphLearner/GraphLearner-Experiments/.venv/bin/python"; do
  if [ -x "$c" ] && "$c" -c "import torch, torch_geometric, scipy, pandas" >/dev/null 2>&1; then PY="$c"; break; fi
done
[ -n "$PY" ] || { echo "No Python environment found (run 'bash run.sh status' once first)"; exit 1; }
AWAKE=""; command -v caffeinate >/dev/null 2>&1 && AWAKE="caffeinate -i"
echo "Python: $PY"
$AWAKE "$PY" posthoc_tests/twitch_2026-10-28/twitch.py
