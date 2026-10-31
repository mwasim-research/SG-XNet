#!/usr/bin/env bash
# SG-XNet v2 experiments (PeerJ CS-2024:10:108662, round 3).
#   bash run.sh smoke          1-2 minute end-to-end test (separate folder results_smoke/)
#   bash run.sh pilot          pilot study: tuning + transfer (5 splits, budgets 1/5/20) + public splits
#   bash run.sh full           full study (10 splits, budgets 1/3/5/10/20, negative controls, ablation)
#   bash run.sh baselines      pretraining baselines: DGI, GRACE (self-supervised), DANN, UDAGCN (domain adaptation)
#   bash run.sh all            pilot, then full study, then baselines (each re-uses what the previous one produced)
#   bash run.sh explore        EXPLORATORY (after the main study): gated SG-XNet-G, own tuning, same splits/conditions
#   bash run.sh explore-smoke  1-2 minute test of the exploratory plan (in results_smoke/, after "smoke")
#   bash run.sh status         progress of the running stage
#   bash run.sh analyze-pilot  rebuild results/analysis/SUMMARY_pilot.md (also analyze-full, -baselines, -explore)
#   bash run.sh figures-pilot  vector figures in results/figures/ (also figures-full, figures-explore)
#   bash run.sh size-full      time estimate for the full study from the job times measured so far (also size-explore)
# Safe to stop (Ctrl-C or close the window) and start again: finished jobs are skipped.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore

find_python() {
  for c in ".venv/bin/python" "../../GraphLearner/GraphLearner-Experiments/.venv/bin/python"; do
    if [ -x "$c" ] && "$c" -c "import torch, torch_geometric, scipy, pandas" >/dev/null 2>&1; then echo "$c"; return; fi
  done
}
PY="${PYTHON:-$(find_python || true)}"
if [ -z "$PY" ]; then
  echo "First run: creating a Python environment in .venv (a few minutes)..."
  BASE="$(command -v python3.11 || command -v python3.12 || command -v python3.10 || command -v python3)"
  "$BASE" -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
  PY=".venv/bin/python"
fi
WORKERS="${WORKERS:-$("$PY" -c 'import os; print(max(1, (os.cpu_count() or 2) - 2))')}"
echo "Python: $PY | $("$PY" -c 'import torch, torch_geometric as g; print("torch", torch.__version__, "| PyG", g.__version__)') | workers: $WORKERS"
AWAKE=""
if command -v caffeinate >/dev/null 2>&1; then AWAKE="caffeinate -i"; fi   # keeps the Mac awake while running

case "${1:-status}" in
  smoke)  SGX_RESULTS=results_smoke SGX_QUICK=1 $AWAKE "$PY" -m sgx.cli run --plan smoke --workers "$WORKERS" ;;
  pilot)  $AWAKE "$PY" -m sgx.cli run --plan pilot --workers "$WORKERS" ;;
  full)   $AWAKE "$PY" -m sgx.cli run --plan full --workers "$WORKERS" ;;
  status) "$PY" -m sgx.cli status ;;
  baselines) $AWAKE "$PY" -m sgx.cli run --plan baselines --workers "$WORKERS" ;;
  all) $AWAKE "$PY" -m sgx.cli run --plan pilot --workers "$WORKERS" && \
       $AWAKE "$PY" -m sgx.cli run --plan full --workers "$WORKERS" && \
       $AWAKE "$PY" -m sgx.cli run --plan baselines --workers "$WORKERS" ;;
  explore) $AWAKE "$PY" -m sgx.cli run --plan explore --workers "$WORKERS" ;;
  explore-smoke) SGX_RESULTS=results_smoke SGX_QUICK=1 $AWAKE "$PY" -m sgx.cli run --plan explore_smoke --workers "$WORKERS" ;;
  analyze-pilot) "$PY" -m sgx.cli analyze --plan pilot ;;
  analyze-full)  "$PY" -m sgx.cli analyze --plan full ;;
  analyze-baselines) "$PY" -m sgx.cli analyze --plan baselines ;;
  analyze-explore)   "$PY" -m sgx.cli analyze --plan explore ;;
  figures-pilot) "$PY" -m sgx.cli figures --plan pilot ;;
  figures-full)  "$PY" -m sgx.cli figures --plan full ;;
  figures-explore) "$PY" -m sgx.cli figures --plan explore ;;
  size-full)      "$PY" -m sgx.cli size --plan full --workers "$WORKERS" ;;
  size-baselines) "$PY" -m sgx.cli size --plan baselines --workers "$WORKERS" ;;
  size-explore)   "$PY" -m sgx.cli size --plan explore --workers "$WORKERS" ;;
  *) echo "usage: bash run.sh smoke|pilot|full|baselines|all|explore|explore-smoke|status|analyze-{pilot,full,baselines,explore}|figures-{pilot,full,explore}|size-{full,baselines,explore}"
     exit 1 ;;
esac
