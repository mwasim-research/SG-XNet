# Overnight post hoc run, 27–28 Oct 2026

Started from the VS Code task "SG-XNet 7: Overnight post hoc tests". It reads the main study's results in `results/` without changing them, and writes only to `results_posthoc_overnight/`.

What it runs is described at the top of `overnight.py`:
1. L2-SP fine-tuning for all 6 models.
2. Two cloud tests repeated at full size: SG-XNet inside UDAGCN, and representation drift (CKA).
3. Transfer-aware tuning for GCN and SG-XNet.

Progress is in `results_posthoc_overnight/status.json` and `results_posthoc_overnight/logs/overnight.log`.
