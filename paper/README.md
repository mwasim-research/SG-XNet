# Scripts behind the numbers in the revised manuscript

Run from the repository root after `bash scripts/unpack_results.sh`. All scripts read only the released result files
(paths are relative to the repository root); `rev2.py` additionally reads the processed dataset statistics in
`data/cache/` (created by the pipeline when the datasets are present).

| Script | Produces |
|---|---|
| `ms_tables.py` | Table 4 (aligned pairs: scratch, fine-tuned, frozen, zero-shot, FT−ZS), Table 9 (self-supervised and domain-adaptation baselines), the SG-XNet-in-UDAGCN comparison; writes `paper/output/numbers.json` |
| `ms_supp.py` | Supplementary Tables S2–S6 (selected configurations, per-target results, self-supervised baselines on all targets, transfer-aware tuning, source-validation scores); writes `paper/output/supp_tables.tex` |
| `ms_figs.py` | Figure 4 (when transfer pays off), Figure 5 (average ranks), Figure 7 (CKA vs. loss), Figure 8 (L2-SP), as PDF/EPS/SVG/PNG plus CSV |
| `heat3.py` | Figure 3 (fine-tuning gain heatmap; failed pretraining blank; stars = Holm within each model) |
| `ov_an.py` | Post hoc analyses: L2-SP (Table 8), SG-XNet in UDAGCN, CKA, transfer-aware tuning |
| `tw_an.py` | Twitch results (Table 10; Supplementary Table S10) |
| `rev2.py` | Additional checks of the first revision round (sensitivity to the failed run, matched pairs, model size, Twitch per country; Supplementary Tables S8–S10) |
| `n2_an.py`, `e1b.py` | Source-only check (Supplementary Table S11; `e1b.py` gives the main-study comparison columns with the main study's exclusion) |
| `e2_an.py` | CKA under L2-SP (Supplementary Table S12) |
| `rv3.py` | Fine-tuned minus frozen (Supplementary Table S13), selected epochs and learning rates, pre-stated test on split subsets and with Holm over 15 tests (Table 5 pooled column, Supplementary Table S14) |
| `rv3b.py` | Within-model CKA analysis |
| `../posthoc/cloud_2026-10-27/a_tables.py` | Friedman tests behind Figure 5 (`ms_figs.py` computes the ranks and the critical difference) |

Figures 1–2 (model and protocol diagrams) and Figure 6 (budget curves) are made by the pipeline:
`bash run.sh figures-full` (written to `results/figures/`).
