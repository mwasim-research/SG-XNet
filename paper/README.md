# Scripts behind the numbers in the revised manuscript

Run from the repository root after `bash scripts/unpack_results.sh`. All scripts read only the released result files
(paths are relative to the repository root); `rev2.py` additionally reads the processed dataset statistics in
`data/cache/` (created by the pipeline when the datasets are present).

| Script | Produces |
|---|---|
| `ms_tables.py` | Table 4 (aligned pairs: scratch, fine-tuned, frozen, zero-shot, FT−ZS), Table 8 (self-supervised and domain-adaptation baselines), the SG-XNet-in-UDAGCN comparison; writes `paper/output/numbers.json` |
| `ms_supp.py` | Supplementary Tables S2–S6 (selected configurations, per-target results, self-supervised baselines on all targets, transfer-aware tuning, source-validation scores); writes `paper/output/supp_tables.tex` |
| `ms_figs.py` | Figure 4 (when transfer pays off), Figure 5 (CKA vs. loss), Figure 6 (L2-SP), Supplementary Figure S1 (average ranks), as PDF/EPS/SVG/PNG plus CSV |
| `heat3.py` | Figure 3 (fine-tuning gain heatmap; failed pretraining blank; stars = Holm within each model) |
| `ov_an.py` | Post hoc analyses: L2-SP (Table 7), SG-XNet in UDAGCN, CKA, transfer-aware tuning |
| `tw_an.py` | Twitch results (Supplementary Tables S14 and S15) |
| `rev2.py` | Additional checks of the first revision round (sensitivity to the failed run, matched pairs, model size, Twitch per country; Supplementary Tables S8, S9 and S15) |
| `n2_an.py`, `e1b.py` | Source-only check (Supplementary Table S10; `e1b.py` gives the main-study comparison columns with the main study's exclusion) |
| `e2_an.py` | CKA under L2-SP (Supplementary Table S11) |
| `rv3.py` | Fine-tuned minus frozen (Supplementary Table S12), selected epochs and learning rates (Supplementary S1), pre-stated test on split subsets and with Holm over 15 tests (Table 5 pooled column, Supplementary Table S13) |
| `rv3b.py` | Within-model CKA analysis (Supplementary S9) |
| `../posthoc/cloud_2026-10-27/a_tables.py` | Friedman tests behind Supplementary Figure S1 (`ms_figs.py` computes the ranks and the critical difference) |

Figures 1–2 (model and protocol diagrams) and Supplementary Figure S2 (SG-XNet budget curves) are made by the pipeline:
`bash run.sh figures-full` (written to `results/figures/`).
