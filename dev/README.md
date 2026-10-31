# Development experiments (not part of the evaluation)

`devexp.py` tests candidate changes that were designed on 22 Oct 2026, after the pilot's tuning results:

- **E1 — fine-tuning protocols for SG-XNet:** freeze only the first (GraphSAGE) layer (`frz1`); linear probe then fine-tune (`lpft`); weight-space interpolation between pretrained and fine-tuned weights (`wise`). They are compared with scratch, plain fine-tuning and the frozen body.
- **E2 — gated SG-XNet (`sgxg`):** every layer is a learned per-channel mix of a GCN step and a GraphSAGE-mean step. It is compared with SG-XNet, GCN and GraphSAGE.

## Rules

- Only dev splits are used: seeds 2000–2003 for the target splits and for pretraining. The evaluation splits 0–9 are never touched.
- Decisions use validation scores only. `python dev/devexp.py summary results_dev` prints nothing else. Test scores are kept in the run files but stay sealed until the design is frozen.
- The decision rules were written down before any results: see the project note `SG-XNet_improvement_ideas_2026-10-22`. A candidate that passes is added to the main study with the same tuning budget as every model; protocol changes apply to all models; the original SG-XNet stays in every table.
- Results go to `results_dev/`, which the study's analysis never reads.

## Commands

```
python dev/devexp.py run results_dev 2       # 2 worker processes (fits beside a running study)
python dev/devexp.py summary results_dev     # validation-only summary
```
