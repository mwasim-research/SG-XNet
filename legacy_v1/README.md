# Version 1 (superseded)

These are the original scripts of SG-XNet version 1 (CiteSeer→Cora fine-tuning with GCN, GraphSAGE, GAT, GIN and
SG-XNet), kept unchanged for the record; `README_v1.md` is the original README.

**Do not use these results.** Version 1 trained each model once on one split with fixed hyperparameters, computed
metrics on all nodes averaged over 200 epochs, had no validation-based model selection, used different learning rates
for SG-XNet and the baselines, and its "feature extraction" runs did not freeze any layer. The revised paper withdraws
all version 1 numbers. Use the version 2 code and results at the top level of this repository.
