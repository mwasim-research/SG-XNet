SG-XNet v2 - post hoc tests run on a Linux cloud machine (not on the Mac), 25-26 Oct 2026.
Same code as the Mac (v2c, sgx/ md5 as deployed), same rules (learning rate from {0.01, 0.005, 0.001} on validation,
pretraining learning rate on source validation, 10 splits, test at best-validation epoch).

1. ablzs/  - zero-shot scores of the six ablation variants on ACMv9 -> DBLPv7 (k = 5, 20; splits 0-9), computed from the
             Mac's pretrained checkpoints. SG-XNet's zero-shot reproduced the Mac's values exactly (max difference 0.0000).
2. swap/   - settings-swap test: SG-XNet with GCN's selected settings, and all-GCN with SG-XNet's selected settings
             (pretraining on ACMv9, 10 seeds; zero-shot and fine-tuning on DBLPv7 at k = 5, 20 and Citationv1 at k = 1, 5).
             Check cell (SG-XNet's own settings, seed 0, DBLPv7 k5): cloud 69.86 / 72.06 vs Mac 70.01 / 72.21 (zero-shot / fine-tuned),
             i.e. about 0.15 points from platform differences in training.
These are post hoc analyses, prompted by the main ablation; label them as such in the paper.
