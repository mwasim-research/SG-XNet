"""Implementation check against a textbook GCN (Kipf & Welling, 2017), written as plain PyG code.

Both models are trained by the pipeline's rules on the same Cora splits:
- k = 20 labelled nodes per class, 30 validation nodes per class, the rest for testing;
- Adam with weight decay 5e-4, dropout 0.5, 64 hidden units;
- learning rate chosen from {0.01, 0.005, 0.001} on validation;
- early stopping on validation accuracy, patience 100.

The splits are development splits (seeds 2000-2004), not the evaluation splits.

Run from the project folder:  python -m tests.check_reference
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from sgx import data, models, splits, train

torch.set_num_threads(1)
g = data.load("cora")
SEEDS = [2000, 2001, 2002, 2003, 2004]


class Textbook(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c1, self.c2 = GCNConv(g.in_dim, 64), GCNConv(64, g.num_classes)

    def forward(self, _g):
        x = F.dropout(g.x, 0.5, self.training)
        x = F.dropout(F.relu(self.c1(x, g.ei)), 0.5, self.training)
        return self.c2(x, g.ei)


def run(make):
    vals, tests = [], []
    for s in SEEDS:
        masks = splits.budget_split(g.y_np, 20, seed=s, multilabel=False)
        runs = []
        for lr in (0.01, 0.005, 0.001):
            torch.manual_seed(1000 + s)
            best, _ = train.fit(make(), g, masks, lr, 5e-4, max_epochs=1000, patience=100)
            runs.append(best)
        b = max(runs, key=lambda r: (r["val"], -r["val_loss"]))
        vals.append(b["val"])
        tests.append(b["test"])
    return 100 * np.mean(vals), 100 * np.mean(tests)


cfg = dict(K=2, hidden=64, dropout=0.5, wd=5e-4)
for name, make in [("textbook PyG GCN", Textbook),
                   ("pipeline GCN (layout v2b)", lambda: models.build("gcn", cfg, g.in_dim, g.num_classes))]:
    v, t = run(make)
    print(f"{name:28s} validation {v:5.1f}   test {t:5.1f}")
