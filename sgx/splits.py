"""Train/validation/test splits.

* budget_split: k labelled training nodes per class, 30 validation nodes per class, all remaining nodes are test
  (the protocol of Shchur et al., 2018, with a variable label budget k).
* ratio_split: 80/10/10 stratified split, used for the (well-labelled) source graphs.
* public_split: the public Planetoid split, or one of the 10 public splits of Platonov et al. (2023).
"""
from __future__ import annotations

import numpy as np
import torch

VAL_PER_CLASS = 30


def _class_members(y: np.ndarray, multilabel: bool):
    if multilabel:
        return [np.where(y[:, c] > 0)[0] for c in range(y.shape[1])]
    return [np.where(y == c)[0] for c in range(int(y.max()) + 1)]


def budget_split(y: np.ndarray, k: int, seed: int, multilabel: bool, val_per_class: int = VAL_PER_CLASS):
    rng = np.random.default_rng(seed)
    n = len(y)
    assigned = np.zeros(n, bool)
    members = _class_members(y, multilabel)
    order = rng.permutation(len(members))
    train, val = [], []
    for c in order:
        idx = members[c][~assigned[members[c]]]
        pick = rng.permutation(idx)[:k]
        assigned[pick] = True
        train.append(pick)
    for c in order:
        idx = members[c][~assigned[members[c]]]
        pick = rng.permutation(idx)[:val_per_class]
        assigned[pick] = True
        val.append(pick)
    return _masks(n, np.concatenate(train), np.concatenate(val), np.where(~assigned)[0])


def ratio_split(y: np.ndarray, seed: int, multilabel: bool, train: float = 0.8, val: float = 0.1):
    rng = np.random.default_rng(seed)
    n = len(y)
    if multilabel:  # stratify on the first label
        key = y.argmax(1)
    else:
        key = y
    tr, va, te = [], [], []
    for c in np.unique(key):
        idx = rng.permutation(np.where(key == c)[0])
        a, b = int(round(train * len(idx))), int(round((train + val) * len(idx)))
        tr.append(idx[:a]); va.append(idx[a:b]); te.append(idx[b:])
    return _masks(n, np.concatenate(tr), np.concatenate(va), np.concatenate(te))


def public_split(public: dict, i: int):
    j = i % public["train"].shape[0]
    return {s: torch.from_numpy(np.asarray(public[s][j], dtype=bool)) for s in ("train", "val", "test")}


def _masks(n, tr, va, te):
    out = {}
    for name, idx in (("train", tr), ("val", va), ("test", te)):
        m = torch.zeros(n, dtype=torch.bool)
        m[torch.from_numpy(np.asarray(idx, dtype=np.int64))] = True
        out[name] = m
    return out
