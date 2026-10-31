"""Hyper-parameter search space, configuration sampling and selection (Reviewer 5, request 2).

Every model is tuned separately for every dataset by random search over the same space, including depth (number
of message-passing layers K) and width. The same random sample of configurations is used for every dataset.
Selection: highest mean validation metric over the tuning splits (ties: lowest mean validation loss).
"""
from __future__ import annotations

import itertools
import json
import os
import zlib

import numpy as np

from . import paths

SPACE = {
    "K": [2, 3, 4, 5],
    "hidden": [64, 128, 256],
    "dropout": [0.0, 0.2, 0.5, 0.7],
    "lr": [0.01, 0.005, 0.001],
    "wd": [0.0, 5e-4, 5e-3],
    # layout v2b (22 Oct 2026): the standard tricks of Luo et al. (2024), tuned for every model
    "residual": [False, True],
    "norm": [False, True],
}
EXTRA = {"gat": {"heads": [1, 4, 8]}, "sage": {"aggr": ["mean", "max"]}, "sgx": {"aggr": ["mean", "max"]}}
FINAL_LRS = [0.01, 0.005, 0.001]  # every final run picks its learning rate from this set on validation


def grid(model: str) -> list[dict]:
    space = dict(SPACE, **EXTRA.get(model, {}))
    keys = list(space)
    return [dict(zip(keys, vals)) for vals in itertools.product(*[space[k] for k in keys])]


def sample_configs(model: str, n: int) -> list[dict]:
    g = grid(model)
    rng = np.random.default_rng(zlib.crc32(model.encode()))
    idx = rng.choice(len(g), size=min(n, len(g)), replace=False)
    out = []
    for cid, i in enumerate(sorted(idx)):
        c = dict(g[i])
        c["cid"] = cid
        out.append(c)
    return out


def selected_path(proto: str) -> str:
    return os.path.join(paths.RESULTS, f"selected_{proto}.json")


def select(proto: str, tune_rows: list[dict], write: bool = True) -> dict:
    """tune_rows: results of tuning jobs. Returns {dataset: {model: cfg}} and (unless write=False) writes
    selected_<proto>.json."""
    acc: dict = {}
    for r in tune_rows:
        key = (r["dataset"], r["model"], r["cfg"]["cid"])
        acc.setdefault(key, {"cfg": r["cfg"], "val": [], "val_loss": []})
        acc[key]["val"].append(r["val"])
        acc[key]["val_loss"].append(r["val_loss"])
    n_splits = max((len(v["val"]) for v in acc.values()), default=0)
    best: dict = {}
    for (d, m, cid), v in acc.items():
        if len(v["val"]) < n_splits:
            continue  # incomplete configuration
        score = (float(np.mean(v["val"])), -float(np.mean(v["val_loss"])))
        cur = best.get((d, m))
        if cur is None or score > cur[0]:
            best[(d, m)] = (score, v["cfg"])
    out: dict = {}
    for (d, m), (score, cfg) in sorted(best.items()):
        cfg = {k: v for k, v in cfg.items()}
        cfg["tune_val"] = score[0]
        out.setdefault(d, {})[m] = cfg
    if write:
        paths.ensure(paths.RESULTS)
        with open(selected_path(proto), "w") as f:
            json.dump(out, f, indent=1)
    return out


def merge_selected(proto: str, new: dict) -> dict:
    """Adds the (dataset, model) entries of `new` to selected_<proto>.json and leaves every existing entry of the
    other models exactly as it was (used by exploratory plans, so the main study's selection never changes)."""
    sel = load_selected(proto)
    for d, ms in new.items():
        sel.setdefault(d, {}).update(ms)
    tmp = selected_path(proto) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(sel, f, indent=1)
    os.replace(tmp, selected_path(proto))
    return sel


def load_selected(proto: str) -> dict:
    with open(selected_path(proto)) as f:
        return json.load(f)
