"""Experiment plans and job executors.

A job is a small dict; its result is written to `job['out']` (JSON). Finished jobs are skipped, so every stage
can be stopped and restarted. Job types: tune, pretrain, target, zeroshot, public.
"""
from __future__ import annotations

import os
import time

import torch

from . import configs, data, models, paths, splits, train

# (source, target) transfer pairs
PAIRS_X1 = [("citeseer", "cora"), ("dblp", "pubmed")]  # the manuscript's original pairs
PAIRS_X2 = [("acmv9", "dblpv7"), ("acmv9", "citationv1"), ("citationv1", "acmv9"),
            ("citationv1", "dblpv7"), ("dblpv7", "acmv9"), ("dblpv7", "citationv1")]  # shared features and labels
PAIRS_X6 = [("squirrel_filtered", "chameleon_filtered"), ("amazon_ratings", "roman_empire"),
            ("citeseer", "chameleon_filtered")]  # heterophilic targets
ALL_PAIRS = PAIRS_X1 + PAIRS_X2 + PAIRS_X6
ABLATION_PAIRS = [("citeseer", "cora"), ("acmv9", "dblpv7"), ("squirrel_filtered", "chameleon_filtered")]
ABLATION_MODELS = ["sgx", "sgx_order", "sgx_last", "sgx_sage2", "sgx_nodrop", "abl_allgcn", "abl_allsage"]

TUNE_DATASETS = {
    "budget20": ["cora", "citeseer", "pubmed", "acmv9", "citationv1", "dblpv7", "roman_empire", "amazon_ratings",
                 "chameleon_filtered", "squirrel_filtered"],
    "public": ["cora", "citeseer", "pubmed", "roman_empire", "amazon_ratings", "chameleon_filtered",
               "squirrel_filtered"],
}

PLANS = {
    "smoke": dict(n_cfg=2, tune_splits=1, budgets=[5], splits=[0], models=["gcn", "sgx", "gat"],
                  conds=["scratch", "finetune", "frozen", "zeroshot", "ctrl_shuffle", "ctrl_random"],
                  pairs=[("citeseer", "cora"), ("acmv9", "dblpv7")], tune={"budget20": ["cora", "dblpv7"],
                                                                         "public": ["cora", "chameleon_filtered"]},
                  ablation=True, ablation_pairs=[("citeseer", "cora")], public_splits=[0, 1]),
    "pilot": dict(n_cfg=30, tune_splits=3, budgets=[1, 5, 20], splits=list(range(5)), models=models.MAIN_MODELS,
                  conds=["scratch", "finetune", "frozen", "zeroshot"], pairs=ALL_PAIRS, tune=TUNE_DATASETS,
                  ablation=False, ablation_pairs=[], public_splits=list(range(10))),
    "full": dict(n_cfg=30, tune_splits=3, budgets=[1, 3, 5, 10, 20], splits=list(range(10)), models=models.MAIN_MODELS,
                 conds=["scratch", "finetune", "frozen", "zeroshot", "ctrl_shuffle", "ctrl_random"], pairs=ALL_PAIRS,
                 tune=TUNE_DATASETS, ablation=True, ablation_pairs=ABLATION_PAIRS, public_splits=list(range(10))),
}
PTYPE = {"finetune": "real", "frozen": "real", "zeroshot": "real", "ctrl_shuffle": "shuffle", "ctrl_random": "random",
         # candidate fine-tuning protocols (22 Oct 2026; used only if adopted under the pre-specified rule)
         "frz1": "real", "lpft": "real", "wise": "real"}
# self-supervised pretraining conditions: (method, graph pretrained on)
SSL_CONDS = {"ssl_dgi": ("dgi", "source"), "ssl_grace": ("grace", "source"),
             "ssl_dgi_self": ("dgi", "self"), "ssl_grace_self": ("grace", "self")}
DA_METHODS = {"dann": ["gcn", "sgx"], "udagcn": ["gcn"]}  # domain adaptation (pairs with shared features only)

# baselines plan (Reviewer 5, concern 8): SSL pretraining and domain adaptation for GCN and SG-XNet
PLANS["baselines"] = dict(n_cfg=30, tune_splits=3, budgets=[1, 5, 20], splits=list(range(10)), models=["gcn", "sgx"],
                          conds=[], pairs=ALL_PAIRS, tune=TUNE_DATASETS, ablation=False, ablation_pairs=[],
                          public_splits=[], ssl_conds=list(SSL_CONDS), ssl_models=["gcn", "sgx"], da=True)
PLANS["smoke"].update(ssl_conds=["ssl_dgi", "ssl_grace", "ssl_dgi_self"], ssl_models=["gcn", "sgx"], da=True)

# exploratory plan (23 Oct 2026). NOT part of the pre-specified design: the gated SG-XNet ("SG-XNet-G", layout
# "sgxg") was designed after the pilot's tuning results, tested on development splits (dev/devexp.py) and did not
# meet the pre-specified adoption rule. It is evaluated here as an explicitly exploratory variant, with its own
# tuning (same budget as every model: 30 configurations x 3 tuning splits) and the same evaluation splits, label
# budgets, transfer pairs and conditions as the main models of the full study (no negative controls, no ablation).
# Its tuned configurations are added to selected_*.json without changing any existing entry (select_merge).
EXPLORE_NOTE = ("Exploratory analysis. SG-XNet-G (every layer a learned per-channel mix of a GCN and a GraphSAGE-mean "
                "step) was designed after the pilot (22 Oct 2026), tested on development splits, and did not meet "
                "the pre-specified adoption rule. It is evaluated here with the same tuning budget, evaluation "
                "splits, label budgets, transfer pairs and conditions as the main models; its results are reported "
                "separately from the pre-specified comparisons.")
PLANS["explore"] = dict(n_cfg=30, tune_splits=3, budgets=[1, 3, 5, 10, 20], splits=list(range(10)), models=["sgxg"],
                        conds=["scratch", "finetune", "frozen", "zeroshot"], pairs=ALL_PAIRS, tune=TUNE_DATASETS,
                        ablation=False, ablation_pairs=[], public_splits=list(range(10)), select_merge=True,
                        focus="sgxg", show_models=models.MAIN_MODELS + ["sgxg"], gain_models=["sgxg", "sgx"],
                        vs_conds=["scratch", "finetune", "frozen", "zeroshot"], show_ablation=False,
                        note=EXPLORE_NOTE)
# end-to-end test of the exploratory plan (run after the smoke plan, in the same results folder)
PLANS["explore_smoke"] = dict(PLANS["explore"], n_cfg=2, tune_splits=1, budgets=[5], splits=[0],
                              pairs=[("citeseer", "cora"), ("acmv9", "dblpv7")],
                              tune={"budget20": ["cora", "dblpv7"], "public": ["cora", "chameleon_filtered"]},
                              public_splits=[0, 1], show_models=["gcn", "sgx", "gat", "sgxg"])
GATED_MODELS = {"sgxg"}


def aligned(src: str, tgt: str) -> bool:
    return src in data.ALIGNED and tgt in data.ALIGNED


def _cfg_for(sel: dict, tgt: str, model: str) -> dict:
    base = "sgx" if model.startswith("sgx_") or model.startswith("abl_") else model
    return sel[tgt][base]


# ----------------------------------------------------------------------------- plans
def tune_jobs(plan: dict) -> list[dict]:
    jobs = []
    for proto, dsets in plan["tune"].items():
        for d in dsets:
            for m in plan["models"]:
                for cfg in configs.sample_configs(m, plan["n_cfg"]):
                    for s in range(plan["tune_splits"]):
                        out = os.path.join(paths.RESULTS, "tune", proto, d, m, f"c{cfg['cid']:02d}_s{s}.json")
                        jobs.append(dict(type="tune", proto=proto, dataset=d, model=m, cfg=cfg, split=s, out=out))
    return jobs


def _model_list(plan, pair):
    ms = list(plan["models"])
    if plan["ablation"] and pair in plan["ablation_pairs"]:
        ms += [m for m in ABLATION_MODELS if m not in ms]
    return ms


def pretrain_jobs(plan: dict, sel: dict) -> list[dict]:
    jobs, seen = [], set()
    ptypes = sorted({PTYPE[c] for c in plan["conds"] if c in PTYPE})
    for src, tgt in plan["pairs"]:
        for m in _model_list(plan, (src, tgt)):
            cfg = _cfg_for(sel, tgt, m)
            for s in plan["splits"]:
                for pt in ptypes:
                    if m not in plan["models"] and pt != "real":
                        continue  # ablation variants: real pretraining only
                    tag = f"{src}__{tgt}"
                    out = os.path.join(paths.RESULTS, "pretrain", tag, m, f"{pt}_s{s}.json")
                    if out in seen:
                        continue
                    seen.add(out)
                    ckpt = os.path.join(paths.CKPT, tag, m, f"{pt}_s{s}.pt")
                    jobs.append(dict(type="pretrain", source=src, target=tgt, model=m, cfg=cfg, seed=s, ptype=pt,
                                     parts=["encoder", "body", "out"] if aligned(src, tgt) else ["body"],
                                     ckpt=ckpt, out=out))
    return jobs


def target_jobs(plan: dict, sel: dict) -> list[dict]:
    jobs, seen = [], set()
    for src, tgt in plan["pairs"]:
        al = aligned(src, tgt)
        for m in _model_list(plan, (src, tgt)):
            cfg = _cfg_for(sel, tgt, m)
            conds = plan["conds"] if m in plan["models"] else ["scratch", "finetune"]
            for k in plan["budgets"]:
                if m not in plan["models"] and k not in (5, 20):
                    continue  # ablation variants at two budgets only
                for s in plan["splits"]:
                    for c in conds:
                        if c == "zeroshot" and not al:
                            continue
                        sub = "scratch" if c == "scratch" else f"{c}@{src}"
                        out = os.path.join(paths.RESULTS, "target", tgt, m, sub, f"k{k}_s{s}.json")
                        if out in seen:
                            continue
                        seen.add(out)
                        job = dict(type="zeroshot" if c == "zeroshot" else "target", source=None if c == "scratch" else src,
                                   target=tgt, model=m, cfg=cfg, cond=c, k=k, split=s, aligned=al, out=out)
                        if c != "scratch":
                            job["ckpt"] = os.path.join(paths.CKPT, f"{src}__{tgt}", m, f"{PTYPE[c]}_s{s}.pt")
                        jobs.append(job)
    return jobs


def public_jobs(plan: dict, sel: dict) -> list[dict]:
    jobs = []
    for d in plan["tune"]["public"]:
        for m in plan["models"]:
            for s in plan["public_splits"]:
                out = os.path.join(paths.RESULTS, "public", d, m, f"s{s}.json")
                jobs.append(dict(type="public", dataset=d, model=m, cfg=sel[d][m], split=s, out=out))
    return jobs


def _ssl_tag(cond, src, tgt):
    method, where = SSL_CONDS[cond]
    graph = src if where == "source" else tgt
    parts = ["encoder", "body"] if (where == "self" or aligned(src, tgt)) else ["body"]
    return method, graph, f"{graph}__{tgt}", parts


def sslpre_jobs(plan: dict, sel: dict) -> list[dict]:
    jobs, seen = [], set()
    for src, tgt in plan["pairs"]:
        for m in plan.get("ssl_models", []):
            cfg = sel[tgt][m]
            for s in plan["splits"]:
                for c in plan.get("ssl_conds", []):
                    method, graph, tag, parts = _ssl_tag(c, src, tgt)
                    out = os.path.join(paths.RESULTS, "sslpre", tag, m, f"{method}_s{s}.json")
                    if out in seen:
                        continue
                    seen.add(out)
                    jobs.append(dict(type="sslpre", method=method, graph=graph, target=tgt, model=m, cfg=cfg, seed=s,
                                     parts=parts, ckpt=os.path.join(paths.CKPT, "ssl", tag, m, f"{method}_s{s}.pt"),
                                     out=out))
    return jobs


def ssl_target_jobs(plan: dict, sel: dict) -> list[dict]:
    jobs, seen = [], set()
    for src, tgt in plan["pairs"]:
        for m in plan.get("ssl_models", []):
            cfg = sel[tgt][m]
            for c in plan.get("ssl_conds", []):
                method, graph, tag, parts = _ssl_tag(c, src, tgt)
                for variant in (c, f"{c}_probe"):  # fine-tune everything / linear probe on the frozen network
                    sub = variant if SSL_CONDS[c][1] == "self" else f"{variant}@{src}"
                    for k in plan["budgets"]:
                        for s in plan["splits"]:
                            out = os.path.join(paths.RESULTS, "target", tgt, m, sub, f"k{k}_s{s}.json")
                            if out in seen:
                                continue
                            seen.add(out)
                            jobs.append(dict(type="target", source=None if SSL_CONDS[c][1] == "self" else src,
                                             target=tgt, model=m, cfg=cfg, cond=variant, k=k, split=s,
                                             aligned=aligned(src, tgt), parts=parts,
                                             freeze=parts if variant.endswith("_probe") else None,
                                             ckpt=os.path.join(paths.CKPT, "ssl", tag, m, f"{method}_s{s}.pt"), out=out))
    return jobs


def da_jobs(plan: dict, sel: dict) -> list[dict]:
    jobs = []
    if not plan.get("da"):
        return jobs
    for src, tgt in plan["pairs"]:
        if not aligned(src, tgt):
            continue
        for method, ms in DA_METHODS.items():
            for m in ms:
                for k in plan["budgets"]:
                    for s in plan["splits"]:
                        out = os.path.join(paths.RESULTS, "target", tgt, m, f"{method}@{src}", f"k{k}_s{s}.json")
                        jobs.append(dict(type="da", method=method, source=src, target=tgt, model=m, cfg=sel[tgt][m],
                                         cond=method, k=k, split=s, aligned=True, out=out))
    return jobs


# ----------------------------------------------------------------------------- executors
def _seed(s: int):
    torch.manual_seed(s)


def _gates(model, state=None):
    """Mean weight of the GCN branch in every gated layer (SG-XNet-G only; 0 = pure GraphSAGE, 1 = pure GCN),
    taken at the best-validation epoch when that epoch's state is given. None for every other model."""
    layers = [m for m in model.modules() if isinstance(m, models._Gate)]
    if not layers:
        return None
    if state is not None:
        model.load_state_dict(state)
    return [round(m.gate(), 4) for m in layers]


def exec_tune(job):
    g = data.load(job["dataset"])
    if job["proto"] == "budget20":
        masks = splits.budget_split(g.y_np, 20, seed=1000 + job["split"], multilabel=g.multilabel)
    else:
        masks = splits.public_split(g.public, job["split"])
    _seed(job["split"])
    cfg = job["cfg"]
    model = models.build(job["model"], cfg, g.in_dim, g.num_classes)
    best, _ = train.fit(model, g, masks, cfg["lr"], cfg["wd"], max_epochs=500, patience=50)
    return best


def _source_graph(job):
    g = data.load(job["source"])
    if job["ptype"] == "shuffle":
        g = data.shuffled_labels(g, seed=7 + job["seed"])
    elif job["ptype"] == "random":
        g = data.random_like(g, seed=13 + job["seed"])
    return g


def exec_pretrain(job):
    """Supervised pretraining on the source graph, with the target's tuned architecture (needed to transfer the
    layers). The learning rate is chosen from FINAL_LRS on the source validation split, the same rule as every
    target run (23 Oct 2026: using the target-tuned learning rate made GCN pretraining collapse on the ACDNE sources
    of two targets, source validation micro-F1 about 33%). Every learning-rate trial is recorded in `lr_trials`."""
    g = _source_graph(job)
    masks = splits.ratio_split(g.y_np, seed=job["seed"], multilabel=g.multilabel)
    cfg = job["cfg"]
    tg = data.load(job["target"])
    out_dim = g.num_classes
    trials = []
    for lr in configs.FINAL_LRS:
        _seed(job["seed"])  # identical initialisation for every learning rate
        model = models.build(job["model"], cfg, g.in_dim, out_dim)
        res, state = train.fit(model, g, masks, lr, cfg["wd"], max_epochs=1000, patience=100, keep_state=True)
        if state is None:  # diverged at the first epoch; keep the initial weights so dependants can still run
            state = model.state_dict()
        res["lr"] = lr
        trials.append((res, state, model))
    best, state, model = max(trials, key=lambda t: (t[0]["val"], -t[0]["val_loss"]))
    train.save_state(state, job["parts"], job["ckpt"])
    best = dict(best)
    best["lr_trials"] = [{k: r.get(k) for k in ("lr", "val", "val_loss", "epoch", "epochs", "diverged")}
                         for r, _, _ in trials]
    best["target_in_dim"], best["target_classes"] = tg.in_dim, tg.num_classes
    if job["model"] in GATED_MODELS:
        best["gates"] = _gates(model, state)
    return best


def _target_masks(g, job):
    return splits.budget_split(g.y_np, job["k"], seed=job["split"], multilabel=g.multilabel)


def exec_target(job):
    g = data.load(job["target"])
    masks = _target_masks(g, job)
    cfg = job["cfg"]
    state = train.load_state(job["ckpt"]) if job["cond"] != "scratch" else None
    parts = job.get("parts") or (["encoder", "body", "out"] if job["aligned"] else ["body"])
    runs = []
    cond = job["cond"]
    for lr in configs.FINAL_LRS:
        _seed(1000 + job["split"])  # identical encoder/head initialisation for every condition of a split
        model = models.build(job["model"], cfg, g.in_dim, g.num_classes)
        if state is not None:
            train.transfer(model, state, parts)
            if cond == "frozen":
                train.freeze(model, ["encoder", "body"] if job["aligned"] else ["body"])
            elif cond == "frz1":  # only the first message-passing layer (and a transferred encoder) stays fixed
                train.freeze_prefixes(model, (["encoder."] if job["aligned"] else []) + ["body.0."])
            elif job.get("freeze"):
                train.freeze(model, job["freeze"])
        if cond == "lpft":  # linear probe (frozen body), then fine-tune everything at a tenth of the learning rate
            train.freeze(model, ["encoder", "body"] if job["aligned"] else ["body"])
            b1, s1 = train.fit(model, g, masks, lr, cfg["wd"], max_epochs=1000, patience=100, keep_state=True)
            if s1 is not None:
                model.load_state_dict(s1)
            train.unfreeze(model)
            b2, _ = train.fit(model, g, masks, lr / 10, cfg["wd"], max_epochs=1000, patience=100)
            best = max([b1, b2], key=lambda r: (r["val"], -r["val_loss"]))
            best["phase"] = 2 if best is b2 else 1
        elif cond == "wise":  # interpolate pretrained and fine-tuned weights; alpha chosen on validation
            _, sft = train.fit(model, g, masks, lr, cfg["wd"], max_epochs=1000, patience=100, keep_state=True)
            best = train.wise_select(model, g, masks, state, sft or model.state_dict(), parts)
        else:
            gated = job["model"] in GATED_MODELS  # keep the best epoch's state only to read its gates
            best, st = train.fit(model, g, masks, lr, cfg["wd"], max_epochs=1000, patience=100, keep_state=gated)
            if gated:
                best["gates"] = _gates(model, st)
        best["lr"] = lr
        runs.append(best)
    sel = max(runs, key=lambda r: (r["val"], -r["val_loss"]))
    return {"sel": sel, "runs": runs, "n_train": int(masks["train"].sum()), "n_val": int(masks["val"].sum()),
            "n_test": int(masks["test"].sum()), "metric": train.primary_name(g.multilabel)}


def exec_zeroshot(job):
    g = data.load(job["target"])
    masks = _target_masks(g, job)
    _seed(1000 + job["split"])
    model = models.build(job["model"], job["cfg"], g.in_dim, g.num_classes)
    train.transfer(model, train.load_state(job["ckpt"]), ["encoder", "body", "out"])
    res = train.evaluate(model, g, masks)
    if job["model"] in GATED_MODELS:
        res["gates"] = _gates(model)
    return {"sel": res, "runs": [res], "n_train": 0, "n_val": int(masks["val"].sum()),
            "n_test": int(masks["test"].sum()), "metric": train.primary_name(g.multilabel)}


def exec_public(job):
    g = data.load(job["dataset"])
    masks = splits.public_split(g.public, job["split"])
    cfg = job["cfg"]
    runs = []
    for lr in configs.FINAL_LRS:
        _seed(1000 + job["split"])
        model = models.build(job["model"], cfg, g.in_dim, g.num_classes)
        gated = job["model"] in GATED_MODELS
        best, st = train.fit(model, g, masks, lr, cfg["wd"], max_epochs=1000, patience=100, keep_state=gated)
        if gated:
            best["gates"] = _gates(model, st)
        best["lr"] = lr
        runs.append(best)
    sel = max(runs, key=lambda r: (r["val"], -r["val_loss"]))
    return {"sel": sel, "runs": runs, "metric": train.primary_name(g.multilabel)}


def exec_sslpre(job):
    from . import baselines

    g = data.load(job["graph"])
    state, info = baselines.pretrain_ssl(job["method"], job["model"], job["cfg"], g, job["seed"])
    train.save_state(state, job["parts"], job["ckpt"])
    return info


def exec_da(job):
    from . import baselines

    gs, gt = data.load(job["source"]), data.load(job["target"])
    ms = splits.ratio_split(gs.y_np, seed=job["split"], multilabel=gs.multilabel)
    mt = _target_masks(gt, job)
    runs = []
    for lr in configs.FINAL_LRS:
        best = baselines.fit_da(job["method"], job["model"], job["cfg"], gs, ms, gt, mt, lr, seed=1000 + job["split"])
        best["lr"] = lr
        runs.append(best)
    sel = max(runs, key=lambda r: (r["val"], -r["val_loss"]))
    return {"sel": sel, "runs": runs, "n_train": int(mt["train"].sum()), "n_val": int(mt["val"].sum()),
            "n_test": int(mt["test"].sum()), "metric": train.primary_name(gt.multilabel)}


EXEC = {"tune": exec_tune, "pretrain": exec_pretrain, "target": exec_target, "zeroshot": exec_zeroshot,
        "public": exec_public, "sslpre": exec_sslpre, "da": exec_da}


def run_job(job) -> dict:
    t0 = time.time()
    res = EXEC[job["type"]](job)
    meta = {k: v for k, v in job.items() if k not in ("out",)}
    flat = job["type"] in ("tune", "pretrain", "sslpre")
    return {**meta, **(res if flat else {}), **({} if flat else {"result": res}), "seconds": round(time.time() - t0, 2)}
