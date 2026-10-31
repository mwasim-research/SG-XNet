"""Development experiments for SG-XNet tweaks (started 22 Oct 2026).

Integrity rules for this script:
  * DEV splits only (seeds 2000-2003 for the target splits and for pretraining). The evaluation splits 0-9 of the
    pilot and the full study are never used here.
  * Decisions use VALIDATION scores only. Test scores are stored (so nothing has to be re-run later) but are not
    printed by `summary`, and are not to be read until the design of any new variant has been frozen in writing.

E1  transfer protocol for SG-XNet (configs from the pilot's tuning):
      scratch | ft (fine-tune all) | frz (frozen body) | frz1 (only the first, GraphSAGE, layer frozen)
      | lpft (frozen-body phase, then fine-tune all at lr/10) | wise (weight-space interpolation between the
      pretrained and the fine-tuned weights, alpha chosen on validation)
E2  architecture: "sgxg", a gated hybrid in which every body layer is w * GCN(h) + (1 - w) * SAGE_mean(h) with a
      learned per-channel gate w = sigmoid(a) (initialised at 0.5), compared with SG-XNet, GCN and GraphSAGE.

Run from the project folder (results go to results_dev/, separate from results/ so that the pilot and full-study
analysis never sees them):
    python dev/devexp.py run results_dev [workers]
    python dev/devexp.py summary results_dev
"""
import glob
import json
import multiprocessing as mp
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import torch  # noqa: E402

torch.set_num_threads(1)
from sgx import data, jobs, models, splits, train  # noqa: E402

SEL_PATH = os.path.join(ROOT, "results", "selected_budget20.json")
LRS = (0.01, 0.005, 0.001)
DEV = [2000, 2001, 2002, 2003]
E1_PAIRS = [("acmv9", "dblpv7"), ("citeseer", "cora"), ("squirrel_filtered", "chameleon_filtered")]
E1_VARIANTS = ["scratch", "ft", "frz", "frz1", "lpft", "wise"]
E2_DATASETS = ["cora", "citeseer", "acmv9", "chameleon_filtered", "squirrel_filtered", "roman_empire"]


# the gated hybrid layer is sgx.models._Gate (layout "sgxg"); it is not part of any plan of the main study
assert "sgxg" in models.LAYOUTS and "gate" in models._LAYER


def cfg_for(tgt, model):
    sel = json.load(open(SEL_PATH))
    base = "sgx" if model == "sgxg" else model
    return dict(sel[tgt][base])


def masks_for(g, k, split):
    return splits.budget_split(g.y_np, k, seed=split, multilabel=g.multilabel)


# ----------------------------------------------------------------------------- pretraining (cached)
def pretrained(out, src, tgt, model, seed):
    path = os.path.join(out, "ckpt", f"{src}__{tgt}", f"{model}_s{seed}.pt")
    if os.path.exists(path):
        return torch.load(path)
    cfg = cfg_for(tgt, model)
    g = data.load(src)
    masks = splits.ratio_split(g.y_np, seed=seed, multilabel=g.multilabel)
    torch.manual_seed(seed)
    m = models.build(model, cfg, g.in_dim, g.num_classes)
    _, state = train.fit(m, g, masks, cfg["lr"], cfg["wd"], max_epochs=1000, patience=100, keep_state=True)
    state = state or m.state_dict()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)
    return state


# ----------------------------------------------------------------------------- target runs
def _freeze(m, prefixes):
    for name, p in m.named_parameters():
        if any(name.startswith(x) for x in prefixes):
            p.requires_grad_(False)


def _unfreeze(m):
    for p in m.parameters():
        p.requires_grad_(True)


@torch.no_grad()
def _eval(m, g, masks):
    m.eval()
    o = m(g)
    r = train.evaluate(m, g, masks)
    r["val_loss"] = float(train.loss_fn(o[masks["val"]], g.y[masks["val"]], g.multilabel))
    return r


def run_target(job):
    out, model, src, tgt, k, split, variant = (job[x] for x in ("out", "model", "src", "tgt", "k", "split", "variant"))
    g = data.load(tgt)
    masks = masks_for(g, k, split)
    cfg = cfg_for(tgt, model)
    al = bool(src) and jobs.aligned(src, tgt)
    parts = ["encoder", "body", "out"] if al else ["body"]
    frozen = ["encoder.", "body."] if al else ["body."]
    first = (["encoder."] if al else []) + ["body.0."]
    state = pretrained(out, src, tgt, model, job["pseed"]) if variant != "scratch" else None
    runs = []
    for lr in LRS:
        torch.manual_seed(1000 + split)
        m = models.build(model, cfg, g.in_dim, g.num_classes)
        if state is not None:
            train.transfer(m, state, parts)
        if variant in ("scratch", "ft", "frz", "frz1"):
            if variant == "frz":
                _freeze(m, frozen)
            if variant == "frz1":
                _freeze(m, first)
            best, _ = train.fit(m, g, masks, lr, cfg["wd"])
        elif variant == "lpft":
            _freeze(m, frozen)
            b1, s1 = train.fit(m, g, masks, lr, cfg["wd"], keep_state=True)
            if s1 is not None:
                m.load_state_dict(s1)
            _unfreeze(m)
            b2, _ = train.fit(m, g, masks, lr / 10, cfg["wd"])
            best = max([b1, b2], key=lambda r: (r["val"], -r["val_loss"]))
            best["phase"] = 2 if best is b2 else 1
        elif variant == "wise":
            _, sft = train.fit(m, g, masks, lr, cfg["wd"], keep_state=True)
            sft = sft or m.state_dict()
            cands = []
            for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
                mix = {}
                for key_, v in sft.items():
                    if key_.split(".")[0] in parts and key_ in state and v.is_floating_point():
                        mix[key_] = (1 - alpha) * state[key_].float() + alpha * v
                    else:
                        mix[key_] = v
                m.load_state_dict(mix)
                r = _eval(m, g, masks)
                r["alpha"] = alpha
                cands.append(r)
            best = max(cands, key=lambda r: (r["val"], -r["val_loss"], r["alpha"]))
        else:
            raise ValueError(variant)
        best["lr"] = lr
        runs.append(best)
    sel = max(runs, key=lambda r: (r["val"], -r["val_loss"]))
    return dict(job, sel=sel, runs=runs)


def run_one(job):
    t0 = time.time()
    try:
        res = run_target(job)
        res["seconds"] = round(time.time() - t0, 1)
        with open(job["path"] + ".tmp", "w") as f:
            json.dump(res, f)
        os.replace(job["path"] + ".tmp", job["path"])
        return job["path"], None
    except Exception:  # noqa: BLE001
        import traceback
        return job["path"], traceback.format_exc()


def job_list(out):
    js = []

    def add(exp, model, src, tgt, k, split, variant, pseed=None):
        name = f"{exp}__{model}__{src or 'none'}__{tgt}__k{k}__s{split}__{variant}.json"
        js.append(dict(out=out, exp=exp, model=model, src=src, tgt=tgt, k=k, split=split, variant=variant,
                       pseed=pseed if pseed is not None else split, path=os.path.join(out, "runs", name)))

    for s in DEV:  # E1: protocol variants for SG-XNet
        for src, tgt in E1_PAIRS:
            for k in (1, 5):
                for v in E1_VARIANTS:
                    add("E1", "sgx", None if v == "scratch" else src, tgt, k, s, v)
    for s in DEV[:3]:  # E2: gated hybrid, from scratch at k = 20 (and the same for GCN, SAGE, SG-XNet)
        for d in E2_DATASETS:
            for model in ("sgxg", "sgx", "gcn", "sage"):
                add("E2", model, None, d, 20, s, "scratch")
    for s in DEV:  # E2: gated hybrid, transfer at k = 5
        for src, tgt in E1_PAIRS:
            for v in ("scratch", "ft"):
                add("E2t", "sgxg", None if v == "scratch" else src, tgt, 5, s, v)
    return js


def run(out, workers):
    os.makedirs(os.path.join(out, "runs"), exist_ok=True)
    todo = [j for j in job_list(out) if not os.path.exists(j["path"])]
    # split-major order, so that every comparison has a first (noisy) estimate early and is refined split by split
    todo.sort(key=lambda j: (DEV.index(j["split"]), {"E2": 0, "E1": 1, "E2t": 2}[j["exp"]]))
    # pretraining first (cheap, cached), so that parallel workers do not train the same source model twice
    need = sorted({(j["src"], j["tgt"], j["model"], j["pseed"]) for j in todo if j["variant"] != "scratch"})
    print(f"{len(todo)} runs to do; {len(need)} pretrained models needed", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers) as pool:
        list(pool.starmap(pretrained, [(out, *n) for n in need]))
        print(f"pretraining done ({time.time() - t0:.0f} s)", flush=True)
        done = 0
        for path, err in pool.imap_unordered(run_one, todo):
            done += 1
            if err:
                print("ERROR", os.path.basename(path), err.strip().splitlines()[-1], flush=True)
            if done % 20 == 0 or done == len(todo):
                print(f"{done}/{len(todo)} runs ({time.time() - t0:.0f} s)", flush=True)


def summary(out):
    """VALIDATION scores only (mean over dev splits, in %)."""
    rows = [json.load(open(p)) for p in glob.glob(os.path.join(out, "runs", "*.json"))]
    from collections import defaultdict
    agg = defaultdict(list)
    for r in rows:
        agg[(r["exp"], r["model"], r["src"], r["tgt"], r["k"], r["variant"])].append(100 * r["sel"]["val"])
    for exp in ("E1", "E2", "E2t"):
        print(f"\n== {exp} (validation, mean over dev splits)")
        keys = sorted((k for k in agg if k[0] == exp), key=lambda k: tuple("" if x is None else str(x) for x in k))
        if exp == "E1":
            for src, tgt in E1_PAIRS:
                for k in (1, 5):
                    cells = []
                    for v in E1_VARIANTS:
                        vals = agg.get(("E1", "sgx", None if v == "scratch" else src, tgt, k, v), [])
                        cells.append(f"{v} {sum(vals) / len(vals):5.1f} (n={len(vals)})" if vals else f"{v} -")
                    print(f"{src}->{tgt} k={k}: " + " | ".join(cells))
        else:
            for key in keys:
                vals = agg[key]
                print(f"{key[1]:5s} {str(key[2]):18s} -> {key[3]:20s} k={key[4]:2d} {key[5]:8s} "
                      f"{sum(vals) / len(vals):5.1f} (n={len(vals)})")


if __name__ == "__main__":
    cmd, outdir = sys.argv[1], os.path.abspath(sys.argv[2])
    if cmd == "run":
        run(outdir, int(sys.argv[3]) if len(sys.argv) > 3 else 2)
    else:
        summary(outdir)
