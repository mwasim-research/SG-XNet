"""(base copied from overnight_2026-10-27)
Post hoc experiments, night of 27-28 Oct 2026 (after the main study; to be reported as post hoc).

Runs on the Mac with the v2c code in ../../sgx, read-only. It reads the main study's results (selected
configurations, checkpoints, learning rates) from ../../results and writes ONLY to ../../results_posthoc_overnight.

  Stage 1  l2sp   L2-SP fine-tuning (Li et al., ICML 2018): during fine-tuning, the penalty alpha * ||theta - theta0||^2
                  keeps the weights near the pretrained ones (theta0), replacing weight decay. 6 models x 6 aligned pairs,
                  k = 1 and 5, 10 splits. Learning rate = the one the main study selected for plain fine-tuning of that
                  run; alpha in {0.001, 0.01, 0.1, 1} chosen on target validation (test never used).
  Stage 2  uda    UDAGCN with the SG-XNet backbone at full size (6 pairs, k = 1, 5, 20, 10 splits), same protocol as the
                  main DA stage; the global PPMI view also replaces SG-XNet's mean-aggregation matrix (row-normalised).
           cka    Representation drift: linear CKA between body embeddings before and after fine-tuning, 6 models x 6
                  aligned pairs, k = 1 and 5, 10 splits (+ all-GCN with SG-XNet's settings, ACMv9->DBLPv7, k = 5).
  Stage 3  tt     Transfer-aware tuning for GCN and SG-XNet: the same 30 candidate configurations as the main tuning,
                  scored by fine-tuned validation accuracy after pretraining (one source per target, seed 0; k = 5;
                  validation nodes of splits 0-2), then the selected configuration is evaluated like the main study
                  (10 pretraining seeds, both sources, zero-shot and fine-tuned, k = 1 and 5, 10 splits).
Resumable: finished jobs are skipped. Progress: results_posthoc_overnight/status.json and logs/overnight.log.
"""
import json
import multiprocessing as mp
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(HERE))
MAIN = os.environ.get("SGX_MAIN_RESULTS", os.path.join(PROJECT, "results"))
OUT = os.environ.get("SGX_POSTHOC_OUT", os.path.join(PROJECT, "results_posthoc_night2"))
os.environ["SGX_RESULTS"] = OUT  # anything the sgx package writes goes to OUT, never to the main results
sys.path.insert(0, PROJECT)
TEST = os.environ.get("OVERNIGHT_TEST") == "1"

MODELS = ["gcn", "sage", "gat", "gin", "mlp", "sgx"]
AL = ["acmv9", "citationv1", "dblpv7"]
PAIRS = [(s, t) for t in AL for s in AL if s != t]
INVALID = {("gat", "acmv9", "dblpv7")}  # failed pretraining in the main study
ALPHAS = [0.001, 0.01, 0.1, 1.0]
SPLITS = [0] if TEST else list(range(10))


def log(msg):
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(os.path.join(OUT, "logs", "night2.log"), "a") as f:
        f.write(line + "\n")


def status(stage, done, total, errors, started):
    el = time.time() - started
    rate = done / el if el > 0 else 0
    eta = (total - done) / rate / 60 if rate > 0 else None
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "status.json.tmp"), "w") as f:
        json.dump(dict(stage=stage, done=done, total=total, errors=errors, percent=round(100 * done / max(total, 1), 1),
                       eta_min=None if eta is None else round(eta, 1), updated=time.strftime("%Y-%m-%d %H:%M:%S")), f)
    os.replace(os.path.join(OUT, "status.json.tmp"), os.path.join(OUT, "status.json"))


# ----------------------------------------------------------------------------- worker side
def _init():
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    _patch_ppmi()


def _patch_ppmi():
    import numpy as np
    import scipy.sparse as sp
    from sgx import baselines, data, paths
    if getattr(baselines, "_sgx_patched", False):
        return
    orig = baselines.ppmi_view

    def view(g, steps=3):
        v = orig(g, steps)
        m = sp.load_npz(os.path.join(paths.CACHE, f"{g.name}_ppmi{steps}.npz"))
        d = np.asarray(m.sum(1)).ravel()
        d[d == 0] = 1
        rm = (sp.diags(1.0 / d) @ m).tocsr()
        v.A_mean, v.A_mean_t = data._csr_from_scipy(rm), data._csr_from_scipy(rm.T)
        return v
    baselines.ppmi_view = view
    baselines._sgx_patched = True


def _fit_l2sp(model, g, masks, lr, alpha, theta0, max_epochs=1000, patience=100):
    """train.fit with weight decay replaced by alpha * sum ||theta - theta0||^2 over the transferred parameters."""
    import math
    import torch
    from sgx import train
    if train.QUICK:
        max_epochs, patience = 5, 2
    tr, va, te = masks["train"], masks["val"], masks["test"]
    ml, key = g.multilabel, train.primary_name(g.multilabel)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.Adam([p for _, p in named], lr=lr, weight_decay=0.0)
    best, ep = {"val": -1.0, "val_loss": math.inf, "epoch": -1}, 0
    for ep in range(max_epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        out = model(g)
        loss = train.loss_fn(out[tr], g.y[tr], ml)
        pen = sum(((p - theta0[n]) ** 2).sum() for n, p in named if n in theta0)
        total = loss + alpha * pen
        if not torch.isfinite(total):
            best["diverged"] = True
            break
        total.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            out = model(g)
            vl = float(train.loss_fn(out[va], g.y[va], ml))
            vm = train.metrics(out[va], g.y[va], ml)
            if vm[key] > best["val"] + 1e-9 or (abs(vm[key] - best["val"]) <= 1e-9 and vl < best["val_loss"]):
                tm = train.metrics(out[te], g.y[te], ml)
                best = {"val": vm[key], "val_loss": vl, "epoch": ep, "test": tm[key],
                        **{f"val_{a}": b for a, b in vm.items()}, **{f"test_{a}": b for a, b in tm.items()}}
        if ep - best["epoch"] >= patience:
            break
    best["epochs"] = ep + 1
    best.setdefault("test", float("nan"))
    return best


def _cka(x, y):
    import numpy as np
    x = x - x.mean(0)
    y = y - y.mean(0)
    return float(np.linalg.norm(y.T @ x) ** 2 / (np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)))


def ex_l2sp(job):
    import torch
    from sgx import data, models, train, jobs as J
    g = data.load(job["target"])
    masks = J._target_masks(g, job)
    st0 = train.load_state(job["ckpt"])
    runs = []
    for a in ALPHAS:
        J._seed(1000 + job["split"])
        net = models.build(job["model"], job["cfg"], g.in_dim, g.num_classes)
        train.transfer(net, st0, ["encoder", "body", "out"])
        theta0 = {n: p.detach().clone() for n, p in net.named_parameters()}
        r = _fit_l2sp(net, g, masks, job["lr"], a, theta0)
        r["alpha"] = a
        runs.append(r)
    sel = max(runs, key=lambda r: (r["val"], -r["val_loss"]))
    return {"sel": sel, "runs": runs, "lr": job["lr"], "metric": train.primary_name(g.multilabel)}


def ex_cka(job):
    import torch
    from sgx import data, models, train, jobs as J
    g = data.load(job["target"])
    masks = J._target_masks(g, job)
    J._seed(1000 + job["split"])
    net = models.build(job["model"], job["cfg"], g.in_dim, g.num_classes)
    train.transfer(net, train.load_state(job["ckpt"]), ["encoder", "body", "out"])
    net.eval()
    with torch.no_grad():
        z0 = net.embed(g).numpy()
        zs = train.evaluate(net, g, masks)["test"]
    w0 = {a: b.detach().clone() for a, b in net.state_dict().items() if a.startswith("body.") and b.is_floating_point()}
    best, state = train.fit(net, g, masks, job["lr"], job["cfg"]["wd"], max_epochs=1000, patience=100, keep_state=True)
    if state is not None:
        net.load_state_dict(state)
    net.eval()
    with torch.no_grad():
        z1 = net.embed(g).numpy()
    w1 = net.state_dict()
    num = sum(float(((w1[a] - w0[a]) ** 2).sum()) for a in w0)
    den = sum(float((w0[a] ** 2).sum()) for a in w0)
    return {"cka": _cka(z0, z1), "wchange": (num / den) ** 0.5, "zs_test": zs, "ft_test": best["test"],
            "main_ft_test": job.get("main_ft_test"), "epoch": best["epoch"]}


def ex_main(job):
    from sgx import jobs as J
    return J.EXEC[job["type"]](job)


EX = {"l2sp": ex_l2sp, "cka": ex_cka}


def run_one(job):
    t0 = time.time()
    try:
        res = EX[job["type"]](job) if job["type"] in EX else ex_main(job)
        meta = {k: v for k, v in job.items() if k != "out"}
        flat = job["type"] == "pretrain"
        rec = {**meta, **(res if flat else {"result": res}), "seconds": round(time.time() - t0, 2)}
        os.makedirs(os.path.dirname(job["out"]), exist_ok=True)
        tmp = job["out"] + f".tmp{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(rec, f, default=str)
        os.replace(tmp, job["out"])
        return "ok", job["out"], ""
    except Exception:
        err = traceback.format_exc()
        ep = os.path.join(OUT, "errors", os.path.relpath(job["out"], OUT) + ".txt")
        os.makedirs(os.path.dirname(ep), exist_ok=True)
        with open(ep, "w") as f:
            f.write(err)
        return "err", job["out"], err.strip().splitlines()[-1]


def run_stage(name, jobs, pool):
    todo = [j for j in jobs if not os.path.exists(j["out"])]
    log(f"[{name}] {len(jobs)} jobs: {len(jobs) - len(todo)} already done, {len(todo)} to run")
    if not todo:
        return 0
    started, done, errs, last = time.time(), 0, 0, time.time()
    status(name, 0, len(todo), 0, started)
    for st, out, msg in pool.imap_unordered(run_one, todo):
        done += 1
        if st == "err":
            errs += 1
            log(f"[{name}] ERROR {os.path.relpath(out, OUT)}: {msg}")
        status(name, done, len(todo), errs, started)
        if time.time() - last > 600 or done == len(todo):
            last = time.time()
            s = json.load(open(os.path.join(OUT, "status.json")))
            log(f"[{name}] {done}/{len(todo)} ({s['percent']}%)  ETA {s['eta_min']} min  errors {errs}")
    log(f"[{name}] finished: {done - errs} ok, {errs} errors, {(time.time() - started) / 60:.1f} min")
    return errs



# ============================================================================ night 2 (30 Oct 2026)
# E1  Target-label-free model selection. The configuration of every model is the one selected on the SOURCE graph
#     (tuning on the source's own labels), pretraining selects its learning rate on source validation, zero-shot uses
#     no target label at all, and the target runs (scratch, fine-tuned, frozen) use a validation set of only k nodes
#     per class (instead of 30) for learning-rate choice and early stopping. 6 aligned pairs x 6 models x k = 1, 5 x
#     10 splits.
# E2  CKA under L2-SP: the L2-SP runs of the overnight study (same alpha and learning rate) are repeated and the CKA
#     between the body's embeddings before and after fine-tuning is recorded.
N1 = os.environ.get("SGX_N1_RESULTS", os.path.join(PROJECT, "results_posthoc_overnight"))


def _init2():
    _init()
    from sgx import jobs as J, splits
    J._target_masks = lambda g, job: splits.budget_split(g.y_np, job["k"], seed=job["split"], multilabel=g.multilabel,
                                                          val_per_class=job.get("vpc", splits.VAL_PER_CLASS))


def ex_l2cka(job):
    import torch
    from sgx import data, models, train, jobs as J
    g = data.load(job["target"])
    masks = J._target_masks(g, job)
    J._seed(1000 + job["split"])
    net = models.build(job["model"], job["cfg"], g.in_dim, g.num_classes)
    train.transfer(net, train.load_state(job["ckpt"]), ["encoder", "body", "out"])
    net.eval()
    with torch.no_grad():
        z0 = net.embed(g).numpy()
    theta0 = {n: p.detach().clone() for n, p in net.named_parameters()}
    # same procedure as ex_l2sp, at the alpha selected there, keeping the best-validation state
    import math
    from sgx import train as T
    tr, va, te = masks["train"], masks["val"], masks["test"]
    ml, key = g.multilabel, T.primary_name(g.multilabel)
    named = [(n, p) for n, p in net.named_parameters() if p.requires_grad]
    opt = torch.optim.Adam([p for _, p in named], lr=job["lr"], weight_decay=0.0)
    best, state = {"val": -1.0, "val_loss": math.inf, "epoch": -1}, None
    for ep in range(1000):
        net.train(); opt.zero_grad(set_to_none=True)
        out = net(g)
        loss = T.loss_fn(out[tr], g.y[tr], ml) + job["alpha"] * sum(((p - theta0[n]) ** 2).sum() for n, p in named)
        loss.backward(); opt.step(); net.eval()
        with torch.no_grad():
            out = net(g); vl = float(T.loss_fn(out[va], g.y[va], ml)); vm = T.metrics(out[va], g.y[va], ml)
            if vm[key] > best["val"] + 1e-9 or (abs(vm[key] - best["val"]) <= 1e-9 and vl < best["val_loss"]):
                best = {"val": vm[key], "val_loss": vl, "epoch": ep, "test": T.metrics(out[te], g.y[te], ml)[key]}
                state = {a: b.detach().clone() for a, b in net.state_dict().items()}
        if ep - best["epoch"] >= 100:
            break
    net.load_state_dict(state); net.eval()
    with torch.no_grad():
        z1 = net.embed(g).numpy()
    return {"cka": _cka(z0, z1), "test": best["test"], "l2sp_test": job["l2sp_test"], "alpha": job["alpha"], "epoch": best["epoch"]}


EX["l2cka"] = ex_l2cka


def main():
    workers = int(os.environ.get("WORKERS", max(1, (os.cpu_count() or 2) - 2)))
    os.makedirs(OUT, exist_ok=True)
    log(f"===== night-2 post hoc run | workers {workers} | main results {MAIN} | output {OUT} | test {TEST}")
    sel = json.load(open(os.path.join(MAIN, "selected_budget20.json")))
    pairs = [("acmv9", "dblpv7")] if TEST else PAIRS
    ms = ["gcn", "sgx"] if TEST else MODELS
    ks = [5] if TEST else [1, 5]
    errors = 0
    with mp.get_context("spawn").Pool(workers, initializer=_init2) as pool:
        # E1 pretraining with the source-selected configuration
        pre, tg = [], []
        for src, tgt in pairs:
            for m in ms:
                cfg = sel[src][m]
                for s in SPLITS:
                    ck = os.path.join(OUT, "e1", "ckpt", f"{src}__{tgt}", m, f"real_s{s}.pt")
                    pre.append(dict(type="pretrain", source=src, target=tgt, model=m, cfg=cfg, seed=s, ptype="real",
                                    parts=["encoder", "body", "out"], ckpt=ck,
                                    out=os.path.join(OUT, "e1", "pretrain", f"{src}__{tgt}", m, f"real_s{s}.json")))
                    for k in ks:
                        for c in ["scratch", "finetune", "frozen", "zeroshot"]:
                            j = dict(type="zeroshot" if c == "zeroshot" else "target", source=src, target=tgt, model=m,
                                     cfg=cfg, cond=c, k=k, split=s, aligned=True, vpc=k,
                                     out=os.path.join(OUT, "e1", "target", tgt, m, f"{c}@{src}", f"k{k}_s{s}.json"))
                            if c != "scratch":
                                j["ckpt"] = ck
                            tg.append(j)
        errors += run_stage("E1a pretrain (source-selected configuration)", pre, pool)
        errors += run_stage("E1b target (k validation nodes per class; zero-shot without target labels)", tg, pool)
        # E2 CKA under L2-SP
        e2 = []
        for p in glob_l2sp():
            d = json.load(open(p))
            if TEST and not (d["model"] in ms and d["k"] in ks and d["split"] in SPLITS and (d["source"], d["target"]) in pairs):
                continue
            r = d["result"]
            e2.append(dict(type="l2cka", source=d["source"], target=d["target"], model=d["model"], cfg=d["cfg"], k=d["k"],
                           split=d["split"], lr=r["lr"], alpha=r["sel"]["alpha"], l2sp_test=r["sel"]["test"], aligned=True,
                           ckpt=os.path.join(MAIN, "ckpt", f"{d['source']}__{d['target']}", d["model"], f"real_s{d['split']}.pt"),
                           out=os.path.join(OUT, "e2_l2sp_cka", d["target"], d["model"], f"l2sp@{d['source']}", f"k{d['k']}_s{d['split']}.json")))
        errors += run_stage("E2 CKA under L2-SP", e2, pool)
    log(f"===== night-2 run done ({errors} job errors)")


def glob_l2sp():
    import glob
    return sorted(glob.glob(os.path.join(N1, "l2sp", "**", "*.json"), recursive=True))


if __name__ == "__main__":
    main()
