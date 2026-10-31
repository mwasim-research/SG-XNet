"""Post hoc generalisation check on a second aligned benchmark (28 Oct 2026; to be reported as post hoc).

Twitch (Rozemberczki et al., 2021, MUSAE): one graph per country, nodes are streamers, edges mutual follows,
features = games played/liked from one shared 3,170-game vocabulary (binary, L2-normalised), label = whether the
streamer uses explicit language ("mature"). Three countries, DE (9,498 nodes), ENGB (7,126) and FR (6,551), give six
directed aligned pairs (shared features and labels), the same design as the ACDNE benchmark.

Protocol = the main study's (v2c code in ../../sgx, read-only):
  tuning   30 configurations x 3 tuning splits per (target, model), k = 20 per class, select on validation
  pretrain 10 seeds per pair and model; learning rate chosen on source validation
  target   scratch, fine-tune, frozen, zero-shot at k = 1 and 5 labels per class, 10 splits
  l2sp     L2-SP fine-tuning (as in the overnight run): lr = the one selected for plain fine-tuning of that run,
           alpha in {0.001, 0.01, 0.1, 1} chosen on validation
Writes only to ../../results_posthoc_twitch (and the processed graphs to data/cache/twitch_*.npz).
"""
import json
import math
import multiprocessing as mp
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(HERE))
OUT = os.environ.get("SGX_POSTHOC_OUT", os.path.join(PROJECT, "results_posthoc_twitch"))
os.environ["SGX_RESULTS"] = OUT
sys.path.insert(0, PROJECT)
TEST = os.environ.get("TWITCH_TEST") == "1"

COUNTRIES = {"twitch_de": "DE", "twitch_engb": "ENGB", "twitch_fr": "FR"}
MODELS = ["gcn", "sage", "gat", "gin", "mlp", "sgx"]
NAMES = list(COUNTRIES)
PAIRS = [(s, t) for t in NAMES for s in NAMES if s != t]
ALPHAS = [0.001, 0.01, 0.1, 1.0]
VOCAB = 3170


def register():
    import numpy as np
    import scipy.sparse as sp
    from sgx import data, paths

    def _load_twitch(spec):
        import csv
        c = spec["file"]
        d = os.path.join(paths.RAW, "twitch")
        feats = json.load(open(os.path.join(d, f"{c}.json")))
        tgt = list(csv.DictReader(open(os.path.join(d, f"{c}_target.csv"))))
        n = len(tgt)
        y = np.zeros(n, np.int64)
        for r in tgt:
            y[int(r["new_id"])] = 1 if r["mature"] == "True" else 0
        rows, cols = [], []
        for k, v in feats.items():
            rows += [int(k)] * len(v)
            cols += list(v)
        x = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, VOCAB))
        x.sum_duplicates()
        x.data[:] = 1.0
        e = np.loadtxt(os.path.join(d, f"{c}_edges.csv"), delimiter=",", skiprows=1, dtype=np.int64).T
        return x, e, y, None

    data._LOADERS["twitch"] = _load_twitch
    for name, c in COUNTRIES.items():
        data.DATASETS[name] = dict(loader="twitch", file=c, norm="l2", group="aligned")
        data.ALIGNED.add(name)


def log(msg):
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(os.path.join(OUT, "logs", "twitch.log"), "a") as f:
        f.write(line + "\n")


def status(stage, done, total, errors, started):
    el = time.time() - started
    rate = done / el if el > 0 else 0
    eta = (total - done) / rate / 60 if rate > 0 else None
    with open(os.path.join(OUT, "status.json.tmp"), "w") as f:
        json.dump(dict(stage=stage, done=done, total=total, errors=errors, percent=round(100 * done / max(total, 1), 1),
                       eta_min=None if eta is None else round(eta, 1), updated=time.strftime("%Y-%m-%d %H:%M:%S")), f)
    os.replace(os.path.join(OUT, "status.json.tmp"), os.path.join(OUT, "status.json"))


def _init():
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    register()


def _fit_l2sp(model, g, masks, lr, alpha, theta0, max_epochs=1000, patience=100):
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
        loss = train.loss_fn(out[tr], g.y[tr], ml) + alpha * sum(((p - theta0[n]) ** 2).sum() for n, p in named)
        if not torch.isfinite(loss):
            best["diverged"] = True
            break
        loss.backward()
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


def ex_l2sp(job):
    from sgx import data, models, train, jobs as J
    g = data.load(job["target"])
    masks = J._target_masks(g, job)
    st0 = train.load_state(job["ckpt"])
    lr = json.load(open(job["ft_result"]))["result"]["sel"]["lr"]
    runs = []
    for a in ALPHAS:
        J._seed(1000 + job["split"])
        net = models.build(job["model"], job["cfg"], g.in_dim, g.num_classes)
        train.transfer(net, st0, ["encoder", "body", "out"])
        theta0 = {n: p.detach().clone() for n, p in net.named_parameters()}
        r = _fit_l2sp(net, g, masks, lr, a, theta0)
        r["alpha"] = a
        runs.append(r)
    sel = max(runs, key=lambda r: (r["val"], -r["val_loss"]))
    return {"sel": sel, "runs": runs, "lr": lr, "metric": train.primary_name(g.multilabel)}


def run_one(job):
    t0 = time.time()
    try:
        from sgx import jobs as J
        res = ex_l2sp(job) if job["type"] == "l2sp" else J.EXEC[job["type"]](job)
        meta = {k: v for k, v in job.items() if k != "out"}
        flat = job["type"] in ("pretrain", "tune")
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
    todo.sort(key=lambda j: (j.get("target") or j.get("dataset") or "", j.get("source") or "", j.get("model", "")))
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


def main():
    register()
    from sgx import configs, data
    workers = int(os.environ.get("WORKERS", max(1, (os.cpu_count() or 2) - 2)))
    os.makedirs(OUT, exist_ok=True)
    log(f"===== Twitch post hoc run | workers {workers} | output {OUT} | test {TEST}")
    for n in NAMES:
        st = data.prepare(n)
        log(f"[prepare] {n}: {st['nodes']} nodes, {st['edges']} edges, {st['features']} features, "
            f"{st['classes']} classes, h_edge={st['h_edge']:.3f}")
    models_ = ["gcn", "sgx"] if TEST else MODELS
    n_cfg, tsplits = (2, 1) if TEST else (30, 3)
    splits_ = [0] if TEST else list(range(10))
    ks = [5] if TEST else [1, 5]
    pairs = [("twitch_de", "twitch_engb")] if TEST else PAIRS
    errors = 0
    with mp.get_context("spawn").Pool(workers, initializer=_init) as pool:
        # 1. tuning on each target (same budget as the main study)
        tune = [dict(type="tune", proto="budget20", dataset=t, model=m, cfg=c, split=s,
                     out=os.path.join(OUT, "tune", t, m, f"c{c['cid']:02d}_s{s}.json"))
                for t in sorted({t for _, t in pairs}) for m in models_ for c in configs.sample_configs(m, n_cfg)
                for s in range(tsplits)]
        errors += run_stage("1 tune", tune, pool)
        rows = [json.load(open(j["out"])) for j in tune if os.path.exists(j["out"])]
        sel = configs.select("budget20", rows, write=False)
        with open(os.path.join(OUT, "selected_twitch.json"), "w") as f:
            json.dump(sel, f, indent=1)
        log("[select] " + json.dumps({t: {m: c["cid"] for m, c in v.items()} for t, v in sel.items()}))
        # 2. pretraining
        pre = []
        for src, tgt in pairs:
            for m in models_:
                for s in splits_:
                    pre.append(dict(type="pretrain", source=src, target=tgt, model=m, cfg=sel[tgt][m], seed=s, ptype="real",
                                    parts=["encoder", "body", "out"],
                                    ckpt=os.path.join(OUT, "ckpt", f"{src}__{tgt}", m, f"real_s{s}.pt"),
                                    out=os.path.join(OUT, "pretrain", f"{src}__{tgt}", m, f"real_s{s}.json")))
        errors += run_stage("2 pretrain", pre, pool)
        # 3. target runs
        tg, seen = [], set()
        for src, tgt in pairs:
            for m in models_:
                for k in ks:
                    for s in splits_:
                        for c in ["scratch", "finetune", "frozen", "zeroshot"]:
                            sub = "scratch" if c == "scratch" else f"{c}@{src}"
                            out = os.path.join(OUT, "target", tgt, m, sub, f"k{k}_s{s}.json")
                            if out in seen:
                                continue
                            seen.add(out)
                            j = dict(type="zeroshot" if c == "zeroshot" else "target", source=None if c == "scratch" else src,
                                     target=tgt, model=m, cfg=sel[tgt][m], cond=c, k=k, split=s, aligned=True, out=out)
                            if c != "scratch":
                                j["ckpt"] = os.path.join(OUT, "ckpt", f"{src}__{tgt}", m, f"real_s{s}.pt")
                            tg.append(j)
        errors += run_stage("3 target", tg, pool)
        # 4. L2-SP fine-tuning
        l2 = []
        for src, tgt in pairs:
            for m in models_:
                for k in ks:
                    for s in splits_:
                        l2.append(dict(type="l2sp", source=src, target=tgt, model=m, cfg=sel[tgt][m], k=k, split=s, aligned=True,
                                       ckpt=os.path.join(OUT, "ckpt", f"{src}__{tgt}", m, f"real_s{s}.pt"),
                                       ft_result=os.path.join(OUT, "target", tgt, m, f"finetune@{src}", f"k{k}_s{s}.json"),
                                       out=os.path.join(OUT, "target", tgt, m, f"l2sp@{src}", f"k{k}_s{s}.json")))
        errors += run_stage("4 l2sp", l2, pool)
    log(f"===== Twitch run done ({errors} job errors)")


if __name__ == "__main__":
    main()
