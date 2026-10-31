"""Wall-clock estimate for a plan, from the run times measured so far (the 'seconds' field of every result file).

    python -m sgx.cli size --plan full --workers 10

Jobs whose result file already exists are not counted. For a job type that has not been measured yet, the estimate
falls back to the closest measured jobs: same dataset and model, then same dataset, then the median of the type
(and pretraining controls borrow the time of real pretraining on the same source graph).
"""
from __future__ import annotations

import glob
import json
import os
import statistics
from collections import defaultdict

from . import configs, jobs, paths

ABL_BASE = {m: "sgx" for m in jobs.ABLATION_MODELS}
# models without measurements of their own: (base model, cost factor). A gated SG-XNet-G layer computes a GCN and a
# GraphSAGE-mean step; on the development runs (same configuration) it took about 1.8x as long as SG-XNet.
COST_PROXY = {"sgxg": ("sgx", 1.8)}


def _keys(job: dict) -> list[tuple]:
    """Most specific first."""
    t = job["type"]
    m = ABL_BASE.get(job.get("model"), job.get("model"))
    if t == "tune":
        return [(t, job["proto"], job["dataset"], m), (t, None, job["dataset"], m), (t, None, job["dataset"], None)]
    if t == "pretrain":
        return [(t, job["source"], m, job.get("ptype")), (t, job["source"], m, None), (t, job["source"], None, None)]
    if t in ("target", "zeroshot"):
        c = job["cond"]
        fam = "frozen" if c == "frozen" or c.endswith("_probe") else ("zeroshot" if t == "zeroshot" else "train")
        return [(t, job["target"], m, fam), (t, job["target"], m, None), (t, job["target"], None, None)]
    if t == "public":
        return [(t, job["dataset"], m), (t, job["dataset"], None)]
    if t == "sslpre":
        return [(t, job["graph"], m, job["method"]), (t, None, m, job["method"]), (t, None, None, job["method"])]
    if t == "da":
        return [(t, job["target"], m, job["method"]), (t, None, m, job["method"]), (t, None, None, job["method"])]
    return [(t,)]


def measured() -> dict[tuple, list[float]]:
    times = defaultdict(list)
    for p in glob.glob(os.path.join(paths.RESULTS, "**", "*.json"), recursive=True):
        if os.sep + "analysis" + os.sep in p or os.sep + "errors" + os.sep in p:
            continue
        try:
            with open(p) as f:
                r = json.load(f)
        except (OSError, ValueError):
            continue
        if not isinstance(r, dict) or "seconds" not in r or "type" not in r:
            continue
        for k in _keys(r):
            times[k].append(float(r["seconds"]))
        times[(r["type"],)].append(float(r["seconds"]))
    return times


def _estimate(job, med) -> tuple[float, str]:
    proxy = COST_PROXY.get(job.get("model"))
    if proxy and _keys(job)[0] not in med:  # not measured yet: the base model's time on the same job, scaled
        base, factor = proxy
        est, _ = _estimate(dict(job, model=base), med)
        if est == est:
            return est * factor, f"{base} x {factor}"
    for i, k in enumerate(_keys(job)):
        if k in med:
            return med[k], ["exact", "close", "same dataset"][min(i, 2)]
    if job["type"] == "pretrain":  # controls: same cost as real pretraining on that source
        k = ("pretrain", job["source"], None, None)
        if k in med:
            return med[k], "close"
    k = (job["type"],)
    return (med[k], "type median") if k in med else (float("nan"), "unknown")


STAGE_JOBS = {
    "tune": lambda plan, sel: jobs.tune_jobs(plan),
    "pretrain": lambda plan, sel: jobs.pretrain_jobs(plan, sel["budget20"]),
    "target": lambda plan, sel: jobs.target_jobs(plan, sel["budget20"]),
    "public": lambda plan, sel: jobs.public_jobs(plan, sel["public"]) if plan["public_splits"] else [],
    "sslpre": lambda plan, sel: jobs.sslpre_jobs(plan, sel["budget20"]),
    "ssltarget": lambda plan, sel: jobs.ssl_target_jobs(plan, sel["budget20"]),
    "da": lambda plan, sel: jobs.da_jobs(plan, sel["budget20"]),
}


def estimate(plan_name: str, workers: int) -> str:
    plan = jobs.PLANS[plan_name]
    times = measured()
    med = {k: statistics.median(v) for k, v in times.items()}
    sel = {}
    for proto in ("budget20", "public"):
        try:
            sel[proto] = configs.load_selected(proto)
        except (OSError, ValueError):
            sel[proto] = None
    lines = [f"Estimate for plan '{plan_name}' on {workers} workers (from {sum(len(v) for k, v in times.items() if len(k) == 1)} "
             "measured jobs)", "", "| stage | jobs to run | done | CPU hours | wall-clock hours | basis |",
             "|---|---|---|---|---|---|"]
    total = 0.0
    for st, make in STAGE_JOBS.items():
        if st != "tune" and sel.get("budget20") is None:
            lines.append(f"| {st} | ? | ? | ? | ? | needs the tuning results first |")
            continue
        try:
            js = make(plan, sel)
        except (KeyError, TypeError):
            lines.append(f"| {st} | ? | ? | ? | ? | selected configurations missing for some datasets |")
            continue
        todo = [j for j in js if not os.path.exists(j["out"])]
        secs, basis = 0.0, defaultdict(int)
        for j in todo:
            s, b = _estimate(j, med)
            basis[b] += 1
            secs += 0 if s != s else s
        total += secs
        bs = ", ".join(f"{v} {k}" for k, v in sorted(basis.items(), key=lambda x: -x[1])) or "-"
        lines.append(f"| {st} | {len(todo)} | {len(js) - len(todo)} | {secs / 3600:.1f} | {secs / 3600 / workers:.1f} | {bs} |")
    lines += ["", f"Total: {total / 3600:.1f} CPU hours = about {total / 3600 / workers:.1f} hours on {workers} workers "
              "(plus a small tail at the end of each stage)."]
    return "\n".join(lines)
