"""Parallel, resumable job runner with a progress file (results/status.json) and a log (results/logs/run.log)."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
import traceback

from . import paths


def log(msg: str):
    paths.ensure(paths.LOGS)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(os.path.join(paths.LOGS, "run.log"), "a") as f:
        f.write(line + "\n")


def _atomic_json(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _init_worker():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _run_one(job):
    from . import jobs as J

    t0 = time.time()
    try:
        res = J.run_job(job)
        _atomic_json(job["out"], res)
        return ("ok", job["out"], time.time() - t0, "")
    except Exception:  # record and continue
        err = traceback.format_exc()
        epath = job["out"].replace(paths.RESULTS, os.path.join(paths.RESULTS, "errors"), 1) + ".txt"
        os.makedirs(os.path.dirname(epath), exist_ok=True)
        with open(epath, "w") as f:
            f.write(json.dumps({k: v for k, v in job.items() if k != "cfg"}, default=str) + "\n\n" + err)
        return ("err", job["out"], time.time() - t0, err.strip().splitlines()[-1])


def write_status(stage, done, total, errors, started, extra=None):
    el = time.time() - started
    rate = done / el if el > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else None
    _atomic_json(os.path.join(paths.RESULTS, "status.json"), dict(
        stage=stage, done=done, total=total, errors=errors, percent=round(100 * done / max(total, 1), 1),
        jobs_per_min=round(60 * rate, 1), eta_min=None if eta is None else round(eta / 60, 1),
        started=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
        updated=time.strftime("%Y-%m-%d %H:%M:%S"), **(extra or {})))


def run(stage: str, jobs: list[dict], workers: int) -> int:
    todo = [j for j in jobs if not os.path.exists(j["out"])]
    # group by dataset so each worker re-uses its cached graphs
    todo.sort(key=lambda j: (j.get("dataset") or j.get("target") or "", j.get("source") or "", j.get("model", "")))
    log(f"[{stage}] {len(jobs)} jobs: {len(jobs) - len(todo)} already done, {len(todo)} to run on {workers} workers")
    if not todo:
        return 0
    started, done, errors, last, last_log = time.time(), 0, 0, 0.0, time.time()
    write_status(stage, 0, len(todo), 0, started)
    if workers <= 1:
        _init_worker()
        it = map(_run_one, todo)
        pool = None
    else:
        pool = mp.get_context("spawn").Pool(workers, initializer=_init_worker, maxtasksperchild=300)
        it = pool.imap_unordered(_run_one, todo, chunksize=1)
    try:
        for status, out, secs, msg in it:
            done += 1
            if status == "err":
                errors += 1
                log(f"[{stage}] ERROR {os.path.relpath(out, paths.RESULTS)}: {msg}")
            now = time.time()
            if now - last > 20 or done == len(todo):
                write_status(stage, done, len(todo), errors, started)
                last = now
                if now - last_log > 600 or done == len(todo):  # a progress line in run.log every 10 minutes
                    st = json.load(open(os.path.join(paths.RESULTS, "status.json")))
                    log(f"[{stage}] {done}/{len(todo)} ({st['percent']}%)  {st['jobs_per_min']} jobs/min  "
                        f"ETA {st['eta_min']} min  errors {errors}")
                    last_log = now
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    log(f"[{stage}] finished: {done - errors} ok, {errors} errors, {round((time.time() - started) / 60, 1)} min")
    return errors
