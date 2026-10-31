"""Command line.

    python -m sgx.cli run --plan pilot          run (or resume) every stage of a plan
    python -m sgx.cli status                    progress of the running stage
    python -m sgx.cli analyze --plan pilot      results/analysis/SUMMARY_pilot.md and CSV statistics
    python -m sgx.cli figures --plan pilot      vector figures in results/figures/
    python -m sgx.cli size --plan full          wall-clock estimate from the job times measured so far
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

from . import configs, data, jobs, paths, runner

STAGES = ["prepare", "tune", "select", "pretrain", "target", "public", "sslpre", "ssltarget", "da", "analyze"]


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 2)


def _load_rows(pattern: str) -> list[dict]:
    rows = []
    for p in glob.glob(pattern, recursive=True):
        with open(p) as f:
            rows.append(json.load(f))
    return rows


def stage_prepare(plan):
    names = set()
    for dsets in plan["tune"].values():
        names |= set(dsets)
    for s, t in plan["pairs"]:
        names |= {s, t}
    rows = []
    for n in sorted(names, key=list(data.DATASETS).index):
        t0 = time.time()
        st = data.prepare(n)
        rows.append(st)
        runner.log(f"[prepare] {n}: {st['nodes']} nodes, {st['edges']} edges, {st['features']} features, "
                   f"{st['classes']} classes, h_edge={st['h_edge']:.3f} ({time.time() - t0:.1f}s)")
    paths.ensure(paths.RESULTS)
    with open(os.path.join(paths.RESULTS, "datasets.json"), "w") as f:
        json.dump(rows, f, indent=1)


def stage_select(plan):
    for proto in plan["tune"]:
        rows = _load_rows(os.path.join(paths.RESULTS, "tune", proto, "**", "*.json"))
        if plan.get("select_merge"):  # exploratory plan: add its own models, never change the main selection
            rows = [r for r in rows if r["model"] in plan["models"]]
            new = configs.select(proto, rows, write=False)
            configs.merge_selected(proto, new)
            n = sum(len(v) for v in new.values())
            runner.log(f"[select] {proto}: added {n} (dataset, model) configurations for {', '.join(plan['models'])} "
                       f"from {len(rows)} tuning runs; existing entries unchanged")
            continue
        sel = configs.select(proto, rows)
        n = sum(len(v) for v in sel.values())
        runner.log(f"[select] {proto}: selected {n} (dataset, model) configurations from {len(rows)} tuning runs")


def run_plan(plan_name: str, workers: int, stages: list[str]):
    plan = jobs.PLANS[plan_name]
    paths.ensure(paths.RESULTS, paths.LOGS)
    runner.log(f"===== plan '{plan_name}' | stages {','.join(stages)} | workers {workers} | results {paths.RESULTS}")
    errors = 0
    for st in stages:
        if st == "prepare":
            stage_prepare(plan)
        elif st == "tune":
            errors += runner.run("tune", jobs.tune_jobs(plan), workers)
        elif st == "select":
            stage_select(plan)
        elif st == "pretrain":
            errors += runner.run("pretrain", jobs.pretrain_jobs(plan, configs.load_selected("budget20")), workers)
        elif st == "target":
            errors += runner.run("target", jobs.target_jobs(plan, configs.load_selected("budget20")), workers)
        elif st == "public":
            if plan["public_splits"]:
                errors += runner.run("public", jobs.public_jobs(plan, configs.load_selected("public")), workers)
        elif st == "sslpre":
            errors += runner.run("sslpre", jobs.sslpre_jobs(plan, configs.load_selected("budget20")), workers)
        elif st == "ssltarget":
            errors += runner.run("ssltarget", jobs.ssl_target_jobs(plan, configs.load_selected("budget20")), workers)
        elif st == "da":
            errors += runner.run("da", jobs.da_jobs(plan, configs.load_selected("budget20")), workers)
        elif st == "analyze":
            from . import analysis

            out = analysis.summarise(plan_name)
            runner.log(f"[analyze] summary written to {os.path.relpath(out, paths.ROOT)}")
    runner.log(f"===== plan '{plan_name}' done ({errors} job errors)")
    return errors


def status():
    p = os.path.join(paths.RESULTS, "status.json")
    if os.path.exists(p):
        print(json.dumps(json.load(open(p)), indent=1))
    for st in ["tune", "pretrain", "target", "public", "sslpre"]:  # ssltarget and da results live under target/
        n = len(glob.glob(os.path.join(paths.RESULTS, st, "**", "*.json"), recursive=True))
        print(f"{st:9s} results: {n}")
    n_err = len(glob.glob(os.path.join(paths.RESULTS, "errors", "**", "*.txt"), recursive=True))
    print(f"errors   : {n_err}")
    lp = os.path.join(paths.LOGS, "run.log")
    if os.path.exists(lp):
        print("--- last log lines ---")
        print("".join(open(lp).readlines()[-8:]))


def main():
    ap = argparse.ArgumentParser(prog="sgx")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--plan", default="pilot", choices=list(jobs.PLANS))
    r.add_argument("--workers", type=int, default=default_workers())
    r.add_argument("--stages", default=",".join(STAGES))
    sub.add_parser("status")
    a = sub.add_parser("analyze")
    a.add_argument("--plan", default="pilot", choices=list(jobs.PLANS))
    sub.add_parser("prepare")
    fg = sub.add_parser("figures")
    fg.add_argument("--plan", default="pilot", choices=list(jobs.PLANS))
    sz = sub.add_parser("size")
    sz.add_argument("--plan", default="full", choices=list(jobs.PLANS))
    sz.add_argument("--workers", type=int, default=default_workers())
    args = ap.parse_args()
    if args.cmd == "run":
        raise SystemExit(1 if run_plan(args.plan, args.workers, args.stages.split(",")) else 0)
    if args.cmd == "status":
        status()
    if args.cmd == "analyze":
        from . import analysis

        print(analysis.summarise(args.plan))
    if args.cmd == "prepare":
        stage_prepare(jobs.PLANS["full"])
    if args.cmd == "figures":
        from . import figures

        for path in figures.make_all(args.plan):
            print(os.path.relpath(path, paths.ROOT))
    if args.cmd == "size":
        from . import sizing

        print(sizing.estimate(args.plan, args.workers))


if __name__ == "__main__":
    main()
