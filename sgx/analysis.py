"""Aggregation and statistics.

All comparisons are paired by split (same labelled nodes, same test nodes, same initialisation of the
dataset-specific encoder/head). We report mean +- standard deviation over splits and the mean paired difference.

Significance (the stars): two-sided paired t-test over splits, Holm-adjusted within a family of related tests -
one family per (condition, model) in the transfer-gain table and one per baseline in the SG-XNet-vs-baseline table.
The Wilcoxon signed-rank p-value is reported alongside in the CSV files as a non-parametric check. It cannot be the
primary test here: with n splits whose differences all have the same sign its smallest two-sided p-value is
2 / 2**n (0.0625 for 5 splits, 0.002 for 10), so after a Holm correction over more than about 25 tests no cell could
ever reach 0.05, however large and consistent the effect.
"""
from __future__ import annotations

import glob
import json
import math
import os

import numpy as np
import pandas as pd
from scipy import stats

from . import jobs, paths

NAMES = {"gcn": "GCN", "sage": "GraphSAGE", "gat": "GAT", "gin": "GIN", "mlp": "MLP", "sgx": "SG-XNet",
         "sgx_order": "SG-XNet (GCN-SAGE order)", "sgx_last": "SG-XNet (SAGE last)", "sgx_sage2": "SG-XNet (2 SAGE)",
         "sgx_nodrop": "SG-XNet (no dropout)", "abl_allgcn": "all-GCN (SG-XNet cfg)", "abl_allsage": "all-SAGE (SG-XNet cfg)",
         "sgxg": "SG-XNet-G"}


TRANSFER_CONDS = ["finetune", "frozen", "zeroshot", "ctrl_shuffle", "ctrl_random",
                  "ssl_dgi", "ssl_dgi_probe", "ssl_grace", "ssl_grace_probe",
                  "ssl_dgi_self", "ssl_dgi_self_probe", "ssl_grace_self", "ssl_grace_self_probe", "dann", "udagcn"]


def load_target() -> pd.DataFrame:
    rows = []
    for p in glob.glob(os.path.join(paths.RESULTS, "target", "**", "*.json"), recursive=True):
        with open(p) as f:
            r = json.load(f)
        s = r["result"]["sel"]
        row = dict(target=r["target"], source=r.get("source"), model=r["model"], cond=r["cond"], k=r["k"],
                   split=r["split"], test=100 * s["test"], val=100 * s["val"], metric=r["result"]["metric"],
                   lr=s.get("lr"), epoch=s.get("epoch"), macro_f1=100 * s.get("test_macro_f1", float("nan")))
        if s.get("gates") is not None:  # SG-XNet-G only: GCN weight of every layer at the selected epoch
            row["gates"] = s["gates"]
        rows.append(row)
    return pd.DataFrame(rows)


def load_public() -> pd.DataFrame:
    rows = []
    for p in glob.glob(os.path.join(paths.RESULTS, "public", "**", "*.json"), recursive=True):
        with open(p) as f:
            r = json.load(f)
        s = r["result"]["sel"]
        row = dict(dataset=r["dataset"], model=r["model"], split=r["split"], test=100 * s["test"])
        if s.get("gates") is not None:
            row["gates"] = s["gates"]
        rows.append(row)
    return pd.DataFrame(rows)


def holm(p: list[float]) -> list[float]:
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    idx = np.argsort(p[ok])
    vals = p[ok][idx]
    m = len(vals)
    adj = np.maximum.accumulate([(m - i) * v for i, v in enumerate(vals)]) if m else np.array([])
    tmp = np.empty(m)
    tmp[idx] = np.minimum(adj, 1.0)
    out[ok] = tmp
    return out.tolist()


def paired(a: pd.Series, b: pd.Series) -> dict:
    """a, b indexed by split. Returns mean difference a-b, its std, t-test and Wilcoxon p, wins."""
    common = a.index.intersection(b.index)
    d = (a.loc[common] - b.loc[common]).astype(float)
    n = len(d)
    res = dict(n=n, diff=float(d.mean()) if n else float("nan"), sd=float(d.std(ddof=1)) if n > 1 else float("nan"),
               wins=int((d > 0).sum()), losses=int((d < 0).sum()), p_t=float("nan"), p_w=float("nan"))
    if n > 1 and d.std(ddof=1) > 0:
        res["p_t"] = float(stats.ttest_rel(a.loc[common], b.loc[common]).pvalue)
    if n > 0 and (d != 0).any():
        try:
            res["p_w"] = float(stats.wilcoxon(d).pvalue)
        except ValueError:
            pass
    return res


def _holm_families(st: pd.DataFrame) -> pd.DataFrame:
    """Adds p_holm: Holm adjustment of the paired t-test p-values within each value of the 'family' column."""
    if st.empty:
        return st.assign(p_holm=[])
    st = st.copy()
    st["p_holm"] = np.nan
    for _, idx in st.groupby("family").groups.items():
        st.loc[idx, "p_holm"] = holm(st.loc[idx, "p_t"].tolist())
    return st


def fmt(mean, sd, digits=1):
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "-"
    return f"{mean:.{digits}f} ± {sd:.{digits}f}" if not math.isnan(sd) else f"{mean:.{digits}f}"


def md_table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _series(df, **kw):
    q = df
    for k, v in kw.items():
        q = q[q[k].isna()] if v is None else q[q[k] == v]
    return q.set_index("split")["test"]


def _gate_section(df: pd.DataFrame, pub: pd.DataFrame, ds: list[dict]) -> list[str]:
    """SG-XNet-G only: weight of the GCN branch of every layer (0 = GraphSAGE-mean step, 1 = GCN step, 0.5 at
    initialisation), at the selected epoch, mean over splits; and its rank correlation with edge homophily."""
    have_t = "gates" in df.columns and df["gates"].notna().any()
    have_p = "gates" in pub.columns and pub["gates"].notna().any()
    if not (have_t or have_p):
        return []
    hom = {d["name"]: d["h_edge"] for d in ds}

    def layer_means(series) -> list[float]:
        arrs = [list(x) for x in series if isinstance(x, list)]
        width = min(len(a) for a in arrs)
        return list(np.mean([a[:width] for a in arrs], axis=0))

    out = ["## 6. Learned gates of SG-XNet-G (weight of the GCN branch per layer; 0 = GraphSAGE-mean step, "
           "1 = GCN step, 0.5 at initialisation; mean over splits at the selected epoch)", ""]
    rows, xs, ys = [], [], []
    if have_p:
        q = pub[pub["gates"].notna()]
        for d in dict.fromkeys(q.dataset):
            g = layer_means(q[q.dataset == d].gates)
            rows.append([d, "public splits", "-", f"{hom.get(d, float('nan')):.3f}", ", ".join(f"{v:.2f}" for v in g),
                         f"{np.mean(g):.2f}", int((q.dataset == d).sum())])
            if d in hom:
                xs.append(hom[d])
                ys.append(float(np.mean(g)))
    if have_t:
        q = df[df["gates"].notna() & (df.cond == "scratch")]
        if not q.empty:
            kmax = int(q.k.max())
            for t in dict.fromkeys(q.target):
                qq = q[(q.target == t) & (q.k == kmax)]
                if qq.empty:
                    continue
                g = layer_means(qq.gates)
                rows.append([t, "scratch", kmax, f"{hom.get(t, float('nan')):.3f}", ", ".join(f"{v:.2f}" for v in g),
                             f"{np.mean(g):.2f}", len(qq)])
    out += [md_table(["dataset", "runs", "k", "edge homophily", "GCN weight by layer (first ... output)", "mean", "n"],
                     rows), ""]
    if len(xs) >= 4:
        rho, p = stats.spearmanr(xs, ys)
        out += [f"Spearman correlation between edge homophily and the mean GCN weight over the public-split datasets: "
                f"rho = {rho:.2f} (p = {p:.3f}, {len(xs)} datasets).", ""]
    return out


def summarise(plan_name: str) -> str:
    """Plan keys used here (defaults = the pre-specified main study): show_models (models in the tables; default
    plan['models']), gain_models (models in the transfer-gain table; default all shown models), focus (the model
    compared with every other model in section 3; default 'sgx'), vs_conds (conditions of section 3; default scratch
    and finetune), show_ablation (default True), note (printed under the title)."""
    plan = jobs.PLANS[plan_name]
    focus = plan.get("focus", "sgx")
    out_dir = os.path.join(paths.RESULTS, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    df = load_target()
    lines = [f"# SG-XNet v2 - {plan_name} summary", "",
             f"Target-run results: {len(df)}. Metric: accuracy (micro-F1 for acmv9/citationv1/dblpv7), test set, "
             "epoch chosen on validation, mean ± std over splits.", ""]
    if plan.get("note"):
        lines += [f"> {plan['note']}", ""]
    if df.empty:
        path = os.path.join(out_dir, f"SUMMARY_{plan_name}.md")
        open(path, "w").write("\n".join(lines + ["No target results yet."]))
        return path
    df.to_csv(os.path.join(out_dir, f"target_runs_{plan_name}.csv"), index=False)
    ds_path = os.path.join(paths.RESULTS, "datasets.json")
    ds = []
    if os.path.exists(ds_path):
        ds = json.load(open(ds_path))
        lines += ["## 0. Datasets", "", md_table(
            ["dataset", "group", "nodes", "edges", "features", "classes", "edge homophily", "adjusted homophily"],
            [[d["name"], d["group"], d["nodes"], d["edges"], d["features"], d["classes"], f"{d['h_edge']:.3f}",
              "-" if d["h_adj"] != d["h_adj"] else f"{d['h_adj']:.3f}"] for d in ds]), ""]
    main_models = [m for m in plan.get("show_models", plan["models"]) if m in df.model.unique()]
    gain_models = [m for m in plan.get("gain_models", main_models) if m in df.model.unique()]
    targets = [t for t in dict.fromkeys(t for _, t in plan["pairs"]) if t in df.target.unique()]

    # ---- 1. from-scratch comparison
    lines += ["## 1. From scratch (no transfer)", ""]
    for k in sorted(df.k.unique()):
        rows = []
        for t in targets:
            sc = df[(df.target == t) & (df.cond == "scratch") & (df.k == k)]
            means = {m: sc[sc.model == m].test.mean() for m in main_models}
            best = max(means.values()) if means else float("nan")
            r = [t]
            for m in main_models:
                v = sc[sc.model == m].test
                cell = fmt(v.mean(), v.std(ddof=1)) if len(v) else "-"
                r.append(f"**{cell}**" if len(v) and abs(v.mean() - best) < 1e-9 else cell)
            rows.append(r)
        lines += [f"Label budget k = {k} per class", "", md_table(["target"] + [NAMES[m] for m in main_models], rows), ""]

    # ---- 2. transfer gains
    lines += ["## 2. Transfer gain = condition - scratch (percentage points, paired by split)", ""]
    trows, stat_rows = [], []
    for s, t in plan["pairs"]:
        for m in gain_models:
            for k in sorted(df.k.unique()):
                base = _series(df, target=t, model=m, cond="scratch", k=k)
                if base.empty:
                    continue
                row = dict(pair=f"{s}→{t}", model=NAMES[m], k=k, scratch=fmt(base.mean(), base.std(ddof=1)), cells={})
                for c in TRANSFER_CONDS:
                    cs = _series(df, target=t, model=m, cond=c, k=k, source=None if "_self" in c else s)
                    if cs.empty:
                        continue
                    pr = paired(cs, base)
                    row["cells"][c] = len(stat_rows)
                    stat_rows.append(dict(source=s, target=t, model=m, k=k, cond=c, gain=pr["diff"], sd=pr["sd"],
                                          n=pr["n"], wins=pr["wins"], losses=pr["losses"], p_t=pr["p_t"],
                                          p_w=pr["p_w"], family=f"{c}|{m}"))
                trows.append(row)
    stats_df = _holm_families(pd.DataFrame(stat_rows))
    conds = [c for c in TRANSFER_CONDS if any(c in r["cells"] for r in trows)]
    rows = []
    for r in trows:
        cells = [r["pair"], r["model"], r["k"], r["scratch"]]
        for c in conds:
            if c not in r["cells"]:
                cells.append("-")
                continue
            q = stats_df.iloc[r["cells"][c]]
            cells.append(f"{q.gain:+.1f}{'*' if q.p_holm < 0.05 else ''} ({q.wins}/{q.n})")
        rows.append(cells)
    if not stats_df.empty:
        stats_df.to_csv(os.path.join(out_dir, f"transfer_gains_{plan_name}.csv"), index=False)
    lines += ["Cell: mean difference (* = paired t-test, Holm-adjusted within each condition and model, p < 0.05) "
              "(splits where the condition beat scratch / splits). Full statistics, including Wilcoxon p-values: "
              f"transfer_gains_{plan_name}.csv.", "",
              md_table(["pair", "model", "k", "scratch"] + conds, rows), ""]

    # ---- 3. SG-XNet (or the plan's focus model) vs each other model
    what = "SG-XNet minus baseline" if focus == "sgx" else f"{NAMES[focus]} minus each model"
    lines += [f"## 3. {what} (percentage points, paired by split)", ""]
    rows, stat_rows = [], []
    cmp_models = [m for m in main_models if m != focus]
    for t in targets:
        for k in sorted(df.k.unique()):
            for c in plan.get("vs_conds", ["scratch", "finetune"]):
                srcs = [None] if c == "scratch" else [s for s, tt in plan["pairs"] if tt == t]
                for s in srcs:
                    a = _series(df, target=t, model=focus, cond=c, k=k, source=s)
                    if a.empty:
                        continue
                    cells = [t, k, c if s is None else f"{c}@{s}", fmt(a.mean(), a.std(ddof=1))]
                    for m in cmp_models:
                        b = _series(df, target=t, model=m, cond=c, k=k, source=s)
                        if b.empty:
                            cells.append("-")
                            continue
                        pr = paired(a, b)
                        cells.append(len(stat_rows))
                        stat_rows.append(dict(target=t, k=k, cond=c, source=s, baseline=m, sgx_minus_baseline=pr["diff"],
                                              sd=pr["sd"], n=pr["n"], wins=pr["wins"], losses=pr["losses"],
                                              p_t=pr["p_t"], p_w=pr["p_w"], family=m,
                                              **({} if focus == "sgx" else {"model": focus})))
                    rows.append(cells)
    stats_df = _holm_families(pd.DataFrame(stat_rows))
    out_rows = []
    for cells in rows:
        new = cells[:4]
        for c in cells[4:]:
            if isinstance(c, int):
                q = stats_df.iloc[c]
                new.append(f"{q.sgx_minus_baseline:+.1f}{'*' if q.p_holm < 0.05 else ''}")
            else:
                new.append(c)
        out_rows.append(new)
    vs_csv = f"sgx_vs_baselines_{plan_name}.csv" if focus == "sgx" else f"{focus}_vs_models_{plan_name}.csv"
    if not stats_df.empty:
        stats_df.to_csv(os.path.join(out_dir, vs_csv), index=False)
    each = "baseline" if focus == "sgx" else "compared model"
    lines += [f"Cell: {what} (* = paired t-test, Holm-adjusted within each {each}, p < 0.05). "
              f"Full statistics: {vs_csv}.", "",
              md_table(["target", "k", "condition", NAMES[focus]] + [NAMES[m] for m in cmp_models], out_rows), ""]

    # ---- 4. public splits
    pub = load_public()
    if not pub.empty:
        lines += ["## 4. Public splits (Planetoid public split x 10 seeds; Platonov et al. 10 splits)", ""]
        rows = []
        for d in dict.fromkeys(pub.dataset):
            q = pub[pub.dataset == d]
            means = {m: q[q.model == m].test.mean() for m in main_models}
            best = max(v for v in means.values() if not math.isnan(v))
            r = [d]
            for m in main_models:
                v = q[q.model == m].test
                cell = fmt(v.mean(), v.std(ddof=1)) if len(v) else "-"
                r.append(f"**{cell}**" if len(v) and abs(v.mean() - best) < 1e-9 else cell)
            rows.append(r)
        lines += [md_table(["dataset"] + [NAMES[m] for m in main_models], rows), ""]

    # ---- 5. ablation (if present)
    abl = [m for m in jobs.ABLATION_MODELS if m in df.model.unique() and m != "sgx"]
    if abl and plan.get("show_ablation", True):
        lines += ["## 5. Ablation (SG-XNet configuration; difference to SG-XNet in percentage points)", ""]
        rows = []
        for s, t in plan["ablation_pairs"]:
            for k in (5, 20):
                for c in ("scratch", "finetune"):
                    src = None if c == "scratch" else s
                    a = _series(df, target=t, model="sgx", cond=c, k=k, source=src)
                    if a.empty:
                        continue
                    cells = [f"{s}→{t}", k, c, fmt(a.mean(), a.std(ddof=1))]
                    for m in abl:
                        b = _series(df, target=t, model=m, cond=c, k=k, source=src)
                        cells.append("-" if b.empty else f"{b.mean() - a.loc[a.index.intersection(b.index)].mean():+.1f}")
                    rows.append(cells)
        lines += [md_table(["pair", "k", "cond", "SG-XNet"] + [NAMES[m] for m in abl], rows), ""]

    # ---- 6. learned gates (SG-XNet-G only)
    lines += _gate_section(df, pub, ds)

    path = os.path.join(out_dir, f"SUMMARY_{plan_name}.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path
