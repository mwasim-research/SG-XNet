"""Publication figures for the revised manuscript (the editor asked for vector figures).

Every figure is written to <results>/figures/<name>.{pdf,eps,svg,png}: PDF and EPS keep the text as embedded
TrueType (editable), SVG stores text as outlines so it renders identically everywhere, and the PNG (300 dpi) is a
preview. Next to each figure, <name>.csv holds exactly the numbers it plots (the table view of the figure).

    python -m sgx.cli figures --plan pilot        # data figures from the finished runs + the two diagrams

Design rules (shared with the tables): one metric per axis; mean over splits with a 95% t-interval; transfer
gains and model differences are paired by split; stars mark p < 0.05 in a two-sided paired t-test, Holm-adjusted
within the family named in each function (see analysis.py for why the Wilcoxon test, reported in the CSV files,
cannot be the primary test with 5-10 splits). Colour never carries meaning alone:
every series also has its own marker and a legend. The three categorical hues (blue, orange, aqua) pass the
all-pairs colour-vision-deficiency and normal-vision separation checks on a white background.
"""
from __future__ import annotations

import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import patches  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from scipy import stats  # noqa: E402

from . import analysis, data, jobs, paths  # noqa: E402

# ----------------------------------------------------------------------------- palette and style
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
BLUE_LIGHT = "#86b6ef"  # blue ramp step 250
INK, INK2, MUTED, GRID, AXIS, MID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#f0efec"
BLUE_WASH = "#cde2fb"  # blue ramp step 100

COND = {  # colour, marker, legend label
    "scratch": (INK2, "o", "From scratch"),
    "finetune": (BLUE, "s", "Fine-tuned (supervised pretraining)"),
    "frozen": (ORANGE, "^", "Frozen pretrained body"),
    "zeroshot": (AQUA, "D", "Zero-shot (no target training)"),
    "ctrl_shuffle": (BLUE_LIGHT, "v", "Control: shuffled source labels"),
    "ctrl_random": (MUTED, "X", "Control: random source graph"),
    "ssl": (ORANGE, "P", "Self-supervised pretraining (best of DGI, GRACE)"),
    "da": (AQUA, "h", "Domain adaptation (best of DANN, UDAGCN)"),
}
DS = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed", "dblp": "DBLP", "acmv9": "ACMv9",
      "citationv1": "Citationv1", "dblpv7": "DBLPv7", "roman_empire": "roman-empire",
      "amazon_ratings": "amazon-ratings", "chameleon_filtered": "chameleon-f", "squirrel_filtered": "squirrel-f"}
CODE = {"cora": "Co", "citeseer": "CS", "pubmed": "PM", "dblp": "DB", "acmv9": "A", "citationv1": "C",
        "dblpv7": "D", "roman_empire": "RE", "amazon_ratings": "AR", "chameleon_filtered": "Ch",
        "squirrel_filtered": "Sq"}  # short codes for point labels (A, C, D as in the ACDNE literature)
MODELS = ["gcn", "sage", "gat", "gin", "mlp", "sgx"]
SHORT = {"gcn": "GCN", "sage": "SAGE", "gat": "GAT", "gin": "GIN", "mlp": "MLP", "sgx": "SG-XNet", "sgxg": "SG-XNet-G",
         "sgx_order": "GCN first, then SAGE", "sgx_last": "SAGE as last layer", "sgx_sage2": "Two SAGE layers",
         "sgx_nodrop": "No dropout", "abl_allgcn": "All GCN layers", "abl_allsage": "All SAGE layers"}
GROUPS = [("Original pairs", jobs.PAIRS_X1, AXIS), ("Shared features and labels", jobs.PAIRS_X2, AXIS),
          ("Heterophilic targets", jobs.PAIRS_X6, AXIS)]
GROUP_STYLE = {"Original pairs": (BLUE, "o"), "Shared features and labels": (ORANGE, "s"),
               "Heterophilic targets": (AQUA, "^")}
FULL_W = 7.0  # inches (page width)


def style():
    from matplotlib import font_manager

    have = {f.name for f in font_manager.fontManager.ttflist}
    sans = next((f for f in ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"] if f in have), "DejaVu Sans")
    plt.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": sans, "mathtext.it": f"{sans}:italic",
                         "mathtext.bf": f"{sans}:bold", "mathtext.sf": sans})
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK,
        "axes.titlecolor": INK, "axes.titlelocation": "left", "xtick.color": INK2, "ytick.color": INK2,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.minor.size": 0, "ytick.minor.size": 0, "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-", "lines.linewidth": 1.4,
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round", "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path", "savefig.dpi": 300, "figure.dpi": 100,
        "axes.unicode_minus": True,
    })


def out_dir() -> str:
    d = os.path.join(paths.RESULTS, "figures")
    os.makedirs(d, exist_ok=True)
    return d


def save(fig, name: str, table: pd.DataFrame | None = None) -> str:
    d = out_dir()
    for ext in ("pdf", "eps", "svg", "png"):
        fig.savefig(os.path.join(d, f"{name}.{ext}"), facecolor="white")
    if table is not None:
        table.to_csv(os.path.join(d, f"{name}.csv"), index=False)
    plt.close(fig)
    return os.path.join(d, name)


def pair_label(s: str, t: str) -> str:
    return f"{DS[s]} → {DS[t]}"


def metric_label(t: str) -> str:
    return "micro-F1 (%)" if t in data.ALIGNED else "accuracy (%)"


# ----------------------------------------------------------------------------- statistics
def mean_ci(v) -> tuple[float, float, int]:
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    n = len(v)
    if n == 0:
        return float("nan"), float("nan"), 0
    if n == 1:
        return float(v[0]), float("nan"), 1
    return float(v.mean()), float(stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / math.sqrt(n)), n


def paired_diff(a: pd.Series, b: pd.Series, test: str) -> dict:
    """a - b over common splits: mean, 95% CI half-width, p-value of the paired test, n."""
    common = a.index.intersection(b.index)
    d = (a.loc[common] - b.loc[common]).astype(float)
    m, h, n = mean_ci(d.values)
    pr = analysis.paired(a, b)
    return dict(diff=m, ci=h, n=n, p=pr["p_t"], p_w=pr["p_w"])


def series(df, **kw) -> pd.Series:
    return analysis._series(df, **kw)


def test_for(plan_name: str) -> str:
    return "t"


def holm_by(cells: dict, family) -> dict:
    """Holm adjustment of cells[key]['p'] within groups of keys sharing family(key)."""
    groups, out = {}, {}
    for key in cells:
        groups.setdefault(family(key), []).append(key)
    for keys in groups.values():
        out.update(zip(keys, analysis.holm([cells[k]["p"] for k in keys])))
    return out


def main_ks(df, ks=None):
    have = sorted(df.k.unique())
    return [k for k in (ks or (1, 5, 20)) if k in have] or have


def _diverging():
    """blue (positive) - neutral grey - red (negative), interpolated in OKLab for even perceptual steps."""
    def to_lab(h):
        c = np.array(to_rgb(h))
        lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        m1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                       [0.0883024619, 0.2817188376, 0.6299787005]])
        m2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                       [0.0259040371, 0.7827717662, -0.8086757660]])
        return m2 @ np.cbrt(m1 @ lin), m1, m2

    def from_lab(lab, m1, m2):
        lin = np.linalg.solve(m1, np.linalg.solve(m2, lab) ** 3)
        lin = np.clip(lin, 0, 1)
        return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)

    (r, m1, m2), (g, _, _), (b, _, _) = to_lab(RED), to_lab(MID), to_lab(BLUE)
    cols = [from_lab(r + (g - r) * t, m1, m2) for t in np.linspace(0, 1, 128)]
    cols += [from_lab(g + (b - g) * t, m1, m2) for t in np.linspace(0, 1, 128)]
    cm = LinearSegmentedColormap.from_list("sgx_div", np.clip(cols, 0, 1))
    cm.set_bad("white")
    return cm


def _text_color(rgba) -> str:
    r, g, b = rgba[:3]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return INK if lum > 0.45 else "white"


# ----------------------------------------------------------------------------- figure: label-budget curves
def fig_budget_curves(df: pd.DataFrame, plan_name: str, model: str = "sgx",
                      conds=("scratch", "finetune", "frozen", "zeroshot")) -> str | None:
    pairs = [p for p in jobs.ALL_PAIRS if not df[(df.target == p[1]) & (df.model == model)].empty]
    if not pairs:
        return None
    ks = sorted(df.k.unique())
    ncol = 4
    nrow = math.ceil((len(pairs) + 1) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(FULL_W, 1.75 * nrow), layout="constrained")
    axes = np.atleast_2d(axes)
    dodge = {c: (i - (len(conds) - 1) / 2) * 0.035 for i, c in enumerate(conds)}
    rows = []
    for ax, (s, t) in zip(axes.flat, pairs):
        for c in conds:
            q = df[(df.target == t) & (df.model == model) & (df.cond == c)]
            q = q[q.source.isna()] if c == "scratch" else q[q.source == s]
            if q.empty:
                continue
            xs, ms, hs = [], [], []
            for k in ks:
                m, h, n = mean_ci(q[q.k == k].test.values)
                if n:
                    xs.append(k * 10 ** dodge[c])
                    ms.append(m)
                    hs.append(0 if math.isnan(h) else h)
                    rows.append(dict(source=s, target=t, model=model, cond=c, k=k, mean=round(m, 2),
                                     ci95=round(h, 2) if not math.isnan(h) else None, n=n))
            col, mk, _ = COND[c]
            ax.errorbar(xs, ms, yerr=hs, color=col, marker=mk, ms=4.2, mew=0.8, mec="white", lw=1.4,
                        elinewidth=0.8, capsize=0, zorder=3 if c != "scratch" else 2)
        ax.set_xscale("log")
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlim(min(ks) / 1.35, max(ks) * 1.35)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.set_title(pair_label(s, t), fontsize=7, pad=3)
        ax.set_ylabel(metric_label(t), fontsize=7)
    for ax in axes.flat[len(pairs):]:
        ax.axis("off")
    for ax in axes[-1]:
        if ax.has_data():
            ax.set_xlabel("labelled nodes per class (k)", fontsize=7)
    handles = [plt.Line2D([], [], color=COND[c][0], marker=COND[c][1], ms=4.5, mec="white", mew=0.8, lw=1.4,
                          label=COND[c][2]) for c in conds]
    leg_ax = axes.flat[len(pairs)] if len(pairs) < nrow * ncol else None
    if leg_ax is not None:
        leg_ax.legend(handles=handles, loc="center left", fontsize=7, handlelength=2.2, borderaxespad=0)
        leg_ax.text(0.0, 0.08, f"{SHORT[model]}; mean and 95% CI over splits", transform=leg_ax.transAxes,
                    fontsize=6.5, color=INK2)
    else:
        fig.legend(handles=handles, loc="outside lower center", ncol=len(conds))
    return save(fig, f"fig_budget_curves_{model}_{plan_name}", pd.DataFrame(rows))


# ----------------------------------------------------------------------------- figure: heatmaps
def _heatmap_panels(mats, row_labels, col_labels, ks, title_fmt, cbar_label, name, table, stars, vmax=None,
                    group_breaks=()):
    finite = [abs(v) for m in mats for v in np.asarray(m, float).ravel() if not math.isnan(v)]
    if not finite:
        return None
    if vmax is None:
        vmax = max(5.0, math.ceil(max(finite) / 5) * 5)
    cm = _diverging()
    nr, nc = len(row_labels), len(col_labels)
    h = 0.55 + 0.19 * nr
    fig, axes = plt.subplots(1, len(ks), figsize=(FULL_W, h), layout="constrained", sharey=True)
    axes = np.atleast_1d(axes)
    mesh = None
    for j, (ax, k, M, S) in enumerate(zip(axes, ks, mats, stars)):
        M = np.ma.masked_invalid(np.asarray(M, float))
        mesh = ax.pcolormesh(np.arange(nc + 1), np.arange(nr + 1), M, cmap=cm, vmin=-vmax, vmax=vmax,
                             edgecolors="white", linewidth=1.0)
        for i in range(nr):
            for c in range(nc):
                v = M[i, c]
                if np.ma.is_masked(v):
                    continue
                ax.text(c + 0.5, i + 0.5, f"{v:.1f}{'*' if S[i][c] else ''}".replace("-", "\u2212"), ha="center",
                        va="center", fontsize=6, color=_text_color(cm((v + vmax) / (2 * vmax))))
        ax.set_xlim(0, nc)
        ax.set_ylim(nr, 0)
        ax.set_xticks(np.arange(nc) + 0.5)
        ax.set_xticklabels(col_labels, rotation=40, ha="right", rotation_mode="anchor", fontsize=6.5)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        for b in group_breaks:
            ax.axhline(b, color=INK2, lw=0.7)
        ax.set_title(title_fmt.format(k=k), fontsize=7.5, pad=3)
        if j == 0:
            ax.set_yticks(np.arange(nr) + 0.5)
            ax.set_yticklabels(row_labels, fontsize=6.5)
    cb = fig.colorbar(mesh, ax=axes, shrink=0.8, aspect=25, pad=0.01)
    cb.set_label(cbar_label, fontsize=7)
    cb.ax.tick_params(labelsize=6.5, length=2)
    cb.outline.set_visible(False)
    cb.solids.set_rasterized(False)  # keep the colour bar vector too (matplotlib rasterises long colour bars)
    return save(fig, name, pd.DataFrame(table))


def fig_gain_heatmap(df: pd.DataFrame, plan_name: str, cond: str = "finetune", ks=None,
                     model_list=None) -> str | None:
    """Transfer gain (condition - scratch) for every pair x model at k = 1, 5, 20. Holm family: one model (column).
    model_list: the columns (default: the six main models)."""
    ks = main_ks(df, ks)
    pairs, breaks = [], []
    for _, ps, _ in GROUPS:
        ps = [p for p in ps if not df[(df.target == p[1]) & (df.cond == cond) & (df.source == p[0])].empty]
        if ps:
            if pairs:
                breaks.append(len(pairs))
            pairs += ps
    models = [m for m in (model_list or MODELS) if m in df.model.unique()]
    if not pairs or not models:
        return None
    test = test_for(plan_name)
    cells, table = {}, []
    for k in ks:
        for s, t in pairs:
            for m in models:
                a = series(df, target=t, model=m, cond=cond, k=k, source=s)
                b = series(df, target=t, model=m, cond="scratch", k=k, source=None)
                if a.empty or b.empty:
                    continue
                cells[(k, s, t, m)] = paired_diff(a, b, test)
    adj = holm_by(cells, lambda key: key[3])
    mats, stars = [], []
    for k in ks:
        M = np.full((len(pairs), len(models)), np.nan)
        S = [[False] * len(models) for _ in pairs]
        for i, (s, t) in enumerate(pairs):
            for j, m in enumerate(models):
                c = cells.get((k, s, t, m))
                if c is None:
                    continue
                M[i, j] = c["diff"]
                S[i][j] = adj[(k, s, t, m)] < 0.05
                table.append(dict(k=k, source=s, target=t, model=m, gain=round(c["diff"], 2),
                                  ci95=round(c["ci"], 2) if not math.isnan(c["ci"]) else None, n=c["n"],
                                  p_t=c["p"], p_holm=adj[(k, s, t, m)], p_wilcoxon=c["p_w"]))
        mats.append(M)
        stars.append(S)
    what = {"finetune": "fine-tuning", "frozen": "the frozen pretrained body"}.get(cond, cond)
    return _heatmap_panels(mats, [pair_label(s, t) for s, t in pairs], [SHORT[m] for m in models], ks,
                           "k = {k} labels per class", f"gain of {what} over scratch (points)",
                           f"fig_gain_heatmap_{cond}_{plan_name}", table, stars, group_breaks=breaks)


def fig_vs_baselines(df: pd.DataFrame, plan_name: str, ks=None, focus: str = "sgx") -> str | None:
    """SG-XNet (or `focus`) minus each other model, from scratch and after fine-tuning. Holm family: one compared
    model (column)."""
    ks = main_ks(df, ks)
    base = [m for m in MODELS if m != focus and m in df.model.unique()]
    if focus not in df.model.unique() or not base:
        return None
    rows_def, breaks = [], []
    targets = list(dict.fromkeys(t for _, ps, _ in GROUPS for _, t in ps))
    for t in targets:
        if df[(df.target == t) & (df.model == focus)].empty:
            continue
        if rows_def:
            breaks.append(len(rows_def))
        rows_def.append((t, "scratch", None))
        rows_def += [(t, "finetune", s) for s, tt in jobs.ALL_PAIRS if tt == t]
    test = test_for(plan_name)
    cells = {}
    for k in ks:
        for t, c, s in rows_def:
            a = series(df, target=t, model=focus, cond=c, k=k, source=s)
            for m in base:
                b = series(df, target=t, model=m, cond=c, k=k, source=s)
                if not a.empty and not b.empty:
                    cells[(k, t, c, s, m)] = paired_diff(a, b, test)
    if not cells:
        return None
    adj = holm_by(cells, lambda key: key[4])
    mats, stars, table = [], [], []
    for k in ks:
        M = np.full((len(rows_def), len(base)), np.nan)
        S = [[False] * len(base) for _ in rows_def]
        for i, (t, c, s) in enumerate(rows_def):
            for j, m in enumerate(base):
                v = cells.get((k, t, c, s, m))
                if v is None:
                    continue
                M[i, j], S[i][j] = v["diff"], adj[(k, t, c, s, m)] < 0.05
                table.append(dict(k=k, target=t, cond=c, source=s, baseline=m, sgx_minus_baseline=round(v["diff"], 2),
                                  ci95=round(v["ci"], 2) if not math.isnan(v["ci"]) else None, n=v["n"], p_t=v["p"],
                                  p_holm=adj[(k, t, c, s, m)], p_wilcoxon=v["p_w"]))
        mats.append(M)
        stars.append(S)
    labels = [f"{DS[t]}: scratch" if c == "scratch" else f"{DS[t]}: from {DS[s]}" for t, c, s in rows_def]
    if focus == "sgx":
        cbar, name = "SG-XNet minus baseline (points)", f"fig_sgx_vs_baselines_{plan_name}"
    else:
        cbar, name = f"{SHORT[focus]} minus model (points)", f"fig_{focus}_vs_models_{plan_name}"
    return _heatmap_panels(mats, labels, [SHORT[m] for m in base], ks, "k = {k} labels per class",
                           cbar, name, table, stars, group_breaks=breaks)


# ----------------------------------------------------------------------------- figure: grouped gain bars
def fig_gain_bars(df: pd.DataFrame, plan_name: str, conds: list[str], name: str, model: str = "sgx",
                  k: int = 5, resolve=None) -> str | None:
    """Horizontal bars: gain over scratch for several conditions, one row group per pair. Holm family: one condition.
    `resolve(df, s, t, model, k, cond)` may map a composite condition (e.g. 'ssl') to a concrete one."""
    pairs = [p for p in jobs.ALL_PAIRS if not df[(df.target == p[1]) & (df.model == model) & (df.k == k)].empty]
    if not pairs:
        return None
    test = test_for(plan_name)
    cells = {}
    for s, t in pairs:
        b = series(df, target=t, model=model, cond="scratch", k=k, source=None)
        for c in conds:
            cc, src = (resolve(df, s, t, model, k, c) if resolve else (c, s))
            if cc is None:
                continue
            a = series(df, target=t, model=model, cond=cc, k=k, source=src)
            if not a.empty and not b.empty:
                cells[(s, t, c)] = dict(paired_diff(a, b, test), concrete=cc)
    if len({key[2] for key in cells}) < 2:  # nothing to compare yet (e.g. the pilot has no controls)
        return None
    adj = holm_by(cells, lambda key: key[2])
    nb = len(conds)
    slot = 0.8 / nb
    fig, ax = plt.subplots(figsize=(FULL_W * 0.62, 0.42 + 0.26 * len(pairs) * max(1, nb / 2.2)), layout="constrained")
    table = []
    for i, (s, t) in enumerate(pairs):
        for j, c in enumerate(conds):
            v = cells.get((s, t, c))
            if v is None:
                continue
            y = i - 0.4 + slot * (j + 0.5)
            col, mk, _ = COND[c]
            ax.barh(y, v["diff"], height=slot * 0.82, color=col, zorder=2)
            if not math.isnan(v["ci"]):
                ax.plot([v["diff"] - v["ci"], v["diff"] + v["ci"]], [y, y], color=INK, lw=0.7, zorder=3)
            if adj[(s, t, c)] < 0.05:
                x = v["diff"] + (v["ci"] if not math.isnan(v["ci"]) else 0) * np.sign(v["diff"] or 1)
                ax.text(x + 0.4 * np.sign(v["diff"] or 1), y, "*", va="center",
                        ha="left" if v["diff"] >= 0 else "right", fontsize=7, color=INK)
            table.append(dict(source=s, target=t, model=model, k=k, cond=c, concrete_cond=v["concrete"],
                              gain=round(v["diff"], 2), ci95=None if math.isnan(v["ci"]) else round(v["ci"], 2),
                              n=v["n"], p_t=v["p"], p_holm=adj[(s, t, c)], p_wilcoxon=v["p_w"]))
    ax.axvline(0, color=INK2, lw=0.8, zorder=1)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([pair_label(s, t) for s, t in pairs], fontsize=6.8)
    ax.set_ylim(len(pairs) - 0.5, -0.5)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"gain over training {SHORT[model]} from scratch (points), k = {k}", fontsize=7)
    handles = [patches.Patch(color=COND[c][0], label=COND[c][2]) for c in conds]
    fig.legend(handles=handles, loc="outside lower center", ncol=1, fontsize=6.5)
    return save(fig, f"{name}_{model}_k{k}_{plan_name}", pd.DataFrame(table))


def best_variant(prefixes: list[str]):
    """Resolver choosing, per (pair, model, k), the variant with the best mean VALIDATION score (no test peeking)."""
    def resolve(df, s, t, model, k, cond):
        cands = []
        for c in df.cond.unique():
            if not any(c.startswith(p) for p in prefixes):
                continue
            q = df[(df.target == t) & (df.model == model) & (df.k == k) & (df.cond == c)]
            q = q[q.source.isna()] if "_self" in c else q[q.source == s]
            if not q.empty:
                cands.append((q.val.mean(), c, None if "_self" in c else s))
        if not cands:
            return None, None
        _, c, src = max(cands)
        return c, src
    return resolve


# ----------------------------------------------------------------------------- figure: ablation
def fig_ablation(df: pd.DataFrame, plan_name: str, k: int = 5) -> str | None:
    """Variant minus SG-XNet for the ablation pairs. Holm family: one panel (pair and condition)."""
    variants = [m for m in jobs.ABLATION_MODELS if m != "sgx" and m in df.model.unique()]
    pairs = [p for p in jobs.ABLATION_PAIRS if not df[(df.target == p[1]) & df.model.isin(variants)].empty]
    if not variants or not pairs:
        return None
    test = test_for(plan_name)
    conds = ["scratch", "finetune"]
    cells = {}
    for s, t in pairs:
        for c in conds:
            src = None if c == "scratch" else s
            a = series(df, target=t, model="sgx", cond=c, k=k, source=src)
            for m in variants:
                b = series(df, target=t, model=m, cond=c, k=k, source=src)
                if not a.empty and not b.empty:
                    cells[(s, t, c, m)] = paired_diff(b, a, test)
    if not cells:
        return None
    adj = holm_by(cells, lambda key: key[:3])
    fig, axes = plt.subplots(len(conds), len(pairs), figsize=(FULL_W, 0.6 + 1.25 * len(conds)), layout="constrained",
                             sharey=True, squeeze=False)
    table = []
    for r, c in enumerate(conds):
        for j, (s, t) in enumerate(pairs):
            ax = axes[r, j]
            for i, m in enumerate(variants):
                v = cells.get((s, t, c, m))
                if v is None:
                    continue
                ax.barh(i, v["diff"], height=0.62, color=BLUE if v["diff"] >= 0 else RED, zorder=2)
                if not math.isnan(v["ci"]):
                    ax.plot([v["diff"] - v["ci"], v["diff"] + v["ci"]], [i, i], color=INK, lw=0.7, zorder=3)
                if adj[(s, t, c, m)] < 0.05:
                    ax.text(v["diff"] + (v["ci"] if not math.isnan(v["ci"]) else 0) * np.sign(v["diff"] or 1)
                            + 0.3 * np.sign(v["diff"] or 1), i, "*", va="center",
                            ha="left" if v["diff"] >= 0 else "right", fontsize=7)
                table.append(dict(source=s, target=t, cond=c, k=k, variant=m, variant_minus_sgx=round(v["diff"], 2),
                                  ci95=None if math.isnan(v["ci"]) else round(v["ci"], 2), n=v["n"], p_t=v["p"],
                                  p_holm=adj[(s, t, c, m)], p_wilcoxon=v["p_w"]))
            ax.axvline(0, color=INK2, lw=0.8, zorder=1)
            ax.grid(axis="x")
            ax.set_axisbelow(True)
            ax.tick_params(axis="y", length=0)
            ax.spines["left"].set_visible(False)
            ax.set_yticks(range(len(variants)))
            ax.set_yticklabels([SHORT[m] for m in variants], fontsize=6.8)
            ax.set_ylim(len(variants) - 0.5, -0.5)
            ttl = f"{DS[t]}, from scratch" if c == "scratch" else f"{DS[t]}, fine-tuned from {DS[s]}"
            ax.set_title(ttl, fontsize=7.5, pad=3)
            if r == len(conds) - 1:
                ax.set_xlabel(f"variant minus SG-XNet (points), k = {k}", fontsize=7)
    return save(fig, f"fig_ablation_k{k}_{plan_name}", pd.DataFrame(table))


# ----------------------------------------------------------------------------- figure: homophily vs gain
LABEL_OFFSET = {("citeseer", "cora"): (5, 4, "left"), ("dblp", "pubmed"): (5, -9, "left"),
                ("squirrel_filtered", "chameleon_filtered"): (5, -9, "left"),
                ("amazon_ratings", "roman_empire"): (-5, 5, "right"), ("citeseer", "chameleon_filtered"): (-5, 5, "right")}


def fig_homophily(df: pd.DataFrame, plan_name: str, model: str = "sgx", ks=None) -> str | None:
    """Fine-tuning gain against the source-target homophily gap: one point per pair, 95% CI whiskers."""
    p = os.path.join(paths.RESULTS, "datasets.json")
    if not os.path.exists(p):
        return None
    ds = {d["name"]: d for d in json.load(open(p))}
    ks = main_ks(df, ks)
    test = test_for(plan_name)
    fig, axes = plt.subplots(1, len(ks), figsize=(FULL_W, 2.3), layout="constrained", sharey=True, squeeze=False)
    table, any_pt = [], False
    for ax, k in zip(axes[0], ks):
        for gname, ps, _ in GROUPS:
            col, mk = GROUP_STYLE[gname]
            pts = []
            for s, t in ps:
                a = series(df, target=t, model=model, cond="finetune", k=k, source=s)
                b = series(df, target=t, model=model, cond="scratch", k=k, source=None)
                if a.empty or b.empty or s not in ds or t not in ds:
                    continue
                v = paired_diff(a, b, test)
                gap = abs(ds[s]["h_edge"] - ds[t]["h_edge"])
                ax.errorbar(gap, v["diff"], yerr=0 if math.isnan(v["ci"]) else v["ci"], color=col, marker=mk, ms=5,
                            mec="white", mew=0.8, elinewidth=0.8, capsize=0, lw=0, zorder=3)
                pts.append((gap, v["diff"], s, t))
                any_pt = True
                table.append(dict(k=k, source=s, target=t, group=gname, h_edge_source=ds[s]["h_edge"],
                                  h_edge_target=ds[t]["h_edge"], homophily_gap=round(gap, 3),
                                  gain=round(v["diff"], 2), ci95=None if math.isnan(v["ci"]) else round(v["ci"], 2)))
            if not pts:
                continue
            if gname == "Shared features and labels":  # six nearby points: one label for the cluster
                ax.annotate("A/C/D pairs", (max(q[0] for q in pts), float(np.mean([q[1] for q in pts]))),
                            xytext=(7, 0), textcoords="offset points", fontsize=6, color=INK2, va="center")
            else:
                for gap, d, s, t in pts:
                    dx, dy, ha = LABEL_OFFSET.get((s, t), (5, 3, "left"))
                    ax.annotate(f"{CODE[s]}\u2192{CODE[t]}", (gap, d), xytext=(dx, dy), textcoords="offset points",
                                fontsize=6, color=INK2, ha=ha)
        ax.axhline(0, color=INK2, lw=0.8, zorder=1)
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.set_title(f"k = {k} labels per class", fontsize=7.5, pad=3)
    if not any_pt:
        plt.close(fig)
        return None
    axes[0, 0].set_ylabel(f"fine-tuning gain, {SHORT[model]} (points)", fontsize=7)
    fig.supxlabel("absolute difference in edge homophily between source and target", fontsize=7)
    handles = [plt.Line2D([], [], color=GROUP_STYLE[g][0], marker=GROUP_STYLE[g][1], lw=0, ms=5, mec="white",
                          label=g) for g, _, _ in GROUPS]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, fontsize=6.8)
    return save(fig, f"fig_homophily_gap_{model}_{plan_name}", pd.DataFrame(table))


# ----------------------------------------------------------------------------- diagrams (no data)
def _box(ax, x, y, w, h, text, fc="white", ec=AXIS, lw=0.8, fs=7, color=INK, weight="normal", r=0.06):
    ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec,
                                        lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, weight=weight,
            zorder=3, linespacing=1.25)


def _arrow(ax, x0, y0, x1, y1, color=INK2, lw=0.9, style="-|>", ls="-"):
    ax.add_patch(patches.FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=7, color=color,
                                         lw=lw, linestyle=ls, shrinkA=0, shrinkB=0, zorder=1))


def fig_model_diagram() -> str:
    H = 1.8
    fig = plt.figure(figsize=(FULL_W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FULL_W)
    ax.set_ylim(0, H)
    ax.axis("off")
    y, h = 0.78, 0.56
    _box(ax, 0.08, y, 0.78, h, "node features\n$X_G$  ($n \\times d_G$)\n+ graph $A_G$", fc="white")
    _arrow(ax, 0.86, y + h / 2, 1.06, y + h / 2)
    _box(ax, 1.06, y, 0.86, h, "encoder\nlinear $d_G \\to h$\n(graph-specific)", fc=MID)
    _arrow(ax, 1.92, y + h / 2, 2.17, y + h / 2)
    ax.add_patch(patches.FancyBboxPatch((2.17, y - 0.17), 3.0, h + 0.34, boxstyle="round,pad=0,rounding_size=0.08",
                                        fc=BLUE_WASH, ec=BLUE, lw=0.9, zorder=1))
    ax.text(2.17 + 3.0 / 2, y + h + 0.29, "transferable body: $K-1$ hidden message-passing layers",
            ha="center", va="center", fontsize=7, color=INK)
    _box(ax, 2.3, y, 0.92, h, "GraphSAGE\n(mean or max)\nlayer 1", fc="white", ec=BLUE)
    _arrow(ax, 3.22, y + h / 2, 3.4, y + h / 2)
    _box(ax, 3.4, y, 0.72, h, "GCN\nlayer 2", fc="white", ec=BLUE)
    ax.text(4.3, y + h / 2, "\u2026", ha="center", va="center", fontsize=10, color=INK2)
    _arrow(ax, 4.12, y + h / 2, 4.22, y + h / 2, style="-")
    _arrow(ax, 4.38, y + h / 2, 4.46, y + h / 2)
    _box(ax, 4.46, y, 0.6, h, "GCN\nlayer\n$K-1$", fc="white", ec=BLUE)
    _arrow(ax, 5.17, y + h / 2, 5.38, y + h / 2)
    _box(ax, 5.38, y, 0.94, h, "output layer\nGCN layer $K$\n$h \\to C_G$", fc=MID)
    _arrow(ax, 6.32, y + h / 2, 6.5, y + h / 2)
    ax.text(6.55, y + h / 2, "class\nscores", ha="left", va="center", fontsize=7, color=INK)
    note = ("re-initialised for the target graph;\ncopied too when source and target\nshare features and labels")
    ax.text(3.67, 0.3, "copied from the source model in every transfer condition", ha="center", va="center",
            fontsize=6.8, color=INK2)
    ax.text(1.49, 0.3, note, ha="center", va="center", fontsize=6.3, color=INK2, linespacing=1.2)
    ax.text(5.85, 0.3, note, ha="center", va="center", fontsize=6.3, color=INK2, linespacing=1.2)
    return save(fig, "fig_model_diagram")


def fig_protocol_diagram() -> str:
    W, H = FULL_W, 3.3
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    # 1  pretraining
    ax.text(0.1, H - 0.17, "1  Pretraining", fontsize=7.5, weight="bold", color=INK, va="center")
    rows = [(2.35, "source graph $G_s$\n80 / 10 / 10 % split", "supervised training,\nearly stopping on\n"
             "source validation", "$\\theta_s$", True),
            (1.62, "negative controls:\nshuffled source labels,\nor a random graph", "the same\nsupervised training",
             "$\\theta_{ctrl}$", False),
            (0.89, "self-supervised baselines:\nDGI, GRACE on $G_s$\nor on $G_t$ (no labels)", "contrastive\ntraining",
             "$\\theta_{ssl}$", False)]
    for y, a, b, c, main in rows:
        _box(ax, 0.1, y, 1.32, 0.5, a, fc="white" if main else MID, fs=6.4)
        _arrow(ax, 1.42, y + 0.25, 1.62, y + 0.25)
        _box(ax, 1.62, y, 1.22, 0.5, b, fc=BLUE_WASH if main else MID, ec=BLUE if main else AXIS, fs=6.4)
        _arrow(ax, 2.84, y + 0.25, 3.04, y + 0.25)
        _box(ax, 3.04, y, 0.5, 0.5, c, fc="white", ec=BLUE if main else AXIS, fs=8)
        ax.plot([3.54, 3.74], [y + 0.25, y + 0.25], color=MUTED, lw=0.8, zorder=1)
    ax.plot([3.74, 3.74], [rows[2][0] + 0.25, rows[0][0] + 0.25], color=MUTED, lw=0.8, zorder=1)
    # 2  target graph
    x0, wt = 4.0, 2.9
    ax.text(x0, H - 0.17, "2  Training and testing on the target graph", fontsize=7.5, weight="bold", color=INK,
            va="center")
    _box(ax, x0, 2.35, wt, 0.5, "target graph $G_t$: $k$ labelled nodes per class ($k$ = 1, 3, 5, 10, 20),\n"
         "30 validation nodes per class, all other nodes for testing", fc="white", fs=6.4)
    conds = [("from scratch", "random initialisation"),
             ("fine-tuned", "start from $\\theta$, train all weights"),
             ("frozen body", "start from $\\theta$, body fixed, output layer trained"),
             ("zero-shot", "use $\\theta_s$ as it is (aligned graphs only)"),
             ("domain adaptation", "DANN, UDAGCN: joint training on $G_s$ and $G_t$")]
    ys = [1.95 - i * 0.31 for i in range(len(conds))]
    for (a, b), yy in zip(conds, ys):
        ax.add_patch(patches.FancyBboxPatch((x0, yy), wt, 0.25, boxstyle="round,pad=0,rounding_size=0.05", fc="white",
                                            ec=AXIS, lw=0.7, zorder=2))
        ax.text(x0 + 0.08, yy + 0.125, a, fontsize=6.6, weight="bold", color=INK, va="center", zorder=3)
        ax.text(x0 + 1.12, yy + 0.125, b, fontsize=6.3, color=INK2, va="center", zorder=3)
    _arrow(ax, x0 + wt / 2, 2.35, x0 + wt / 2, ys[0] + 0.26)
    top, bot, xb = ys[1] + 0.25, ys[3], x0 - 0.07
    ax.plot([xb + 0.05, xb, xb, xb + 0.05], [top, top, bot, bot], color=MUTED, lw=0.8, zorder=1)
    _arrow(ax, 3.74, (top + bot) / 2, xb, (top + bot) / 2, color=MUTED, lw=0.8)
    ax.text(0.1, 0.3, "Model selection, identical for every condition: the learning rate (0.01, 0.005 or 0.001) and the "
            "stopping epoch are chosen on the target validation nodes, and the\ntest score is read at that epoch. All "
            "conditions of a split share the labelled nodes, the test nodes and the initial weights, so differences "
            "are paired.", fontsize=6.3, color=INK2, va="center", linespacing=1.3)
    return save(fig, "fig_protocol_diagram")


# ----------------------------------------------------------------------------- entry point
def make_all(plan_name: str) -> list[str]:
    style()
    made = [fig_model_diagram(), fig_protocol_diagram()]
    df = analysis.load_target()
    if df.empty:
        return made
    plan = jobs.PLANS[plan_name]
    focus = plan.get("focus", "sgx")
    if focus != "sgx":  # exploratory plan: its own model next to the six main models
        made.append(fig_budget_curves(df, plan_name, model=focus))
        made.append(fig_vs_baselines(df, plan_name, focus=focus))
        for c in ("finetune", "frozen"):
            made.append(fig_gain_heatmap(df, plan_name, c, model_list=MODELS + [focus]))
        return [m for m in made if m]
    for m in ("sgx", "gcn"):
        made.append(fig_budget_curves(df, plan_name, model=m))
    for c in ("finetune", "frozen"):
        made.append(fig_gain_heatmap(df, plan_name, c))
    made.append(fig_vs_baselines(df, plan_name))
    made.append(fig_homophily(df, plan_name))
    for k in (1, 5, 20):
        if k in plan["budgets"]:
            made.append(fig_gain_bars(df, plan_name, ["finetune", "ctrl_shuffle", "ctrl_random"],
                                      "fig_negative_controls", k=k))
            made.append(fig_gain_bars(df, plan_name, ["finetune", "ssl", "da"], "fig_pretraining_baselines", k=k,
                                      resolve=lambda d, s, t, m, kk, c: (c, s) if c == "finetune" else
                                      best_variant(["ssl_"] if c == "ssl" else ["dann", "udagcn"])(d, s, t, m, kk, c)))
    for k in (5, 20):
        made.append(fig_ablation(df, plan_name, k=k))
    return [m for m in made if m]
