"""Dataset loading, preprocessing and caching.

All graphs are made undirected, self-loops are removed and duplicate edges merged. Planetoid/CitationFull
bag-of-words features are row-normalised to unit L1 norm (the standard Kipf & Welling preprocessing); the ACDNE
word-count features (6,775-word vocabulary, ~35 non-zeros per row) are normalised to unit L2 norm, because
L1-normalised entries (~0.03) are too small to train from with weight decay; embedding features
(Platonov et al., 2023) are used as provided.
The processed graphs are cached as plain numpy arrays (.npz) so the cache does not depend on library versions.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import torch

from . import paths

# name -> how to load it. group: 'citation' (Planetoid/CitationFull), 'aligned' (shared vocabulary and
# label set; ACDNE cross-network benchmark), 'heterophilic' (Platonov et al., 2023).
DATASETS = {
    "cora": dict(loader="planetoid", pyg_name="Cora", norm="row", group="citation"),
    "citeseer": dict(loader="planetoid", pyg_name="CiteSeer", norm="row", group="citation"),
    "pubmed": dict(loader="planetoid", pyg_name="PubMed", norm="row", group="citation"),
    "dblp": dict(loader="citationfull", pyg_name="DBLP", norm="row", group="citation"),
    "acmv9": dict(loader="acdne", file="acmv9.mat", norm="l2", group="aligned"),
    "citationv1": dict(loader="acdne", file="citationv1.mat", norm="l2", group="aligned"),
    "dblpv7": dict(loader="acdne", file="dblpv7.mat", norm="l2", group="aligned"),
    "roman_empire": dict(loader="platonov", file="roman_empire.npz", norm=None, group="heterophilic"),
    "amazon_ratings": dict(loader="platonov", file="amazon_ratings.npz", norm=None, group="heterophilic"),
    "chameleon_filtered": dict(loader="platonov", file="chameleon_filtered.npz", norm=None, group="heterophilic"),
    "squirrel_filtered": dict(loader="platonov", file="squirrel_filtered.npz", norm=None, group="heterophilic"),
}
ALIGNED = {k for k, v in DATASETS.items() if v["group"] == "aligned"}


# ----------------------------------------------------------------------------- raw loaders
def _load_planetoid(spec):
    from torch_geometric.datasets import Planetoid

    d = Planetoid(os.path.join(paths.RAW, "planetoid"), spec["pyg_name"])[0]
    public = {"train": d.train_mask.numpy()[None], "val": d.val_mask.numpy()[None], "test": d.test_mask.numpy()[None]}
    return d.x.numpy(), d.edge_index.numpy(), d.y.numpy(), public


def _load_citationfull(spec):
    from torch_geometric.datasets import CitationFull

    d = CitationFull(os.path.join(paths.RAW, "citationfull"), spec["pyg_name"])[0]
    return d.x.numpy(), d.edge_index.numpy(), d.y.numpy(), None


def _load_acdne(spec):
    import scipy.io as sio

    m = sio.loadmat(os.path.join(paths.RAW, "acdne", spec["file"]))
    x = sp.csr_matrix(m["attrb"], dtype=np.float32)
    a = sp.coo_matrix(m["network"])
    edges = np.stack([a.row, a.col]).astype(np.int64)
    g = m["group"]
    y = (g.toarray() if sp.issparse(g) else np.asarray(g)).astype(np.float32)
    return x, edges, y, None


def _load_platonov(spec):
    d = np.load(os.path.join(paths.RAW, "platonov", spec["file"]))
    public = {"train": d["train_masks"], "val": d["val_masks"], "test": d["test_masks"]}
    return d["node_features"].astype(np.float32), d["edges"].T.astype(np.int64), d["node_labels"].astype(np.int64), public


_LOADERS = {"planetoid": _load_planetoid, "citationfull": _load_citationfull, "acdne": _load_acdne, "platonov": _load_platonov}


# ----------------------------------------------------------------------------- preprocessing
def undirected(edges: np.ndarray, n: int) -> np.ndarray:
    e = edges[:, edges[0] != edges[1]].astype(np.int64)
    e = np.concatenate([e, e[::-1]], axis=1)
    key = np.unique(e[0] * n + e[1])
    return np.stack([key // n, key % n])


def l2_normalise(x):
    x = sp.csr_matrix(x, dtype=np.float32)
    s = np.sqrt(np.asarray(x.multiply(x).sum(1)).ravel())
    s[s == 0] = 1.0
    return (sp.diags(1.0 / s) @ x).tocsr().astype(np.float32)


def row_normalise(x):
    if sp.issparse(x):
        x = sp.csr_matrix(x, dtype=np.float32)
        s = np.asarray(x.sum(1)).ravel()
        s[s == 0] = 1.0
        return (sp.diags(1.0 / s) @ x).tocsr().astype(np.float32)
    s = x.sum(1, keepdims=True)
    s[s == 0] = 1.0
    return (x / s).astype(np.float32)


def homophily(edge_index: np.ndarray, y: np.ndarray, multilabel: bool) -> dict:
    """Edge homophily and adjusted homophily (Platonov et al., 2023) on the undirected edge list."""
    src, dst = edge_index
    if multilabel:
        same = (y[src] * y[dst]).sum(1) > 0
        return {"h_edge": float(same.mean()), "h_adj": float("nan")}
    h_edge = float((y[src] == y[dst]).mean())
    deg = np.bincount(src, minlength=len(y)).astype(np.float64)
    dk = np.bincount(y, weights=deg, minlength=int(y.max()) + 1)
    p2 = float(((dk / dk.sum()) ** 2).sum())
    return {"h_edge": h_edge, "h_adj": (h_edge - p2) / (1.0 - p2)}


def prepare(name: str, force: bool = False) -> dict:
    """Load a raw dataset, clean it and write data/cache/<name>.npz plus <name>.json (statistics)."""
    paths.ensure(paths.CACHE)
    out = os.path.join(paths.CACHE, f"{name}.npz")
    meta_path = os.path.join(paths.CACHE, f"{name}.json")
    if os.path.exists(out) and os.path.exists(meta_path) and not force:
        with open(meta_path) as f:
            return json.load(f)
    spec = DATASETS[name]
    x, edges, y, public = _LOADERS[spec["loader"]](spec)
    n = x.shape[0]
    ei = undirected(edges, n)
    if spec["norm"] == "row":
        x = row_normalise(x)
    elif spec["norm"] == "l2":
        x = l2_normalise(x)
    multilabel = y.ndim == 2
    arrays = dict(edge_index=ei, y=y, multilabel=np.array(multilabel))
    # Storage: dense for every graph except the three ACDNE graphs (6,775-word vocabulary, ~60M entries each), which
    # are kept sparse. On Apple silicon dense products run on the matrix units and are 3-16x faster than PyTorch's
    # generic sparse kernel; for the ACDNE graphs the sparse path saves ~250 MB per graph and process.
    if sp.issparse(x) and spec["loader"] != "acdne":
        x = np.asarray(x.todense(), dtype=np.float32)
    if sp.issparse(x):
        x = sp.csr_matrix(x)
        arrays.update(x_data=x.data.astype(np.float32), x_indices=x.indices.astype(np.int64),
                      x_indptr=x.indptr.astype(np.int64), x_shape=np.array(x.shape))
    else:
        arrays.update(x=np.asarray(x, dtype=np.float32))
    if public is not None:
        arrays.update(public_train=public["train"], public_val=public["val"], public_test=public["test"])
    np.savez_compressed(out, **arrays)
    deg = np.bincount(ei[0], minlength=n)
    stats = dict(
        name=name, group=spec["group"], nodes=int(n), edges=int(ei.shape[1] // 2), features=int(x.shape[1]),
        classes=int(y.shape[1] if multilabel else y.max() + 1), multilabel=bool(multilabel),
        multilabel_fraction=float((y.sum(1) > 1).mean()) if multilabel else 0.0,
        avg_degree=float(deg.mean()), isolated=int((deg == 0).sum()),
        feature_density=float((x != 0).sum() / (x.shape[0] * x.shape[1])) if sp.issparse(x) else float((x != 0).mean()),
        public_splits=int(public["train"].shape[0]) if public is not None else 0,
        **homophily(ei, y, multilabel),
    )
    with open(meta_path, "w") as f:
        json.dump(stats, f, indent=1)
    return stats


# ----------------------------------------------------------------------------- in-memory graph
@dataclass
class Graph:
    name: str
    x: torch.Tensor  # dense float32, or torch.sparse_csr
    ei: torch.Tensor  # 2 x E, undirected, no self-loops
    gcn_ei: torch.Tensor  # with self-loops, for pre-normalised GCN propagation
    gcn_ew: torch.Tensor
    y: torch.Tensor  # long (n,) or float (n, C)
    num_classes: int
    multilabel: bool
    public: dict | None = None
    y_np: np.ndarray = field(default=None, repr=False)
    # pre-computed sparse propagation matrices (rows = receiving node)
    A_gcn: torch.Tensor = field(default=None, repr=False)  # D^-1/2 (A + I) D^-1/2, symmetric
    A_mean: torch.Tensor = field(default=None, repr=False)  # D^-1 A (mean over neighbours)
    A_mean_t: torch.Tensor = field(default=None, repr=False)  # its transpose (for the backward pass)
    A_sum: torch.Tensor = field(default=None, repr=False)  # A (sum over neighbours), symmetric
    xt: torch.Tensor = field(default=None, repr=False)  # transpose of a sparse feature matrix
    xt_perm: torch.Tensor = field(default=None, repr=False)  # xt.values() == x.values()[xt_perm]

    @property
    def n(self) -> int:
        return self.x.shape[0]

    @property
    def in_dim(self) -> int:
        return self.x.shape[1]

    @property
    def sparse_x(self) -> bool:
        return self.x.layout != torch.strided


def gcn_norm(ei: torch.Tensor, n: int):
    loop = torch.arange(n)
    e = torch.cat([ei, torch.stack([loop, loop])], 1)
    deg = torch.bincount(e[1], minlength=n).float()
    w = deg[e[0]].rsqrt() * deg[e[1]].rsqrt()
    return e, w


def _csr_from_scipy(m) -> torch.Tensor:
    m = sp.csr_matrix(m, dtype=np.float32)
    m.sort_indices()
    return torch.sparse_csr_tensor(torch.from_numpy(m.indptr.astype(np.int64)), torch.from_numpy(m.indices.astype(np.int64)),
                                   torch.from_numpy(m.data.astype(np.float32)), size=m.shape)


def propagation_matrices(ei_np: np.ndarray, n: int) -> dict:
    """Sparse matrices for GCN (normalised, with self-loops), mean and sum aggregation. Row i aggregates the
    messages sent to node i (PyG 'source_to_target' flow: edge (j, i) sends h_j to i)."""
    src, dst = ei_np
    a = sp.csr_matrix((np.ones(len(src), np.float32), (dst, src)), shape=(n, n))
    a_hat = a + sp.identity(n, dtype=np.float32, format="csr")
    d = np.asarray(a_hat.sum(1)).ravel()
    dinv = 1.0 / np.sqrt(d)
    a_gcn = sp.diags(dinv) @ a_hat @ sp.diags(dinv)
    deg = np.asarray(a.sum(1)).ravel()
    deg[deg == 0] = 1.0
    a_mean = sp.diags(1.0 / deg) @ a
    return dict(A_gcn=_csr_from_scipy(a_gcn), A_mean=_csr_from_scipy(a_mean), A_mean_t=_csr_from_scipy(a_mean.T),
                A_sum=_csr_from_scipy(a))


def make_graph(name, x, ei_np, y_np, multilabel, public=None) -> Graph:
    xt_tr = xt_perm = None
    if sp.issparse(x):
        x = sp.csr_matrix(x, dtype=np.float32)
        x.sort_indices()
        xt = _csr_from_scipy(x)
        pos = sp.csr_matrix((np.arange(x.nnz, dtype=np.float64), x.indices, x.indptr), shape=x.shape).T.tocsr()
        pos.sort_indices()
        xt_perm = torch.from_numpy(pos.data.astype(np.int64))
        xt_tr = _csr_from_scipy(x.T)
        assert torch.equal(xt_tr.values(), xt.values()[xt_perm])
    else:
        xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    ei_np = np.ascontiguousarray(ei_np, dtype=np.int64)
    ei = torch.from_numpy(ei_np)
    n = xt.shape[0]
    gei, gew = gcn_norm(ei, n)
    if multilabel:
        y = torch.from_numpy(np.asarray(y_np, dtype=np.float32))
        c = y.shape[1]
    else:
        y = torch.from_numpy(np.asarray(y_np, dtype=np.int64))
        c = int(y.max()) + 1
    return Graph(name, xt, ei, gei, gew, y, c, multilabel, public, y_np=np.asarray(y_np), xt=xt_tr, xt_perm=xt_perm,
                 **propagation_matrices(ei_np, n))


_CACHE: dict = {}


def load(name: str) -> Graph:
    """Load a prepared graph (kept in a small per-process cache)."""
    if name in _CACHE:
        return _CACHE[name]
    path = os.path.join(paths.CACHE, f"{name}.npz")
    if not os.path.exists(path):
        prepare(name)
    d = np.load(path)
    if "x" in d.files:
        x = d["x"]
    else:
        x = sp.csr_matrix((d["x_data"], d["x_indices"], d["x_indptr"]), shape=tuple(d["x_shape"]))
    public = None
    if "public_train" in d.files:
        public = {"train": d["public_train"], "val": d["public_val"], "test": d["public_test"]}
    g = make_graph(name, x, d["edge_index"], d["y"], bool(d["multilabel"]), public)
    if len(_CACHE) >= 4:  # keep memory bounded in worker processes
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[name] = g
    return g


def random_like(g: Graph, seed: int) -> Graph:
    """Negative-control source: Erdos-Renyi graph with the same number of nodes/edges, random features of the
    same dimension (same sparsity for sparse features), and random labels with the same label space."""
    rng = np.random.default_rng(seed)
    n, m = g.n, g.ei.shape[1] // 2
    src = rng.integers(0, n, size=m)
    dst = rng.integers(0, n, size=m)
    ei = undirected(np.stack([src, dst]), n)
    if g.sparse_x:
        indptr = g.x.crow_indices().numpy()
        nnz_rows = np.diff(indptr)
        k = rng.choice(nnz_rows, size=n)
        rows = np.repeat(np.arange(n), k)
        cols = rng.integers(0, g.in_dim, size=int(k.sum()))
        x = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, g.in_dim))
        x.sum_duplicates()
        x.data[:] = 1.0
        x = l2_normalise(x) if DATASETS.get(g.name, {}).get("norm") == "l2" else row_normalise(x)
    else:
        x = rng.standard_normal((n, g.in_dim)).astype(np.float32)
    if g.multilabel:
        y = np.zeros((n, g.num_classes), np.float32)
        y[np.arange(n), rng.integers(0, g.num_classes, size=n)] = 1.0
    else:
        y = rng.integers(0, g.num_classes, size=n).astype(np.int64)
    return make_graph(f"random({g.name})", x, ei, y, g.multilabel)


def shuffled_labels(g: Graph, seed: int) -> Graph:
    """Negative-control source: the real graph with node labels randomly permuted across nodes."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(g.n)
    y = g.y_np[perm]
    x = g.x
    if g.sparse_x:
        x = sp.csr_matrix((g.x.values().numpy(), g.x.col_indices().numpy(), g.x.crow_indices().numpy()), shape=tuple(g.x.shape))
    else:
        x = g.x.numpy()
    return make_graph(f"shuffled({g.name})", x, g.ei.numpy(), y, g.multilabel)


def stats_table() -> list[dict]:
    rows = []
    for name in DATASETS:
        rows.append(prepare(name))
    return rows
