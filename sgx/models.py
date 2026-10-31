"""Models (layout v2b, 22 Oct 2026).

Every model has the same three parts:

    encoder : dataset-specific linear projection x (d_in) -> h; Glorot-initialised, no bias.
              Re-initialised on a new graph unless the feature space is shared.
    body    : the first K - 1 message-passing layers, h -> h. This is the part transferred between graphs.
    out     : the K-th message-passing layer, h -> classes. Re-initialised unless the label space is shared.

Input dropout before the encoder; ReLU after every hidden layer and dropout before every message-passing layer
after the first. Two tuned options for the hidden layers, as in Luo et al. (2024): residual connections
(h <- h + ReLU(layer(h))) and LayerNorm before the activation.

Layout v2a, used for the first pilot run (21-22 Oct), had K hidden-to-hidden layers, a separate linear head and a
default-initialised encoder. Checked against a textbook two-layer GCN (Kipf & Welling, 2017) trained by the same
rules on the same Cora splits, v2a lost about 5 points and v2b about 1 point. That point is the remaining cost of
the dataset-specific input projection, which is what lets every hidden layer be transferred between graphs with
different feature spaces. tests/check_reference.py repeats the comparison.

SG-XNet = [GraphSAGE] + [GCN] * (K - 1) (manuscript, Algorithm 1): the GraphSAGE layer is the first hidden layer
and is transferred in every transfer condition.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv

# body layer layouts; K = number of message-passing layers
LAYOUTS = {
    "gcn": lambda K: ["gcn"] * K,
    "sage": lambda K: ["sage"] * K,
    "gat": lambda K: ["gat"] * K,
    "gin": lambda K: ["gin"] * K,
    "mlp": lambda K: ["mlp"] * K,
    "sgx": lambda K: ["sage"] + ["gcn"] * (K - 1),  # SG-XNet
    # ablations of SG-XNet (Reviewer 5, request 4)
    "sgx_order": lambda K: ["gcn", "sage"] + ["gcn"] * (K - 2),  # SAGE moved to second position
    "sgx_last": lambda K: ["gcn"] * (K - 1) + ["sage"],  # SAGE last
    "sgx_sage2": lambda K: ["sage", "sage"] + ["gcn"] * (K - 2),  # two SAGE layers
    "sgx_nodrop": lambda K: ["sage"] + ["gcn"] * (K - 1),  # SG-XNet without dropout
    "abl_allgcn": lambda K: ["gcn"] * K,  # SG-XNet configuration, all layers GCN
    "abl_allsage": lambda K: ["sage"] * K,  # SG-XNet configuration, all layers SAGE
    # candidate variant (not part of any plan unless adopted under the pre-specified rule of 22 Oct 2026)
    "sgxg": lambda K: ["gate"] * K,  # gated SG-XNet: every layer a learned GCN/SAGE mix
}
MAIN_MODELS = ["gcn", "sage", "gat", "gin", "mlp", "sgx"]


class SpMM(torch.autograd.Function):
    """y = A @ X for a sparse CSR matrix A and dense X, with a pre-computed transpose At for the backward pass
    (torch's own backward re-transposes A on every call, which dominated the run time)."""

    @staticmethod
    def forward(ctx, A, At, X):
        ctx.At = At
        return torch.sparse.mm(A, X)

    @staticmethod
    def backward(ctx, G):
        return None, None, torch.sparse.mm(ctx.At, G.contiguous())


_CHUNK = 1 << 22  # gathered elements per block in SegmentMax (16 MB of float32)


def max_plan(g) -> dict:
    """In-neighbour lists grouped into degree buckets (1, 2, 3-4, 5-8, ...), cached on the graph object.
    Shorter lists are padded by repeating the node's first neighbour, which leaves the maximum unchanged."""
    cached = getattr(g, "_max_plan", None)
    if cached is not None and cached[0] is g.ei:
        return cached[1]
    src, dst = g.ei[0].numpy(), g.ei[1].numpy()
    n = g.n
    order = np.lexsort((src, dst))
    s, d = src[order], dst[order]
    deg = np.bincount(d, minlength=n)
    ptr = np.zeros(n + 1, np.int64)
    ptr[1:] = np.cumsum(deg)
    buckets, hi, top = [], 1, max(int(deg.max()) if n else 1, 1)
    while True:
        lo = hi // 2 + 1 if hi > 1 else 1
        nodes = np.nonzero((deg >= lo) & (deg <= hi))[0]
        if len(nodes):
            col = np.arange(hi)[None, :]
            pos = np.where(col < deg[nodes][:, None], ptr[nodes][:, None] + col, ptr[nodes][:, None])
            buckets.append((torch.from_numpy(nodes), torch.from_numpy(s[pos].astype(np.int64))))
        if hi >= top:
            break
        hi *= 2
    plan = {"n": n, "buckets": buckets, "has": torch.from_numpy(deg > 0)}
    g._max_plan = (g.ei, plan)
    return plan


class SegmentMax(torch.autograd.Function):
    """Max over each node's in-neighbours, as PyG's max aggregation (nodes without neighbours get 0).

    PyG scatters over every edge in forward and backward; here the forward works on degree buckets and keeps the
    arg-max, so the backward touches one value per output entry instead of one per edge and feature. When several
    neighbours share the maximum, PyG splits the gradient between them and this routes it to one of them; such ties
    only occur at zeros produced by ReLU or dropout, where both versions pass no gradient further back, so training
    is the same (tests/check_equivalence.py compares gradients with respect to the pre-activation inputs)."""

    @staticmethod
    def forward(ctx, h, plan):
        hdim = h.shape[1]
        out = h.new_zeros(plan["n"], hdim)
        arg = torch.zeros(plan["n"], hdim, dtype=torch.long)
        for nodes, nbr in plan["buckets"]:
            deg = nbr.shape[1]
            step = max(1, _CHUNK // (deg * hdim))
            for a in range(0, len(nodes), step):
                nb = nbr[a:a + step]
                x = h.index_select(0, nb.reshape(-1)).view(nb.shape[0], deg, hdim)
                m, i = x.max(1)
                rows = nodes[a:a + step]
                out[rows] = m
                arg[rows] = nb.gather(1, i)
        ctx.save_for_backward(arg, plan["has"])
        ctx.n_in = h.shape[0]
        return out

    @staticmethod
    def backward(ctx, go):
        arg, has = ctx.saved_tensors
        hdim = go.shape[1]
        gflat = (go * has[:, None].to(go.dtype)).reshape(-1)
        flat = (arg * hdim + torch.arange(hdim)).reshape(-1)
        gh = go.new_zeros(ctx.n_in * hdim)
        gh.index_add_(0, flat, gflat)
        return gh.view(ctx.n_in, hdim), None


class SparseLinear(nn.Module):
    """Bias-free linear projection that also accepts a sparse CSR input matrix (with its transpose g.xt).

    No bias: the encoder is a linear factorisation of the first graph layer's weight matrix (that layer adds its own
    bias after propagation). A bias here would add the same offset to every node and swamp small normalised
    bag-of-words inputs."""

    def __init__(self, din: int, dout: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dout, din))
        nn.init.xavier_uniform_(self.weight)  # Glorot, like the first layer of a textbook GCN (layout v2b)

    def forward(self, x, xt=None):
        if x.layout != torch.strided:
            return SpMM.apply(x, xt, self.weight.t()) if xt is not None else torch.sparse.mm(x, self.weight.t())
        return F.linear(x, self.weight)


# The layers below keep PyG's modules (and therefore PyG's parameter initialisation) but propagate with
# pre-computed sparse adjacency matrices; the arithmetic is identical to the PyG forward passes
# (checked to < 1e-6 in tests/check_equivalence.py) and about 3x faster on CPU. Every layer takes (din, dout):
# hidden layers map h -> h, the output layer h -> classes.
class _GCN(nn.Module):
    def __init__(self, din, dout, **_):
        super().__init__()
        self.conv = GCNConv(din, dout, normalize=False, add_self_loops=False)

    def forward(self, h, g):
        out = SpMM.apply(g.A_gcn, g.A_gcn, self.conv.lin(h))  # symmetric normalised adjacency incl. self-loops
        return out + self.conv.bias


class _SAGE(nn.Module):
    def __init__(self, din, dout, aggr="mean", **_):
        super().__init__()
        self.aggr = aggr
        self.conv = SAGEConv(din, dout, aggr=aggr)

    def forward(self, h, g):
        if self.aggr == "mean":
            return self.conv.lin_l(SpMM.apply(g.A_mean, g.A_mean_t, h)) + self.conv.lin_r(h)
        if self.aggr == "max":
            return self.conv.lin_l(SegmentMax.apply(h, max_plan(g))) + self.conv.lin_r(h)
        return self.conv(h, g.ei)


class _GAT(nn.Module):
    def __init__(self, din, dout, heads=1, dropout=0.0, out=False, **_):
        super().__init__()
        if out:  # output layer: a single head, as in Velickovic et al. (2018) for transductive tasks
            self.conv = GATConv(din, dout, heads=1, concat=False, dropout=dropout)
        else:
            self.conv = GATConv(din, dout // heads, heads=heads, concat=True, dropout=dropout)

    def forward(self, h, g):
        return self.conv(h, g.ei)


class _GIN(nn.Module):
    def __init__(self, din, dout, **_):
        super().__init__()
        self.conv = GINConv(nn.Sequential(nn.Linear(din, din), nn.ReLU(), nn.Linear(din, dout)), train_eps=True)

    def forward(self, h, g):
        return self.conv.nn((1 + self.conv.eps) * h + SpMM.apply(g.A_sum, g.A_sum, h))


class _MLP(nn.Module):
    def __init__(self, din, dout, **_):
        super().__init__()
        self.lin = nn.Linear(din, dout)

    def forward(self, h, g):
        return self.lin(h)


class _Gate(nn.Module):
    """Gated hybrid layer (candidate "SG-XNet-G", designed 22 Oct 2026 after the pilot's tuning results):
    w * GCN(h) + (1 - w) * SAGE_mean(h), with a learned per-channel gate w = sigmoid(a), a = 0 at initialisation
    (an equal mix). The GCN branch smooths over the neighbourhood including the node itself; the SAGE branch keeps
    the node's own representation separate from the mean of its neighbours (lin_r vs lin_l)."""

    def __init__(self, din, dout, **_):
        super().__init__()
        self.gcn = _GCN(din, dout)
        self.sage = _SAGE(din, dout, aggr="mean")
        self.a = nn.Parameter(torch.zeros(dout))

    def forward(self, h, g):
        w = torch.sigmoid(self.a)
        return w * self.gcn(h, g) + (1 - w) * self.sage(h, g)

    def gate(self) -> float:
        """Mean weight of the GCN branch (0 = pure SAGE, 1 = pure GCN)."""
        return float(torch.sigmoid(self.a).mean())


_LAYER = {"gcn": _GCN, "sage": _SAGE, "gat": _GAT, "gin": _GIN, "mlp": _MLP, "gate": _Gate}


class _Hidden(nn.Module):
    """One hidden block: message-passing layer, optional LayerNorm (before the activation, as in Luo et al., 2024),
    ReLU. Residual connections are added by Net.body_forward."""

    def __init__(self, layer: nn.Module, hidden: int, norm: bool):
        super().__init__()
        self.layer = layer
        self.norm = nn.LayerNorm(hidden) if norm else None

    def forward(self, h, g):
        z = self.layer(h, g)
        if self.norm is not None:
            z = self.norm(z)
        return F.relu(z)


class Net(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, layers: list[str], hidden: int, dropout: float,
                 heads: int = 1, aggr: str = "mean", residual: bool = False, norm: bool = False):
        super().__init__()
        assert len(layers) >= 2, "K >= 2: at least one transferable hidden layer and the output layer"
        self.layers = list(layers)
        self.dropout = float(dropout)
        self.residual = bool(residual)
        kw = dict(heads=heads, aggr=aggr, dropout=dropout)
        self.encoder = SparseLinear(in_dim, hidden)
        self.body = nn.ModuleList([_Hidden(_LAYER[k](hidden, hidden, **kw), hidden, norm) for k in layers[:-1]])
        self.out = _LAYER[layers[-1]](hidden, out_dim, out=True, **kw)

    def encode(self, g):
        """Input dropout + dataset-specific linear projection."""
        x = g.x
        if x.layout == torch.strided:
            return self.encoder(F.dropout(x, self.dropout, self.training))
        # sparse bag-of-words input: dropout on the non-zero entries (identical to dense input dropout)
        xt = g.xt
        if self.training and self.dropout > 0:
            vals = F.dropout(x.values(), self.dropout, True)
            x = torch.sparse_csr_tensor(x.crow_indices(), x.col_indices(), vals, x.shape)
            xt = torch.sparse_csr_tensor(g.xt.crow_indices(), g.xt.col_indices(), vals[g.xt_perm], g.xt.shape)
        return self.encoder(x, xt)

    def body_forward(self, h, g):
        """The transferable hidden layers, applied to encoder outputs h; returns the node embeddings."""
        for i, block in enumerate(self.body):
            if i > 0:
                h = F.dropout(h, self.dropout, self.training)
            z = block(h, g)
            h = h + z if self.residual else z
        return h

    def embed(self, g):
        return self.body_forward(self.encode(g), g)

    def classify(self, h, g):
        """The output layer (the K-th message-passing layer) applied to embeddings h."""
        return self.out(F.dropout(h, self.dropout, self.training), g)

    def forward(self, g):
        return self.classify(self.embed(g), g)


def build(model: str, cfg: dict, in_dim: int, out_dim: int) -> Net:
    layers = LAYOUTS[model](int(cfg["K"]))
    dropout = 0.0 if model == "sgx_nodrop" else cfg["dropout"]
    return Net(in_dim, out_dim, layers, int(cfg["hidden"]), dropout, heads=int(cfg.get("heads", 1)),
               aggr=cfg.get("aggr", "mean"), residual=bool(cfg.get("residual", False)),
               norm=bool(cfg.get("norm", False)))


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
