"""Pretraining and domain-adaptation baselines (Reviewer 5, concern 8 and request 5).

Self-supervised pretraining (encoder + body, no labels), followed by the same fine-tuning protocol as supervised
transfer:
  * DGI   - Deep Graph Infomax (Velickovic et al., 2019): local-global mutual information, row-shuffled corruption.
  * GRACE - Zhu et al. (2020): two views (edge dropping + feature masking), node-level InfoNCE.
Both are run on the source graph ("@source") and on the target graph itself ("self", unlabelled target nodes).

Domain adaptation for the pairs with a shared feature space (ACDNE graphs):
  * DANN   - domain-adversarial training with a gradient-reversal layer (Ganin et al., 2016) on the GNN embedding.
  * UDAGCN - Wu et al. (2020): local (adjacency) + global (PPMI) graph convolutions with shared weights, attention
             fusion, domain-adversarial loss and target entropy minimisation. Re-implementation; the PPMI matrix is
             computed from 3-step random-walk transition probabilities instead of sampled walks.
In every case the labelled target nodes of the split are used (semi-supervised adaptation) and model selection uses
the target validation nodes, exactly as for the other methods.
"""
from __future__ import annotations

import math
import os

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import data, models, paths, train


# ----------------------------------------------------------------------------- graph views
class View:
    """A graph with some attributes replaced (augmented edges/features or another propagation matrix)."""

    _ATTRS = ("name", "x", "xt", "xt_perm", "ei", "gcn_ei", "gcn_ew", "A_gcn", "A_mean", "A_mean_t", "A_sum", "y",
              "num_classes", "multilabel", "y_np")

    def __init__(self, g, **over):
        for a in self._ATTRS:
            setattr(self, a, getattr(g, a, None))
        for k, v in over.items():
            setattr(self, k, v)

    @property
    def n(self):
        return self.x.shape[0]

    @property
    def in_dim(self):
        return self.x.shape[1]

    @property
    def sparse_x(self):
        return self.x.layout != torch.strided


def drop_edges(g, p: float, rng: np.random.Generator):
    ei = g.ei.numpy()
    und = ei[:, ei[0] < ei[1]]
    keep = und[:, rng.random(und.shape[1]) >= p]
    new = data.undirected(keep, g.n)
    mats = data.propagation_matrices(new, g.n)
    gei, gew = data.gcn_norm(torch.from_numpy(new), g.n)
    return dict(ei=torch.from_numpy(new), gcn_ei=gei, gcn_ew=gew, **mats)


def mask_features(g, p: float, gen: torch.Generator):
    keep = (torch.rand(g.in_dim, generator=gen) >= p).float()
    if g.sparse_x:
        vals = g.x.values() * keep[g.x.col_indices()]
        x = torch.sparse_csr_tensor(g.x.crow_indices(), g.x.col_indices(), vals, g.x.shape)
        xt = torch.sparse_csr_tensor(g.xt.crow_indices(), g.xt.col_indices(), vals[g.xt_perm], g.xt.shape)
        return dict(x=x, xt=xt)
    return dict(x=g.x * keep[None, :])


# ----------------------------------------------------------------------------- self-supervised pretraining
class DGI(nn.Module):
    def __init__(self, net: models.Net, hidden: int):
        super().__init__()
        self.net = net
        self.W = nn.Parameter(torch.empty(hidden, hidden))
        nn.init.xavier_uniform_(self.W)

    def loss(self, g, gen):
        h0 = self.net.encode(g)
        perm = torch.randperm(g.n, generator=gen)
        pos = self.net.body_forward(h0, g)
        neg = self.net.body_forward(h0[perm], g)  # row-shuffled features (the encoder is linear, so this equals
        s = torch.sigmoid(pos.mean(0))            # shuffling the input rows)
        v = self.W @ s
        lp, ln = pos @ v, neg @ v
        return F.binary_cross_entropy_with_logits(lp, torch.ones_like(lp)) + \
            F.binary_cross_entropy_with_logits(ln, torch.zeros_like(ln))


class GRACE(nn.Module):
    def __init__(self, net: models.Net, hidden: int, tau: float = 0.5, pe=(0.2, 0.4), pf=(0.3, 0.4), max_nodes=2048):
        super().__init__()
        self.net, self.tau, self.pe, self.pf, self.max_nodes = net, tau, pe, pf, max_nodes
        self.proj = nn.Sequential(nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, hidden))

    def _semi(self, a, b):
        s_ab = torch.exp(a @ b.t() / self.tau)
        s_aa = torch.exp(a @ a.t() / self.tau)
        return -torch.log(s_ab.diag() / (s_ab.sum(1) + s_aa.sum(1) - s_aa.diag()))

    def loss(self, g, gen, rng):
        z = []
        for pe, pf in zip(self.pe, self.pf):
            v = View(g, **drop_edges(g, pe, rng), **mask_features(g, pf, gen))
            z.append(self.net.embed(v))
        idx = torch.randperm(g.n, generator=gen)[: self.max_nodes]  # node subsample keeps memory bounded
        h1 = F.normalize(self.proj(z[0][idx]), dim=1)
        h2 = F.normalize(self.proj(z[1][idx]), dim=1)
        return 0.5 * (self._semi(h1, h2) + self._semi(h2, h1)).mean()


def pretrain_ssl(method: str, model_name: str, cfg: dict, g, seed: int):
    """Returns (state_dict of the network, info). Dropout is disabled during self-supervised pretraining."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = models.build(model_name, cfg, g.in_dim, g.num_classes)
    net.dropout = 0.0
    h = int(cfg["hidden"])
    if method == "dgi":
        mod, max_epochs, patience, lr = DGI(net, h), 500, 20, 1e-3
    elif method == "grace":
        mod, max_epochs, patience, lr = GRACE(net, h), 200, 10_000, 5e-4
    else:
        raise ValueError(method)
    if train.QUICK:
        max_epochs = 3
    opt = torch.optim.Adam(mod.parameters(), lr=lr, weight_decay=1e-5)
    best, best_ep, state, ep = math.inf, -1, None, 0
    for ep in range(max_epochs):
        mod.train()
        opt.zero_grad(set_to_none=True)
        loss = mod.loss(g, gen) if method == "dgi" else mod.loss(g, gen, rng)
        if not torch.isfinite(loss):
            break
        loss.backward()
        opt.step()
        if float(loss) < best:
            best, best_ep = float(loss), ep
            state = {a: b.detach().clone() for a, b in net.state_dict().items()}
        if ep - best_ep >= patience:
            break
    if state is None:
        state = net.state_dict()
    return state, {"ssl_loss": best, "ssl_best_epoch": best_ep, "ssl_epochs": ep + 1}


# ----------------------------------------------------------------------------- domain adaptation
class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def ppmi_view(g, steps: int = 3):
    """Global-consistency graph of UDAGCN: positive PMI of random-walk co-occurrence (cached per dataset)."""
    path = os.path.join(paths.CACHE, f"{g.name}_ppmi{steps}.npz")
    if os.path.exists(path):
        m = sp.load_npz(path)
    else:
        src, dst = g.ei.numpy()
        a = sp.csr_matrix((np.ones(len(src), np.float32), (dst, src)), shape=(g.n, g.n))
        deg = np.asarray(a.sum(1)).ravel()
        deg[deg == 0] = 1
        p = sp.diags(1.0 / deg) @ a
        acc, pk = p.copy(), p.copy()
        for _ in range(steps - 1):
            pk = pk @ p
            acc = acc + pk
        acc = (acc / steps).tocoo()
        total = acc.sum()
        row = np.asarray(acc.sum(1)).ravel()
        col = np.asarray(acc.sum(0)).ravel()
        pmi = np.log(acc.data * total / (row[acc.row] * col[acc.col]))
        keep = pmi > 0
        m = sp.csr_matrix((pmi[keep].astype(np.float32), (acc.row[keep], acc.col[keep])), shape=(g.n, g.n))
        m = ((m + m.T) * 0.5).tocsr()
        m = m + sp.identity(g.n, dtype=np.float32, format="csr")
        d = np.asarray(m.sum(1)).ravel()
        dinv = 1.0 / np.sqrt(d)
        m = (sp.diags(dinv) @ m @ sp.diags(dinv)).tocsr()
        os.makedirs(paths.CACHE, exist_ok=True)
        sp.save_npz(path, m)
    return View(g, A_gcn=data._csr_from_scipy(m))


def entropy(out, multilabel):
    if multilabel:
        p = torch.sigmoid(out)
        return -(p * torch.log(p + 1e-8) + (1 - p) * torch.log(1 - p + 1e-8)).mean()
    p = F.softmax(out, 1)
    return -(p * torch.log(p + 1e-8)).sum(1).mean()


def fit_da(method: str, model_name: str, cfg: dict, gs, ms, gt, mt, lr: float, seed: int,
           max_epochs: int = 1000, patience: int = 100, schedule_epochs: int = 200):
    """DANN or UDAGCN; returns the best-validation record on the target (same format as train.fit)."""
    if train.QUICK:
        max_epochs, patience = 5, 2
    torch.manual_seed(seed)
    net = models.build(model_name, cfg, gt.in_dim, gt.num_classes)
    h = int(cfg["hidden"])
    disc = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.5), nn.Linear(h, 1))
    mods = nn.ModuleList([net, disc])
    if method == "udagcn":
        att = nn.Linear(h, 1)
        mods.append(att)
        gs_glob, gt_glob = ppmi_view(gs), ppmi_view(gt)

        def emb(g, g_glob):
            z = torch.stack([net.embed(g), net.embed(g_glob)], 1)  # shared weights (local, global)
            a = F.softmax(att(z).squeeze(-1), 1)
            return (a.unsqueeze(-1) * z).sum(1)
    else:
        def emb(g, _):
            return net.embed(g)
        gs_glob = gt_glob = None
    opt = torch.optim.Adam(mods.parameters(), lr=lr, weight_decay=cfg["wd"])
    ml = gt.multilabel
    key = train.primary_name(ml)
    best = {"val": -1.0, "val_loss": math.inf, "epoch": -1}
    dom_y = torch.cat([torch.zeros(gs.n), torch.ones(gt.n)])
    ep = 0
    for ep in range(max_epochs):
        mods.train()
        opt.zero_grad(set_to_none=True)
        lam = 2.0 / (1.0 + math.exp(-10.0 * min(1.0, ep / schedule_epochs))) - 1.0
        zs, zt = emb(gs, gs_glob), emb(gt, gt_glob)
        out_s, out_t = net.classify(zs, gs), net.classify(zt, gt)
        loss = train.loss_fn(out_s[ms["train"]], gs.y[ms["train"]], ml) + \
            train.loss_fn(out_t[mt["train"]], gt.y[mt["train"]], ml)
        d = disc(_GRL.apply(torch.cat([zs, zt]), lam)).squeeze(-1)
        loss = loss + F.binary_cross_entropy_with_logits(d, dom_y)
        if method == "udagcn":
            loss = loss + min(1.0, ep / schedule_epochs) * entropy(out_t, ml)
        if not torch.isfinite(loss):
            best["diverged"] = True
            break
        loss.backward()
        opt.step()
        mods.eval()
        with torch.no_grad():
            out = net.classify(emb(gt, gt_glob), gt)
            vl = float(train.loss_fn(out[mt["val"]], gt.y[mt["val"]], ml))
            vm = train.metrics(out[mt["val"]], gt.y[mt["val"]], ml)
            if vm[key] > best["val"] + 1e-9 or (abs(vm[key] - best["val"]) <= 1e-9 and vl < best["val_loss"]):
                tm = train.metrics(out[mt["test"]], gt.y[mt["test"]], ml)
                best = {"val": vm[key], "val_loss": vl, "epoch": ep, "test": tm[key],
                        **{f"val_{a}": b for a, b in vm.items()}, **{f"test_{a}": b for a, b in tm.items()}}
        if ep - best["epoch"] >= patience:
            break
    best["epochs"] = ep + 1
    best.setdefault("test", float("nan"))
    return best
