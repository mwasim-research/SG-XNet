"""Training, evaluation and the transfer protocol.

Model selection always uses the validation set: training stops when the validation metric has not improved for
`patience` epochs and the test metrics are those of the epoch with the best validation metric (ties broken by
validation loss). Test labels are never used for any decision.
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F

QUICK = os.environ.get("SGX_QUICK") == "1"  # smoke test: a handful of epochs only


def loss_fn(out, y, multilabel):
    return F.binary_cross_entropy_with_logits(out, y) if multilabel else F.cross_entropy(out, y)


@torch.no_grad()
def metrics(out, y, multilabel) -> dict:
    C = out.shape[1]
    if not multilabel:
        pred = out.argmax(1)
        cm = torch.bincount(y * C + pred, minlength=C * C).view(C, C).float()  # rows: true, cols: predicted
        tp = cm.diag()
        fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
        present = (cm.sum(0) + cm.sum(1)) > 0
        f1 = 2 * tp / (2 * tp + fp + fn).clamp(min=1e-12)
        return {"acc": float((pred == y).float().mean()), "macro_f1": float(f1[present].mean())}
    # multi-label: predict the top-k labels, k = number of true labels of the node (ACDNE / UDAGCN protocol)
    k = y.sum(1).long()
    order = out.argsort(1, descending=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(C).expand_as(order).contiguous())
    pred = (ranks < k[:, None]).float()
    tp, fp, fn = (pred * y).sum(0), (pred * (1 - y)).sum(0), ((1 - pred) * y).sum(0)
    micro = 2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum()).clamp(min=1e-12)
    present = (y.sum(0) + pred.sum(0)) > 0
    macro = (2 * tp / (2 * tp + fp + fn).clamp(min=1e-12))[present].mean()
    return {"micro_f1": float(micro), "macro_f1": float(macro)}


def primary_name(multilabel: bool) -> str:
    return "micro_f1" if multilabel else "acc"


def fit(model, g, masks, lr, wd, max_epochs=1000, patience=100, params=None, keep_state=False):
    if QUICK:
        max_epochs, patience = min(max_epochs, 5), 2
    tr, va, te = masks["train"], masks["val"], masks["test"]
    ml = g.multilabel
    key = primary_name(ml)
    params = params if params is not None else [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=wd)
    best = {"val": -1.0, "val_loss": math.inf, "epoch": -1}
    state = None
    ep = 0
    for ep in range(max_epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        out = model(g)
        loss = loss_fn(out[tr], g.y[tr], ml)
        if not torch.isfinite(loss):
            best["diverged"] = True
            break
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            out = model(g)
            vl = float(loss_fn(out[va], g.y[va], ml))
            vm = metrics(out[va], g.y[va], ml)
            v = vm[key]
            if v > best["val"] + 1e-9 or (abs(v - best["val"]) <= 1e-9 and vl < best["val_loss"]):
                tm = metrics(out[te], g.y[te], ml)
                best = {"val": v, "val_loss": vl, "epoch": ep, "train_loss": float(loss),
                        **{f"val_{a}": b for a, b in vm.items()}, **{f"test_{a}": b for a, b in tm.items()}}
                best["test"] = tm[key]
                if keep_state:
                    state = {a: b.detach().clone() for a, b in model.state_dict().items()}
        if ep - best["epoch"] >= patience:
            break
    best["epochs"] = ep + 1
    best.setdefault("test", float("nan"))
    return best, state


@torch.no_grad()
def evaluate(model, g, masks) -> dict:
    model.eval()
    out = model(g)
    ml = g.multilabel
    key = primary_name(ml)
    vm, tm = metrics(out[masks["val"]], g.y[masks["val"]], ml), metrics(out[masks["test"]], g.y[masks["test"]], ml)
    return {"val": vm[key], "test": tm[key], **{f"val_{a}": b for a, b in vm.items()},
            **{f"test_{a}": b for a, b in tm.items()}}


# ----------------------------------------------------------------------------- transfer helpers
def transfer(model, state: dict, parts) -> list[str]:
    """Copy the named parts ('encoder', 'body', 'out') of a pretrained state into model. Returns copied keys."""
    own = model.state_dict()
    copied = []
    for k, v in state.items():
        if k.split(".")[0] in parts:
            if k not in own or own[k].shape != v.shape:
                raise ValueError(f"cannot transfer {k}: shape {tuple(v.shape)} vs {tuple(own.get(k, torch.empty(0)).shape)}")
            own[k] = v.float()
            copied.append(k)
    model.load_state_dict(own)
    return copied


def freeze(model, parts):
    for name, p in model.named_parameters():
        if name.split(".")[0] in parts:
            p.requires_grad_(False)


def freeze_prefixes(model, prefixes):
    for name, p in model.named_parameters():
        if any(name.startswith(x) for x in prefixes):
            p.requires_grad_(False)


def unfreeze(model):
    for p in model.parameters():
        p.requires_grad_(True)


@torch.no_grad()
def wise_select(model, g, masks, pre: dict, ft: dict, parts, alphas=(0.0, 0.25, 0.5, 0.75, 1.0)) -> dict:
    """WiSE-FT (Wortsman et al., 2022): theta = (1 - a) * pretrained + a * fine-tuned for the transferred parts
    (the others keep their fine-tuned values); a is chosen on validation (ties: larger a, i.e. closer to the
    fine-tuned model). The model is left at the selected weights."""
    cands = []
    for a in alphas:
        mix = {k: ((1 - a) * pre[k].float() + a * v) if (k.split(".")[0] in parts and k in pre and v.is_floating_point())
               else v for k, v in ft.items()}
        model.load_state_dict(mix)
        model.eval()
        out = model(g)
        r = evaluate(model, g, masks)
        r["val_loss"] = float(loss_fn(out[masks["val"]], g.y[masks["val"]], g.multilabel))
        r["alpha"] = a
        cands.append((r, mix))
    best, mix = max(cands, key=lambda c: (c[0]["val"], -c[0]["val_loss"], c[0]["alpha"]))
    model.load_state_dict(mix)
    return best


def save_state(state: dict, parts, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keep = {}
    for k, v in state.items():
        if k.split(".")[0] in parts:
            keep[k] = v.half() if v.is_floating_point() and v.numel() > 100_000 else v  # big encoders in fp16
    tmp = path + ".tmp"
    torch.save(keep, tmp)
    os.replace(tmp, path)


def load_state(path: str) -> dict:
    return {k: v.float() if v.is_floating_point() else v for k, v in torch.load(path, map_location="cpu").items()}
