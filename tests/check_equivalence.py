"""Checks that the matrix-based propagation in sgx.models equals PyG's message passing (outputs and gradients).

Run from the project folder:  python -m tests.check_equivalence
"""
import torch

from sgx import data, models

torch.manual_seed(0)
worst = 0.0
for name in ["cora", "acmv9", "chameleon_filtered", "amazon_ratings"]:
    g = data.load(name)
    z = torch.randn(g.n, 32, requires_grad=True)
    for kind, layer in [("gcn", models._GCN(32, 32)), ("sage", models._SAGE(32, 32, aggr="mean")),
                        ("gin", models._GIN(32, 32)), ("sagemax", models._SAGE(32, 32, aggr="max")),
                        ("sagemax-relu", models._SAGE(32, 32, aggr="max"))]:
        # "sagemax-relu": input after ReLU, so many neighbours tie at zero (as in every layer after the first);
        # gradients are compared with respect to the pre-activation input z, i.e. what training actually uses
        h = torch.relu(z) if kind.endswith("relu") else z
        ours = layer(h, g)
        if kind == "gcn":
            ref = layer.conv(h, g.gcn_ei, g.gcn_ew)
        else:
            ref = layer.conv(h, g.ei)
        d_out = float((ours - ref).abs().max())
        go = torch.autograd.grad(ours.pow(2).sum(), [z] + list(layer.parameters()), retain_graph=True)
        gr = torch.autograd.grad(ref.pow(2).sum(), [z] + list(layer.parameters()))
        d_grad = max(float((a - b).abs().max() / b.abs().max().clamp(min=1e-12)) for a, b in zip(go, gr))
        worst = max(worst, d_out, d_grad)
        print(f"{name:20s} {kind:12s} max |out diff| = {d_out:.2e}   relative grad diff = {d_grad:.2e}")
    if g.sparse_x:
        enc = models.SparseLinear(g.in_dim, 16)
        a = enc(g.x, g.xt)
        b = enc(g.x)
        ga, = torch.autograd.grad(a.pow(2).sum(), enc.weight)
        gb, = torch.autograd.grad(b.pow(2).sum(), enc.weight)
        d = max(float((a - b).abs().max()), float((ga - gb).abs().max() / gb.abs().max()))
        worst = max(worst, d)
        print(f"{name:20s} sparse encoder with pre-computed transpose: max diff = {d:.2e}")
print("OK" if worst < 1e-4 else "MISMATCH", f"(worst {worst:.2e})")
