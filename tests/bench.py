"""Speed benchmark of the input projection and propagation on this machine (single thread).
Writes results/bench.txt. Run:  python -m tests.bench"""
import os
import platform
import time

import torch
import torch.nn.functional as F

from sgx import data, models, paths

torch.set_num_threads(1)
lines = [f"platform {platform.platform()} | torch {torch.__version__} | mkl {torch.backends.mkl.is_available()} | "
         f"threads {torch.get_num_threads()}"]


def bench(fn, reps=8):
    for _ in range(2):
        fn()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    return 1000 * (time.perf_counter() - t) / reps


for name in ["pubmed", "acmv9", "cora", "roman_empire"]:
    g = data.load(name)
    H = 256
    W = torch.nn.Parameter(torch.randn(H, g.in_dim) * 0.01)
    res = {}
    if g.sparse_x:
        def csr():
            W.grad = None
            models.SpMM.apply(g.x, g.xt, W.t()).sum().backward()
        cols, offs, vals = g.x.col_indices(), g.x.crow_indices()[:-1].contiguous(), g.x.values()
        Wt = torch.nn.Parameter(W.detach().t().contiguous())

        def ebag():
            Wt.grad = None
            F.embedding_bag(cols, Wt, offs, mode="sum", per_sample_weights=vals).sum().backward()
        res["encoder csr"] = bench(csr)
        res["encoder embedding_bag"] = bench(ebag)
        if g.n * g.in_dim < 80_000_000:
            xd = g.x.to_dense()

            def dense():
                W.grad = None
                F.linear(xd, W).sum().backward()
            res["encoder dense"] = bench(dense)
    else:
        xd = g.x

        def dense():
            W.grad = None
            F.linear(xd, W).sum().backward()
        res["encoder dense"] = bench(dense)
    h = torch.randn(g.n, H, requires_grad=True)
    res["gcn propagate"] = bench(lambda: models.SpMM.apply(g.A_gcn, g.A_gcn, h).sum().backward())
    res["dense linear h->h"] = bench(lambda: F.linear(h, torch.randn(H, H)).sum().backward())
    lines.append(f"{name:13s} " + " | ".join(f"{k} {v:.1f} ms" for k, v in res.items()))
    print(lines[-1], flush=True)

os.makedirs(paths.RESULTS, exist_ok=True)
with open(os.path.join(paths.RESULTS, "bench.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("written", os.path.join(paths.RESULTS, "bench.txt"))
