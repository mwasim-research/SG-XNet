"""B1 (post hoc, 27 Oct 2026): how far do node representations move when a pretrained model is fine-tuned?
Linear CKA (Kornblith et al., 2019) between the embeddings (output of the transferable body) before and after
fine-tuning, on all target nodes; plus the relative change of the body weights. Same code (v2c), same splits, same
selected configurations and the learning rate the main study selected for that run."""
import os, sys, json, time, traceback
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *(['..']*2))))
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, 'b1_results')
UP = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), *(['..']*2))), 'results')
SW = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'cloud_2026-10-26', 'swap', 'results')

def cka(x, y):
    x = x - x.mean(0); y = y - y.mean(0)
    return float(np.linalg.norm(y.T @ x) ** 2 / (np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)))

def variants(sel, runs):
    v = []
    for t, ks, ms in [('citationv1', [1, 5], ['gcn', 'sage', 'gat', 'gin', 'mlp', 'sgx']), ('dblpv7', [5], ['gcn', 'sage', 'gin', 'mlp', 'sgx', 'abl_allgcn'])]:
        for m in ms:
            cfg = sel[t]['sgx' if m.startswith('abl_') else m]
            v.append((t, ks, m, m, cfg, os.path.join(UP, 'ckpt', f'acmv9__{t}', m), 'csv'))
        v.append((t, ks, 'sgx@gcncfg', 'sgx', sel[t]['gcn'], os.path.join(SW, 'ckpt', f'acmv9__{t}', 'sgx@gcncfg'), 'swap'))
        if t == 'citationv1':
            v.append((t, ks, 'allgcn@sgxcfg', 'abl_allgcn', sel[t]['sgx'], os.path.join(SW, 'ckpt', f'acmv9__{t}', 'allgcn@sgxcfg'), 'swap'))
    return v

def main():
    os.environ['OMP_NUM_THREADS'] = '1'
    import torch; torch.set_num_threads(1)
    from sgx import data, models, train, jobs as J
    sel = json.load(open(os.path.join(UP, 'selected_budget20.json')))
    runs = pd.read_csv(os.path.join(UP, 'analysis', 'target_runs_full.csv'))
    runs = runs[(runs.cond == 'finetune') & (runs.source == 'acmv9')]
    for t, ks, tag, m, cfg, ckdir, lrsrc in variants(sel, runs):
        g = data.load(t)
        for k in ks:
            for s in range(5):
                out = os.path.join(OUT, t, tag, f'k{k}_s{s}.json')
                if os.path.exists(out): continue
                t0 = time.time()
                try:
                    if lrsrc == 'csv':
                        r = runs[(runs.target == t) & (runs.model == m) & (runs.k == k) & (runs.split == s)]
                        lr, mac_test = float(r.lr.iloc[0]), float(r.test.iloc[0])
                    else:
                        d = json.load(open(os.path.join(SW, 'target', t, tag, 'finetune@acmv9', f'k{k}_s{s}.json')))
                        lr, mac_test = d['result']['sel']['lr'], 100 * d['result']['sel']['test']
                    masks = J._target_masks(g, dict(k=k, split=s))
                    J._seed(1000 + s)
                    net = models.build(m, cfg, g.in_dim, g.num_classes)
                    st0 = train.load_state(os.path.join(ckdir, f'real_s{s}.pt'))
                    train.transfer(net, st0, ['encoder', 'body', 'out'])
                    net.eval()
                    with torch.no_grad():
                        z0 = net.embed(g).numpy(); zs = train.evaluate(net, g, masks)['test']
                    w0 = {a: b.detach().clone() for a, b in net.state_dict().items() if a.startswith('body.')}
                    best, state = train.fit(net, g, masks, lr, cfg['wd'], max_epochs=1000, patience=100, keep_state=True)
                    if state is not None: net.load_state_dict(state)
                    net.eval()
                    with torch.no_grad():
                        z1 = net.embed(g).numpy()
                    w1 = {a: b for a, b in net.state_dict().items() if a.startswith('body.')}
                    num = sum(float(((w1[a] - w0[a]).float() ** 2).sum()) for a in w0 if w0[a].is_floating_point())
                    den = sum(float((w0[a].float() ** 2).sum()) for a in w0 if w0[a].is_floating_point())
                    rec = dict(target=t, tag=tag, model=m, k=k, split=s, lr=lr, cka=cka(z0, z1), wchange=(num / den) ** 0.5,
                               zs_test=100 * zs, ft_test=100 * best['test'], mac_ft_test=mac_test, epoch=best['epoch'],
                               seconds=round(time.time() - t0, 1))
                    os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(rec, open(out, 'w'))
                    print(time.strftime('%H:%M:%S'), t, tag, k, s, f"CKA {rec['cka']:.3f} dW {rec['wchange']:.3f} FT {rec['ft_test']:.1f} (Mac {mac_test:.1f})", f"{rec['seconds']/60:.1f} min", flush=True)
                except Exception:
                    print('ERR', out, traceback.format_exc()[-600:], flush=True)
    print('done', flush=True)

if __name__ == '__main__':
    main()
