"""B2 (post hoc, 27 Oct 2026): UDAGCN with the SG-XNet backbone. Same code (v2c) and protocol as the Mac's DA stage
(3 learning rates, selection on target validation, labelled target nodes of the split). For SG-XNet's GraphSAGE layer
the global (PPMI) view also replaces the mean-aggregation matrix with the row-normalised PPMI matrix."""
import os, sys, json, time, traceback
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *(['..']*2))))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'b2_results')
UP = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), *(['..']*2))), 'results')
PAIRS = [(s, t) for s in ['acmv9', 'citationv1', 'dblpv7'] for t in ['acmv9', 'citationv1', 'dblpv7'] if s != t]

def patch():
    import numpy as np, scipy.sparse as sp
    from sgx import baselines, data, paths
    orig = baselines.ppmi_view
    def view(g, steps=3):
        v = orig(g, steps)
        m = sp.load_npz(os.path.join(paths.CACHE, f"{g.name}_ppmi{steps}.npz"))
        d = np.asarray(m.sum(1)).ravel(); d[d == 0] = 1
        rm = (sp.diags(1.0 / d) @ m).tocsr()
        v.A_mean = data._csr_from_scipy(rm); v.A_mean_t = data._csr_from_scipy(rm.T)
        return v
    baselines.ppmi_view = view

def main():
    os.environ['OMP_NUM_THREADS'] = '1'
    import torch; torch.set_num_threads(1)
    patch()
    from sgx import jobs as J
    sel = json.load(open(os.path.join(UP, 'selected_budget20.json')))
    js = [dict(type='da', method='udagcn', source=s, target=t, model='sgx', cfg=sel[t]['sgx'], cond='udagcn', k=k, split=sp,
               aligned=True, out=os.path.join(OUT, t, 'sgx', f'udagcn@{s}', f'k{k}_s{sp}.json'))
          for t0 in ['dblpv7', 'acmv9', 'citationv1'] for sp in range(3) for k in [1, 5] for s, t in PAIRS if t == t0]
    if os.environ.get('REV'): js = js[::-1]
    print('jobs', len(js), flush=True)
    for i, job in enumerate(js):
        if os.path.exists(job['out']): continue
        t0 = time.time()
        try:
            res = J.EXEC['da'](job)
            os.makedirs(os.path.dirname(job['out']), exist_ok=True)
            rec = {**{k: v for k, v in job.items() if k != 'out'}, 'result': res, 'seconds': round(time.time() - t0, 1)}
            json.dump(rec, open(job['out'] + '.tmp', 'w'), default=str); os.replace(job['out'] + '.tmp', job['out'])
            print(time.strftime('%H:%M:%S'), i + 1, '/', len(js), job['source'], '->', job['target'], 'k', job['k'], 's', job['split'],
                  f"test {100*res['sel']['test']:.1f}", f'{(time.time()-t0)/60:.1f} min', flush=True)
        except Exception:
            print('ERR', job['out'], traceback.format_exc()[-600:], flush=True)
    print('done', flush=True)

if __name__ == '__main__':
    main()
