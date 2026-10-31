import os, sys, json, time
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')))
from sgx import jobs
UP = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')), 'results')
sel = json.load(open(os.path.join(UP, 'selected_budget20.json')))
mac = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mac_values.json')))
VAR = ['sgx', 'sgx_order', 'sgx_last', 'sgx_sage2', 'sgx_nodrop', 'abl_allgcn', 'abl_allsage']
zs = {}; t0 = time.time()
for m in VAR:
    cfg = jobs._cfg_for(sel, 'dblpv7', m)
    for k in [5, 20]:
        for s in range(10):
            job = dict(type='zeroshot', source='acmv9', target='dblpv7', model=m, cfg=cfg, cond='zeroshot', k=k, split=s,
                       aligned=True, out='/dev/null', ckpt=os.path.join(UP, 'ckpt', 'acmv9__dblpv7', m, f'real_s{s}.pt'))
            zs[(m, k, s)] = 100 * jobs.exec_zeroshot(job)['sel']['test']
print('evaluations', len(zs), 'in %.0f s' % (time.time() - t0))
dev = [abs(zs[('sgx', k, s)] - mac[f'sgx|zeroshot|{k}|{s}']) for k in [5, 20] for s in range(10)]
print('sanity: |cloud ZS - Mac ZS| for SG-XNet: max %.4f, mean %.4f points' % (max(dev), np.mean(dev)))
json.dump({f'{m}|{k}|{s}': v for (m, k, s), v in zs.items()}, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'zs_cloud.json'), 'w'))
print('\nk | model | ZS | FT | FT-ZS | (FT-ZS) - SG-XNet(FT-ZS) [p, splits better]')
for k in [5, 20]:
    base = np.array([mac[f'sgx|finetune|{k}|{s}'] - zs[('sgx', k, s)] for s in range(10)])
    for m in VAR:
        z = np.array([zs[(m, k, s)] for s in range(10)]); f = np.array([mac[f'{m}|finetune|{k}|{s}'] for s in range(10)])
        dd = f - z
        if m == 'sgx':
            print(f'k{k} {m:11s} ZS {z.mean():.1f} FT {f.mean():.1f} FT-ZS {dd.mean():+.1f}')
        else:
            x = dd - base; p = stats.ttest_1samp(x, 0).pvalue
            print(f'k{k} {m:11s} ZS {z.mean():.1f} FT {f.mean():.1f} FT-ZS {dd.mean():+.1f}  vs SG-XNet {x.mean():+.1f} (p {p:.3f}, {int((x > 0).sum())}/10 better)')
