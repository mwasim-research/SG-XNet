"""Configuration-swap test (post hoc, 25 Oct): is SG-XNet's robustness to fine-tuning a property of its
architecture or of its selected configuration? Same code (v2c) and rules as the main study; results kept apart."""
import os, sys, json, time, traceback
import multiprocessing as mp
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *(['..']*3))))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
UP = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), *(['..']*3))), 'results')

def make_jobs():
    sel = json.load(open(os.path.join(UP, 'selected_budget20.json')))
    cells = [  # (tag, model, cfg-from, source, target, budgets)
        ('sgx@gcncfg', 'sgx', ('dblpv7', 'gcn'), 'acmv9', 'dblpv7', [5, 20]),
        ('allgcn@sgxcfg', 'abl_allgcn', ('citationv1', 'sgx'), 'acmv9', 'citationv1', [1, 5]),
        ('sgx@gcncfg', 'sgx', ('citationv1', 'gcn'), 'acmv9', 'citationv1', [1, 5]),
        ('check_sgx@sgxcfg', 'sgx', ('dblpv7', 'sgx'), 'acmv9', 'dblpv7', [5]),   # reproduces a Mac cell
    ]
    pre, tgt = [], []
    for tag, m, (ct, cm), src, t, ks in cells:
        cfg = dict(sel[ct][cm])
        seeds = [0] if tag.startswith('check') else range(10)
        for s in seeds:
            ck = os.path.join(OUT, 'ckpt', f'{src}__{t}', tag, f'real_s{s}.pt')
            pre.append(dict(type='pretrain', source=src, target=t, model=m, cfg=cfg, seed=s, ptype='real',
                            parts=['encoder', 'body', 'out'], ckpt=ck, out=os.path.join(OUT, 'pretrain', f'{src}__{t}', tag, f'real_s{s}.json')))
            for k in ks:
                for c in ['zeroshot', 'finetune']:
                    tgt.append(dict(type=c if c == 'zeroshot' else 'target', source=src, target=t, model=m, cfg=cfg, cond=c, k=k, split=s,
                                    aligned=True, ckpt=ck, tag=tag, out=os.path.join(OUT, 'target', t, tag, f'{c}@{src}', f'k{k}_s{s}.json')))
    return pre, tgt

def run_one(job):
    os.environ['OMP_NUM_THREADS'] = '1'
    import torch; torch.set_num_threads(1)
    from sgx import jobs as J
    if os.path.exists(job['out']):
        return job['out'], 'skip', 0
    t0 = time.time()
    try:
        res = J.EXEC[job['type']](job)
        os.makedirs(os.path.dirname(job['out']), exist_ok=True)
        rec = {k: v for k, v in job.items()}
        if job['type'] == 'pretrain':
            rec.update(res)
        else:
            rec['result'] = res
        rec['seconds'] = round(time.time() - t0, 2)
        json.dump(rec, open(job['out'] + '.tmp', 'w'), default=str); os.replace(job['out'] + '.tmp', job['out'])
        return job['out'], 'ok', time.time() - t0
    except Exception:
        return job['out'], 'err ' + traceback.format_exc()[-400:], time.time() - t0

if __name__ == '__main__':
    pre, tgt = make_jobs()
    print('pretrain jobs', len(pre), 'target jobs', len(tgt), flush=True)
    with mp.get_context('spawn').Pool(2) as pool:
        for stage, js in [('pretrain', pre), ('target', tgt)]:
            n = 0
            for out, st, sec in pool.imap_unordered(run_one, js):
                n += 1
                if st != 'skip':
                    print(time.strftime('%H:%M:%S'), stage, n, '/', len(js), st[:300], os.path.relpath(out, OUT), f'{sec/60:.1f} min', flush=True)
    print('done', flush=True)
