import os
"""Numbers for the revised manuscript, computed from the result files only."""
import json, glob, os, numpy as np, pandas as pd
from scipy import stats
P=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); O=os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')), 'paper', 'output')
os.makedirs(O, exist_ok=True)
f=pd.read_csv(f'{P}/results/analysis/target_runs_full.csv'); b=pd.read_csv(f'{P}/results/analysis/target_runs_baselines.csv')
X=pd.concat([f,b]).drop_duplicates(['target','source','model','cond','k','split'])
MS=['gcn','sage','gat','gin','mlp','sgx']; NM={'gcn':'GCN','sage':'GraphSAGE','gat':'GAT','gin':'GIN','mlp':'MLP','sgx':'SG-XNet'}
AL=['acmv9','citationv1','dblpv7']; PAIRS=[(s,t) for t in AL for s in AL if s!=t]
bad=lambda m,s,t: (m=='gat' and s=='acmv9' and t=='dblpv7') or (m in('gat','sgx','sgxg') and s=='citeseer' and t=='chameleon_filtered')
I=X.set_index(['target','source','model','cond','k','split']).test.sort_index()
def unit(m,c,k,pairs=PAIRS):
    """per-split mean over valid pairs (scratch: over targets of those pairs)"""
    v=[]
    for s,t in pairs:
        if bad(m,s,t): continue
        key=(t,np.nan if c=='scratch' else s,m,c,k)
        try: q=X[(X.target==t)&(X.model==m)&(X.cond==c)&(X.k==k)&((X.source.isna()) if (c=='scratch' or '_self' in c) else (X.source==s))].set_index('split').test
        except KeyError: continue
        if len(q): v.append(q.sort_index())
    return pd.concat(v,axis=1).mean(axis=1) if v else None
out={}
# main aligned table
rows=[]
for m in MS:
  for k in [1,5,20]:
    r={'model':m,'k':k}
    for c in ['scratch','finetune','frozen','zeroshot']:
        u=unit(m,c,k); r[c]=(u.mean(),u.std(ddof=1))
    ft,zs=unit(m,'finetune',k),unit(m,'zeroshot',k); r['ftzs']=((ft-zs).mean(),stats.ttest_rel(ft,zs).pvalue)
    rows.append(r)
out['main']=rows
# SSL / DA (gcn, sgx) aligned
ssl=['ssl_dgi','ssl_grace','ssl_dgi_self','ssl_grace_self','ssl_dgi_probe','ssl_grace_probe','ssl_dgi_self_probe','ssl_grace_self_probe','dann','udagcn']
bl=[]
for m in ['gcn','sgx']:
  for k in [1,5,20]:
    r={'model':m,'k':k}
    for c in ['scratch','finetune','frozen']+ssl:
        u=unit(m,c,k); r[c]=None if u is None else u.mean()
    bl.append(r)
# sgx udagcn overnight
U=[json.load(open(p)) for p in glob.glob(f'{P}/results_posthoc_overnight/uda_sgx/**/*.json',recursive=True)]
ud=pd.DataFrame([dict(target=d['target'],source=d['source'],k=d['k'],split=d['split'],test=100*d['result']['sel']['test']) for d in U])
for r in bl:
    if r['model']=='sgx': r['udagcn']=ud[ud.k==r['k']].groupby('split').test.mean().mean()
out['baselines']=bl
json.dump(out,open(f'{O}/numbers.json','w'),default=float,indent=1)
# print compact
for r in rows: print(NM[r['model']],r['k'],' '.join(f"{c}={r[c][0]:.1f}±{r[c][1]:.1f}" for c in ['scratch','finetune','frozen','zeroshot']), f"ftzs={r['ftzs'][0]:+.1f} p={r['ftzs'][1]:.3g}")
for r in bl: print(r['model'],r['k'],{k:(None if v is None else round(v,1)) for k,v in r.items() if k not in('model','k')})
