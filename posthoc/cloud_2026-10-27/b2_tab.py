import os
import json, glob, pandas as pd, numpy as np
from scipy import stats
U=os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), *(['..']*2))), 'results', 'analysis') + os.sep
x=pd.concat([pd.read_csv(U+'target_runs_full.csv'),pd.read_csv(U+'target_runs_baselines.csv')]).drop_duplicates(['target','source','model','cond','k','split'])
r=[json.load(open(p)) for p in glob.glob('b2_results/*/sgx/*/*.json')]
b=pd.DataFrame([dict(target=d['target'],source=d['source'],model='sgx',cond='udagcn_sgx',k=d['k'],split=d['split'],test=100*d['result']['sel']['test']) for d in r])
for t in sorted(b.target.unique()):
  for k in [1,5]:
    bb=b[(b.target==t)&(b.k==k)]; sp=sorted(bb.split.unique()); keys=bb.set_index(['source','split']).index
    def get(m,c):
        q=x[(x.target==t)&(x.k==k)&(x.model==m)&(x.cond==c)].set_index(['source','split']).test
        return q.reindex(keys).groupby(level='split').mean()
    sg=bb.set_index(['source','split']).test.groupby(level='split').mean()
    row={n:get(m,c) for n,(m,c) in {'SGX FT':('sgx','finetune'),'SGX DANN':('sgx','dann'),'GCN UDAGCN':('gcn','udagcn'),'GCN frozen':('gcn','frozen')}.items()}
    s=f"{t} k{k} (splits {sp}, n src-splits {len(bb)}): SGX-UDAGCN {sg.mean():.1f} | "+' | '.join(f"{n} {v.mean():.1f} (diff {np.mean(sg-v):+.1f}, p {stats.ttest_rel(sg,v).pvalue:.2f})" for n,v in row.items())
    print(s)
