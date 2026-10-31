import os
import pandas as pd, numpy as np, json, sys, warnings; warnings.filterwarnings('ignore')
from scipy import stats
U=os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), *(['..']*2))), 'results') + os.sep
f=pd.read_csv(U+'analysis/target_runs_full.csv'); b=pd.read_csv(U+'analysis/target_runs_baselines.csv')
AL=['acmv9','citationv1','dblpv7']
x=pd.concat([f,b]).drop_duplicates(['target','source','model','cond','k','split'])
x=x[x.target.isin(AL)&x.k.isin([1,5,20])&(x.model.isin(['gcn','sgx','sage','gat','gin','mlp']))]
x=x[~((x.model=='gat')&(x.source=='acmv9')&(x.target=='dblpv7'))]
x['app']=x.model+':'+x.cond
x=x[x.cond.isin(['scratch','finetune','frozen','zeroshot','ssl_grace_self','ssl_dgi_self','dann','udagcn'])]
# per (target,k): mean over splits and sources -> block
p=x.groupby(['target','k','app']).test.mean().unstack('app')
p=p.dropna(axis=1)
rk=p.rank(axis=1,ascending=False).mean().sort_values()
N,K=p.shape; cd=stats.studentized_range.ppf(0.95,K,np.inf)/np.sqrt(2)*np.sqrt(K*(K+1)/(6*N))
print(f'approaches on aligned targets: N={N} blocks (target x k), K={K}, CD={cd:.1f}'); print(rk.head(12).round(1).to_string())
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *(['..']*2))))
from sgx import models
sel=json.load(open(U+'selected_budget20.json')); ds={d['name']:d for d in json.load(open(U+'datasets.json'))}
rows=[]
for t in ['acmv9','citationv1','dblpv7','cora','pubmed','chameleon_filtered','roman_empire']:
    r={'target':t}
    for m in ['gcn','sage','gat','gin','mlp','sgx']:
        c=sel[t][m]; net=models.build(m,c,ds[t]['features'],ds[t]['classes'])
        r[m]=f"{sum(p.numel() for n,p in net.named_parameters() if not n.startswith('encoder'))/1e3:.0f}k (+{net.encoder.weight.numel()/1e3:.0f}k enc) K{c['K']} h{c['hidden']}"
    rows.append(r)
print(pd.DataFrame(rows).to_markdown(index=False))
