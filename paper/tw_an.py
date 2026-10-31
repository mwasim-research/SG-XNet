import os
os.chdir(os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')), 'results_posthoc_twitch'))
import json, glob, numpy as np, pandas as pd
from scipy import stats
R=[]
for p in glob.glob('target/*/*/*/*.json'):
    d=json.load(open(p)); R.append(dict(target=d['target'],source=d.get('source'),model=d['model'],cond=p.split('/')[3].split('@')[0],k=d['k'],split=d['split'],test=100*d['result']['sel']['test']))
D=pd.DataFrame(R); D['source']=D.source.fillna('-')
MS=['gcn','sage','gat','gin','mlp','sgx']
sc=D[D.cond=='scratch'].set_index(['target','model','k','split']).test
X=D[D.cond!='scratch'].copy(); X['sc']=[sc[(r.target,r.model,r.k,r.split)] for r in X.itertuples()]
zs=D[D.cond=='zeroshot'].set_index(['target','source','model','k','split']).test
print('counts', D.groupby('cond').size().to_dict())
print('\n## mean over 6 pairs x 10 splits (n per model)')
for k in [1,5]:
  rows=[]
  for m in MS:
    q=X[(X.model==m)&(X.k==k)]; r={'model':m,'k':k}
    r['scratch']=sc.xs((m,k),level=('model','k')).mean()
    for c in ['finetune','frozen','zeroshot','l2sp']:
        qc=q[q.cond==c]
        if len(qc)==0: continue
        r[c]=qc.test.mean(); r[c+'_gain']=(qc.test-qc.sc).mean(); r[c+'_p']=stats.ttest_rel(qc.test,qc.sc).pvalue; r[c+'_n']=len(qc)
    ft=q[q.cond=='finetune']; z=np.array([zs[(a.target,a.source,m,k,a.split)] for a in ft.itertuples()])
    r['FT-ZS']=(ft.test.values-z).mean(); r['FT-ZS_p']=stats.ttest_rel(ft.test.values,z).pvalue
    l=q[q.cond=='l2sp']
    if len(l): 
        zl=np.array([zs[(a.target,a.source,m,k,a.split)] for a in l.itertuples()]); fl=np.array([X[(X.cond=='finetune')&(X.target==a.target)&(X.source==a.source)&(X.model==m)&(X.k==k)&(X.split==a.split)].test.iloc[0] for a in l.itertuples()])
        r['L2SP-ZS']=(l.test.values-zl).mean(); r['L2SP-FT']=(l.test.values-fl).mean(); r['L2SP-FT_p']=stats.ttest_rel(l.test.values,fl).pvalue
    rows.append(r)
  print(pd.DataFrame(rows).round(3).to_string())
print('\n## per pair FT gain (mean over models), k1/k5')
g=X[X.cond=='finetune'].groupby(['source','target','k']).apply(lambda q:(q.test-q.sc).mean()).unstack('k'); print(g.round(2))
print('\n## per target best approach (mean over sources)')
for t in sorted(D.target.unique()):
  for k in [1,5]:
    q=D[(D.target==t)&(D.k==k)].groupby(['model','cond']).test.mean().sort_values(ascending=False); print(t,k,q.head(4).round(1).to_dict())
# pre-stated-like test: SGX FT-ZS minus baseline FT-ZS, k1&5 pooled, unit split
print('\n## SG-XNet FT-ZS minus baseline FT-ZS (k=1,5; mean over pairs; unit=split)')
def u(m):
    ft=X[(X.model==m)&(X.cond=='finetune')]
    v=ft.test.values-np.array([zs[(a.target,a.source,m,a.k,a.split)] for a in ft.itertuples()])
    return pd.Series(v,index=pd.MultiIndex.from_frame(ft[['target','source','k','split']])).groupby(level='split').mean()
s=u('sgx')
for m in ['gcn','sage','gat','gin','mlp']:
    b=u(m); print(m, round((s-b).mean(),2), 'p', round(stats.ttest_rel(s,b).pvalue,4))
# rank of fine-tuned models
p=X[X.cond=='finetune'].groupby(['target','source','k','model']).test.mean().unstack('model')[MS]
print('\nFT avg rank', p.rank(axis=1,ascending=False).mean().round(2).to_dict(), 'Friedman p', stats.friedmanchisquare(*[p[c] for c in MS]).pvalue)
