import os, sys, numpy as np, pandas as pd
P=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); sys.path.insert(0,P)
import matplotlib; matplotlib.use('Agg')
from sgx import figures as F
OUT=P+'/results_posthoc_review3/figures'; F.out_dir=lambda: OUT
g=pd.read_csv(P+'/results/analysis/transfer_gains_full.csv'); g=g[g.cond=='finetune']
bad0={('acmv9','dblpv7','gat'),('citeseer','chameleon_filtered','gat'),('citeseer','chameleon_filtered','sgx')}
# Holm within each model over all pairs and the five budgets, with the failed-pretraining cells excluded from the family
g=g[[(s_,t_,m_) not in bad0 for s_,t_,m_ in zip(g.source,g.target,g.model)]].copy()
def _holm(p):
    p=np.asarray(p,float); o=np.argsort(p); m=len(p); out=np.empty(m); cur=0.0
    for i,j in enumerate(o): cur=max(cur,min(1.0,(m-i)*p[j])); out[j]=cur
    return out
for _m,_idx in g.groupby('model').groups.items(): g.loc[_idx,'p_holm']=_holm(g.loc[_idx,'p_t'].values)
models=['gcn','sage','gat','gin','mlp','sgx']
bad={('acmv9','dblpv7','gat'),('citeseer','chameleon_filtered','gat'),('citeseer','chameleon_filtered','sgx')}
pairs,breaks=[],[]
for _,ps,_ in F.GROUPS:
    ps=[p for p in ps if not g[(g.source==p[0])&(g.target==p[1])].empty]
    if ps:
        if pairs: breaks.append(len(pairs))
        pairs+=ps
print(pairs,breaks)
ks=[1,5,20]; I=g.set_index(['k','source','target','model'])
mats,stars,table=[],[],[]
for k in ks:
    M=np.full((len(pairs),len(models)),np.nan); S=[[False]*len(models) for _ in pairs]
    for i,(s,t) in enumerate(pairs):
        for j,m in enumerate(models):
            if (s,t,m) in bad or (k,s,t,m) not in I.index: continue
            r=I.loc[(k,s,t,m)]; M[i,j]=r.gain; S[i][j]=bool(r.p_holm<0.05)
            table.append(dict(k=k,source=s,target=t,model=m,gain=round(r.gain,2),p_t=r.p_t,p_holm=r.p_holm))
    mats.append(M); stars.append(S)
F._heatmap_panels(mats,[F.pair_label(s,t) for s,t in pairs],[F.SHORT[m] for m in models],ks,"k = {k} labels per class","gain of fine-tuning over scratch (points)","fig_gain_heatmap_finetune_full",table,stars,group_breaks=breaks)
T=pd.DataFrame(table); print('stars',sum(T.p_holm<0.05),'cells',len(T))
