import os
import warnings; warnings.filterwarnings('ignore')
import json, glob, os, numpy as np, pandas as pd
from scipy import stats
P=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
X=pd.concat([pd.read_csv(f'{P}/results/analysis/target_runs_full.csv'),pd.read_csv(f'{P}/results/analysis/target_runs_baselines.csv')]).drop_duplicates(['target','source','model','cond','k','split'])
MS=['gcn','sage','gat','gin','mlp','sgx']; AL=['acmv9','citationv1','dblpv7']; PAIRS=[(s,t) for t in AL for s in AL if s!=t]
bad={('acmv9','dblpv7','gat')}
I=X.set_index(['target','source','model','cond','k','split']).test.sort_index()
def holm(p):
    p=np.array(p); o=np.argsort(p); out=np.empty(len(p)); m=len(p); cur=0
    for i,j in enumerate(o): cur=max(cur,min(1,(m-i)*p[j])); out[j]=cur
    return out
def u(m,c,ks,pairs=PAIRS,splits=range(10)):
    v=[]
    for s,t in pairs:
        if (s,t,m) in bad: continue
        for k in ks: v.append(I.loc[(t,s,m,c,k)].loc[list(splits)])
    return pd.concat(v,axis=1).mean(axis=1)
print('## A: FT-ZS and FT-frozen, aligned, pooled per split')
for ks in [[1],[3],[5],[10],[20],[1,3,5]]:
    row=[]
    for m in MS:
        ft,zs,fz=u(m,'finetune',ks),u(m,'zeroshot',ks),u(m,'frozen',ks)
        row.append(f"{m}: FT-ZS {(ft-zs).mean():+.1f} FT-Fz {(ft-fz).mean():+.1f}")
    print(ks,' | '.join(row))
for ks in [[1],[5],[1,3,5]]:
    a=u('sgx','finetune',ks)-u('sgx','frozen',ks); ps=[];ds=[]
    for m in MS[:-1]:
        b=u(m,'finetune',ks)-u(m,'frozen',ks); ds.append((a-b).mean()); ps.append(stats.ttest_rel(a,b).pvalue)
    print('SGX-other FT-frozen',ks,[f"{m} {d:+.1f} p{h:.3f}" for m,d,h in zip(MS[:-1],ds,holm(ps))])
print('## B: selected epoch and lr, aligned finetune')
F=X[(X.target.isin(AL))&(X.source.isin(AL))&(X.cond=='finetune')&(X.model.isin(MS))&~((X.model=='gat')&(X.source=='acmv9')&(X.target=='dblpv7'))]  # main models, failed pretraining excluded
print(F.groupby(['model'])['epoch'].describe()[['50%','mean']].round(1).T.to_string())
print('share epoch==0 by model,k'); print(F.assign(e0=F.epoch==0).groupby(['model','k']).e0.mean().unstack().round(2).to_string())
print('lr share'); print(F.groupby('model').lr.value_counts(normalize=True).unstack().round(2).to_string())
print('## D: pre-stated test, all splits and splits 5-9')
def prestated(splits):
    res={}
    for t in AL:
        srcs=[s for s in AL if s!=t]; ps=[];ds=[]
        for m in MS[:-1]:
            use=[s for s in srcs if (s,t,m) not in bad]
            pr=[(s,t) for s in use]
            a=u('sgx','finetune',[1,3,5],pr,splits)-u('sgx','zeroshot',[1,3,5],pr,splits)
            b=u(m,'finetune',[1,3,5],pr,splits)-u(m,'zeroshot',[1,3,5],pr,splits)
            ds.append((a-b).mean()); ps.append(stats.ttest_rel(a,b).pvalue)
        h=holm(ps); res[t]=[(m,round(d,1),round(x,3)) for m,d,x in zip(MS[:-1],ds,h)]
    return res
for name,sp in [('all',range(10)),('5-9',range(5,10)),('0-4',range(5))]:
    r=prestated(sp); print(name)
    for t in AL: print('  ',t,r[t])
    met={m:sum(1 for t in AL for (mm,d,p) in r[t] if mm==m and d>0 and p<0.05) for m in MS[:-1]}; print('  targets met',met)
print('## pooled over targets per split (10 units) and Holm over 15')
allp=[]
for m in MS[:-1]:
    A=[];B=[]
    for t in AL:
        pr=[(s,t) for s in AL if s!=t and (s,t,m) not in bad]
        A.append(u('sgx','finetune',[1,3,5],pr)-u('sgx','zeroshot',[1,3,5],pr)); B.append(u(m,'finetune',[1,3,5],pr)-u(m,'zeroshot',[1,3,5],pr))
    a=pd.concat(A,axis=1).mean(axis=1); b=pd.concat(B,axis=1).mean(axis=1)
    print(m,'pooled10',round((a-b).mean(),1),f"{stats.ttest_rel(a,b).pvalue:.2g}")
res=prestated(range(10)); flat=[(t,m,d) for t in AL for (m,d,p) in res[t]]
raw=[]
for t in AL:
    srcs=[s for s in AL if s!=t]
    for m in MS[:-1]:
        pr=[(s,t) for s in srcs if (s,t,m) not in bad]
        a=u('sgx','finetune',[1,3,5],pr)-u('sgx','zeroshot',[1,3,5],pr); b=u(m,'finetune',[1,3,5],pr)-u(m,'zeroshot',[1,3,5],pr)
        raw.append(stats.ttest_rel(a,b).pvalue)
h=holm(raw); 
met={m:sum(1 for (t,mm,d),p in zip(flat,h) if mm==m and d>0 and p<0.05) for m in MS[:-1]}; print('Holm over 15, targets met',met)
