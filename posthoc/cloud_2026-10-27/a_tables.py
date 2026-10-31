import os
import pandas as pd, numpy as np
from scipy import stats
U=os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), *(['..']*2))), 'results', 'analysis') + os.sep
f=pd.read_csv(U+'target_runs_full.csv'); b=pd.read_csv(U+'target_runs_baselines.csv')
MAIN=['gcn','sage','gat','gin','mlp','sgx']; f=f[f.model.isin(MAIN)]
AL={'acmv9','citationv1','dblpv7'}
bad=lambda r: (r.model=='gat' and r.source=='acmv9' and r.target=='dblpv7') or (r.model in('gat','sgx') and r.source=='citeseer' and r.target=='chameleon_filtered')
f=f[~f.apply(bad,axis=1)]
f['aligned']=f.target.isin(AL)&f.source.fillna('').isin(AL)
sc=f[f.cond=='scratch'].groupby(['target','model','k','split']).test.mean()
def cellmeans(cond):
    x=f[f.cond==cond].copy(); x['sc']=[sc.get((t,m,k,s),np.nan) for t,m,k,s in zip(x.target,x.model,x.k,x.split)]
    return x
print('## A2 negative-transfer rate (cells = pair x k; mean over 10 splits; % of cells below reference; sig = paired t p<0.05)')
rows=[]
for m in MAIN:
    r={'model':m}
    for cond in ['finetune','frozen']:
        x=cellmeans(cond); x=x[x.model==m]
        for al,name in [(True,'aligned'),(False,'other')]:
            g=x[x.aligned==al].groupby(['target','source','k'])
            d=g.apply(lambda q:(q.test-q.sc).mean()); p=g.apply(lambda q: stats.ttest_rel(q.test,q.sc).pvalue if len(q)>1 else np.nan)
            r[f'{cond}<scratch {name}']=f"{100*(d<0).mean():.0f}% ({100*((d<0)&(p<0.05)).mean():.0f}% sig) n{len(d)}"
    ft=f[(f.cond=='finetune')&(f.model==m)&f.aligned].set_index(['target','source','k','split']).test
    zs=f[(f.cond=='zeroshot')&(f.model==m)].set_index(['target','source','k','split']).test
    j=pd.concat([ft,zs],axis=1,keys=['ft','zs']).dropna().groupby(level=[0,1,2])
    d=j.apply(lambda q:(q.ft-q.zs).mean()); p=j.apply(lambda q: stats.ttest_rel(q.ft,q.zs).pvalue)
    r['finetune<zeroshot aligned']=f"{100*(d<0).mean():.0f}% ({100*((d<0)&(p<0.05)).mean():.0f}% sig) n{len(d)}"
    rows.append(r)
print(pd.DataFrame(rows).to_markdown(index=False))
print('\n## A4 spread over splits (mean SD over cells), aligned and other')
x=f.groupby(['target','source','model','cond','k','aligned']).test.std().reset_index()
x.loc[x.cond=='scratch','aligned']=x.loc[x.cond=='scratch','target'].isin(AL)
print(x[x.cond.isin(['scratch','finetune','frozen','zeroshot'])].groupby(['cond','aligned','model']).test.mean().unstack('model')[MAIN].round(2).to_markdown())
# paired: SD of FT vs SD of scratch per cell
print('\n## A5 macro-F1: FT - ZS (aligned, mean over 30 cells) and fine-tuned macro-F1 vs SG-XNet')
for m in MAIN:
    ft=f[(f.cond=='finetune')&(f.model==m)&f.aligned].set_index(['target','source','k','split']).macro_f1
    zs=f[(f.cond=='zeroshot')&(f.model==m)].set_index(['target','source','k','split']).macro_f1
    j=pd.concat([ft,zs],axis=1,keys=['ft','zs']).dropna(); dd=(j.ft-j.zs).groupby(level=[0,1,2]).mean()
    mi=f[(f.cond=='finetune')&(f.model==m)&f.aligned].set_index(['target','source','k','split']).test
    mz=f[(f.cond=='zeroshot')&(f.model==m)].set_index(['target','source','k','split']).test
    jm=pd.concat([mi,mz],axis=1,keys=['ft','zs']).dropna(); dm=(jm.ft-jm.zs).groupby(level=[0,1,2]).mean()
    print(f"{m}: macroF1 FT-ZS mean {dd.mean():+.1f} worst {dd.min():+.1f} below0 {int((dd<0).sum())}/{len(dd)} | microF1 mean {dm.mean():+.1f}; FT macro-F1 mean {j.ft.mean():.1f}")
# pre-stated test with macro F1 (k1,3,5; unit split; mean over sources,k)
print('\n## A5b pre-stated test with macro-F1 (SG-XNet FT-ZS minus baseline FT-ZS, k=1,3,5)')
def ftzs(m,col):
    ft=f[(f.cond=='finetune')&(f.model==m)&f.aligned&f.k.isin([1,3,5])].set_index(['target','source','k','split'])[col]
    zs=f[(f.cond=='zeroshot')&(f.model==m)&f.k.isin([1,3,5])].set_index(['target','source','k','split'])[col]
    j=pd.concat([ft,zs],axis=1,keys=['ft','zs']).dropna(); return (j.ft-j.zs)
for m in ['gcn','sage','gat','gin','mlp']:
    out=[]
    for t in ['acmv9','citationv1','dblpv7']:
        a=ftzs('sgx','macro_f1'); c=ftzs(m,'macro_f1')
        a=a.xs(t,level=0); c=c.xs(t,level=0); idx=a.index.intersection(c.index)
        ua=a.loc[idx].groupby(level='split').mean(); uc=c.loc[idx].groupby(level='split').mean()
        out.append(f"{t} {np.mean(ua-uc):+.1f} (p {stats.ttest_rel(ua,uc).pvalue:.3f})")
    print(m, ' | '.join(out))
print('\n## A3 average rank of fine-tuned models (blocks = pair x k, complete blocks only; 1 = best)')
for al,name in [(True,'aligned'),(False,'other'),(None,'all')]:
    x=f[f.cond=='finetune']; x=x if al is None else x[x.aligned==al]
    p=x.groupby(['target','source','k','model']).test.mean().unstack('model')[MAIN].dropna()
    rk=p.rank(axis=1,ascending=False); N,K=p.shape
    fr=stats.friedmanchisquare(*[p[c] for c in MAIN]).pvalue
    q=stats.studentized_range.ppf(0.95,K,np.inf)/np.sqrt(2); cd=q*np.sqrt(K*(K+1)/(6*N))
    print(name, f'N={N} Friedman p={fr:.2g} CD={cd:.2f}', rk.mean().round(2).to_dict())
