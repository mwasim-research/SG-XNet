import json, glob, os, numpy as np, pandas as pd
from scipy import stats
P=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); N=P+'/results_posthoc_night2'
X=pd.concat([pd.read_csv(f'{P}/results/analysis/target_runs_full.csv'),pd.read_csv(f'{P}/results/analysis/target_runs_baselines.csv')]).drop_duplicates(['target','source','model','cond','k','split'])
MS=['gcn','sage','gat','gin','mlp','sgx']
# pretrain source val
pr=[json.load(open(p)) for p in glob.glob(N+'/e1/pretrain/**/*.json',recursive=True)]
print('pretrain keys',list(pr[0].keys()), list(pr[0].get('result',{}).keys())[:12])
R=[]
for p in glob.glob(N+'/e1/target/**/*.json',recursive=True):
    d=json.load(open(p)); R.append(dict(target=d['target'],source=d.get('source'),model=d['model'],cond=d['cond'],k=d['k'],split=d['split'],test=100*d['result']['sel']['test']))
E=pd.DataFrame(R); E.to_csv(os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')), 'results_posthoc_night2', 'analysis', 'e1_runs.csv'),index=False)
print(E.groupby(['cond','k']).size())
print(E[E.cond=='scratch'].source.unique())
S=pd.DataFrame([dict(source=d['source'],target=d['target'],model=d['model'],seed=d['seed'],val=100*d['val']) for d in pr])
sv=S.groupby(['source','target','model']).val.mean().unstack('model').round(1); print(sv)
best=sv.max(axis=1); bad=set((s,t,m) for (s,t),r in sv.iterrows() for m in MS if r[m]<0.5*best[(s,t)]); print('failed',bad)
AL=['acmv9','citationv1','dblpv7']; PAIRS=[(s,t) for t in AL for s in AL if s!=t]
I=E.set_index(['target','source','model','cond','k','split']).test
Xm=X.set_index(['target','source','model','cond','k','split']).test
MAINBAD={('acmv9','dblpv7','gat')}  # main-study exclusion (failed pretraining), used for the main-study columns
def unit(src,m,c,k,pairs=PAIRS,excl=None):
    v=[]
    for s,t in pairs:
        if (s,t,m) in (bad if excl is None else excl): continue
        v.append(src.loc[(t,s,m,c,k)].sort_index())
    return pd.concat(v,axis=1).mean(axis=1)
rows=[]
for k in [1,5]:
  for m in MS:
    sc,ft,fz,zs=[unit(I,m,c,k) for c in ['scratch','finetune','frozen','zeroshot']]
    mft,mzs=unit(Xm,m,'finetune',k,excl=MAINBAD),unit(Xm,m,'zeroshot',k,excl=MAINBAD)
    nb=sum(I.loc[(t,s,m,'finetune',k)].mean()<I.loc[(t,s,m,'zeroshot',k)].mean() for s,t in PAIRS if (s,t,m) not in bad)
    rows.append(dict(k=k,model=m,scratch=sc.mean(),FT=ft.mean(),frozen=fz.mean(),ZS=zs.mean(),FT_ZS=(ft-zs).mean(),p=stats.ttest_rel(ft,zs).pvalue,
      FT_scr=(ft-sc).mean(),fz_scr=(fz-sc).mean(),ZS_scr=(zs-sc).mean(),pairsFT_lt_ZS=nb,main_FT=mft.mean(),main_ZS=mzs.mean(),main_FT_ZS=(mft-mzs).mean()))
T=pd.DataFrame(rows); pd.set_option('display.width',250); print(T.round(2).to_string())
T.to_csv(os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')), 'results_posthoc_night2', 'analysis', 'e1_table.csv'),index=False)
print('per pair FT-ZS')
pp=[]
for k in [1,5]:
  for s,t in PAIRS:
    r={'k':k,'pair':f'{s}->{t}'}
    for m in MS: r[m]=np.nan if (s,t,m) in bad else I.loc[(t,s,m,'finetune',k)].mean()-I.loc[(t,s,m,'zeroshot',k)].mean()
    pp.append(r)
print(pd.DataFrame(pp).round(1).to_string())
# SGX vs others FT-ZS paired
for k in [1,5]:
  a=unit(I,'sgx','finetune',k)-unit(I,'sgx','zeroshot',k)
  for m in MS[:-1]:
    b=unit(I,m,'finetune',k)-unit(I,m,'zeroshot',k); print('k',k,'sgx vs',m,'diff',round((a-b).mean(),2),'p',round(stats.ttest_rel(a,b).pvalue,4))
# best approach among frozen vs ZS vs FT
