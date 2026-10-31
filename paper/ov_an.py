import json, glob, os, numpy as np, pandas as pd
os.chdir(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from scipy import stats
O='results_posthoc_overnight'
f=pd.concat([pd.read_csv('results/analysis/target_runs_full.csv'),pd.read_csv('results/analysis/target_runs_baselines.csv')]).drop_duplicates(['target','source','model','cond','k','split'])
M=f.set_index(['target','source','model','cond','k','split']).test
def rows(pat):
    out=[]
    for p in glob.glob(pat,recursive=True):
        d=json.load(open(p)); out.append(d)
    return out
print('## 1 L2-SP')
L=pd.DataFrame([dict(target=d['target'],source=d['source'],model=d['model'],k=d['k'],split=d['split'],test=100*d['result']['sel']['test'],alpha=d['result']['sel']['alpha']) for d in rows(O+'/l2sp/**/*.json')])
res=[]
for m in ['gcn','sage','gat','gin','mlp','sgx']:
  for k in [1,5]:
    q=L[(L.model==m)&(L.k==k)]
    ft=np.array([M[(r.target,r.source,m,'finetune',k,r.split)] for r in q.itertuples()])
    zs=np.array([M[(r.target,r.source,m,'zeroshot',k,r.split)] for r in q.itertuples()])
    fz=np.array([M[(r.target,r.source,m,'frozen',k,r.split)] for r in q.itertuples()])
    l=q.test.values
    # cell-level below-ZS rate
    q2=q.assign(ft=ft,zs=zs,l=l); c=q2.groupby(['target','source']).mean(numeric_only=True)
    res.append(dict(model=m,k=k,FT=ft.mean(),L2SP=l.mean(),ZS=zs.mean(),frozen=fz.mean(),L2SP_FT=(l-ft).mean(),p_FT=stats.ttest_rel(l,ft).pvalue,
       FT_ZS=(ft-zs).mean(),L2SP_ZS=(l-zs).mean(),p_ZS=stats.ttest_rel(l,zs).pvalue,L2SP_frozen=(l-fz).mean(),
       cells_below_ZS_FT=int((c.ft<c.zs).sum()),cells_below_ZS_L2SP=int((c.l<c.zs).sum()),ncell=len(c),alpha_med=np.median(q.alpha)))
print(pd.DataFrame(res).round(3).to_string())
print('alpha counts', L.alpha.value_counts().to_dict())
# SGX vs GCN under L2SP
for k in [1,5]:
    a=L[(L.model=='sgx')&(L.k==k)].set_index(['target','source','split']).test; b=L[(L.model=='gcn')&(L.k==k)].set_index(['target','source','split']).test
    print('L2SP sgx-gcn k',k, round((a-b).mean(),2), stats.ttest_rel(a,b.loc[a.index]).pvalue)
print('## 2 UDA sgx full')
U=pd.DataFrame([dict(target=d['target'],source=d['source'],k=d['k'],split=d['split'],test=100*d['result']['sel']['test']) for d in rows(O+'/uda_sgx/**/*.json')])
for t in ['acmv9','citationv1','dblpv7','ALL']:
  for k in [1,5,20]:
    q=U[(U.k==k)&((U.target==t)|(t=='ALL'))]
    g=lambda m,c: np.array([M[(r.target,r.source,m,c,k,r.split)] for r in q.itertuples()])
    a=q.test.values; o=[]
    for n,(m,c) in {'sgxFT':('sgx','finetune'),'sgxDANN':('sgx','dann'),'gcnUDA':('gcn','udagcn'),'gcnFrozen':('gcn','frozen')}.items():
        b=g(m,c); o.append(f"{n} {b.mean():.1f} ({(a-b).mean():+.1f}, p {stats.ttest_rel(a,b).pvalue:.3f})")
    print(t,k,f"sgxUDA {a.mean():.1f} |",' | '.join(o))
print('## 3 CKA')
C=pd.DataFrame([dict(target=d['target'],source=d['source'],model=d['model'],k=d['k'],split=d['split'],cka=d['result']['cka'],dw=d['result']['wchange'],zs=100*d['result']['zs_test'],ft=100*d['result']['ft_test'],main=d['result']['main_ft_test']) for d in rows(O+'/cka/**/*.json')])
C['ftzs']=C.ft-C.zs; print('reproduce: within 0.05:',int(((C.ft-C.main).abs()<0.05).sum()),'of',len(C))
print(C.groupby(['k','model'])[['cka','dw','ftzs']].mean().round(3).unstack('k').to_string())
for k in [1,5]:
    a=C[(C.k==k)].groupby('model')[['cka','ftzs']].mean(); print('k',k,'model-level spearman',round(stats.spearmanr(a.cka,a.ftzs).statistic,2))
    cc=C[C.k==k].groupby(['target','source','model'])[['cka','ftzs']].mean(); print('  cell-level spearman', stats.spearmanr(cc.cka,cc.ftzs))
    s=C[(C.k==k)&(C.model=='sgx')].set_index(['target','source','split']).cka
    for o in ['gcn','sage','gat','gin','mlp']:
        b=C[(C.k==k)&(C.model==o)].set_index(['target','source','split']).cka; idx=s.index.intersection(b.index)
        print('  sgx-',o, round((s[idx]-b[idx]).mean(),3),'p',round(stats.ttest_rel(s[idx],b[idx]).pvalue,4))
q=C[(C.model.isin(['sgx','abl_allgcn']))&(C.target=='dblpv7')&(C.source=='acmv9')&(C.k==5)].groupby('model')[['cka','ftzs']].mean(); print(q)
print('## 4 transfer tuning')
print(json.dumps(json.load(open(O+'/tt/selected_transfer.json'))))
T=pd.DataFrame([dict(target=d['target'],source=d['source'],model=d['model'],cond=d['cond'],k=d['k'],split=d['split'],test=100*d['result']['sel']['test']) for d in rows(O+'/tt/target/**/*.json')])
T=T.set_index(['target','source','model','cond','k','split']).test
for t in ['acmv9','citationv1','dblpv7','ALL']:
  for k in [1,5]:
    o=[]; v={}
    for m in ['gcn','sgx']:
        idx=[i for i in T.index if i[2]==m and i[3]=='finetune' and i[4]==k and (t=='ALL' or i[0]==t)]
        ft=np.array([T[i] for i in idx]); zs=np.array([T[(i[0],i[1],m,'zeroshot',k,i[5])] for i in idx])
        oft=np.array([M[(i[0],i[1],m,'finetune',k,i[5])] for i in idx]); ozs=np.array([M[(i[0],i[1],m,'zeroshot',k,i[5])] for i in idx])
        v[m]=(ft-zs, idx)
        o.append(f"{m}: TT ZS {zs.mean():.1f} FT {ft.mean():.1f} FT-ZS {(ft-zs).mean():+.1f} | orig ZS {ozs.mean():.1f} FT {oft.mean():.1f} FT-ZS {(oft-ozs).mean():+.1f} | TT-orig FT {(ft-oft).mean():+.1f} p {stats.ttest_rel(ft,oft).pvalue:.3f}")
    d=v['sgx'][0]-v['gcn'][0]
    print(t,k,' || '.join(o), f"|| sgx(FT-ZS)-gcn(FT-ZS) {d.mean():+.1f} p {stats.ttest_rel(v['sgx'][0],v['gcn'][0]).pvalue:.4f}")
