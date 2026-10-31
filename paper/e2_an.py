import json, glob, os, numpy as np, pandas as pd
from scipy import stats
P=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
L=pd.DataFrame([dict(target=d['target'],source=d['source'],model=d['model'],k=d['k'],split=d['split'],cka_l2=d['result']['cka'],l2=100*d['result']['test'],alpha=d['alpha']) for d in (json.load(open(p)) for p in glob.glob(P+'/results_posthoc_night2/e2_l2sp_cka/**/*.json',recursive=True))])
C=pd.DataFrame([dict(target=d['target'],source=d['source'],model=d['model'],k=d['k'],split=d['split'],cka_ft=d['result']['cka'],ft=100*d['result']['ft_test'],zs=100*d['result']['zs_test']) for d in (json.load(open(p)) for p in glob.glob(P+'/results_posthoc_overnight/cka/**/*.json',recursive=True))])
print(len(L),len(C)); print(L.groupby(['model','k']).size().unstack())
J=L.merge(C,on=['target','source','model','k','split']); print('merged',len(J))
J.to_csv(os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')), 'results_posthoc_night2', 'analysis', 'e2_merged.csv'),index=False)
pd.set_option('display.width',250)
r=[]
for m in sorted(J.model.unique()):
  for k in sorted(J.k.unique()):
    q=J[(J.model==m)&(J.k==k)]
    if not len(q): continue
    u=q.groupby('split')[['cka_ft','cka_l2','ft','l2','zs']].mean()
    r.append(dict(model=m,k=k,n=len(q),cka_ft=u.cka_ft.mean(),cka_l2=u.cka_l2.mean(),d_cka=(u.cka_l2-u.cka_ft).mean(),p_cka=stats.ttest_rel(u.cka_l2,u.cka_ft).pvalue,
      ft=u.ft.mean(),l2=u.l2.mean(),zs=u.zs.mean(),d_acc=(u.l2-u.ft).mean(),p_acc=stats.ttest_rel(u.l2,u.ft).pvalue))
print(pd.DataFrame(r).round(3).to_string())
# within cells: does the CKA increase predict accuracy gain?
cell=J.groupby(['target','source','model','k'])[['cka_ft','cka_l2','ft','l2']].mean(); cell['dc']=cell.cka_l2-cell.cka_ft; cell['da']=cell.l2-cell.ft
print('cells',len(cell),'spearman dCKA vs dAcc',stats.spearmanr(cell.dc,cell.da))
print('cells with CKA up',(cell.dc>0).sum(),'acc up',(cell.da>0).sum())
# pooled split-level
u=J.groupby(['k','split'])[['cka_ft','cka_l2','ft','l2']].mean()
for k in sorted(J.k.unique()):
  z=u.loc[k]; print('pooled k',k,'cka',z.cka_ft.mean().round(3),'->',z.cka_l2.mean().round(3),'p',stats.ttest_rel(z.cka_l2,z.cka_ft).pvalue,'acc',z.ft.mean().round(2),'->',z.l2.mean().round(2))
print(J.alpha.value_counts())
