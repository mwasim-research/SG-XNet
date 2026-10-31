import os
import warnings; warnings.filterwarnings('ignore')
import json, glob, os, numpy as np, pandas as pd
from scipy import stats
P=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
C=pd.DataFrame([dict(target=d['target'],source=d['source'],model=d['model'],k=d['k'],split=d['split'],cka=d['result']['cka'],ftzs=100*(d['result']['ft_test']-d['result']['zs_test'])) for d in (json.load(open(p)) for p in glob.glob(f'{P}/results_posthoc_overnight/cka/**/*.json',recursive=True))])
C=C[C.model.isin(['gcn','sage','gat','gin','mlp','sgx'])]
for k in [1,5]:
    cell=C[C.k==k].groupby(['target','source','model'])[['cka','ftzs']].mean().reset_index()
    print('k',k,'cells',len(cell),'all-cell rho',round(stats.spearmanr(cell.cka,cell.ftzs).statistic,2))
    w=[]
    for m,g in cell.groupby('model'):
        r=stats.spearmanr(g.cka,g.ftzs).statistic; w.append(r); print('   within',m,len(g),round(r,2))
    cc=cell.copy(); cc['c']=cc.cka-cc.groupby('model').cka.transform('mean'); cc['f']=cc.ftzs-cc.groupby('model').ftzs.transform('mean')
    print('  pooled within-model (centred) rho',round(stats.spearmanr(cc.c,cc.f).statistic,2),'p',round(stats.spearmanr(cc.c,cc.f).pvalue,3))
    mm=cell.groupby('model')[['cka','ftzs']].mean(); print('  between-model rho',round(stats.spearmanr(mm.cka,mm.ftzs).statistic,2))
    # split-level within cell
    s=C[C.k==k].copy(); s['c']=s.cka-s.groupby(['target','source','model']).cka.transform('mean'); s['f']=s.ftzs-s.groupby(['target','source','model']).ftzs.transform('mean')
    print('  within-cell split-level rho',round(stats.spearmanr(s.c,s.f).statistic,2),'p',f"{stats.spearmanr(s.c,s.f).pvalue:.2g}")
