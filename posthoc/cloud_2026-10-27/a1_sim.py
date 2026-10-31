import os
import pandas as pd, numpy as np, json, sys, warnings; warnings.filterwarnings('ignore')
from scipy import stats
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *(['..']*2)))); from sgx import data
U=os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), *(['..']*2))), 'results') + os.sep
f=pd.read_csv(U+'analysis/target_runs_full.csv'); f=f[f.model.isin(['gcn','sage','gat','gin','mlp','sgx'])]
bad=((f.model=='gat')&(f.source=='acmv9')&(f.target=='dblpv7'))|((f.model.isin(['gat','sgx']))&(f.source=='citeseer')&(f.target=='chameleon_filtered'))
f=f[~bad]
sc=f[f.cond=='scratch'].groupby(['target','model','k']).test.mean()
pairs=f[f.cond=='finetune'][['source','target']].drop_duplicates().values.tolist()
G={}
def stats_of(n):
    if n in G: return G[n]
    g=data.load(n); ei=g.ei.numpy(); deg=np.bincount(ei[1],minlength=g.n)
    y=g.y_np; cls=np.bincount(y,minlength=g.num_classes)/len(y) if y.ndim==1 else y.mean(0)/y.mean(0).sum()
    x=g.x.to_dense().numpy() if g.x.layout!=__import__('torch').strided else g.x.numpy()
    G[n]=dict(deg=deg,cls=cls,x=x,h=float((y[ei[0]]==y[ei[1]]).mean()) if y.ndim==1 else float((y[ei[0]]==y[ei[1]]).all(1).mean()),n=g.n,d=x.shape[1]); return G[n]
def mmd_lin(a,b,rng=np.random.default_rng(0)):
    a=a[rng.choice(len(a),min(3000,len(a)),replace=False)]; b=b[rng.choice(len(b),min(3000,len(b)),replace=False)]
    a=a/ (np.linalg.norm(a,axis=1,keepdims=True)+1e-9); b=b/(np.linalg.norm(b,axis=1,keepdims=True)+1e-9)
    return float(np.linalg.norm(a.mean(0)-b.mean(0)))
rows=[]
for s,t in pairs:
    S,T=stats_of(s),stats_of(t); al=S['d']==T['d'] and s in ('acmv9','citationv1','dblpv7')
    ft=f[(f.cond=='finetune')&(f.source==s)&(f.target==t)].groupby(['model','k']).test.mean()
    fz=f[(f.cond=='frozen')&(f.source==s)&(f.target==t)].groupby(['model','k']).test.mean()
    gft=np.mean([v-sc[(t,m,k)] for (m,k),v in ft.items()]); gfz=np.mean([v-sc[(t,m,k)] for (m,k),v in fz.items()])
    g1=np.mean([v-sc[(t,m,k)] for (m,k),v in ft.items() if k==1])
    jsd=float(stats.entropy(0.5*(S['cls']+T['cls']),base=2)-0.5*stats.entropy(S['cls'],base=2)-0.5*stats.entropy(T['cls'],base=2)) if al else np.nan
    rows.append(dict(pair=f'{s}->{t}',shared=al,gain_ft=round(gft,1),gain_ft_k1=round(g1,1),gain_frozen=round(gfz,1),
        dh=round(abs(S['h']-T['h']),3),h_t=round(T['h'],3),deg_ks=round(stats.ks_2samp(S['deg'],T['deg']).statistic,3),
        logdeg_w1=round(stats.wasserstein_distance(np.log1p(S['deg']),np.log1p(T['deg'])),3),
        size_ratio=round(np.log(S['n']/T['n']),2),label_jsd=None if not al else round(jsd,3),feat_mmd=None if not al else round(mmd_lin(S['x'],T['x']),3)))
d=pd.DataFrame(rows).sort_values('gain_ft',ascending=False); print(d.to_markdown(index=False))
print('\nSpearman with fine-tuning gain (all 11 pairs):')
for c in ['shared','dh','h_t','deg_ks','logdeg_w1','size_ratio']:
    r=stats.spearmanr(d[c].astype(float),d.gain_ft); print(f'  {c}: rho {r.statistic:+.2f} p {r.pvalue:.3f}')
a=d[d.shared]; print('Aligned only (6 pairs):')
for c in ['dh','deg_ks','logdeg_w1','label_jsd','feat_mmd','size_ratio']:
    r=stats.spearmanr(a[c].astype(float),a.gain_ft); r2=stats.spearmanr(a[c].astype(float),a.gain_ft_k1); print(f'  {c}: rho {r.statistic:+.2f} (k1 {r2.statistic:+.2f})')
o=d[~d.shared]; print('Other only (5 pairs):')
for c in ['dh','h_t','deg_ks','logdeg_w1']:
    r=stats.spearmanr(o[c].astype(float),o.gain_ft); print(f'  {c}: rho {r.statistic:+.2f}')
