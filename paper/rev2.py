import json, glob, os, numpy as np, pandas as pd
from scipy import stats
P=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
f=pd.read_csv(f'{P}/results/analysis/target_runs_full.csv')
MS=['gcn','sage','gat','gin','mlp','sgx']; AL=['acmv9','citationv1','dblpv7']
I=f.set_index(['target','source','model','cond','k','split']).test
def holm(p):
    p=np.asarray(p,float); m=len(p); o=np.argsort(p); a=np.empty(m); run=0
    for i,j in enumerate(o): run=max(run,(m-i)*p[j]); a[j]=min(1,run)
    return a
print('== (a) main table on the 5 pairs common to all models (drop acmv9->dblpv7) and GAT incl. failed run')
P5=[(s,t) for t in AL for s in AL if s!=t and not (s=='acmv9' and t=='dblpv7')]; P6=[(s,t) for t in AL for s in AL if s!=t]
def u(m,c,k,pairs):
    if c=='scratch':
        ts=sorted({t for _,t in pairs}); return np.mean([[I[(t,np.nan,m,c,k,s)] if False else f[(f.target==t)&(f.model==m)&(f.cond=='scratch')&(f.k==k)&(f.split==s)].test.iloc[0] for s in range(10)] for t in ts],axis=0)
    return np.mean([[I[(t,src,m,c,k,s)] for s in range(10)] for src,t in pairs],axis=0)
for k in [1,5,20]:
    row=[]
    for m in MS:
        ft,zs=u(m,'finetune',k,P5),u(m,'zeroshot',k,P5)
        row.append(f"{m}: FT {ft.mean():.1f} FR {u(m,'frozen',k,P5).mean():.1f} ZS {zs.mean():.1f} FT-ZS {(ft-zs).mean():+.1f}{'*' if stats.ttest_rel(ft,zs).pvalue<0.05 else ''}")
    print('k',k,' | '.join(row))
    ft,zs=u('gat','finetune',k,P6),u('gat','zeroshot',k,P6); print('   GAT incl failed (6 pairs): FT %.1f FR %.1f ZS %.1f FT-ZS %+.1f'%(ft.mean(),u('gat','frozen',k,P6).mean(),zs.mean(),(ft-zs).mean()))
print('== (b) pre-stated test vs GAT incl. failed run, and on matched sources')
def unit_t(m,t,srcs):
    return np.mean([[np.mean([I[(t,s,m,'finetune',k,sp)]-I[(t,s,m,'zeroshot',k,sp)] for k in [1,3,5]]) for sp in range(10)] for s in srcs],axis=0)
for t in AL:
    srcs=[s for s in AL if s!=t]
    a=unit_t('sgx',t,srcs); b=unit_t('gat',t,srcs); print(t,'all sources incl failed: %+.1f p %.3f'%((a-b).mean(),stats.ttest_rel(a,b).pvalue))
    if t=='dblpv7':
        a=unit_t('sgx',t,['citationv1']); b=unit_t('gat',t,['citationv1']); print('   matched source citationv1 only: %+.1f p %.3f'%((a-b).mean(),stats.ttest_rel(a,b).pvalue))
print('== (e) ablation Holm (family = all 36 variant x pair x k comparisons, fine-tuned)')
V=['sgx_order','sgx_last','sgx_sage2','sgx_nodrop','abl_allgcn','abl_allsage']; rows=[]
for s,t in [('citeseer','cora'),('acmv9','dblpv7'),('squirrel_filtered','chameleon_filtered')]:
    for k in [5,20]:
        a=np.array([I[(t,s,'sgx','finetune',k,sp)] for sp in range(10)])
        for v in V:
            b=np.array([I[(t,s,v,'finetune',k,sp)] for sp in range(10)]); rows.append((s,t,k,v,(b-a).mean(),stats.ttest_rel(b,a).pvalue))
ph=holm([r[5] for r in rows]); print('significant after Holm:',[(r[0],r[2],r[3],round(r[4],1),round(h,3)) for r,h in zip(rows,ph) if h<0.05])
print('unadjusted p<0.05:',[(r[0],r[2],r[3],round(r[4],1),round(r[5],3)) for r in rows if r[5]<0.05])
print('== (c) Twitch per target: majority rate, accuracy and macro-F1 (FT/frozen/scratch/ZS), k=1,5')
T=[]
for p in glob.glob(f'{P}/results_posthoc_twitch/target/*/*/*/*.json'):
    d=json.load(open(p)); s=d['result']['sel']; T.append(dict(target=d['target'],model=d['model'],cond=p.split('/')[-2].split('@')[0],k=d['k'],split=d['split'],acc=100*s['test'],mf1=100*s.get('test_macro_f1',np.nan)))
T=pd.DataFrame(T)
for t in ['twitch_de','twitch_engb','twitch_fr']:
    y=json.load(open(f'{P}/data/cache/{t}.json'))
    import numpy as _n; z=_n.load(f'{P}/data/cache/{t}.npz'); yy=z['y']; maj=max(yy.mean(),1-yy.mean())*100
    print(t,'majority %.1f'%maj)
    for k in [1,5]:
        q=T[(T.target==t)&(T.k==k)].groupby(['model','cond'])[['acc','mf1']].mean()
        print('  k',k,' '.join(f"{m}:" + "/".join(f"{q.loc[(m,c),'acc']:.1f}({q.loc[(m,c),'mf1']:.1f})" for c in ['scratch','finetune','frozen','zeroshot','l2sp']) for m in MS))
