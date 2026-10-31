import json, glob, os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
P=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); FD=f'{P}/results_posthoc_overnight/figures'; os.makedirs(FD,exist_ok=True)
plt.rcParams.update({'font.family':'sans-serif','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42})
NM={'gcn':'GCN','sage':'GraphSAGE','gat':'GAT','gin':'GIN','mlp':'MLP','sgx':'SG-XNet','abl_allgcn':'all-GCN (SG-XNet settings)'}
COL={'gcn':'#4C72B0','sage':'#DD8452','gat':'#55A868','gin':'#8172B3','mlp':'#937860','sgx':'#C44E52','abl_allgcn':'#222222'}
MK={'gcn':'o','sage':'s','gat':'^','gin':'D','mlp':'v','sgx':'*','abl_allgcn':'X'}
def save(fig,name):
    for e in ['pdf','eps','svg','png']: fig.savefig(f'{FD}/{name}.{e}',dpi=300,bbox_inches='tight')
    plt.close(fig)
# 1 CKA vs FT-ZS
C=pd.DataFrame([dict(target=d['target'],source=d['source'],model=d['model'],k=d['k'],cka=d['result']['cka'],ftzs=100*(d['result']['ft_test']-d['result']['zs_test'])) for d in (json.load(open(p)) for p in glob.glob(f'{P}/results_posthoc_overnight/cka/**/*.json',recursive=True))])
cell=C.groupby(['k','target','source','model'])[['cka','ftzs']].mean().reset_index(); cell.to_csv(f'{FD}/fig_cka_drift.csv',index=False)
fig,axs=plt.subplots(1,2,figsize=(7.2,3.0),sharey=True)
for ax,k in zip(axs,[1,5]):
    q=cell[(cell.k==k)&(cell.model!='abl_allgcn')]
    for m in ['gcn','sage','gat','gin','mlp','sgx']:
        z=q[q.model==m]; ax.scatter(z.cka,z.ftzs,s=40 if m=='sgx' else 22,marker=MK[m],color=COL[m],label=NM[m],edgecolor='white',linewidth=0.4)
    r=stats.spearmanr(q.cka,q.ftzs); ax.axhline(0,color='grey',lw=0.6,ls='--')
    ax.set_title(f'k = {k} label{"s" if k>1 else ""} per class   (Spearman ρ = {r.statistic:.2f})',fontsize=9)
    ax.set_xlabel('CKA similarity of body embeddings\nbefore vs after fine-tuning')
axs[0].set_ylabel('fine-tuned − zero-shot (points)'); axs[1].legend(frameon=False,fontsize=7.5,loc='lower right')
save(fig,'fig_cka_drift')
# 2 L2-SP vs plain vs frozen, minus zero-shot, aligned pairs
f=pd.read_csv(f'{P}/results/analysis/target_runs_full.csv'); I=f.set_index(['target','source','model','cond','k','split']).test
L=pd.DataFrame([dict(target=d['target'],source=d['source'],model=d['model'],k=d['k'],split=d['split'],test=100*d['result']['sel']['test']) for d in (json.load(open(p)) for p in glob.glob(f'{P}/results_posthoc_overnight/l2sp/**/*.json',recursive=True))])
rows=[]
for m in ['gcn','sage','gat','gin','mlp','sgx']:
  for k in [1,5]:
    q=L[(L.model==m)&(L.k==k)]
    g=lambda c: np.array([I[(r.target,r.source,m,c,k,r.split)] for r in q.itertuples()])
    zs=g('zeroshot'); rows.append(dict(model=m,k=k,plain=(g('finetune')-zs).mean(),l2sp=(q.test.values-zs).mean(),frozen=(g('frozen')-zs).mean()))
D=pd.DataFrame(rows); D.to_csv(f'{FD}/fig_l2sp.csv',index=False)
fig,axs=plt.subplots(1,2,figsize=(7.2,2.8),sharey=True); w=0.26
for ax,k in zip(axs,[1,5]):
    d=D[D.k==k]; x=np.arange(len(d))
    ax.bar(x-w,d.plain,w,label='plain fine-tuning',color='#999999'); ax.bar(x,d.l2sp,w,label='L2-SP fine-tuning',color='#4C72B0'); ax.bar(x+w,d.frozen,w,label='frozen body',color='#DD8452')
    ax.axhline(0,color='black',lw=0.6); ax.set_xticks(x); ax.set_xticklabels([NM[m] for m in d.model],rotation=30,ha='right'); ax.set_title(f'k = {k}',fontsize=9)
axs[0].set_ylabel('accuracy − zero-shot (points)'); axs[1].legend(frameon=False,fontsize=7.5)
save(fig,'fig_l2sp')
# 3 when does transfer work: mean FT gain over scratch per pair (all models), k=1, incl. Twitch
sc=f[f.cond=='scratch'].set_index(['target','model','k','split']).test
bad=lambda m,s,t: (m=='gat' and s=='acmv9' and t=='dblpv7') or (m in('gat','sgx') and s=='citeseer' and t=='chameleon_filtered')
ft=f[(f.cond=='finetune')&(f.model.isin(['gcn','sage','gat','gin','mlp','sgx']))].copy(); ft=ft[[not bad(a,b,c) for a,b,c in zip(ft.model,ft.source,ft.target)]]
ft['gain']=ft.test.values-np.array([sc[(r.target,r.model,r.k,r.split)] for r in ft.itertuples()])
g=ft.groupby(['source','target','k']).gain.mean().unstack('k')
T=[json.load(open(p)) for p in glob.glob(f'{P}/results_posthoc_twitch/target/*/*/*/*.json')]
T=pd.DataFrame([dict(target=d['target'],source=d.get('source'),model=d['model'],cond=p,k=d['k'],split=d['split'],test=100*d['result']['sel']['test']) for d,p in zip(T,[q.split('/')[-2].split('@')[0] for q in glob.glob(f'{P}/results_posthoc_twitch/target/*/*/*/*.json')])])
ts=T[T.cond=='scratch'].set_index(['target','model','k','split']).test; tf=T[T.cond=='finetune'].copy(); tf['gain']=tf.test.values-np.array([ts[(r.target,r.model,r.k,r.split)] for r in tf.itertuples()])
tg=tf.groupby(['source','target','k']).gain.mean().unstack('k')
LB={'acmv9':'ACMv9','citationv1':'Citationv1','dblpv7':'DBLPv7','citeseer':'CiteSeer','cora':'Cora','dblp':'DBLP','pubmed':'PubMed','squirrel_filtered':'squirrel-f','chameleon_filtered':'chameleon-f','amazon_ratings':'amazon-ratings','roman_empire':'roman-empire','twitch_de':'Twitch-DE','twitch_engb':'Twitch-EN','twitch_fr':'Twitch-FR'}
items=[]
for (s,t),v in g.iterrows(): items.append((f'{LB[s]}→{LB[t]}',v[1],v[20],'ACDNE (shared features and labels)' if s in('acmv9','citationv1','dblpv7') else 'no shared features/labels'))
for (s,t),v in tg.iterrows(): items.append((f'{LB[s]}→{LB[t]}',v[1],np.nan,'Twitch (shared features and labels)'))
W=pd.DataFrame(items,columns=['pair','k1','k20','group']).sort_values('k1'); W.to_csv(f'{FD}/fig_when_transfer.csv',index=False)
cg={'ACDNE (shared features and labels)':'#4C72B0','Twitch (shared features and labels)':'#55A868','no shared features/labels':'#999999'}
fig,ax=plt.subplots(figsize=(6.0,4.6)); y=np.arange(len(W))
ax.barh(y,W.k1,color=[cg[x] for x in W.group]); ax.set_yticks(y); ax.set_yticklabels(W.pair,fontsize=7.5); ax.axvline(0,color='black',lw=0.6)
ax.set_xlabel('fine-tuning gain over training from scratch, k = 1 (points; mean over models)')
from matplotlib.patches import Patch; ax.legend(handles=[Patch(color=v,label=k) for k,v in cg.items()],frameon=False,fontsize=7.5,loc='lower right')
save(fig,'fig_when_transfer')
# 4 average rank of fine-tuned models (aligned / other), with CD
MS=['gcn','sage','gat','gin','mlp','sgx']; f['al']=f.target.isin(['acmv9','citationv1','dblpv7'])&f.source.isin(['acmv9','citationv1','dblpv7'])
fig,axs=plt.subplots(1,2,figsize=(7.2,2.4)); res={}
for ax,(al,name) in zip(axs,[(True,'aligned pairs (shared features and labels)'),(False,'other pairs')]):
    x=f[(f.cond=='finetune')&(f.al==al)&(f.model.isin(MS))]; x=x[[not bad(a,b,c) for a,b,c in zip(x.model,x.source,x.target)]]
    p=x.groupby(['target','source','k','model']).test.mean().unstack('model')[MS].dropna(); rk=p.rank(axis=1,ascending=False).mean().sort_values()
    N,K=p.shape; cd=stats.studentized_range.ppf(0.95,K,np.inf)/np.sqrt(2)*np.sqrt(K*(K+1)/(6*N)); res[name]=(rk.to_dict(),cd,N)
    ax.barh(range(K),rk.values,color=[COL[m] for m in rk.index]); ax.set_yticks(range(K)); ax.set_yticklabels([NM[m] for m in rk.index]); ax.invert_yaxis()
    ax.plot([rk.values[0],rk.values[0]+cd],[-0.55,-0.55],color='black',lw=1.5); ax.text(rk.values[0]+cd+0.08,-0.55,'CD',va='center',fontsize=7)
    ax.set_xlim(0,6.2); ax.set_xlabel('average rank (1 = best)'); ax.set_title(f'{name}; N = {N}, CD = {cd:.2f}',fontsize=8)
save(fig,'fig_rank'); json.dump(res,open(f'{FD}/fig_rank.json','w'),indent=1)
print('ok', os.listdir(FD))
