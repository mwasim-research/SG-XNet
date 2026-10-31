import json, glob, os, numpy as np, pandas as pd
from scipy import stats
P=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); O=os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')), 'paper', 'output'); os.makedirs(O,exist_ok=True)
NM={'gcn':'GCN','sage':'GraphSAGE','gat':'GAT','gin':'GIN','mlp':'MLP','sgx':'SG-XNet','sgxg':'SG-XNet-G'}
DN={'cora':'Cora','citeseer':'CiteSeer','pubmed':'PubMed','dblp':'DBLP','acmv9':'ACMv9','citationv1':'Citationv1','dblpv7':'DBLPv7','roman_empire':'roman-empire','amazon_ratings':'amazon-ratings','chameleon_filtered':'chameleon-f','squirrel_filtered':'squirrel-f','twitch_de':'Twitch-DE','twitch_engb':'Twitch-EN','twitch_fr':'Twitch-FR'}
out=[]
def cfgstr(c,m):
    s=f"K{c['K']}, {c['hidden']}, d{c['dropout']}, lr{c['lr']:g}, wd{c['wd']:g}"
    if c.get('residual'): s+=', res'
    if c.get('norm'): s+=', LN'
    if m=='gat': s+=f", h{c['heads']}"
    if m in('sage','sgx'): s+=f", {c['aggr']}"
    return s
sel=json.load(open(f'{P}/results/selected_budget20.json'))
out.append(r"\begin{table}[!htbp]\centering\scriptsize\caption{Selected configurations (tuning protocol: 30 configurations $\times$ 3 tuning splits, 20 labels per class). K: layers; hidden units; d: dropout; lr; wd: weight decay; res: residual; LN: LayerNorm; h: GAT heads; aggregator for GraphSAGE and SG-XNet.}\label{tab:s2}\begin{adjustbox}{max width=\linewidth}\begin{tabular}{l"+"l"*6+r"}\toprule Target & "+" & ".join(NM[m] for m in ['gcn','sage','gat','gin','mlp','sgx'])+r"\\\midrule")
for t in ['cora','pubmed','acmv9','citationv1','dblpv7','roman_empire','chameleon_filtered']:
    out.append(DN[t]+" & "+" & ".join(cfgstr(sel[t][m],m) for m in ['gcn','sage','gat','gin','mlp','sgx'])+r"\\")
tw=json.load(open(f'{P}/results_posthoc_twitch/selected_twitch.json'))
for t in ['twitch_de','twitch_engb','twitch_fr']:
    out.append(DN[t]+" & "+" & ".join(cfgstr(tw[t][m],m) for m in ['gcn','sage','gat','gin','mlp','sgx'])+r"\\")
out.append(r"\bottomrule\end{tabular}\end{adjustbox}\end{table}")
# per-target aligned table
f=pd.read_csv(f'{P}/results/analysis/target_runs_full.csv'); b=pd.read_csv(f'{P}/results/analysis/target_runs_baselines.csv')
X=pd.concat([f,b]).drop_duplicates(['target','source','model','cond','k','split'])
bad=lambda m,s,t: (m=='gat' and s=='acmv9' and t=='dblpv7') or (m in('gat','sgx') and s=='citeseer' and t=='chameleon_filtered')
def val(t,m,c,k,src=None):
    q=X[(X.target==t)&(X.model==m)&(X.cond==c)&(X.k==k)]
    if c=='scratch' or '_self' in c: q=q[q.source.isna()]
    else:
        q=q[[not bad(m,s,t) for s in q.source]]
        if src: q=q[q.source==src]
    if len(q)==0: return None
    return q.groupby('split').test.mean().mean()
out.append(r"\begin{table}[!htbp]\centering\scriptsize\caption{Every target and model (mean over the target's sources and 10 splits; micro-F1 on ACDNE graphs, accuracy otherwise): from scratch / fine-tuned / frozen / zero-shot (aligned targets only). Failed pretrainings excluded.}\label{tab:s3}\begin{adjustbox}{max width=\linewidth}\begin{tabular}{llcccccc}\toprule Target & $k$ & GCN & GraphSAGE & GAT & GIN & MLP & SG-XNet\\\midrule")
for t in ['acmv9','citationv1','dblpv7','cora','pubmed','chameleon_filtered','roman_empire']:
    for k in [1,3,5,10,20]:
        cells=[]
        for m in ['gcn','sage','gat','gin','mlp','sgx']:
            v=[val(t,m,c,k) for c in (['scratch','finetune','frozen','zeroshot'] if t in('acmv9','citationv1','dblpv7') else ['scratch','finetune','frozen'])]
            cells.append(" / ".join('--' if x is None else f"{x:.1f}" for x in v))
        out.append((DN[t] if k==1 else '')+f" & {k} & "+" & ".join(cells)+r"\\")
    out.append(r"\midrule")
out[-1]=r"\bottomrule\end{tabular}\end{adjustbox}\end{table}"
# SSL on non-aligned targets, gcn & sgx
ssl=['ssl_dgi','ssl_grace','ssl_dgi_self','ssl_grace_self','ssl_dgi_probe','ssl_grace_probe','ssl_dgi_self_probe','ssl_grace_self_probe']
SN={'ssl_dgi':'DGI src FT','ssl_grace':'GRACE src FT','ssl_dgi_self':'DGI tgt FT','ssl_grace_self':'GRACE tgt FT','ssl_dgi_probe':'DGI src probe','ssl_grace_probe':'GRACE src probe','ssl_dgi_self_probe':'DGI tgt probe','ssl_grace_self_probe':'GRACE tgt probe'}
out.append(r"\begin{table}[!htbp]\centering\scriptsize\caption{Self-supervised pretraining on every target (GCN / SG-XNet; $k=1$, 5, 20; mean over the target's sources and 10 splits). chameleon-f: source squirrel-f only.}\label{tab:s4}\begin{adjustbox}{max width=\linewidth}\begin{tabular}{l"+"c"*7+r"}\toprule Method & "+" & ".join(DN[t] for t in ['acmv9','citationv1','dblpv7','cora','pubmed','chameleon_filtered','roman_empire'])+r"\\\midrule")
for c in ['scratch','finetune','frozen']+ssl:
    row=[]
    for t in ['acmv9','citationv1','dblpv7','cora','pubmed','chameleon_filtered','roman_empire']:
        src='squirrel_filtered' if t=='chameleon_filtered' else None
        v=[[val(t,m,c,k,src) for k in [1,5,20]] for m in ['gcn','sgx']]
        row.append("; ".join("/".join('--' if x is None else f"{x:.1f}" for x in vv) for vv in v))
    out.append({'scratch':'From scratch','finetune':'Supervised FT','frozen':'Supervised frozen'}.get(c,SN.get(c,c))+" & "+" & ".join(row)+r"\\")
out.append(r"\bottomrule\end{tabular}\end{adjustbox}\end{table}")
# transfer-aware tuning
ttsel=json.load(open(f'{P}/results_posthoc_overnight/tt/selected_transfer.json'))
out.append(r"\begin{table}[!htbp]\centering\scriptsize\caption{Transfer-aware tuning (post hoc): configurations selected by fine-tuned validation accuracy after pretraining, against those selected for training from scratch.}\label{tab:s5}\begin{tabular}{lll}\toprule Target, model & Selected for scratch (main study) & Selected for transfer\\\midrule")
for t in ['acmv9','citationv1','dblpv7']:
    for m in ['gcn','sgx']:
        out.append(f"{DN[t]}, {NM[m]} & {cfgstr(sel[t][m],m)} & {cfgstr(ttsel[t][m],m)}"+r"\\")
out.append(r"\bottomrule\end{tabular}\end{table}")
# pretraining source validation (median over seeds)
rows=[]
for p in sorted(glob.glob(f'{P}/results/pretrain/*')):
    tag=os.path.basename(p); s,t=tag.split('__'); r=[]
    for m in ['gcn','sage','gat','gin','mlp','sgx']:
        v=[json.load(open(q))['val'] for q in glob.glob(f'{p}/{m}/real_s*.json')]
        r.append(f"{100*np.median(v):.1f}" if v else '--')
    rows.append(f"{DN[s]}$\\rightarrow${DN[t]} & "+" & ".join(r)+r"\\")
out.append(r"\begin{table}[!htbp]\centering\scriptsize\caption{Source validation score of supervised pretraining (median over 10 seeds). Pretraining runs scoring below half of the best model on the pair (GAT on ACMv9$\rightarrow$DBLPv7; GAT and SG-XNet on CiteSeer$\rightarrow$chameleon-f) are treated as failed and excluded.}\label{tab:s6}\begin{tabular}{lcccccc}\toprule Pair & GCN & GraphSAGE & GAT & GIN & MLP & SG-XNet\\\midrule"+"\n".join(rows)+r"\bottomrule\end{tabular}\end{table}")
open(f'{O}/supp_tables.tex','w').write("\n".join(out)); print('ok', len(out))
