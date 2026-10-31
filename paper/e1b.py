import os
import warnings; warnings.filterwarnings('ignore')
import os
src=open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'n2_an.py')).read().split("print('per pair")[0]
exec(src)
mb={('acmv9','dblpv7','gat')}
def unitb(src,m,c,k,b):
    v=[src.loc[(t,s,m,c,k)].sort_index() for s,t in PAIRS if (s,t,m) not in b]; return pd.concat(v,axis=1).mean(axis=1)
for k in [1,5]:
  for m in MS:
    a=unitb(Xm,m,'finetune',k,mb)-unitb(Xm,m,'zeroshot',k,mb)
    fz=unitb(I,m,'frozen',k,set()); zs=unitb(I,m,'zeroshot',k,set()); zm=unitb(Xm,m,'zeroshot',k,mb); ft=unitb(I,m,'finetune',k,set())
    d=ft-zs
    print(k,m,'mainFTZS',round(a.mean(),1),'| E1 FT-ZS',round(d.mean(),1),'sd',round(d.std(ddof=1),1),'p',f"{stats.ttest_rel(ft,zs).pvalue:.2g}",'| frozen-ZS',round((fz-zs).mean(),1),'p',round(stats.ttest_rel(fz,zs).pvalue,3),'| ZS',round(zs.mean(),1),'mainZS',round(zm.mean(),1))
