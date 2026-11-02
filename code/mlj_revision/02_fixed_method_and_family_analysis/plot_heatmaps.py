import shutil
from pathlib import Path
import json,subprocess,os
os.environ['MPLBACKEND']='Agg'
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
H=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
def load(n):
 j=json.loads((H/(n+'.export.json')).read_text());return pd.DataFrame(j['values'],columns=j['columns'])
S=load('fixed_method_scenario_results');F=load('fixed_method_ranks');methods=list(range(3,28));names=S.drop_duplicates('method_number').set_index('method_number').method_name
labels=[f'{n:03d}  {names[n]}' for n in methods]
ends=[1,11,20,23,26,28,37,38,39,40];starts=[0]+ends[:-1]
legend='F1 Reference   F2 Prevalence   F3 Positive-label availability   F4 Separation   F5 Dimensionality\nF6 Correlation   F7 Joint prevalence/labels   F8 Curvature   F9 Feature interaction   F10 Unequal covariance'
def frame(ax,matrix,cmap,vmin,vmax,xlabels):
 ax.set_facecolor('#e5e7eb');pc=ax.pcolormesh(np.arange(matrix.shape[1]+1),np.arange(26),np.ma.masked_invalid(matrix),cmap=cmap,vmin=vmin,vmax=vmax,rasterized=False)
 ax.set_ylim(25,0);ax.set_yticks(np.arange(25)+.5,labels=labels,fontsize=9);ax.set_xticks(np.arange(matrix.shape[1])+.5,labels=xlabels,fontsize=8,rotation=90);ax.tick_params(length=0,pad=5)
 for y in [3,8]:ax.axhline(y,color='white',lw=2)
 return pc
meta=[]
for p in ['fixed_method_heatmap.pdf','family_rank_heatmap.pdf']:assert not (H/p).exists()
with PdfPages(H/'fixed_method_heatmap.pdf',metadata={'Title':'Fixed PU methods across all 40 scenarios','Subject':'Archived ranking and common quantile BA; Task 1C pilot separate'}) as pdf:
 specs=[('AUROC','ranking_archive'),('AP','ranking_archive'),('BalancedAccuracy','Task1A_quantile_primary'),('BalancedAccuracy','Task1C_PU_BA_pilot')]
 for metric,policy in specs:
  for design in ['one_sample','two_sample']:
   p=S[(S.design==design)&(S.metric==metric)&(S.policy==policy)&S.practical_PU].copy();pilot=policy.endswith('pilot')
   if not pilot:p.loc[~p.complete_100_reps,'mean_value']=np.nan
   matrix=p.pivot(index='method_number',columns='scenario_index',values='mean_value').reindex(index=methods,columns=range(1,41)).to_numpy(float)
   fig,ax=plt.subplots(figsize=(16,10));fig.subplots_adjust(left=.20,right=.92,bottom=.17,top=.84)
   limits=(.45,.82) if metric=='AUROC' else (0,.75) if metric=='AP' else (.48,.75)
   pc=frame(ax,matrix,'viridis',*limits,[f'{i:02d}' for i in range(1,41)])
   for x in ends[:-1]:ax.axvline(x,color='white',lw=1.5)
   for j,(a,b) in enumerate(zip(starts,ends)):ax.text((a+b)/2,-.55,f'F{j+1}',ha='center',va='bottom',fontsize=9,color='#374151')
   cb=fig.colorbar(pc,ax=ax,fraction=.023,pad=.018);cb.set_label(metric if metric!='BalancedAccuracy' else 'Balanced accuracy')
   subtitle='Published raw-ranking archive; 100 repetitions per cell.' if policy=='ranking_archive' else 'Task 1A common prevalence quantile; retained threshold applied to refitted test model.' if not pilot else 'Task 1C PU-BA pilot only; selected train-only model, one repetition per observed cell.'
   fig.suptitle(f'{metric}: fixed methods, {design.replace("_"," ")}',x=.20,ha='left',y=.965,fontsize=18,fontweight='bold')
   fig.text(.20,.918,subtitle,fontsize=11)
   missing='Gray cells are unavailable. Partial TM003 recovery (eight single repetitions) is excluded from 100-repetition means.' if design=='two_sample' and policy=='Task1A_quantile_primary' else 'Gray cells are unavailable; this pilot cannot provide a 40-setting or ten-family summary.' if pilot else 'Rows follow the fixed catalog. Supervised references are excluded. No group maximum is selected.'
   fig.text(.20,.885,missing,fontsize=10,color='#4b5563');ax.set_xlabel('Scenario number (S for one sample, TS for two sample)',labelpad=9)
   fig.text(.20,.035,legend,fontsize=9,linespacing=1.6,color='#4b5563');pdf.savefig(fig);plt.close(fig);meta.append(dict(file='fixed_method_heatmap.pdf',metric=metric,policy=policy,design=design,finite_cells=int(np.isfinite(matrix).sum())))
with PdfPages(H/'family_rank_heatmap.pdf',metadata={'Title':'Equal-family ranks of fixed PU methods'}) as pdf:
 for metric,policy in specs[:3]:
  for design in ['one_sample','two_sample']:
   q=F[(F.design==design)&(F.metric==metric)&(F.policy==policy)&F.practical_PU];sensitivity=metric=='BalancedAccuracy' and design=='two_sample';col='available_catalog_mean_rank' if sensitivity else 'mean_rank'
   fam=q[q.rank_level=='within_family'].pivot(index='method_number',columns='family_id',values=col).reindex(index=methods,columns=range(1,11));last=q[q.rank_level=='equal_10_family'].set_index('method_number')[col].reindex(methods);matrix=np.c_[fam.to_numpy(float),last.to_numpy(float)]
   fig,ax=plt.subplots(figsize=(13.5,10));fig.subplots_adjust(left=.24,right=.92,bottom=.17,top=.85)
   pc=frame(ax,matrix,'viridis_r',1,24 if sensitivity else 25,[f'F{i}' for i in range(1,11)]+['Equal-family']);ax.axvline(10,color='white',lw=3)
   for i in range(25):
    for j in range(11):
     v=matrix[i,j]
     if np.isfinite(v):ax.text(j+.5,i+.5,f'{v:.1f}',ha='center',va='center',fontsize=8,color='white' if v>14 else '#111827')
   cb=fig.colorbar(pc,ax=ax,fraction=.026,pad=.02);cb.set_label('Mean rank (lower is better)')
   title=f'{metric}: family ranks, {design.replace("_"," ")}'
   fig.suptitle(title,x=.24,ha='left',y=.965,fontsize=17,fontweight='bold')
   subtitle='SENSITIVITY: 24 complete methods; TM003 unavailable. This is not a full 25-method BA ranking.' if sensitivity else 'Ranks within scenario among all 25 PU methods; average settings, then weight ten families equally.'
   fig.text(.24,.913,subtitle,fontsize=10);fig.text(.24,.88,'Fixed catalog order; ties receive average ranks. Gray means unavailable, not last place.',fontsize=10,color='#4b5563')
   fig.text(.24,.035,legend,fontsize=8.7,linespacing=1.6,color='#4b5563');pdf.savefig(fig);plt.close(fig);meta.append(dict(file='family_rank_heatmap.pdf',metric=metric,policy=policy,design=design,rank_catalog_n=24 if sensitivity else 25,finite_cells=int(np.isfinite(matrix).sum())))
(H/'heatmap_manifest.json').write_text(json.dumps(meta,indent=2))
preview=H/'pdf_previews';preview.mkdir(exist_ok=True)
renderer=Path(shutil.which('pdftoppm') or 'pdftoppm')
for stem in ['fixed_method_heatmap','family_rank_heatmap']:
 subprocess.run([str(renderer),'-scale-to','1700','-png',str(H/(stem+'.pdf')),str(preview/stem)],check=True)
print('Saved and rendered 8 fixed-method pages and 6 family-rank pages')
