from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import json,math,sys
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
H=Path(__file__).resolve().parent
FAMILIES=['reference','prevalence','positive-label availability','class separation','dimensionality','predictor correlation','joint prevalence-label availability','curved boundary','feature interaction','unequal class covariance']
ENGINES=['AdaBoost','GBM','XGBoost','LightGBM','CatBoost']
SHORT=['Reference','Prevalence','Label availability','Separation','Dimensionality','Correlation','Prevalence x label availability','Curved boundary','Feature interaction','Unequal covariance']
COLORS=['#4D4D4D','#5597BF','#5EAA92','#8E86B7','#DAA843','#BA7D59','#94A654','#E79AB2','#C84046','#5379B5']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':10,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.axisbelow':True})
def read_table(name):
 obj=json.loads((H/(name+'.export.json')).read_text());return pd.DataFrame(obj['values'],columns=obj['columns'])
d=read_table('rn_selection_quality_all')
s=d[(d.study=='main')&(d.stage=='final_train_plus_validation')].copy()
INTERACTION_ONLY='--interaction-only' in sys.argv
manifest=json.loads((H/'figure_manifest.json').read_text()) if INTERACTION_ONLY else []
def save_scatter(name,x,outcome,xlabel):
 if INTERACTION_ONLY:return
 p=H/(name+'.pdf');assert not p.exists()
 limit=math.ceil(np.nanmax(abs(s[outcome]))*100/5)*5
 with PdfPages(p) as pdf:
  for design in ['one_sample','two_sample']:
   fig,axs=plt.subplots(2,3,figsize=(12,8.5),sharex=True,sharey=True)
   sub=s[s.design==design]
   for ax,en in zip(axs.flat,ENGINES):
    v=sub[sub.engine==en]
    for fa,col in zip(FAMILIES,COLORS):
     z=v[v.scenario_family==fa]
     ax.scatter(z[x],100*z[outcome],s=5,alpha=.23,color=col,linewidth=0,rasterized=True)
     means=z.groupby('scenario_id')[[x,outcome]].mean()
     ax.scatter(means[x],100*means[outcome],s=26,color=col,edgecolor='white',linewidth=.5,zorder=5)
    ax.axhline(0,color='#333333',linewidth=.8)
    ax.set(xlim=(0,1),ylim=(-limit,limit),title=en,xlabel=xlabel,ylabel='RN - naive '+outcome.removeprefix('Delta_')+' (percentage points)')
    ax.grid(alpha=.15)
   ax=axs.flat[-1];ax.axis('off')
   handles=[plt.Line2D([0],[0],marker='o',color='none',markerfacecolor=c,markeredgecolor='none',label=l,markersize=6) for l,c in zip(SHORT,COLORS)]
   ax.legend(handles=handles,loc='upper left',frameon=False,fontsize=9)
   ax.text(0,.05,'Small points: individual repetitions\nLarge points: scenario means\n100 repetitions per scenario and engine\nFinal train + validation RN selection\nReconstructed sets; no archived index match',transform=ax.transAxes,fontsize=9,va='bottom',linespacing=1.5)
   fig.suptitle('RN selection and matched performance | '+design.replace('_',' '),fontsize=16,y=.98)
   fig.subplots_adjust(left=.075,right=.985,top=.91,bottom=.08,hspace=.35,wspace=.25)
   pdf.savefig(fig);plt.close(fig)
 manifest.append(dict(file=p.name,pages=2))
save_scatter('rn_purity_vs_delta_auroc','RN_precision','Delta_AUROC','RN precision')
save_scatter('rn_recall_vs_delta_auroc','RN_negative_recall','Delta_AUROC','True-negative recall')
save_scatter('rn_selection_fraction_vs_delta_auroc','RN_selection_fraction','Delta_AUROC','RN fraction of U')
save_scatter('rn_purity_vs_delta_ap','RN_precision','Delta_AP','RN precision')
if not INTERACTION_ONLY:
 f=read_table('rn_family_balanced_summary')
 p=H/'rn_family_summary.pdf';assert not p.exists()
 with PdfPages(p) as pdf:
  for outcome in ['Delta_AUROC','Delta_AP']:
   fig,axs=plt.subplots(1,2,figsize=(11.8,8.4),sharey=True)
   maxval=float(f[outcome].abs().max()*100);lim=math.ceil(maxval/2)*2
   ims=[]
   for ax,design in zip(axs,['one_sample','two_sample']):
    v=f[(f.design==design)&(f.summary_type=='within_family')].pivot(index='scenario_family',columns='engine',values=outcome).reindex(index=FAMILIES,columns=ENGINES)*100
    eq=f[(f.design==design)&(f.summary_type=='equal_10_family_mean')].set_index('engine')[outcome].reindex(ENGINES).to_numpy()*100
    vals=np.vstack([v.to_numpy(),eq])
    im=ax.imshow(vals,cmap='RdBu',norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim),aspect='auto');ims.append(im)
    for (i,j),value in np.ndenumerate(vals):ax.text(j,i,f'{value:+.2f}',ha='center',va='center',color='white' if abs(value)>.55*lim else '#202020',fontsize=10)
    ax.set_xticks(range(5),ENGINES,rotation=35,ha='right');ax.set_yticks(range(11),SHORT+['Equal-family mean']);ax.set_title(design.replace('_',' '))
    ax.axhline(9.5,color='black',linewidth=1.8)
   fig.suptitle('Fixed RN - naive '+outcome.removeprefix('Delta_')+' by scenario family',fontsize=16,y=.97)
   fig.subplots_adjust(left=.25,right=.9,top=.89,bottom=.16,wspace=.13)
   cax=fig.add_axes([.92,.20,.016,.61]);fig.colorbar(ims[0],cax=cax,label='Difference (percentage points)')
   fig.text(.25,.025,'Each cell averages settings within a family; the final row weights all ten families equally.\nThe five columns are fixed matched engine comparisons. No group maximum is taken.',fontsize=10)
   pdf.savefig(fig);plt.close(fig)
 manifest.append(dict(file=p.name,pages=2))
p=H/'rn_interaction_diagnostic.pdf';assert not p.exists() or INTERACTION_ONLY
scenarios=[1,26,38,39,40];labels=['Reference','p = 500','Curved','Interaction','Unequal covariance']
cat=s[(s.engine=='CatBoost')&s.scenario_number.isin(scenarios)]
with PdfPages(p) as pdf:
 for design in ['one_sample','two_sample']:
  fig,axs=plt.subplots(2,3,figsize=(12,8.3));z=cat[cat.design==design]
  specs=[('RN_precision','RN precision'),('RN_negative_recall','Negative recall'),('RN_selection_fraction','Selected fraction of U'),('Delta_AUROC','RN - naive AUROC (pp)'),('Delta_AP','RN - naive AP (pp)'),('hardest_negative_quartile_recall','Recall of hardest negative quartile')]
  for ax,(col,title) in zip(axs.flat,specs):
   for i,num in enumerate(scenarios):
    vals=z[z.scenario_number==num][col].dropna().to_numpy();scale=100 if col.startswith('Delta') else 1
    # Fixed deterministic jitter, only for visibility; no subsampling.
    jitter=np.linspace(-.15,.15,len(vals))
    ax.scatter(i+jitter,scale*vals,s=8,alpha=.3,color=COLORS[FAMILIES.index(z[z.scenario_number==num].scenario_family.iloc[0])],linewidth=0,rasterized=True)
    mean=scale*vals.mean();se=scale*vals.std(ddof=1)/math.sqrt(len(vals))
    ax.errorbar(i,mean,yerr=1.96*se,fmt='o',color='black',markersize=5,capsize=3,zorder=5)
   ax.set_xticks(range(5),labels,rotation=30,ha='right');ax.set_title(title);ax.grid(axis='y',alpha=.2)
   if col.startswith('Delta'):
    allv=cat[col].dropna()*100;lim=math.ceil(abs(allv).max()/5)*5;ax.set_ylim(-lim,lim);ax.axhline(0,color='black',linewidth=.7)
   else:ax.set_ylim(0,1)
  fig.suptitle('RN-CatBoost mechanism contrasts | '+design.replace('_',' '),fontsize=16,y=.98)
  fig.subplots_adjust(left=.07,right=.985,top=.90,bottom=.20,wspace=.27,hspace=.62)
  fig.text(.07,.015,'Points: all 100 repetitions. Black symbols: means with 95% Monte Carlo intervals within each fixed scenario.\nHardest negatives are the top DGP likelihood-ratio quartile of true negatives in U, used retrospectively only.',fontsize=9)
  pdf.savefig(fig);plt.close(fig)
 # Score-distribution page: means and average within-set quantiles, not uncertainty intervals.
 fig,axs=plt.subplots(2,2,figsize=(12,8.5),sharex=True)
 groups=['selected_true_negative','unselected_true_negative','hidden_positive_U'];gc=['#287A57','#3C78A8','#C14444']
 for row,design in enumerate(['one_sample','two_sample']):
  z=cat[cat.design==design]
  for col,score in enumerate(['score','LR']):
   ax=axs[row,col]
   for j,(g,color) in enumerate(zip(groups,gc)):
    mean=[];lo=[];hi=[]
    for num in scenarios:
     v=z[z.scenario_number==num]
     mean.append(v[f'{score}_{g}_q50'].mean());lo.append(v[f'{score}_{g}_q25'].mean());hi.append(v[f'{score}_{g}_q75'].mean())
    xx=np.arange(5)+(j-1)*.19
    ax.errorbar(xx,mean,yerr=[np.array(mean)-lo,np.array(hi)-mean],fmt='o',color=color,capsize=3,markersize=4,label=g.replace('_',' '))
   ax.set_xticks(range(5),labels,rotation=25,ha='right');ax.grid(axis='y',alpha=.2)
   ax.set_title(design.replace('_',' ')+' | '+('Original logistic spy score' if score=='score' else 'Exact DGP log likelihood ratio'))
   if score=='score':ax.set_ylim(0,1)
   else:ax.axhline(0,color='gray',linewidth=.7)
 axs[0,0].legend(loc='upper right',fontsize=8,frameon=False)
 fig.suptitle('RN-CatBoost: composition of the selected and discarded sets',fontsize=16,y=.98)
 fig.subplots_adjust(left=.075,right=.985,top=.90,bottom=.20,wspace=.23,hspace=.50)
 fig.text(.075,.02,'Symbols show the mean across 100 repetitions of each within-set median; bars show mean within-set quartiles.\nThese bars describe score distributions, not Monte Carlo uncertainty. True labels and DGP scores enter only retrospective diagnosis.',fontsize=9)
 pdf.savefig(fig);plt.close(fig)
if not INTERACTION_ONLY:manifest.append(dict(file=p.name,pages=3))
(H/'figure_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
