"""Publication figures for the completed core CatBoost selector intervention."""
from pathlib import Path
import sys
import json
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
H=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.labelsize':10,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
names=['Naive_all_U','All_U_RN_fixed_final','RN_random_size_matched','RN_logistic','RN_nonlinear_size_matched','RN_nonlinear','Oracle_true_negative_size_matched']
labels=['Naive all U\n(archived)','All U, RN\nparameters','Random\nsize matched','Logistic RN','Nonlinear\nsize matched','Nonlinear\nfull set','Oracle TN\nsize matched']
short=dict(zip(names,labels))
colors=dict(zip(names,['#555555','#999999','#A67738','#C55A47','#007A78','#456FB4','#8064A2']))
scenarios=[1,26,38,39,40];slabels=['Reference','p = 500','Curved','Interaction','Unequal\ncovariance']
q=pd.DataFrame(json.loads((H/'plot_summary.json').read_text()))
q=q[(q.scope=='core')&(q.stage=='final_train_plus_validation')]
u=pd.DataFrame(json.loads((H/'plot_condition_uncertainty.json').read_text()))
u=u[(u.scope=='core')&(u.stage=='final_train_plus_validation')]
mc=pd.DataFrame(json.loads((H/'plot_paired_uncertainty.json').read_text()))
mc=mc[(mc.scope=='core')&(mc.stage=='final_train_plus_validation')]
dist=pd.DataFrame(json.loads((H/'plot_interaction_distribution.json').read_text()))
manifest=[]
ONLY=set(sys.argv[1:])
def output(name,pages,fn):
 if ONLY and name not in ONLY:
  manifest.append(dict(file=name,pages=pages));return
 path=H/name
 if path.exists() and not ONLY:raise FileExistsError(path)
 with PdfPages(path,metadata={'Title':name.replace('_',' ').replace('.pdf',''),'Author':'PU Boosting Benchmark'}) as pdf:
  fn(pdf)
 manifest.append(dict(file=name,pages=pages))
def probability_axis(ax):
 ax.set_ylim(0,1);ax.set_yticks(np.linspace(0,1,6));ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
def points(ax,de,metric,scenario=39,order=names):
 data=u[(u.design==de)&(u.scenario_number==scenario)].set_index('condition')
 for i,c in enumerate(order):
  row=data.loc[c];mean=row[metric+'_mean'];se=row[metric+'_MCSE']
  ax.errorbar(i,mean,yerr=1.96*se,fmt='o',color=colors[c],capsize=3,markersize=6,lw=1.1)
 ax.set_xticks(range(len(order)));ax.set_xticklabels([short[c] for c in order],fontsize=8);ax.set_xlim(-.65,len(order)-.35);probability_axis(ax);ax.set_ylabel(metric)
def fig1(pdf):
 fig,axes=plt.subplots(2,2,figsize=(12.6,9),sharey=True)
 for r,de in enumerate(['one_sample','two_sample']):
  for col,metric in enumerate(['AUROC','AP']):
   points(axes[r,col],de,metric);axes[r,col].set_title(de.replace('_',' ').title()+' | '+metric)
 fig.suptitle('Feature interaction: fixed final learner, different negative subsets',fontsize=15,y=.97)
 fig.subplots_adjust(left=.06,right=.985,top=.90,bottom=.12,hspace=.48,wspace=.16)
 fig.text(.06,.035,'Points: means of 100 paired repetitions. Bars: 95% Monte Carlo intervals within this fixed scenario.\nRandom and oracle metrics average ten draws per repetition. All-U control uses RN parameters and RN row order; archived naive uses its original parameters and row order.',fontsize=9)
 pdf.savefig(fig);plt.close(fig)
output('interaction_selector_comparison.pdf',1,fig1)
def fig2(pdf):
 cs=['RN_logistic','RN_nonlinear','RN_nonlinear_size_matched','RN_random_size_matched','Oracle_true_negative_size_matched']
 fig,axes=plt.subplots(2,3,figsize=(13.5,8.7))
 for row,de in enumerate(['one_sample','two_sample']):
  for col,(metric,title) in enumerate([('RN_precision','RN purity'),('RN_negative_recall','Negative recall'),('RN_selection_fraction','Selected U fraction')]):
   ax=axes[row,col];matrix=np.array([[float(q[(q.design==de)&(q.scenario_number==s)&(q.condition==c)][metric].iloc[0]) for s in scenarios] for c in cs])
   im=ax.imshow(matrix,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
   for i in range(len(cs)):
    for j in range(len(scenarios)):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if matrix[i,j]>.63 else '#17212D')
   ax.set_xticks(range(5));ax.set_xticklabels(slabels,fontsize=9,rotation=28,ha='right',rotation_mode='anchor')
   ax.set_yticks(range(5));ax.set_yticklabels([short[c].replace('\n',' ') for c in cs] if col==0 else ['']*5,fontsize=9)
   ax.set_title(de.replace('_',' ').title()+' | '+title)
 fig.suptitle('Selection quality across the five core scenarios',fontsize=15,y=.97)
 fig.subplots_adjust(left=.20,right=.92,top=.90,bottom=.18,wspace=.16,hspace=.44)
 cax=fig.add_axes([.945,.19,.012,.65]);fig.colorbar(im,cax=cax,label='Fraction (full 0–1 scale)')
 fig.text(.20,.035,'Each cell averages 100 repetitions. Random/oracle draws are first averaged within each repetition.\nHigh purity can coexist with low negative coverage. Oracle subsets are retrospective diagnostics.',fontsize=9)
 pdf.savefig(fig);plt.close(fig)
output('selector_quality_by_scenario.pdf',1,fig2)
def fig3(pdf):
 fig,axes=plt.subplots(3,2,figsize=(11.8,9.6),sharex=True,sharey=True)
 for col,de in enumerate(['one_sample','two_sample']):
  for row,(metric,title) in enumerate([('jaccard','Jaccard overlap'),('logistic_in_other','Fraction of logistic RN retained'),('other_in_logistic','Fraction of nonlinear RN shared')]):
   ax=axes[row,col]
   for c,shift,mark in [('RN_nonlinear',-.06,'o'),('RN_nonlinear_size_matched',.06,'s')]:
    data=q[(q.design==de)&q.condition.eq(c)].set_index('scenario_number')
    vals=[data.loc[s,metric] for s in scenarios];ax.plot(np.arange(5)+shift,vals,marker=mark,color=colors[c],label=short[c].replace('\n',' '),lw=1)
   probability_axis(ax);ax.set_title(de.replace('_',' ').title()+' | '+title);ax.set_xticks(range(5));ax.set_xticklabels(slabels,fontsize=9)
 axes[0,0].legend(loc='lower left',frameon=False)
 fig.suptitle('How much does nonlinear selection change the selected set?',fontsize=15,y=.98)
 fig.subplots_adjust(left=.07,right=.98,top=.92,bottom=.10,wspace=.17,hspace=.36)
 fig.text(.07,.025,'Means across 100 paired repetitions. Equal set size does not imply equal selected observations.\nJaccard = intersection / union; the two directional fractions use different denominators.',fontsize=9)
 pdf.savefig(fig);plt.close(fig)
output('selector_overlap.pdf',1,fig3)
def fig4(pdf):
 contrasts=[('All_U_RN_fixed_final minus RN_random_size_matched','All U minus random\n(same final parameters)'),('RN_nonlinear_size_matched minus RN_logistic','Nonlinear size matched\nminus logistic'),('Oracle_true_negative_size_matched minus RN_random_size_matched','Oracle negatives\nminus random'),('Naive_all_U minus All_U_RN_fixed_final','Archived naive minus\nRN-protocol all U')]
 for de in ['one_sample','two_sample']:
  fig,axes=plt.subplots(2,2,figsize=(12.6,9))
  for col,metric in enumerate(['AUROC','AP']):
   points(axes[0,col],de,metric);axes[0,col].set_title('Feature interaction | '+metric)
   ax=axes[1,col]
   bound=.005
   for i,(contrast,label) in enumerate(contrasts):
    r=mc[(mc.design==de)&mc.scenario_number.eq(39)&mc.outcome.eq(metric)&mc.contrast.eq(contrast)].iloc[0]
    mean=100*r.mean_paired_difference;err=196*r.MCSE;bound=max(bound,abs(mean)+err)
    ax.errorbar(i,mean,yerr=err,fmt='o',capsize=3,color=['#A67738','#007A78','#8064A2','#555555'][i],markersize=7)
   bound=max(2,np.ceil(bound/2)*2);ax.set_ylim(-bound,bound);ax.axhline(0,color='#333333',lw=.8);ax.grid(axis='y',alpha=.18)
   ax.set_xticks(range(len(contrasts)));ax.set_xticklabels([label for _,label in contrasts],fontsize=8);ax.set_xlim(-.5,3.5)
   ax.set_ylabel('Paired '+metric+' difference (pp)');ax.set_title('Controlled contrasts')
  fig.suptitle(de.replace('_',' ').title()+': separating retained size and selection composition',fontsize=15,y=.98)
  fig.subplots_adjust(left=.07,right=.985,top=.90,bottom=.14,wspace=.19,hspace=.50)
  fig.text(.07,.035,'Controlled subset fits share RN parameters and the P-then-U row-order rule. Archived naive also differs in row order.\nRandom/oracle outcomes average ten draws; intervals use 100 paired case differences. These contrasts do not imply a unique causal decomposition.',fontsize=9)
  pdf.savefig(fig);plt.close(fig)
output('size_vs_identity_decomposition.pdf',2,fig4)
def fig5(pdf):
 cs=['RN_logistic','RN_nonlinear_size_matched','RN_nonlinear','RN_random_size_matched']
 variables=[('exact_DGP_log_likelihood_ratio','Exact DGP log likelihood ratio'),('x0','Informative coordinate x0'),('x1','Informative coordinate x1'),('x0_x1_product','Interaction product x0 × x1'),('x2','Main-effect coordinate x2')]
 for de in ['one_sample','two_sample']:
  fig,axes=plt.subplots(2,3,figsize=(14,9.5));flat=axes.ravel()
  for ax,(var,title) in zip(flat,variables):
   yy=[];lab=[]
   base=dist[(dist.design==de)&dist.condition.eq('RN_logistic')&dist.group.eq('full_true_negative')&dist.score_type.eq(var)].iloc[0]
   ax.axvline(base.q50,color='#666666',ls='--',lw=.8,label='Full-negative median')
   for i,c in enumerate(cs):
    for j,(group,marker) in enumerate([('selected_true_negative','o'),('discarded_true_negative','s')]):
     r=dist[(dist.design==de)&dist.condition.eq(c)&dist.group.eq(group)&dist.score_type.eq(var)].iloc[0]
     y=-(2*i+j);ax.plot([r.q10,r.q90],[y,y],color=colors[c],lw=1,alpha=.7);ax.plot([r.q25,r.q75],[y,y],color=colors[c],lw=3,alpha=.8);ax.scatter(r.q50,y,color=colors[c],s=24,marker=marker,zorder=3)
     yy.append(y);lab.append({'RN_logistic':'Logistic RN','RN_nonlinear_size_matched':'Nonlinear SM','RN_nonlinear':'Nonlinear RN','RN_random_size_matched':'Random SM'}[c]+' | '+('selected' if j==0 else 'discarded'))
   ax.set_yticks(yy);ax.set_yticklabels(lab,fontsize=8.5);ax.set_title(title);ax.grid(axis='x',alpha=.18);ax.set_ylim(-7.7,.7)
  flat[5].axis('off')
  flat[5].text(.02,.90,'How to read these panels',weight='bold',fontsize=12)
  flat[5].text(.02,.79,'Circle: selected true negatives\nSquare: discarded true negatives\n\nThin line: 10th–90th quantiles\nThick line: 25th–75th quantiles\nPoint: median\nDashed line: full-negative median\n\nSM = size matched\n\nMarks average within-set quantiles\nfrom 100 repetitions. Random\nsubsets first average ten draws\nwithin each repetition.\n\nThese are not pooled quantiles\nor confidence intervals.',va='top',fontsize=10,linespacing=1.5)
  fig.suptitle('Feature interaction | '+de.replace('_',' ').title()+' | negative-set composition',fontsize=15,y=.98)
  fig.subplots_adjust(left=.14,right=.98,top=.91,bottom=.10,wspace=.75,hspace=.33)
  fig.text(.16,.025,'The Bayes score uses the exact known DGP only after selection. Informative-coordinate diagnostics also use hidden labels retrospectively.\nThe nonlinear size-matched and logistic sets have identical counts; differences in these distributions reflect selection composition.',fontsize=9)
  pdf.savefig(fig);plt.close(fig)
output('interaction_negative_distribution.pdf',2,fig5)
(H/'figure_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
