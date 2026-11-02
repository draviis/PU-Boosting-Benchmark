"""Deterministic matplotlib figures; no model fitting or score selection."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator
from display_specs import *
from display_data import SELECTORS,SELNAMES
COLORS=["#0072B2","#D55E00","#009E73","#CC79A7","#E69F00","#56B4E9"]
MARKERS=["o","s","^","D","v","P"]
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.labelsize":10,"axes.titlesize":11,
 "xtick.labelsize":9,"ytick.labelsize":9,"legend.fontsize":9,"figure.dpi":100,
 "savefig.dpi":300,"pdf.fonttype":42,"ps.fonttype":42,"axes.spines.top":False,
 "axes.spines.right":False,"axes.unicode_minus":False,"axes.grid":False,
 "lines.linewidth":1.25,"lines.markersize":4.5})
def title(ax,label,text):ax.set_title(f"({label}) {text}",loc="left",fontsize=11,pad=9)
def short(m):return SHORT.get(m,METHODS[m].replace(" ","\n",1))
def heat(ax,values,rows,cols,vmin,vmax,cmap="viridis",zero=False,showy=True):
 kw={"norm":TwoSlopeNorm(vmin=vmin,vcenter=0,vmax=vmax)} if zero else {"vmin":vmin,"vmax":vmax}
 im=ax.pcolormesh(np.arange(len(cols)+1),np.arange(len(rows)+1),np.ma.masked_invalid(values),cmap=cmap,rasterized=False,shading="flat",**kw)
 ax.set_ylim(len(rows),0)
 ax.set_yticks(np.arange(len(rows))+.5,rows if showy else [""]*len(rows))
 ax.set_xticks(np.arange(len(cols))+.5,cols,rotation=43,ha="right",rotation_mode="anchor")
 ax.tick_params(length=0,pad=4)
 for spine in ax.spines.values():spine.set_visible(False)
 return im
def cb(fig,im,cax,label):
 c=fig.colorbar(im,cax=cax);c.solids.set_rasterized(False);c.solids.set_edgecolor("face");c.set_label(label,fontsize=10);c.ax.tick_params(labelsize=9)
def forest_panel(ax,g,rows,categories,labels,ycol,catcol,mean="mean",lo="lo",hi="hi",scale=1):
 offsets=np.linspace(-.23,.23,len(categories))
 for j,(cat,label) in enumerate(zip(categories,labels)):
  a=g[g[catcol]==cat].set_index(ycol).reindex(rows)
  x=scale*a[mean].to_numpy();low=scale*a[lo].to_numpy();high=scale*a[hi].to_numpy()
  ax.errorbar(x,np.arange(len(rows))+offsets[j],xerr=np.vstack([x-low,high-x]),fmt=MARKERS[j],color=COLORS[j],capsize=2,label=label,ms=4)
 ax.set_yticks(range(len(rows)),rows);ax.invert_yaxis();ax.axvline(0,color="#666666",lw=.8,ls="--")
 ax.xaxis.set_major_locator(MaxNLocator(5))
 ax.margins(y=.10)
def figures(id,d):
 if id=="fig_core_fixed_methods":
  fig=plt.figure(figsize=(7.2,8.2))
  for row,metric in enumerate(["AUROC","AP"]):
   extent=max(1,float(d[d.metric==metric].difference_pp.abs().max()))*1.04
   for col,design in enumerate(DESIGNS):
    ax=fig.add_axes([.255+.345*col,.57-.43*row,.275,.315])
    a=d[(d.design==design)&(d.metric==metric)]
    mat=a.pivot(index="family_id",columns="method_number",values="difference_pp").reindex(index=range(1,11),columns=[5,10,25,20,27])
    im=heat(ax,mat.to_numpy(),FSHORT,[short(m) for m in [5,10,25,20,27]],-extent,extent,"RdBu_r",True,col==0)
    title(ax,"abcd"[row*2+col],f"{DNAME[design]} {metric}")
   cb(fig,im,fig.add_axes([.91,.57-.43*row,.018,.315]),f"{metric} difference (pp)")
  yield fig,"Ten original scenario families; fixed configurations",None
 elif id=="fig_nonlinear_decomposition":
  fig,axes=plt.subplots(2,2,figsize=(7.2,6.4))
  fig.subplots_adjust(left=.10,right=.98,top=.94,bottom=.16,hspace=.60,wspace=.27)
  cats=["learner_gain","CS_additional_gain","RN_additional_gain"];labels=["Learner gain","CS addition","RN addition"]
  for row,metric in enumerate(["AUROC","AP"]):
   allg=d[d.metric==metric];low=allg.MC_lower_pp.min();high=allg.MC_upper_pp.max();pad=(high-low)*.08
   for col,design in enumerate(DESIGNS):
    ax=axes[row,col];g=allg[allg.design==design]
    forest_panel(ax,g,[f"NL{i:02}" for i in range(1,7)],cats,labels,"scenario_id","contrast","mean_difference_pp","MC_lower_pp","MC_upper_pp")
    ax.set_xlim(low-pad,high+pad);title(ax,"abcd"[row*2+col],f"{DNAME[design]} {metric}")
    ax.set_xlabel(f"Difference in {metric}\n(percentage points)")
  handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=3,loc="lower center",bbox_to_anchor=(.5,.01),frameon=False)
  yield fig,"CatBoost learner-versus-strategy decomposition",None
 elif id in ["fig_s_core_full_fixed_methods","fig_s_nonlinear_full"]:
  nonlinear=id=="fig_s_nonlinear_full"
  for metric in ["AUROC","AP"]:
   for block,methods in BLOCKS:
    fig=plt.figure(figsize=(7.2,6.1 if not nonlinear else 5.0))
    for col,design in enumerate(DESIGNS):
     ax=fig.add_axes([.255+.345*col,.245,.275,.60])
     g=d[(d.design==design)&d.method_number.isin(methods)]
     if nonlinear:
      mat=g.pivot(index="scenario_id",columns="method_number",values=metric+"_mean").reindex(index=[f"NL{i:02}" for i in range(1,7)],columns=methods)
      rows=mat.index.tolist()
     else:
      g=g[g.metric==metric];mat=g.pivot(index="family_id",columns="method_number",values="mean_value").reindex(index=range(1,11),columns=methods);rows=FSHORT
     im=heat(ax,mat.to_numpy(),rows,[short(m) for m in methods],0,1,showy=col==0)
     title(ax,"ab"[col],DNAME[design])
     if nonlinear:
      countmat=g.pivot(index="scenario_id",columns="method_number",values=metric+"_n").reindex(index=mat.index,columns=methods)
      for y,x in zip(*np.where(countmat.to_numpy()<100)):ax.text(x+.5,y+.5,"*",ha="center",va="center",fontsize=12,color="white")
    cb(fig,im,fig.add_axes([.91,.245,.018,.60]),metric+" (0 to 1)")
    fig.text(.255,.945,block+" / "+metric,fontsize=11)
    note=("All six nonlinear DGPs. * n=99; all other cells n=100. Missing refits are not imputed." if nonlinear else "Within-family means over settings; all 25 PU configurations retained across blocks.")
    yield fig,block+" / "+metric,note
 elif id=="fig_s_bayes_regret":
  vmin=min(0,float(d.regret.min()));vmax=float(d.regret.max())
  for start in [1,21]:
   fig=plt.figure(figsize=(7.2,7.5))
   for col,design in enumerate(DESIGNS):
    ax=fig.add_axes([.10+.38*col,.18,.31,.72]);g=d[d.design==design]
    rows=list(range(start,start+20));mat=g.pivot(index="scenario_index",columns="method_number",values="regret").reindex(index=rows,columns=REPS)
    im=heat(ax,mat.to_numpy(),[f"S{i:03}" for i in rows],[short(m) for m in REPS],vmin,vmax,"magma_r",showy=True)
    title(ax,"ab"[col],DNAME[design])
   cb(fig,im,fig.add_axes([.865,.18,.018,.72]),"Bayes AUROC - method AUROC")
   yield fig,f"Original settings {start:02}--{start+19:02}","All 40 settings use the existing validated Task-2 Bayes reference. Lower regret is better. The six conceptual representatives are fixed in advance."
 elif id=="fig_s_threshold_sensitivity":
  x=d[(d.study=="simulation")&d.method_number.isin(REPS)]
  rules=["B_same_model_quantile","C_PU_BA","D_same_model_oracle"];labels=["A_same_model quantile","PU-estimated BA pilot","Oracle BA diagnostic"]
  for design in DESIGNS:
   scenarios=sorted(x[x.design==design].scenario_id.unique())
   for begin in [0,4]:
    fig,axes=plt.subplots(2,2,figsize=(7.2,7.0));fig.subplots_adjust(left=.09,right=.98,bottom=.22,top=.88,hspace=.75,wspace=.25)
    for i,(ax,scenario) in enumerate(zip(axes.flat,scenarios[begin:begin+4])):
     for j,rule in enumerate(rules):
      g=x[(x.design==design)&(x.scenario_id==scenario)&(x.rule==rule)].set_index("method_number").reindex(REPS)
      ax.plot(range(6),g.BalancedAccuracy,marker=MARKERS[j],color=COLORS[j],label=labels[j],lw=.8)
     ax.set_xticks(range(6),[short(m) for m in REPS],rotation=45,ha="right");ax.set_ylim(0,1)
     ax.set_ylabel("Balanced accuracy");title(ax,"abcd"[i],scenario)
    handles,labels0=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels0,ncol=1,loc="lower center",bbox_to_anchor=(.5,.0),frameon=False)
    fig.text(.09,.985,DNAME[design]+" / one repetition per setting",fontsize=11,va="top")
    yield fig,DNAME[design]+f" / simulation pilot {begin+1}--{begin+4}",None
 elif id=="fig_s_rn_mechanism":
  f=d.groupby(["design","scenario_family"],sort=False).mean(numeric_only=True).reset_index()
  order=["reference","prevalence","positive-label availability","class separation","dimensionality","predictor correlation","joint prevalence-label availability","curved boundary","feature interaction","unequal class covariance"]
  # Use source family ids via original scenario order rather than fuzzy string ordering.
  order=d.sort_values("scenario_number").scenario_family.drop_duplicates().tolist();assert len(order)==10
  fig,axes=plt.subplots(2,2,figsize=(7.2,7.65));fig.subplots_adjust(left=.25,right=.98,bottom=.20,top=.94,hspace=.42,wspace=.28)
  for col,design in enumerate(DESIGNS):
   g=f[f.design==design].set_index("scenario_family").reindex(order)
   for j,(field,label) in enumerate([("RN_precision_mean","RN purity"),("RN_negative_recall_mean","True-negative recall / retention"),("RN_selection_fraction_mean","U retention")]):
    axes[0,col].plot(g[field],np.arange(10)+(j-1)*.18,MARKERS[j],color=COLORS[j],label=label)
   for j,metric in enumerate(["AUROC","AP"]):
    axes[1,col].plot(100*g["Delta_"+metric+"_mean"],np.arange(10)+(j-.5)*.18,MARKERS[j],color=COLORS[j],label=metric)
   for row in [0,1]:
    ax=axes[row,col];ax.set_yticks(range(10),FSHORT if col==0 else [""]*10);ax.invert_yaxis();title(ax,"abcd"[row*2+col],DNAME[design])
   axes[0,col].set_xlim(0,1);axes[0,col].set_xlabel("Fraction")
   axes[1,col].axvline(0,color="gray",ls="--",lw=.8);axes[1,col].set_xlabel("RN minus naive CatBoost\n(percentage points)")
  lim=100*f[["Delta_AUROC_mean","Delta_AP_mean"]].abs().to_numpy().max()*1.15
  for ax in axes[1]:ax.set_xlim(-lim,lim)
  handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=1,loc="lower center",bbox_to_anchor=(.5,.012),frameon=False)
  handles,labels=axes[1,0].get_legend_handles_labels();axes[1,1].legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,-.25),ncol=2,frameon=False)
  yield fig,"RN CatBoost: negative-set quality and ranking changes","Ten design-defined families; settings are equally weighted within each family. True-negative recall and the fraction of available true negatives retained are the same quantity."
 elif id=="fig_s_rn_selector_sensitivity":
  for design in DESIGNS:
   fig,axes=plt.subplots(1,2,figsize=(7.2,5.3));fig.subplots_adjust(left=.19,right=.98,top=.90,bottom=.30,wspace=.35)
   for i,metric in enumerate(["AUROC","AP"]):
    ax=axes[i];g=d[(d.design==design)&(d.metric==metric)]
    for j,cond in enumerate(SELECTORS):
     a=g[g.condition==cond].set_index("scenario_number").reindex([1,26,38,39,40])
     ax.plot(a["mean"],np.arange(5)+(j-2)*.11,MARKERS[j],color=COLORS[j],label=SELNAMES[j])
    ax.set_yticks(range(5),[SCENARIOS[n] for n in [1,26,38,39,40]] if i==0 else [""]*5);ax.invert_yaxis();ax.set_xlim(0,1);ax.set_xlabel(metric+" (0 to 1)")
    title(ax,"ab"[i],metric)
   handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=2,loc="lower center",bbox_to_anchor=(.5,.02),frameon=False)
   fig.text(.19,.985,DNAME[design]+" / fixed CatBoost",fontsize=11,va="top")
   yield fig,DNAME[design]+" / five selection diagnostics",None
 elif id=="fig_s_supplied_prior_sensitivity":
  a=d.groupby(["design","method_number","metric","prior_ratio"])["mean"].mean().reset_index()
  fig,axes=plt.subplots(2,2,figsize=(7.2,6.6));fig.subplots_adjust(left=.12,right=.98,top=.93,bottom=.18,hspace=.55,wspace=.30)
  for row,metric in enumerate(["AUROC","AP"]):
   lim=max(.1,a[a.metric==metric]["mean"].abs().max())*1.15
   for col,design in enumerate(DESIGNS):
    ax=axes[row,col]
    for j,m in enumerate([4,5,26,27]):
     g=a[(a.design==design)&(a.metric==metric)&(a.method_number==m)]
     ax.plot(g.prior_ratio,g["mean"],marker=MARKERS[j],color=COLORS[j],label=METHODS[m])
    ax.axhline(0,color="gray",ls="--",lw=.8);ax.set_ylim(-lim,lim);ax.set_xticks([.8,.9,1.,1.1,1.2]);ax.set_xlabel("Supplied / true prior")
    ax.set_ylabel(f"{metric} change\n(percentage points)");title(ax,"abcd"[row*2+col],DNAME[design])
  handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=2,loc="lower center",bbox_to_anchor=(.5,.015),frameon=False)
  yield fig,"Equal-eight-setting supplied-prior sensitivity","Curves are equally weighted means of eight designed settings, not confidence bands or estimates over a random population of scenarios."
 elif id=="fig_s_estimated_prior":
  for (study,design,budget),g in d.groupby(["study","design","budget"],sort=True):
   keys=sorted(g.scenario_id.unique());n=len(keys)
   fig,axes=plt.subplots(1,3,figsize=(7.2,5.6));fig.subplots_adjust(left=.22,right=.98,top=.89,bottom=.22,wspace=.30)
   cover=g.drop_duplicates("scenario_id").set_index("scenario_id").reindex(keys)
   # Whiskers are distribution quantiles; the mean can be outside the central quantile range.
   axes[0].hlines(np.arange(n),cover.ratio_q05,cover.ratio_q95,color=COLORS[0],lw=1)
   axes[0].plot(cover.ratio_mean,np.arange(n),"o",color=COLORS[0])
   axes[0].axvline(1,color="gray",ls="--",lw=.8);axes[0].set_xlim(0,max(1.3,float(cover[["ratio_q95","ratio_mean"]].to_numpy().max())*1.1))
   axes[0].set_xlabel("Estimated / true\nprior");title(axes[0],"a","KM2 prior ratio")
   for i,metric in enumerate(["AUROC","AP"],1):
    x=g[g.metric==metric];ax=axes[i]
    for j,m in enumerate([4,5,26,27]):
     a=x[x.method_number==m].set_index("scenario_id").reindex(keys)
     ax.plot(a["mean"],np.arange(n)+(j-1.5)*.12,MARKERS[j],color=COLORS[j],label=METHODS[m])
    lim=max(1.,float(d[(d.study==study)&(d.metric==metric)]["mean"].abs().max()))*1.1
    ax.set_xlim(-lim,lim);ax.axvline(0,color="gray",ls="--",lw=.8);ax.set_xlabel(metric+" change\n(percentage points)");title(ax,"abc"[i],metric)
   for i,ax in enumerate(axes):
    ax.set_yticks(range(n),[k.replace("_"," ") for k in keys] if i==0 else [""]*n);ax.invert_yaxis();ax.xaxis.set_major_locator(MaxNLocator(3))
   handles,labels=axes[1].get_legend_handles_labels();fig.legend(handles,labels,ncol=2,loc="lower center",bbox_to_anchor=(.5,.015),frameon=False)
   fig.text(.22,.985,f"KM2 / {'simulation' if study=='main' else 'real data'} / {DNAME[design]}"+(f" / {budget}" if budget else ""),fontsize=11,va="top")
   yield fig,f"{study} / {DNAME[design]} / {budget}","Performance panels show paired means. The accompanying table and numeric CSV retain uncertainty and case counts. Prior whiskers are empirical 5th--95th percentiles, not confidence intervals."
 elif id=="fig_s_neural_simulation":
  cats=["nnPU / proxy-AUROC","nnPU / risk","uPU / proxy-AUROC","uPU / risk"]
  for design in DESIGNS:
   fig,axes=plt.subplots(1,2,figsize=(7.2,5.1));fig.subplots_adjust(left=.19,right=.98,top=.90,bottom=.29,wspace=.3)
   rows=[1,12,26,38,39,40]
   for i,metric in enumerate(["AUROC","AP"]):
    g=d[(d.design==design)&(d.metric==metric)]
    forest_panel(axes[i],g,rows,cats,cats,"scenario_number","contrast")
    axes[i].set_yticks(range(6),[SCENARIOS[r] for r in rows] if i==0 else [""]*6)
    allg=d[d.metric==metric];lo=allg.lo.min();hi=allg.hi.max();pad=(hi-lo)*.10
    axes[i].set_xlim(lo-pad,hi+pad);axes[i].set_xlabel(f"Neural minus linear {metric}\n(percentage points)");title(axes[i],"ab"[i],metric)
   handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=2,loc="lower center",bbox_to_anchor=(.5,.015),frameon=False)
   fig.text(.19,.985,DNAME[design]+" / exploratory, 10 paired repetitions",fontsize=11,va="top")
   yield fig,DNAME[design]+" / matched linear-neural contrasts",None
 else:raise KeyError(id)
