"""Publication figures, drawn only from completed Task 6 tables."""
from pathlib import Path
import json
import numpy as np,pandas as pd,matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
H=Path(__file__).resolve().parent
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.titlesize":10,"axes.labelsize":9,
                     "legend.fontsize":8,"pdf.fonttype":42,"ps.fonttype":42,
                     "axes.spines.top":False,"axes.spines.right":False})
ac=pd.read_csv(H/"prior_estimation_accuracy_case.csv",float_precision="round_trip")
mc=pd.read_csv(H/"estimated_prior_mc_uncertainty.csv",float_precision="round_trip")
diff=pd.read_csv(H/"estimated_prior_simulation_differences.csv",float_precision="round_trip")
real=pd.read_csv(H/"estimated_prior_real_summary.csv",float_precision="round_trip")
sids=[1,2,3,11,26,38,39,40]
labels={1:"Reference",2:"pi = 0.01",3:"pi = 0.05",11:"pi = 0.50",26:"p = 500",38:"Curved",39:"Interaction",40:"Unequal covariance"}
method_names={4:"Linear uPU",5:"Linear nnPU",26:"AdaPU",27:"PU gradient boosting"}
datasets={"credit_default":"Credit Default","dry_bean":"Dry Bean","htru2":"HTRU2","online_shoppers":"Online Shoppers",
          "optical_digits":"Optical Digits","phishing":"Phishing Websites","rice":"Rice","spambase":"Spambase"}
colors=dict(zip(sids,["#0072B2","#D55E00","#E69F00","#009E73","#CC79A7","#56B4E9","#6B4C9A","#666666"]))
dcolors=dict(zip(datasets,list(colors.values())))
designs=[("one_sample","One-sample"),("two_sample","Two-sample")]
manifest=[]
def path(name):
 p=H/name
 if p.exists():raise FileExistsError(p)
 return p
def finish(fig,name,pages=1):
 fig.savefig(path(name),bbox_inches="tight");plt.close(fig)
 manifest.append(dict(file=name,pages=pages))
def identity(ax):
 ax.plot([0,1],[0,1],color="#555555",ls="--",lw=1,zorder=0)
 ax.set(xlim=(0,1),ylim=(0,1));ax.grid(alpha=.18)
# Full-range axes disclose large estimation errors without truncation.
name="prior_estimation_accuracy.pdf"
with PdfPages(path(name)) as pdf:
 for study,title in [("main","Simulation priors"),("real","Semi-synthetic real-data priors")]:
  fig,axes=plt.subplots(1,2,figsize=(10.8,5.0),sharex=True,sharey=True)
  for ax,(design,dlabel) in zip(axes,designs):
   g=ac[(ac.study==study)&(ac.design==design)&(ac.status=="success")]
   categories=sids if study=="main" else list(datasets)
   for cat in categories:
    z=g[g.scenario_number==cat] if study=="main" else g[g.dataset==cat]
    ax.scatter(z.true_pi,z.pi_hat,s=12,alpha=.4,color=colors[cat] if study=="main" else dcolors[cat],
               label=labels[cat] if study=="main" else datasets[cat],rasterized=False)
   identity(ax);ax.set_title(dlabel+"; valid n = "+str(len(g)))
   ax.set_xlabel("Generating pi" if study=="main" else "Oracle empirical training-background pi")
  axes[0].set_ylabel("Estimated population prevalence")
  handles,labs=axes[0].get_legend_handles_labels()
  fig.legend(handles,labs,loc="lower center",ncol=4,frameon=False,bbox_to_anchor=(.5,-.02))
  fig.suptitle(title+": training-only KM2",y=.99)
  fig.subplots_adjust(bottom=.22,top=.88,wspace=.12)
  pdf.savefig(fig,bbox_inches="tight");plt.close(fig)
manifest.append(dict(file=name,pages=2))
fig,axes=plt.subplots(1,2,figsize=(11,6.5))
for ax,study,title in zip(axes,["main","real"],["Simulations","Semi-synthetic real data"]):
 g=ac[(ac.study==study)&(ac.design=="one_sample")&(ac.status=="success")]
 categories=sids if study=="main" else list(datasets)
 for cat in categories:
  z=g[g.scenario_number==cat] if study=="main" else g[g.dataset==cat]
  ax.scatter(z.true_c,z.c_hat,s=13,alpha=.45,color=colors[cat] if study=="main" else dcolors[cat],
             label=labels[cat] if study=="main" else datasets[cat])
 identity(ax);ax.set_title(title+"; valid n = "+str(len(g)))
 ax.set_xlabel("Oracle SCAR labeling probability c");ax.set_ylabel("Estimated c")
 ax.legend(loc="upper center",bbox_to_anchor=(.5,-.18),fontsize=7,frameon=False,ncol=2)
fig.suptitle("One-sample labeling-probability estimation",y=.99)
fig.tight_layout(rect=(0,.22,1,.94));finish(fig,"c_estimation_accuracy.pdf")
name="estimated_vs_oracle_performance.pdf"
limits={}
for metric in ["AUROC","AP"]:
 g=mc[(mc.metric==metric)&mc.method_number.isin(method_names)]
 vals=pd.concat([g.MC_lower,g.MC_upper]).dropna().to_numpy(float)*100
 limits[metric]=max(.1,float(np.max(np.abs(vals)))*1.12) if len(vals) else 1
with PdfPages(path(name)) as pdf:
 for design,dlabel in designs:
  fig,axes=plt.subplots(4,2,figsize=(11.8,12.2),sharex=True)
  for row,(n,mname) in enumerate(method_names.items()):
   for col,metric in enumerate(["AUROC","AP"]):
    ax=axes[row,col]
    z=mc[(mc.design==design)&(mc.method_number==n)&(mc.metric==metric)].set_index("scenario_number").reindex(sids)
    y=z.mean_difference.to_numpy(float)*100
    lo=z.MC_lower.to_numpy(float)*100;hi=z.MC_upper.to_numpy(float)*100
    ax.axhline(0,color="#555555",lw=.9)
    ax.errorbar(np.arange(8),y,yerr=np.vstack([y-lo,hi-y]),fmt="o",color="#176B88",capsize=3,ms=4,lw=1)
    ax.set_ylim(-limits[metric],limits[metric]);ax.grid(axis="y",alpha=.2)
    ax.set_title(mname+" - "+metric)
    ax.set_ylabel("Estimated minus oracle (pp)")
    ax.set_xticks(range(8),[labels[s] for s in sids],rotation=35,ha="right")
    for x,(yi,ni) in enumerate(zip(y,z.n_valid)):
     if np.isfinite(yi) and ni<100:ax.annotate("n="+str(int(ni)),(x,yi),xytext=(0,6),textcoords="offset points",ha="center",fontsize=7)
  fig.suptitle(dlabel+": prior-dependent methods with full original model selection",fontsize=13,y=.995)
  fig.text(.5,.005,"Points: mean paired differences. Bars: 95% Monte Carlo intervals within each fixed DGP.",ha="center",fontsize=9)
  fig.tight_layout(rect=(0,.035,1,.97));pdf.savefig(fig,bbox_inches="tight");plt.close(fig)
manifest.append(dict(file=name,pages=2))
name="prior_error_vs_auroc_loss.pdf"
with PdfPages(path(name)) as pdf:
 for design,dlabel in designs:
  fig,axes=plt.subplots(4,2,figsize=(11.8,12.2))
  for row,(n,mname) in enumerate(method_names.items()):
   g=diff[(diff.design==design)&(diff.method_number==n)]
   for col,error in enumerate(["absolute_prior_error","relative_prior_error"]):
    ax=axes[row,col]
    for sid in sids:
     z=g[g.scenario_number==sid].dropna(subset=[error,"Delta_AUROC"])
     ax.scatter(z[error],z.Delta_AUROC*100,s=10,alpha=.45,color=colors[sid],label=labels[sid])
    ax.axhline(0,color="#555555",lw=.8);ax.axvline(0,color="#777777",lw=.7,ls="--")
    ax.set_title(mname);ax.grid(alpha=.15)
    ax.set_xlabel("Signed prior error: estimated pi - pi" if col==0 else "Relative prior error: (estimated pi - pi) / pi")
    ax.set_ylabel("AUROC change (pp)")
  handles,labs=axes[0,0].get_legend_handles_labels()
  fig.legend(handles,labs,loc="lower center",ncol=4,frameon=False,bbox_to_anchor=(.5,-.005))
  fig.suptitle(dlabel+": prior error and AUROC change (descriptive associations)",fontsize=13,y=.995)
  fig.tight_layout(rect=(0,.07,1,.97));pdf.savefig(fig,bbox_inches="tight");plt.close(fig)
manifest.append(dict(file=name,pages=2))
fig,axes=plt.subplots(1,2,figsize=(12,6.0),sharey=True)
palette=["#0072B2","#009E73","#D55E00","#CC79A7"]
for ax,metric in zip(axes,["AUROC","AP"]):
 g=real[(real.summary_level=="dataset_equal_four_conditions")&(real.metric==metric)]
 for j,((n,label),color) in enumerate(zip(method_names.items(),palette)):
  z=g[g.method_number==n].set_index("dataset").reindex(list(datasets))
  ax.scatter(z.mean_difference*100,np.arange(8)+(j-1.5)*.15,s=32,label=label,color=color)
 ax.axvline(0,color="#555555",lw=1)
 ax.set_xlabel("Estimated minus oracle "+metric+" (pp)")
 ax.set_yticks(range(8),list(datasets.values()));ax.grid(axis="x",alpha=.2)
 ax.set_title(metric)
axes[0].invert_yaxis()
handles,labs=axes[0].get_legend_handles_labels();fig.legend(handles,labs,loc="lower center",ncol=4,frameon=False)
fig.suptitle("Real data: equal weighting of four design/budget conditions within each dataset",y=.99)
fig.text(.5,.06,"Each condition averages its repeated paired splits; datasets are not treated as repeated cases.",ha="center",fontsize=8)
fig.tight_layout(rect=(0,.1,1,.94));finish(fig,"estimated_prior_real_data.pdf")
fig,axes=plt.subplots(1,2,figsize=(11.5,6),sharey=True)
for ax,(design,dlabel) in zip(axes,designs):
 g=ac[(ac.study=="main")&(ac.design==design)&(ac.status=="success")]
 values=[g.loc[g.scenario_number==sid,"prior_ratio"].dropna().to_numpy(float) for sid in sids]
 boxes=ax.boxplot(values,positions=range(8),vert=False,patch_artist=True,whis=(5,95),widths=.55,
                  medianprops={"color":"black","lw":1.2},flierprops={"markersize":2,"alpha":.35})
 for sid,box in zip(sids,boxes["boxes"]):box.set_facecolor(colors[sid]);box.set_alpha(.6)
 ax.axvspan(.8,1.2,color="#999999",alpha=.25,label="Old +/-20% range")
 ax.axvline(1,color="black",ls="--",lw=.9)
 ax.set_xscale("log");ax.set_xlim(.01,100)
 ax.set_xticks([.01,.03,.1,.3,1,3,10,30,100],["0.01","0.03","0.1","0.3","1","3","10","30","100"])
 ax.set_yticks(range(8),[labels[s] for s in sids]);ax.set_title(dlabel)
 ax.set_xlabel("Estimated pi / true pi (logarithmic axis)");ax.grid(axis="x",alpha=.15)
axes[0].invert_yaxis();axes[1].legend(loc="upper right",frameon=False)
fig.suptitle("Empirical prior error relative to the archived sensitivity range",y=.99)
fig.text(.5,.015,"Boxes: 25th-75th percentiles; whiskers: 5th-95th percentiles; points: remaining estimates.",ha="center",fontsize=9)
fig.tight_layout(rect=(0,.055,1,.95));finish(fig,"prior_error_distribution.pdf")
with (H/"figure_manifest.json").open("x") as f:json.dump(manifest,f,indent=2)
print(json.dumps(manifest,indent=2))
