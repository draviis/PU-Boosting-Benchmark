"""Exact displayed table values and LaTeX formatting."""
import re
import numpy as np
import pandas as pd
from display_specs import *
from display_data import read, SELECTORS, SELNAMES
def score(x):return "--" if pd.isna(x) else f"{x:.3f}"
def pp(x):return "--" if pd.isna(x) else f"{0. if abs(x)<.005 else x:+.2f}"
def count(x):return "--" if pd.isna(x) else str(int(x))
def interval(lo,hi):return "--" if pd.isna(lo) or pd.isna(hi) else f"[{pp(lo)}, {pp(hi)}]"
def block(m):return "Block 1: nonboosting PU" if m<=5 else ("Block 3: native PU" if m>=26 else "Block 2: boosting")
def panel(design,other=""):return DNAME[design]+(" - "+other if other else "")
def table_values(id,d):
 rows=[];note=""
 if "same_engine" in id:
  for (strategy,engine),g in d.groupby(["strategy","engine"],sort=False):
   r={"Strategy":strategy,"Engine":engine}
   for metric in ["AUROC","AP"]:
    for design,short in zip(DESIGNS,["OS","TS"]):r[f"{metric} {short}"]=pp(g[(g.metric==metric)&(g.design==design)].difference_pp.iloc[0])
   rows.append(r)
  note="Entries are signed percentage-point differences. OS: one-sample; TS: two-sample. No BA comparison is included."
  if "40setting" in id:note="Unweighted 40-setting sensitivity summary. This is not the headline weighting scheme. "+note
 elif id=="tab_s_core_family_balanced_all_methods":
  for m in range(3,28):
   g=d[d.method_number==m];r={"Panel":block(m),"Method":METHODS[m]}
   for metric in ["AUROC","AP"]:
    for design,short in zip(DESIGNS,["OS","TS"]):r[f"{metric} {short}"]=score(g[(g.metric==metric)&(g.design==design)].mean_value.iloc[0])
   rows.append(r)
  note="First average settings within each family, then weight all ten families equally. Every scenario/method mean has 100 repetitions. OS: one-sample; TS: two-sample."
 elif id=="tab_core_bayes_reference" or id=="tab_s_bayes_regret":
  for (design,scenario),g in d.groupby(["design","scenario_index"],sort=True):
   r={"Panel":panel(design),"Scenario":SCENARIOS.get(scenario,f"S{scenario:03}"),"Bayes AUROC":score(g.Bayes_AUROC.iloc[0])}
   if id=="tab_s_bayes_regret":r["Scenario"]=f"S{scenario:03}"
   for m in REPS:
    a=g[g.method_number==m].iloc[0]
    r[METHODS[m]]=f"{score(a.mean_value)} ({score(a.regret)})" if id=="tab_core_bayes_reference" else score(a.regret)
   rows.append(r)
  note=("Cells show AUROC (Bayes minus AUROC regret)." if id=="tab_core_bayes_reference" else "Method cells show Bayes AUROC minus observed AUROC; lower regret is better.")+" Bayes references are the existing Task-2 analytic or exact-likelihood-ratio Monte Carlo values; Monte Carlo reference uncertainty is retained in Task 2. All means use 100 repetitions. Sxxx identifies the corresponding original setting (TSxxx in two-sample). No new Bayes approximation is introduced."
 elif id=="tab_nonlinear_headline":
  for _,g in d.iterrows():
   r={"Panel":panel(g.design),"DGP":NLNAMES[int(g.scenario_id[-2:])-1],"Bayes AUROC":score(g.Bayes_AUROC)}
   for m,key in [(3,"corrected_logistic"),(10,"naive_CatBoost"),(25,"CS_CatBoost"),(20,"RN_CatBoost"),(27,"PU_gradient_boosting")]:
    r[METHODS[m]]=score(g[key+"_AUROC"])+("*" if g[key+"_n"]==99 else "")
   rows.append(r)
  note="AUROC means from final train+validation refits. * NL02 two-sample RN CatBoost uses n=99; one refit produced zero reliable negatives and remains unavailable. All other entries use n=100. Bayes values are the frozen million-observation calibration estimates. NL01 and NL05 have identical true Bayes information; their displayed estimates differ only by calibration Monte Carlo error."
 elif "threshold" in id:
  for (study,design,case,m),g in d.groupby(["study","design","case","method_number"],sort=True):
   a=g.iloc[0]; label=a.scenario_id if study=="simulation" else f"{a.dataset}, c={a.c:.2f}"
   r={"Panel":panel(design,f"{study}: {label}"),"Method":METHODS[m]}
   for rule,name in [("B_same_model_quantile","A_same_model"),("C_PU_BA","PU-estimated BA"),("D_same_model_oracle","Oracle BA")]:r[name]=score(g[g.rule==rule].BalancedAccuracy.iloc[0])
   r["Verified"]="yes" if g.eligible_primary.all() else "no"
   rows.append(r)
  note="Balanced accuracy from one saved split/repetition per setting. -- indicates an unverified replay, not a zero. Six real method/case replays are unverified; 1,194/1,200 method/case results are verified for each rule. A_same_model is the common prevalence quantile; oracle BA uses true validation labels only. PU-estimated BA remains a limited pilot."
 elif "rn_mechanism" in id:
  for _,g in d.iterrows():
   name=f"S{int(g.scenario_number):03}"+(" Interaction" if g.scenario_number==39 else "")
   rows.append({"Panel":panel(g.design),"Scenario":name,"RN purity":score(g.RN_precision_mean),"TN retained / available":score(g.RN_negative_recall_mean),"U retention":score(g.RN_selection_fraction_mean),"AUROC change (pp)":pp(100*g.Delta_AUROC_mean),"AP change (pp)":pp(100*g.Delta_AP_mean),"n":count(g.n)})
  note="TN retained / available is both true-negative recall and the fraction of available true negatives retained. U retention includes hidden positives. Means over 100 repetitions. Scenario identifiers correspond to Task 2. RN masks were reconstructed from exact archived data/code because selected indices were not originally stored."
 elif "rn_selector" in id:
  for (design,sn),g in d.groupby(["design","scenario_number"],sort=True):
   for condition,label in zip(SELECTORS,SELNAMES):
    x=g[g.condition==condition];a=x[x.metric=="AUROC"].iloc[0];b=x[x.metric=="AP"].iloc[0]
    rows.append({"Panel":panel(design,SCENARIOS[int(sn)]),"Selector/control":label,"AUROC":score(a["mean"]),"AP":score(b["mean"]),"n AUROC":count(a.n),"n AP":count(b.n)})
  note="Fixed CatBoost final-refit parameters. Multiple random/oracle draws are first averaged within repetition. Absolute means are conditional on the original Task-4 protocol; these controls are diagnostics, not additional benchmark configurations."
 elif "supplied_prior" in id:
  for (design,scenario,metric,m),g in d.groupby(["design","scenario_id","metric","method_number"],sort=True):
   r={"Panel":panel(design,scenario+" / "+metric),"Method":METHODS[m]}
   for ratio in [.8,.9,1.,1.1,1.2]:
    a=g[g.prior_ratio==ratio].iloc[0];r[f"{ratio:.2f}"]=pp(a["mean"])
   r["n"]=count(g.n.min());rows.append(r)
  note="Columns are supplied/true prior ratios; entries are paired mean changes from the correct supplied prior in percentage points. All cells use 100 pairs. The ratio-one difference is the observed baseline compared with itself. Full-precision MCSE and 95% Monte Carlo intervals are in the accompanying numeric audit CSV; the figure gives an equally weighted eight-setting summary."
 elif "estimated_prior" in id:
  for _,g in d.sort_values(["study","design","budget","metric","scenario_id","method_number"]).iterrows():
   setting=g.scenario_id if g.study=="main" else g.dataset
   rows.append({"Panel":panel(g.design,f"{g.study}, {g.budget}, {g.metric}"),"Setting":setting.replace("_"," "),"Method":METHODS[g.method_number],"Prior ratio":f"{g.ratio_mean:.2f}","Change (pp)":pp(g["mean"]),"95% interval":interval(g.lo,g.hi),"n / requested":f"{int(g.n)}/{int(g.n_expected)}"})
  note="KM2 only. Prior ratio is mean estimated/true prevalence; n refers to matched performance pairs. Simulation intervals quantify within-setting Monte Carlo error; real-data intervals are conditional repeated-split uncertainty and not a population sampling guarantee. All available cases are retained; no failed fits are imputed."
 elif id=="tab_s_nonlinear_all_methods":
  for _,g in d.sort_values(["design","scenario_id","method_number"]).iterrows():
   rows.append({"Panel":panel(g.design,g.scenario_id),"Method":METHODS[g.method_number],"AUROC":score(g.AUROC_mean),"n AUROC":count(g.AUROC_n),"AP":score(g.AP_mean),"n AP":count(g.AP_n)})
  for _,g in read("failures").iterrows():
   rows.append({"Panel":"Unavailable final refits (zero-based repetition index)","Method":f"{g.scenario} {DNAME[g.design]}, r{g.repetition:03}: {g.method}","AUROC":"--","n AUROC":"0","AP":"--","n AP":"0"})
  note="Means use available final refits and are conditional on refit success where n=99 (out of 100 requested). The six unavailable refits are retained explicitly above. Reason in every case: selected RN parameters yielded zero reliable negatives during final train+validation refit. No imputation, fallback model, or reselection is applied. Selected-model hard-classification results are not substituted for ranking results."
 elif "unique_score_ranks" in id:
  for (design,scenario,m),g in d.groupby(["design","scenario_id","retained_method_number"],sort=True):
   a=g[g.metric=="AUROC"].iloc[0];b=g[g.metric=="AP"].iloc[0]
   name=a.ranking_entry
   if 6<=m<=10:name="Naive/Corrected "+ENGINES[m-6]
   if m in [3,4,5,26,27]:name=METHODS[m]
   rows.append({"Panel":panel(design,scenario),"Ranking entry":name,"AUROC":score(a.mean_value),"AUC rank":f"{a['rank']:g}","AP":score(b.mean_value),"AP rank":f"{b['rank']:g}","n":count(a.n_repetitions)})
  note="Twenty unique ranking entries per nonlinear DGP/design. Verified naive/corrected same-engine aliases retain one score without averaging. Ranks use the saved Task-7B mean-score rule, with average ties after 12-decimal rounding. n=99 entries are conditional on successful refits. Original 25-configuration rank files remain unchanged. The Task-2 two-sample AdaBoost exception remains separate in its 21-entry ranking universe; it does not apply to these nonlinear aliases."
 elif "neural_simulation" in id:
  for (design,sn,contrast),g in d.groupby(["design","scenario_number","contrast"],sort=True):
   a=g[g.metric=="AUROC"].iloc[0];b=g[g.metric=="AP"].iloc[0]
   rows.append({"Panel":panel(design,SCENARIOS[int(sn)]),"Matched contrast":contrast,"AUROC (pp)":pp(a["mean"]),"AUROC interval":interval(a.lo,a.hi),"AP (pp)":pp(b["mean"]),"AP interval":interval(b.lo,b.hi),"n":"10"})
  note="Exploratory: 10 paired repetitions per setting. Neural minus linear with matched risk and selection criterion. Intervals are the existing paired t intervals (df=9); they are retained as saved. Proxy-AUROC denotes the PU validation tuning criterion, not test-data selection."
 else:raise KeyError(id)
 return pd.DataFrame(rows).fillna("--"),note

def esc(s):
 s=str(s)
 for a,b in [("\\",r"\textbackslash{}"),("&",r"\&"),("%",r"\%"),("_",r"\_"),("#",r"\#")]:s=s.replace(a,b)
 return s
HEADERS={"Logistic probability correction":r"\shortstack{Logistic\\correction}","PU gradient boosting":r"\shortstack{PU gradient\\boosting}",
"Bayes AUROC":r"\shortstack{Bayes\\AUROC}","nnPU logistic":r"\shortstack{nnPU\\logistic}",
"Naive CatBoost":r"\shortstack{Naive\\CatBoost}","CS CatBoost":r"\shortstack{CS\\CatBoost}","RN CatBoost":r"\shortstack{RN\\CatBoost}",
"TN retained / available":r"\shortstack{TN retained\\ / available}","AUROC change (pp)":r"\shortstack{AUROC\\change (pp)}","AP change (pp)":r"\shortstack{AP\\change (pp)}",
"AUROC interval":r"\shortstack{AUROC 95\%\\interval}","AP interval":r"\shortstack{AP 95\%\\interval}",
"n / requested":r"\shortstack{n /\\requested}","Prior ratio":r"\shortstack{Prior\\ratio}","95% interval":r"95\% interval",
"AUROC (pp)":r"\shortstack{AUROC\\(pp)}","AP (pp)":r"\shortstack{AP\\(pp)}","PU-estimated BA":r"\shortstack{PU-estimated\\BA}",
"A_same_model":r"\shortstack{A\_same\_model\\quantile}","Oracle BA":r"\shortstack{Oracle BA\\diagnostic}"}
def widths(id,cols):
 n=len(cols);total=6.95-(n-1)*.0833
 # Column widths include no scaling: font remains small (10pt in 11pt audit document).
 if id=="tab_core_bayes_reference":return [1.02]+[(total-1.02)/7]*7
 if id=="tab_nonlinear_headline":return [1.48]+[(total-1.48)/6]*6
 if id=="tab_s_bayes_regret":return [.58]+[(total-.58)/7]*7
 if "estimated_prior" in id:return [1.12,1.35,.62,.70,1.15,total-4.94]
 if "rn_mechanism" in id:return [.95,.65,.85,.72,.80,.80,total-4.77]
 if "neural_simulation" in id:return [1.32,.73,1.25,.73,1.25,total-5.28]
 if "same_engine" in id:return [1.4,.85]+[(total-2.25)/4]*4
 if "supplied_prior" in id:return [1.65]+[(total-1.65)/(n-1)]*(n-1)
 first=2.0 if n<=6 else 1.85
 return [first]+[(total-first)/(n-1)]*(n-1)
def latex(id,t,note,caption):
 cols=[c for c in t.columns if c!="Panel"];ws=widths(id,cols)
 align="@{}"+''.join((r">{\raggedright\arraybackslash}p" if i==0 or (cols[0]=="Strategy" and i==1) else r">{\centering\arraybackslash}p")+f"{{{w:.3f}in}}" for i,w in enumerate(ws))+"@{}"
 lines=[r"% Requires booktabs, array, longtable. Compile from repository root.",
 r"\begingroup",r"\small",r"\setlength{\tabcolsep}{3pt}",r"\renewcommand{\arraystretch}{1.18}"]
 groups=list(t.groupby("Panel",sort=False)) if "Panel" in t else [("",t)]
 for i,(name,g) in enumerate(groups):
  head=" & ".join(HEADERS.get(c,esc(c)) for c in cols)+r" \\"
  lines += [r"\begin{longtable}{"+align+"}"]
  if i==0:lines += [r"\caption{"+esc(caption)+r"}\label{tab:"+id.replace("_","-")+r"}\\"]
  section=(("Panel "+chr(65+i)+": ") if id in ["tab_core_bayes_reference","tab_nonlinear_headline"] else "")+name
  sectionrow=([r"\multicolumn{"+str(len(cols))+r"}{@{}l}{\textbf{"+esc(section)+r"}}\\",r"\addlinespace[3pt]"] if name else [])
  lines += [r"\toprule"]+sectionrow+[head,r"\midrule",r"\endfirsthead",r"\toprule"]+sectionrow+[head,r"\midrule",r"\endhead",r"\bottomrule",r"\endfoot"]
  for _,r in g.iterrows():
   cells=[]
   for c in cols:
    v=str(r[c])
    if re.match(r"^\d\.\d{3} \(-?\d\.\d{3}\)$",v):
     a,b=v.split(" ",1);v=r"\shortstack{"+esc(a)+r"\\"+esc(b)+"}"
    else:v=esc(v)
    cells.append(v)
   lines.append(" & ".join(cells)+r" \\")
   if id in ["tab_core_bayes_reference","tab_nonlinear_headline"]:lines.append(r"\addlinespace[2pt]")
  lines += [r"\end{longtable}"]
 lines += [r"\noindent\textit{Note.} "+esc(note),r"\par\endgroup"]
 return "\n".join(lines)+"\n"
