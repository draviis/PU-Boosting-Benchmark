"""Read-only post-processing of immutable saved Task outputs."""
from functools import lru_cache
import numpy as np
import pandas as pd
from display_specs import *

@lru_cache(None)
def read(key):
 p=ROOT/source_path(key)
 d=pd.read_csv(p,low_memory=False)
 print(f"SOURCE {source_path(key).as_posix()}: {len(d)} rows",flush=True)
 return d

def catalog(d,col="method_number"):
 assert set(d[col].unique())==set(range(3,28)),(col,sorted(d[col].unique()))
def paired(g,value):
 v=g[value]
 n=int(v.notna().sum())
 mean=v.mean(); se=v.std(ddof=1)/np.sqrt(n) if n>1 else np.nan
 return pd.Series(dict(mean=mean,n=n,n_expected=len(v),MCSE=se,lo=mean-1.96*se,hi=mean+1.96*se))
def aggregate(d,by,value):
 return d.groupby(by,dropna=False,sort=True).apply(lambda g:paired(g,value),include_groups=False).reset_index()
def core():
 d=read("core").query("policy == 'ranking_archive' and metric in ['AUROC','AP'] and method_number >= 3").copy()
 catalog(d); assert len(d)==4000 and d.n_repetitions.eq(100).all()
 assert set(d.scenario_index)==set(range(1,41))
 return d
def family():
 d=read("family").query("policy == 'ranking_archive' and metric in ['AUROC','AP'] and method_number >= 3").copy()
 catalog(d)
 f=d[d.summary_type.eq("within_family")]
 assert len(f)==1000 and f.complete.all() and set(f.family_id)==set(range(1,11))
 return d
def core_family():
 d=family().query("summary_type == 'within_family'").copy()
 # Independently check equal-setting aggregation from scenario-level source.
 check=core().groupby(["design","method_number","metric","family_id"]).mean_value.mean()
 got=d.set_index(["design","method_number","metric","family_id"]).mean_value
 assert np.allclose(check.sort_index(),got.sort_index(),atol=1e-14,rtol=0)
 d["family_label"]=d.family_id.map(lambda i:FAMILIES[int(i)-1])
 return d
def same_engine(weight):
 d=family().query("summary_type == @weight")
 rows=[]
 for strategy,start in [("Probability correction",11),("RN",16),("CS",21)]:
  for j,engine in enumerate(ENGINES):
   for design in DESIGNS:
    for metric in ["AUROC","AP"]:
     g=d[(d.design==design)&(d.metric==metric)].set_index("method_number")
     rows.append(dict(strategy=strategy,engine=engine,method_number=start+j,naive_number=6+j,design=design,metric=metric,difference_pp=100*(g.loc[start+j,"mean_value"]-g.loc[6+j,"mean_value"]),weight=weight))
 return pd.DataFrame(rows)
def bayes():
 d=core().query("metric == 'AUROC' and method_number in @REPS").copy()
 b=read("bayes")
 d=d.merge(b[["design","scenario_id","Bayes_AUROC","AUROC_source"]],on=["design","scenario_id"],validate="many_to_one")
 d["regret"]=d.Bayes_AUROC-d.mean_value
 assert len(d)==480 and d.Bayes_AUROC.notna().all()
 assert np.allclose(d.regret,d.regret_to_Bayes,atol=1e-13)
 return d
def threshold():
 d=read("threshold")
 d=d[(d.scope=="fixed_pilot")&d.rule.isin(["B_same_model_quantile","C_PU_BA","D_same_model_oracle"])].copy()
 assert len(d)==3600 and d.case.nunique()==48
 assert not d.test_labels_used_for_selection.any()
 catalog(d)
 assert set(d.model_stage)=={"selected_train_only"},d.model_stage.unique()
 # Verify both endpoint rules against Task 1B rather than treating rule names as interchangeable.
 for key,rule in [("thresholdA","B_same_model_quantile"),("thresholdB","D_same_model_oracle")]:
  b=read(key);b=b[(b.scope=="fixed_pilot")&b.method_number.ge(3)]
  if key=="thresholdB":b=b[b.policy=="B_same_model_oracle"]
  a=d[d.rule==rule].merge(b,on=["case","method_number"],suffixes=("_c","_b"),validate="one_to_one")
  assert len(a)==1200
  v=a[a.eligible_primary]
  assert len(v)==1194
  assert np.allclose(v.BalancedAccuracy_c,v.BalancedAccuracy_b,atol=1e-9,equal_nan=True)
  assert not a.test_labels_used_for_threshold_selection.any()
 d["unverified_BA_audit_only"]=d.BalancedAccuracy.where(~d.eligible_primary)
 d.loc[~d.eligible_primary,"BalancedAccuracy"]=np.nan
 d["rule_label"]=d.rule.map({"B_same_model_quantile":"A_same_model quantile","C_PU_BA":"PU-estimated BA pilot","D_same_model_oracle":"Oracle BA diagnostic"})
 return d
def rn():
 d=read("rn")
 d=d[(d.study=="main")&(d.engine=="CatBoost")&(d.stage=="final_train_plus_validation")].copy()
 assert len(d)==8000 and d.groupby(["design","scenario_id"]).size().eq(100).all()
 assert d.RN_negative_recall.notna().all()
 assert np.allclose(d.RN_negative_recall,d.RN_true_negatives/d.hidden_negatives_U,atol=1e-9)
 assert np.allclose(d.discarded_negative_fraction,1-d.RN_negative_recall,atol=1e-9)
 vals=["RN_precision","RN_negative_recall","RN_selection_fraction","Delta_AUROC","Delta_AP"]
 rows=[]
 for keys,g in d.groupby(["design","scenario_id","scenario_number","scenario_family"]):
  r=dict(zip(["design","scenario_id","scenario_number","scenario_family"],keys));r["n"]=len(g)
  for v in vals:
   a=paired(g,v)
   for k in ["mean","MCSE","lo","hi"]:r[v+"_"+k]=a[k]
  rows.append(r)
 out=pd.DataFrame(rows)
 p=read("rnpaired").query("study == 'main' and engine == 'CatBoost'")
 for metric in ["AUROC","AP"]:
  a=out.merge(p[p.outcome=="Delta_"+metric],on=["design","scenario_id"],validate="one_to_one")
  assert np.allclose(a["Delta_"+metric+"_mean"],a.mean_paired_difference,atol=2e-10)
 read("rnfamily")
 return out
SELECTORS=["RN_logistic","RN_nonlinear","RN_nonlinear_size_matched","RN_random_size_matched","Oracle_true_negative_size_matched"]
SELNAMES=["Logistic spy","Nonlinear spy full","Nonlinear size-matched","Random size-matched","Oracle TN size-matched"]
def selector():
 d=read("selector")
 d=d[(d.scope=="core")&(d.study=="main")&(d.engine=="CatBoost")&(d.selector=="CatBoost")&(d.stage=="final_train_plus_validation")&(d.record_type=="case_condition_mean")&d.condition.isin(SELECTORS)].copy()
 assert len(d)==5000
 assert set(d.scenario_number)=={1,26,38,39,40}
 assert not d.duplicated(["design","scenario_id","rep","condition"]).any()
 assert d.groupby(["design","scenario_id","condition"]).size().eq(100).all()
 m=d.melt(id_vars=["design","scenario_id","scenario_number","rep","condition"],value_vars=["AUROC","AP"],var_name="metric",value_name="value")
 out=aggregate(m,["design","scenario_id","scenario_number","condition","metric"],"value")
 out["selector_label"]=out.condition.map(dict(zip(SELECTORS,SELNAMES)))
 assert out.n.eq(100).all()
 read("selector_mc")
 return out
def supplied():
 rows=[]
 for design,key in zip(DESIGNS,["supplied_os","supplied_ts"]):
  d=read(key).copy();d["method_number"]=d.method_id.str[-3:].astype(int)
  d=d[d.method_number.isin([4,5,26,27])&d.metric.isin(["AUROC","AUPRC"])].copy()
  d["metric"]=d.metric.replace({"AUPRC":"AP"});d["design"]=design;d["scenario_id"]=d.source_scenario_id
  assert len(d)==25600 and set(d.prior_ratio)=={.8,.9,1.1,1.2}
  assert d.correct_prior_value.notna().all() and d.misspecified_prior_value.notna().all()
  # Ratio 1 is a self-comparison of observed baseline scores, not a missing run filled in.
  unique=["design","scenario_id","method_number","metric","rep"]
  assert d.groupby(unique).correct_prior_value.nunique().eq(1).all()
  z=d.sort_values("prior_ratio").drop_duplicates(unique).copy()
  z["prior_ratio"]=1.;z["change_misspecified_minus_correct"]=z.correct_prior_value-z.correct_prior_value
  d=pd.concat([d,z],ignore_index=True)
  d["difference_pp"]=100*d.change_misspecified_minus_correct
  a=aggregate(d,["design","scenario_id","method_number","metric","prior_ratio"],"difference_pp")
  assert a.n.eq(100).all() and len(a)==320
  rows.append(a)
 return pd.concat(rows,ignore_index=True)
def estimated():
 cover=read("priorcover").query("record_type == 'estimated_prior_coverage'").copy()
 assert len(cover)==48 and cover.n_requested.eq(cover.n_valid).all()
 s=read("est_sim").query("method_number in [4,5,26,27] and metric in ['AUROC','AP']").copy()
 s["study"]="main";s["dataset"]="";s["budget"]=""
 r=read("est_real").query("summary_level == 'dataset_design_budget' and method_number in [4,5,26,27] and metric in ['AUROC','AP']").copy()
 r["study"]="real";r["scenario_id"]=r.dataset
 perf=pd.concat([s,r],ignore_index=True)
 for c in ["dataset","budget"]:
  cover[c]=cover[c].fillna("");perf[c]=perf[c].fillna("")
 keys=["study","design","scenario_id","dataset","budget"]
 out=perf.merge(cover[keys+["ratio_mean","ratio_q05","ratio_q95","n_requested","n_valid"]],on=keys,validate="many_to_one",suffixes=("","_prior"))
 assert len(out)==384 and out.ratio_mean.notna().all()
 for old,new in [("mean_difference","mean"),("MCSE","MCSE"),("MC_lower","lo"),("MC_upper","hi")]:out[new]=100*out[old]
 out["n"]=out.n_valid;out["n_expected"]=out.n_requested
 return out
def nonlinear():
 d=read("nl").copy();d["method_number"]=d.method_id.str[-3:].astype(int);d=d[d.method_number.ge(3)]
 catalog(d);assert len(d)==300 and d.n_repetitions.eq(100).all()
 raw=read("nlraw");f=read("failures")
 failed=raw[raw.AUROC.isna()|raw.AP.isna()]
 assert len(failed)==len(f)==6
 assert failed.method_id.str[-3:].astype(int).between(16,20).all()
 assert failed[["AUROC","AP"]].isna().all().all()
 assert set(zip(failed.scenario_id,failed.design,failed.rep,failed.method_id))==set(zip(f.scenario,f.design,f.repetition,f.method_id))
 assert not f.imputed.any()
 assert d.AUROC_missing.sum()==d.AP_missing.sum()==6
 for metric in ["AUROC","AP"]:
  check=raw[raw.method_id.str[-3:].astype(int).ge(3)].groupby(["scenario_id","design","method_id"])[metric].agg(["mean","count"])
  a=d.set_index(["scenario_id","design","method_id"]).sort_index()
  assert np.allclose(a[metric+"_mean"],check.sort_index()["mean"],atol=2e-10)
  assert np.array_equal(a[metric+"_n"],check.sort_index()["count"])
 return d
def neural():
 d=read("neural").copy()
 d=d[d.reference.eq(d.method.str.replace("mlp_","linear_",regex=False))].copy()
 assert len(d)==96 and d.n.eq(10).all()
 assert set(d.method)=={"mlp_nnpu_auc","mlp_nnpu_risk","mlp_upu_auc","mlp_upu_risk"}
 d["design"]=d.scheme;d["scenario_id"]=d.scenario;d["scenario_number"]=d.scenario.str[-3:].astype(int)
 d["metric"]=d.metric.replace({"AUPRC":"AP"})
 d["contrast"]=d.method.str.replace("mlp_","",regex=False).map({"nnpu_auc":"nnPU / proxy-AUROC","nnpu_risk":"nnPU / risk","upu_auc":"uPU / proxy-AUROC","upu_risk":"uPU / risk"})
 for old,new in [("mean","mean"),("se","MCSE"),("ci95_low","lo"),("ci95_high","hi")]:d[new]=100*d[old]
 return d
def data_for(id):
 if id in ["fig_core_fixed_methods","fig_s_core_full_fixed_methods"]:
  d=core_family()
  if id=="fig_core_fixed_methods":
   ref=d[d.method_number==3][["design","metric","family_id","mean_value"]].rename(columns={"mean_value":"reference"})
   d=d[d.method_number.isin([5,10,25,20,27])].merge(ref,on=["design","metric","family_id"],validate="many_to_one")
   d["difference_pp"]=100*(d.mean_value-d.reference)
  return d
 if "same_engine" in id:return same_engine("original_40_setting_mean" if "40setting" in id else "equal_10_family_mean")
 if id=="tab_s_core_family_balanced_all_methods":return family().query("summary_type == 'equal_10_family_mean'").copy()
 if "bayes" in id:
  d=bayes()
  return d[d.scenario_index.isin([1,26,38,39,40])].copy() if id=="tab_core_bayes_reference" else d
 if id=="fig_nonlinear_decomposition":
  d=read("decomp").query("engine == 'CatBoost'").copy()
  assert len(d)==72 and d.n_pairs.eq(99).sum()==2 and set(d.scenario_id)=={f"NL{i:02}" for i in range(1,7)}
  return d
 if id=="tab_nonlinear_headline":
  d=read("headline");assert len(d)==12 and d.RN_CatBoost_n.eq(99).sum()==1
  assert len(read("failures"))==6
  read("nlcal");return d.copy()
 if "threshold" in id:return threshold()
 if "rn_mechanism" in id:return rn()
 if "rn_selector" in id:return selector()
 if "supplied_prior" in id:return supplied()
 if "estimated_prior" in id:return estimated()
 if id in ["fig_s_nonlinear_full","tab_s_nonlinear_all_methods"]:return nonlinear()
 if "unique_score_ranks" in id:
  d=read("nlrank");assert len(d)==480 and d.rank_catalog_n.eq(20).all() and not d.alias_values_averaged.any()
  for k in ["alias_audit","alias_exceptions","alias_original","core_ranks","nl_original_rank"]:read(k)
  a=read("alias_original");assert a[a.design=="one_sample"].rank_catalog_n.eq(20).all() and a[a.design=="two_sample"].rank_catalog_n.eq(21).all()
  return d.copy()
 if "neural_simulation" in id:return neural()
 raise KeyError(id)
