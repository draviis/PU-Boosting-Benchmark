"""Aggregate only completed, independently validated Task 6 records."""
from pathlib import Path
import json,math,hashlib
import numpy as np,pandas as pd
H=Path(__file__).resolve().parent;R=H.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def export(name,frame):
 dest=H/(name+".export.ndjson")
 if dest.exists():raise FileExistsError(dest)
 frame=frame.copy()
 for col in frame:
  if frame[col].dtype=="object":
   frame[col]=frame[col].map(lambda x:json.dumps(x,sort_keys=True,separators=(",",":")) if isinstance(x,(list,dict)) else x)
 frame=frame.replace([np.inf,-np.inf],np.nan)
 data=frame.astype(object).where(pd.notna(frame),None)
 with dest.open("x",encoding="utf-8") as f:
  f.write(json.dumps({"columns":list(data.columns)})+"\n")
  for start in range(0,len(data),1000):
   f.write(json.dumps(data.iloc[start:start+1000].values.tolist(),allow_nan=False)+"\n")
 frame.to_pickle(H/(name+".staging.pkl"))
 return dict(file=name+".csv",rows=len(frame),columns=len(frame.columns))
def mc(values):
 a=np.asarray(values,float);a=a[np.isfinite(a)];n=len(a)
 mean=float(a.mean()) if n else None
 sd=float(a.std(ddof=1)) if n>1 else None
 se=sd/math.sqrt(n) if n>1 else None
 return dict(n_valid=n,mean_difference=mean,MCSE=se,MC_lower=mean-1.96*se if se is not None else None,MC_upper=mean+1.96*se if se is not None else None)
cases=json.loads((H/"cases.json").read_text())
canonical={int(mid[-3:]):row["method"] for mid,row in cases[0]["oracle"].items()}
canonical[3]="Block1 | Probability-corrected logistic"
accuracy=[];paired=[];arms=[];estfail=[];fitfail=[]
for c in cases:
 h=H/"results"/c["key"]
 complete=json.loads((h/"complete.json").read_text())
 valid=json.loads((h/"independent_validation.json").read_text())
 assert valid["status"]=="passed" and valid["complete_sha256"]==sha(h/"complete.json")
 e=json.loads((h/"estimate.json").read_text())
 meta={k:c.get(k) for k in ["key","study","design","scenario_id","scenario_number","scenario_family","dataset","budget","rep","seed","fingerprint"]}
 a=meta|{k:e.get(k) for k in ["estimator","parameters","status","failure_reason","kappa_raw","kappa_usable","lambda_hat","pi_hat_raw","pi_U_hat_raw","c_hat_raw","pi_hat","pi_U_hat","c_hat","true_pi","true_pi_U","true_c","empirical_train_pi","empirical_train_pi_U","empirical_train_c","nP","nU","kernel_nP","kernel_nU","kernel_width","RKHS_separation","search_lower_saturation","search_upper_saturation","prior_clipping_applied","training_X_sha256","training_S_sha256","P_indices_sha256","U_indices_sha256","source_code_sha256","estimator_source_sha256","arrays_sha256","seconds"]}
 a.update(estimator_seed=c["estimator_seed"],solver_calls=len(e["solver_calls"]),roundoff_safeguards=len(e["sqrt_safeguards"]),estimate_source="results/"+c["key"]+"/estimate.json")
 for quantity,truth in [("pi","true_pi"),("pi_U","true_pi_U"),("c","true_c")]:
  v=e.get(quantity+"_hat");t=e.get(truth)
  a[quantity+"_error"]=v-t if v is not None and t is not None else None
  a[quantity+"_absolute_error"]=abs(v-t) if v is not None and t is not None else None
  a[quantity+"_relative_absolute_error"]=abs(v-t)/t if v is not None and t is not None and t>0 else None
 a["prior_ratio"]=e["pi_hat"]/e["true_pi"] if e.get("pi_hat") is not None else None
 accuracy.append(a)
 if e["status"]!="success":estfail.append(a)
 for row in complete["rows"]:
  mid=row["method_id"];n=int(mid[-3:])
  p=meta|dict(method_id=mid,method_number=n,method=canonical[n],archived_method_label=row["method"],status=row["status"],prior_status=e["status"],
              ranking_prior_dependent=n in [4,5,26,27],raw_score_rank_invariance_exact=row.get("raw_score_rank_invariance_exact"),
              params_oracle=c["oracle"][mid]["params"],params_estimated=row.get("params"),validation_result=row.get("validation_result"),
              pi_oracle=c["pi"],pi_estimated=e.get("pi_hat"),c_oracle=c["c"],c_estimated=e.get("c_hat"),
              pi_U_estimated=e.get("pi_U_hat"),lambda_hat=e.get("lambda_hat"),
              absolute_prior_error=a["pi_error"],relative_prior_error=a["prior_ratio"]-1 if a["prior_ratio"] is not None else None,
              c_error=a["c_error"],result_source="results/"+c["key"]+"/complete.json",
              ranking_stage="final_train_plus_validation",BA_stage="selected_train_only",
              prediction_hashes=row.get("score_sha256"),failure_reason=row.get("error"))
  for metric in ["AUROC","AP","Brier","LogLoss"]:
   p["oracle_"+metric]=row.get("oracle_"+metric)
   if p["oracle_"+metric] is None and metric in ["Brier","LogLoss"] and n in [4,5]:p["oracle_"+metric]=c["oracle"][mid]["metrics"][metric]
   p["estimated_"+metric]=row.get("estimated_"+metric)
   p["Delta_"+metric]=row.get("Delta_"+metric)
  for metric in ["Brier","LogLoss"]:p["oracle_archived_"+metric]=c["oracle"][mid]["metrics"][metric]
  for arm in ["oracle","estimated"]:
   p[arm+"_BA"]=row.get("BA_"+arm+"_same_model",{}).get("BA")
   p[arm+"_BA_threshold"]=row.get("BA_"+arm+"_same_model",{}).get("threshold")
  p["oracle_probability_convention"]="archived_endpoint_clipped_1e-7" if n in [4,5] else "physical_0_1_logloss_eps1e-7" if n in [3,11,12,13,14,15] else "not_defined"
  p["oracle_Brier_endpoint_error_bound"]=2e-7 if n in [4,5] else 0. if n in [3,11,12,13,14,15] else None
  p["Delta_BA"]=p["estimated_BA"]-p["oracle_BA"] if p["estimated_BA"] is not None and p["oracle_BA"] is not None else None
  p["BA_oracle_coverage"]="unavailable" if p["oracle_BA"] is None else "fixed_pilot" if c["rep"]==1 else "targeted_Task1B_audit"
  if n in [4,5,26,27] and row["status"]=="success":
   f=json.loads((h/(mid+".json")).read_text())
   p.update(prediction_hashes=f["prediction_hashes"],candidate_count=len(f["candidates"]),fit_seconds=f["seconds"],
            changed_selected_parameters=c["oracle"][mid]["params"]!=row["params"])
  elif row["status"]=="fit_failed":fitfail.append(p)
  paired.append(p)
  for arm in ["oracle","estimated"]:
   z={k:p[k] for k in meta}|dict(method_id=mid,method_number=n,method=canonical[n],archived_method_label=row["method"],information_condition=arm,
       prior_status=e["status"],status="archived" if arm=="oracle" else row["status"],
       pi_supplied=c["pi"] if arm=="oracle" else e.get("pi_hat"),
       c_supplied=c["c"] if arm=="oracle" else e.get("c_hat"),
       hyperparameters=p["params_"+arm],validation_result=c["oracle"][mid]["val_metric"] if arm=="oracle" else p["validation_result"],
       ranking_stage=p["ranking_stage"],BA_stage=p["BA_stage"],BA_oracle_coverage=p["BA_oracle_coverage"],result_source=p["result_source"],
       probability_convention=p["oracle_probability_convention"] if arm=="oracle" else "physical_0_1_logloss_eps1e-7" if n in [3,4,5,11,12,13,14,15] else "not_defined",
       oracle_Brier_endpoint_error_bound=p["oracle_Brier_endpoint_error_bound"],
       probability_metric_source="same_archived_q_recorrected" if n in [3,11,12,13,14,15] else "original_defined_probability" if n in [4,5] else "archived_unchanged_if_defined")
   for metric in ["AUROC","AP","Brier","LogLoss","BA"]:z[metric]=p[arm+"_"+metric]
   z["BA_threshold"]=p[arm+"_BA_threshold"]
   if arm=="oracle":z.update(archived_Brier=p["oracle_archived_Brier"],archived_LogLoss=p["oracle_archived_LogLoss"])
   arms.append(z)
ac=pd.DataFrame(accuracy);df=pd.DataFrame(paired);armdf=pd.DataFrame(arms)
assert len(ac)==2240 and len(df)==56000 and len(armdf)==112000
groups=["study","design","scenario_id","dataset","budget"]
summ=[]
for key,g in ac.groupby(groups,dropna=False):
 context=dict(zip(groups,key))
 quantities=["pi","pi_U","c"] if context["design"]=="one_sample" else ["pi"]
 for q in quantities:
  z=g[g.status=="success"];errors=z[q+"_error"].dropna().to_numpy(float);ae=np.abs(errors);n=len(errors)
  rel=z[q+"_relative_absolute_error"].dropna().to_numpy(float)
  summ.append(context|dict(quantity=q,n_requested=len(g),n_valid=n,n_failed=len(g)-n,failure_rate=1-n/len(g),
      bias=float(errors.mean()) if n else None,absolute_bias=float(abs(errors.mean())) if n else None,
      RMSE=float(np.sqrt(np.mean(errors**2))) if n else None,
      mean_absolute_error=float(ae.mean()) if n else None,median_absolute_error=float(np.median(ae)) if n else None,
      median_relative_absolute_error=float(np.median(rel)) if len(rel) else None,
      q90_absolute_error=float(np.quantile(ae,.9)) if n else None,
      lower_saturation_count=int(g.search_lower_saturation.fillna(False).sum()),
      upper_saturation_count=int(g.search_upper_saturation.fillna(False).sum()),
      truth=float(g["true_"+q].iloc[0]) if g["true_"+q].nunique()==1 else None))
accsummary=pd.DataFrame(summ)
sim=df[df.study=="main"].copy();real=df[df.study=="real"].copy()
metrics=["AUROC","AP","Brier","LogLoss","BA"];mcs=[]
for key,g in sim.groupby(["design","scenario_id","scenario_number","method_id","method_number","method"],dropna=False):
 context=dict(zip(["design","scenario_id","scenario_number","method_id","method_number","method"],key))
 for metric in metrics:
  vals=g["Delta_"+metric].dropna()
  mcs.append(context|dict(metric=metric,n_requested=len(g),**mc(vals),oracle_mean_on_pairs=float(g.loc[vals.index,"oracle_"+metric].mean()) if len(vals) else None,
                         estimated_mean_on_pairs=float(g.loc[vals.index,"estimated_"+metric].mean()) if len(vals) else None,
                         prior_failure_count=int((g.prior_status!="success").sum()),fit_failure_count=int((g.status=="fit_failed").sum()),
                         scope="paired_same_model_pilot" if metric=="BA" else "final_refit_rank_or_probability"))
mctable=pd.DataFrame(mcs)
real_summary=[]
for key,g in real.groupby(["dataset","design","budget","method_id","method_number","method"],dropna=False):
 context=dict(zip(["dataset","design","budget","method_id","method_number","method"],key))
 for metric in metrics:
  vals=(g.loc[g.rep==1,"Delta_BA"] if metric=="BA" else g["Delta_"+metric]).dropna();z=mc(vals)
  real_summary.append(context|dict(summary_level="dataset_design_budget",metric=metric,n_requested=len(g),
       **z,oracle_mean=float(g.loc[vals.index,"oracle_"+metric].mean()) if len(vals) else None,
       estimated_mean=float(g.loc[vals.index,"estimated_"+metric].mean()) if len(vals) else None,
       n_conditions=1,complete_required_conditions=bool(len(vals)),prior_failure_count=int((g.prior_status!="success").sum()),
       fit_failure_count=int((g.status=="fit_failed").sum())))
condition=pd.DataFrame(real_summary)
for key,g in condition.groupby(["dataset","method_number","method","metric"],dropna=False):
 context=dict(zip(["dataset","method_number","method","metric"],key));valid=g[g.n_valid>0]
 ok=len(valid)==4
 real_summary.append(context|dict(summary_level="dataset_equal_four_conditions",design="equal_designs",budget="equal_budgets",
       method_id="PU"+str(context["method_number"]).zfill(3),n_conditions=len(valid),complete_required_conditions=ok,
       n_requested=int(g.n_requested.sum()),n_valid=int(g.n_valid.sum()),
       mean_difference=float(valid.mean_difference.mean()) if ok else None,
       oracle_mean=float(valid.oracle_mean.mean()) if ok else None,estimated_mean=float(valid.estimated_mean.mean()) if ok else None,
       MCSE=None,MC_lower=None,MC_upper=None))
dataset=pd.DataFrame([r for r in real_summary if r["summary_level"]=="dataset_equal_four_conditions"])
for key,g in dataset.groupby(["method_number","method","metric"],dropna=False):
 context=dict(zip(["method_number","method","metric"],key));valid=g[g.complete_required_conditions];ok=len(valid)==8
 real_summary.append(context|dict(summary_level="equal_eight_datasets",dataset="equal_eight_datasets",design="equal_designs",budget="equal_budgets",
       method_id="PU"+str(context["method_number"]).zfill(3),n_datasets=len(valid),complete_required_conditions=ok,
       n_requested=int(g.n_requested.sum()),n_valid=int(g.n_valid.sum()),
       mean_difference=float(valid.mean_difference.mean()) if ok else None,
       oracle_mean=float(valid.oracle_mean.mean()) if ok else None,estimated_mean=float(valid.estimated_mean.mean()) if ok else None,
       MCSE=None,MC_lower=None,MC_upper=None))
rs=pd.DataFrame(real_summary)
association=df[df.ranking_prior_dependent].copy()
association["record_type"]="paired_case"
association["prior_absolute_error_magnitude"]=association.absolute_prior_error.abs()
cor=[]
for key,g in association.groupby(groups+["method_id","method_number","method"],dropna=False):
 context=dict(zip(groups+["method_id","method_number","method"],key))
 for error in ["absolute_prior_error","relative_prior_error","c_error"]:
  for metric in ["AUROC","AP"]:
   z=g[[error,"Delta_"+metric]].dropna()
   rho=float(z[error].rank().corr(z["Delta_"+metric].rank())) if len(z)>2 and z[error].nunique()>1 and z["Delta_"+metric].nunique()>1 else None
   cor.append(context|dict(record_type="within_condition_spearman",error_variable=error,performance_variable="Delta_"+metric,
                          n_valid=len(z),spearman_rho=rho,interpretation="descriptive_not_causal"))
error_table=pd.concat([association,pd.DataFrame(cor)],ignore_index=True)
coverage=[]
for key,g in ac.groupby(groups,dropna=False):
 context=dict(zip(groups,key));v=g.loc[g.status=="success","prior_ratio"].dropna();n=len(v)
 z=context|dict(record_type="estimated_prior_coverage",n_requested=len(g),n_valid=n,failure_rate=1-n/len(g),
       ratio_mean=float(v.mean()) if n else None,ratio_median=float(v.median()) if n else None,
       ratio_q05=float(v.quantile(.05)) if n else None,ratio_q25=float(v.quantile(.25)) if n else None,
       ratio_q75=float(v.quantile(.75)) if n else None,ratio_q95=float(v.quantile(.95)) if n else None)
 for width,label in [(.2,"20"),(.1,"10"),(.05,"5")]:
  z["fraction_within_"+label+"pct"]=float(v.between(1-width,1+width,inclusive="both").mean()) if n else None
 coverage.append(z)
old_provenance={}
for design,suffix in [("one_sample","one"),("two_sample","two")]:
 path=R/("publication_preparation/eta_scoring_20261013/collected/prior_"+suffix+"_raw.csv")
 old=pd.read_csv(path,usecols=["source_scenario_id","scenario_id","rep","method_id","prior_ratio","AUROC","AUPRC","data_fingerprint"],float_precision="round_trip")
 old=old[old.method_id.str[-3:].astype(int)>=3]
 baseline=old[old.prior_ratio==1][["scenario_id","rep","method_id","AUROC","AUPRC","data_fingerprint"]]
 merged=old.merge(baseline,on=["scenario_id","rep","method_id"],suffixes=("","_oracle"),validate="many_to_one")
 assert (merged.data_fingerprint==merged.data_fingerprint_oracle).all()
 merged["Delta_AUROC"]=merged.AUROC-merged.AUROC_oracle;merged["Delta_AP"]=merged.AUPRC-merged.AUPRC_oracle
 for key,g in merged.groupby(["source_scenario_id","method_id","prior_ratio"]):
  scenario,method,ratio=key
  coverage.append(dict(record_type="archived_misspecification_context",study="main",design=design,scenario_id=scenario,dataset="",budget="",
      method_id=method,prior_ratio_setting=ratio,n_requested=len(g),n_valid=len(g),
      archived_Delta_AUROC_mean=float(g.Delta_AUROC.mean()),archived_Delta_AP_mean=float(g.Delta_AP.mean()),
      comparison_scope="separate_archived_seeds_context_not_paired_with_Task6",c_treatment="fixed_true_c" if design=="one_sample" else "not_applicable"))
 old_provenance[str(path.relative_to(R))]=sha(path)
failures=pd.DataFrame(estfail,columns=ac.columns)
outputs=[]
for name,frame in [
 ("prior_estimation_accuracy_case",ac),("prior_estimation_accuracy_summary",accsummary),
 ("prior_estimation_failures",failures),
 ("estimated_prior_simulation_results",armdf[armdf.study=="main"]),
 ("estimated_prior_simulation_differences",sim),("estimated_prior_mc_uncertainty",mctable),
 ("estimated_prior_real_results",armdf[armdf.study=="real"]),("estimated_prior_real_summary",rs),
 ("prior_error_vs_performance",error_table),("prior_sensitivity_coverage",pd.DataFrame(coverage))]:
 outputs.append(export(name,frame))
with (H/"analysis_validation.json").open("x") as f:json.dump(dict(status="passed",cases=2240,paired_method_rows=len(df),information_arm_rows=len(armdf),
 estimator_failures=len(estfail),learner_failures=len(fitfail),old_sensitivity_sources=old_provenance,outputs=outputs),f,indent=2)
with (H/"learner_failures.json").open("x") as f:json.dump(fitfail,f,default=str,indent=2)
print(json.dumps(outputs,indent=2))
