"""Tables and exploratory analyses. No benchmark models fitted."""
from pathlib import Path
import json,hashlib,math,collections
import numpy as np,pandas as pd
H=Path(__file__).resolve().parent;R=H.parents[1]
FAMILIES=['reference','prevalence','positive-label availability','class separation','dimensionality','predictor correlation','joint prevalence-label availability','curved boundary','feature interaction','unequal class covariance']
DIAGS=['RN_precision','RN_contamination','RN_negative_recall','RN_positive_capture','RN_selection_fraction','final_training_n','rn_training_positive_fraction']
EXTRA=['RN_n','nP','nU','discarded_negative_fraction','easy_negative_score_gap','hardest_negative_quartile_recall','easiest_negative_quartile_recall','logistic_spy_score_U_AUROC','exact_DGP_LR_U_AUROC','LR_minus_spy_U_AUROC']
METRICS=['Delta_AUROC','Delta_AP','Delta_BA']
def dump_table(name,d):
 d=d.copy()
 for col in d.columns:
  if d[col].dtype==object:d[col]=d[col].map(lambda x:json.dumps(x,separators=(',',':')) if isinstance(x,(list,dict)) else None if isinstance(x,str) and x=='' else x)
 d.to_pickle(H/(name+'.staging.pkl'))
 # JSON serializer retains float precision; artifact-tool authors verified CSV cells.
 obj={'columns':list(d.columns),'values':[[None if pd.isna(x) else x.item() if isinstance(x,np.generic) else x for x in row] for row in d.itertuples(index=False,name=None)]}
 (H/(name+'.export.json')).write_text(json.dumps(obj,separators=(',',':'),allow_nan=False))
 print(name,len(d),len(d.columns),flush=True)
def correlation_rows(d,study):
 out=[]
 groups=[('pooled',{},d)]
 if study=='main':
  groups += [('design_engine',dict(design=de,engine=en),g) for (de,en),g in d.groupby(['design','engine'])]
  groups += [('scenario_engine',dict(design=de,scenario_id=sc,scenario_family=g.scenario_family.iloc[0],engine=en),g) for (de,sc,en),g in d.groupby(['design','scenario_id','engine'])]
  groups += [('family_design_engine',dict(design=de,scenario_family=fa,engine=en),g) for (de,fa,en),g in d.groupby(['design','scenario_family','engine'])]
 else:
  groups += [('dataset_design_engine',dict(dataset=da,design=de,engine=en),g) for (da,de,en),g in d.groupby(['dataset','design','engine'])]
 for scope,key,g in groups:
  for y in METRICS[:2]:
   for x in DIAGS+['hardest_negative_quartile_recall','LR_minus_spy_U_AUROC']:
    if x not in g:continue
    v=g[[x,y]].dropna();n=len(v)
    rho=float(v.rank(method='average').corr().iloc[0,1]) if n>=3 and v[x].nunique()>1 and v[y].nunique()>1 else None
    out.append(dict(analysis_type='spearman',study=study,scope=scope,**key,outcome=y,diagnostic=x,n_valid=n,spearman_rho=rho,p_value=None,interpretation='descriptive; dependent engine rows and designed scenarios; no causal inference'))
 return out
def regressions(d,study):
 rows=[];xs=['RN_precision','RN_negative_recall','RN_selection_fraction']
 models=[('unadjusted',[]),('design_engine_family_adjusted',['design','engine','scenario_family'] if study=='main' else ['design','engine','dataset','budget'])]
 for outcome in METRICS[:2]:
  for model,cats in models:
   v=d.dropna(subset=xs+[outcome]).copy()
   if len(v)<10:continue
   # Scale continuous predictors to SD units for readable coefficient comparison.
   means=v[xs].mean();sd=v[xs].std(ddof=0);Z=(v[xs]-means)/sd
   cat=pd.get_dummies(v[cats].astype(str),drop_first=True,dtype=float) if cats else pd.DataFrame(index=v.index)
   X=np.c_[np.ones(len(v)),Z.to_numpy(),cat.to_numpy()]
   names=['intercept']+xs+list(cat.columns)
   if study=='main':
    sizes=v.groupby(['design','engine','scenario_family']).key.transform('size')
    w=1/sizes.to_numpy(dtype=float)
   else:
    sizes=v.groupby(['dataset','design','budget','engine']).key.transform('size');w=1/sizes.to_numpy(dtype=float)
   w=w/w.mean();y=v[outcome].to_numpy()*100
   coef,_,rank,s=np.linalg.lstsq(X*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)
   fit=X@coef;r2=1-np.sum(w*(y-fit)**2)/np.sum(w*(y-np.average(y,weights=w))**2)
   # Standardized predictor coefficients; categorical references recorded explicitly.
   baseline={c:sorted(v[c].astype(str).unique())[0] for c in cats}
   condition=float(s.max()/s.min()) if s.min()>0 else np.inf
   for j,name in enumerate(names):
    rows.append(dict(analysis_type='descriptive_regression',study=study,scope=model,outcome=outcome,diagnostic=name,n_valid=len(v),coefficient_pp_per_SD=coef[j] if name in xs else None,coefficient_pp_per_unit=coef[j]/sd[name] if name in xs else coef[j],predictor_SD=sd[name] if name in xs else None,weighted_R_squared=r2,design_matrix_rank=int(rank),design_matrix_columns=X.shape[1],condition_number=condition,categorical_reference=json.dumps(baseline),weighting='equal design/engine/family then repetitions' if study=='main' else 'equal dataset/design/budget/engine then repetitions',p_value=None,interpretation='exploratory association; jointly tuned engine parameters; no causal attribution'))
 return rows
def main():
 perf=pd.read_pickle(H/'matched_inputs.pkl');cases=json.loads((H/'cases.json').read_text());case_by={c['key']:c for c in cases}
 results={p.stem:json.loads(p.read_text()) for p in (H/'results').glob('*.json')}
 repaired_keys={x['case'] for x in json.loads((H/'failed_case_original_metadata.json').read_text())}
 assert len(results)==len(cases),('Incomplete selector recovery',len(results),len(cases))
 raw=[];inventory=[];counts=collections.Counter()
 for c in cases:
  res=results[c['key']]
  assert res['data_fingerprint']==c['fingerprint'] and res['original_fingerprint_matched']
  assert res['boosting_fits']==res['hyperparameter_searches']==0
  counts.update(spy_logistic_fits=res['spy_logistic_fits'],cache_hits=res['identical_fit_cache_hits'])
  for mid,m in c['methods'].items():
   p=m['params'];selected=[q for q in res['rows'] if q['spy_fraction']==p['spy_fraction'] and q['spy_quantile']==p['spy_quantile']]
   assert sum(q['stage']=='final_train_plus_validation' for q in selected)==1
   for q in selected:
    if q['stage']=='selected_train_only' and not m['same_model_BA_available']:continue
    raw.append(dict(key=c['key'],method_id=mid,arrays_archive='replayed_RN_sets.tar',arrays_member='sets/'+c['key']+'.npz',replay_audit='results/'+c['key']+'.json',hardware_compatibility_recovery=c['key'] in repaired_keys,**q))
   record={k:c.get(k) for k in ['key','study','design','scenario_id','scenario_family','dataset','budget','rep','seed','fingerprint','source_case_manifest','performance_source','frozen_remote']}
   inventory.append(record|dict(method_id=mid,engine=m['engine'],selected_params=json.dumps(p,sort_keys=True),selected_spy_fraction_recovered=True,selected_spy_quantile_recovered=True,engine_hyperparameters_recovered=True,seed_recovered=True,data_split_fingerprint_recovered=True,hidden_Y_recovered=True,hidden_Y_source='saved standardized audit arrays' if c['study']=='real' else 'exact original DGP with matched full-data fingerprint',archived_RN_indices_recovered=False,archived_candidate_counts_recovered=False,archived_pre_RN_training_score_recovered=False,final_RN_selection_reconstructed=True,train_only_RN_selection_reconstructed=m['same_model_BA_available'],archived_AUROC_AP_recovered=True,archived_test_predictions_recovered=c['study']=='real',same_model_BA_recovered=m['same_model_BA_available'],reconstruction_status='exact_inputs_code_environment_verified_no_archived_RN_set',direct_archived_RN_set_verification=False,replayed_arrays_source='mlj_revision/03_rn_mechanism/sets/'+c['key']+'.npz',replayed_arrays_sha256=res['arrays_sha256'],local_arrays_archive='replayed_RN_sets.tar',local_arrays_member='sets/'+c['key']+'.npz',hardware_compatibility_recovery=c['key'] in repaired_keys,verified_compatible_replay_node='node21' if c['key'] in repaired_keys else None,replay_log_source='results/'+c['key']+'.json',replay_source='original selector reconstructed; not archived indices',candidate_counts_by_repeat_recorded=True,source_code_hashes=json.dumps(res['source_files'],sort_keys=True),original_package_versions=json.dumps(res['packages'],sort_keys=True),original_threads=res['threads'],verification_warning='No archived RN indices/counts/predictions linking directly to the selected set; replayed identity is not independently verified against an original set.'))
 quality=pd.DataFrame(raw)
 merged=perf.merge(quality,on=['key','method_id'],how='left',validate='one_to_many',suffixes=('','_selection'))
 assert len(merged)==43200+int(perf.Delta_BA.notna().sum())
 for c in ['spy_fraction','spy_quantile']:assert np.array_equal(merged[c],merged[c+'_selection']);merged=merged.drop(columns=c+'_selection')
 # Make outcome/model-stage alignment explicit. No refit-transfer BA enters this task.
 merged['ranking_outcome_stage']='final_train_plus_validation'
 merged['BA_outcome_stage']='selected_train_only'
 merged.loc[merged.stage=='selected_train_only',['Delta_AUROC','Delta_AP','RN_AUROC','naive_AUROC','RN_AP','naive_AP']]=np.nan
 merged.loc[merged.stage=='final_train_plus_validation',['Delta_BA','RN_BA','naive_BA']]=np.nan
 nonempty=merged.RN_n>0
 assert np.allclose((merged.RN_precision+merged.RN_contamination)[nonempty],1)
 assert merged.loc[~nonempty,'RN_precision'].isna().all()
 assert np.allclose(merged.RN_selection_fraction,merged.RN_n/merged.nU)
 assert np.allclose(merged.final_training_n,merged.nP+merged.RN_n)
 assert np.allclose(merged.RN_negative_recall,merged.RN_true_negatives/merged.hidden_negatives_U,equal_nan=True)
 final=merged[merged.stage=='final_train_plus_validation'].copy()
 sim=final[final.study=='main'];real=final[final.study=='real']
 # Per-scenario paired MC uncertainty is computed from all original paired outcomes.
 paired=[]
 group_cols=['study','design','scenario_id','dataset','budget','engine']
 for key,g in perf.groupby(group_cols,dropna=False):
  for metric in METRICS:
   v=g[metric].dropna();n=len(v);mean=v.mean() if n else np.nan;sd=v.std(ddof=1) if n>=2 else np.nan;se=sd/math.sqrt(n) if n>=2 else np.nan
   paired.append(dict(zip(group_cols,key))|dict(outcome=metric,n_valid_pairs=n,n_expected_pairs=100 if key[0]=='main' else 20,mean_paired_difference=mean,sd_paired_differences=sd,MCSE=se,MC95_low=mean-1.96*se,MC95_high=mean+1.96*se,positive_difference_fraction=float((v>0).mean()) if n else np.nan,threshold_policy='Task1B_A_same_model' if metric=='Delta_BA' else 'not_applicable',uncertainty_scope='simulation Monte Carlo within fixed scenario' if key[0]=='main' else 'repeated-split uncertainty conditional on this fixed dataset',coverage='complete' if n==(100 if key[0]=='main' else 20) else 'single_repetition_pilot' if n==1 else 'missing' if n==0 else 'partial'))
 paired=pd.DataFrame(paired)
 # Equal settings within family; equal families in preferred overall summaries.
 values=DIAGS+EXTRA+METRICS[:2]
 scenario=sim.groupby(['design','engine','scenario_id','scenario_number','scenario_family'],dropna=False)[values].mean().reset_index()
 ns=sim.groupby(['design','engine','scenario_id']).size().rename('n_repetitions').reset_index();scenario=scenario.merge(ns,on=['design','engine','scenario_id'],validate='one_to_one')
 fam=scenario.groupby(['design','engine','scenario_family'])[values].mean().reset_index()
 fam['n_settings']=fam.scenario_family.map(scenario.groupby('scenario_family').scenario_id.nunique()/2).astype(int)
 fam['summary_type']='within_family'
 eq=fam.groupby(['design','engine'])[values].mean().reset_index();eq['scenario_family']='all_ten_families';eq['n_settings']=40;eq['n_families']=10;eq['summary_type']='equal_10_family_mean'
 family=pd.concat([fam,eq],ignore_index=True)
 old=scenario.groupby(['design','engine'])[values].mean().reset_index();old['n_settings']=40;old['summary_type']='original_40_setting_mean'
 # BA has no full family coverage: do not fill an overall summary from the pilot.
 for tab in [family,old]:tab['Delta_BA']=np.nan;tab['BA_status']='unavailable_complete_family_coverage; pilot kept in paired/per-repetition tables'
 associations=correlation_rows(sim,'main')+correlation_rows(real,'real')+regressions(sim,'main')+regressions(real,'real')
 a=pd.DataFrame(associations)
 sc=scenario.copy();sc['analysis_type']='scenario_summary';sc['study']='main';sc['scope']='100_repetition_scenario_mean'
 fs=fam.copy();fs['analysis_type']='family_summary';fs['study']='main';fs['scope']='settings_averaged_equally_within_family'
 quality_perf=pd.concat([a,sc,fs],ignore_index=True)
 spyrows=[]
 for study,sub in [('main',sim),('real',real)]:
  cols=['design','engine','scenario_id','scenario_family'] if study=='main' else ['dataset','design','budget','engine']
  for key,g in sub.groupby(cols,dropna=False):
   for sf in [.1,.2]:
    for sq in [.01,.05]:
     v=g[(g.spy_fraction==sf)&(g.spy_quantile==sq)]
     spyrows.append(dict(study=study,summary_type='scenario' if study=='main' else 'dataset_design_budget',**dict(zip(cols,key)),spy_fraction=sf,spy_quantile=sq,n_selected=len(v),n_total=len(g),selection_frequency=len(v)/len(g),**{m:v[m].mean() for m in DIAGS+METRICS[:2]}))
 spy=pd.DataFrame(spyrows)
 for st,cols in [('within_family',['design','engine','scenario_family','spy_fraction','spy_quantile']),('original_40_setting',['design','engine','spy_fraction','spy_quantile'])]:
  q=spy[(spy.study=='main')&(spy.summary_type=='scenario')].groupby(cols,dropna=False).agg(selection_frequency=('selection_frequency','mean'),n_selected=('n_selected','sum'),n_total=('n_total','sum')).reset_index();q['study']='main';q['summary_type']=st;spy=pd.concat([spy,q],ignore_index=True)
 q=spy[(spy.study=='main')&(spy.summary_type=='within_family')].groupby(['design','engine','spy_fraction','spy_quantile']).agg(selection_frequency=('selection_frequency','mean'),n_selected=('n_selected','sum'),n_total=('n_total','sum')).reset_index();q['study']='main';q['summary_type']='equal_10_family';spy=pd.concat([spy,q],ignore_index=True)
 # All engines retained in special-scenario diagnostics; CatBoost receives report emphasis.
 interaction=sim[sim.scenario_number.isin([1,2,3,12,26,28,38,39,40])].copy();interaction['record_type']='repetition'
 isc=interaction.groupby(['design','engine','scenario_id','scenario_number','scenario_family'])[values].mean().reset_index();isc['record_type']='scenario_mean'
 interaction=pd.concat([interaction,isc],ignore_index=True)
 realrows=real.copy();realrows['record_type']='repetition'
 rg=real.groupby(['dataset','design','budget','engine'])[DIAGS+EXTRA+METRICS[:2]].mean().reset_index();rg['record_type']='dataset_design_budget_mean';rg['n_splits']=20
 # Dataset-engine means give each of four design/budget conditions equal weight.
 rge=rg.groupby(['dataset','engine'])[DIAGS+EXTRA+METRICS[:2]].mean().reset_index();rge['record_type']='dataset_engine_equal_condition_mean';rge['n_conditions']=4
 realtable=pd.concat([realrows,rg,rge],ignore_index=True)
 outputs={'rn_recovery_inventory':pd.DataFrame(inventory),'rn_selection_quality_all':merged,'rn_matched_performance_all':perf,'rn_quality_vs_performance':quality_perf,'rn_family_balanced_summary':family,'rn_40setting_summary':old,'rn_paired_mc_uncertainty':paired,'rn_spy_parameter_summary':spy,'rn_interaction_diagnostics':interaction,'rn_real_data_diagnostics':realtable}
 for name,d in outputs.items():dump_table(name,d)
 validation=dict(cases=len(cases),pairs=len(perf),final_quality_rows=len(final),train_only_quality_rows=len(merged)-len(final),simulation_final_rows=len(sim),real_final_rows=len(real),same_model_BA_valid_pairs=int(perf.Delta_BA.notna().sum()),boosting_model_fits=0,hyperparameter_searches=0,hidden_Y_only_retrospective=True,archived_RN_set_verification_available=False,**counts)
 (H/'analysis_validation.json').write_text(json.dumps(validation,indent=2));print(json.dumps(validation,indent=2))
if __name__=='__main__':main()
