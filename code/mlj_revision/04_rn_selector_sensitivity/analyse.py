"""Aggregate completed Task 4 fits. Draws are averaged within a case, never treated as MC repetitions."""
from pathlib import Path
import json,math,collections,sys
import numpy as np,pandas as pd
H=Path(__file__).resolve().parent
CORE_ONLY='--core-only' in sys.argv
META=['job_key','case_key','scope','study','design','scenario_id','scenario_number','dataset','budget','rep','seed','engine','selector','data_fingerprint']
KEY=['job_key','stage','condition']
QUALITY=['nP','nU','selected_n','true_negatives_selected','hidden_positives_selected','hidden_negatives_U','hidden_positives_U','RN_precision','RN_contamination','RN_negative_recall','RN_positive_capture','RN_selection_fraction','final_training_n','rn_training_positive_fraction','hard_negative_quartile_recall','easy_negative_quartile_recall']
OVERLAP=['intersection_n','union_n','jaccard','logistic_in_other','other_in_logistic']
OUTCOMES=['AUROC','AP','BA']
def table(name,d):
 for column in d:
  if pd.api.types.is_string_dtype(d[column]) or d[column].dtype==object:
   d[column]=d[column].map(lambda x:json.dumps(x,sort_keys=True,separators=(',',':')) if isinstance(x,(dict,list)) else None if isinstance(x,str) and not x else x)
 assert not (H/(name+'.csv')).exists(),name
 d.to_pickle(H/(name+'.staging.pkl'))
 with (H/(name+'.export.ndjson')).open('w',encoding='utf-8') as f:
  f.write(json.dumps(dict(columns=list(d.columns)))+'\n')
  for start in range(0,len(d),1000):
   values=[[None if pd.isna(x) else x.item() if isinstance(x,np.generic) else x for x in row] for row in d.iloc[start:start+1000].itertuples(index=False,name=None)]
   f.write(json.dumps(values,separators=(',',':'),allow_nan=False)+'\n')
 print(name,len(d),len(d.columns),flush=True)
def main():
 case=[];quality=[];overlap=[];perf=[];distribution=[]
 seen=set()
 for path in sorted((H/'results').glob('*/complete.json')):
  d=json.loads(path.read_text())
  if CORE_ONLY and d['scope']!='core':continue
  meta={k:d.get(k) for k in META};assert d['job_key'] not in seen;seen.add(d['job_key'])
  case.append(meta|dict(original_RN_params=d['original_RN_params'],final_RN_params=d['final_params'],original_naive_params=d['original_naive_params'],naive_final_params_differ=d['final_params']!=d['original_naive_params'],task3_arrays_sha256=d['task3_arrays_sha256'],task3_record_sha256=d['task3_record_sha256'],worker_sha256=d['worker_sha256'],source_code_hashes=d['source_code_hashes'],packages=d['packages'],threads=d['threads'],hostname=d['hostname'],runtime_seconds=d['seconds'],same_model_BA_available=any(p['stage']=='selected_train_only' for p in d['performance']),fixed_final_aliases=d['fixed_final_aliases'],logistic_archive_max_abs_difference=max(abs(v) for z in d['logistic_archive_checks'] for v in z['archived_metric_differences'].values()),RN_training_order='P first then selected U in original index order',archived_naive_training_order='original development row order',hyperparameter_searches=d['hyperparameter_searches'],hidden_Y_practical_selection=d['hidden_Y_practical_selection'],archive_source='original immutable benchmark outcomes; Task3 reconstructed logistic mask; Task4 new final fits',case_record=path.relative_to(H).as_posix()))
  for field,dest in [('quality',quality),('overlap',overlap),('performance',perf)]:
   for row in d[field]:dest.append(meta|row|dict(record_type='individual_fit_or_archive'))
  # Scores and coordinate distributions remain draw-specific; no averaged quantile is labelled a pooled quantile.
  for row in d['distributions']:distribution.append(meta|row)
 case=pd.DataFrame(case);q=pd.DataFrame(quality);o=pd.DataFrame(overlap);p=pd.DataFrame(perf);dist=pd.DataFrame(distribution)
 pilot=case[case.scope.eq('core')]
 assert len(pilot)==1000 and (pilot.groupby(['design','scenario_number']).size()==100).all()
 decision=json.loads((H/'pilot_decision.json').read_text())
 if not CORE_ONLY and (H/'secondary_jobs.json').exists():assert sum(case.scope.eq('secondary'))==400
 if not CORE_ONLY and decision['decision']=='A':
  assert sum(case.scope.eq('engine_extension'))==2000
  assert sum(case.scope.eq('selector_robustness'))==1000
  assert sum(case.scope.eq('real_extension'))==640
 if not CORE_ONLY and decision['decision']=='B':assert sum(case.scope.eq('real_extension'))==640
 # Average each repeated-random condition at the CASE level. Single fits are unchanged.
 meta=case[META].set_index('job_key')
 means=[]
 for key,g in p.groupby(KEY,dropna=False):
  row=dict(zip(KEY,key));row.update(meta.loc[key[0]].to_dict())
  row.update(record_type='case_condition_mean',n_draws_expected=10 if 'random_size' in key[2] or key[2].startswith('Oracle_') else 1,n_draws_recorded=len(g),final_params=g.final_params.iloc[0],final_seed=g.final_seed.iloc[0],source=g.source.iloc[0],fixed_final_alias=key[2]+'_fixed_final' if key[2] in ['RN_logistic','RN_nonlinear','RN_nonlinear_size_matched'] else None,selector_condition_alias='RN_XGB_selector' if row['selector']=='XGBoost' and key[2]=='RN_nonlinear' else None)
  for metric in OUTCOMES:
   vals=g[metric].dropna()
   row[metric]=vals.mean() if len(vals)==row['n_draws_expected'] else np.nan
   row[metric+'_n_valid_draws']=len(vals)
   row[metric+'_within_case_draw_SD']=vals.std(ddof=1) if len(vals)>1 else np.nan
  row['all_expected_draws_recorded']=len(g)==row['n_draws_expected']
  assert row['all_expected_draws_recorded']
  means.append(row)
 means=pd.DataFrame(means)
 # Join original naive and logistic comparators within the exact same case/model stage.
 for condition,prefix in [('Naive_all_U','vs_naive'),('RN_logistic','vs_logistic')]:
  reference=means[means.condition.eq(condition)][['job_key','stage']+OUTCOMES]
  means=means.merge(reference.rename(columns={k:k+'_'+prefix+'_reference' for k in OUTCOMES}),on=['job_key','stage'],validate='many_to_one')
  for metric in OUTCOMES:means['Delta_'+metric+'_'+prefix]=means[metric]-means[metric+'_'+prefix+'_reference']
 # Retain draw-level results plus clearly labelled case summaries in delivered performance table.
 p=pd.concat([p,means],ignore_index=True)
 qm=q.groupby(KEY,dropna=False)[[x for x in QUALITY if x in q]].mean().reset_index()
 qm=qm.merge(case[META],on='job_key',validate='many_to_one');qm['record_type']='case_condition_mean'
 qm['aggregation']='mean of draw-specific diagnostic ratios; single deterministic fit unchanged'
 q=pd.concat([q,qm],ignore_index=True)
 om=o.groupby(KEY,dropna=False)[OVERLAP].mean().reset_index().merge(case[META],on='job_key',validate='many_to_one');om['record_type']='case_condition_mean'
 o=pd.concat([o,om],ignore_index=True)
 # Prespecified paired contrasts include controls separating retained size and composition.
 contrasts=[('RN_logistic','Naive_all_U'),('RN_nonlinear','Naive_all_U'),('RN_nonlinear_size_matched','Naive_all_U'),('RN_random_size_matched','Naive_all_U'),('Oracle_true_negative_size_matched','Naive_all_U'),('RN_nonlinear','RN_logistic'),('RN_nonlinear_size_matched','RN_logistic'),('RN_random_size_matched','RN_logistic'),('Oracle_true_negative_size_matched','RN_logistic'),('RN_nonlinear','RN_nonlinear_size_matched'),('RN_nonlinear_size_matched','RN_random_size_matched'),('All_U_RN_fixed_final','RN_random_size_matched'),('All_U_RN_fixed_final','RN_logistic'),('Naive_all_U','All_U_RN_fixed_final'),('Oracle_true_negative_size_matched','RN_random_size_matched')]
 pairs=[]
 for (job,stage),g in means.groupby(['job_key','stage']):
  g=g.set_index('condition')
  for high,low in contrasts:
   if high not in g.index or low not in g.index:continue
   for metric in OUTCOMES:
    if (stage=='selected_train_only')!=(metric=='BA'):continue
    a=g.loc[high,metric];b=g.loc[low,metric]
    pairs.append(dict(job_key=job,stage=stage,condition=high,comparator=low,contrast=high+' minus '+low,outcome=metric,condition_value=a,comparator_value=b,paired_difference=a-b,valid_pair=bool(pd.notna(a) and pd.notna(b)),primary_mechanism_contrast=high=='RN_nonlinear_size_matched' and low=='RN_logistic',threshold_policy='Task1B_A_same_model' if metric=='BA' else 'not_applicable')|meta.loc[job].to_dict())
 pairs=pd.DataFrame(pairs)
 aliases={'RN_logistic':'RN_logistic_fixed_final','RN_nonlinear':'RN_nonlinear_fixed_final','RN_nonlinear_size_matched':'RN_nonlinear_size_matched_fixed_final'}
 pairs['condition_fixed_final_alias']=pairs.condition.map(aliases)
 pairs['comparator_fixed_final_alias']=pairs.comparator.map(aliases)
 pairs['fixed_final_contrast_alias']=[aliases[a]+' minus '+aliases[b] if a in aliases and b in aliases else None for a,b in zip(pairs.condition,pairs.comparator)]

 groupcols=['scope','study','design','scenario_id','scenario_number','dataset','budget','engine','selector','stage','contrast','condition','comparator','outcome','primary_mechanism_contrast']
 summary=[]
 for key,g in pairs.groupby(groupcols,dropna=False):
  z=g.paired_difference.dropna();n=len(z);mean=z.mean() if n else np.nan;sd=z.std(ddof=1) if n>1 else np.nan;se=sd/math.sqrt(n) if n>1 else np.nan
  summary.append(dict(zip(groupcols,key))|dict(n_valid_pairs=n,n_expected_pairs=1 if g.outcome.iloc[0]=='BA' else 100 if g.study.iloc[0]=='main' else 20,mean_paired_difference=mean,sd_paired_differences=sd,MCSE=se,MC95_low=mean-1.96*se,MC95_high=mean+1.96*se,uncertainty_scope='within fixed scenario; not across designed scenarios' if g.study.iloc[0]=='main' else 'repeated splits conditional on fixed dataset; not independent datasets',coverage='one_repetition_secondary_BA_pilot' if g.outcome.iloc[0]=='BA' else 'complete' if n==(100 if g.study.iloc[0]=='main' else 20) else 'partial_or_undefined'))
 mc=pd.DataFrame(summary)
 joined=means.merge(qm[KEY+[x for x in QUALITY if x in qm]],on=KEY,validate='one_to_one').merge(om[KEY+OVERLAP],on=KEY,how='left',validate='one_to_one')
 interaction=joined[joined.scenario_number.eq(39)].copy()
 # Add draw-averaged within-set distribution statistics for convenient per-repetition interaction tables.
 di=dist[dist.scenario_number.eq(39)].copy()
 if len(di):
  di['variable']=di.group+'__'+di.score_type
  wide=di.groupby(KEY+['variable'])[['mean','SD','q10','q25','q50','q75','q90']].mean().unstack('variable')
  wide.columns=[var+'__'+stat for stat,var in wide.columns]
  interaction=interaction.merge(wide.reset_index(),on=KEY,how='left',validate='one_to_one')
  interaction['distribution_aggregation']='single-fit statistics; random/oracle entries average draw-specific statistics, not pooled quantiles'
 outputs=dict(rn_selector_case_level=case,rn_selector_quality=q,rn_selector_overlap=o,rn_selector_performance=p,rn_selector_paired_differences=pairs,rn_selector_mc_uncertainty=mc,rn_selector_easy_negative_diagnostics=dist,rn_interaction_selector_diagnostics=interaction)
 if case.study.eq('real').any():
  real=joined[joined.study.eq('real')].copy();outputs['rn_selector_real_data']=real
  vals=OUTCOMES+QUALITY+['Delta_AUROC_vs_naive','Delta_AP_vs_naive','Delta_AUROC_vs_logistic','Delta_AP_vs_logistic']
  vals=[k for k in vals if k in real]
  rs=real.groupby(['dataset','design','budget','engine','selector','stage','condition'],dropna=False)[vals].mean().reset_index();rs['summary_type']='dataset_design_budget_mean';rs['n_splits']=20
  eq=rs.groupby(['dataset','engine','selector','stage','condition'],dropna=False)[vals].mean().reset_index();eq['summary_type']='dataset_equal_four_condition_mean';eq['n_conditions']=4
  outputs['rn_selector_real_data_summary']=pd.concat([rs,eq],ignore_index=True)
 if not CORE_ONLY:
  for name,d in outputs.items():table(name,d)
 # Smaller scientific summaries for reports and plots, always using one case-level observation.
 joined.to_pickle(H/('core_case_condition_summary.pkl' if CORE_ONLY else 'case_condition_summary.pkl'))
 cols=['scope','study','design','scenario_id','scenario_number','dataset','budget','engine','selector','stage','condition']
 numeric=[k for k in OUTCOMES+QUALITY+OVERLAP+['Delta_AUROC_vs_naive','Delta_AP_vs_naive','Delta_AUROC_vs_logistic','Delta_AP_vs_logistic'] if k in joined]
 plot=joined.groupby(cols,dropna=False)[numeric].mean().reset_index()
 (H/'plot_summary.json').write_text(plot.to_json(orient='records',double_precision=15))
 condition_uncertainty=[]
 for key,g in joined.groupby(cols,dropna=False):
  row=dict(zip(cols,key))
  for metric in OUTCOMES:
   z=g[metric].dropna();n=len(z);se=z.std(ddof=1)/math.sqrt(n) if n>1 else np.nan
   row.update({metric+'_mean':z.mean(),metric+'_n':n,metric+'_MCSE':se})
  condition_uncertainty.append(row)
 (H/'plot_condition_uncertainty.json').write_text(pd.DataFrame(condition_uncertainty).to_json(orient='records',double_precision=15))
 (H/'plot_paired_uncertainty.json').write_text(mc.to_json(orient='records',double_precision=15))
 dp=dist[(dist.scope=='core')&(dist.scenario_number==39)&(dist.stage=='final_train_plus_validation')]
 # Equal draws per random/oracle case make this equivalent to averaging draws, then cases.
 dp=dp.groupby(['design','condition','group','score_type'])[['n','mean','SD','q10','q25','q50','q75','q90']].mean().reset_index()
 (H/'plot_interaction_distribution.json').write_text(dp.to_json(orient='records',double_precision=15))

 validations=dict(status='assembled',cases=len(case),scopes=case.scope.value_counts().to_dict(),primary_case_condition_rows=len(means),individual_fit_or_archive_rows=int(p.record_type.eq('individual_fit_or_archive').sum()),new_fit_rows=int((p.record_type.eq('individual_fit_or_archive')&p.source.eq('Task4_new_fixed_parameter_fit')&p.status.eq('complete')).sum()),reused_identical_fit_rows=int((p.record_type.eq('individual_fit_or_archive')&p.source.eq('Task4_reused_identical_fit')).sum()),draws_averaged_within_case=True,hyperparameter_searches=int(case.hyperparameter_searches.sum()),BA_core_cases=int(case.same_model_BA_available.sum()),old_threshold_BA_used=False,decision=decision['decision'],tables={name:dict(rows=len(d),columns=len(d.columns)) for name,d in outputs.items()})
 (H/('core_analysis_validation.json' if CORE_ONLY else 'analysis_validation.json')).write_text(json.dumps(validations,indent=2));print(json.dumps(validations,indent=2))
if __name__=='__main__':main()
