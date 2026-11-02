"""Fixed catalog analyses from saved repetition results; no model fitting."""
from pathlib import Path
import json,math,hashlib,sys
import pandas as pd,numpy as np
H=Path(__file__).resolve().parent;R=H.parents[1];sys.dont_write_bytecode=True
FAMILIES=json.loads((H/'PLAN.json').read_text())['families']
ENG=['AdaBoost','GBM','XGBoost','LightGBM','CatBoost']
def name(n):
 if n<6:return {1:'Supervised logistic',2:'Supervised CatBoost',3:'Corrected logistic',4:'uPU linear',5:'nnPU linear'}[n]
 if n<26:return ['Naive','Corrected','RN','CS'][(n-6)//5]+' '+ENG[(n-6)%5]
 return 'AdaPU' if n==26 else 'PU gradient boosting'
def group(n):return 'supervised_reference' if n<3 else 'nonboosting_PU' if n<6 else 'naive_boosting' if n<11 else 'PU_aware_boosting'
def main():
 reg=pd.DataFrame(json.loads((H/'scenario_registry.json').read_text()));reg=reg[['design','scenario_id','scenario_index','scenario_label','family_id','family','p','pi','c','delta','rho']]
 raw=[]
 for design,stem in [('one_sample','one'),('two_sample','two')]:
  p=R/f'publication_preparation/eta_scoring_20261013/collected/main_{stem}_raw.csv'
  d=pd.read_csv(p,usecols=['scenario_id','rep','method_id','seed','AUROC','AUPRC','data_fingerprint'],float_precision='round_trip');d['design']=design
  assert len(d)==108000 and not d.duplicated(['scenario_id','rep','method_id']).any();assert d.groupby(['scenario_id','method_id']).rep.nunique().eq(100).all();raw.append(d)
 raw=pd.concat(raw,ignore_index=True);raw['method_number']=raw.method_id.str[-3:].astype(int)
 records=raw.melt(id_vars=['design','scenario_id','rep','method_id','method_number','seed','data_fingerprint'],value_vars=['AUROC','AUPRC'],var_name='metric',value_name='value');records['metric']=records.metric.replace({'AUPRC':'AP'});records['policy']='ranking_archive';records['source']='published_eta_archive';records['model_stage']='historical_refit'
 ba=[];p=H.with_name('01_threshold_reanalysis')/'threshold_reanalysis_all_results.csv'
 cols=['study','design','scenario_id','rep','method_id','seed','policy','BalancedAccuracy','availability']
 for chunk in pd.read_csv(p,usecols=cols,chunksize=100000,float_precision='round_trip',dtype={'seed':'string'}):
  b=chunk[(chunk.study=='main')&(chunk.policy=='A')].copy();ba.append(b)
 ba=pd.concat(ba,ignore_index=True);assert len(ba)==216000
 ba['seed']=pd.to_numeric(ba['seed']).astype(np.int64);ba['rep']=pd.to_numeric(ba['rep']).astype(np.int64)
 ba['value']=ba.BalancedAccuracy;ba['metric']='BalancedAccuracy';ba['policy']='Task1A_quantile_primary';ba['method_number']=ba.method_id.str[-3:].astype(int);ba['source']='Task1_policy_A';ba['model_stage']='historical_refit_threshold_transfer'
 ba=ba.merge(raw[['design','scenario_id','rep','method_id','data_fingerprint']],on=['design','scenario_id','rep','method_id'],validate='one_to_one')
 # Reuse the eight Task1B saved raw-validation / final-refit score pairs for TM003.
 recovered=[];B=H.with_name('01b_threshold_completion_and_refit_audit');inputs=json.loads((H/'input_hashes.json').read_text())
 for c in json.loads((B/'pilot_cases.json').read_text()):
  if c['study']!='main' or c['design']!='two_sample' or c['scope']!='fixed_pilot':continue
  mp=B/'pilot_results'/c['key']/'TM003.json';npz=mp.with_suffix('.npz');meta=json.loads(mp.read_text())
  assert hashlib.sha256(npz.read_bytes()).hexdigest()==meta['arrays_sha256']
  for p in [mp,npz]:inputs[p.relative_to(R).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
  rr=next(x for x in meta['results'] if x['policy']=='A_refit_transfer')
  with np.load(npz) as z:
   cut=float(np.quantile(z['q_val'][z['s_val']==0],1-c['pi_tilde']));assert cut==rr['threshold'];h=z['q_test_refit']>=cut;y=z['y_test'];value=.5*(h[y==1].mean()+(~h[y==0]).mean());assert value==rr['BalancedAccuracy']
  mask=(ba.design=='two_sample')&(ba.scenario_id==c['scenario_id'])&(ba.rep==c['rep'])&(ba.method_id=='TM003');assert mask.sum()==1 and ba.loc[mask,'value'].isna().all()
  ba.loc[mask,'value']=value;ba.loc[mask,'source']='Task1B_saved_raw_score_recovery';recovered.append(dict(case=c['key'],threshold=cut,BA=value))
 (H/'input_hashes.json').write_text(json.dumps(inputs,indent=2));(H/'TM003_no_fit_recovery.json').write_text(json.dumps(recovered,indent=2))
 pilot=pd.read_csv(H.with_name('01c_pu_balanced_accuracy_threshold')/'pu_ba_threshold_results.csv',float_precision='round_trip')
 pilot=pilot[(pilot.study=='simulation')&(pilot.scope=='fixed_pilot')&(pilot.rule=='C_PU_BA')&pilot.eligible_primary].copy();assert len(pilot)==400
 pilot['value']=pilot.BalancedAccuracy;pilot['metric']='BalancedAccuracy';pilot['policy']='Task1C_PU_BA_pilot';pilot['source']='Task1C_verified_pilot';pilot['model_stage']='selected_train_only'
 columns=['design','scenario_id','rep','method_id','method_number','seed','data_fingerprint','metric','value','policy','source','model_stage']
 data=pd.concat([records[columns],ba[columns],pilot[columns]],ignore_index=True);assert not data.duplicated(['design','scenario_id','rep','method_id','metric','policy']).any()
 data.to_pickle(H/'repetition_data.pkl')
 specs=[('AUROC','ranking_archive'),('AP','ranking_archive'),('BalancedAccuracy','Task1A_quantile_primary'),('BalancedAccuracy','Task1C_PU_BA_pilot')]
 agg=data.groupby(['design','scenario_id','method_number','metric','policy']).value.agg(['mean','std','count']).reset_index();alook=agg.set_index(['design','scenario_id','method_number','metric','policy'])
 bayes=pd.DataFrame(json.loads((H/'bayes_results.json').read_text())).set_index(['design','scenario_id'])
 scenario=[]
 for _,r in reg.iterrows():
  for n in range(1,28):
   mid=('M' if r.design=='one_sample' else 'TM')+f'{n:03d}'
   for metric,policy in specs:
    key=(r.design,r.scenario_id,n,metric,policy)
    v=alook.loc[key] if key in alook.index else None;count=int(v['count']) if v is not None else 0;mean=float(v['mean']) if count else np.nan
    status='complete_100_reps' if count==100 else 'pilot_one_rep' if policy.endswith('pilot') and count==1 else 'partial_saved_score_coverage' if count else 'unavailable_no_scores'
    if n<3 and policy.endswith('pilot'):status='not_applicable_supervised_reference'
    ceiling=bayes.loc[(r.design,r.scenario_id),'Bayes_AUROC' if metric=='AUROC' else 'Bayes_BA'] if metric!='AP' else np.nan
    scenario.append(r.to_dict()|dict(method_id=mid,method_number=n,method_name=name(n),method_group=group(n),practical_PU=n>=3,metric=metric,policy=policy,mean_value=mean,n_repetitions=count,n_expected_repetitions=100,complete_100_reps=count==100,status=status,rep_sd=float(v['std']) if count>1 else np.nan,MCSE_mean=float(v['std']/10) if count==100 else np.nan,Bayes_ceiling=ceiling,regret_to_Bayes=ceiling-mean if count else np.nan,model_stage='selected_train_only' if policy.endswith('pilot') else 'historical_refit'))
 scenario=pd.DataFrame(scenario)
 # Scenario ranks: strict full catalog, and explicitly separate available catalog.
 scenario['rank_25_PU']=np.nan;scenario['available_100rep_catalog_rank']=np.nan;scenario['rank_catalog_n']=np.nan;scenario['rank_status']='supervised_not_ranked'
 for key,ix in scenario[scenario.practical_PU].groupby(['design','scenario_id','metric','policy']).groups.items():
  p=scenario.loc[ix];complete=p[p.complete_100_reps];counts=len(complete);ranks=complete.mean_value.round(12).rank(ascending=False,method='average')
  scenario.loc[complete.index,'available_100rep_catalog_rank']=ranks;scenario.loc[ix,'rank_catalog_n']=counts
  if counts==25:scenario.loc[ix,'rank_25_PU']=p.mean_value.round(12).rank(ascending=False,method='average');status='complete_25_methods_100_reps'
  elif key[-1].endswith('pilot') and (p.n_repetitions==1).all():scenario.loc[ix,'rank_25_PU']=p.mean_value.round(12).rank(ascending=False,method='average');status='pilot_25_methods_one_rep'
  else:status='incomplete_25_method_catalog'
  scenario.loc[ix,'rank_status']=status
 # Average settings within family, then families equally. Strict full coverage.
 family=[];rankrows=[]
 for (design,n,metric,policy),part in scenario.groupby(['design','method_number','metric','policy']):
  common=dict(design=design,method_id=part.method_id.iloc[0],method_number=n,method_name=name(n),method_group=group(n),practical_PU=n>=3,metric=metric,policy=policy)
  famvals=[];famranks=[];famavail=[]
  for fid,fp in part.groupby('family_id'):
   ok=bool(fp.complete_100_reps.all());value=fp.mean_value.mean() if ok else np.nan;famvals.append(value)
   rv=fp.rank_25_PU.mean() if ok and fp.rank_25_PU.notna().all() and n>=3 else np.nan;famranks.append(rv)
   av=fp.available_100rep_catalog_rank.mean() if fp.available_100rep_catalog_rank.notna().all() else np.nan;famavail.append(av)
   family.append(common|dict(summary_type='within_family',family_id=fid,family=fp.family.iloc[0],settings_expected=len(fp),settings_complete=int(fp.complete_100_reps.sum()),settings_observed=int(fp.mean_value.notna().sum()),repetitions_available=int(fp.n_repetitions.sum()),mean_value=value,observed_setting_mean=fp.mean_value.mean(),complete=ok,original_setting_weight=len(fp)/40,equal_family_weight=.1))
   rankrows.append(common|dict(rank_level='within_family',family_id=fid,family=fp.family.iloc[0],mean_rank=rv,available_catalog_mean_rank=av,settings_expected=len(fp),settings_complete=int(fp.complete_100_reps.sum()),rank_status='complete' if np.isfinite(rv) else 'incomplete_or_reference'))
  allok=bool(part.complete_100_reps.all());original=part.mean_value.mean() if allok else np.nan;balanced=float(np.mean(famvals)) if np.isfinite(famvals).all() else np.nan
  for st,value in [('original_40_setting_mean',original),('equal_10_family_mean',balanced)]:
   family.append(common|dict(summary_type=st,family='ALL',settings_expected=40,settings_complete=int(part.complete_100_reps.sum()),settings_observed=int(part.mean_value.notna().sum()),repetitions_available=int(part.n_repetitions.sum()),mean_value=value,observed_setting_mean=part.mean_value.mean() if st.startswith('original') else np.nan,complete=allok,family_balanced_minus_original_pp=100*(balanced-original),families_expected=10,families_complete=int(np.isfinite(famvals).sum()),preferred=st=='equal_10_family_mean'))
  rankrows.append(common|dict(rank_level='equal_10_family',family='ALL',mean_rank=float(np.mean(famranks)) if np.isfinite(famranks).all() else np.nan,available_catalog_mean_rank=float(np.mean(famavail)) if np.isfinite(famavail).all() else np.nan,settings_expected=40,settings_complete=int(part.complete_100_reps.sum()),rank_status='complete' if np.isfinite(famranks).all() else 'incomplete_or_reference'))
 ranks=pd.concat([scenario.assign(rank_level='scenario').rename(columns={'rank_25_PU':'mean_rank','available_100rep_catalog_rank':'available_catalog_mean_rank'}),pd.DataFrame(rankrows)],ignore_index=True)
 family=pd.DataFrame(family)
 # One fixed nonboosting baseline, selected once on equal-family AUROC pooled
 # equally across designs, for description; all 3 baselines remain in contrasts.
 nb=family[(family.summary_type=='equal_10_family_mean')&(family.metric=='AUROC')&family.method_number.between(3,5)].groupby('method_number').mean_value.mean();best=int(nb.idxmax())
 selection=dict(criterion='highest equal-family mean AUROC, equal weight across the two designs',best_fixed_nonboosting_number=best,best_fixed_nonboosting_name=name(best),values={str(k):v for k,v in nb.items()},posthoc_global_descriptive_selection=True,all_three_baselines_contrasted=True)
 (H/'fixed_baseline_selection.json').write_text(json.dumps(selection,indent=2))
 comparisons=[(n+10,n,'RN_minus_naive') for n in range(6,11)]+[(n+15,n,'CS_minus_naive') for n in range(6,11)]+[(n,b,'fixed_boost_minus_fixed_nonboost') for b in [3,4,5] for n in list(range(6,11))+list(range(16,28))]
 contrasts=[]
 for (design,sid,metric,policy),part in data[data.method_number>=3].groupby(['design','scenario_id','metric','policy']):
  pivot=part.pivot(index=['rep','seed','data_fingerprint'],columns='method_number',values='value');rr=reg[(reg.design==design)&(reg.scenario_id==sid)].iloc[0]
  for a,b,typ in comparisons:
   vals=(pivot[a]-pivot[b]).dropna() if a in pivot and b in pivot else pd.Series(dtype=float);n=len(vals);mean=vals.mean() if n else np.nan;sd=vals.std(ddof=1) if n>1 else np.nan;se=sd/10 if n==100 else np.nan
   contrasts.append(dict(design=design,scenario_id=sid,scenario_index=rr.scenario_index,family=rr.family,family_id=rr.family_id,metric=metric,policy=policy,contrast_type=typ,method_a_number=a,method_b_number=b,method_a=name(a),method_b=name(b),best_fixed_nonboosting_comparator=b==best,direction='method_a_minus_method_b',n_pairs=n,n_expected=100,mean_difference=mean,mean_difference_pp=100*mean,paired_sd=sd,MCSE=se,MCSE_pp=100*se,MC95_low=mean-1.96*se,MC95_high=mean+1.96*se,MC95_low_pp=100*(mean-1.96*se),MC95_high_pp=100*(mean+1.96*se),status='complete_100_pairs' if n==100 else 'pilot_one_pair_no_MCSE' if n==1 else 'unavailable_or_partial_no_100rep_MCSE',important_structural_contrast=int(rr.scenario_index) in [26,38,39,40]))
 # Add explicit empty rows for pilot scenarios that have no score arrays.
 existing={(x['design'],x['scenario_id'],x['metric'],x['policy']) for x in contrasts}
 for _,rr in reg.iterrows():
  key=(rr.design,rr.scenario_id,'BalancedAccuracy','Task1C_PU_BA_pilot')
  if key not in existing:
   for a,b,typ in comparisons:contrasts.append(dict(design=rr.design,scenario_id=rr.scenario_id,scenario_index=rr.scenario_index,family=rr.family,family_id=rr.family_id,metric=key[2],policy=key[3],contrast_type=typ,method_a_number=a,method_b_number=b,method_a=name(a),method_b=name(b),best_fixed_nonboosting_comparator=b==best,direction='method_a_minus_method_b',n_pairs=0,n_expected=100,status='unavailable_no_pilot_scores',important_structural_contrast=int(rr.scenario_index) in [26,38,39,40]))
 contrasts=pd.DataFrame(contrasts)
 outputs={'fixed_method_scenario_results':scenario,'family_balanced_results':family,'fixed_method_ranks':ranks,'bayes_ceiling_results':bayes.reset_index(),'paired_mc_uncertainty':contrasts}
 for filename,df in outputs.items():
  df.to_pickle(H/(filename+'.staging.pkl'))
  def clean(v):
   if isinstance(v,np.generic):v=v.item()
   if isinstance(v,float) and not math.isfinite(v):return None
   return v
  (H/(filename+'.export.json')).write_text(json.dumps(dict(columns=list(df.columns),values=[[clean(v) for v in row] for row in df.itertuples(index=False,name=None)]),allow_nan=False,separators=(',',':')))
 validation=dict(raw_rows=len(raw),unique_repetitions_each_method_scenario=100,primary_BA_available=int(ba.value.notna().sum()),primary_PU_BA_available=int(ba[(ba.method_number>=3)].value.notna().sum()),TM003_recovered_without_fitting=len(recovered),remaining_TM003_missing=4000-len(recovered),Task1C_pilot_rows=len(pilot),model_fits=0,output_rows={k:len(v) for k,v in outputs.items()},best_fixed_nonboosting=selection)
 (H/'analysis_validation.json').write_text(json.dumps(validation,indent=2));print(json.dumps(validation,indent=2))
if __name__=='__main__':main()
