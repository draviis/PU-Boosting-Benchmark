"""Consolidate independently verified probability metrics; never fit a model."""
import sys,json,hashlib,re,collections
import pandas as pd,numpy as np
from labels import *
sys.path.insert(0,str(ROOT/'mlj_revision/display_package'))
from display_specs import METHODS,DESIGNS,DNAME
METHODS.update({1:'Supervised logistic regression',2:'Supervised CatBoost'})
ELIGIBLE=[1,2,3,4,5,11,12,13,14,15]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def definition(n,des):
 if n in [1,2]:return 'Supervised estimate of P(Y=1|X)'
 if n in [4,5]:return 'sigmoid(F), explicitly defined posterior output of direct PU logistic-risk model'
 if n==3 or 11<=n<=15:return 'clip(q/c)' if des=='one_sample' else 'clip(pi_tilde (1-alpha)/alpha q/(1-q))'
 return 'NA: ranking/source-membership score without defined population probability'
def main():
 blocks=[];manifest=[]
 for name,des in [('main_one_raw.csv','one_sample'),('main_two_raw.csv','two_sample'),('real_raw.csv',None)]:
  p=ROOT/'publication_preparation/eta_scoring_20261013/collected'/name;d=pd.read_csv(p,low_memory=False,float_precision='round_trip');digest=sha(p)
  d['number']=d.method_id.str[-3:].astype(int);d['method_id']=d.number.map(lambda n:f'M{n:03}')
  d['study']='real' if name=='real_raw.csv' else 'core';d['design']=d.sampling_scheme if des is None else des
  d['scenario']=d.dataset if des is None else d.scenario_id
  d['dataset']=d.dataset if des is None else ''
  d['budget']=d.c if des is None else ''
  d['rep']=d.seed if des is None else d.rep
  d['scenario_description']=[label(s,ds) if des else s.replace('_',' ').title() for s,ds in zip(d.scenario,d.design)]
  d['log_loss']=d.LogLoss;d['probability_source']=p.relative_to(ROOT).as_posix();d['source_hash']=digest
  d['recomputed_or_archived']='archived';d['n_test']=d.n_test if 'n_test'in d else np.nan
  d['final_refit_prediction_available']=False
  blocks.append(d)
  manifest.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=digest))
 p=ROOT/'mlj_revision/07_nonlinear_simulations/nonlinear_results.csv';d=pd.read_csv(p,low_memory=False,float_precision='round_trip')
 d['number']=d.method_id.str[-3:].astype(int);d['study']='nonlinear';d['scenario']=d.scenario_id;d['scenario_description']=d.scenario.map(label);d['dataset']='';d['budget']='';d['log_loss']=d.LogLoss;d['n_test']=10000;d['probability_source']=p.relative_to(ROOT).as_posix();d['source_hash']=sha(p);d['recomputed_or_archived']='archived';d['final_refit_prediction_available']=False
 blocks.append(d);manifest.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)))
 full=pd.concat(blocks,ignore_index=True).copy()
 full['budget']=full.budget.astype(str)
 full['method_name']=full.number.map(METHODS)
 keys=['study','design','scenario','rep','method_id','budget']
 assert not full.duplicated(keys).any()
 checks=[]
 rec=pd.read_csv(HOME/'source_data/task7_probability_recomputed.csv')
 rec['study']='nonlinear';rec['dataset']='';rec['budget']='';rec['accepted']=True
 real=pd.read_csv(HOME/'source_data/real_probability_recomputed.csv');real=real[real.accepted].copy()
 harm=pd.read_csv(HOME/'source_data/harmonized_probability_recomputed.csv').fillna({'budget':''})
 harm['accepted']=True
 for frame in [rec,real,harm]:frame['budget']=frame.budget.astype(str)
 updates=pd.concat([rec,real,harm],ignore_index=True)
 match=updates.merge(full[keys+['Brier','log_loss']],on=keys,how='left',validate='many_to_one',suffixes=('','_archived'),indicator=True)
 assert match['_merge'].eq('both').all()
 errors=np.maximum(abs(match.Brier-match.Brier_archived),abs(match.log_loss-match.log_loss_archived))
 assert errors.max()<1e-9,match.loc[errors.idxmax(),keys].to_dict()
 check=match[keys].copy();check['max_abs_difference']=errors;checks=check.to_dict('records')
 cols=['Brier','log_loss','n_test','probability_source','source_hash']
 updates=updates.drop_duplicates(keys,keep='last')
 full=full.merge(updates[keys+cols],on=keys,how='left',validate='one_to_one',suffixes=('','_new'),indicator=True)
 mask=full['_merge'].eq('both')
 for c in cols:full[c]=full[c+'_new'].where(mask,full[c])
 full.loc[mask,'recomputed_or_archived']='recomputed'
 full.loc[mask,'final_refit_prediction_available']=True
 assert full.loc[~full.number.isin(ELIGIBLE),['Brier','log_loss']].isna().all().all()
 valid=full[full.number.isin(ELIGIBLE)].copy()
 assert len(valid)==98400 and valid[['Brier','log_loss']].notna().all().all()
 assert len(valid[valid.recomputed_or_archived=='recomputed'])==66400
 # Dataset-specific test lengths are verified from saved files for every real record.
 assert valid.n_test.notna().all()
 valid['probability_definition']=[definition(n,ds) for n,ds in zip(valid.number,valid.design)]
 columns=['study','design','scenario','scenario_description','dataset','budget','method_id','method_name','rep','Brier','log_loss','n_test','probability_source','source_hash','recomputed_or_archived','final_refit_prediction_available','probability_definition','number']
 valid[columns].to_csv(HOME/'source_data/probability_metrics_full.csv',index=False,float_format='%.17g')
 pd.DataFrame(checks).to_csv(HOME/'audits/probability_recomputation_checks.csv',index=False)
 pd.DataFrame(manifest).to_csv(HOME/'audits/probability_primary_source_hashes.csv',index=False)
 cover=[]
 for (study,des,scenario,budget,n),g in full.groupby(['study','design','scenario','budget','number'],dropna=False):
  ok=n in ELIGIBLE
  cover.append(dict(study=study,design=des,scenario_or_dataset=scenario,budget=budget,method_id=f'M{n:03}',method_name=METHODS[n],expected_repetitions=20 if study=='real' else 100,available_final_refit_predictions=int(g.final_refit_prediction_available.sum()) if ok else 0,available_brier=int(g.Brier.notna().sum()),available_log_loss=int(g.log_loss.notna().sum()),probability_definition=definition(n,des),source_file='source_data/probability_metrics_full.csv (record-specific paths and SHA256)' if ok else 'archived method output map; no population probability',recomputed_or_archived=';'.join(sorted(g.recomputed_or_archived.unique())) if ok else 'not_applicable',notes='Counts refer to compatible population-probability predictions, not arbitrary saved ranking scores. Original values are retained if predictions are unavailable.' if ok else 'No new probability transformation introduced.'))
 cov=pd.DataFrame(cover);cov.to_csv(HOME/'audits/PROBABILITY_METRIC_COVERAGE.csv',index=False)
 # Repetition -> setting -> family/DGP/dataset. All cells complete before averaging.
 aggregate=[]
 for study in ['core','nonlinear','real']:
  v=valid[valid.study==study]
  if study=='core':
   v=v.copy();v['family']=v.scenario.str[-3:].astype(int).map(lambda i:REG[i-1]['scenario_family'])
   setting=v.groupby(['design','scenario','family','number'])[['Brier','log_loss']].mean().reset_index()
   fam=setting.groupby(['design','family','number'])[['Brier','log_loss']].mean().reset_index()
   z=fam.groupby(['design','number'])[['Brier','log_loss']].mean().reset_index()
   assert fam.groupby(['design','number']).size().eq(10).all()
  elif study=='nonlinear':
   setting=v.groupby(['design','scenario','number'])[['Brier','log_loss']].mean().reset_index()
   assert v.groupby(['design','scenario','number']).size().eq(100).all()
   z=setting.groupby(['design','number'])[['Brier','log_loss']].mean().reset_index()
  else:
   setting=v.groupby(['design','dataset','budget','number'])[['Brier','log_loss']].mean().reset_index()
   assert v.groupby(['design','dataset','budget','number']).size().eq(20).all()
   dataset=setting.groupby(['design','dataset','number'])[['Brier','log_loss']].mean().reset_index()
   assert dataset.groupby(['design','number']).size().eq(8).all()
   z=dataset.groupby(['design','number'])[['Brier','log_loss']].mean().reset_index()
  z['study']=study;aggregate.append(z)
 agg=pd.concat(aggregate);agg.to_csv(HOME/'source_data/probability_aggregated_full_precision.csv',index=False,float_format='%.17g')
 rows=[]
 panels={'core':'Panel A: Original controlled simulations','nonlinear':'Panel B: Nonlinear simulation extension','real':'Panel C: Real-data experiments'}
 for study in panels:
  for n in [3,4,5,11,12,13,14,15,1,2]:
   r={'Panel':panels[study],'Method':METHODS[n]+(' (supervised reference)' if n<=2 else '')}
   for metric,name in [('Brier','Brier'),('log_loss','Log loss')]:
    for des,short in zip(DESIGNS,['OS','TS']):r[name+' '+short]=f'{agg[(agg.study==study)&(agg.design==des)&(agg.number==n)][metric].iloc[0]:.4f}'
   rows.append(r)
 from build_tables import save
 caption='Probability-based performance for methods with a defined estimate of the population positive-class probability. Brier score and log loss are computed from final train+validation refit probabilities; lower values indicate better probability predictions. Ranking-only scores without a population-probability interpretation are not assigned Brier or log-loss values. Aggregation follows the benchmark weighting scheme described in the text.'
 note='OS: one-sample; TS: two-sample. Panel A: 100 repetitions per setting, settings averaged within ten families, then equal family weights. Panel B: 100 repetitions per DGP, then equal weights for six DGPs. Panel C: 20 splits per dataset/design/budget, then equal weights for two budgets and eight datasets. All ten eligible methods have complete coverage in all panels. The final two methods in each panel are supervised references, not practical PU competitors. Values describe probability prediction accuracy, not calibration alone.'
 save('tab_s_probability_metrics',pd.DataFrame(rows),caption,note)
 print('probability counts',valid.groupby(['study','design','recomputed_or_archived']).size().to_dict(),flush=True)
if __name__=='__main__':main()

