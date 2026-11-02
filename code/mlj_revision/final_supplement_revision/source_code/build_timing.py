"""Collect only completed standardized timing records; never launch a model."""
import json,hashlib,sys
from pathlib import Path
import pandas as pd,numpy as np
from labels import ROOT,HOME
sys.path.insert(0,str(ROOT/'mlj_revision/display_package'))
from display_specs import METHODS
METHODS.update({1:'Supervised logistic regression',2:'Supervised CatBoost'})
from build_tables import save
H=HOME;T=H/'timing'
def group(n):
 if n<=2:return 'Supervised references'
 if n<=5:return 'Nonboosting PU'
 if n<=10:return 'Naive boosting'
 if n<=15:return 'Probability-corrected boosting'
 if n<=20:return 'Reliable-negative boosting'
 if n<=25:return 'Cost-sensitive boosting'
 return 'Native PU boosting'
def main():
 complete=json.loads((T/'cluster_records/completion.json').read_text())
 assert len(complete)==20 and all(c==0 for i,c in complete),complete
 base=T/'cluster_records/results';files=sorted(base.glob('*/M[0-9][0-9][0-9].json'))
 assert len(files)==540,len(files)
 assert not list(base.glob('*/*failure*.json'))
 rows=[]
 for path in files:
  r=json.loads(path.read_text());n=int(r['method_id'][1:]);r['method_name']=METHODS[n];r['method_group']=group(n)
  r['archived_timer_definition']=r['timer_definition']
  r['timer_definition']='Full frozen candidate search, validation selection, final refit and internal score/threshold preparation. Excludes data generation, imports/warmup, test metric computation, result/prediction CSV/NPZ writing and plotting; includes redirected fitting-diagnostic logging.'
  r['timing_source']=str(path.relative_to(H));r['timing_source_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
  r['package_versions']=json.dumps(r['package_versions'],sort_keys=True);r['affinity']=json.dumps(r['affinity'])
  rows.append(r)
 d=pd.DataFrame(rows)
 assert not d.duplicated(['design','setting','rep','method_id']).any()
 assert d.groupby(['design','setting','method_id']).size().eq(5).all()
 assert d.groupby(['design','setting','rep']).data_fingerprint.nunique().eq(1).all()
 assert all(d[c].nunique()==1 for c in ['cpu','threads','python_version','package_versions','host','external_workers'])
 assert np.isfinite(d[['runtime_seconds','standalone_runtime_seconds']]).all().all() and d.runtime_seconds.gt(0).all()
 assert d.threads.eq(1).all() and d.external_workers.eq(20).all()
 assert d.candidate_count.notna().all() and d.candidate_count.gt(0).all()
 alloc=json.loads((T/'cluster_records/allocation_manifest.json').read_text())
 assert len(set(alloc['cores']))==20
 local_sources=pd.read_csv(T/'frozen_fitting_source_hashes.csv').set_index('path').sha256
 remote_sources=pd.read_csv(T/'cluster_records/frozen_fitting_source_hashes_remote.csv').set_index('path').sha256
 assert local_sources.equals(remote_sources.reindex(local_sources.index))
 native=pd.read_csv(T/'native_adapu_source_hashes.csv')
 remote_native={line.split(None,1)[1].strip().split('/PUBenchmark/')[-1]:line.split()[0] for line in (T/'native_adapu_source_hashes_remote.txt').read_text(encoding='utf-8-sig').splitlines()}
 assert all(remote_native[r.path]==r.sha256 for r in native.itertuples())
 assert alloc['worker_sha256']==hashlib.sha256((H/'source_code/timing_worker.py').read_bytes()).hexdigest()
 envs=[json.loads(p.read_text()) for p in base.glob('*/environment.json')]
 assert len(envs)==20 and all(len(e['affinity'])==1 for e in envs)
 assert all(p['num_threads']==1 for e in envs for p in e['threadpools'])
 # Verify shared-fit charging exactly for every case.
 for _,g in d.groupby(['design','setting','rep']):
  v=g.set_index('method_id')
  for n in range(11,16):
   a=v.loc[f'M{n:03}'];b=v.loc[f'M{n-5:03}']
   assert a.runtime_seconds==a.incremental_runtime_seconds
   assert abs(a.standalone_runtime_seconds-b.standalone_runtime_seconds-a.incremental_runtime_seconds)<1e-10
 required=['design','setting','setting_description','rep','method_id','method_name','method_group','n_train','p','candidate_count','runtime_seconds','standalone_runtime_seconds','incremental_runtime_seconds','threads','cpu','python_version','package_versions','timer_definition']
 d=d[required+[c for c in d if c not in required]].sort_values(['design','setting','rep','method_id'])
 d.to_csv(T/'computational_time_raw.csv',index=False,float_format='%.17g')
 summary=[]
 for (des,setting,mid),g in d.groupby(['design','setting','method_id'],sort=False):
  q=np.quantile(g.standalone_runtime_seconds,[.25,.5,.75])
  iq=np.quantile(g.incremental_runtime_seconds.dropna(),[.25,.5,.75]) if g.incremental_runtime_seconds.notna().any() else [np.nan]*3
  summary.append(dict(design=des,setting=setting,method_id=mid,method_name=g.method_name.iloc[0],method_group=g.method_group.iloc[0],n=5,median_seconds=q[1],q1_seconds=q[0],q3_seconds=q[2],incremental_median_seconds=iq[1],incremental_q1_seconds=iq[0],incremental_q3_seconds=iq[2],candidate_count_min=g.candidate_count.min(),candidate_count_max=g.candidate_count.max()))
 s=pd.DataFrame(summary);s.to_csv(T/'computational_time_summary_full_precision.csv',index=False,float_format='%.17g')
 s[s.incremental_median_seconds.notna()].to_csv(T/'incremental_correction_summary.csv',index=False,float_format='%.17g')
 order=list(range(3,28))+[1,2];cells=[('one_sample','reference','Reference OS'),('two_sample','reference','Reference TS'),('one_sample','high_dimension','$p=500$ OS'),('two_sample','high_dimension','$p=500$ TS')]
 rows=[]
 for n in order:
  r={'Panel':'Standalone pipeline time (reference: s; high dimension: min): '+group(n),'Method':METHODS[n]}
  for des,setting,col in cells:
   q=s[(s.design==des)&(s.setting==setting)&(s.method_id==f'M{n:03}')].iloc[0]
   vals=np.array([q.median_seconds,q.q1_seconds,q.q3_seconds])/(60 if setting=='high_dimension' else 1)
   def fmt(x):return f'{x:.3g}' if x<1 else f'{x:.2f}' if setting=='high_dimension' else f'{x:.1f}'
   r[col]=fmt(vals[0])+' ['+fmt(vals[1])+', '+fmt(vals[2])+']'
  rows.append(r)
 for n in range(11,16):
  r={'Panel':'Incremental correction only (milliseconds)','Method':METHODS[n]}
  for des,setting,col in cells:
   q=s[(s.design==des)&(s.setting==setting)&(s.method_id==f'M{n:03}')].iloc[0]
   r[col]=f'{1000*q.incremental_median_seconds:.3f} [{1000*q.incremental_q1_seconds:.3f}, {1000*q.incremental_q3_seconds:.3f}]'
  rows.append(r)
 caption='Standardized computational time for the core PU configurations. Entries are median end-to-end fitting times with interquartile ranges over five fixed data realizations. Timing includes the candidate hyperparameter search, validation-based model selection, and final refit, but excludes data generation and metric computation. All methods were timed on the same hardware with controlled thread counts. The timing experiment is separate from the predictive benchmark and does not alter any reported performance result.'
 note='OS: one-sample; TS: two-sample. Brackets give the 25th and 75th percentiles. Standalone reference times are seconds and high-dimensional times are minutes; incremental corrections in the final panel are milliseconds. Raw times are always retained in seconds. CPU: '+d.cpu.iloc[0]+'. Python '+d.python_version.iloc[0]+'. One library thread per worker; at most 20 case workers pinned to distinct physical cores on one exclusive node. Active concurrency decreases as cases finish. All methods share the same data within a case. Warm-up is discarded. Timers include internal scoring/threshold preparation and redirected fitting diagnostics. Corrected pipelines reuse the matching naive fit: standalone time adds the measured correction; the fit is not charged twice to the shared benchmark. Candidate counts and software versions are in the raw CSV. Candidate counts are validation candidates/checkpoints from the frozen selection hooks, not counts of separate trajectories; AdaPU can evaluate several stages from each beta trajectory. The two supervised references are separate comparators. The exploratory neural extension uses a distinct protocol and is excluded.'
 save('tab_s_computational_time',pd.DataFrame(rows),caption,note)
 counts=d.groupby(['method_id','method_name','method_group']).candidate_count.agg(['min','max']).reset_index();counts.to_csv(T/'candidate_counts.csv',index=False)
 result=dict(completed_cases=20,completed_method_records=540,failures=0,cpu=d.cpu.iloc[0],python=d.python_version.iloc[0],packages=json.loads(d.package_versions.iloc[0]),node=d.host.iloc[0],library_threads=1,maximum_external_workers=20,all_same_case_fingerprints=True,all_shared_fit_charges_verified=True,warmup_discarded=True,raw_repetitions=5)
 (T/'TIMING_VALIDATION.json').write_text(json.dumps(result,indent=2),encoding='utf8')
 (T/'STANDARDIZED_TIMING_REPORT.md').write_text('# Standardized timing study\n\nCompleted 20 data cases and 540 configuration records: reference and 500 predictors, both designs, five fixed realizations, 25 PU configurations plus two supervised references. No timing case failed. No predictive result was replaced.\n\n'+note+'\n\nThe worker metadata originally abbreviated the exclusion as "file I/O". The raw-table timer_definition makes the actual scope explicit: result and prediction files are outside the timer, while redirected fitting diagnostics may add logging cost. The original wording is retained in archived_timer_definition; no time is subtracted or estimated.\n\nMethods ran in the frozen order recorded in the worker, not a randomized order. These are descriptive wall-clock timings on a shared-memory node, not hardware-independent complexity measures or single-process isolation estimates. Five repetitions give limited information on timing variability. Quartiles use the linear empirical-quantile convention. Candidate counts refer to evaluated validation candidates/checkpoints, not independent fitted trajectories. The timing uses generated reference/high-dimensional data with fixed seeds 2026102000 through 2026102019; all methods within a case share its fingerprint. There were 440 complete fitting pipelines and 100 incremental corrections.\n\nSoftware versions:\n\n'+ '\n'.join('- '+k+': '+str(v) for k,v in result['packages'].items())+'\n',encoding='utf8')
 print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
