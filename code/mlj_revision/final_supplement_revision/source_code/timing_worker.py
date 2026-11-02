"""Separate timing study. Frozen fitting functions; no predictive result writes."""
from pathlib import Path
import os,sys,time,json,importlib,importlib.metadata,inspect,ast,copy,contextlib,hashlib,traceback,socket,subprocess
H=Path(__file__).resolve().parents[1];R=H.parents[1];T=R/'mlj_revision/07_nonlinear_simulations'
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','PU_BENCHMARK_N_JOBS']:os.environ[k]='1'
dep=R/'phase3_confirmatory_S001_S040_100reps/Ada-PU/src'
os.environ['ADAPU_SRC']=str(dep)
sys.path[:0]=[str(dep),str(dep.parent)]
def save(p,d):
 with p.open('x') as f:json.dump(d,f,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x))
def run(i):
 import numpy as np,torch
 from threadpoolctl import threadpool_limits,threadpool_info
 design=['one_sample','two_sample'][i//10];p=[10,500][i%10//5];rep=i%5
 dest=H/'timing/results'/f'{design}_p{p}_r{rep}';dest.mkdir(parents=True,exist_ok=True)
 os.environ['ADAPU_CACHE_DIR']=str(dest/'cache')
 sys.path.insert(0,str(T/'sources'/design))
 mod=importlib.import_module('confirmatory_benchmark' if design=='one_sample' else 'two_sample_benchmark')
 base=mod.base;repair=mod.repaired
 sys.path.insert(0,str(T/'sources'));import scoring_harmonization as harm
 assert base.ADAPU_AVAILABLE and repair.ADAPU_AVAILABLE
 versions={k:importlib.metadata.version(k) for k in ['numpy','scipy','pandas','scikit-learn','torch','catboost','xgboost-cpu','lightgbm']}
 seed=2026102000+i
 with threadpool_limits(limits=1):
  torch.set_num_threads(1);torch.set_num_interop_threads(1)
  data=(mod.generate_structural_one_sample_scar(structural_type='linear_gaussian',p=p,seed=seed) if design=='one_sample' else mod.generate_two_sample_scar(p=p,seed=seed))
  fingerprint=base.experiment_fingerprint(data)
  # Warm-up each engine and linear/torch path once per worker; discarded.
  X=np.array([[-2.,0],[-1.,1],[0.,-1],[1.,0],[2.,1],[3.,-1]]); y=np.array([0,0,0,1,1,1])
  with (dest/'warmup.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
   base.LogisticRegression(max_iter=100).fit(X,y)
   for eng in ['AdaBoost','GBM','XGBoost','LightGBM','CatBoost']:
    pa=dict(base.ENGINE_GRIDS[eng][0])
    for k in ['n_estimators','iterations']:
     if k in pa:pa[k]=2
    base.make_engine(eng,pa,seed).fit(X,y)
   repair.AdaBoost_PU(n_estimators=2,prior=.2,random_selection=False,beta=0.).fit(X[y==1].astype(np.float32),X.astype(np.float32))
   w=torch.nn.Linear(2,1);opt=torch.optim.Adam(w.parameters());loss=w(torch.tensor(X,dtype=torch.float32)).square().mean();loss.backward();opt.step()
  env=dict(host=socket.gethostname(),cpu=next(v.split(':',1)[1].strip() for v in Path('/proc/cpuinfo').read_text().splitlines() if v.startswith('model name')),affinity=sorted(os.sched_getaffinity(0)),python=sys.version,packages=versions,threadpools=threadpool_info(),threads=1,external_workers=20,data_fingerprint=fingerprint,seed=seed)
  if not (dest/'environment.json').exists():save(dest/'environment.json',env)
  durations={};scores={};records=[]
  true_train=data.train.y_true.copy();true_val=data.val.y_true.copy()
  for n in list(range(1,11))+list(range(16,28))+list(range(11,16)):
   mid=f'M{n:03}'
   if (dest/(mid+'.json')).exists():
    prior=json.loads((dest/(mid+'.json')).read_text());durations[n]=prior['standalone_runtime_seconds']
    if 6<=n<=10: scores[n]=np.load(dest/(mid+'_q.npy'))
    continue
   data.train.y_true[:]=true_train if n<=2 else -1;data.val.y_true[:]=true_val if n<=2 else -1
   data.test.y_true[:]=-1
   count=[0];qfinal=[None];start=None
   def recorder(v):count[0]+=1
   def evaluate(method,y_test,score_test,threshold,probability=None,extra=None):
    qfinal[0]=np.asarray(score_test)
    return dict(extra=extra)
   base.evaluate_test=evaluate
   if 11<=n<=15:
    start=time.perf_counter()
    q=scores[n-5]
    if design=='one_sample':probs=base.clip_prob(q/data.c)
    else:
     # TS naive ranking is already the raw engine probability.
     probs=mod._probability_from_log_density_ratio(mod._case_control_log_density_ratio(q,float(np.r_[data.train.s,data.val.s].mean())),data.pi)
    elapsed=time.perf_counter()-start;standalone=durations[n-5]+elapsed;incremental=elapsed
    count[0]=records[[r['method_id'] for r in records].index(f'M{n-5:03}')]['candidate_count'] if f'M{n-5:03}' in [r['method_id'] for r in records] else None
    fn=None
   else:
    if n==1:fn=base.fit_oracle_logistic;args=(data,)
    elif n==2:fn=base.fit_oracle_catboost;args=(data,)
    elif n==3:fn=mod.fit_case_control_known_pi if design=='two_sample' else harm.fit_logistic_probability_correction;args=(data,) if design=='two_sample' else (base,data)
    elif n in [4,5]:fn=repair.fit_pu_logistic_baseline;args=(data,)
    elif n==26:fn=repair.fit_adapu_one_sample;args=(data,)
    elif n==27:fn=mod.fit_pu_gradient_boosting if design=='two_sample' else repair.fit_pu_gradient_boosting;args=(data,)
    else:
     fn=mod.fit_one_boosting_cell if design=='two_sample' else base.fit_one_boosting_cell
     args=(data,'Naive' if n<=10 else 'RN' if n<=20 else 'CostSensitive',['AdaBoost','GBM','XGBoost','LightGBM','CatBoost'][(n-6)%5])
    # Counting hooks only; unchanged scientific statements.
    sys.path.insert(0,str(T));from worker import instrument
    fit,ast_hash=instrument(fn,recorder);fit.__globals__['evaluate_test']=evaluate
    try:
     with (dest/(mid+'.log')).open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
      start=time.perf_counter()
      fit(*args,**({'non_negative':n==5} if n in [4,5] else {}))
      elapsed=time.perf_counter()-start
     standalone=elapsed;incremental=None
    except Exception:
     save(dest/(mid+'_failure.json'),dict(traceback=traceback.format_exc(),elapsed=time.perf_counter()-start));raise
    if 6<=n<=10:
     scores[n]=qfinal[0].copy();np.save(dest/(mid+'_q.npy'),scores[n])
   durations[n]=standalone
   r=dict(design=design,setting='reference' if p==10 else 'high_dimension',setting_description='Reference' if p==10 else 'High dimension (p=500)',rep=rep,method_id=mid,n_train=len(data.train.X),p=p,candidate_count=count[0],runtime_seconds=elapsed,standalone_runtime_seconds=standalone,incremental_runtime_seconds=incremental,threads=1,cpu=env['cpu'],python_version=sys.version.split()[0],package_versions=versions,timer_definition='Candidate search, validation selection, final refit and score preparation; excludes data generation, imports/warmup, test metric evaluation and file I/O.',data_fingerprint=fingerprint,seed=seed,host=env['host'],affinity=env['affinity'],source_function=fn.__name__ if fn else 'shared_naive_fit_plus_probability_correction',external_workers=20)
   save(dest/(mid+'.json'),r);records.append(r);print(i,mid,round(standalone,3),flush=True)
  save(dest/'complete.json',dict(methods=27,design=design,p=p,rep=rep))
if __name__=='__main__':run(int(sys.argv[1]))


