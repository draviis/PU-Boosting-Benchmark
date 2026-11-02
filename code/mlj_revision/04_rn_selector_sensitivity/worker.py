"""Task 4: fixed-parameter selector intervention; no hyperparameter search."""
from pathlib import Path
import argparse,hashlib,importlib,importlib.metadata,json,math,os,socket,sys,time,traceback,warnings
H=Path(__file__).resolve().parent;ROOT=H.parents[1];T3=ROOT/'mlj_revision/03_rn_mechanism'
sys.dont_write_bytecode=True
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def ah(a):return recovery.ah(a)
def clean(x):return recovery.clean(x)
def write_json(p,x):
 assert not p.exists(),p
 temp=p.with_name(p.name+'.writing')
 temp.write_text(json.dumps(clean(x),separators=(',',':'),allow_nan=False))
 temp.rename(p)
def save_npz(p,arrays):
 assert not p.exists(),p
 temp=p.with_name(p.name+'.writing.npz');np.savez_compressed(temp,**arrays);temp.rename(p)
def derived_seed(case_seed,stage,condition,draw):
 text=f'Task4|{case_seed}|{stage}|{condition}|{draw}'
 return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4],'little')
def stats(a):
 a=np.asarray(a,float)
 if not len(a):return dict(n=0,mean=None,SD=None,q10=None,q25=None,q50=None,q75=None,q90=None)
 return dict(n=len(a),mean=float(a.mean()),SD=float(a.std(ddof=1)) if len(a)>1 else None,**dict(zip(['q10','q25','q50','q75','q90'],map(float,np.quantile(a,[.1,.25,.5,.75,.9])))))
def size_match_indices(U,counts,mean_score,m):
 assert 0<=m<=len(U) and len(U)==len(counts)==len(mean_score)
 assert np.isfinite(mean_score).all()
 order=np.lexsort((U,mean_score,-np.asarray(counts,dtype=int)))
 return np.sort(U[order[:m]]).astype(np.int32)
def random_subset(U,m,seed):
 return np.sort(np.random.default_rng(seed).choice(U,size=m,replace=False)).astype(np.int32)
def quality(selected,s,y):
 U=np.flatnonzero(s==0);mark=np.zeros(len(s),bool);mark[selected]=True;uy=y[U]
 z=mark[U];neg=uy==0;pos=uy==1;m=len(selected);tn=int((z&neg).sum());fp=int((z&pos).sum());nP=int((s==1).sum())
 return dict(nP=nP,nU=len(U),selected_n=m,true_negatives_selected=tn,hidden_positives_selected=fp,hidden_negatives_U=int(neg.sum()),hidden_positives_U=int(pos.sum()),RN_precision=tn/m if m else None,RN_contamination=fp/m if m else None,RN_negative_recall=tn/neg.sum() if neg.any() else None,RN_positive_capture=fp/pos.sum() if pos.any() else None,RN_selection_fraction=m/len(U),final_training_n=nP+m,rn_training_positive_fraction=nP/(nP+m),empty_RN=m==0)
def overlap(a,b):
 inter=len(np.intersect1d(a,b));union=len(np.union1d(a,b))
 return dict(intersection_n=inter,union_n=union,jaccard=inter/union if union else 1.,logistic_in_other=inter/len(a) if len(a) else None,other_in_logistic=inter/len(b) if len(b) else None)
def load_data(c):
 base,ns,gen,bp,hashes=recovery.namespaces(c)
 if c['study']=='real':
  src=ROOT/c['data_remote']
  for name in ['manifest.json','data_audit_only.npz']:assert sha(src/name)==c['files'][name]
  with np.load(src/'data_audit_only.npz') as z:
   splits={n:base.PUSplit(X=z[n+'_X'].copy(),y_true=z[n+'_y'].copy(),s=z[n+'_s'].copy()) for n in ['train','val','test']}
  data=base.PUExperiment(**splits,pi=c['pi_tilde'],c=c['c'] or 0,p=c['p'],delta=np.nan,delta_sq=np.nan,signal=np.nan,n_signal=0,rho=np.nan,seed=c['seed'])
 else:data=gen(**c['generator'])
 assert base.experiment_fingerprint(data)==c['fingerprint'],('data fingerprint mismatch',c['key'])
 true={n:getattr(data,n).y_true.copy() for n in ['train','val','test']}
 for n in true:getattr(data,n).y_true[:]=-1
 # Load only original model constructor and scorer, never search/training pipelines.
 ns.update(CatBoostClassifier=CatBoostClassifier,XGBClassifier=XGBClassifier,LGBMClassifier=LGBMClassifier,N_JOBS=c['threads'])
 recovery.definitions(bp,['make_engine','model_score'],ns)
 return data,true,ns,bp,hashes

def nonlinear_select(X,s,c,params,ns,bp,selector,old_spies):
 captures=[];configs=[]
 class ReplacementSpy:
  def __init__(self,max_iter,random_state):
   assert max_iter==3000
   if selector=='CatBoost':
    cfg=dict(iterations=200,learning_rate=.05,depth=4,loss_function='Logloss',verbose=False,allow_writing_files=False,random_seed=random_state,thread_count=c['threads'])
    self.model=CatBoostClassifier(**cfg)
   else:
    assert selector=='XGBoost'
    cfg=dict(n_estimators=200,learning_rate=.05,max_depth=2,subsample=.8,colsample_bytree=.8,objective='binary:logistic',eval_metric='logloss',random_state=random_state,n_jobs=c['threads'])
    self.model=XGBClassifier(**cfg)
   configs.append(cfg)
  def fit(self,x,y):self.model.fit(x,y);return self
  def predict_proba(self,x):return self.model.predict_proba(x)
 def capture(b,spy_idx,spy_scores,unl_idx,unl_scores,selected_unl,threshold,rn_counts):
  captures.append(dict(b=b,spy_indices=spy_idx.copy(),spy_scores=spy_scores.copy(),U_scores=unl_scores.copy(),threshold=float(threshold),candidate_count=len(selected_unl),counts=rn_counts.copy()))
 func,ast_hash=recovery.instrument(bp,dict(ns,LogisticRegression=ReplacementSpy),capture)
 mask=func(X,s,spy_fraction=params['spy_fraction'],spy_quantile=params['spy_quantile'],n_repeats=10,selection_frequency=.70,seed=c['seed'])
 spies=np.stack([z['spy_indices'] for z in captures])
 np.testing.assert_array_equal(spies,old_spies)
 U=np.flatnonzero(s==0);scores=np.stack([z['U_scores'] for z in captures]);counts=captures[-1]['counts'][U]
 np.testing.assert_array_equal(counts,(scores<np.array([z['threshold'] for z in captures])[:,None]).sum(0))
 np.testing.assert_array_equal(mask[U],counts>=7)
 return dict(selected=np.flatnonzero(mask).astype(np.int32),counts=counts.astype(np.int8),mean_score=scores.mean(0),scores=scores,spies=spies,spy_scores=np.stack([z['spy_scores'] for z in captures]),thresholds=np.array([z['threshold'] for z in captures]),candidate_counts=np.array([z['candidate_count'] for z in captures]),configs=configs,original_selector_ast_sha256=ast_hash)

def fit_condition(c,j,ns,stage,X,s,Xval,sval,Xtest,selected,condition,draw,seed_draw,params,ty,out):
 name=stage+'__'+condition+f'__{draw:02d}'
 jp=out/(name+'.json');ap=out/(name+'.npz')
 if jp.exists():
  d=json.loads(jp.read_text());assert sha(ap)==d['arrays_sha256']
  assert d['selected_set_sha256']==ah(selected) and d['final_params']==params and d['final_seed']==c['seed']
  return d
 start=time.perf_counter()
 common=dict(stage=stage,condition=condition,draw=draw,draw_seed=seed_draw,selected_n=len(selected),selected_set_sha256=ah(selected),final_params=params,final_seed=c['seed'],selector=j['selector'],engine=j['engine'],source='Task4_new_fixed_parameter_fit',hyperparameter_searches=0,selection_uses_hidden_Y=condition=='Oracle_true_negative_size_matched',final_training_uses_hidden_Y=False)
 arr=dict(selected_indices=selected,y_test=ty)
 if not len(selected):
  d=common|dict(status='undefined_empty_RN',AUROC=None,AP=None,BA=None,threshold=None,fit_seconds=0)
 else:
  P=np.flatnonzero(s==1)
  # Original RN protocol orders all P first, then selected U in original index order.
  Xfit=np.vstack([X[P],X[selected]]);yfit=np.r_[np.ones(len(P),dtype=int),np.zeros(len(selected),dtype=int)]
  assert np.all(s[selected]==0) and np.all(selected[:-1]<selected[1:])
  model=ns['make_engine'](j['engine'],params,seed=c['seed'])
  with warnings.catch_warnings(record=True) as ws:
   warnings.simplefilter('always');model.fit(Xfit,yfit)
  pred=np.asarray(ns['model_score'](model,Xtest),float);arr['score_test']=pred
  threshold=None;ba=None;val_fraction=None;pop_n=None
  if stage=='selected_train_only':
   val=np.asarray(ns['model_score'](model,Xval),float);pop=np.ones(len(sval),bool) if c['design']=='one_sample' else sval==0
   assert pop.any()
   threshold=float(np.quantile(val[pop],1-c['pi_tilde']))
   arr.update(score_val=val,validation_population_mask=pop,test_prediction=pred>=threshold)
   ba=float(balanced_accuracy_score(ty,pred>=threshold));val_fraction=float((val[pop]>=threshold).mean());pop_n=int(pop.sum())
  d=common|dict(status='complete',AUROC=float(roc_auc_score(ty,pred)) if stage=='final_train_plus_validation' else None,AP=float(average_precision_score(ty,pred)) if stage=='final_train_plus_validation' else None,BA=ba,threshold=threshold,threshold_policy='Task1B_A_same_model' if stage=='selected_train_only' else 'not_applicable',threshold_population='complete_validation' if c['design']=='one_sample' else 'validation_U_only',threshold_population_n=pop_n,validation_positive_fraction=val_fraction,threshold_uses_test_labels=False,final_model_class=type(model).__name__,final_model_effective_params=model.get_params(),training_X_sha256=ah(Xfit),training_observed_labels_sha256=ah(yfit),fit_seconds=time.perf_counter()-start,warnings=sorted(set(str(w.message) for w in ws)))
 save_npz(ap,arr);d['arrays_path']=ap.name;d['arrays_sha256']=sha(ap);write_json(jp,d)
 return d

def run(c,j):
 start=time.perf_counter();out=H/'results'/j['job_key'];out.mkdir(parents=True,exist_ok=True)
 if (out/'complete.json').exists():
  d=json.loads((out/'complete.json').read_text());assert d['data_fingerprint']==c['fingerprint']
  print(j['job_key'],'already complete',flush=True);return
 data,true,ns,bp,hashes=load_data(c)
 method_id=next(mid for mid,m in c['methods'].items() if m['engine']==j['engine']);method=c['methods'][method_id];params=method['params']
 final_params={k:v for k,v in params.items() if k not in ['spy_fraction','spy_quantile']}
 oldj=T3/'results'/(c['key']+'.json');oldp=T3/'sets'/(c['key']+'.npz')
 assert sha(oldj)==c['task3_result_sha256'];assert sha(oldp)==c['task3_arrays_sha256']
 task3=json.loads(oldj.read_text());old=np.load(oldp)
 stages=[('final_train_plus_validation',np.vstack([data.train.X,data.val.X]),np.r_[data.train.s,data.val.s],np.r_[true['train'],true['val']])]
 if j['include_BA']:stages.append(('selected_train_only',data.train.X,data.train.s,true['train']))
 diagnostics=[];overlaps=[];performances=[];selection_records=[];dist=[];arrays={};runtime=[];log_checks=[]
 for stage,X,s,y in stages:
  U=np.flatnonzero(s==0);P=np.flatnonzero(s==1);prefix=stage+f"_f{int(params['spy_fraction']*100):02d}_q{int(params['spy_quantile']*100):02d}"
  oldrow=next(q for q in task3['rows'] if q['stage']==stage and q['spy_fraction']==params['spy_fraction'] and q['spy_quantile']==params['spy_quantile'])
  assert ah(X)==oldrow['X_sha256'] and ah(s)==oldrow['S_sha256']
  logistic=old[prefix+'_selected_indices'].copy();logq=old[prefix+'_mean_spy_score_U'].copy()
  assert ah(logistic)==oldrow['selected_indices_sha256']
  selection_file=out/(stage+'__nonlinear_selection.npz');selection_meta=out/(stage+'__nonlinear_selection.json')
  if selection_meta.exists():
   sm=json.loads(selection_meta.read_text());assert sha(selection_file)==sm['arrays_sha256']
   with np.load(selection_file) as z:nl={k:z[k].copy() for k in z.files}
   nl['configs']=sm['configs'];nl['original_selector_ast_sha256']=sm['original_selector_ast_sha256']
  else:
   t=time.perf_counter();nl=nonlinear_select(X,s,c,params,ns,bp,j['selector'],old[prefix+'_spy_indices'])
   save_npz(selection_file,{k:v for k,v in nl.items() if isinstance(v,np.ndarray)})
   sm=dict(configs=nl['configs'],original_selector_ast_sha256=nl['original_selector_ast_sha256'],arrays_sha256=sha(selection_file),seconds=time.perf_counter()-t,selection_uses_hidden_Y=False,source_data_fingerprint=c['fingerprint'],source_X_sha256=ah(X),source_S_sha256=ah(s),spy_fraction=params['spy_fraction'],spy_quantile=params['spy_quantile'],seed=c['seed'],selector=j['selector'])
   write_json(selection_meta,sm)
  runtime.append(dict(stage=stage,selector_seconds=sm['seconds']))
  m=len(logistic);matched=size_match_indices(U,nl['counts'],nl['mean_score'],m)
  assert len(matched)==m and np.isin(matched,U).all()
  conditions=[('RN_logistic',0,logistic,None,logq),('RN_nonlinear',0,nl['selected'],None,nl['mean_score']),('RN_nonlinear_size_matched',0,matched,None,nl['mean_score']),('All_U_RN_fixed_final',0,U.astype(np.int32),None,None)]
  # Practical random subsets use only U indices and a separately derived deterministic seed.
  for draw in range(1,j['random_draws']+1):
   seed=derived_seed(c['seed'],stage,'random',draw);chosen=random_subset(U,m,seed)
   conditions.append(('RN_random_size_matched',draw,chosen,seed,None))
  # The only selection branch allowed to inspect hidden Y is this named oracle diagnostic.
  if c['study']=='main':
   negatives=U[y[U]==0]
   for draw in range(1,j['oracle_draws']+1):
    seed=derived_seed(c['seed'],stage,'oracle',draw);chosen=np.sort(np.random.default_rng(seed).choice(negatives,size=min(m,len(negatives)),replace=False)).astype(np.int32)
    conditions.append(('Oracle_true_negative_size_matched',draw,chosen,seed,None))
  arrays[stage+'__U_indices']=U.astype(np.int32);arrays[stage+'__hidden_Y_U']=y[U].astype(np.int8);arrays[stage+'__logistic_mean_spy_score']=logq;arrays[stage+'__nonlinear_mean_spy_score']=nl['mean_score']
  lr=recovery.exact_llr(X[U],c['generator']) if c['study']=='main' else None
  if lr is not None:arrays[stage+'__exact_DGP_LR_U']=lr
  if c.get('scenario_number')==39:arrays[stage+'__interaction_X_U']=X[U,:3]
  for condition,draw,selected,seed_draw,ownscore in conditions:
   selected=np.asarray(selected,dtype=np.int32)
   mark=np.isin(U,selected)
   common=dict(stage=stage,condition=condition,draw=draw,draw_seed=seed_draw,selected_set_sha256=ah(selected))
   qi=quality(selected,s,y)
   if lr is not None:
    neg=y[U]==0;hard=neg&(lr>=np.quantile(lr[neg],.75));easy=neg&(lr<=np.quantile(lr[neg],.25))
    qi.update(hard_negative_quartile_recall=float(mark[hard].mean()),easy_negative_quartile_recall=float(mark[easy].mean()))
   diagnostics.append(common|qi|dict(oracle_shortfall=condition=='Oracle_true_negative_size_matched' and len(selected)<m,selection_uses_hidden_Y=condition=='Oracle_true_negative_size_matched'))
   ov=common|overlap(logistic,selected);overlaps.append(ov)
   groups={'selected_true_negative':mark&(y[U]==0),'discarded_true_negative':~mark&(y[U]==0),'hidden_positive_U':y[U]==1,'full_true_negative':y[U]==0}
   scoretypes={'logistic_spy_reference':logq,'nonlinear_spy_reference':nl['mean_score']}
   if ownscore is not None:scoretypes['own_spy_score']=ownscore
   if lr is not None:scoretypes['exact_DGP_log_likelihood_ratio']=lr
   if c.get('scenario_number')==39:
    scoretypes.update(x0=X[U,0],x1=X[U,1],x2=X[U,2],x0_x1_product=X[U,0]*X[U,1])
   for group,mask in groups.items():
    for scoretype,score in scoretypes.items():dist.append(common|dict(group=group,score_type=scoretype)|stats(score[mask]))
   selection_records.append(common|dict(selected_n=len(selected),logistic_selected_n=m,size_matched=condition in ['RN_nonlinear_size_matched','RN_random_size_matched','Oracle_true_negative_size_matched'],indices_artifact=stage+'__'+condition+f'__{draw:02d}.npz',selector_information_only=condition!='Oracle_true_negative_size_matched'))
   fit=fit_condition(c,j,ns,stage,X,s,data.val.X,data.val.s,data.test.X,selected,condition,draw,seed_draw,final_params,true['test'],out)
   performances.append(fit)
   if condition=='RN_logistic':
    archive=method['archive'];metrics=['AUROC','AP'] if stage=='final_train_plus_validation' else ['BA']
    diffs={k:fit[k]-archive['RN_'+k] for k in metrics if archive.get('RN_'+k) is not None}
    # Same tolerance used in prior validated benchmark reconstruction; preserve actual deviations.
    assert all(abs(v)<=5e-5 for v in diffs.values()),('archived logistic performance mismatch',c['key'],diffs)
    log_checks.append(dict(stage=stage,archived_metric_differences=diffs,tolerance=5e-5,passed=True))
   print(j['job_key'],stage,condition,draw,'done',round(fit.get('fit_seconds',0),2),'s',flush=True)
  # Archived naive benchmark is immutable. It may use different selected final parameters.
  ar=method['archive'];naive_params=json.loads(ar['naive_selected_params'])
  performances.append(dict(stage=stage,condition='Naive_all_U',draw=0,source='archived_original_naive',status='complete' if stage=='final_train_plus_validation' or ar.get('naive_BA') is not None else 'missing_archived_same_model_BA',AUROC=ar['naive_AUROC'] if stage=='final_train_plus_validation' else None,AP=ar['naive_AP'] if stage=='final_train_plus_validation' else None,BA=ar.get('naive_BA') if stage=='selected_train_only' else None,final_params=naive_params,final_seed=c['seed'],different_params_from_RN=naive_params!=final_params,selected_n=len(U),threshold_policy='Task1B_A_same_model' if stage=='selected_train_only' else 'not_applicable',performance_source=ar['performance_source'],predictions_source=ar.get('saved_prediction_path'),new_fit=False))
  qi=quality(U.astype(np.int32),s,y);diagnostics.append(dict(stage=stage,condition='Naive_all_U',draw=0,draw_seed=None,selected_set_sha256=ah(U.astype(np.int32)),**qi,selection_uses_hidden_Y=False))
 arrays_path=out/'diagnostic_arrays.npz'
 if arrays_path.exists():pass
 else:save_npz(arrays_path,arrays)
 summary=dict(job_key=j['job_key'],case_key=c['key'],scope=j['scope'],study=c['study'],design=c['design'],scenario_id=c['scenario_id'],scenario_number=c.get('scenario_number'),dataset=c.get('dataset'),budget=c.get('budget'),rep=c['rep'],seed=c['seed'],engine=j['engine'],selector=j['selector'],data_fingerprint=c['fingerprint'],method_id=method_id,original_RN_params=params,final_params=final_params,original_naive_params=json.loads(method['archive']['naive_selected_params']),source_code_hashes=hashes,task3_arrays_sha256=c['task3_arrays_sha256'],task3_record_sha256=c['task3_result_sha256'],selection_records=selection_records,quality=diagnostics,overlap=overlaps,performance=performances,distributions=dist,logistic_archive_checks=log_checks,selector_runtime=runtime,diagnostic_arrays_sha256=sha(arrays_path),worker_sha256=sha(Path(__file__)),packages=versions,threads=c['threads'],hostname=socket.gethostname(),seconds=time.perf_counter()-start,hyperparameter_searches=0,hidden_Y_practical_selection=False,hidden_Y_oracle_selection=True if c['study']=='main' and j['oracle_draws'] else False,fixed_final_aliases={x+'_fixed_final':x for x in ['RN_logistic','RN_nonlinear','RN_nonlinear_size_matched']})
 write_json(out/'complete.json',summary)
 print(j['job_key'],'COMPLETE',round(summary['seconds'],2),'s',flush=True)
def main():
 global np,recovery,CatBoostClassifier,XGBClassifier,LGBMClassifier,roc_auc_score,average_precision_score,balanced_accuracy_score,versions
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',default='pilot_jobs.json');ap.add_argument('--task',type=int,required=True);args=ap.parse_args()
 assert Path(args.manifest).name==args.manifest
 j=json.loads((H/args.manifest).read_text())[args.task];c=json.loads((H/'cases.json').read_text())[j['case_index']]
 for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','PU_BENCHMARK_N_JOBS']:os.environ[key]=str(c['threads'])
 import numpy as np
 from catboost import CatBoostClassifier
 from xgboost import XGBClassifier
 from lightgbm import LGBMClassifier
 from sklearn.linear_model import LogisticRegression
 from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score
 from scipy.special import logsumexp
 from threadpoolctl import threadpool_limits
 import task3_recovery as recovery
 recovery.np=np;recovery.LogisticRegression=LogisticRegression;recovery.logsumexp=logsumexp
 expected={'numpy':'2.2.6','scipy':'1.15.3','scikit-learn':'1.7.2','catboost':'1.2.10','xgboost-cpu':'3.2.0','lightgbm':'4.7.0'}
 versions={k:importlib.metadata.version(k) for k in expected};assert versions==expected
 try:
  with threadpool_limits(limits=c['threads']):run(c,j)
 except Exception as e:
  err=H/'errors';err.mkdir(exist_ok=True);p=err/(j['job_key']+'.json')
  if not p.exists():write_json(p,dict(job=j,case=c['key'],error=repr(e),traceback=traceback.format_exc(),worker_sha256=sha(Path(__file__))))
  raise
if __name__=='__main__':main()
