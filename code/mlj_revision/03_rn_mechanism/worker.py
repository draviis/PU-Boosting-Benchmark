"""Replay ONLY original selected spy RN selection. No boosting/tuning functions loaded."""
from pathlib import Path
import argparse,ast,copy,dataclasses,hashlib,importlib.metadata,json,math,os,sys,time,traceback,types,warnings
H=Path(__file__).resolve().parent;R=H.parents[1];sys.dont_write_bytecode=True
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ah(a):
 a=np.ascontiguousarray(a);h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes());return h.hexdigest()
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.generic):x=x.item()
 if isinstance(x,float) and not math.isfinite(x):return None
 return x
def definitions(path,names,ns):
 tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
 assert {n.name for n in nodes}==set(names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
 return nodes
def namespaces(c):
 folder=R/c['frozen_remote'];hashes={p.name:sha(p) for p in folder.glob('*.py')}
 for name,digest in c['frozen_hashes'].items():assert hashes[name]==digest,('frozen source mismatch',name)
 ns=dict(np=np,math=math,hashlib=hashlib,dataclass=dataclasses.dataclass,LogisticRegression=LogisticRegression)
 bp=folder/'base_phase1_benchmark.py'
 definitions(bp,['PUSplit','PUExperiment','experiment_fingerprint','toeplitz_covariance','calibrate_mean_vector','generate_split','generate_one_sample_scar','select_reliable_negatives_spy'],ns)
 base=types.SimpleNamespace(**ns)
 structures={'linear_gaussian','curved_boundary','multiplicative_interaction','unequal_covariance','many_irrelevant_variables'}
 dn=dict(np=np,math=math,base=base,Callable=__import__('typing').Callable,STRUCTURAL_TYPES=structures,logsumexp=logsumexp)
 if c['design']=='one_sample':
  dn['_ORIGINAL_GENERATOR']=base.generate_one_sample_scar
  definitions(folder/'confirmatory_benchmark.py',['_validate_common','_latent_equal_covariance_split','_unequal_covariance_split','_mixed_interaction_split','generate_structural_one_sample_scar'],dn)
  gen=dn['generate_structural_one_sample_scar']
 else:
  dn.update(SAMPLING_SCHEME='two_sample_case_control_scar',DGP_IMPLEMENTATION='deterministic_toeplitz_ar1_v1',_ACTIVE_VALIDATION_U_MASK=None)
  definitions(folder/'two_sample_benchmark.py',['_validate_design','_toeplitz_precision_times','_calibrate_deterministic_toeplitz_mean','_deterministic_toeplitz_standard_normal','_linear_conditional_sample','_curved_conditional_sample','_interaction_conditional_sample','_unequal_conditional_sample','generate_two_sample_scar'],dn)
  gen=dn['generate_two_sample_scar']
 return base,ns,gen,bp,hashes
def instrument(bp,ns,capture):
 tree=ast.parse(bp.read_text());node=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='select_reliable_negatives_spy'))
 original_dump=ast.dump(node,include_attributes=False)
 loop=next(n for n in node.body if isinstance(n,ast.For))
 assert isinstance(loop.body[-1],ast.AugAssign)
 hook=ast.parse('_capture(b, spy_idx, spy_scores, unl_idx, unl_scores, selected_unl, threshold, rn_counts)').body[0]
 loop.body.append(hook);instrumented=copy.deepcopy(node);loop.body.pop()
 assert ast.dump(node,include_attributes=False)==original_dump
 env=dict(ns,_capture=capture);exec(compile(ast.fix_missing_locations(ast.Module(body=[instrumented],type_ignores=[])),str(bp),'exec'),env)
 return env['select_reliable_negatives_spy'],hashlib.sha256(original_dump.encode()).hexdigest()
def exact_llr(X,g):
 kind=g['structural_type'];d=g['delta']
 if kind in ['linear_gaussian','many_irrelevant_variables']:
  p=g['p'];rho=g['rho'];S=rho**np.abs(np.subtract.outer(np.arange(p),np.arange(p)));direction=np.zeros(p);direction[:g['n_signal']]=1;v=np.linalg.solve(S,direction);mu=d*direction/np.sqrt(direction@v);return X@np.linalg.solve(S,mu)-d*d/2
 if kind=='curved_boundary':
  z=X.copy();z[:,1]-=g['transform_strength']*(X[:,0]**2-1);return z[:,1:4].sum(1)*(d/math.sqrt(3))-d*d/2
 if kind=='multiplicative_interaction':
  a=g['transform_strength'];b=g['main_effect_shift'];u=a*(X[:,0]+X[:,1]);v=a*(X[:,0]-X[:,1]);return np.logaddexp(u,-u)-np.logaddexp(v,-v)+b*X[:,2]-b*b/2
 r=g['positive_variance_ratio'];mu=d/math.sqrt(3);return .5*((1-1/r)*(X[:,:3]**2).sum(1)+(2*mu/r)*X[:,:3].sum(1)-d*d/r-3*math.log(r))
def distribution(a,prefix):
 if not len(a):return {prefix+'_n':0}|{prefix+'_'+k:None for k in ['mean','q05','q25','q50','q75','q95']}
 q=np.quantile(a,[.05,.25,.5,.75,.95]);return {prefix+'_n':len(a),prefix+'_mean':float(a.mean())}|{prefix+'_'+k:float(v) for k,v in zip(['q05','q25','q50','q75','q95'],q)}
def run(c,verify=False):
 dest=H/'results'/(c['key']+'.json')
 if dest.exists():
  old=json.loads(dest.read_text());assert sha(H/'sets'/(c['key']+'.npz'))==old['arrays_sha256'];return
 start=time.perf_counter()
 base,ns,gen,bp,hashes=namespaces(c)
 source_rows={}
 if c['study']=='real':
  src=R/c['data_remote']
  for name in ['manifest.json','data_audit_only.npz']:assert sha(src/name)==c['files'][name],('data file mismatch',name)
  with np.load(src/'data_audit_only.npz') as z:
   splits={n:base.PUSplit(X=z[n+'_X'].copy(),y_true=z[n+'_y'].copy(),s=z[n+'_s'].copy()) for n in ['train','val','test']}
   source_rows={n:z[n+'_source_row'].copy() for n in ['train','val','test']}
  data=base.PUExperiment(**splits,pi=c['pi_tilde'],c=c['c'] or 0,p=c['p'],delta=np.nan,delta_sq=np.nan,signal=np.nan,n_signal=0,rho=np.nan,seed=c['seed'])
 else:data=gen(**c['generator'])
 fingerprint=base.experiment_fingerprint(data);assert fingerprint==c['fingerprint'],('data fingerprint mismatch',c['key'],fingerprint)
 # Hidden Y is copied out and then redacted from all model-facing split objects.
 true={n:getattr(data,n).y_true.copy() for n in ['train','val','test']}
 for n in true:getattr(data,n).y_true[:]=-1
 primary=[('final_train_plus_validation',np.vstack([data.train.X,data.val.X]),np.r_[data.train.s,data.val.s],np.r_[true['train'],true['val']],list(c['methods']))]
 train_mids=[mid for mid,v in c['methods'].items() if v['same_model_BA_available']]
 if train_mids:primary.append(('selected_train_only',data.train.X,data.train.s,true['train'],train_mids))
 rows=[];arrays={};actual_fits=0;cache_hits=0;warnings_all=[];verification=[]
 for stage,X,s,y,mids in primary:
  X.setflags(write=False);s.setflags(write=False)
  U=np.flatnonzero(s==0);nP=int((s==1).sum());nU=len(U);uid=stage
  arrays[uid+'_U_indices']=U.astype(np.int32);arrays[uid+'_hidden_Y_U']=y[U].astype(np.int8)
  if source_rows:arrays[uid+'_source_rows']=np.r_[source_rows['train'],source_rows['val']] if stage=='final_train_plus_validation' else source_rows['train']
  combos=sorted({(c['methods'][m]['params']['spy_fraction'],c['methods'][m]['params']['spy_quantile']) for m in mids})
  cache={}
  class CachedLogistic:
   def __init__(self,**kwargs):self.kwargs=kwargs
   def fit(self,Xt,yt):
    nonlocal actual_fits,cache_hits
    k=(self.kwargs['random_state'],ah(Xt),ah(yt))
    if k not in cache:
     model=LogisticRegression(**self.kwargs)
     with warnings.catch_warnings(record=True) as ws:
      warnings.simplefilter('always');model.fit(Xt,yt)
     warnings_all.extend(str(w.message) for w in ws);cache[k]=model;actual_fits+=1
    else:cache_hits+=1
    self.model=cache[k];return self
   def predict_proba(self,Xt):return self.model.predict_proba(Xt)
  env=dict(ns,LogisticRegression=CachedLogistic)
  # Both reference and instrumented code accept only X, observed S and fixed parameters.
  for sf,sq in combos:
   captures=[]
   def capture(b,spy_idx,spy_scores,unl_idx,unl_scores,selected_unl,threshold,rn_counts):
    captures.append(dict(b=b,spy_idx=spy_idx.copy(),spy_scores=spy_scores.copy(),unl_scores=unl_scores.copy(),candidate_count=len(selected_unl),threshold=float(threshold),counts=rn_counts.copy()))
   selector,ast_hash=instrument(bp,env,capture)
   rn=selector(X,s,spy_fraction=sf,spy_quantile=sq,n_repeats=10,selection_frequency=.70,seed=c['seed'])
   assert len(captures)==10
   counts=captures[-1]['counts'];assert np.array_equal(rn, (counts>=7)&(s==0))
   assert not rn[s==1].any();assert np.all((counts>=0)&(counts<=10))
   if verify:
    # Independent uninstrumented source call with fresh logistic fits, same inputs.
    original=ns['select_reliable_negatives_spy'];original_mask=original(X,s,spy_fraction=sf,spy_quantile=sq,n_repeats=10,selection_frequency=.70,seed=c['seed'])
    actual_fits+=10;assert np.array_equal(rn,original_mask)
    verification.append(dict(stage=stage,spy_fraction=sf,spy_quantile=sq,uninstrumented_original_mask_exactly_equal=True))
   q=np.mean(np.stack([t['unl_scores'] for t in captures]),axis=0)
   selected=rn[U];yy=y[U];neg=yy==0;pos=yy==1;nr=int(selected.sum());tn=int((selected&neg).sum());fp=int((selected&pos).sum())
   prefix=stage+f'_f{int(sf*100):02d}_q{int(sq*100):02d}'
   arrays[prefix+'_selected_indices']=np.flatnonzero(rn).astype(np.int32)
   arrays[prefix+'_selection_counts_U']=counts[U].astype(np.int8)
   arrays[prefix+'_mean_spy_score_U']=q
   arrays[prefix+'_spy_indices']=np.stack([t['spy_idx'] for t in captures]).astype(np.int32)
   arrays[prefix+'_spy_scores']=np.stack([t['spy_scores'] for t in captures])
   arrays[prefix+'_thresholds']=np.array([t['threshold'] for t in captures])
   arrays[prefix+'_candidate_counts']=np.array([t['candidate_count'] for t in captures],dtype=np.int32)
   quality=dict(stage=stage,spy_fraction=sf,spy_quantile=sq,nP=nP,nU=nU,hidden_positives_U=int(pos.sum()),hidden_negatives_U=int(neg.sum()),RN_n=nr,RN_true_negatives=tn,RN_hidden_positives=fp,RN_precision=tn/nr if nr else None,RN_contamination=fp/nr if nr else None,RN_negative_recall=tn/int(neg.sum()) if neg.any() else None,RN_positive_capture=fp/int(pos.sum()) if pos.any() else None,RN_selection_fraction=nr/nU,final_training_n=nP+nr,rn_training_positive_fraction=nP/(nP+nr),empty_RN=nr==0,discarded_true_negatives=int(neg.sum())-tn,discarded_negative_fraction=1-tn/int(neg.sum()) if neg.any() else None,diagnostic_score='mean of 10 original logistic-spy positive-class scores on original U',score_uses_hidden_Y=False,selection_uses_hidden_Y=False,selected_indices_sha256=ah(np.flatnonzero(rn).astype(np.int32)),selection_counts_sha256=ah(counts[U].astype(np.int8)),X_sha256=ah(X),S_sha256=ah(s),source_selector_ast_sha256=ast_hash,arrays_prefix=prefix,archive_RN_indices_available=False,archive_RN_counts_available=False,direct_original_RN_set_verification=False,reconstruction_status='exact_inputs_code_environment_verified_no_archived_RN_set',candidate_counts=[t['candidate_count'] for t in captures],spy_thresholds=[t['threshold'] for t in captures],logistic_spy_score_U_AUROC=float(roc_auc_score(yy,q)) if neg.any() and pos.any() else None)
   groups={'selected_true_negative':selected&neg,'unselected_true_negative':~selected&neg,'hidden_positive_U':pos}
   for group,mask in groups.items():quality.update(distribution(q[mask],'score_'+group))
   if neg.any() and pos.any():
    quality['easy_negative_score_gap']=quality['score_unselected_true_negative_mean']-quality['score_selected_true_negative_mean'] if tn and (~selected&neg).any() else None
   if c['study']=='main':
    lr=exact_llr(X[U],c['generator']);arrays[prefix+'_exact_DGP_LR_U']=lr
    quality['exact_DGP_LR_U_AUROC']=float(roc_auc_score(yy,lr))
    quality['LR_minus_spy_U_AUROC']=quality['exact_DGP_LR_U_AUROC']-quality['logistic_spy_score_U_AUROC']
    for group,mask in groups.items():quality.update(distribution(lr[mask],'LR_'+group))
    hard=neg&(lr>=np.quantile(lr[neg],.75));easy=neg&(lr<=np.quantile(lr[neg],.25))
    quality.update(hardest_negative_quartile_recall=float(selected[hard].mean()),easiest_negative_quartile_recall=float(selected[easy].mean()),selected_negative_near_LR_boundary_fraction=float((abs(lr[selected&neg])<=.5).mean()) if tn else None,all_negative_near_LR_boundary_fraction=float((abs(lr[neg])<=.5).mean()))
    if c['scenario_number']==39:
     product=X[U,0]*X[U,1]
     for group,mask in groups.items():
      quality[group+'_x0x1_mean']=float(product[mask].mean()) if mask.any() else None
      quality[group+'_x2_mean']=float(X[U,2][mask].mean()) if mask.any() else None
   rows.append(quality)
 out=H/'sets'/(c['key']+'.npz');out.parent.mkdir(exist_ok=True);np.savez_compressed(out,**arrays)
 result=dict(case=c['key'],task=c['task'],data_fingerprint=fingerprint,original_fingerprint_matched=True,source_files=hashes,packages=versions,threads=c['threads'],seed=c['seed'],rows=rows,arrays_sha256=sha(out),boosting_fits=0,hyperparameter_searches=0,spy_logistic_fits=actual_fits,identical_fit_cache_hits=cache_hits,hidden_Y_only_retrospective=True,direct_archived_set_evidence=False,uninstrumented_verification=verification,warnings=list(sorted(set(warnings_all))),seconds=time.perf_counter()-start,worker_sha256=sha(Path(__file__)))
 dest.parent.mkdir(exist_ok=True);dest.write_text(json.dumps(clean(result),separators=(',',':'),allow_nan=False))
 print(c['task'],c['key'],'done',len(rows),'sets',round(result['seconds'],2),'s',flush=True)
def main():
 global np,LogisticRegression,logsumexp,roc_auc_score,versions
 ap=argparse.ArgumentParser();ap.add_argument('--indices');ap.add_argument('--batch',type=int);ap.add_argument('--batch-size',type=int,default=20);ap.add_argument('--verify',action='store_true');args=ap.parse_args()
 cases=json.loads((H/'cases.json').read_text())
 idx=[int(x) for x in args.indices.split(',')] if args.indices else list(range(args.batch*args.batch_size,min(len(cases),(args.batch+1)*args.batch_size)))
 # Preserve the archived numerical thread settings; batches are grouped by design/source.
 for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:os.environ[k]=str(cases[idx[0]]['threads'])
 import numpy as np
 from scipy.special import logsumexp
 from sklearn.linear_model import LogisticRegression
 from sklearn.metrics import roc_auc_score
 from threadpoolctl import threadpool_limits
 versions={k:importlib.metadata.version(k) for k in ['numpy','scipy','scikit-learn']}
 assert versions=={'numpy':'2.2.6','scipy':'1.15.3','scikit-learn':'1.7.2'}
 for i in idx:
  try:
   with threadpool_limits(limits=cases[i]['threads']):run(cases[i],verify=args.verify)
  except Exception as e:
   (H/'errors').mkdir(exist_ok=True)
   (H/'errors'/(cases[i]['key']+'.json')).write_text(json.dumps(dict(case=cases[i]['key'],task=i,error=repr(e),traceback=traceback.format_exc()),indent=2))
   print(i,cases[i]['key'],'ERROR',repr(e),flush=True)
if __name__=='__main__':main()
