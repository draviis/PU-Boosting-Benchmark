"""Exact-DGP ceilings; executes only audited distribution functions, no estimators."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import ast,json,math,hashlib,sys
import numpy as np
from scipy.special import ndtr,logsumexp
from scipy.stats import ncx2
H=Path(__file__).resolve().parent;R=H.parents[1];sys.dont_write_bytecode=True

def load_functions(path,names):
 tree=ast.parse(path.read_text(encoding='utf-8-sig'));nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names]
 assert {x.name for x in nodes}==set(names)
 namespace={'np':np,'math':math,'logsumexp':logsumexp}
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
 return namespace,{x.name:x.lineno for x in nodes}

def metrics(pos,neg):
 ns=np.sort(neg);ps=np.sort(pos);n=len(pos);m=len(neg)
 v=(np.searchsorted(ns,pos,'left')+np.searchsorted(ns,pos,'right'))/(2*m)
 w=1-(np.searchsorted(ps,neg,'left')+np.searchsorted(ps,neg,'right'))/(2*n)
 auc=float(v.mean());se=math.sqrt(float(v.var(ddof=1)/n+w.var(ddof=1)/m))
 tpr=float((pos>=0).mean());tnr=float((neg<0).mean());ba=.5*(tpr+tnr);base=.5*math.sqrt(tpr*(1-tpr)/(n-1)+tnr*(1-tnr)/(m-1))
 return dict(mc_AUROC=auc,mc_AUROC_MCSE=se,mc_BA=ba,mc_BA_MCSE=base,mc_TPR=tpr,mc_TNR=tnr,mc_n_per_class=n,mc_n_total=n+m)

def independent_llr(kind,x):
 if kind=='curved_boundary':
  z=x.copy();z[:,1]-=1.5*(x[:,0]**2-1);return z[:,1:4].sum(axis=1)*(.9/math.sqrt(3))-.9**2/2
 if kind=='multiplicative_interaction':
  a=.67;b=.6;u=a*(x[:,0]+x[:,1]);v=a*(x[:,0]-x[:,1]);return np.logaddexp(u,-u)-np.logaddexp(v,-v)+b*x[:,2]-.5*b*b
 r=1.5;mu=.9/math.sqrt(3);return .5*((1-1/r)*(x[:,:3]**2).sum(axis=1)+(2*mu/r)*x[:,:3].sum(axis=1)-.9**2/r-3*math.log(r))

def main():
 fn=R/'publication_preparation/eta_scoring_20261013/frozen/main/two_sample_benchmark.py'
 ns,lines=load_functions(fn,['_curved_conditional_sample','_interaction_conditional_sample','_unequal_conditional_sample','_toeplitz_precision_times','_calibrate_deterministic_toeplitz_mean'])
 bp=R/'publication_preparation/scoring_harmonization_20261012/frozen/main/base_phase1_benchmark.py';bn,blines=load_functions(bp,['toeplitz_covariance','calibrate_mean_vector'])
 registry=json.loads((H/'scenario_registry.json').read_text());gaussian={};calibration=[]
 for s in registry:
  if s['scenario_index']>37:continue
  key=(s['p'],s['rho'],s['delta'])
  if key not in gaussian:
   p,rho,delta=key;S=bn['toeplitz_covariance'](p,rho);mu=bn['calibrate_mean_vector'](p,3,S,delta);mu2,precision,d2=ns['_calibrate_deterministic_toeplitz_mean'](p,3,rho,delta)
   achieved=float(mu@np.linalg.solve(S,mu));assert abs(achieved-delta**2)<1e-12;assert np.max(abs(mu-mu2))<1e-12;assert abs(d2-delta**2)<1e-12
   gaussian[key]=(float(ndtr(delta/math.sqrt(2))),float(ndtr(delta/2)))
   calibration.append(dict(p=p,rho=rho,delta=delta,achieved_delta_squared=achieved,one_two_mean_max_difference=float(np.max(abs(mu-mu2)))))
 n=5_000_000;batch=100_000;mc={};checks=[]
 for kind,seed in [('curved_boundary',202610160238),('multiplicative_interaction',202610160239),('unequal_covariance',202610160240)]:
  rng=np.random.default_rng(seed);score_sets=[];mu=np.zeros(10);mu[[1,2,3] if kind=='curved_boundary' else [0,1,2]]=.9/math.sqrt(3)
  for y in [1,0]:
   scores=[]
   for j in range(n//batch):
    yy=np.full(batch,y,dtype=np.int64)
    if kind=='curved_boundary':x,score=ns['_curved_conditional_sample'](batch,yy,mu,1.5,rng)
    elif kind=='multiplicative_interaction':x,score=ns['_interaction_conditional_sample'](batch,yy,.67,.6,10,rng)
    else:x,score=ns['_unequal_conditional_sample'](batch,yy,mu,1.5,rng)
    difference=float(np.max(np.abs(score-independent_llr(kind,x))));assert difference<1e-12
    if j==0:checks.append(dict(kind=kind,class_label=y,llr_max_difference=difference))
    scores.append(score)
   score_sets.append(np.concatenate(scores))
  values=metrics(*score_sets);values.update(mc_seed=seed,exact_LR=True,trained_model=False)
  if kind=='curved_boundary':
   a,b=float(ndtr(.9/math.sqrt(2))),float(ndtr(.9/2));assert abs(values['mc_AUROC']-a)<5*values['mc_AUROC_MCSE'];assert abs(values['mc_BA']-b)<5*values['mc_BA_MCSE']
   values.update(Bayes_AUROC=a,Bayes_BA=b,AUROC_source='analytic_invertible_Gaussian_transform',BA_source='analytic_invertible_Gaussian_transform')
  elif kind=='unequal_covariance':
   r=1.5;delta=.9;k=3;radius=r*delta**2/(r-1)**2+k*r*math.log(r)/(r-1)
   ba=float(.5*(ncx2.sf(radius/r,k,r*delta**2/(r-1)**2)+ncx2.cdf(radius,k,delta**2/(r-1)**2)))
   assert abs(values['mc_BA']-ba)<5*values['mc_BA_MCSE']
   values.update(Bayes_AUROC=values['mc_AUROC'],Bayes_BA=ba,AUROC_source='exact_LR_independent_MC',BA_source='analytic_noncentral_chi_square_CDF',radius_squared=radius)
  else:values.update(Bayes_AUROC=values['mc_AUROC'],Bayes_BA=values['mc_BA'],AUROC_source='exact_LR_independent_MC',BA_source='exact_LR_independent_MC')
  mc[kind]=values;print(kind,json.dumps(values),flush=True)
 result=[]
 for s in registry:
  base={k:s[k] for k in ['design','scenario_id','scenario_index','family','family_id','p','pi','delta','rho','structural_type','transform_strength','main_effect_shift','positive_variance_ratio']}
  if s['scenario_index']<=37:
   a,b=gaussian[(s['p'],s['rho'],s['delta'])];v=dict(Bayes_AUROC=a,Bayes_BA=b,AUROC_source='analytic_shared_covariance_Gaussian',BA_source='analytic_shared_covariance_Gaussian',exact_LR=True,trained_model=False,mc_n_total=0)
  else:v=mc[s['structural_type']]
  result.append(dict(base,**v,BA_rule='log(f1/f0)>=0; equivalently posterior>=pi',Bayes_AUROC_MCSE=v.get('mc_AUROC_MCSE',0.) if v['AUROC_source']=='exact_LR_independent_MC' else 0.,Bayes_BA_MCSE=v.get('mc_BA_MCSE',0.) if v['BA_source']=='exact_LR_independent_MC' else 0.))
 (H/'bayes_results.json').write_text(json.dumps(result,indent=2))
 (H/'bayes_validation.json').write_text(json.dumps(dict(gaussian_calibrations=calibration,independent_LR_checks=checks,distribution_source=str(fn.relative_to(R)),distribution_source_sha256=hashlib.sha256(fn.read_bytes()).hexdigest(),source_function_lines=lines,base_source_function_lines=blines,large_MC_total_observations=30_000_000,class_stratified_sampling=True,same_population_both_designs=True,model_fits=0),indent=2))
if __name__=='__main__':main()
