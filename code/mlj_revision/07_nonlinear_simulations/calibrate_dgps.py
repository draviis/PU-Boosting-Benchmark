"""Parameter calibration and independent Bayes integration only. No learner imports/fits."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import datetime,hashlib,importlib.metadata,json,time,sys
sys.dont_write_bytecode=True
import numpy as np
from scipy.special import ndtri
from nonlinear_dgps import SCENARIOS,config,streams,conditional,population,exact_log_lr,analytic_ceiling,auc_mc
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,x):
 with (H/name).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def search(sn):
 cfg=config(sn);lo,hi=cfg['bracket'];history=[]
 def trial(theta):
    rs=streams(sn,11);y=np.r_[np.ones(50000,dtype=np.int8),np.zeros(50000,dtype=np.int8)]
    X=conditional(sn,theta,y,rs[0],rs[1],full=False)
    score=exact_log_lr(sn,theta,X);m=auc_mc(y,score)
    rec=dict(scenario_id=cfg['id'],parameter=theta,auroc=m['bayes_auroc'],mcse=m['auroc_mcse'],n_per_class=50000)
    history.append(rec);return rec['auroc']
 al,ah=trial(lo),trial(hi)
 assert al<cfg['target']<ah,(sn,al,ah)
 for i in range(12):
    mid=(lo+hi)/2
    if trial(mid)<cfg['target']:lo=mid
    else:hi=mid
 best=min(history,key=lambda r:abs(r['auroc']-cfg['target']))
 return best['parameter'],history
def main():
 assert json.loads((H/'dgp_unit_validation.json').read_text())['passed']
 initial=json.loads((H/'plan_precalibration_fingerprint.json').read_text())
 assert sha(H/'NONLINEAR_DGP_PLAN.md')==initial['sha256']
 save('calibration_started.json',dict(utc=now(),learner_fits=0,
     plan_sha256=initial['sha256'],code_hashes={p.name:sha(p) for p in H.glob('*.py')},
     versions={k:importlib.metadata.version(k) for k in ['numpy','scipy','scikit-learn']},
     parameter_search_phase=11,confirmation_phase=22,confirmation_n=1000000,chunk_size=10000))
 pars={};history=[]
 for sn in range(1,7):
    cfg=config(sn)
    if sn==2:theta=np.sqrt(cfg['target']/(1-cfg['target']));method='analytic_inverse'
    elif sn==4:theta=float(ndtri(cfg['target'])/np.sqrt(2));method='analytic_inverse'
    elif sn==5:theta=pars['NL01']['value'];method='same_signal_as_NL01'
    else:theta,hh=search(sn);history+=hh;method='fixed_seed_12_step_bisection'
    pars[cfg['id']]=dict(value=float(theta),name=cfg['parameter'],selection_method=method,target=cfg['target'],p=cfg['p'],q=cfg['q'])
    print(cfg['id'],'selected',cfg['parameter'],round(theta,9),flush=True)
 save('parameter_search_history.json',history)
 save('frozen_dgp_parameters.json',dict(parameters=pars,utc=now(),frozen_before_confirmation=True,main_run_approval='pending_plan_review'))
 (H/'calibration_populations').mkdir(exist_ok=False)
 result=[];start=time.perf_counter()
 for sn in range(1,7):
    cfg=config(sn);theta=pars[cfg['id']]['value'];rs=streams(sn,22);n=1000000;chunk=10000
    labels=np.empty(n,dtype=np.int8);scores=np.empty(n,dtype=np.float64);xhash=hashlib.sha256()
    for begin in range(0,n,chunk):
        X,y=population(sn,theta,min(chunk,n-begin),rs,full=True);llr=exact_log_lr(sn,theta,X)
        assert np.isfinite(llr).all()
        labels[begin:begin+len(y)]=y;scores[begin:begin+len(y)]=llr
        xhash.update(np.ascontiguousarray(X).tobytes())
    f=H/'calibration_populations'/(cfg['id']+'.npz')
    with f.open('xb') as out:np.savez_compressed(out,Y=labels,exact_log_likelihood_ratio=scores)
    m=auc_mc(labels,scores);aa,bb=analytic_ceiling(sn,theta)
    within=bool(.70<m['auroc_mc95_lower'] and m['auroc_mc95_upper']<.90 and abs(m['bayes_auroc']-cfg['target'])<=.015)
    analytic_ok=bool((aa is None or abs(m['bayes_auroc']-aa)<=4*m['auroc_mcse']) and
                     (bb is None or abs(m['bayes_ba']-bb)<=4*m['ba_mcse']))
    row=dict(scenario_id=cfg['id'],scenario=cfg['name'],family=cfg['family'],p=cfg['p'],
       n_signal_coordinates=cfg['q'],n_noise=cfg['p']-cfg['q'],parameter=cfg['parameter'],parameter_value=theta,
       target_bayes_auroc=cfg['target'],n_population=n,prior_pi=.2,exact_likelihood_ratio=True,
       **m,auroc_target_error=m['bayes_auroc']-cfg['target'],
       analytic_bayes_auroc=aa,analytic_bayes_ba=bb,difficulty_accepted=within,analytic_check_passed=analytic_ok,
       full_feature_matrix_generated=True,confirmation_seed_entropy=[20261017,7,sn,22],
       chunk_size=chunk,full_X_sha256=xhash.hexdigest(),
       score_file=f.relative_to(H).as_posix(),score_file_sha256=sha(f),learner_fits=0)
    result.append(row)
    save('calibration_populations/'+cfg['id']+'_summary.json',row)
    print(cfg['id'],'confirmation AUROC',round(m['bayes_auroc'],6),'MCSE',round(m['auroc_mcse'],6),
          'BA',round(m['bayes_ba'],6),'accepted',within,'elapsed',round(time.perf_counter()-start,1),flush=True)
 save('nonlinear_dgp_calibration.json',result)
 a,b=result[0],result[4];delta=a['bayes_auroc']-b['bayes_auroc'];dse=np.sqrt(a['auroc_mcse']**2+b['auroc_mcse']**2)
 bad=[r['scenario_id'] for r in result if not(r['difficulty_accepted'] and r['analytic_check_passed'])]
 pair_ok=bool(abs(delta)<=4*dse)
 save('calibration_validation.json',dict(passed=not bad and pair_ok,failed_scenarios=bad,
      identical_signal_pair_check=dict(scenarios=['NL01','NL05'],difference=delta,mcse_difference=dse,
          within_four_mcse=pair_ok,parameters_identical=pars['NL01']['value']==pars['NL05']['value']),
      full_populations=6,total_independent_confirmation_observations=6000000,learner_fits=0,
      main_run_approval='pending_plan_review',elapsed_seconds=time.perf_counter()-start,utc=now()))
 assert not bad and pair_ok,(bad,delta,dse)
 print('Calibration passed; full learner runs remain unstarted.',flush=True)
if __name__=='__main__':main()
