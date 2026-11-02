"""Rerun only AdaPU with the exact saved arrays and frozen tuning implementation."""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data-root',type=Path,help='Portable inputs root with one directory per cell key')
    p.add_argument('--benchmark-root',type=Path,default=HERE.parents[1])
    p.add_argument('--repair-root',type=Path,default=HERE.parent/'adapu_rng_repair')
    p.add_argument('--allow-environment-difference',action='store_true',help='Local validation only; collector rejects these runs')
    args=p.parse_args()
    plan=json.loads((HERE/'plan.json').read_text())
    assert 1<=args.task<=len(plan['cells'])
    cell=plan['cells'][args.task-1]
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    src=(args.data_root/cell['key']) if args.data_root else (args.benchmark_root/cell['local_data_dir'])
    for name,expected in cell['files'].items():
        assert digest(src/name)==expected, f'Input artifact changed: {name}'
    meta=json.loads((src/'manifest.json').read_text())
    repair=args.repair_root.resolve()
    for name,expected in plan['repaired_sources'].items():
        assert digest(repair/'src'/name)==expected, f'Repair source changed: {name}'
    os.environ['ADAPU_SRC']=str(repair/'src')
    os.environ['ADAPU_CACHE_DIR']=str(out/'cache')
    threads=str(meta['worker_threads'])
    for name in ('PU_BENCHMARK_N_JOBS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):
        os.environ[name]=threads
    sys.path[:0]=[str(repair),str(repair/'src')]
    phase='phase3_confirmatory_S001_S040_100reps' if cell['scheme']=='one_sample' else 'phase5_two_sample_scar_confirmatory_100reps'
    for rel,expected in cell['original_source_sha256'].items():
        assert digest(HERE/'frozen_code'/rel)==expected, rel
    sys.path.insert(0,str(HERE/'frozen_code'/phase/'code'))
    module=importlib.import_module('confirmatory_benchmark' if cell['scheme']=='one_sample' else 'two_sample_benchmark')
    base,repaired=module.base,module.repaired
    import numpy as np
    import pandas as pd
    import adapu
    import decisionstump
    from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,balanced_accuracy_score,matthews_corrcoef
    assert Path(adapu.__file__).resolve()==repair/'src/adapu.py'
    assert Path(decisionstump.__file__).resolve().parent in (repair,repair/'src')
    assert repaired.ADAPU_AVAILABLE
    versions={}
    for package in meta['packages']:
        try:
            versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            if package=='xgboost-cpu':
                versions[package]=importlib.metadata.version('xgboost')
            else:
                raise
    environment_matches=versions==meta['packages']
    if not args.allow_environment_difference:
        assert environment_matches, f'Environment differs: current={versions}, expected={meta["packages"]}'
    with np.load(src/'data_audit_only.npz',allow_pickle=False) as saved:
        splits={name:base.PUSplit(X=saved[f'{name}_X'].copy(),y_true=saved[f'{name}_y'].copy(),s=saved[f'{name}_s'].copy()) for name in ('train','val','test')}
        test_rows=saved['test_source_row'].copy()
    data=base.PUExperiment(**splits,pi=meta['population_prior_supplied'],
                           c=meta['labeling_probability_one_sample'] if cell['scheme']=='one_sample' else 0.,
                           p=splits['train'].X.shape[1],delta=np.nan,delta_sq=np.nan,signal=np.nan,n_signal=0,rho=np.nan,seed=cell['seed'])
    assert base.experiment_fingerprint(data)==cell['data_fingerprint'], 'Saved-data fingerprint mismatch'
    if cell['scheme']=='two_sample':
        module._ACTIVE_VALIDATION_U_MASK=data.val.s==0
    original= pd.read_csv(src/'results.csv',float_precision='round_trip')
    old=original.loc[original.method_id==cell['method_id']].iloc[0]
    data.train.y_true[:]=-1
    data.val.y_true[:]=-1
    masked_fingerprint=base.experiment_fingerprint(data)
    trace=[]
    native_class=repaired.AdaBoost_PU
    class AuditedAdaPU(native_class):
        def fit(self,*a,**kw):
            answer=super().fit(*a,**kw)
            stumps=[(float(s.feature_idx),float(s.threshold),s.inequal,float(s.estimator_weight)) for s in self.estimators]
            trace.append(dict(beta=self.beta,requested_estimators=self.n_estimators,fitted_estimators=len(self.estimators),
                              stumps_sha256=hashlib.sha256(json.dumps(stumps).encode()).hexdigest()))
            return answer
    repaired.AdaBoost_PU=AuditedAdaPU
    evaluate=repaired.evaluate_test
    captured={}
    def capture(method,y_test,score_test,threshold,probability=None,extra=None):
        score=np.asarray(score_test,dtype=float)
        assert probability is None
        captured['frame']=pd.DataFrame(dict(source_row=test_rows,y_true=y_test,score=score,prediction=(score>=threshold).astype(int)))
        return evaluate(method,y_test,score_test,threshold,probability,extra)
    repaired.evaluate_test=capture
    print(f'Running full frozen AdaPU tuning/refit: {cell["key"]}',flush=True)
    start=time.perf_counter()
    result=base.fit_adapu_one_sample(data)
    elapsed=time.perf_counter()-start
    assert base.experiment_fingerprint(data)==masked_fingerprint, 'Fitter mutated input data'
    assert len(trace)==6, f'Expected all five beta candidates and final refit, got {len(trace)}'
    frame=captured['frame']
    assert np.isfinite(frame.score).all()
    old_prediction=pd.read_csv(src/f'predictions/{cell["method_id"]}.csv',float_precision='round_trip')
    np.testing.assert_array_equal(frame.source_row,old_prediction.source_row)
    np.testing.assert_array_equal(frame.y_true,old_prediction.y_true)
    frame.to_csv(out/'predictions.csv',index=False)
    # Recompute from serialized output, independently of the benchmark evaluator.
    saved=pd.read_csv(out/'predictions.csv',float_precision='round_trip')
    y,s,h=saved.y_true,saved.score,saved.prediction
    metrics=dict(AUROC=roc_auc_score(y,s),AUPRC=average_precision_score(y,s),F1=f1_score(y,h,zero_division=0),BalancedAccuracy=balanced_accuracy_score(y,h),MCC=matthews_corrcoef(y,h))
    for name,value in metrics.items():
        np.testing.assert_allclose(result[name],value,rtol=0,atol=1e-12)
    np.testing.assert_array_equal(h,(s>=result['threshold']).astype(int))
    result.update(dataset=cell['dataset'],c=meta['labeling_probability_one_sample'],method_id=cell['method_id'],
                  method_block='Block3',seed=cell['seed'],sampling_scheme=cell['scheme'],comparison_role='pu_method',
                  status='ok',data_fingerprint=cell['data_fingerprint'],fit_and_tune_seconds=elapsed,repair_revision='adapu_rng_v1')
    result['best_params']=json.dumps(result['best_params'])
    pd.DataFrame([result]).to_csv(out/'results.csv',index=False)
    provenance=dict(cell=cell['key'],task=cell['task'],plan_sha256=digest(HERE/'plan.json'),
                    input_hashes=cell['files'],source_sha256=cell['original_source_sha256'],
                    repaired_source_sha256=plan['repaired_sources'],extension=str(decisionstump.__file__),
                    extension_sha256=digest(Path(decisionstump.__file__)),python=sys.version,packages=versions,
                    threads=int(threads),environment_matches_original=environment_matches,
                    validation_only=args.allow_environment_difference,full_grid_fits=trace,
                    score_sha256=hashlib.sha256(frame.score.to_numpy().tobytes()).hexdigest(),
                    metric_deltas={k:float(v)-float(old[k]) for k,v in metrics.items()})
    provenance['workflow_sha256']=digest(Path(__file__))
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    (out/'complete.json').write_text(json.dumps(dict(key=cell['key'],verified=True,
                                  predictions_sha256=digest(out/'predictions.csv'),results_sha256=digest(out/'results.csv'),
                                  provenance_sha256=digest(out/'provenance.json'))),encoding='utf-8')
    print(json.dumps(dict(key=cell['key'],seconds=elapsed,best_params=result['best_params'],metrics=metrics),indent=2))


if __name__=='__main__':
    main()
