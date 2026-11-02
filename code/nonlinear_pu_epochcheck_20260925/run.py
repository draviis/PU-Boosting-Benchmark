import argparse,copy,functools,hashlib,importlib,importlib.metadata,inspect,json,os,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def load(cell):
    sys.path.insert(0,str(HERE/'frozen'/cell['phase']))
    return importlib.import_module('confirmatory_benchmark' if cell['scheme']=='one_sample' else 'two_sample_benchmark')

def generate(module,cell):
    fn=module.generate_structural_one_sample_scar if cell['scheme']=='one_sample' else module.generate_two_sample_scar
    integers={'n_train','n_val','n_test','n_p_train','n_u_train','n_p_val','n_u_val','p','n_signal'}
    args={}
    for name in inspect.signature(fn).parameters:
        if name=='seed': args[name]=cell['seed']
        elif name in cell['definition']:
            value=cell['definition'][name]
            args[name]=value if name=='structural_type' else (int(float(value)) if name in integers else float(value))
    return fn(**args)

def metrics(y,score,threshold,probability):
    import numpy as np
    from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,balanced_accuracy_score,matthews_corrcoef,brier_score_loss,log_loss
    hard=(score>=threshold).astype(int)
    result=dict(AUROC=roc_auc_score(y,score),AUPRC=average_precision_score(y,score),F1=f1_score(y,hard,zero_division=0),BalancedAccuracy=balanced_accuracy_score(y,hard),MCC=matthews_corrcoef(y,hard))
    if probability is not None:
        prob=np.clip(probability,1e-7,1-1e-7)
        result.update(Brier=brier_score_loss(y,prob),LogLoss=log_loss(y,prob))
    return result

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--task',type=int,required=True)
    parser.add_argument('--output',type=Path,default=HERE/'results')
    parser.add_argument('--reference-root',type=Path,default=Path.home()/'PUBenchmark/nonlinear_pu_pilot_20260925')
    parser.add_argument('--smoke',action='store_true',help='Small validation fixture, never accepted by collector')
    args=parser.parse_args()
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','PU_BENCHMARK_N_JOBS']:
        os.environ[key]='4'
    os.environ['ADAPU_SRC']=str(HERE/'unused_adapu')
    import numpy as np,pandas as pd,torch
    from models import fit
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    plan=json.loads((HERE/'plan.json').read_text()); cell=plan['cells'][args.task-1]
    assert 1<=args.task<=120
    for rel,expected in plan['provenance'].items():
        if rel.startswith('frozen/'):
            assert sha(HERE/rel)==expected
    if args.smoke:
        cell=copy.deepcopy(cell); cell['seed']=979999000+args.task
    module=load(cell); base=module.base
    base.CatBoostClassifier=functools.partial(base.CatBoostClassifier,allow_writing_files=False)
    module.repaired.CatBoostClassifier=functools.partial(module.repaired.CatBoostClassifier,allow_writing_files=False)
    data=generate(module,cell)
    if args.smoke:
        # Preserve observations of both groups in a small technical fixture.
        for split in (data.train,data.val):
            take=np.r_[np.flatnonzero(split.s==1)[:25],np.flatnonzero(split.s==0)[:100]]
            split.X=split.X[take]; split.s=split.s[take]; split.y_true=split.y_true[take]
        data.test.X=data.test.X[:200]; data.test.s=data.test.s[:200]; data.test.y_true=data.test.y_true[:200]
    fingerprint=base.experiment_fingerprint(data)
    reference_meta=None
    if not args.smoke:
        reference_folder=args.reference_root/'results'/f'task_{cell["task"]:03d}'
        reference_meta=json.loads((reference_folder/'metadata.json').read_text())
        assert reference_meta['cell']==cell and not reference_meta['smoke']
        assert reference_meta['plan_sha256']==plan['reference_plan_sha256']
        assert reference_meta['data_fingerprint']==fingerprint, 'Original data fingerprint changed'
        for package,expected in reference_meta['packages'].items():
            assert importlib.metadata.version(package)==expected, f'Environment changed: {package}'
        original_marker=json.loads((reference_folder/'complete.json').read_text())
        assert original_marker['results_sha256']==sha(reference_folder/'results.csv')
    out=args.output/f'task_{args.task:03d}'; out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'data.npz',**{f'{name}_{field}':getattr(getattr(data,name),field) for name in ('train','val','test') for field in ('X','s','y_true')})
    counts={name:dict(P=int((getattr(data,name).s==1).sum()),U=int((getattr(data,name).s==0).sum())) for name in ('train','val')}
    assert all(v['P']>0 and v['U']>0 for v in counts.values()),'Empty group; do not redraw seed'
    data.train.y_true[:]=-1; data.val.y_true[:]=-1
    masked=base.experiment_fingerprint(data)
    if cell['scheme']=='two_sample': module._ACTIVE_VALIDATION_U_MASK=data.val.s==0
    rows=[]; audits={}
    methods=[(kind,nn,selection) for kind in ('linear','mlp') for nn in (False,True) for selection in ('auc','risk')]
    for kind,nn,selection in methods:
        name=f'{kind}_{"nnpu" if nn else "upu"}_{selection}'
        print(name,flush=True)
        try:
            extra=dict(max_epochs=20,check_every=5,patience=2,widths=(4,),decays=(1e-4,)) if args.smoke else {}
            fit_result=fit(data.train.X,data.train.s,data.val.X,data.val.s,data.test.X,pi=data.pi,c=data.c,seed=data.seed,nonnegative=nn,kind=kind,selection=selection,**extra)
            score,prob,threshold=fit_result.pop('score'),fit_result.pop('probability'),fit_result['threshold']
            measured=metrics(data.test.y_true,score,threshold,prob)
            pd.DataFrame(dict(y=data.test.y_true,score=score,probability=prob,prediction=(score>=threshold).astype(int))).to_csv(out/f'{name}_predictions.csv',index=False)
            audits[name]=fit_result
            rows.append(dict(method=name,status='ok',**measured,threshold=threshold,seconds=fit_result['seconds'],best_params=json.dumps(fit_result['best_params'])))
        except Exception:
            rows.append(dict(method=name,status='failed',error=traceback.format_exc()))
    reference_hashes={}
    if not args.smoke:
        old=pd.read_csv(reference_folder/'results.csv',float_precision='round_trip')
        for name in ('RN_CatBoost','CostSensitive_CatBoost'):
            matching=old[old.method==name]
            assert len(matching)==1
            record=matching.iloc[0].to_dict()
            assert record['status']=='ok' and record['data_fingerprint']==fingerprint
            file=reference_folder/f'{name}_predictions.csv'
            pred=pd.read_csv(file,float_precision='round_trip')
            np.testing.assert_array_equal(pred.y,data.test.y_true)
            probability=None if pred.probability.isna().all() else pred.probability.to_numpy()
            for key,value in metrics(pred.y,pred.score.to_numpy(),record['threshold'],probability).items():
                np.testing.assert_allclose(record[key],value,atol=1e-12,rtol=0)
            (out/file.name).write_bytes(file.read_bytes())
            reference_hashes[file.name]=sha(file)
            record['execution_role']='reused_original_contextual_comparator'
            rows.append(record)
    assert base.experiment_fingerprint(data)==masked,'Data mutated'
    table=pd.DataFrame(rows).assign(task=cell['task'],scenario=cell['scenario'],scheme=cell['scheme'],seed=cell['seed'],data_fingerprint=fingerprint)
    table.to_csv(out/'results.csv',index=False)
    (out/'diagnostics.json').write_text(json.dumps(audits,indent=2))
    (out/'metadata.json').write_text(json.dumps(dict(cell=cell,smoke=args.smoke,counts=counts,plan_sha256=sha(HERE/'plan.json'),
        max_epochs=1500,reference_prediction_hashes=reference_hashes,
        reference_metadata_sha256=None if args.smoke else sha(reference_folder/'metadata.json'),
        reference_results_sha256=None if args.smoke else sha(reference_folder/'results.csv'),
        code_sha256={p.name:sha(p) for p in HERE.glob('*.py')},packages={name:importlib.metadata.version(name) for name in ('numpy','pandas','scipy','torch','scikit-learn','catboost')},python=sys.version,
        data_fingerprint=fingerprint),indent=2))
    if any(r['status']!='ok' for r in rows): raise RuntimeError('Some methods failed; outputs retained')
    (out/'complete.json').write_text(json.dumps(dict(rows=len(rows),results_sha256=sha(out/'results.csv'))))
    print(f'Completed {len(rows)} methods',flush=True)

if __name__=='__main__': main()
