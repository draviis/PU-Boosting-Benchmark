"""Fit controlled neural/linear PU models on the untouched saved real-data splits."""
import argparse,hashlib,importlib.metadata,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,balanced_accuracy_score,matthews_corrcoef,brier_score_loss,log_loss
from models import fit
from context_sources import comparator_ids
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def metrics(y,score,threshold,probability):
    h=score>=threshold
    r=dict(AUROC=roc_auc_score(y,score),AUPRC=average_precision_score(y,score),F1=f1_score(y,h,zero_division=0),
           BalancedAccuracy=balanced_accuracy_score(y,h),MCC=matthews_corrcoef(y,h))
    p=np.clip(probability,1e-7,1-1e-7)
    r.update(Brier=brier_score_loss(y,p),LogLoss=log_loss(y,p));return r

def main():
    p=argparse.ArgumentParser();p.add_argument('--task',required=True,type=int)
    p.add_argument('--input-root',type=Path,default=Path.home()/'PUBenchmark/adapu_rng_release_20260925/adapu_workflow/inputs')
    p.add_argument('--benchmark-root',type=Path);p.add_argument('--output',type=Path,default=HERE/'results')
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    plan=json.loads((HERE/'plan.json').read_text());cell=plan['cells'][a.task-1];assert cell['task']==a.task
    src=a.benchmark_root/cell['local_data_dir'] if a.benchmark_root else a.input_root/cell['key']
    for name,digest in cell['files'].items():assert sha(src/name)==digest,name
    meta=json.loads((src/'manifest.json').read_text());assert not meta['test_labels_used_for_prior']
    assert meta['data_fingerprint']==cell['data_fingerprint']
    packages={}
    for name,expected in meta['packages'].items():
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            if name=='xgboost-cpu':packages[name]=importlib.metadata.version('xgboost')
            else:raise
    environment_matches=packages==meta['packages']
    if a.smoke:
        recipe=dict(max_epochs=20,learning_rates=[.001,.003],widths=[4],decays=[.0001],check_every=5,patience=2)
        recipe_sha=None
    else:
        assert environment_matches,'Production environment must match saved real-data study'
        recipe_path=HERE/'frozen_training_rule.json'
        assert recipe_path.exists(),'Training diagnostic must be reviewed and a rule frozen before production runs'
        frozen=json.loads(recipe_path.read_text());assert frozen['status']=='frozen_after_validation_review'
        assert frozen['test_scores_used_for_rule'] is False
        recipe=frozen['fit_settings'];recipe_sha=sha(recipe_path)
        assert set(recipe)=={'max_epochs','learning_rates','widths','decays','check_every','patience'}
    with np.load(src/'data_audit_only.npz',allow_pickle=False) as data:
        tx,ts,vx,vs,te=[data[k].copy() for k in ('train_X','train_s','val_X','val_s','test_X')]
    if a.smoke:
        ti=np.r_[np.flatnonzero(ts==1)[:25],np.flatnonzero(ts==0)[:100]]
        vi=np.r_[np.flatnonzero(vs==1)[:25],np.flatnonzero(vs==0)[:100]]
        tx,ts,vx,vs,te=tx[ti],ts[ti],vx[vi],vs[vi],te[:200]
    arrays=[tx,ts,vx,vs,te];before=[hashlib.sha256(x.tobytes()).hexdigest() for x in arrays]
    out=a.output/f'task_{a.task:03d}';out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    pending={}
    for kind in ['linear','mlp']:
        for nn in [False,True]:
            for selection in ['auc','risk']:
                name=f'{kind}_{"nnpu" if nn else "upu"}_{selection}';print(name,flush=True)
                pending[name]=fit(tx,ts,vx,vs,te,pi=meta['population_prior_supplied'],
                    c=meta['labeling_probability_one_sample'] if cell['scheme']=='one_sample' else 0.,
                    seed=cell['seed'],kind=kind,nonnegative=nn,selection=selection,**recipe)
                assert all(c['status']=='ok' for c in pending[name]['candidates'])
    assert before==[hashlib.sha256(x.tobytes()).hexdigest() for x in arrays]
    # Test outcomes enter only after every configuration has been selected and fitted.
    with np.load(src/'data_audit_only.npz',allow_pickle=False) as data:
        y=data['test_y'].copy();test_rows=data['test_source_row'].copy()
    if a.smoke:y,test_rows=y[:200],test_rows[:200]
    records=[];diagnostics={}
    for method,result in pending.items():
        score=result.pop('score');prob=result.pop('probability');threshold=result['threshold']
        pd.DataFrame(dict(source_row=test_rows,y=y,score=score,probability=prob,prediction=(score>=threshold).astype(int))).to_csv(out/f'{method}_predictions.csv',index=False)
        records.append(dict(method=method,status='ok',**metrics(y,score,threshold,prob),threshold=threshold,seconds=result['seconds'],execution_role='new_controlled_fit'))
        diagnostics[method]=result
    if not a.smoke:
        original=pd.read_csv(src/'results.csv',float_precision='round_trip')
        for code,name in comparator_ids(cell):
            part=original[original.method_id==code];assert len(part)==1 and part.iloc[0]['status']=='ok'
            row=part.iloc[0].to_dict();row.update(method=name,execution_role='reused_original_contextual_result');records.append(row)
    table=pd.DataFrame(records).assign(task=a.task,dataset=cell['dataset'],budget=cell['budget'],scheme=cell['scheme'],seed=cell['seed'],data_fingerprint=cell['data_fingerprint'])
    table.to_csv(out/'results.csv',index=False)
    (out/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    (out/'metadata.json').write_text(json.dumps(dict(cell=cell['key'],smoke=a.smoke,input_hashes=cell['files'],
       plan_sha256=sha(HERE/'plan.json'),recipe_sha256=recipe_sha,fit_settings=recipe,environment_matches=environment_matches,
       packages=packages,code_sha256={p.name:sha(p) for p in HERE.glob('*.py')},hidden_development_labels_loaded=False),indent=2))
    (out/'complete.json').write_text(json.dumps({n:sha(out/n) for n in ['results.csv','metadata.json','diagnostics.json']}))
    print('Completed',len(records),'methods',flush=True)

if __name__=='__main__':main()
