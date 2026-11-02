"""Approved Task 7: original catalog, final ranking, same-selected-model hard metrics."""
from pathlib import Path
import argparse,ast,copy,contextlib,hashlib,importlib,importlib.metadata,inspect,json,math,os,sys,time,traceback
H=Path(__file__).resolve().parent;R=H.parents[1];sys.dont_write_bytecode=True
BLOCKS={0:[1,2,3,4,5],1:[6,11,16,21],2:[7,12,17,22],3:[8,13,18,23],4:[9,14,19,24],5:[10,15,20,25],6:[26],7:[27]}
EXPECTED={'numpy':'2.2.6','scipy':'1.15.3','pandas':'2.3.3','scikit-learn':'1.7.2','torch':'2.5.1+cpu','catboost':'1.2.10','xgboost-cpu':'3.2.0','lightgbm':'4.7.0'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if hasattr(x,'item') and getattr(x,'ndim',1)==0:x=x.item()
    if isinstance(x,float) and not math.isfinite(x):return None
    return x
def save(path,obj):
    with path.open('x') as f:json.dump(clean(obj),f,indent=2,allow_nan=False)
def ah(x):
    import numpy as np
    x=np.ascontiguousarray(x);return hashlib.sha256(str(x.dtype).encode()+str(x.shape).encode()+x.tobytes()).hexdigest()
def threshold(q,s,design,pi):
    import numpy as np
    q=np.asarray(q,dtype=float);s=np.asarray(s)
    assert q.shape==s.shape and np.isfinite(q).all() and 0<pi<1
    assert design in ['one_sample','two_sample']
    mask=(s==0) if design=='two_sample' else np.ones(len(s),bool)
    v=q[mask];cut=float(np.quantile(v,1-pi,method='linear'))
    cnt=np.unique(v,return_counts=True)[1]
    return cut,dict(validation_population='validation_U_only' if design=='two_sample' else 'all_validation',validation_n=len(v),validation_unique_scores=len(cnt),validation_max_tie_mass=float(cnt.max()/len(v)),validation_ties_at_threshold=int((v==cut).sum()),validation_predicted_positive_fraction=float((v>=cut).mean()),validation_material_prevalence_deviation=bool(abs((v>=cut).mean()-pi)>.02),validation_population_mask_sha256=ah(mask))
def instrument(fn,recorder):
    node=ast.parse(inspect.getsource(fn)).body[0];original=ast.dump(node,include_attributes=False);hooks=[]
    class Add(ast.NodeTransformer):
        in_engine_loop=False
        def visit_For(self,n):
            previous=self.in_engine_loop
            if isinstance(n.target,ast.Name) and n.target.id=='engine_param':self.in_engine_loop=True
            n=self.generic_visit(n);self.in_engine_loop=previous
            return n
        def visit_Assign(self,n):
            names=[t.id for t in n.targets if isinstance(t,ast.Name)]
            add=[]
            if isinstance(n.value,ast.Dict):
                for name in ['candidate','best']:
                    if name in names:add+=ast.parse(name+'["_t7_model"] = model').body
            if any(v in names for v in ['val_auc','val_score']):
                add+=ast.parse('_t7_record(locals())').body
            # An empty RN set creates an unfittable one-class candidate in some
            # original engines. Continue only within the training tuning grid.
            tuple_names=[v.id for t in n.targets if isinstance(t,ast.Tuple) for v in t.elts if isinstance(v,ast.Name)]
            if self.in_engine_loop and fn.__name__=='fit_one_boosting_cell' and 'y_fit' in tuple_names:
                add+=ast.parse("if strategy == 'RN' and np.unique(y_fit).size < 2:\n    _t7_record(dict(strategy_param=strategy_param,engine_param=engine_param,invalid_candidate=True,invalid_reason='one_class_RN_training',training_positive_count=int((y_fit==1).sum()),training_negative_count=int((y_fit==0).sum())))\n    continue").body
            hooks.extend(add);return [n]+add if add else n
        def visit_Expr(self,n):
            c=n.value
            if isinstance(c,ast.Call) and isinstance(c.func,ast.Attribute) and c.func.attr=='append' and isinstance(c.func.value,ast.Name) and c.func.value.id=='candidates':
                add=ast.parse('_t7_record(candidates[-1])').body;hooks.extend(add);return [n]+add
            return self.generic_visit(n)
    changed=Add().visit(copy.deepcopy(node));dumps={ast.dump(n,include_attributes=False) for n in hooks}
    class Remove(ast.NodeTransformer):
        def generic_visit(self,n):
            if ast.dump(n,include_attributes=False) in dumps:return None
            return super().generic_visit(n)
    assert ast.dump(Remove().visit(copy.deepcopy(changed)),include_attributes=False)==original
    env=dict(fn.__globals__,_t7_record=recorder)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[changed],type_ignores=[])),inspect.getsourcefile(fn),'exec'),env)
    return env[fn.__name__],hashlib.sha256(original.encode()).hexdigest()

def run(case,block,out,parity):
    import numpy as np,torch
    from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score,f1_score,matthews_corrcoef,brier_score_loss,log_loss
    from nonlinear_dgps import pu_split,exact_log_lr,config
    versions={k:importlib.metadata.version(k) for k in EXPECTED};assert versions==EXPECTED,versions
    for name,digest in json.loads((H/'source_hashes.json').read_text()).items():assert sha(H/name)==digest,name
    freeze=json.loads((H/'execution_manifest.json').read_text())
    for name,digest in freeze['frozen_files'].items():
        if name.endswith('.zip'):continue
        assert sha(H/name)==digest,name
    sys.path.insert(0,str(H/'sources'/case['design']))
    module=importlib.import_module('two_sample_benchmark' if case['design']=='two_sample' else 'confirmatory_benchmark')
    base=module.base;repair=module.repaired
    sys.path.insert(0,str(H/'sources'));import scoring_harmonization
    assert base.ADAPU_AVAILABLE and repair.ADAPU_AVAILABLE
    sn=case['sn'];cfg=config(sn);parameter=json.loads((H/'frozen_dgp_parameters.json').read_text())['parameters'][cfg['id']]['value']
    true={};splits={};fingerprints={}
    for short,long in [('train','train'),('val','validation'),('test','test')]:
        v,y=pu_split(sn,parameter,case['design'],case['rep'],long);true[short]=y.copy()
        splits[short]=base.PUSplit(X=v['X'],s=v['S'],y_true=np.full(len(y),-1,dtype=np.int8))
        fingerprints[short]={k:ah(x) for k,x in dict(X=v['X'],S=v['S'],Y=y).items()}
    data=base.PUExperiment(**splits,pi=.2,c=.3 if case['design']=='one_sample' else 0.,p=cfg['p'],delta=np.nan,delta_sq=np.nan,signal=np.nan,n_signal=cfg['q'],rho=np.nan,seed=case['model_seed'])
    if case['design']=='two_sample':module._ACTIVE_VALIDATION_U_MASK=data.val.s==0
    targets=out/'evaluation_targets.npz'
    if not targets.exists():np.savez_compressed(targets,**{k+'_Y':y for k,y in true.items()},val_S=data.val.s,train_S=data.train.s,bayes_test_log_lr=exact_log_lr(sn,parameter,data.test.X))
    else:
        with np.load(targets) as z:
            for k,y in true.items():np.testing.assert_array_equal(z[k+'_Y'],y)
    def metric(arrays,prob=None):
        cut,diag=threshold(arrays['selected_val_raw'],data.val.s,case['design'],data.pi)
        selected=arrays['selected_test_raw'];final=arrays.get('refit_test_raw');pred=selected>=cut;ty=true['test']
        row=dict(AUROC=float(roc_auc_score(ty,final)) if final is not None else None,AP=float(average_precision_score(ty,final)) if final is not None else None,BA=float(balanced_accuracy_score(ty,pred)),F1=float(f1_score(ty,pred,zero_division=0)),MCC=float(matthews_corrcoef(ty,pred)),threshold=cut,**diag,
          test_predicted_positive_fraction=float(pred.mean()),test_material_prevalence_deviation=bool(abs(pred.mean()-.2)>.02),test_ties_at_threshold=int((selected==cut).sum()),test_max_tie_mass=float(np.unique(selected,return_counts=True)[1].max()/len(selected)),prediction_sha256=ah(pred),ranking_model='refit_train_plus_validation' if final is not None else 'unavailable_empty_RN_refit',refit_available=final is not None,refit_failure_reason=None if final is not None else 'empty_RN_refit',hard_metric_model='selected_train_only',threshold_transfer_used=False,test_labels_used_for_selection=False,selected_AUROC_diagnostic=float(roc_auc_score(ty,selected)),selected_AP_diagnostic=float(average_precision_score(ty,selected)))
        if prob is not None:row.update(Brier=float(brier_score_loss(ty,prob)),LogLoss=float(log_loss(ty,prob,labels=[0,1])))
        return row
    def corrected(q):
        if case['design']=='one_sample':return base.clip_prob(q/data.c)
        alpha=float(np.r_[data.train.s,data.val.s].mean())
        return module._probability_from_log_density_ratio(module._case_control_log_density_ratio(q,alpha),data.pi)
    catalog={m['canonical_method_id']:m for m in json.loads((H/'proposed_method_catalog.json').read_text())['methods'] if m['design']==case['design']}
    done=[]
    for n in BLOCKS[block]:
        mid=f'M{n:03d}';dest=out/(mid+'.json')
        if dest.exists():
            saved=json.loads(dest.read_text());assert sha(out/(mid+'.npz'))==saved['arrays_sha256']
            assert saved['worker_sha256'] in {sha(Path(__file__)),sha(H/'worker.py'),sha(H/'worker_rn_recovery.py')}
            done.append(mid);continue
        start=time.perf_counter();candidates=[];captured={};parity_record=None
        for name in ['train','val']:getattr(data,name).y_true[:]=true[name] if n<=2 else -1
        assert np.all(data.test.y_true==-1)
        def record(v):
            keep=['invalid_candidate','invalid_reason','training_positive_count','training_negative_count','C','c_value','params','engine_param','strategy_param','val_auc','val_score','l2','best_epoch','val_risk','correction_count','minimum_negative_component','beta','requested_estimators','effective_estimators']
            candidates.append(clean({k:v[k] for k in keep if k in v}))
            if 6<=n<=10 and case['design']=='two_sample':
                q=v.get('raw_val')
                if q is not None:
                    transformed=module._case_control_log_density_ratio(q,float(data.train.s.mean()))
                    assert abs(base.pu_proxy_auc(data.val.s,q)-base.pu_proxy_auc(data.val.s,transformed))<1e-14
        def evaluate(method,y_test,score_test,threshold,probability=None,extra=None):
            assert np.all(y_test==-1)
            loc=inspect.currentframe().f_back.f_locals;best=loc['best']
            fm=loc.get('final_model',loc.get('final'));assert fm is not None
            if n in [4,5]:
                sm=loc['selected_model']
                def scorer(model,X):
                    with torch.no_grad():return model(torch.tensor(X,dtype=torch.float32)).cpu().numpy()
            elif n==26:
                sm=best['_t7_model']
                def scorer(model,X):
                    stage=best['effective_estimators'] if model is sm else len(model.estimators)
                    return model.staged_predict(X.astype(np.float32),stage,test=False)
            elif n==27:
                sm=best['_t7_model']
                def scorer(model,X):return model.decision_function(X)
            else:
                sm=best['_t7_model']
                def scorer(model,X):return base.model_score(model,X)
            assert sm is not fm
            for name,model in [('selected',sm),('refit',fm)]:
                for split in ['val','test']:captured[name+'_'+split+'_raw']=np.asarray(scorer(model,getattr(data,split).X),dtype=float).copy()
            if 'score_val' in best and n in [26,27]:np.testing.assert_allclose(captured['selected_val_raw'],best['score_val'],rtol=0,atol=0)
            source_final=np.asarray(score_test,float);captured['source_refit_test_ranking']=source_final.copy()
            if probability is not None:captured['refit_test_probability']=np.asarray(probability,float).copy()
            if n==3 and case['design']=='two_sample':
                expected=module._case_control_log_density_ratio(captured['refit_test_raw'],float(np.r_[data.train.s,data.val.s].mean()))
            else:expected=captured['refit_test_raw']
            np.testing.assert_allclose(source_final,expected,rtol=0,atol=1e-12)
            assert all(np.isfinite(v).all() for v in captured.values())
            return dict(extra=clean(extra),row=metric(captured,captured.get('refit_test_probability')))
        try:
            if 11<=n<=15:
                twin=f'M{n-5:03d}';original=json.loads((out/(twin+'.json')).read_text())
                with np.load(out/(twin+'.npz')) as z:captured={k:z[k].copy() for k in ['selected_val_raw','selected_test_raw','refit_val_raw','refit_test_raw']}
                captured['refit_test_probability']=corrected(captured['refit_test_raw'])
                result=dict(row=metric(captured,captured['refit_test_probability']),extra=original['selection'])
                candidates=original['candidates'];ast_hash=original['source_function_AST_sha256'];shared=twin
                for key in ['AUROC','AP','BA','F1','MCC','threshold','prediction_sha256']:assert result['row'][key]==original['row'][key],key
            else:
                if n==1:fn=base.fit_oracle_logistic;args=()
                elif n==2:fn=base.fit_oracle_catboost;args=()
                elif n==3:fn=module.fit_case_control_known_pi if case['design']=='two_sample' else scoring_harmonization.fit_logistic_probability_correction;args=()
                elif n in [4,5]:fn=repair.fit_pu_logistic_baseline;args=()
                elif n==26:fn=repair.fit_adapu_one_sample;args=()
                elif n==27:fn=module.fit_pu_gradient_boosting if case['design']=='two_sample' else repair.fit_pu_gradient_boosting;args=()
                else:
                    fn=module.fit_one_boosting_cell if case['design']=='two_sample' else base.fit_one_boosting_cell
                    args=('Naive' if n<=10 else 'RN' if n<=20 else 'CostSensitive',['AdaBoost','GBM','XGBoost','LightGBM','CatBoost'][(n-6)%5])
                fit,ast_hash=instrument(fn,record);fit.__globals__['evaluate_test']=evaluate;base.evaluate_test=evaluate
                callargs=(base,data) if n==3 and case['design']=='one_sample' else (data,)+args
                kwargs={'non_negative':n==5} if n in [4,5] else {}
                logpath=out/(mid+'_fit_'+str(time.time_ns())+'.log')
                with logpath.open('x') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                    try:result=fit(*callargs,**kwargs)
                    except Exception as refit_error:
                        assert 16<=n<=20
                        trace=refit_error.__traceback__;loc=None
                        while trace is not None:
                            if trace.tb_frame.f_code.co_name=='fit_one_boosting_cell':loc=trace.tb_frame.f_locals
                            trace=trace.tb_next
                        assert loc is not None and loc.get('best') is not None
                        # Both original functions have already selected best and
                        # constructed the final estimator before this failure.
                        final_key='final' if case['design']=='two_sample' else 'final_model'
                        assert final_key in loc
                        yy=np.asarray(loc['y_fit' if case['design']=='two_sample' else 'y_dev_fit'])
                        assert len(yy)>0 and np.all(yy==1),'Not an empty-negative final refit'
                        best=loc['best'];sm=best['_t7_model']
                        for split in ['val','test']:
                            captured['selected_'+split+'_raw']=np.asarray(base.model_score(sm,getattr(data,split).X),dtype=float).copy()
                        assert all(np.isfinite(v).all() for v in captured.values())
                        evidence=dict(reason='empty_RN_refit',refit_positive_count=int(len(yy)),refit_negative_count=0,refit_pseudo_label_sha256=ah(yy),original_exception=repr(refit_error),traceback=traceback.format_exc(),selected_model_retained=True,no_refit_fallback=True,no_parameter_reselection=True)
                        result=dict(extra=dict(best_params={**best['strategy_param'],**best['engine_param']},val_metric=best['val_auc'],refit_failure_reason='empty_RN_refit'),row=metric(captured),refit_failure_evidence=evidence)
                shared=None
                if parity:
                    # Independent uninstrumented fitting on preflight only. Same full grids.
                    def plain_eval(method,y_test,score_test,threshold,probability=None,extra=None):
                        return dict(score=np.asarray(score_test,float).copy(),extra=clean(extra))
                    env=dict(fn.__globals__,evaluate_test=plain_eval);base.evaluate_test=plain_eval
                    import types
                    original_fn=types.FunctionType(fn.__code__,env,fn.__name__,fn.__defaults__,fn.__closure__)
                    with (out/(mid+'_parity_'+str(time.time_ns())+'.log')).open('x') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):plain=original_fn(*callargs,**kwargs)
                    np.testing.assert_allclose(plain['score'],captured['source_refit_test_ranking'],rtol=0,atol=1e-12)
                    assert plain['extra']==result['extra'],(plain['extra'],result['extra'])
                    parity_record=dict(full_grid_uninstrumented_equal=True,max_score_difference=float(np.max(np.abs(plain['score']-captured['source_refit_test_ranking']))))
            row=dict(case=case['key'],scenario_id=cfg['id'],family=cfg['family'],design=case['design'],rep=case['rep'],method_id=mid,method=catalog[mid]['source_label'],supervised_reference=n<=2,p=cfg['p'],model_seed=case['model_seed'],**result['row'])
            arrpath=out/(mid+'.npz');assert not arrpath.exists();np.savez_compressed(arrpath,**captured)
            save(dest,dict(row=row,refit_failure_evidence=result.get('refit_failure_evidence'),empty_refit_protocol_sha256=sha(H/'rn_empty_refit_protocol.json'),selection=result['extra'],candidates=candidates,shared_fitted_models_with=shared,original_scientific_AST_unchanged=not 16<=n<=20,original_source_files_unchanged=True,recovery_change='skip_one_class_RN_training_candidates' if 16<=n<=20 else 'none',recovery_protocol_sha256=sha(H/'rn_recovery_protocol.json'),source_function_AST_sha256=ast_hash,score_hashes={k:ah(v) for k,v in captured.items()},arrays_sha256=sha(arrpath),worker_sha256=sha(Path(__file__)),execution_manifest_sha256=sha(H/'execution_manifest.json'),data_fingerprints=fingerprints,packages=versions,seconds=time.perf_counter()-start,parity=parity_record,practical_PU_hidden_Y_redacted=n>2,threshold_function_accepts_no_Y=True))
            print(case['key'],mid,'complete',round(time.perf_counter()-start,2),flush=True);done.append(mid)
        except Exception as exc:
            save(out/(mid+'_failure_'+str(time.time_ns())+'.json'),dict(error=repr(exc),traceback=traceback.format_exc(),case=case,method=mid))
            continue
    if len(done)!=len(BLOCKS[block]):raise RuntimeError(dict(case=case['key'],block=block,completed=done,reason='Some methods failed; successful later methods retained'))
    save(out/'complete.json',dict(case=case,block=block,completed_methods=done,expected_methods=BLOCKS[block],empty_refit_protocol_sha256=sha(H/'rn_empty_refit_protocol.json'),recovery_protocol_sha256=sha(H/'rn_recovery_protocol.json'),worker_sha256=sha(Path(__file__)),execution_manifest_sha256=sha(H/'execution_manifest.json'),data_fingerprints=fingerprints,targets_sha256=sha(targets),packages=versions))

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=int,required=True);p.add_argument('--block',type=int,required=True);p.add_argument('--preflight',action='store_true');p.add_argument('--parity',action='store_true');a=p.parse_args()
    case=json.loads((H/('preflight_cases.json' if a.preflight else 'main_cases.json')).read_text())[a.case]
    out=H/('preflight_results' if a.preflight else 'results')/case['key']/('block'+str(a.block));out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():
        d=json.loads((out/'complete.json').read_text());assert d['worker_sha256']==sha(Path(__file__));print('Already complete',case['key'],a.block);return
    for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','PU_BENCHMARK_N_JOBS']:os.environ[k]=str(case['threads'])
    dep=R/'phase3_confirmatory_S001_S040_100reps/Ada-PU/src'
    assert sha(dep/'adapu.py')=='3eee5aee7a6819f98c4bf26bee56efe9b73b3094094d5a8a350cf278bd7541fa'
    assert sha(dep/'decisionstump.pyx')=='be42f8b8868f621e4b15c4ca7069cc7154848432848acba558eee28bd1ff4741'
    assert sha(dep/'decisionstump.cpython-310-x86_64-linux-gnu.so')=='83068a7c4b2f92e6cda64118f113ba63bb746b70277af7e77f98bd5def840ce3'
    os.environ['ADAPU_SRC']=str(dep);os.environ['ADAPU_CACHE_DIR']=str(out/'cache')
    sys.path[:0]=[str(dep),str(dep.parent)]
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=case['threads']):run(case,a.block,out,a.parity)
if __name__=='__main__':main()
