"""Task 6: training-only KM2, exact prior-dependent searches, archived-score reuse."""
from pathlib import Path
import argparse,ast,copy,contextlib,hashlib,importlib,importlib.metadata,inspect,json,math,os,sys,time,traceback
H=Path(__file__).resolve().parent; R=H.parents[1];sys.dont_write_bytecode=True
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if hasattr(x,"item") and getattr(x,"ndim",1)==0:x=x.item()
    if isinstance(x,float) and not math.isfinite(x):return None
    return x
def save_json(path,obj):
    with path.open("x") as f:json.dump(clean(obj),f,indent=2,allow_nan=False)
def instrument(fn,recorder):
    source=inspect.getsource(fn);node=ast.parse(source).body[0]
    original=ast.dump(node,include_attributes=False)
    hooks=[]
    class Capture(ast.NodeTransformer):
        def visit_Assign(self,n):
            if any(isinstance(t,ast.Name) and t.id=="candidate" for t in n.targets) and isinstance(n.value,ast.Dict):
                added=ast.parse('candidate["_task6_model"] = model\n_task6_record(candidate)').body
                hooks.extend(added);return [n]+added
            return n
        def visit_Expr(self,n):
            call=n.value
            if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr=="append" and isinstance(call.func.value,ast.Name) and call.func.value.id=="candidates":
                added=ast.parse('_task6_record(candidates[-1])').body
                hooks.extend(added);return [n]+added
            return self.generic_visit(n)
    updated=Capture().visit(copy.deepcopy(node))
    # Remove only appended hooks to prove no original scientific statement changed.
    hook_dumps={ast.dump(n,include_attributes=False) for n in hooks}
    class Remove(ast.NodeTransformer):
        def generic_visit(self,n):
            if ast.dump(n,include_attributes=False) in hook_dumps:return None
            return super().generic_visit(n)
    restored=Remove().visit(copy.deepcopy(updated))
    assert ast.dump(restored,include_attributes=False)==original
    env=dict(fn.__globals__,_task6_record=recorder)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[updated],type_ignores=[])),inspect.getsourcefile(fn),"exec"),env)
    return env[fn.__name__],hashlib.sha256(original.encode()).hexdigest()

def run(c,out,estimate_only=False):
    import numpy as np,pandas as pd,torch
    from scipy.special import expit
    from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score,brier_score_loss,log_loss
    from prior_estimator import estimate_prior,array_hash
    versions={k:importlib.metadata.version(k) for k in ["numpy","scipy","pandas","scikit-learn","torch","catboost","xgboost-cpu","lightgbm","cvxopt"]}
    expected={"numpy":"2.2.6","scipy":"1.15.3","pandas":"2.3.3","scikit-learn":"1.7.2","torch":"2.5.1+cpu","catboost":"1.2.10","xgboost-cpu":"3.2.0","lightgbm":"4.7.0","cvxopt":"1.3.2"}
    assert versions==expected,versions
    frozen=R/c["frozen_remote"]
    for name,digest in c["frozen_hashes"].items():assert sha(frozen/name)==digest
    sys.path.insert(0,str(frozen))
    module=importlib.import_module("two_sample_benchmark" if c["design"]=="two_sample" else "confirmatory_benchmark")
    base=module.base; repaired=module.repaired
    import data_recovery
    from scipy.special import logsumexp
    from sklearn.linear_model import LogisticRegression
    data_recovery.np=np;data_recovery.logsumexp=logsumexp;data_recovery.LogisticRegression=LogisticRegression
    _,_,gen,_,_=data_recovery.namespaces(c)
    if c["study"]=="main":data=gen(**c["generator"])
    else:
        src=R/c["data_remote"]
        for name in ["manifest.json","data_audit_only.npz"]:assert sha(src/name)==c["files"][name]
        with np.load(src/"data_audit_only.npz") as z:
            splits={n:base.PUSplit(X=z[n+"_X"].copy(),s=z[n+"_s"].copy(),y_true=z[n+"_y"].copy()) for n in ["train","val","test"]}
        data=base.PUExperiment(**splits,pi=c["pi"],c=c["c"] or 0,p=c["p"],delta=np.nan,delta_sq=np.nan,signal=np.nan,n_signal=0,rho=np.nan,seed=c["seed"])
    assert base.experiment_fingerprint(data)==c["fingerprint"],"source fingerprint"
    true={n:getattr(data,n).y_true.copy() for n in ["train","val","test"]}
    for n in true:getattr(data,n).y_true[:]=-1
    ty=true["test"]
    targets=out/"evaluation_targets.npz"
    if not targets.exists():np.savez_compressed(targets,**{n+"_Y":v for n,v in true.items()},val_S=data.val.s)
    ep=out/"estimate.json"
    if ep.exists():
        est=json.loads(ep.read_text());assert est["training_X_sha256"]==array_hash(np.asarray(data.train.X,dtype=np.float64))
        assert est["training_S_sha256"]==array_hash(data.train.s)
    else:
        start=time.perf_counter()
        est,arrays=estimate_prior(data.train.X,data.train.s,c["design"],c["estimator_seed"])
        est.update(case=c["key"],data_fingerprint=c["fingerprint"],seconds=time.perf_counter()-start,packages=versions,
                   source_code_sha256=sha(H/"prior_estimator.py"),estimator_source_sha256=sha(H/"sources/upstream_Kernel_MPE_grad_threshold.py"),
                   true_pi=c["pi"],true_c=c["c"],true_pi_U=c["pi"]*(1-(c["c"] or 0))/(1-c["pi"]*(c["c"] or 0)),
                   empirical_train_pi=float(true["train"].mean()),
                   empirical_train_pi_U=float(true["train"][data.train.s==0].mean()),
                   empirical_train_c=float(data.train.s.sum()/true["train"].sum()),
                   hidden_labels_used_in_estimator=False)
        np.savez_compressed(out/"estimate_arrays.npz",**arrays)
        est["arrays_sha256"]=sha(out/"estimate_arrays.npz");save_json(ep,est)
    print(c["key"],"prior",est["status"],est.get("pi_hat"),flush=True)
    if estimate_only:return
    assert sha(out/"estimate_arrays.npz")==est["arrays_sha256"]
    if c["design"]=="two_sample":module._ACTIVE_VALIDATION_U_MASK=data.val.s==0
    rows=[];raw_sources={};reused={}
    def metrics(q,p=None):
        q=np.asarray(q,float);assert np.isfinite(q).all() and q.shape==ty.shape
        z=dict(AUROC=float(roc_auc_score(ty,q)),AP=float(average_precision_score(ty,q)))
        if p is not None:
            p=np.asarray(p,float);assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
            z.update(Brier=float(brier_score_loss(ty,p)),LogLoss=float(log_loss(ty,np.clip(p,1e-7,1-1e-7),labels=[0,1])))
        return z
    def probability(q,pi,cval):
        q=np.asarray(q,float)
        if c["design"]=="one_sample":return np.clip(q/cval,0,1)
        alpha=float(np.r_[data.train.s,data.val.s].mean())
        q=np.clip(q,np.nextafter(0.,1.),np.nextafter(1.,0.))
        lr=np.log(q)-np.log1p(-q)+math.log1p(-alpha)-math.log(alpha)
        return np.exp(np.minimum(math.log(pi)+lr,0.))
    def ba(qv,qt,pi):
        population=qv[data.val.s==0] if c["design"]=="two_sample" else qv
        cut=float(np.quantile(population,1-pi))
        return dict(BA=float(balanced_accuracy_score(ty,qt>=cut)),threshold=cut,
                    validation_population="U_only" if c["design"]=="two_sample" else "complete_PU",
                    validation_n=len(population),validation_positive_fraction=float((population>=cut).mean()),
                    test_positive_fraction=float((qt>=cut).mean()),test_labels_used_for_threshold=False)
    prefix="M" if c["design"]=="one_sample" else "TM"
    scorebase=R/c["score_archive_remote"]
    if (scorebase/"scores.npz").exists():
        stamp=json.loads((scorebase/"complete.json").read_text())
        assert sha(scorebase/"scores.npz")==stamp["scores_sha256"]
        raw_sources[str((scorebase/"scores.npz").relative_to(R))]=sha(scorebase/"scores.npz")
        with np.load(scorebase/"scores.npz") as z:
            np.testing.assert_array_equal(ty,z["y_test"])
            for n in [3,11,12,13,14,15]:
                key=f"{prefix}{n:03d}_q_test"
                if key in z:reused[f"{prefix}{n:03d}"]=z[key].copy()
    if c["study"]=="real":
        for n in [3,11,12,13,14,15]:
            mid=f"{prefix}{n:03d}"
            if mid in reused:continue
            source_mid=f"{prefix}{n-5:03d}" if n>=11 else mid
            path=R/c["data_remote"]/"predictions"/(source_mid+".csv")
            frame=pd.read_csv(path,float_precision="round_trip")
            np.testing.assert_array_equal(frame.y_true,ty)
            q=frame.score.to_numpy()
            if n==3:
                assert c["design"]=="two_sample"
                alpha=float(np.r_[data.train.s,data.val.s].mean())
                q=expit(q-math.log((1-alpha)/alpha))
            else:assert c["oracle"][mid]["params"]==c["oracle"][source_mid]["params"]
            reused[mid]=q;raw_sources[str(path.relative_to(R))]=sha(path)
    assert set(reused)=={f"{prefix}{n:03d}" for n in [3,11,12,13,14,15]},"missing saved source q"
    for mid,info in c["oracle"].items():
        n=int(mid[-3:])
        if n in [4,5,26,27]:continue
        archive=info["metrics"]
        row=dict(method_id=mid,method=info["method"],status="success" if est["status"]=="success" else "prior_failed",
                 oracle_AUROC=archive["AUROC"],oracle_AP=archive["AUPRC"],
                 estimated_AUROC=archive["AUROC"],estimated_AP=archive["AUPRC"],
                 Delta_AUROC=0.,Delta_AP=0.,ranking_reused=True,
                 invariance="identical_archived_ranking_no_new_fit",params=info["params"],
                 oracle_archived_Brier=archive["Brier"],oracle_archived_LogLoss=archive["LogLoss"])
        if mid in reused:
            q=reused[mid];oracle=metrics(q,probability(q,c["pi"],c["c"]))
            delta={m:oracle[m]-archive["AUPRC" if m=="AP" else m] for m in ["AUROC","AP"]}
            assert max(abs(v) for v in delta.values())<=5e-5,("archive rank recovery",mid,delta)
            row.update(score_sha256=array_hash(q),oracle_q_recovery_differences=delta,
                       oracle_Brier=oracle["Brier"],oracle_LogLoss=oracle["LogLoss"])
            if est["status"]=="success":
                changed=metrics(q,probability(q,est["pi_hat"],est["c_hat"]))
                assert changed["AUROC"]==oracle["AUROC"] and changed["AP"]==oracle["AP"]
                row.update(estimated_Brier=changed["Brier"],estimated_LogLoss=changed["LogLoss"],
                           Delta_Brier=changed["Brier"]-oracle["Brier"],Delta_LogLoss=changed["LogLoss"]-oracle["LogLoss"],
                           raw_score_rank_invariance_exact=True)
        elif archive["Brier"] is not None:
            row.update(oracle_Brier=archive["Brier"],estimated_Brier=archive["Brier"],Delta_Brier=0.,
                       oracle_LogLoss=archive["LogLoss"],estimated_LogLoss=archive["LogLoss"],Delta_LogLoss=0.)
        same=R/c["same_model_archive_remote"]/(mid+".npz")
        if same.exists() and est["status"]=="success":
            stamp=json.loads(same.with_suffix(".json").read_text());assert sha(same)==stamp["arrays_sha256"]
            with np.load(same) as z:
                np.testing.assert_array_equal(z["y_test"],ty)
                row["BA_oracle_same_model"]=ba(z["q_val"],z["q_test_same"],c["pi"])
                row["BA_estimated_same_model"]=ba(z["q_val"],z["q_test_same"],est["pi_hat"])
            row["BA_score_source"]=str(same.relative_to(R))
        rows.append(row)
    reused_path=out/"reused_source_scores.npz"
    if not reused_path.exists():np.savez_compressed(reused_path,**reused)
    if est["status"]=="success":
        data.pi=est["pi_hat"];data.c=est["c_hat"] if c["design"]=="one_sample" else 0.
        for n in [4,5,26,27]:
            mid=f"{prefix}{n:03d}";dest=out/(mid+".json")
            if dest.exists():
                old=json.loads(dest.read_text());assert sha(out/(mid+".npz"))==old["arrays_sha256"]
                rows.append(old["row"]);continue
            start=time.perf_counter();candidates=[];captured={}
            def record(candidate):
                candidates.append(clean({k:v for k,v in candidate.items() if k not in ["model","_task6_model","state","score_val"]}))
            def evaluate(method,y_test,score_test,threshold,probability=None,extra=None):
                assert np.all(y_test==-1)
                loc=inspect.currentframe().f_back.f_locals
                qf=np.asarray(score_test).copy()
                if n in [4,5]:
                    model=loc["selected_model"]
                    with torch.no_grad():qv=model(torch.tensor(data.val.X,dtype=torch.float32)).cpu().numpy();qt=model(torch.tensor(data.test.X,dtype=torch.float32)).cpu().numpy()
                else:
                    best=loc["best"];model=best.get("_task6_model",best.get("model"))
                    assert model is not None
                    qv=np.asarray(best["score_val"]).copy()
                    qt=model.staged_predict(data.test.X.astype(np.float32),best["effective_estimators"],test=False) if n==26 else model.decision_function(data.test.X)
                captured.update(q_val=qv,q_test_selected=qt,q_test_final=qf)
                if probability is not None:captured["probability_final"]=np.asarray(probability).copy()
                return dict(metrics(qf,probability),extra=clean(extra),BA_same_model=ba(qv,qt,data.pi))
            fn=repaired.fit_pu_logistic_baseline if n in [4,5] else repaired.fit_adapu_one_sample if n==26 else module.fit_pu_gradient_boosting if c["design"]=="two_sample" else repaired.fit_pu_gradient_boosting
            try:
                fit,ast_hash=instrument(fn,record)
                # Use a shallow namespace copy to change only final evaluation.
                fit.__globals__["evaluate_test"]=evaluate
                if "base" in fit.__globals__:
                    base.evaluate_test=evaluate
                with (out/(mid+"_fit.log")).open("x") as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                    result=fit(data,non_negative=n==5) if n in [4,5] else fit(data)
                assert all(np.isfinite(v).all() for v in captured.values())
                archive=c["oracle"][mid]["metrics"]
                row=dict(method_id=mid,method=c["oracle"][mid]["method"],status="success",ranking_reused=False,
                         oracle_AUROC=archive["AUROC"],oracle_AP=archive["AUPRC"],estimated_AUROC=result["AUROC"],estimated_AP=result["AP"],
                         Delta_AUROC=result["AUROC"]-archive["AUROC"],Delta_AP=result["AP"]-archive["AUPRC"],
                         params=result["extra"]["best_params"],validation_result=result["extra"].get("val_metric"),
                         BA_estimated_same_model=result["BA_same_model"],
                         oracle_archived_Brier=archive["Brier"],oracle_archived_LogLoss=archive["LogLoss"])
                for metric in ["Brier","LogLoss"]:
                    if metric in result and archive[metric] is not None:
                        row.update({f"estimated_{metric}":result[metric],f"oracle_{metric}":archive[metric],f"Delta_{metric}":result[metric]-archive[metric]})
                same=R/c["same_model_archive_remote"]/(mid+".npz")
                if same.exists():
                    stamp=json.loads(same.with_suffix(".json").read_text());assert sha(same)==stamp["arrays_sha256"]
                    with np.load(same) as z:
                        np.testing.assert_array_equal(z["y_test"],ty)
                        row["BA_oracle_same_model"]=ba(z["q_val"],z["q_test_same"],c["pi"])
                    row["BA_score_source"]=str(same.relative_to(R))
                np.savez_compressed(out/(mid+".npz"),**captured)
                record_out=dict(row=row,candidates=candidates,original_scientific_AST_unchanged=True,
                                source_function_AST_sha256=ast_hash,arrays_sha256=sha(out/(mid+".npz")),
                                prediction_hashes={k:array_hash(v) for k,v in captured.items()},diagnostics=result["extra"],
                                packages=versions,seconds=time.perf_counter()-start,seed=c["seed"])
                save_json(dest,record_out);rows.append(row)
                print(c["key"],mid,"done",round(time.perf_counter()-start,2),flush=True)
            except Exception as exc:
                row=dict(method_id=mid,method=c["oracle"][mid]["method"],status="fit_failed",error=repr(exc),
                         oracle_AUROC=c["oracle"][mid]["metrics"]["AUROC"],oracle_AP=c["oracle"][mid]["metrics"]["AUPRC"])
                save_json(out/(mid+"_failure.json"),dict(row=row,candidates=candidates,traceback=traceback.format_exc()))
                rows.append(row);print(c["key"],mid,"FAILED",repr(exc),flush=True)
    else:
        for n in [4,5,26,27]:
            mid=f"{prefix}{n:03d}"
            rows.append(dict(method_id=mid,method=c["oracle"][mid]["method"],status="prior_failed",
                             oracle_AUROC=c["oracle"][mid]["metrics"]["AUROC"],oracle_AP=c["oracle"][mid]["metrics"]["AUPRC"]))
    assert len(rows)==25
    save_json(out/"complete.json",dict(case=c["key"],fingerprint=c["fingerprint"],rows=rows,
              estimate_sha256=sha(ep),reused_scores_sha256=sha(reused_path),raw_score_sources=raw_sources,
              targets_sha256=sha(targets),packages=versions,source_hashes=c["frozen_hashes"],
              worker_sha256=sha(Path(__file__)),cases_sha256=sha(H/"cases.json"),
              original_data_fingerprint_verified=True,hidden_Y_redacted=True,
              estimator_training_only=True,threshold_transfer_used=False))
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--task",type=int,required=True);ap.add_argument("--estimate-only",action="store_true");args=ap.parse_args()
    c=json.loads((H/"cases.json").read_text())[args.task]
    out=H/"results"/c["key"];out.mkdir(parents=True,exist_ok=True)
    if (out/"complete.json").exists():
        d=json.loads((out/"complete.json").read_text());assert d["worker_sha256"]==sha(Path(__file__))
        print("already complete",c["key"]);return
    for k in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","BLIS_NUM_THREADS","PU_BENCHMARK_N_JOBS"]:os.environ[k]=str(c["threads"])
    repair=R/("adapu_rng_release_20260925/adapu_rng_repair" if c["study"]=="real" else c["phase"]+"/Ada-PU")
    os.environ["ADAPU_SRC"]=str(repair/"src");os.environ["ADAPU_CACHE_DIR"]=str(out/"cache")
    sys.path[:0]=[str(H/"vendor_cluster"),str(repair),str(repair/"src")]
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=c["threads"]):run(c,out,args.estimate_only)
if __name__=="__main__":main()
