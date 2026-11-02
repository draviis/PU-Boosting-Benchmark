from pathlib import Path
import inspect,json,math,sys
import numpy as np
from prior_estimator import source_namespace,derive_priors,estimate_prior,VERSION,PARAMS
H=Path(__file__).resolve().parent
out=H/"estimator_unit_validation.json"
if out.exists():raise FileExistsError(out)
checks={}
max_error=0.
for pi in [.001,.01,.05,.2,.5,.95]:
 for c in [.01,.1,.3,.9,1.]:
  lam=pi*c; pu=pi*(1-c)/(1-lam)
  r=derive_priors(pu,lam,1-lam,"one_sample")
  max_error=max(max_error,abs(r["pi_hat"]-pi),abs(r["c_hat"]-c))
assert max_error<1e-12
checks["population_identity_max_error"]=max_error
rng=np.random.default_rng(93771)
sim=[]
for pi,c in [(.01,.3),(.05,.1),(.2,.3),(.5,.9)]:
 y=rng.random(500000)<pi;s=y&(rng.random(len(y))<c)
 pu=y[~s].mean()
 r=derive_priors(pu,int(s.sum()),int((~s).sum()),"one_sample")
 assert abs(r["pi_hat"]-y.mean())<1e-14
 assert abs(r["c_hat"]-s.sum()/y.sum())<1e-14
 sim.append(dict(pi=pi,c=c,empirical_pi=float(y.mean()),empirical_c=float(s.sum()/y.sum())))
checks["simulated_SCAR_identity_cases"]=sim
assert not derive_priors(-1,3,97,"one_sample")["priors_admissible"]
assert not derive_priors(1.1,3,97,"two_sample")["priors_admissible"]
checks["invalid_priors_rejected"]=True
d=dict(solver_calls=[],sqrt_safeguards=[])
env=source_namespace(d)
u=np.array([[-.2],[1.2]])
sol,dist=env["find_nearest_valid_distribution"](u,np.eye(2))
assert abs(dist-.08)<1e-6
assert np.max(abs(np.array(sol["x"]).ravel()-[0,1]))<1e-6
checks["analytic_simplex_projection_error"]=float(abs(dist-.08))
# Independent directly loaded upstream function namespace; no wrapper safeguards.
P=rng.normal(3,1,size=(35,2))
U=np.vstack([rng.normal(0,1,size=(55,2)),rng.normal(3,1,size=(25,2))])
X=np.vstack([P,U]);s=np.r_[np.ones(len(P),dtype=np.int8),np.zeros(len(U),dtype=np.int8)]
result,arrays=estimate_prior(X,s,"two_sample",1234)
assert result["status"]=="success",result
mean=X.mean(0);sd=X.std(0)
reference=source_namespace()["wrapper"]((X[arrays["U_indices"]]-mean)/sd,(X[arrays["P_indices"]]-mean)/sd)
np.testing.assert_array_equal(result["raw_output"],reference)
checks["official_wrapper_KM2_second_output_exact"]=True
# API boundary excludes labels and holds estimator data fixed under diagnostic-label changes.
assert list(inspect.signature(estimate_prior).parameters)==["X_train","S_train","design","seed"]
again,_=estimate_prior(X,s,"two_sample",1234)
assert result==again
checks["deterministic_X_S_only_API"]=True
bad,_=estimate_prior(X[:2],[0,0],"one_sample",1)
assert bad["status"]=="failed" and bad["pi_hat"] is None
badfinite,_=estimate_prior(np.full_like(X,np.nan),s,"two_sample",1)
assert badfinite["status"]=="failed" and badfinite["pi_hat"] is None
constant,_=estimate_prior(np.ones_like(X),s,"two_sample",1)
assert constant["status"]=="failed" and constant["pi_hat"] is None
checks["failure_cases_preserved_without_oracle_fallback"]=3
rec=dict(status="passed",estimator=VERSION,parameters=PARAMS,checks=checks,
         numerical_smoke_output=result,example_failure_records=[bad,badfinite,constant])
out.write_text(json.dumps(rec,indent=2))
print(json.dumps(dict(status="passed",checks=checks),indent=2))
