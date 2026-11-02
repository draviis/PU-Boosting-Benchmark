"""Verified wrapper around the official KM2 functions; X/S only."""
from pathlib import Path
import hashlib, math, sys
import numpy as np
HERE = Path(__file__).resolve().parent
if (HERE / "vendor_local").exists():
    sys.path.insert(0, str(HERE / "vendor_local"))
from cvxopt import matrix, solvers, spmatrix

VERSION = "official-KM2-2016-task6-wrapper-v1"
PARAMS = dict(max_per_group=1000, epsilon=0.04, lambda_upper_bound=8.0,
              bandwidth_multipliers=np.logspace(-1, 1, 5).tolist(),
              standardization="all_training_X", negative_sqrt_tolerance=1e-10)
SOURCE = HERE / "sources/upstream_Kernel_MPE_grad_threshold.py"

def array_hash(a):
    a=np.ascontiguousarray(a)
    h=hashlib.sha256()
    h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()

def source_namespace(diagnostics=None):
    """Preserve author's scientific function bodies and default solver options."""
    text=SOURCE.read_text(encoding="utf-8")
    text=text[:text.index("if __name__")]
    text="\n".join(line for line in text.splitlines() if not line.startswith(
        ("from cvxopt", "from math", "import numpy", "import matplotlib", "plt.close", "import scipy")))
    env=dict(np=np, matrix=matrix, spmatrix=spmatrix, sqrt=math.sqrt, solvers=solvers)
    if diagnostics is not None:
        class CheckedSolvers:
            options=solvers.options
            def coneqp(self,*args,**kwargs):
                result=solvers.coneqp(*args,**kwargs)
                record={key:result.get(key) for key in [
                    "status","gap","relative gap","primal infeasibility","dual infeasibility","iterations"]}
                diagnostics["solver_calls"].append(record)
                if result["status"]!="optimal":
                    raise ValueError("solver_status_"+str(result["status"]))
                return result
        def checked_sqrt(v):
            v=float(v)
            if not math.isfinite(v): raise ValueError("nonfinite_squared_distance")
            if v<0:
                diagnostics["sqrt_safeguards"].append(v)
                if v < -PARAMS["negative_sqrt_tolerance"]:
                    raise ValueError("materially_negative_squared_distance")
                v=0.
            return math.sqrt(v)
        env.update(sqrt=checked_sqrt, solvers=CheckedSolvers())
    exec(compile(text,str(SOURCE),"exec"),env)
    return env

def derive_priors(kappa,nP,nU,design):
    lam=float(nP/(nP+nU)) if nP+nU else float("nan")
    if design=="one_sample":
        pi=lam+(1-lam)*kappa
        c=lam/pi if pi>0 else float("nan")
        pi_u=kappa
    elif design=="two_sample":
        pi=kappa; c=None; pi_u=kappa
    else: raise ValueError("unknown_design")
    admissible=math.isfinite(kappa) and 0<=kappa<1 and math.isfinite(pi) and 0<pi<1
    if design=="one_sample":admissible=admissible and math.isfinite(c) and 0<c<=1
    return dict(lambda_hat=lam,pi_hat_raw=pi,pi_U_hat_raw=pi_u,c_hat_raw=c,
                pi_hat=pi if admissible else None,pi_U_hat=pi_u if admissible else None,
                c_hat=c if admissible else None,priors_admissible=bool(admissible))

def estimate_prior(X_train,S_train,design,seed):
    """No Y, validation data or test data are accepted."""
    d=dict(estimator=VERSION,parameters=PARAMS.copy(),seed=int(seed),
           status="started",failure_reason=None,kappa_raw=None,kappa_usable=None,
           solver_calls=[],sqrt_safeguards=[],raw_output=None)
    arrays={}
    try:
        X=np.asarray(X_train,dtype=np.float64); s=np.asarray(S_train)
        if X.ndim!=2 or s.shape!=(len(X),) or not np.isin(s,[0,1]).all():
            raise ValueError("invalid_X_S_shape_or_labels")
        P=np.flatnonzero(s==1); U=np.flatnonzero(s==0)
        d.update(nP=int(len(P)),nU=int(len(U)),n_features=int(X.shape[1]),
                 training_X_sha256=array_hash(X),training_S_sha256=array_hash(s))
        if min(len(P),len(U))<2:raise ValueError("insufficient_P_or_U")
        if not np.isfinite(X).all():raise ValueError("nonfinite_features")
        mean=X.mean(0); scale=X.std(0)
        scale[scale==0]=1
        arrays.update(scale_mean=mean,scale_sd=scale)
        rng=np.random.default_rng(seed)
        ip=np.sort(rng.choice(P,min(len(P),PARAMS["max_per_group"]),replace=False))
        iu=np.sort(rng.choice(U,min(len(U),PARAMS["max_per_group"]),replace=False))
        arrays.update(P_indices=ip,U_indices=iu)
        xp=(X[ip]-mean)/scale; xu=(X[iu]-mean)/scale
        d.update(kernel_nP=len(ip),kernel_nU=len(iu),P_indices_sha256=array_hash(ip),
                 U_indices_sha256=array_hash(iu),scale_mean_sha256=array_hash(mean),
                 scale_sd_sha256=array_hash(scale))
        env=source_namespace(d)
        original_width=env["compute_best_rbf_kernel_width"]
        def width(a,b):
            w,K=original_width(a,b)
            if not math.isfinite(w) or w<=0:raise ValueError("invalid_kernel_width")
            v=np.r_[np.full(len(a),1/len(a)),np.full(len(b),-1/len(b))]
            sep=float(v@K@v)
            if not math.isfinite(sep) or sep<=0:raise ValueError("zero_or_invalid_RKHS_separation")
            d.update(kernel_width=float(w),RKHS_separation=math.sqrt(sep))
            return w,K
        env["compute_best_rbf_kernel_width"]=width
        KM1,KM2=env["wrapper"](xu,xp)
        d.update(raw_output=[float(KM1),float(KM2)],kappa_raw=float(KM2),
                 lambda_raw=float(1/(1-KM2)))
        d.update(derive_priors(float(KM2),len(P),len(U),design))
        if not d["priors_admissible"]:raise ValueError("inadmissible_derived_priors")
        d.update(kappa_usable=float(KM2),status="success",
                 search_upper_saturation=d["lambda_raw"]>=8.-.04,
                 search_lower_saturation=d["lambda_raw"]<=1.+.04,
                 prior_clipping_applied=False)
    except Exception as exc:
        d.update(status="failed",failure_reason=type(exc).__name__+": "+str(exc),
                 pi_hat=None,pi_U_hat=None,c_hat=None)
    return d,arrays
