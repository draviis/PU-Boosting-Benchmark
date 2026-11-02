
"""
PHASE 1 PU BOOSTING BENCHMARK — ONE-SAMPLE SCAR
================================================

Blocks (27 fitted methods)
--------------------------
Block 0: Supervised oracle references (true Y available for development)
    - Oracle logistic regression
    - Oracle CatBoost

Block 1: Ordinary PU baselines
    - uPU logistic
    - nnPU logistic
    - Elkan-Noto / probability correction with known c

Block 2: PU strategy × standard boosting engine
    PU strategies:
        - Naive U-as-negative
        - Probability correction (known c)
        - Reliable-negative (simple quantile RN selector)
        - Cost-sensitive PU
    Boosting engines:
        - AdaBoost
        - Gradient Boosting (GBM)
        - XGBoost
        - LightGBM
        - CatBoost

    Total = 4 × 5 = 20 combinations

Block 3: Native PU boosting
    - AdaPU
    - PU Gradient Boosting

Total = 2 supervised references + 3 ordinary baselines
        + 20 wrapper/engine combinations + 2 native PU methods
        = 27 methods. Oracle methods are excluded from PU-only ranks.

IMPORTANT DESIGN RULES
----------------------
1. d_train and d_val expose only X and S to model fitting / tuning.
2. y_train_true and y_val_true are retained ONLY for simulation diagnostics.
3. d_test is fully labelled and is used ONLY once for final evaluation.
4. pi and c are assumed KNOWN in this first experiment.
5. SCAR: P(S=1 | Y=1, X)=c and P(S=1 | Y=0, X)=0.
6. For one-sample PU, the positive prevalence inside the unlabeled subset is

       pi_U = pi(1-c) / (1-pi*c).

   AdaPU is therefore supplied pi_U, because it is fitted to P vs U.

INSTALL ONCE
------------
pip install numpy pandas scipy scikit-learn torch xgboost lightgbm catboost

AdaPU:
    clone the authors' Ada-PU repository,
    compile decisionstump.pyx, and set ADAPU_SRC below.

Notes
-----
- "XGB" here means XGBoost.
- The RN implementation below is deliberately a simple, transparent
  benchmark RN strategy based on a lower quantile of a preliminary P-vs-U
  score. It is NOT intended to represent every published RN selector.
"""

# ============================================================
# 0. Imports and configuration
# ============================================================

import argparse
import hashlib
import os
import sys
import copy
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    precision_recall_curve,
)

import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent


def _positive_int_from_environment(name, default=1):
    """Read a positive integer environment variable with a safe fallback."""
    try:
        return max(1, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


# Slurm sets SLURM_CPUS_PER_TASK. PU_BENCHMARK_N_JOBS is also convenient
# outside Slurm and takes precedence when it is explicitly provided.
N_JOBS = _positive_int_from_environment(
    "PU_BENCHMARK_N_JOBS",
    _positive_int_from_environment("SLURM_CPUS_PER_TASK", 1),
)
torch.set_num_threads(N_JOBS)

# Third-party boosting engines
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

from sklearn.tree import DecisionTreeRegressor

# import copy
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# ------------------------------------------------------------
# AdaPU can be configured through ADAPU_SRC on the cluster. The relative
# candidates make the common layout PUBenchmark/ and Ada-PU/ portable.
# ------------------------------------------------------------
_adapu_environment = os.environ.get("ADAPU_SRC")
_adapu_candidates = [
    SCRIPT_DIR.parent / "Ada-PU" / "src",
    SCRIPT_DIR.parent / "Ada-PU" / "Ada-PU" / "src",
    # Standard Phase 1 layout:
    #   <work-root>/PUBenchmark/phase1_pilot_28_scenarios/code/
    #   <work-root>/Ada-PU/src/
    SCRIPT_DIR.parents[2] / "Ada-PU" / "src",
    SCRIPT_DIR.parents[2] / "Ada-PU" / "Ada-PU" / "src",
]

if _adapu_environment:
    ADAPU_SRC = str(Path(_adapu_environment).expanduser().resolve())
else:
    _existing_adapu_candidates = [
        Path(candidate).resolve()
        for candidate in _adapu_candidates
        if Path(candidate).is_dir()
    ]
    _compiled_adapu_candidates = [
        candidate
        for candidate in _existing_adapu_candidates
        if (
            list(candidate.glob("decisionstump*.so"))
            or list(candidate.glob("decisionstump*.pyd"))
        )
    ]
    ADAPU_SRC = str(next(
        iter(_compiled_adapu_candidates or _existing_adapu_candidates),
        Path(_adapu_candidates[0]).resolve(),
    ))

ADAPU_AVAILABLE = False
if os.path.isdir(ADAPU_SRC):
    if ADAPU_SRC not in sys.path:
        sys.path.insert(0, ADAPU_SRC)
    try:
        from adapu import AdaBoost_PU
        ADAPU_AVAILABLE = True
    except Exception as e:
        print("AdaPU could not be imported:", repr(e))


RANDOM_SEED = 123


# ============================================================
# 1. Data structures
# ============================================================

@dataclass
class PUSplit:
    X: np.ndarray
    y_true: np.ndarray
    s: np.ndarray


@dataclass
class PUExperiment:
    train: PUSplit
    val: PUSplit
    test: PUSplit

    pi: float
    c: float
    p: int
    delta: float
    delta_sq: float
    signal: float
    n_signal: int
    rho: float
    seed: int

    @property
    def pi_u(self):
        """
        Positive prevalence among S=0 observations under one-sample SCAR.
        """
        return self.pi * (1.0 - self.c) / (1.0 - self.pi * self.c)


def experiment_fingerprint(data):
    """Return a stable fingerprint of the exact train/validation/test data."""
    digest = hashlib.sha256()

    for split_name, split in (
        ("train", data.train),
        ("val", data.val),
        ("test", data.test),
    ):
        digest.update(split_name.encode("utf-8"))
        for array in (split.X, split.y_true, split.s):
            contiguous = np.ascontiguousarray(array)
            digest.update(str(contiguous.dtype).encode("ascii"))
            digest.update(str(contiguous.shape).encode("ascii"))
            digest.update(contiguous.tobytes())

    return digest.hexdigest()


# ============================================================
# 2. Synthetic one-sample SCAR generator
# ============================================================

def toeplitz_covariance(p, rho):
    idx = np.arange(p)
    return rho ** np.abs(np.subtract.outer(idx, idx))


def calibrate_mean_vector(p, n_signal, Sigma, delta):
    """Calibrate a sparse mean shift to the requested Mahalanobis distance.

    For direction v=(1,...,1,0,...,0), return

        mu = delta * v / sqrt(v.T @ Sigma^{-1} @ v)

    so that mu.T @ Sigma^{-1} @ mu = delta**2.  This keeps statistical
    separation fixed when p or rho changes.
    """
    if delta <= 0:
        raise ValueError("delta must be positive.")
    if not (1 <= n_signal <= p):
        raise ValueError("n_signal must be between 1 and p.")

    direction = np.zeros(p, dtype=float)
    direction[:n_signal] = 1.0
    solved = np.linalg.solve(Sigma, direction)
    denominator = float(np.sqrt(direction @ solved))
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("Could not calibrate the Mahalanobis mean shift.")
    return delta * direction / denominator


def generate_split(
    n,
    p,
    pi,
    c,
    mu_pos,
    Sigma,
    rng,
):
    """
    Generate one independent population split.

    Y ~ Bernoulli(pi)
    X | Y=0 ~ N(0, Sigma)
    X | Y=1 ~ N(mu_pos, Sigma)

    SCAR:
        S | Y=1 ~ Bernoulli(c)
        S | Y=0 = 0
    """

    y = rng.binomial(1, pi, size=n).astype(np.int64)

    X = np.empty((n, p), dtype=np.float64)

    pos = y == 1
    neg = ~pos

    X[pos] = rng.multivariate_normal(
        mean=mu_pos,
        cov=Sigma,
        size=pos.sum()
    )

    X[neg] = rng.multivariate_normal(
        mean=np.zeros(p),
        cov=Sigma,
        size=neg.sum()
    )

    s = np.zeros(n, dtype=np.int64)
    s[pos] = rng.binomial(1, c, size=pos.sum()).astype(np.int64)

    return PUSplit(X=X, y_true=y, s=s)


def generate_one_sample_scar(
    n_train=5000,
    n_val=5000,
    n_test=10000,
    p=10,
    pi=0.20,
    c=0.30,
    delta=0.90,
    n_signal=3,
    rho=0.0,
    seed=123,
):
    """
    Generate independent train/validation/test draws from the SAME
    underlying population and SCAR labeling mechanism.

    y_true is stored for all splits, but train/validation y_true must
    never be used to fit or tune the PU learners.
    """

    if not (0 < pi < 1):
        raise ValueError("pi must be in (0,1).")
    if not (0 < c <= 1):
        raise ValueError("c must be in (0,1].")
    if n_signal > p:
        raise ValueError("n_signal cannot exceed p.")
    if not (0 <= rho < 1):
        raise ValueError("rho must be in [0,1).")

    rng = np.random.default_rng(seed)

    Sigma = toeplitz_covariance(p, rho)

    mu_pos = calibrate_mean_vector(
        p=p,
        n_signal=n_signal,
        Sigma=Sigma,
        delta=delta,
    )
    achieved_delta_sq = float(mu_pos @ np.linalg.solve(Sigma, mu_pos))
    if not np.isclose(achieved_delta_sq, delta ** 2, rtol=1e-10, atol=1e-12):
        raise RuntimeError(
            "Mahalanobis calibration failed: "
            f"requested delta^2={delta ** 2}, achieved={achieved_delta_sq}."
        )

    train = generate_split(
        n_train, p, pi, c, mu_pos, Sigma, rng
    )
    val = generate_split(
        n_val, p, pi, c, mu_pos, Sigma, rng
    )
    test = generate_split(
        n_test, p, pi, c, mu_pos, Sigma, rng
    )

    return PUExperiment(
        train=train,
        val=val,
        test=test,
        pi=pi,
        c=c,
        p=p,
        delta=delta,
        delta_sq=achieved_delta_sq,
        # Retained as an interpretable diagnostic: equal shift applied to
        # each of the n_signal active coordinates after delta calibration.
        signal=float(mu_pos[0]),
        n_signal=n_signal,
        rho=rho,
        seed=seed,
    )


# ============================================================
# 3. Diagnostics
# ============================================================

def split_diagnostics(split, name):
    n = len(split.y_true)

    return {
        "split": name,
        "n": n,
        "n_true_positive": int((split.y_true == 1).sum()),
        "n_true_negative": int((split.y_true == 0).sum()),
        "true_positive_rate": split.y_true.mean(),
        "labelled_rate": split.s.mean(),
        "label_rate_given_positive":
            split.s[split.y_true == 1].mean()
            if np.any(split.y_true == 1) else np.nan,
        "positive_rate_in_U":
            split.y_true[split.s == 0].mean()
            if np.any(split.s == 0) else np.nan,
        "n_labelled_positive": int((split.s == 1).sum()),
        "n_unlabelled": int((split.s == 0).sum()),
    }


def print_experiment_diagnostics(data):
    out = pd.DataFrame([
        split_diagnostics(data.train, "train"),
        split_diagnostics(data.val, "val"),
        split_diagnostics(data.test, "test"),
    ])

    print("\n=== DGP diagnostics ===")
    print(f"pi            = {data.pi:.4f}")
    print(f"c             = {data.c:.4f}")
    print(f"delta         = {data.delta:.4f}")
    print(f"delta^2       = {data.delta_sq:.4f}")
    print(f"active-coordinate shift = {data.signal:.6f}")
    print(f"theoretical pi_U = {data.pi_u:.4f}")
    print(out.to_string(index=False))
    print()


# ============================================================
# 4. Common PU helpers
# ============================================================

def get_P_U(split):
    """
    Return the observed labelled-positive and unlabeled feature matrices.

    No true Y is used.
    """
    Xp = split.X[split.s == 1]
    Xu = split.X[split.s == 0]
    return Xp, Xu


def pu_proxy_auc(s, score):
    """
    PU-only validation ranking criterion.

    It uses observed S only:
       S=1 = labelled positive
       S=0 = unlabeled

    Larger is better.
    """
    if len(np.unique(s)) < 2:
        return np.nan
    return roc_auc_score(s, score)


def prior_quantile_threshold(score_val, pi):
    """
    PU-only hard-classification threshold using known target prevalence pi.

    Choose the score cutoff so approximately pi of validation observations
    are declared positive.

    This avoids using y_val_true.
    """
    return float(np.quantile(score_val, 1.0 - pi))


def clip_prob(prob):
    return np.clip(np.asarray(prob, dtype=float), 1e-7, 1.0 - 1e-7)


def evaluate_test(
    method,
    y_test,
    score_test,
    threshold,
    probability=None,
    extra=None,
):
    """
    Common final test evaluation.

    score_test:
        ranking score; larger = more positive

    threshold:
        chosen from validation only

    probability:
        optional estimate of P(Y=1|X), only supplied when meaningful
    """

    score_test = np.asarray(score_test, dtype=float)
    pred_test = (score_test >= threshold).astype(int)

    result = {
        "method": method,
        "AUROC": roc_auc_score(y_test, score_test),
        "AUPRC": average_precision_score(y_test, score_test),
        "F1": f1_score(y_test, pred_test, zero_division=0),
        "BalancedAccuracy":
            balanced_accuracy_score(y_test, pred_test),
        "MCC": matthews_corrcoef(y_test, pred_test),
        "Brier": np.nan,
        "LogLoss": np.nan,
        "threshold": threshold,
    }

    if probability is not None:
        p = clip_prob(probability)
        result["Brier"] = brier_score_loss(y_test, p)
        result["LogLoss"] = log_loss(y_test, p)

    if extra:
        result.update(extra)

    return result


# ============================================================
# 5. Block 0 — supervised oracle references
# ============================================================

def oracle_f1_threshold(y_val, score_val):
    """Choose an F1-maximising threshold using true oracle validation labels."""
    if len(np.unique(y_val)) < 2:
        raise ValueError("Oracle validation data must contain both classes.")

    precision, recall, thresholds = precision_recall_curve(y_val, score_val)
    if len(thresholds) == 0:
        raise ValueError("No oracle validation thresholds were available.")
    f1_values = (
        2.0 * precision[:-1] * recall[:-1]
        / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    )
    best_index = int(np.nanargmax(f1_values))
    return float(thresholds[best_index]), float(f1_values[best_index])


def fit_oracle_logistic(data, C_grid=(0.1, 1.0, 10.0)):
    """Supervised logistic reference trained/tuned with the latent true Y."""
    best = None
    for C in C_grid:
        model = LogisticRegression(
            C=C,
            max_iter=3000,
            random_state=data.seed,
        )
        model.fit(data.train.X, data.train.y_true)
        val_probability = model.predict_proba(data.val.X)[:, 1]
        val_auc = roc_auc_score(data.val.y_true, val_probability)
        if best is None or val_auc > best["val_auc"]:
            best = {
                "C": C,
                "model": model,
                "val_probability": val_probability,
                "val_auc": val_auc,
            }

    threshold, validation_f1 = oracle_f1_threshold(
        data.val.y_true,
        best["val_probability"],
    )
    X_dev = np.vstack([data.train.X, data.val.X])
    y_dev = np.concatenate([data.train.y_true, data.val.y_true])
    final_model = LogisticRegression(
        C=best["C"],
        max_iter=3000,
        random_state=data.seed,
    )
    final_model.fit(X_dev, y_dev)
    test_probability = final_model.predict_proba(data.test.X)[:, 1]

    return evaluate_test(
        method="Block0 | Oracle logistic regression",
        y_test=data.test.y_true,
        score_test=test_probability,
        threshold=threshold,
        probability=test_probability,
        extra={
            "best_params": {"C": best["C"]},
            "val_metric": best["val_auc"],
            "validation_F1_at_threshold": validation_f1,
            "comparison_role": "supervised_reference",
            "uses_true_y_for_development": True,
        },
    )


def fit_oracle_catboost(data):
    """Supervised CatBoost reference trained/tuned with the latent true Y."""
    if CatBoostClassifier is None:
        raise ImportError("catboost is required for the oracle CatBoost reference.")

    best = None
    for params in ENGINE_GRIDS["CatBoost"]:
        model = make_engine("CatBoost", params, data.seed)
        model.fit(data.train.X, data.train.y_true)
        val_probability = model_score(model, data.val.X)
        val_auc = roc_auc_score(data.val.y_true, val_probability)
        if best is None or val_auc > best["val_auc"]:
            best = {
                "params": dict(params),
                "model": model,
                "val_probability": val_probability,
                "val_auc": val_auc,
            }

    threshold, validation_f1 = oracle_f1_threshold(
        data.val.y_true,
        best["val_probability"],
    )
    X_dev = np.vstack([data.train.X, data.val.X])
    y_dev = np.concatenate([data.train.y_true, data.val.y_true])
    final_model = make_engine("CatBoost", best["params"], data.seed)
    final_model.fit(X_dev, y_dev)
    test_probability = model_score(final_model, data.test.X)

    return evaluate_test(
        method="Block0 | Oracle CatBoost",
        y_test=data.test.y_true,
        score_test=test_probability,
        threshold=threshold,
        probability=test_probability,
        extra={
            "best_params": best["params"],
            "val_metric": best["val_auc"],
            "validation_F1_at_threshold": validation_f1,
            "comparison_role": "supervised_reference",
            "uses_true_y_for_development": True,
        },
    )


# ============================================================
# 6. Block 1 — Elkan-Noto, uPU, nnPU
# ============================================================

# ------------------------------------------------------------
# 6A. Elkan-Noto with KNOWN c
# ------------------------------------------------------------

def fit_elkan_noto_known_c(data, C_grid=(0.1, 1.0, 10.0)):
    """
    With known c under SCAR:

        q(x)   = P(S=1 | X=x)
        eta(x) = P(Y=1 | X=x) = q(x)/c.

    Hyperparameter C is selected using PU-only validation AUC based on S_val.
    """

    best = None

    for C in C_grid:
        model = LogisticRegression(
            C=C,
            max_iter=3000,
            random_state=data.seed,
        )
        model.fit(data.train.X, data.train.s)

        q_val = model.predict_proba(data.val.X)[:, 1]
        val_score = pu_proxy_auc(data.val.s, q_val)

        if best is None or val_score > best["val_score"]:
            best = {
                "C": C,
                "val_score": val_score,
                "model": model,
            }

    # Refit on train + validation with chosen hyperparameter
    X_dev = np.vstack([data.train.X, data.val.X])
    s_dev = np.concatenate([data.train.s, data.val.s])

    final_model = LogisticRegression(
        C=best["C"],
        max_iter=3000,
        random_state=data.seed,
    )
    final_model.fit(X_dev, s_dev)

    # Threshold is learned from validation predictions from the selected
    # train-only model, NOT from test.
    q_val = best["model"].predict_proba(data.val.X)[:, 1]
    eta_val = clip_prob(q_val / data.c)
    threshold = prior_quantile_threshold(eta_val, data.pi)

    q_test = final_model.predict_proba(data.test.X)[:, 1]
    eta_test = clip_prob(q_test / data.c)

    return evaluate_test(
        method="Block1 | Elkan-Noto known-c",
        y_test=data.test.y_true,
        score_test=eta_test,
        threshold=threshold,
        probability=eta_test,
        extra={
            "best_params": {"C": best["C"]},
            "val_metric": best["val_score"],
        },
    )


# ============================================================
# 6B. uPU / nnPU linear PU-risk baselines
# ============================================================



# ------------------------------------------------------------
# Linear scoring model
# ------------------------------------------------------------

class LinearPUModel(nn.Module):
    """
    Linear decision function

        f(x) = beta_0 + x^T beta
    """

    def __init__(self, p):
        super().__init__()
        self.linear = nn.Linear(p, 1)

    def forward(self, X):
        return self.linear(X).squeeze(1)


# ------------------------------------------------------------
# One-sample SCAR PU risk components
# ------------------------------------------------------------

def one_sample_pu_components_torch(
    model,
    Xp,
    Xu,
    pi,
    c,
):
    """
    Compute the components of the one-sample SCAR PU risk.

    Population positive prior:
        pi = P(Y=1)

    SCAR labeling probability:
        c = P(S=1 | Y=1)

    Positive prevalence in the unlabeled subset:

        pi_U = P(Y=1 | S=0)
             = pi(1-c) / (1-pi*c)

    Logistic losses:

        R_P^+ = E_P[ log(1 + exp(-f(X))) ]

        R_P^- = E_P[ log(1 + exp(f(X))) ]

        R_U^- = E_U[ log(1 + exp(f(X))) ]

    Since

        R_U^- = pi_U R_P^- + (1-pi_U) R_N^-

    we obtain

        R_N^-
        =
        [R_U^- - pi_U R_P^-] / (1-pi_U).

    Therefore the original population PN risk is

        R
        =
        pi R_P^+
        +
        (1-pi)/(1-pi_U)
        [R_U^- - pi_U R_P^-].
    """

    pi_u = (
        pi * (1.0 - c)
        /
        (1.0 - pi * c)
    )

    # Model scores
    fp = model(Xp)
    fu = model(Xu)

    # Positive observations evaluated as positive
    Rp_pos = F.softplus(-fp).mean()

    # Positive observations evaluated as negative
    Rp_neg = F.softplus(fp).mean()

    # Unlabeled observations evaluated as negative
    Ru_neg = F.softplus(fu).mean()

    # This estimates:
    #     (1 - pi_U) R_N^-
    raw_negative_component = (
        Ru_neg
        -
        pi_u * Rp_neg
    )

    # Rescaling needed to recover the contribution
    # of negatives to the ORIGINAL population PN risk.
    negative_scale = (
        (1.0 - pi)
        /
        (1.0 - pi_u)
    )

    positive_component = (
        pi * Rp_pos
    )

    negative_component = (
        negative_scale
        * raw_negative_component
    )

    return {
        "pi_u": pi_u,
        "Rp+": Rp_pos,
        "Rp-": Rp_neg,
        "Ru-": Ru_neg,

        # Before one-sample scaling
        "raw_negative_component":
            raw_negative_component,

        # Contribution on original PN-risk scale
        "negative_component":
            negative_component,

        "positive_component":
            positive_component,
    }


# ------------------------------------------------------------
# uPU objective
# ------------------------------------------------------------

def one_sample_upu_objective(
    model,
    Xp,
    Xu,
    pi,
    c,
):
    """
    One-sample SCAR uPU objective.

        R_uPU
        =
        positive_component
        +
        negative_component
    """

    comp = one_sample_pu_components_torch(
        model=model,
        Xp=Xp,
        Xu=Xu,
        pi=pi,
        c=c,
    )

    objective = (
        comp["positive_component"]
        +
        comp["negative_component"]
    )

    return objective, comp


# ------------------------------------------------------------
# Kiryo-style nnPU training objective
# ------------------------------------------------------------

def one_sample_nnpu_training_objective(
    model,
    Xp,
    Xu,
    pi,
    c,
    beta=0.0,
    gamma=1.0,
):
    """
    nnPU optimization with Kiryo-style gradient-ascent correction.

    Let

        R_N_hat

    denote the estimated negative-risk contribution.

    If

        R_N_hat >= -beta,

    minimize the ordinary PU risk:

        R_P + R_N_hat.

    If

        R_N_hat < -beta,

    do NOT simply clamp it to zero.

    Instead minimize

        -gamma * R_N_hat,

    which is equivalent to gradient ascent on R_N_hat
    and pushes the estimated negative risk upward.

    Parameters
    ----------
    beta : float
        Non-negative correction threshold.
        Default = 0.

    gamma : float
        Strength of gradient-ascent correction.
        Default = 1.
    """

    comp = one_sample_pu_components_torch(
        model=model,
        Xp=Xp,
        Xu=Xu,
        pi=pi,
        c=c,
    )

    positive_component = comp[
        "positive_component"
    ]

    negative_component = comp[
        "negative_component"
    ]

    # Kiryo-style correction
    if negative_component.item() < -beta:

        objective = (
            -gamma
            * negative_component
        )

        correction_active = True

    else:

        objective = (
            positive_component
            +
            negative_component
        )

        correction_active = False

    return (
        objective,
        comp,
        correction_active
    )


# ------------------------------------------------------------
# Validation risk for uPU
# ------------------------------------------------------------

@torch.no_grad()
def evaluate_upu_risk(
    model,
    Xp,
    Xu,
    pi,
    c,
):
    """
    Validation uPU risk.
    """

    objective, comp = one_sample_upu_objective(
        model=model,
        Xp=Xp,
        Xu=Xu,
        pi=pi,
        c=c,
    )

    return (
        float(objective.item()),
        {
            k:
            (
                float(v.item())
                if torch.is_tensor(v)
                else float(v)
            )
            for k, v in comp.items()
        }
    )


# ------------------------------------------------------------
# Validation criterion for nnPU
# ------------------------------------------------------------

@torch.no_grad()
def evaluate_nnpu_risk(
    model,
    Xp,
    Xu,
    pi,
    c,
):
    """
    Non-negative validation criterion.

    For MODEL SELECTION we use

        positive_component
        +
        max(0, negative_component).

    This is the nnPU empirical-risk estimator.

    Notice:
        training uses the Kiryo gradient-ascent correction,
        whereas validation uses the non-negative risk itself.
    """

    comp = one_sample_pu_components_torch(
        model=model,
        Xp=Xp,
        Xu=Xu,
        pi=pi,
        c=c,
    )

    positive_component = comp[
        "positive_component"
    ]

    negative_component = comp[
        "negative_component"
    ]

    risk = (
        positive_component
        +
        torch.clamp(
            negative_component,
            min=0.0
        )
    )

    return (
        float(risk.item()),
        {
            k:
            (
                float(v.item())
                if torch.is_tensor(v)
                else float(v)
            )
            for k, v in comp.items()
        }
    )


# ------------------------------------------------------------
# Main fitting function
# ------------------------------------------------------------

def fit_pu_logistic_baseline(
    data,
    non_negative=False,

    # nnPU correction parameters
    beta=0.0,
    gamma=1.0,

    # tuning
    l2_grid=(
        0.0,
        1e-4,
        1e-3,
        1e-2
    ),

    lr=0.01,
    max_epochs=1500,
    check_every=20,
    patience_checks=15,
):
    """
    Fit either uPU or nnPU linear classifier.

    Parameters
    ----------
    data
        PUExperiment object.

    non_negative
        False -> uPU
        True  -> nnPU

    beta, gamma
        Used only for nnPU.

    Model selection
    ---------------
    - training uses d_train
    - hyperparameters / early stopping use d_val
    - y_val_true is NEVER used
    - final evaluation uses d_test
    """

    torch.manual_seed(data.seed)
    np.random.seed(data.seed)

    # --------------------------------------------------------
    # Extract observed P and U
    # --------------------------------------------------------

    Xp_train_np, Xu_train_np = get_P_U(
        data.train
    )

    Xp_val_np, Xu_val_np = get_P_U(
        data.val
    )

    Xp_train = torch.tensor(
        Xp_train_np,
        dtype=torch.float32
    )

    Xu_train = torch.tensor(
        Xu_train_np,
        dtype=torch.float32
    )

    Xp_val = torch.tensor(
        Xp_val_np,
        dtype=torch.float32
    )

    Xu_val = torch.tensor(
        Xu_val_np,
        dtype=torch.float32
    )

    candidates = []

    # ========================================================
    # Hyperparameter tuning
    # ========================================================

    for l2 in l2_grid:

        torch.manual_seed(data.seed)

        model = LinearPUModel(
            data.p
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=l2,
        )

        best_state = copy.deepcopy(
            model.state_dict()
        )

        best_val_risk = np.inf
        best_epoch = 0

        no_improve = 0

        # Track how often nnPU correction activates
        correction_count = 0

        # ====================================================
        # Training epochs
        # ====================================================

        for epoch in range(
            1,
            max_epochs + 1
        ):

            model.train()
            optimizer.zero_grad()

            # ----------------------------------------------
            # uPU
            # ----------------------------------------------
            if not non_negative:

                objective, comp = (
                    one_sample_upu_objective(
                        model=model,
                        Xp=Xp_train,
                        Xu=Xu_train,
                        pi=data.pi,
                        c=data.c,
                    )
                )

                correction_active = False

            # ----------------------------------------------
            # nnPU
            # ----------------------------------------------
            else:

                (
                    objective,
                    comp,
                    correction_active
                ) = (
                    one_sample_nnpu_training_objective(
                        model=model,
                        Xp=Xp_train,
                        Xu=Xu_train,
                        pi=data.pi,
                        c=data.c,
                        beta=beta,
                        gamma=gamma,
                    )
                )

                if correction_active:
                    correction_count += 1

            # ----------------------------------------------
            # Parameter update
            # ----------------------------------------------

            objective.backward()
            optimizer.step()

            # =================================================
            # Validation / early stopping
            # =================================================

            if epoch % check_every == 0:

                model.eval()

                if not non_negative:

                    val_risk, _ = (
                        evaluate_upu_risk(
                            model=model,
                            Xp=Xp_val,
                            Xu=Xu_val,
                            pi=data.pi,
                            c=data.c,
                        )
                    )

                else:

                    val_risk, _ = (
                        evaluate_nnpu_risk(
                            model=model,
                            Xp=Xp_val,
                            Xu=Xu_val,
                            pi=data.pi,
                            c=data.c,
                        )
                    )

                # ------------------------------------------
                # Early stopping
                # ------------------------------------------

                if (
                    val_risk
                    <
                    best_val_risk - 1e-8
                ):

                    best_val_risk = val_risk
                    best_epoch = epoch

                    best_state = copy.deepcopy(
                        model.state_dict()
                    )

                    no_improve = 0

                else:

                    no_improve += 1

                if (
                    no_improve
                    >= patience_checks
                ):
                    break

        # ----------------------------------------------------
        # Restore best validation state
        # ----------------------------------------------------

        model.load_state_dict(
            best_state
        )

        candidates.append({
            "l2":
                l2,

            "best_epoch":
                best_epoch,

            "val_risk":
                best_val_risk,

            "state":
                copy.deepcopy(
                    model.state_dict()
                ),

            "correction_count":
                correction_count,
        })

    # ========================================================
    # Choose hyperparameter configuration
    # ========================================================

    best = min(
        candidates,
        key=lambda z: z["val_risk"]
    )

    # ========================================================
    # Refit using train + validation
    # ========================================================

    X_dev = np.vstack([
        data.train.X,
        data.val.X
    ])

    s_dev = np.concatenate([
        data.train.s,
        data.val.s
    ])

    Xp_dev = torch.tensor(
        X_dev[s_dev == 1],
        dtype=torch.float32
    )

    Xu_dev = torch.tensor(
        X_dev[s_dev == 0],
        dtype=torch.float32
    )

    torch.manual_seed(
        data.seed
    )

    final_model = LinearPUModel(
        data.p
    )

    optimizer = torch.optim.Adam(
        final_model.parameters(),
        lr=lr,
        weight_decay=best["l2"],
    )

    final_epochs = max(
        best["best_epoch"],
        check_every
    )

    final_correction_count = 0

    # --------------------------------------------------------
    # Final fitting
    # --------------------------------------------------------

    for epoch in range(
        final_epochs
    ):

        final_model.train()
        optimizer.zero_grad()

        if not non_negative:

            objective, comp = (
                one_sample_upu_objective(
                    model=final_model,
                    Xp=Xp_dev,
                    Xu=Xu_dev,
                    pi=data.pi,
                    c=data.c,
                )
            )

        else:

            (
                objective,
                comp,
                correction_active
            ) = (
                one_sample_nnpu_training_objective(
                    model=final_model,
                    Xp=Xp_dev,
                    Xu=Xu_dev,
                    pi=data.pi,
                    c=data.c,
                    beta=beta,
                    gamma=gamma,
                )
            )

            if correction_active:
                final_correction_count += 1

        objective.backward()
        optimizer.step()

    final_model.eval()

    # ========================================================
    # Reconstruct validation-selected model
    # ========================================================

    selected_model = LinearPUModel(
        data.p
    )

    selected_model.load_state_dict(
        best["state"]
    )

    selected_model.eval()

    # ========================================================
    # Predictions
    # ========================================================

    with torch.no_grad():

        X_val_t = torch.tensor(
            data.val.X,
            dtype=torch.float32
        )

        X_test_t = torch.tensor(
            data.test.X,
            dtype=torch.float32
        )

        # Validation ranking score
        score_val = (
            selected_model(
                X_val_t
            )
            .cpu()
            .numpy()
        )

        # Final test score
        score_test = (
            final_model(
                X_test_t
            )
            .cpu()
            .numpy()
        )

        # Logistic probability transformation
        prob_test = (
            torch.sigmoid(
                final_model(
                    X_test_t
                )
            )
            .cpu()
            .numpy()
        )

    # --------------------------------------------------------
    # PU-valid hard classification threshold
    # --------------------------------------------------------

    threshold = (
        prior_quantile_threshold(
            score_val,
            data.pi
        )
    )

    # --------------------------------------------------------
    # Final risk diagnostics
    # --------------------------------------------------------

    if not non_negative:

        _, final_components = (
            evaluate_upu_risk(
                model=final_model,
                Xp=Xp_dev,
                Xu=Xu_dev,
                pi=data.pi,
                c=data.c,
            )
        )

        method_name = (
            "uPU logistic"
        )

    else:

        _, final_components = (
            evaluate_nnpu_risk(
                model=final_model,
                Xp=Xp_dev,
                Xu=Xu_dev,
                pi=data.pi,
                c=data.c,
            )
        )

        method_name = (
            "nnPU logistic"
        )

    # ========================================================
    # Final test evaluation
    # ========================================================

    return evaluate_test(
        method=(
            f"Block1 | {method_name}"
        ),

        y_test=data.test.y_true,

        score_test=score_test,

        threshold=threshold,

        probability=prob_test,

        extra={

            "best_params": {
                "l2":
                    best["l2"],

                "epochs":
                    final_epochs,

                "beta":
                    beta
                    if non_negative
                    else np.nan,

                "gamma":
                    gamma
                    if non_negative
                    else np.nan,
            },

            # Convert risk into larger-is-better display metric
            "val_metric":
                -best["val_risk"],

            # Extremely useful diagnostic:
            # if zero, nnPU behaved exactly like uPU
            "nnpu_correction_count":
                final_correction_count
                if non_negative
                else 0,

            "pi_u":
                final_components[
                    "pi_u"
                ],

            "final_negative_component":
                final_components[
                    "negative_component"
                ],

            "final_raw_negative_component":
                final_components[
                    "raw_negative_component"
                ],
        },
    )


# ============================================================
# 7. Standard boosting engine factory
# ============================================================

ENGINE_GRIDS = {
    "AdaBoost": [
        {"n_estimators": 100, "learning_rate": 0.05, "depth": 1},
        {"n_estimators": 200, "learning_rate": 0.05, "depth": 1},
        {"n_estimators": 100, "learning_rate": 0.10, "depth": 2},
        {"n_estimators": 200, "learning_rate": 0.10, "depth": 2},
    ],

    "GBM": [
        {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 2},
        {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 2},
        {"n_estimators": 100, "learning_rate": 0.10, "max_depth": 3},
        {"n_estimators": 200, "learning_rate": 0.10, "max_depth": 3},
    ],

    "XGBoost": [
        {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 2},
        {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 2},
        {"n_estimators": 100, "learning_rate": 0.10, "max_depth": 3},
        {"n_estimators": 200, "learning_rate": 0.10, "max_depth": 3},
    ],

    "LightGBM": [
        {"n_estimators": 100, "learning_rate": 0.05, "num_leaves": 7},
        {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 7},
        {"n_estimators": 100, "learning_rate": 0.10, "num_leaves": 15},
        {"n_estimators": 200, "learning_rate": 0.10, "num_leaves": 15},
    ],

    "CatBoost": [
        {"iterations": 100, "learning_rate": 0.05, "depth": 4},
        {"iterations": 200, "learning_rate": 0.05, "depth": 4},
        {"iterations": 100, "learning_rate": 0.10, "depth": 6},
        {"iterations": 200, "learning_rate": 0.10, "depth": 6},
    ],
}


def make_engine(engine_name, params, seed):
    if engine_name == "AdaBoost":
        stump = DecisionTreeClassifier(
            max_depth=params["depth"],
            random_state=seed,
        )

        return AdaBoostClassifier(
            estimator=stump,
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            random_state=seed,
        )

    if engine_name == "GBM":
        return GradientBoostingClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            random_state=seed,
        )

    if engine_name == "XGBoost":
        if XGBClassifier is None:
            raise ImportError(
                "xgboost is not installed. Run: pip install xgboost"
            )

        return XGBClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=N_JOBS,
        )

    if engine_name == "LightGBM":
        if LGBMClassifier is None:
            raise ImportError(
                "lightgbm is not installed. Run: pip install lightgbm"
            )

        return LGBMClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            num_leaves=params["num_leaves"],
            random_state=seed,
            verbosity=-1,
            n_jobs=N_JOBS,
        )

    if engine_name == "CatBoost":
        if CatBoostClassifier is None:
            raise ImportError(
                "catboost is not installed. Run: pip install catboost"
            )

        return CatBoostClassifier(
            iterations=params["iterations"],
            learning_rate=params["learning_rate"],
            depth=params["depth"],
            loss_function="Logloss",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=N_JOBS,
        )

    raise ValueError(f"Unknown engine: {engine_name}")


def model_score(model, X):
    """
    Return probability-like ranking score.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):
        return model.decision_function(X)

    return model.predict(X).astype(float)


# ============================================================
# 8. Block 2 PU strategy helpers
# ============================================================

def fit_preliminary_pu_scorer(X, s, seed):
    """
    Simple logistic P-vs-U model used only for RN selection.
    """
    model = LogisticRegression(
        C=1.0,
        max_iter=3000,
        random_state=seed,
    )
    model.fit(X, s)
    return model


def select_reliable_negatives_spy(
    X,
    s,
    spy_fraction=0.15,
    spy_quantile=0.05,
    n_repeats=10,
    selection_frequency=0.70,
    seed=123
):
    """
    Spy-based reliable-negative selection.

    Parameters
    ----------
    X : ndarray
        Feature matrix.

    s : ndarray
        Observed PU label:
            1 = labelled positive
            0 = unlabeled

    spy_fraction : float
        Fraction of labelled positives temporarily moved into U.

    spy_quantile : float
        Lower quantile of spy scores used as RN threshold.

    n_repeats : int
        Number of independent spy repetitions.

    selection_frequency : float
        U observation must be selected as RN in at least this
        fraction of repetitions.

    seed : int
        Random seed.

    Returns
    -------
    rn_mask : boolean ndarray
        True for observations selected as reliable negatives.
    """

    rng = np.random.default_rng(seed)

    pos_idx = np.where(s == 1)[0]
    unl_idx = np.where(s == 0)[0]

    n_pos = len(pos_idx)

    if n_pos < 5:
        raise ValueError(
            "Too few labelled positives for Spy RN selection."
        )

    n_spy = max(
        1,
        int(np.floor(spy_fraction * n_pos))
    )

    rn_counts = np.zeros(len(X), dtype=int)

    for b in range(n_repeats):

        # ----------------------------------------------
        # 1. Select spies from known positives
        # ----------------------------------------------
        spy_idx = rng.choice(
            pos_idx,
            size=n_spy,
            replace=False
        )

        remain_pos_idx = np.setdiff1d(
            pos_idx,
            spy_idx
        )

        # ----------------------------------------------
        # 2. Build temporary P-vs-U* problem
        # ----------------------------------------------
        X_temp = np.vstack([
            X[remain_pos_idx],
            X[unl_idx],
            X[spy_idx]
        ])

        y_temp = np.concatenate([
            np.ones(
                len(remain_pos_idx),
                dtype=int
            ),
            np.zeros(
                len(unl_idx) + len(spy_idx),
                dtype=int
            )
        ])

        # ----------------------------------------------
        # 3. Preliminary classifier
        # ----------------------------------------------
        model = LogisticRegression(
            max_iter=3000,
            random_state=seed + b
        )

        model.fit(
            X_temp,
            y_temp
        )

        # ----------------------------------------------
        # 4. Obtain scores for spies and U
        # ----------------------------------------------
        spy_scores = model.predict_proba(
            X[spy_idx]
        )[:, 1]

        unl_scores = model.predict_proba(
            X[unl_idx]
        )[:, 1]

        # ----------------------------------------------
        # 5. Spy-derived threshold
        # ----------------------------------------------
        threshold = np.quantile(
            spy_scores,
            spy_quantile
        )

        # U examples below threshold are candidate RN
        selected_unl = unl_idx[
            unl_scores < threshold
        ]

        rn_counts[selected_unl] += 1

    # --------------------------------------------------
    # 6. Aggregate repetitions
    # --------------------------------------------------
    required_count = int(
        np.ceil(
            selection_frequency
            * n_repeats
        )
    )

    rn_mask = (
        rn_counts >= required_count
    )

    # Only U can be reliable negatives
    rn_mask[s == 1] = False

    return rn_mask

def prepare_strategy_training_data(
    split,
    strategy,
    strategy_param,
    seed,
):
    """
    Return X_fit, y_fit, sample_weight for the selected wrapper strategy.
    """

    X = split.X
    s = split.s

    # ------------------------------------------
    # Naive and probability correction
    # ------------------------------------------
    if strategy in ("Naive", "ProbCorrection"):
        return X, s, None

    # ------------------------------------------
    # Cost-sensitive
    # ------------------------------------------
    if strategy == "CostSensitive":
        w_u = strategy_param["w_u"]

        weights = np.where(
            s == 1,
            1.0,
            w_u,
        ).astype(float)

        return X, s, weights

    # ------------------------------------------
    # Reliable-negative
    # ------------------------------------------
    if strategy == "RN":
    
        rn_mask = select_reliable_negatives_spy(
            X,
            s,
            spy_fraction=strategy_param["spy_fraction"],
            spy_quantile=strategy_param["spy_quantile"],
            n_repeats=10,
            selection_frequency=0.70,
            seed=seed
        )
    
        p_mask = s == 1
    
        X_fit = np.vstack([
            X[p_mask],
            X[rn_mask]
        ])
    
        y_fit = np.concatenate([
            np.ones(
                p_mask.sum(),
                dtype=int
            ),
            np.zeros(
                rn_mask.sum(),
                dtype=int
            )
        ])
    
        return X_fit, y_fit, None

    raise ValueError(f"Unknown strategy: {strategy}")


STRATEGY_GRIDS = {
    "Naive": [{}],

    "ProbCorrection": [{}],

    "RN": [
        {
            "spy_fraction": 0.10,
            "spy_quantile": 0.01
        },
        {
            "spy_fraction": 0.10,
            "spy_quantile": 0.05
        },
        {
            "spy_fraction": 0.20,
            "spy_quantile": 0.01
        },
        {
            "spy_fraction": 0.20,
            "spy_quantile": 0.05
        }
    ],

    "CostSensitive": [
        {"w_u": 0.10},
        {"w_u": 0.25},
        {"w_u": 0.50},
        {"w_u": 1.00},
    ],
}


def fit_one_boosting_cell(
    data,
    strategy,
    engine_name,
):
    """
    Tune one PU-strategy × boosting-engine cell using d_val only.

    Selection metric:
        PU proxy AUROC = AUC(S_val, score_val).

    After choosing hyperparameters:
        refit on train + validation
        evaluate once on test Y.
    """

    best = None

    for strategy_param in STRATEGY_GRIDS[strategy]:
        for engine_param in ENGINE_GRIDS[engine_name]:

            X_fit, y_fit, weights = prepare_strategy_training_data(
                data.train,
                strategy,
                strategy_param,
                seed=data.seed,
            )

            model = make_engine(
                engine_name,
                engine_param,
                seed=data.seed,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                if weights is None:
                    model.fit(X_fit, y_fit)
                else:
                    model.fit(
                        X_fit,
                        y_fit,
                        sample_weight=weights,
                    )

            score_val_raw = model_score(
                model,
                data.val.X,
            )

            # Probability correction changes scale but not ranking.
            if strategy == "ProbCorrection":
                score_val = clip_prob(
                    score_val_raw / data.c
                )
            else:
                score_val = score_val_raw

            val_auc = pu_proxy_auc(
                data.val.s,
                score_val,
            )

            candidate = {
                "strategy_param": strategy_param,
                "engine_param": engine_param,
                "val_auc": val_auc,
                "train_model": model,
                "score_val": score_val,
            }

            if (
                best is None
                or candidate["val_auc"] > best["val_auc"]
            ):
                best = candidate

    # Validation-derived threshold BEFORE refit
    threshold = prior_quantile_threshold(
        best["score_val"],
        data.pi,
    )

    # --------------------------------------------------------
    # Refit on train + validation using selected parameters
    # --------------------------------------------------------
    dev_split = PUSplit(
        X=np.vstack([
            data.train.X,
            data.val.X,
        ]),
        y_true=np.concatenate([
            data.train.y_true,
            data.val.y_true,
        ]),  # stored but NOT used below
        s=np.concatenate([
            data.train.s,
            data.val.s,
        ]),
    )

    X_dev_fit, y_dev_fit, weights_dev = (
        prepare_strategy_training_data(
            dev_split,
            strategy,
            best["strategy_param"],
            seed=data.seed,
        )
    )

    final_model = make_engine(
        engine_name,
        best["engine_param"],
        seed=data.seed,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if weights_dev is None:
            final_model.fit(
                X_dev_fit,
                y_dev_fit,
            )
        else:
            final_model.fit(
                X_dev_fit,
                y_dev_fit,
                sample_weight=weights_dev,
            )

    score_test_raw = model_score(
        final_model,
        data.test.X,
    )

    probability = None

    if strategy == "ProbCorrection":
        score_test = clip_prob(
            score_test_raw / data.c
        )
        probability = score_test
    else:
        score_test = score_test_raw

    result = evaluate_test(
        method=f"Block2 | {strategy} + {engine_name}",
        y_test=data.test.y_true,
        score_test=score_test,
        threshold=threshold,
        probability=probability,
        extra={
            "best_params": {
                **best["strategy_param"],
                **best["engine_param"],
            },
            "val_metric": best["val_auc"],
        },
    )

    return result


def run_block2(data):
    """
    Run all 20 wrapper × engine combinations.
    """

    results = []

    strategies = [
        "Naive",
        "ProbCorrection",
        "RN",
        "CostSensitive",
    ]

    engines = [
        "AdaBoost",
        "GBM",
        "XGBoost",
        "LightGBM",
        "CatBoost",
    ]

    for strategy in strategies:
        for engine in engines:
            print(
                f"Running Block 2: {strategy} + {engine}"
            )

            try:
                res = fit_one_boosting_cell(
                    data,
                    strategy,
                    engine,
                )
                results.append(res)

            except Exception as e:
                print(
                    f"FAILED: {strategy} + {engine}: {e}"
                )

                results.append({
                    "method":
                        f"Block2 | {strategy} + {engine}",
                    "AUROC": np.nan,
                    "AUPRC": np.nan,
                    "F1": np.nan,
                    "BalancedAccuracy": np.nan,
                    "MCC": np.nan,
                    "Brier": np.nan,
                    "LogLoss": np.nan,
                    "threshold": np.nan,
                    "best_params": {"error": repr(e)},
                    "val_metric": np.nan,
                })

    return results


# ============================================================
# 9. Block 3 — AdaPU
# ============================================================

def fit_adapu_one_sample(
    data,
    beta_grid=(1e-5, 1e-4, 1e-3),
    n_estimators_grid=(50, 100, 200),
    random_selection=1,
):
    """
    AdaPU tuned using PU-only validation AUC.

    Important:
        AdaPU is fitted to observed P and U.
        In a one-sample SCAR split the positive fraction inside U is pi_U,
        not the population prevalence pi.

    Therefore:
        prior = data.pi_u.

    AdaPU's Cython decision stump expects float32.
    """

    if not ADAPU_AVAILABLE:
        raise RuntimeError(
            "AdaPU is unavailable. Check ADAPU_SRC and compile "
            "decisionstump.pyx first."
        )

    Xp_train, Xu_train = get_P_U(data.train)

    Xp_train = Xp_train.astype(np.float32)
    Xu_train = Xu_train.astype(np.float32)

    X_val = data.val.X.astype(np.float32)
    X_test = data.test.X.astype(np.float32)

    best = None

    for beta in beta_grid:
        for n_estimators in n_estimators_grid:

            try:
                model = AdaBoost_PU(
                    n_estimators=n_estimators,
                    prior=data.pi_u,
                    random_selection=random_selection,
                    beta=beta,
                )

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(
                        Xp_train,
                        Xu_train,
                    )

                n_fitted = len(model.estimators)

                if n_fitted == 0:
                    continue

                score_val = model.staged_predict(
                    X_val,
                    n_fitted,
                    test=False,
                )

                val_auc = pu_proxy_auc(
                    data.val.s,
                    score_val,
                )

                candidate = {
                    "beta": beta,
                    "requested_estimators": n_estimators,
                    "n_fitted": n_fitted,
                    "val_auc": val_auc,
                    "model": model,
                    "score_val": score_val,
                }

                if (
                    best is None
                    or candidate["val_auc"] > best["val_auc"]
                ):
                    best = candidate

            except Exception as e:
                print(
                    "AdaPU candidate failed:",
                    beta,
                    n_estimators,
                    repr(e),
                )

    if best is None:
        raise RuntimeError(
            "All AdaPU tuning candidates failed."
        )

    threshold = prior_quantile_threshold(
        best["score_val"],
        data.pi,
    )

    # --------------------------------------------------------
    # Refit AdaPU on train + validation
    # --------------------------------------------------------
    X_dev = np.vstack([
        data.train.X,
        data.val.X,
    ])
    s_dev = np.concatenate([
        data.train.s,
        data.val.s,
    ])

    Xp_dev = X_dev[s_dev == 1].astype(np.float32)
    Xu_dev = X_dev[s_dev == 0].astype(np.float32)

    final_model = AdaBoost_PU(
        n_estimators=best["requested_estimators"],
        prior=data.pi_u,
        random_selection=random_selection,
        beta=best["beta"],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_model.fit(
            Xp_dev,
            Xu_dev,
        )

    n_fitted_final = len(final_model.estimators)

    score_test = final_model.staged_predict(
        X_test,
        n_fitted_final,
        test=False,
    )

    return evaluate_test(
        method="Block3 | AdaPU",
        y_test=data.test.y_true,
        score_test=score_test,
        threshold=threshold,
        probability=None,
        extra={
            "best_params": {
                "beta": best["beta"],
                "n_estimators":
                    best["requested_estimators"],
            },
            "n_estimators_fitted":
                n_fitted_final,
            "prior_given_to_AdaPU":
                data.pi_u,
            "val_metric":
                best["val_auc"],
        },
    )


# ============================================================
# PU Gradient boosting
# ============================================================

class PUGradientBoosting:
    """
    Positive-Unlabeled Gradient Boosting.

    Implements functional gradient boosting using the
    one-sample SCAR PU-risk decomposition.

    F_m(x) = F_{m-1}(x) + learning_rate * h_m(x)
    """

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        min_samples_leaf=10,
        random_state=123,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        self.trees_ = []
        self.init_score_ = 0.0

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -30, 30)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(
        self,
        X_pos,
        X_unl,
        pi,
        c,
    ):
        """
        Fit using observed positive and unlabeled samples.

        Parameters
        ----------
        X_pos : ndarray
            Labelled positive observations.

        X_unl : ndarray
            Unlabelled observations.

        pi : float
            Population positive prior P(Y=1).

        c : float
            SCAR labeling probability P(S=1|Y=1).
        """

        rng = np.random.default_rng(
            self.random_state
        )

        X_pos = np.asarray(
            X_pos,
            dtype=float
        )

        X_unl = np.asarray(
            X_unl,
            dtype=float
        )

        n_p = X_pos.shape[0]
        n_u = X_unl.shape[0]

        # ----------------------------------------------------
        # Positive prevalence among U in one-sample SCAR
        # ----------------------------------------------------

        pi_u = (
            pi * (1.0 - c)
            /
            (1.0 - pi * c)
        )

        self.pi_ = pi
        self.c_ = c
        self.pi_u_ = pi_u

        # One-sample scaling factor
        scale = (
            (1.0 - pi)
            /
            (1.0 - pi_u)
        )

        self.scale_ = scale

        # ----------------------------------------------------
        # Combined training feature matrix
        # ----------------------------------------------------

        X = np.vstack([
            X_pos,
            X_unl
        ])

        # Current ensemble predictions
        F_pos = np.full(
            n_p,
            self.init_score_,
            dtype=float
        )

        F_unl = np.full(
            n_u,
            self.init_score_,
            dtype=float
        )

        self.trees_ = []

        # ====================================================
        # Boosting iterations
        # ====================================================

        for m in range(
            self.n_estimators
        ):

            # Current probabilities
            p_pos = self._sigmoid(
                F_pos
            )

            p_unl = self._sigmoid(
                F_unl
            )

            # ------------------------------------------------
            # Gradient contribution for positive observations
            #
            # pi R_P^+
            # -
            # scale*pi_u R_P^-
            # ------------------------------------------------

            grad_pos = (
                (pi / n_p)
                * (p_pos - 1.0)
                -
                (
                    scale
                    * pi_u
                    / n_p
                )
                * p_pos
            )

            # ------------------------------------------------
            # Gradient contribution for U
            #
            # scale R_U^-
            # ------------------------------------------------

            grad_unl = (
                scale
                / n_u
            ) * p_unl

            # Negative gradients = pseudo-residuals
            residual_pos = -grad_pos
            residual_unl = -grad_unl

            residual = np.concatenate([
                residual_pos,
                residual_unl
            ])

            # ------------------------------------------------
            # Fit weak regression tree to PU pseudo-residuals
            # ------------------------------------------------

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=(
                    self.random_state + m
                ),
            )

            tree.fit(
                X,
                residual
            )

            self.trees_.append(
                tree
            )

            # ------------------------------------------------
            # Update current ensemble scores
            # ------------------------------------------------

            F_pos += (
                self.learning_rate
                * tree.predict(X_pos)
            )

            F_unl += (
                self.learning_rate
                * tree.predict(X_unl)
            )

        return self

    def decision_function(
        self,
        X,
        n_estimators=None,
    ):
        X = np.asarray(
            X,
            dtype=float
        )

        if n_estimators is None:
            n_estimators = len(
                self.trees_
            )

        score = np.full(
            X.shape[0],
            self.init_score_,
            dtype=float
        )

        for tree in self.trees_[
            :n_estimators
        ]:
            score += (
                self.learning_rate
                * tree.predict(X)
            )

        return score

    def predict_proba(
        self,
        X,
        n_estimators=None,
    ):
        score = self.decision_function(
            X,
            n_estimators=n_estimators,
        )

        p = self._sigmoid(
            score
        )

        return np.column_stack([
            1.0 - p,
            p
        ])

    def predict(
        self,
        X,
        threshold=0.0,
    ):
        score = self.decision_function(
            X
        )

        return (
            score >= threshold
        ).astype(int)

PU_GB_GRID = [
    {
        "n_estimators": 50,
        "learning_rate": 0.05,
        "max_depth": 1,
    },
    {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 1,
    },
    {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 2,
    },
    {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 2,
    },
]
def fit_pu_gradient_boosting(
    data,
    param_grid=PU_GB_GRID,
):
    """
    Tune PU Gradient Boosting using PU-only validation AUC.

    No Y_val is used.
    """

    Xp_train, Xu_train = get_P_U(
        data.train
    )

    best = None

    # ========================================================
    # Hyperparameter tuning
    # ========================================================

    for params in param_grid:

        model = PUGradientBoosting(
            n_estimators=params[
                "n_estimators"
            ],
            learning_rate=params[
                "learning_rate"
            ],
            max_depth=params[
                "max_depth"
            ],
            min_samples_leaf=10,
            random_state=data.seed,
        )

        model.fit(
            X_pos=Xp_train,
            X_unl=Xu_train,
            pi=data.pi,
            c=data.c,
        )

        score_val = (
            model.decision_function(
                data.val.X
            )
        )

        val_auc = pu_proxy_auc(
            data.val.s,
            score_val
        )

        candidate = {
            "params": params,
            "val_auc": val_auc,
            "model": model,
            "score_val": score_val,
        }

        if (
            best is None
            or val_auc
            > best["val_auc"]
        ):
            best = candidate

    # --------------------------------------------------------
    # Validation-derived hard threshold
    # --------------------------------------------------------

    threshold = (
        prior_quantile_threshold(
            best["score_val"],
            data.pi
        )
    )

    # ========================================================
    # Refit on train + validation
    # ========================================================

    X_dev = np.vstack([
        data.train.X,
        data.val.X
    ])

    s_dev = np.concatenate([
        data.train.s,
        data.val.s
    ])

    Xp_dev = X_dev[
        s_dev == 1
    ]

    Xu_dev = X_dev[
        s_dev == 0
    ]

    final_model = PUGradientBoosting(
        n_estimators=best[
            "params"
        ]["n_estimators"],

        learning_rate=best[
            "params"
        ]["learning_rate"],

        max_depth=best[
            "params"
        ]["max_depth"],

        min_samples_leaf=10,

        random_state=data.seed,
    )

    final_model.fit(
        X_pos=Xp_dev,
        X_unl=Xu_dev,
        pi=data.pi,
        c=data.c,
    )

    # ========================================================
    # Test predictions
    # ========================================================

    score_test = (
        final_model.decision_function(
            data.test.X
        )
    )

    # I would NOT yet claim these are calibrated probabilities.
    # Therefore probability=None for Brier / LogLoss.
    return evaluate_test(
        method=(
            "Block3 | PU Gradient Boosting"
        ),

        y_test=data.test.y_true,

        score_test=score_test,

        threshold=threshold,

        probability=None,

        extra={
            "best_params":
                best["params"],

            "val_metric":
                best["val_auc"],

            "pi_u":
                final_model.pi_u_,
        },
    )





# ============================================================
# 10. Run one complete scenario/repetition benchmark
# ============================================================

def run_full_benchmark(
    n_train=5000,
    n_val=5000,
    n_test=10000,
    p=10,
    pi=0.20,
    c=0.30,
    delta=0.90,
    n_signal=3,
    rho=0.0,
    seed=123,
):
    print(
        "\nGenerating configured PU synthetic benchmark..."
    )

    data = generate_one_sample_scar(
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        p=p,
        pi=pi,
        c=c,
        delta=delta,
        n_signal=n_signal,
        rho=rho,
        seed=seed,
    )
    shared_data_fingerprint = experiment_fingerprint(data)
    # The DGP uses its own Generator above. Seed library-level model RNGs only
    # after the data have been created so model randomness is reproducible too.
    np.random.seed(seed)
    torch.manual_seed(seed)

    print_experiment_diagnostics(data)

    results = []

    # --------------------------------------------------------
    # Block 0: supervised references. These deliberately use
    # latent true Y in train/validation and are not PU competitors.
    # --------------------------------------------------------
    print("Running Block 0: Oracle logistic regression")
    results.append(fit_oracle_logistic(data))

    print("Running Block 0: Oracle CatBoost")
    results.append(fit_oracle_catboost(data))

    # --------------------------------------------------------
    # Block 1
    # --------------------------------------------------------
    print("Running Block 1: Elkan-Noto")
    results.append(
        fit_elkan_noto_known_c(data)
    )

    print("Running Block 1: uPU")
    results.append(
        fit_pu_logistic_baseline(
            data,
            non_negative=False,
        )
    )

    print("Running Block 1: nnPU")
    
    results.append(
        fit_pu_logistic_baseline(
            data,
            non_negative=True,
            beta=0.0,
            gamma=1.0
        )
    )

    # --------------------------------------------------------
    # Block 2
    # --------------------------------------------------------
    results.extend(
        run_block2(data)
    )

    # --------------------------------------------------------
    # Block 3
    # --------------------------------------------------------
    
    # ========================================================
    # AdaPU
    # ========================================================
    
    print("Running Block 3: AdaPU")
    
    try:
        results.append(
            fit_adapu_one_sample(data)
        )
    
    except Exception as e:
    
        print(
            "AdaPU failed:",
            repr(e)
        )
    
        results.append({
            "method":
                "Block3 | AdaPU",
    
            "AUROC":
                np.nan,
    
            "AUPRC":
                np.nan,
    
            "F1":
                np.nan,
    
            "BalancedAccuracy":
                np.nan,
    
            "MCC":
                np.nan,
    
            "Brier":
                np.nan,
    
            "LogLoss":
                np.nan,
    
            "threshold":
                np.nan,
    
            "best_params":
                {"error": repr(e)},
    
            "val_metric":
                np.nan,
        })
    
    
    # ========================================================
    # PU Gradient Boosting
    # ========================================================
    
    print(
        "Running Block 3: "
        "PU Gradient Boosting"
    )
    
    try:
        results.append(
            fit_pu_gradient_boosting(data)
        )
    
    except Exception as e:
    
        print(
            "PU Gradient Boosting failed:",
            repr(e)
        )
    
        results.append({
            "method":
                "Block3 | PU Gradient Boosting",
    
            "AUROC":
                np.nan,
    
            "AUPRC":
                np.nan,
    
            "F1":
                np.nan,
    
            "BalancedAccuracy":
                np.nan,
    
            "MCC":
                np.nan,
    
            "Brier":
                np.nan,
    
            "LogLoss":
                np.nan,
    
            "threshold":
                np.nan,
    
            "best_params":
                {"error": repr(e)},
    
            "val_metric":
                np.nan,
        })
    
    
    # ========================================================
    # Collect results
    # ========================================================
    
    results_df = pd.DataFrame(
        results
    )

    results_df["method_id"] = results_df["method"].map(METHOD_ID_BY_NAME)
    if results_df["method_id"].isna().any():
        unknown = sorted(results_df.loc[results_df["method_id"].isna(), "method"])
        raise RuntimeError(f"Methods missing from the fixed catalog: {unknown}")
    results_df["method_block"] = results_df["method"].str.extract(
        r"^(Block\d+)"
    )[0]
    results_df["comparison_role"] = np.where(
        results_df["method_block"].eq("Block0"),
        "supervised_reference",
        "pu_method",
    )
    
    
    # Sort only for display;
    # block labels remain in method names.
    display_cols = [
        "method",
        "AUROC",
        "AUPRC",
        "F1",
        "BalancedAccuracy",
        "MCC",
        "Brier",
        "LogLoss",
        "val_metric",
        "best_params",
    ]
    
    
    # Add missing columns if necessary
    for col in display_cols:
    
        if col not in results_df.columns:
    
            results_df[col] = np.nan
    
    
    # ========================================================
    # Display final results
    # ========================================================
    
    print(
        "\n=== FINAL TEST RESULTS ==="
    )
    
    print(
        results_df[
            display_cols
        ]
        .sort_values(
            by="AUROC",
            ascending=False,
            na_position="last",
        )
        .to_string(
            index=False
        )
    )

    if experiment_fingerprint(data) != shared_data_fingerprint:
        raise RuntimeError(
            "A benchmark method modified the shared train/validation/test data."
        )
    results_df["data_fingerprint"] = shared_data_fingerprint
    
    
    return data, results_df



# ============================================================
# 11. Repeated Monte Carlo benchmark
# ============================================================

TEST_METRICS = [
    "AUROC",
    "AUPRC",
    "F1",
    "BalancedAccuracy",
    "MCC",
    "Brier",
    "LogLoss",
]

EXPECTED_METHOD_COUNT = 27
EXPECTED_PU_METHOD_COUNT = 25

METHOD_NAMES_IN_FIXED_ORDER = [
    "Block0 | Oracle logistic regression",
    "Block0 | Oracle CatBoost",
    "Block1 | Elkan-Noto known-c",
    "Block1 | uPU logistic",
    "Block1 | nnPU logistic",
    *[
        f"Block2 | {strategy} + {engine}"
        for strategy in ["Naive", "ProbCorrection", "RN", "CostSensitive"]
        for engine in ["AdaBoost", "GBM", "XGBoost", "LightGBM", "CatBoost"]
    ],
    "Block3 | AdaPU",
    "Block3 | PU Gradient Boosting",
]
METHOD_ID_BY_NAME = {
    method: f"M{index:03d}"
    for index, method in enumerate(METHOD_NAMES_IN_FIXED_ORDER, start=1)
}

if len(METHOD_ID_BY_NAME) != EXPECTED_METHOD_COUNT:
    raise RuntimeError("The fixed Phase 1 method catalog must contain 27 methods.")


def rank_and_summarize(raw_results):
    """Add paired ranks and summaries, keeping oracle references separate."""
    raw_results = raw_results.copy()

    raw_results["comparison_role"] = np.where(
        raw_results["method"].str.startswith("Block0 |"),
        "supervised_reference",
        "pu_method",
    )
    raw_results["method_block"] = raw_results["method"].str.extract(
        r"^(Block\d+)"
    )[0]

    for metric in TEST_METRICS:
        raw_results[metric] = pd.to_numeric(
            raw_results[metric], errors="coerce"
        )

    raw_results["AUROC_rank_all_27"] = (
        raw_results.groupby("rep")["AUROC"]
        .rank(ascending=False, method="average")
    )
    raw_results["AUPRC_rank_all_27"] = (
        raw_results.groupby("rep")["AUPRC"]
        .rank(ascending=False, method="average")
    )
    raw_results["AUROC_rank_PU_25"] = np.nan
    raw_results["AUPRC_rank_PU_25"] = np.nan
    pu_mask = raw_results["comparison_role"].eq("pu_method")
    raw_results.loc[pu_mask, "AUROC_rank_PU_25"] = (
        raw_results.loc[pu_mask].groupby("rep")["AUROC"]
        .rank(ascending=False, method="average")
    )
    raw_results.loc[pu_mask, "AUPRC_rank_PU_25"] = (
        raw_results.loc[pu_mask].groupby("rep")["AUPRC"]
        .rank(ascending=False, method="average")
    )
    # Backward-compatible names now mean ranks among PU methods only.
    raw_results["AUROC_rank"] = raw_results["AUROC_rank_PU_25"]
    raw_results["AUPRC_rank"] = raw_results["AUPRC_rank_PU_25"]

    summary_rows = []
    for method, df_method in raw_results.groupby("method"):
        row = {
            "method": method,
            "method_block": df_method["method_block"].iloc[0],
            "comparison_role": df_method["comparison_role"].iloc[0],
            "n_repetitions_available": df_method["rep"].nunique(),
        }

        for metric in TEST_METRICS:
            values = df_method[metric].dropna()
            n = len(values)
            row[f"{metric}_n"] = n

            if n == 0:
                mean = sd = median = se = ci_low = ci_high = np.nan
            else:
                mean = values.mean()
                median = values.median()
                if n > 1:
                    sd = values.std(ddof=1)
                    se = sd / np.sqrt(n)
                    ci_low = mean - 1.96 * se
                    ci_high = mean + 1.96 * se
                else:
                    sd = se = ci_low = ci_high = np.nan

            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
            row[f"{metric}_median"] = median
            row[f"{metric}_se"] = se
            row[f"{metric}_ci95_low"] = ci_low
            row[f"{metric}_ci95_high"] = ci_high

        row["mean_AUROC_rank_PU_25"] = df_method["AUROC_rank_PU_25"].mean()
        row["mean_AUPRC_rank_PU_25"] = df_method["AUPRC_rank_PU_25"].mean()
        row["mean_AUROC_rank_all_27"] = df_method["AUROC_rank_all_27"].mean()
        row["mean_AUPRC_rank_all_27"] = df_method["AUPRC_rank_all_27"].mean()
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        "AUROC_mean", ascending=False, na_position="last"
    )
    return raw_results, summary_df


def run_repeated_benchmark(
    n_reps=100,
    base_seed=123,
    rep_ids=None,

    # DGP parameters
    n_train=5000,
    n_val=5000,
    n_test=10000,
    p=10,
    pi=0.20,
    c=0.30,
    delta=0.90,
    n_signal=3,
    rho=0.0,
):

    all_results = []
    all_diagnostics = []
    failed_reps = []

    if rep_ids is None:
        rep_ids = range(1, n_reps + 1)
    else:
        rep_ids = list(rep_ids)

    for rep in rep_ids:

        seed = base_seed + rep - 1

        print("\n" + "=" * 80)
        print(
            f"REPETITION {rep}/{n_reps} "
            f"| seed = {seed}"
        )
        print("=" * 80)

        try:

            # ----------------------------------------------------
            # Run complete benchmark on one newly generated dataset
            # ----------------------------------------------------
            data, results = run_full_benchmark(
                n_train=n_train,
                n_val=n_val,
                n_test=n_test,
                p=p,
                pi=pi,
                c=c,
                delta=delta,
                n_signal=n_signal,
                rho=rho,
                seed=seed,
            )

            # ----------------------------------------------------
            # Tag each method result with repetition information
            # ----------------------------------------------------
            results = results.copy()

            results.insert(
                0,
                "rep",
                rep
            )

            results.insert(
                1,
                "seed",
                seed
            )

            # Useful if you later run several simulation scenarios
            results["n_train"] = n_train
            results["n_val"] = n_val
            results["n_test"] = n_test
            results["p"] = p
            results["pi"] = pi
            results["c"] = c
            results["delta"] = data.delta
            results["delta_sq"] = data.delta_sq
            results["signal_per_active_coordinate"] = data.signal
            results["n_signal"] = n_signal
            results["rho"] = rho
            results["pi_u_theoretical"] = data.pi_u
            # Every method row in a repetition receives the same hash. This
            # makes the shared-data design directly auditable in the CSVs.
            results["data_fingerprint"] = experiment_fingerprint(data)

            all_results.append(results)

            # ----------------------------------------------------
            # Also save DGP diagnostics for every split
            # ----------------------------------------------------
            for split_name, split in [
                ("train", data.train),
                ("val", data.val),
                ("test", data.test),
            ]:

                d = split_diagnostics(
                    split,
                    split_name
                )

                d["rep"] = rep
                d["seed"] = seed
                d["pi"] = pi
                d["c"] = c
                d["delta"] = data.delta
                d["delta_sq"] = data.delta_sq
                d["signal_per_active_coordinate"] = data.signal
                d["rho"] = rho

                all_diagnostics.append(d)

        except Exception as e:

            print(
                f"\nREPETITION {rep} FAILED: "
                f"{repr(e)}"
            )

            failed_reps.append({
                "rep": rep,
                "seed": seed,
                "error": repr(e),
            })

    # ============================================================
    # Combine raw method-level results
    # ============================================================

    if len(all_results) == 0:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(all_diagnostics),
            pd.DataFrame(failed_reps),
        )

    raw_results = pd.concat(
        all_results,
        ignore_index=True
    )

    diagnostics_df = pd.DataFrame(
        all_diagnostics
    )

    failures_df = pd.DataFrame(
        failed_reps
    )

    raw_results, summary_df = rank_and_summarize(raw_results)

    return (
        raw_results,
        summary_df,
        diagnostics_df,
        failures_df,
    )


# ============================================================
# 12. Slurm-safe persistence and collection
# ============================================================

def atomic_to_csv(frame, path):
    """Write a CSV atomically so a cancelled job cannot leave a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def save_combined_results(raw, summary, diagnostics, failures, output_dir, n_reps):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "raw": output_dir / f"pu_benchmark_raw_{n_reps}reps.csv",
        "summary": output_dir / f"pu_benchmark_summary_{n_reps}reps.csv",
        "diagnostics": output_dir / f"pu_benchmark_diagnostics_{n_reps}reps.csv",
        "failures": output_dir / f"pu_benchmark_failures_{n_reps}reps.csv",
    }
    atomic_to_csv(raw, paths["raw"])
    atomic_to_csv(summary, paths["summary"])
    atomic_to_csv(diagnostics, paths["diagnostics"])
    if not failures.empty:
        atomic_to_csv(failures, paths["failures"])
    return paths


def collect_array_results(output_dir, n_reps=100, base_seed=123):
    """Collect per-array-task files and audit Monte Carlo completeness."""
    output_dir = Path(output_dir)
    repetition_dir = output_dir / "repetitions"
    result_frames = []
    diagnostic_frames = []
    report_rows = []

    for rep in range(1, n_reps + 1):
        result_file = repetition_dir / f"rep_{rep:03d}_results.csv"
        diagnostic_file = repetition_dir / f"rep_{rep:03d}_diagnostics.csv"
        issues = []
        method_count = 0
        fingerprint_count = 0
        auroc_count = 0

        if not result_file.is_file():
            issues.append("missing result file")
        else:
            try:
                frame = pd.read_csv(result_file)
                required = {"rep", "seed", "method", "AUROC", "data_fingerprint"}
                missing_columns = sorted(required.difference(frame.columns))
                if missing_columns:
                    issues.append("missing columns: " + ", ".join(missing_columns))
                else:
                    rep_values = set(pd.to_numeric(frame["rep"], errors="coerce").dropna())
                    seed_values = set(pd.to_numeric(frame["seed"], errors="coerce").dropna())
                    method_count = frame["method"].nunique()
                    fingerprint_count = frame["data_fingerprint"].nunique()
                    auroc_count = pd.to_numeric(frame["AUROC"], errors="coerce").notna().sum()

                    if rep_values != {rep}:
                        issues.append(f"unexpected rep values: {sorted(rep_values)}")
                    expected_seed = base_seed + rep - 1
                    if seed_values != {expected_seed}:
                        issues.append(f"unexpected seed values: {sorted(seed_values)}")
                    if method_count != EXPECTED_METHOD_COUNT:
                        issues.append(
                            f"expected {EXPECTED_METHOD_COUNT} methods, found {method_count}"
                        )
                    if len(frame) != EXPECTED_METHOD_COUNT:
                        issues.append(
                            f"expected {EXPECTED_METHOD_COUNT} rows, found {len(frame)}"
                        )
                    if fingerprint_count != 1:
                        issues.append(
                            f"expected one shared data fingerprint, found {fingerprint_count}"
                        )
                    if auroc_count != EXPECTED_METHOD_COUNT:
                        issues.append(
                            f"AUROC available for {auroc_count}/{EXPECTED_METHOD_COUNT} methods"
                        )

                    result_frames.append(frame)
            except Exception as error:
                issues.append(f"could not read result file: {error!r}")

        if not diagnostic_file.is_file():
            issues.append("missing diagnostics file")
        else:
            try:
                diagnostic_frames.append(pd.read_csv(diagnostic_file))
            except Exception as error:
                issues.append(f"could not read diagnostics file: {error!r}")

        report_rows.append({
            "rep": rep,
            "expected_seed": base_seed + rep - 1,
            "method_count": method_count,
            "methods_with_AUROC": auroc_count,
            "data_fingerprint_count": fingerprint_count,
            "complete": len(issues) == 0,
            "issues": "; ".join(issues),
        })

    report = pd.DataFrame(report_rows)
    atomic_to_csv(report, output_dir / "pu_benchmark_collection_report.csv")

    if not result_frames:
        raise RuntimeError(
            f"No repetition result files were found in {repetition_dir}"
        )

    raw = pd.concat(result_frames, ignore_index=True)
    raw, summary = rank_and_summarize(raw)
    diagnostics = (
        pd.concat(diagnostic_frames, ignore_index=True)
        if diagnostic_frames else pd.DataFrame()
    )
    failures = report.loc[~report["complete"]].copy()
    paths = save_combined_results(
        raw, summary, diagnostics, failures, output_dir, n_reps
    )
    return raw, summary, diagnostics, report, paths


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run the PU benchmark locally, run one Slurm array repetition, "
            "or aggregate Slurm result files."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--rep", type=int,
        help="Run this one-based repetition number (used by a Slurm array task).",
    )
    mode.add_argument(
        "--aggregate", action="store_true",
        help="Collect rep_XXX files instead of running models.",
    )
    mode.add_argument(
        "--check-dependencies", action="store_true",
        help="Check optional engines and AdaPU, then exit.",
    )
    parser.add_argument("--n-reps", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=123)
    parser.add_argument(
        "--output-dir", type=Path, default=SCRIPT_DIR / "mc_results"
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--n-val", type=int, default=5000)
    parser.add_argument("--n-test", type=int, default=10000)
    parser.add_argument("--p", type=int, default=10)
    parser.add_argument("--pi", type=float, default=0.20)
    parser.add_argument("--c", type=float, default=0.30)
    parser.add_argument(
        "--delta", type=float, default=0.90,
        help="Target Mahalanobis distance between class means.",
    )
    parser.add_argument("--n-signal", type=int, default=3)
    parser.add_argument("--rho", type=float, default=0.0)
    return parser


def dependency_status():
    status = {
        "XGBoost": XGBClassifier is not None,
        "LightGBM": LGBMClassifier is not None,
        "CatBoost": CatBoostClassifier is not None,
        "AdaPU": ADAPU_AVAILABLE,
    }
    print(f"Python: {sys.version.split()[0]}")
    print(f"Worker threads: {N_JOBS}")
    print(f"AdaPU source: {ADAPU_SRC}")
    for name, available in status.items():
        print(f"{name}: {'OK' if available else 'MISSING'}")
    return all(status.values())


def benchmark_parameters(args):
    return {
        "n_train": args.n_train,
        "n_val": args.n_val,
        "n_test": args.n_test,
        "p": args.p,
        "pi": args.pi,
        "c": args.c,
        "delta": args.delta,
        "n_signal": args.n_signal,
        "rho": args.rho,
    }


def main():
    args = build_argument_parser().parse_args()
    if args.n_reps < 1:
        raise SystemExit("--n-reps must be at least 1")

    if args.check_dependencies:
        raise SystemExit(0 if dependency_status() else 1)

    if args.aggregate:
        raw, summary, diagnostics, report, paths = collect_array_results(
            args.output_dir, n_reps=args.n_reps, base_seed=args.base_seed
        )
        n_complete = int(report["complete"].sum())
        print(f"Collected {n_complete}/{args.n_reps} complete repetitions.")
        print(f"Raw results: {paths['raw']}")
        print(f"Summary: {paths['summary']}")
        print(f"Audit report: {args.output_dir / 'pu_benchmark_collection_report.csv'}")
        if n_complete != args.n_reps and not args.allow_incomplete:
            raise SystemExit(
                "Collection is incomplete. Inspect the audit report, rerun failed "
                "array indices, and aggregate again."
            )
        return

    rep_ids = [args.rep] if args.rep is not None else None
    if args.rep is not None and not (1 <= args.rep <= args.n_reps):
        raise SystemExit("--rep must be between 1 and --n-reps")

    raw, summary, diagnostics, failures = run_repeated_benchmark(
        n_reps=args.n_reps,
        base_seed=args.base_seed,
        rep_ids=rep_ids,
        **benchmark_parameters(args),
    )

    if args.rep is not None:
        repetition_dir = args.output_dir / "repetitions"
        result_file = repetition_dir / f"rep_{args.rep:03d}_results.csv"
        diagnostic_file = repetition_dir / f"rep_{args.rep:03d}_diagnostics.csv"
        failure_file = repetition_dir / f"rep_{args.rep:03d}_failure.csv"

        if raw.empty:
            atomic_to_csv(failures, failure_file)
            raise SystemExit(f"Repetition {args.rep} failed; see {failure_file}")

        atomic_to_csv(raw, result_file)
        atomic_to_csv(diagnostics, diagnostic_file)
        method_count = raw["method"].nunique()
        auroc_count = raw["AUROC"].notna().sum()
        if method_count != EXPECTED_METHOD_COUNT or auroc_count != EXPECTED_METHOD_COUNT:
            problem = pd.DataFrame([{
                "rep": args.rep,
                "seed": args.base_seed + args.rep - 1,
                "method_count": method_count,
                "methods_with_AUROC": auroc_count,
                "error": "One or more of the 27 methods failed.",
            }])
            atomic_to_csv(problem, failure_file)
            if not args.allow_incomplete:
                raise SystemExit(
                    f"Repetition {args.rep} has incomplete method results; "
                    f"see {failure_file} and the Slurm log."
                )
        print(f"Saved repetition {args.rep} to {result_file}")
        return

    # Without --rep, run all repetitions serially (useful off-cluster).
    paths = save_combined_results(
        raw, summary, diagnostics, failures, args.output_dir, args.n_reps
    )
    print(f"Raw results: {paths['raw']}")
    print(f"Summary: {paths['summary']}")


if __name__ == "__main__":
    main()
