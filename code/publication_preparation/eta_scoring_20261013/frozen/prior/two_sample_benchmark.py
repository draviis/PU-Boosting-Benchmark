"""Canonical two-sample (case-control) SCAR PU benchmark.

Every repetition independently samples

* ``P``: observations from the positive class-conditional distribution f1;
* ``U``: observations from the population mixture pi*f1 + (1-pi)*f0;
* ``test``: an independent, fully labelled population mixture.

All 27 methods receive the very same P/U/validation/test arrays.  The legacy
one-sample implementation is vendored only to reuse its engines and repaired
PU-risk optimisers.  Setting its internal ``c`` argument to zero gives exactly
the canonical two-sample risk

    pi R_P^+ + R_U^- - pi R_P^-.

There is no labelling probability ``c`` in this observation design.
"""

from __future__ import annotations

import math
import warnings
from typing import Callable

import numpy as np
import pandas as pd
from scipy.special import logsumexp

import base_phase1_benchmark as base
import repaired_method_library as repaired


STRUCTURAL_TYPES = {
    "linear_gaussian",
    "curved_boundary",
    "multiplicative_interaction",
    "unequal_covariance",
    "many_irrelevant_variables",
}
SAMPLING_SCHEME = "two_sample_case_control_scar"
DGP_IMPLEMENTATION = "deterministic_toeplitz_ar1_v1"

# Updated before each repetition.  It lets every validation-derived prevalence
# threshold use U only; the pooled P+U case-control sample is not a population
# sample and therefore cannot be thresholded at its artificial class fraction.
_ACTIVE_VALIDATION_U_MASK: np.ndarray | None = None
_ORIGINAL_PRIOR_THRESHOLD = base.prior_quantile_threshold


METHOD_NAMES_IN_FIXED_ORDER = [
    "Block0 | Oracle logistic regression",
    "Block0 | Oracle CatBoost",
    "Block1 | CaseControl density-ratio known-pi",
    "Block1 | uPU logistic",
    "Block1 | nnPU logistic",
    *[
        f"Block2 | {strategy} + {engine}"
        for strategy in ["Naive", "ProbCorrection", "RN", "CostSensitive"]
        for engine in ["AdaBoost", "GBM", "XGBoost", "LightGBM", "CatBoost"]
    ],
    "Block3 | AdaPU repaired",
    "Block3 | PU Gradient Boosting repaired",
]
METHOD_ID_BY_NAME = {
    method: f"TM{index:03d}"
    for index, method in enumerate(METHOD_NAMES_IN_FIXED_ORDER, start=1)
}
if len(METHOD_ID_BY_NAME) != 27:
    raise RuntimeError("The two-sample method catalog must contain 27 methods")


def _population_validation_threshold(score_val, pi):
    score_val = np.asarray(score_val, dtype=float)
    mask = _ACTIVE_VALIDATION_U_MASK
    if mask is not None and score_val.shape[0] == mask.shape[0]:
        score_val = score_val[mask]
    return _ORIGINAL_PRIOR_THRESHOLD(score_val, pi)


def _validate_design(p, pi, delta, n_signal, rho, sizes):
    if not 0.0 < pi < 1.0:
        raise ValueError("pi must lie in (0, 1)")
    if not 1 <= n_signal <= p:
        raise ValueError("n_signal must lie between 1 and p")
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must lie in [0, 1)")
    if delta <= 0.0:
        raise ValueError("delta must be positive")
    if any(int(value) < 2 for value in sizes):
        raise ValueError("every sample size must be at least two")


def _toeplitz_precision_times(vector, rho):
    """Apply the exact AR(1) Toeplitz precision matrix in fixed order."""
    vector = np.asarray(vector, dtype=np.float64)
    p = vector.size
    if p == 1 or rho == 0.0:
        return vector.copy()
    denominator = 1.0 - rho * rho
    out = np.empty_like(vector)
    out[0] = (vector[0] - rho * vector[1]) / denominator
    for column in range(1, p - 1):
        out[column] = (
            (1.0 + rho * rho) * vector[column]
            - rho * vector[column - 1]
            - rho * vector[column + 1]
        ) / denominator
    out[-1] = (vector[-1] - rho * vector[-2]) / denominator
    return out


def _calibrate_deterministic_toeplitz_mean(p, n_signal, rho, delta):
    """Calibrate an equal active-coordinate shift without LAPACK calls."""
    direction = np.zeros(p, dtype=np.float64)
    direction[:n_signal] = 1.0
    precision_direction = _toeplitz_precision_times(direction, rho)
    quadratic = sum(
        float(direction[column]) * float(precision_direction[column])
        for column in range(p)
    )
    scale = float(delta) / math.sqrt(quadratic)
    mu = direction * scale
    precision_mu = precision_direction * scale
    achieved_delta_sq = sum(
        float(mu[column]) * float(precision_mu[column])
        for column in range(p)
    )
    return mu, precision_mu, float(achieved_delta_sq)


def _deterministic_toeplitz_standard_normal(n, p, rho, rng):
    """Draw N(0, Toeplitz(rho)) using fixed-order AR(1) operations."""
    innovations = rng.normal(size=(int(n), int(p)))
    if p == 1 or rho == 0.0:
        return innovations
    x = np.empty_like(innovations)
    x[:, 0] = innovations[:, 0]
    innovation_scale = math.sqrt(1.0 - rho * rho)
    for column in range(1, p):
        x[:, column] = (
            rho * x[:, column - 1]
            + innovation_scale * innovations[:, column]
        )
    return x


def _linear_conditional_sample(
    n, y, mu, rho, precision_mu, achieved_delta_sq, rng
):
    x = np.empty((n, mu.size), dtype=np.float64)
    positive = y == 1
    x[positive] = _deterministic_toeplitz_standard_normal(
        int(positive.sum()), mu.size, rho, rng
    ) + mu
    x[~positive] = _deterministic_toeplitz_standard_normal(
        int((~positive).sum()), mu.size, rho, rng
    )
    score = np.zeros(n, dtype=np.float64)
    for column in range(mu.size):
        score += x[:, column] * precision_mu[column]
    score -= 0.5 * achieved_delta_sq
    return x, score


def _curved_conditional_sample(n, y, mu, strength, rng):
    z = rng.normal(size=(n, mu.size)) + y[:, None] * mu
    x = z.copy()
    x[:, 1] = z[:, 1] + strength * (z[:, 0] ** 2 - 1.0)
    return x, z @ mu - 0.5 * float(mu @ mu)


def _interaction_conditional_sample(n, y, shift, main_shift, p, rng):
    x = rng.normal(size=(n, p))
    sign = rng.choice(np.array([-1.0, 1.0]), size=n)
    x[:, 0] += shift * sign
    x[:, 1] += shift * sign * np.where(y == 1, 1.0, -1.0)
    x[:, 2] += main_shift * y
    pos_centres = np.array([[shift, shift], [-shift, -shift]])
    neg_centres = np.array([[shift, -shift], [-shift, shift]])
    log_pos = logsumexp(
        np.stack([-0.5 * np.sum((x[:, :2] - c) ** 2, axis=1) for c in pos_centres]),
        axis=0,
    ) - math.log(2.0)
    log_neg = logsumexp(
        np.stack([-0.5 * np.sum((x[:, :2] - c) ** 2, axis=1) for c in neg_centres]),
        axis=0,
    ) - math.log(2.0)
    score = log_pos - log_neg + main_shift * x[:, 2] - 0.5 * main_shift**2
    return x, score


def _unequal_conditional_sample(n, y, mu, variance_ratio, rng):
    variance = np.ones(mu.size)
    variance[:3] = variance_ratio
    x = rng.normal(size=(n, mu.size))
    positive = y == 1
    x[positive] = x[positive] * np.sqrt(variance) + mu
    q0 = np.sum(x * x, axis=1)
    q1 = np.sum((x - mu) ** 2 / variance, axis=1)
    score = 0.5 * (q0 - q1 - np.log(variance).sum())
    return x, score


def generate_two_sample_scar(
    *,
    n_p_train=300,
    n_u_train=4700,
    n_p_val=300,
    n_u_val=4700,
    n_test=10000,
    p=10,
    pi=0.20,
    delta=0.90,
    n_signal=3,
    rho=0.0,
    seed=123,
    structural_type="linear_gaussian",
    transform_strength=0.0,
    main_effect_shift=0.0,
    positive_variance_ratio=1.0,
):
    """Draw independent P, U and labelled-test samples from one population."""
    global _ACTIVE_VALIDATION_U_MASK
    _validate_design(
        p,
        pi,
        delta,
        n_signal,
        rho,
        (n_p_train, n_u_train, n_p_val, n_u_val, n_test),
    )
    if structural_type not in STRUCTURAL_TYPES:
        raise ValueError(f"unknown structural_type: {structural_type}")
    if structural_type in {"curved_boundary", "multiplicative_interaction", "unequal_covariance"}:
        if p != 10 or n_signal != 3 or not np.isclose(rho, 0.0):
            raise ValueError("nonlinear structural scenarios require p=10, n_signal=3, rho=0")
    if structural_type == "many_irrelevant_variables" and p != 500:
        raise ValueError("many_irrelevant_variables requires p=500")

    rng = np.random.default_rng(seed)
    mu, precision_mu, achieved_delta_sq = _calibrate_deterministic_toeplitz_mean(
        p, n_signal, rho, delta
    )

    if structural_type in {"linear_gaussian", "many_irrelevant_variables"}:
        conditional: Callable = lambda n, y: _linear_conditional_sample(
            n, y, mu, rho, precision_mu, achieved_delta_sq, rng
        )
    elif structural_type == "curved_boundary":
        if transform_strength <= 0:
            raise ValueError("curved_boundary requires transform_strength > 0")
        curved_mu = np.zeros(p)
        curved_mu[[1, 2, 3]] = delta / math.sqrt(3.0)
        mu = curved_mu
        achieved_delta_sq = delta**2
        conditional = lambda n, y: _curved_conditional_sample(
            n, y, curved_mu, transform_strength, rng
        )
    elif structural_type == "multiplicative_interaction":
        if transform_strength <= 0 or main_effect_shift <= 0:
            raise ValueError("interaction scenario requires positive shifts")
        achieved_delta_sq = delta**2
        conditional = lambda n, y: _interaction_conditional_sample(
            n, y, transform_strength, main_effect_shift, p, rng
        )
    else:
        if positive_variance_ratio <= 1:
            raise ValueError("unequal_covariance requires variance ratio > 1")
        unequal_mu = np.zeros(p)
        unequal_mu[:3] = delta / math.sqrt(3.0)
        mu = unequal_mu
        achieved_delta_sq = delta**2
        conditional = lambda n, y: _unequal_conditional_sample(
            n, y, unequal_mu, positive_variance_ratio, rng
        )

    def pure_positive(n):
        y = np.ones(n, dtype=np.int64)
        x, score = conditional(n, y)
        return x, y, score

    def population(n):
        y = rng.binomial(1, pi, size=n).astype(np.int64)
        x, score = conditional(n, y)
        return x, y, score

    xp_train, yp_train, bp_train = pure_positive(n_p_train)
    xu_train, yu_train, bu_train = population(n_u_train)
    xp_val, yp_val, bp_val = pure_positive(n_p_val)
    xu_val, yu_val, bu_val = population(n_u_val)
    xtest, ytest, btest = population(n_test)

    train = base.PUSplit(
        X=np.vstack([xp_train, xu_train]),
        y_true=np.concatenate([yp_train, yu_train]),
        s=np.concatenate([
            np.ones(n_p_train, dtype=np.int64),
            np.zeros(n_u_train, dtype=np.int64),
        ]),
    )
    val = base.PUSplit(
        X=np.vstack([xp_val, xu_val]),
        y_true=np.concatenate([yp_val, yu_val]),
        s=np.concatenate([
            np.ones(n_p_val, dtype=np.int64),
            np.zeros(n_u_val, dtype=np.int64),
        ]),
    )
    test = base.PUSplit(X=xtest, y_true=ytest, s=np.zeros(n_test, dtype=np.int64))
    data = base.PUExperiment(
        train=train,
        val=val,
        test=test,
        pi=pi,
        c=0.0,  # mathematical bridge to the canonical two-sample PU risk
        p=p,
        delta=delta,
        delta_sq=achieved_delta_sq,
        signal=float(mu[np.flatnonzero(mu)[0]]) if np.any(mu) else 0.0,
        n_signal=n_signal,
        rho=rho,
        seed=seed,
    )
    data.sampling_scheme = SAMPLING_SCHEME
    data.n_p_train = int(n_p_train)
    data.n_u_train = int(n_u_train)
    data.n_p_val = int(n_p_val)
    data.n_u_val = int(n_u_val)
    data.structural_type = structural_type
    data.transform_strength = float(transform_strength)
    data.main_effect_shift = float(main_effect_shift)
    data.positive_variance_ratio = float(positive_variance_ratio)
    data.dgp_implementation = DGP_IMPLEMENTATION
    data.bayes_scores = {
        "train": np.concatenate([bp_train, bu_train]),
        "val": np.concatenate([bp_val, bu_val]),
        "test": btest,
    }
    _ACTIVE_VALIDATION_U_MASK = val.s == 0
    return data


def split_diagnostics(split, name):
    is_positive_sample = split.s == 1
    is_unlabeled = ~is_positive_sample
    return {
        "split": name,
        "n": len(split.y_true),
        "n_positive_sample": int(is_positive_sample.sum()),
        "n_unlabeled_sample": int(is_unlabeled.sum()),
        "n_hidden_positive_in_U": int(((split.y_true == 1) & is_unlabeled).sum()),
        "n_hidden_negative_in_U": int(((split.y_true == 0) & is_unlabeled).sum()),
        "empirical_pi_in_U": (
            float(split.y_true[is_unlabeled].mean()) if is_unlabeled.any() else np.nan
        ),
        "case_control_positive_fraction": float(is_positive_sample.mean()),
        "n_true_positive": int((split.y_true == 1).sum()),
        "n_true_negative": int((split.y_true == 0).sum()),
        "true_positive_rate_in_pooled_rows": float(split.y_true.mean()),
    }


def print_experiment_diagnostics(data):
    diagnostics = pd.DataFrame(
        [split_diagnostics(data.train, "train"), split_diagnostics(data.val, "val"), split_diagnostics(data.test, "test")]
    )
    print("\n=== TWO-SAMPLE CASE-CONTROL DGP DIAGNOSTICS ===")
    print(f"pi={data.pi:.4f}; nP/nU train={data.n_p_train}/{data.n_u_train}; delta={data.delta:.4f}")
    print(diagnostics.to_string(index=False))


def _oracle_subset(split):
    # U is an actual population draw; P+U is a deliberately distorted sample.
    mask = split.s == 0
    return split.X[mask], split.y_true[mask]


def fit_oracle_logistic(data, C_grid=(0.1, 1.0, 10.0)):
    x_train, y_train = _oracle_subset(data.train)
    x_val, y_val = _oracle_subset(data.val)
    best = None
    for c_value in C_grid:
        model = base.LogisticRegression(C=c_value, max_iter=3000, random_state=data.seed)
        model.fit(x_train, y_train)
        val_probability = model.predict_proba(x_val)[:, 1]
        val_auc = base.roc_auc_score(y_val, val_probability)
        if best is None or val_auc > best["val_auc"]:
            best = {"C": c_value, "val_auc": val_auc, "probability": val_probability}
    threshold, validation_f1 = base.oracle_f1_threshold(y_val, best["probability"])
    final = base.LogisticRegression(C=best["C"], max_iter=3000, random_state=data.seed)
    final.fit(np.vstack([x_train, x_val]), np.concatenate([y_train, y_val]))
    probability = final.predict_proba(data.test.X)[:, 1]
    return base.evaluate_test(
        "Block0 | Oracle logistic regression",
        data.test.y_true,
        probability,
        threshold,
        probability,
        {
            "best_params": {"C": best["C"]},
            "val_metric": best["val_auc"],
            "validation_F1_at_threshold": validation_f1,
            "comparison_role": "supervised_reference",
            "uses_true_y_for_development": True,
            "oracle_training_sample": "unlabeled_population_rows_only",
        },
    )


def fit_oracle_catboost(data):
    if base.CatBoostClassifier is None:
        raise ImportError("catboost is required")
    x_train, y_train = _oracle_subset(data.train)
    x_val, y_val = _oracle_subset(data.val)
    best = None
    for params in base.ENGINE_GRIDS["CatBoost"]:
        model = base.make_engine("CatBoost", params, data.seed)
        model.fit(x_train, y_train)
        val_probability = base.model_score(model, x_val)
        val_auc = base.roc_auc_score(y_val, val_probability)
        if best is None or val_auc > best["val_auc"]:
            best = {"params": dict(params), "val_auc": val_auc, "probability": val_probability}
    threshold, validation_f1 = base.oracle_f1_threshold(y_val, best["probability"])
    final = base.make_engine("CatBoost", best["params"], data.seed)
    final.fit(np.vstack([x_train, x_val]), np.concatenate([y_train, y_val]))
    probability = base.model_score(final, data.test.X)
    return base.evaluate_test(
        "Block0 | Oracle CatBoost",
        data.test.y_true,
        probability,
        threshold,
        probability,
        {
            "best_params": best["params"],
            "val_metric": best["val_auc"],
            "validation_F1_at_threshold": validation_f1,
            "comparison_role": "supervised_reference",
            "uses_true_y_for_development": True,
            "oracle_training_sample": "unlabeled_population_rows_only",
        },
    )


def _case_control_log_density_ratio(q, alpha):
    """Return log(f_P/f_U) without saturating the ranking score.

    If q(x)=P(sample=P|X=x) in a case-control sample whose sampled-positive
    fraction is alpha, then

        log f_P/f_U = logit(q) + log((1-alpha)/alpha).

    Clipping is restricted to machine precision so exact 0/1 predictions are
    numerically representable.  Population-probability clipping is deliberately
    kept out of this ranking score.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("case-control alpha must lie in (0, 1)")
    q = np.asarray(q, dtype=float)
    # Preserve every representable probability in (0, 1).  Only exact 0/1
    # values are moved to their nearest finite neighbours, so clipping cannot
    # collapse distinct finite engine scores and alter a ranking.
    q = np.clip(q, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))
    return (
        np.log(q)
        - np.log1p(-q)
        + math.log1p(-alpha)
        - math.log(alpha)
    )


def _probability_from_log_density_ratio(log_density_ratio, pi):
    """Convert log(f_P/f_U) to clipped eta only for probability metrics."""
    log_eta = math.log(pi) + np.asarray(log_density_ratio, dtype=float)
    eta = np.exp(np.clip(log_eta, -50.0, 50.0))
    return base.clip_prob(eta)


def _case_control_probability(q, pi, alpha):
    """Backward-compatible probability helper used by algebra tests."""
    return _probability_from_log_density_ratio(
        _case_control_log_density_ratio(q, alpha),
        pi,
    )


def fit_case_control_known_pi(data, C_grid=(0.1, 1.0, 10.0)):
    alpha_train = float(data.train.s.mean())
    best = None
    for c_value in C_grid:
        model = base.LogisticRegression(C=c_value, max_iter=3000, random_state=data.seed)
        model.fit(data.train.X, data.train.s)
        q_val = model.predict_proba(data.val.X)[:, 1]
        val_auc = base.pu_proxy_auc(data.val.s, q_val)
        if best is None or val_auc > best["val_auc"]:
            best = {"C": c_value, "val_auc": val_auc, "model": model, "q_val": q_val}
    score_val = _case_control_log_density_ratio(best["q_val"], alpha_train)
    threshold = _population_validation_threshold(score_val, data.pi)
    x_dev = np.vstack([data.train.X, data.val.X])
    s_dev = np.concatenate([data.train.s, data.val.s])
    alpha_dev = float(s_dev.mean())
    final = base.LogisticRegression(C=best["C"], max_iter=3000, random_state=data.seed)
    final.fit(x_dev, s_dev)
    score_test = _case_control_log_density_ratio(
        final.predict_proba(data.test.X)[:, 1], alpha_dev
    )
    eta_test = _probability_from_log_density_ratio(score_test, data.pi)
    return base.evaluate_test(
        "Block1 | CaseControl density-ratio known-pi",
        data.test.y_true,
        score_test,
        threshold,
        eta_test,
        {
            "best_params": {"C": best["C"]},
            "val_metric": best["val_auc"],
            "case_control_alpha_train": alpha_train,
            "case_control_alpha_dev": alpha_dev,
            "known_population_prior": data.pi,
            "ranking_score_type": "unclipped_log_case_control_density_ratio",
            "probability_score_type": "clipped_pi_times_density_ratio",
        },
    )


def fit_one_boosting_cell(data, strategy, engine_name):
    best = None
    alpha_train = float(data.train.s.mean())
    for strategy_param in base.STRATEGY_GRIDS[strategy]:
        for engine_param in base.ENGINE_GRIDS[engine_name]:
            x_fit, y_fit, weights = base.prepare_strategy_training_data(
                data.train, strategy, strategy_param, seed=data.seed
            )
            model = base.make_engine(engine_name, engine_param, seed=data.seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if weights is None:
                    model.fit(x_fit, y_fit)
                else:
                    model.fit(x_fit, y_fit, sample_weight=weights)
            raw_val = base.model_score(model, data.val.X)
            score_val = (
                _case_control_log_density_ratio(raw_val, alpha_train)
                if strategy == "ProbCorrection"
                else raw_val
            )
            val_auc = base.pu_proxy_auc(data.val.s, score_val)
            candidate = {
                "strategy_param": dict(strategy_param),
                "engine_param": dict(engine_param),
                "val_auc": val_auc,
                "raw_val": raw_val,
                "score_val": score_val,
            }
            if best is None or val_auc > best["val_auc"]:
                best = candidate
    # Derive the cutoff on the untransformed engine score.  Its monotone image
    # below gives exactly the same hard labels as the Naive twin, including
    # when NumPy's quantile uses interpolation between adjacent scores.
    raw_threshold = _population_validation_threshold(best["raw_val"], data.pi)
    dev = base.PUSplit(
        X=np.vstack([data.train.X, data.val.X]),
        y_true=np.concatenate([data.train.y_true, data.val.y_true]),
        s=np.concatenate([data.train.s, data.val.s]),
    )
    x_fit, y_fit, weights = base.prepare_strategy_training_data(
        dev, strategy, best["strategy_param"], seed=data.seed
    )
    final = base.make_engine(engine_name, best["engine_param"], seed=data.seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if weights is None:
            final.fit(x_fit, y_fit)
        else:
            final.fit(x_fit, y_fit, sample_weight=weights)
    raw_test = base.model_score(final, data.test.X)
    alpha_dev = float(dev.s.mean())
    threshold = (
        float(_case_control_log_density_ratio([raw_threshold], alpha_dev)[0])
        if strategy == "ProbCorrection"
        else raw_threshold
    )
    score_test = (
        _case_control_log_density_ratio(raw_test, alpha_dev)
        if strategy == "ProbCorrection"
        else raw_test
    )
    probability = (
        _probability_from_log_density_ratio(score_test, data.pi)
        if strategy == "ProbCorrection"
        else None
    )
    return base.evaluate_test(
        f"Block2 | {strategy} + {engine_name}",
        data.test.y_true,
        score_test,
        threshold,
        probability,
        {
            "best_params": {**best["strategy_param"], **best["engine_param"]},
            "val_metric": best["val_auc"],
            "case_control_alpha_train": alpha_train,
            "case_control_alpha_dev": alpha_dev,
            "ranking_score_type": (
                "unclipped_log_case_control_density_ratio"
                if strategy == "ProbCorrection"
                else "engine_score"
            ),
            "probability_score_type": (
                "clipped_pi_times_density_ratio"
                if strategy == "ProbCorrection"
                else "not_reported"
            ),
        },
    )


def run_block2(data):
    results = []
    for strategy in ["Naive", "ProbCorrection", "RN", "CostSensitive"]:
        for engine in ["AdaBoost", "GBM", "XGBoost", "LightGBM", "CatBoost"]:
            print(f"Running Block 2: {strategy} + {engine}")
            try:
                results.append(fit_one_boosting_cell(data, strategy, engine))
            except Exception as error:
                results.append({
                    "method": f"Block2 | {strategy} + {engine}",
                    **{metric: np.nan for metric in ["AUROC", "AUPRC", "F1", "BalancedAccuracy", "MCC", "Brier", "LogLoss", "threshold", "val_metric"]},
                    "best_params": {"error": repr(error)},
                })
    return results


def two_sample_pu_risk_components_numpy(F_pos, F_unl, pi):
    """Canonical two-sample PU logistic-risk components."""
    F_pos = np.asarray(F_pos, dtype=float)
    F_unl = np.asarray(F_unl, dtype=float)
    positive_component = pi * np.mean(np.logaddexp(0.0, -F_pos))
    raw_negative_component = (
        np.mean(np.logaddexp(0.0, F_unl))
        - pi * np.mean(np.logaddexp(0.0, F_pos))
    )
    return {
        "positive_component": float(positive_component),
        "raw_negative_component": float(raw_negative_component),
        "uPU_risk": float(positive_component + raw_negative_component),
        "nnPU_risk": float(positive_component + max(0.0, raw_negative_component)),
    }


def two_sample_nnpu_negative_gradient_numpy(F_pos, F_unl, pi):
    """Return a first-order functional negative gradient for canonical nnPU.

    The standard canonical two-sample PU loss is linear in each positive score,
    so its positive Hessian is exactly zero.  A first-order projection is used
    explicitly instead of fabricating a near-zero Newton weight.
    """
    F_pos = np.asarray(F_pos, dtype=float)
    F_unl = np.asarray(F_unl, dtype=float)
    n_pos = F_pos.size
    n_unl = F_unl.size
    if n_pos == 0 or n_unl == 0:
        raise ValueError("both P and U must be nonempty")
    components = two_sample_pu_risk_components_numpy(F_pos, F_unl, pi)
    p_pos = 1.0 / (1.0 + np.exp(-np.clip(F_pos, -30.0, 30.0)))
    p_unl = 1.0 / (1.0 + np.exp(-np.clip(F_unl, -30.0, 30.0)))
    correction_active = components["raw_negative_component"] < 0.0

    if correction_active:
        # Minimise -R_N when the unbiased negative component is below zero.
        gradient_pos = (pi / n_pos) * p_pos
        gradient_unl = -(1.0 / n_unl) * p_unl
    else:
        # d/dF_P {pi*l(+1,F_P)-pi*l(-1,F_P)} = -pi.
        gradient_pos = np.full(n_pos, -pi / n_pos, dtype=float)
        gradient_unl = (1.0 / n_unl) * p_unl

    scale = n_pos + n_unl
    negative_gradient = -scale * np.concatenate([gradient_pos, gradient_unl])
    return negative_gradient, components, correction_active


class TwoSamplePUGradientBoosting:
    """First-order functional gradient boosting for canonical two-sample nnPU."""

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
        self.init_score_ = None
        self.correction_count_ = 0
        self.minimum_negative_component_ = np.inf
        self.final_negative_component_ = np.nan
        self.training_risk_history_ = []
        self.minimum_positive_pseudo_residual_ = np.inf

    def fit(self, X_pos, X_unl, pi):
        X_pos = np.asarray(X_pos, dtype=float)
        X_unl = np.asarray(X_unl, dtype=float)
        if X_pos.ndim != 2 or X_unl.ndim != 2 or X_pos.shape[1] != X_unl.shape[1]:
            raise ValueError("P and U must be two-dimensional with equal feature count")
        if not 0.0 < pi < 1.0:
            raise ValueError("pi must lie in (0, 1)")
        clipped_pi = np.clip(pi, 1e-6, 1.0 - 1e-6)
        self.pi_ = float(pi)
        self.pi_u_ = float(pi)
        self.init_score_ = float(np.log(clipped_pi / (1.0 - clipped_pi)))
        X = np.vstack([X_pos, X_unl])
        F_pos = np.full(X_pos.shape[0], self.init_score_, dtype=float)
        F_unl = np.full(X_unl.shape[0], self.init_score_, dtype=float)
        self.trees_ = []
        self.correction_count_ = 0
        self.minimum_negative_component_ = np.inf
        self.training_risk_history_ = []
        self.minimum_positive_pseudo_residual_ = np.inf

        for iteration in range(self.n_estimators):
            pseudo_response, components, correction_active = (
                two_sample_nnpu_negative_gradient_numpy(F_pos, F_unl, pi)
            )
            if correction_active:
                self.correction_count_ += 1
            self.minimum_negative_component_ = min(
                self.minimum_negative_component_,
                components["raw_negative_component"],
            )
            self.training_risk_history_.append(components["nnPU_risk"])
            if not correction_active:
                self.minimum_positive_pseudo_residual_ = min(
                    self.minimum_positive_pseudo_residual_,
                    float(np.min(pseudo_response[: X_pos.shape[0]])),
                )
            tree = base.DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state + iteration,
            )
            # Equal projection weights are intentional: relative P/U risk
            # weights are already encoded in the per-observation gradient.
            tree.fit(X, pseudo_response)
            self.trees_.append(tree)
            F_pos += self.learning_rate * tree.predict(X_pos)
            F_unl += self.learning_rate * tree.predict(X_unl)

        self.final_negative_component_ = two_sample_pu_risk_components_numpy(
            F_pos, F_unl, pi
        )["raw_negative_component"]
        return self

    def decision_function(self, X, n_estimators=None):
        X = np.asarray(X, dtype=float)
        if n_estimators is None:
            n_estimators = len(self.trees_)
        score = np.full(X.shape[0], self.init_score_, dtype=float)
        for tree in self.trees_[:n_estimators]:
            score += self.learning_rate * tree.predict(X)
        return score


TWO_SAMPLE_PU_GB_GRID = [dict(params) for params in repaired.PU_GB_GRID]


def fit_pu_gradient_boosting(data, param_grid=TWO_SAMPLE_PU_GB_GRID):
    """Tune/refit explicit first-order canonical two-sample nnPU boosting."""
    Xp_train, Xu_train = base.get_P_U(data.train)
    best = None
    for params in param_grid:
        model = TwoSamplePUGradientBoosting(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            random_state=data.seed,
        )
        model.fit(Xp_train, Xu_train, data.pi)
        score_val = model.decision_function(data.val.X)
        val_auc = base.pu_proxy_auc(data.val.s, score_val)
        candidate = {
            "params": dict(params),
            "val_auc": val_auc,
            "score_val": score_val,
        }
        if best is None or val_auc > best["val_auc"]:
            best = candidate

    threshold = _population_validation_threshold(best["score_val"], data.pi)
    X_dev = np.vstack([data.train.X, data.val.X])
    s_dev = np.concatenate([data.train.s, data.val.s])
    Xp_dev = X_dev[s_dev == 1]
    Xu_dev = X_dev[s_dev == 0]
    final_model = TwoSamplePUGradientBoosting(
        n_estimators=best["params"]["n_estimators"],
        learning_rate=best["params"]["learning_rate"],
        max_depth=best["params"]["max_depth"],
        min_samples_leaf=best["params"]["min_samples_leaf"],
        random_state=data.seed,
    )
    final_model.fit(Xp_dev, Xu_dev, data.pi)
    score_test = final_model.decision_function(data.test.X)
    return base.evaluate_test(
        method="Block3 | PU Gradient Boosting repaired",
        y_test=data.test.y_true,
        score_test=score_test,
        threshold=threshold,
        probability=None,
        extra={
            "best_params": best["params"],
            "val_metric": best["val_auc"],
            "pi_u": data.pi,
            "nnpu_correction_count": final_model.correction_count_,
            "minimum_negative_component_train": final_model.minimum_negative_component_,
            "final_negative_component": final_model.final_negative_component_,
            "final_raw_negative_component": final_model.final_negative_component_,
            "minimum_positive_pseudo_residual": final_model.minimum_positive_pseudo_residual_,
            "pu_gb_objective": "first_order_canonical_two_sample_nnpu",
            "pu_gb_optimizer": "functional_gradient_equal_projection_weights",
        },
    )


def rank_and_summarize(raw_results):
    ranked, summary = base.rank_and_summarize(raw_results)
    if "method_id" in ranked:
        summary.insert(
            1,
            "method_id",
            summary["method"].map(METHOD_ID_BY_NAME),
        )
    return ranked, summary


def _install_two_sample_hooks():
    base.fit_oracle_logistic = fit_oracle_logistic
    base.fit_oracle_catboost = fit_oracle_catboost
    base.fit_elkan_noto_known_c = fit_case_control_known_pi
    base.fit_pu_logistic_baseline = repaired.fit_pu_logistic_baseline
    base.run_block2 = run_block2
    base.fit_adapu_one_sample = repaired.fit_adapu_one_sample
    base.fit_pu_gradient_boosting = fit_pu_gradient_boosting
    base.prior_quantile_threshold = _population_validation_threshold
    repaired.prior_quantile_threshold = _population_validation_threshold
    base.split_diagnostics = split_diagnostics
    base.print_experiment_diagnostics = print_experiment_diagnostics
    base.METHOD_NAMES_IN_FIXED_ORDER = METHOD_NAMES_IN_FIXED_ORDER
    base.METHOD_ID_BY_NAME = dict(METHOD_ID_BY_NAME)
    # Base's fail-closed fallback labels are aliases to the repaired methods.
    base.METHOD_ID_BY_NAME["Block3 | AdaPU"] = "TM026"
    base.METHOD_ID_BY_NAME["Block3 | PU Gradient Boosting"] = "TM027"


_install_two_sample_hooks()


def run_repeated_benchmark(
    *,
    structural_type,
    transform_strength=0.0,
    main_effect_shift=0.0,
    positive_variance_ratio=1.0,
    n_p_train=300,
    n_u_train=4700,
    n_p_val=300,
    n_u_val=4700,
    **kwargs,
):
    """Run the shared 27-method loop with a configured two-sample generator."""
    def configured_generator(**generator_kwargs):
        generator_kwargs.pop("n_train", None)
        generator_kwargs.pop("n_val", None)
        generator_kwargs.pop("c", None)
        return generate_two_sample_scar(
            n_p_train=n_p_train,
            n_u_train=n_u_train,
            n_p_val=n_p_val,
            n_u_val=n_u_val,
            structural_type=structural_type,
            transform_strength=transform_strength,
            main_effect_shift=main_effect_shift,
            positive_variance_ratio=positive_variance_ratio,
            **generator_kwargs,
        )

    original_generator = base.generate_one_sample_scar
    base.generate_one_sample_scar = configured_generator
    try:
        raw, summary, diagnostics, failures = base.run_repeated_benchmark(
            n_train=n_p_train + n_u_train,
            n_val=n_p_val + n_u_val,
            c=0.0,
            **kwargs,
        )
    finally:
        base.generate_one_sample_scar = original_generator

    metadata = {
        "sampling_scheme": SAMPLING_SCHEME,
        "n_p_train": n_p_train,
        "n_u_train": n_u_train,
        "n_p_val": n_p_val,
        "n_u_val": n_u_val,
        "c_applicable": False,
        "structural_type": structural_type,
        "transform_strength": transform_strength,
        "main_effect_shift": main_effect_shift,
        "positive_variance_ratio": positive_variance_ratio,
        "dgp_implementation": DGP_IMPLEMENTATION,
    }
    for frame in (raw, diagnostics):
        for key, value in metadata.items():
            frame[key] = value
    if not raw.empty:
        raw["method_id"] = raw["method"].map(METHOD_ID_BY_NAME)
        raw["pi_u_theoretical"] = raw["pi"]
    if not summary.empty:
        summary["method_id"] = summary["method"].map(METHOD_ID_BY_NAME)
    return raw, summary, diagnostics, failures


atomic_to_csv = base.atomic_to_csv
experiment_fingerprint = base.experiment_fingerprint
EXPECTED_METHOD_COUNT = 27
EXPECTED_PU_METHOD_COUNT = 25


def dependency_status():
    base_ok = base.dependency_status()
    print(f"Repaired AdaPU: {'OK' if repaired.ADAPU_AVAILABLE else 'MISSING'}")
    return bool(base_ok and repaired.ADAPU_AVAILABLE)
