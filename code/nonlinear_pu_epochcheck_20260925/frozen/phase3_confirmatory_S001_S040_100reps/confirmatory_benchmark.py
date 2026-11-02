"""Confirmatory one-sample-SCAR benchmark with the frozen 27-method catalog.

The original Phase 1 implementation is vendored as ``base_phase1_benchmark``.
Only three method hooks are replaced: linear uPU/nnPU receive the repair-pilot
diagnostics, and stable method IDs M026/M027 use the repaired AdaPU and PU
gradient boosting implementations.  All other fitted methods are unchanged.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
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
_ORIGINAL_GENERATOR = base.generate_one_sample_scar


# Install the repaired algorithms while retaining stable catalog IDs M001-M027.
base.fit_pu_logistic_baseline = repaired.fit_pu_logistic_baseline
base.fit_adapu_one_sample = repaired.fit_adapu_one_sample
base.fit_pu_gradient_boosting = repaired.fit_pu_gradient_boosting
base.METHOD_ID_BY_NAME.update(
    {
        "Block3 | AdaPU repaired": "M026",
        "Block3 | PU Gradient Boosting repaired": "M027",
    }
)


def _validate_common(p: int, pi: float, c: float, delta: float) -> None:
    if not (0.0 < pi < 1.0):
        raise ValueError("pi must lie in (0, 1)")
    if not (0.0 < c <= 1.0):
        raise ValueError("c must lie in (0, 1]")
    if p < 5:
        raise ValueError("Structural scenarios require at least five features")
    if delta <= 0.0:
        raise ValueError("delta must be positive")


def _latent_equal_covariance_split(
    *,
    n: int,
    p: int,
    pi: float,
    c: float,
    mu: np.ndarray,
    rng: np.random.Generator,
    transform: Callable[[np.ndarray], np.ndarray],
) -> tuple[base.PUSplit, np.ndarray]:
    y = rng.binomial(1, pi, size=n).astype(np.int64)
    z = np.empty((n, p), dtype=np.float64)
    positive = y == 1
    negative = ~positive
    z[positive] = rng.normal(size=(int(positive.sum()), p)) + mu
    z[negative] = rng.normal(size=(int(negative.sum()), p))
    x = transform(z)
    s = np.zeros(n, dtype=np.int64)
    s[positive] = rng.binomial(1, c, size=int(positive.sum())).astype(np.int64)
    bayes_score = z @ mu - 0.5 * float(mu @ mu)
    return base.PUSplit(X=x, y_true=y, s=s), bayes_score


def _unequal_covariance_split(
    *,
    n: int,
    p: int,
    pi: float,
    c: float,
    mu: np.ndarray,
    positive_variance: np.ndarray,
    rng: np.random.Generator,
) -> tuple[base.PUSplit, np.ndarray]:
    y = rng.binomial(1, pi, size=n).astype(np.int64)
    x = np.empty((n, p), dtype=np.float64)
    positive = y == 1
    negative = ~positive
    x[positive] = (
        rng.normal(size=(int(positive.sum()), p)) * np.sqrt(positive_variance)
        + mu
    )
    x[negative] = rng.normal(size=(int(negative.sum()), p))
    s = np.zeros(n, dtype=np.int64)
    s[positive] = rng.binomial(1, c, size=int(positive.sum())).astype(np.int64)

    # log f_1(x) - log f_0(x); the class-prior constant does not affect AUROC.
    q0 = np.sum(x * x, axis=1)
    q1 = np.sum((x - mu) ** 2 / positive_variance, axis=1)
    bayes_score = 0.5 * (q0 - q1 - np.log(positive_variance).sum())
    return base.PUSplit(X=x, y_true=y, s=s), bayes_score


def _mixed_interaction_split(
    *,
    n: int,
    p: int,
    pi: float,
    c: float,
    interaction_shift: float,
    main_effect_shift: float,
    rng: np.random.Generator,
) -> tuple[base.PUSplit, np.ndarray]:
    """Generate a two-feature XOR mixture plus one independent main effect."""
    y = rng.binomial(1, pi, size=n).astype(np.int64)
    x = rng.normal(size=(n, p))
    component_sign = rng.choice(np.array([-1.0, 1.0]), size=n)
    x[:, 0] += interaction_shift * component_sign
    x[:, 1] += interaction_shift * component_sign * np.where(y == 1, 1.0, -1.0)
    x[:, 2] += main_effect_shift * y
    positive = y == 1
    s = np.zeros(n, dtype=np.int64)
    s[positive] = rng.binomial(1, c, size=int(positive.sum())).astype(np.int64)

    positive_centres = np.array(
        [[interaction_shift, interaction_shift], [-interaction_shift, -interaction_shift]]
    )
    negative_centres = np.array(
        [[interaction_shift, -interaction_shift], [-interaction_shift, interaction_shift]]
    )
    log_positive = logsumexp(
        np.stack(
            [
                -0.5 * np.sum((x[:, :2] - centre) ** 2, axis=1)
                for centre in positive_centres
            ]
        ),
        axis=0,
    ) - math.log(2.0)
    log_negative = logsumexp(
        np.stack(
            [
                -0.5 * np.sum((x[:, :2] - centre) ** 2, axis=1)
                for centre in negative_centres
            ]
        ),
        axis=0,
    ) - math.log(2.0)
    bayes_score = (
        log_positive
        - log_negative
        + main_effect_shift * x[:, 2]
        - 0.5 * main_effect_shift**2
    )
    return base.PUSplit(X=x, y_true=y, s=s), bayes_score


def generate_structural_one_sample_scar(
    *,
    structural_type: str,
    n_train: int = 5000,
    n_val: int = 5000,
    n_test: int = 10000,
    p: int = 10,
    pi: float = 0.20,
    c: float = 0.30,
    delta: float = 0.90,
    n_signal: int = 3,
    rho: float = 0.0,
    seed: int = 123,
    transform_strength: float = 0.0,
    positive_variance_ratio: float = 1.0,
    main_effect_shift: float = 0.0,
) -> base.PUExperiment:
    """Generate one of the frozen S001--S040 confirmatory populations."""
    if structural_type not in STRUCTURAL_TYPES:
        raise ValueError(f"Unknown structural_type: {structural_type}")
    _validate_common(p, pi, c, delta)
    if structural_type == "linear_gaussian":
        # This is deliberately a direct call to the original Phase 1
        # generator.  It preserves every DGP detail for S001--S037 (including
        # S026) while the wrapper keeps one common runner for all scenarios.
        data = _ORIGINAL_GENERATOR(
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
        data.structural_type = structural_type
        data.transform_strength = 0.0
        data.positive_variance_ratio = 1.0
        data.main_effect_shift = 0.0
        return data

    if n_signal != 3:
        raise ValueError("The structural pilot fixes n_signal=3")
    if not np.isclose(rho, 0.0):
        raise ValueError("The structural pilot fixes rho=0")

    if structural_type == "many_irrelevant_variables":
        if p != 500:
            raise ValueError("many_irrelevant_variables requires p=500")
        data = _ORIGINAL_GENERATOR(
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
        mu = np.zeros(p, dtype=float)
        mu[:3] = delta / math.sqrt(3.0)
        data.bayes_scores = {
            name: split.X @ mu - 0.5 * float(mu @ mu)
            for name, split in (
                ("train", data.train),
                ("val", data.val),
                ("test", data.test),
            )
        }
        data.structural_type = structural_type
        data.transform_strength = 0.0
        data.positive_variance_ratio = 1.0
        data.main_effect_shift = 0.0
        return data

    if p != 10:
        raise ValueError(f"{structural_type} requires p=10")
    rng = np.random.default_rng(seed)
    mu = np.zeros(p, dtype=float)

    if structural_type == "curved_boundary":
        if transform_strength <= 0.0:
            raise ValueError("curved_boundary requires transform_strength > 0")
        mu[[1, 2, 3]] = delta / math.sqrt(3.0)

        def transform(z: np.ndarray) -> np.ndarray:
            x = z.copy()
            x[:, 1] = z[:, 1] + transform_strength * (z[:, 0] ** 2 - 1.0)
            return x

        split_factory = lambda n: _latent_equal_covariance_split(
            n=n, p=p, pi=pi, c=c, mu=mu, rng=rng, transform=transform
        )

    elif structural_type == "multiplicative_interaction":
        if transform_strength <= 0.0:
            raise ValueError(
                "multiplicative_interaction requires transform_strength > 0"
            )
        if main_effect_shift <= 0.0:
            raise ValueError(
                "multiplicative_interaction requires main_effect_shift > 0"
            )
        split_factory = lambda n: _mixed_interaction_split(
            n=n,
            p=p,
            pi=pi,
            c=c,
            interaction_shift=transform_strength,
            main_effect_shift=main_effect_shift,
            rng=rng,
        )

    else:
        if positive_variance_ratio <= 1.0:
            raise ValueError("unequal_covariance requires a variance ratio > 1")
        mu[:3] = delta / math.sqrt(3.0)
        positive_variance = np.ones(p, dtype=float)
        positive_variance[:3] = positive_variance_ratio
        split_factory = lambda n: _unequal_covariance_split(
            n=n,
            p=p,
            pi=pi,
            c=c,
            mu=mu,
            positive_variance=positive_variance,
            rng=rng,
        )

    train, bayes_train = split_factory(n_train)
    val, bayes_val = split_factory(n_val)
    test, bayes_test = split_factory(n_test)
    data = base.PUExperiment(
        train=train,
        val=val,
        test=test,
        pi=pi,
        c=c,
        p=p,
        delta=delta,
        delta_sq=delta**2,
        signal=delta / math.sqrt(3.0),
        n_signal=3,
        rho=0.0,
        seed=seed,
    )
    data.bayes_scores = {"train": bayes_train, "val": bayes_val, "test": bayes_test}
    data.structural_type = structural_type
    data.transform_strength = float(transform_strength)
    data.positive_variance_ratio = float(positive_variance_ratio)
    data.main_effect_shift = float(main_effect_shift)
    return data


def bayes_test_auroc(data: base.PUExperiment) -> float:
    return float(base.roc_auc_score(data.test.y_true, data.bayes_scores["test"]))


def run_repeated_benchmark(
    *,
    structural_type: str,
    transform_strength: float = 0.0,
    positive_variance_ratio: float = 1.0,
    main_effect_shift: float = 0.0,
    **kwargs,
):
    """Run the original 27-method loop with one structural generator installed."""
    def configured_generator(**generator_kwargs):
        return generate_structural_one_sample_scar(
            structural_type=structural_type,
            transform_strength=transform_strength,
            positive_variance_ratio=positive_variance_ratio,
            main_effect_shift=main_effect_shift,
            **generator_kwargs,
        )

    original_generator = base.generate_one_sample_scar
    base.generate_one_sample_scar = configured_generator
    try:
        return base.run_repeated_benchmark(**kwargs)
    finally:
        base.generate_one_sample_scar = original_generator


# Runner-facing exports.
atomic_to_csv = base.atomic_to_csv
experiment_fingerprint = base.experiment_fingerprint
rank_and_summarize = base.rank_and_summarize
split_diagnostics = base.split_diagnostics
EXPECTED_METHOD_COUNT = base.EXPECTED_METHOD_COUNT
EXPECTED_PU_METHOD_COUNT = base.EXPECTED_PU_METHOD_COUNT


def dependency_status() -> bool:
    base_ok = base.dependency_status()
    print(f"Repaired AdaPU: {'OK' if repaired.ADAPU_AVAILABLE else 'MISSING'}")
    return bool(base_ok and repaired.ADAPU_AVAILABLE)
