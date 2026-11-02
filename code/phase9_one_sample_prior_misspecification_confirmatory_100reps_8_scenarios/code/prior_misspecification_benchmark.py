"""Run all 27 frozen one-sample SCAR methods under a supplied class prior."""

from __future__ import annotations

import numpy as np
import pandas as pd

import one_sample_benchmark as benchmark


PRIOR_FIT_IDS = {"M004", "M005", "M026", "M027"}
PRIOR_PROBABILITY_IDS = {"M004", "M005"}
PRIOR_THRESHOLD_IDS = {f"M{index:03d}" for index in range(3, 28)}
EXPECTED_RANKING_INVARIANT_IDS = (
    {f"M{index:03d}" for index in range(1, 28)} - PRIOR_FIT_IDS
)
EXPECTED_FULL_METRIC_INVARIANT_IDS = {"M001", "M002"}
EXPECTED_PROBABILITY_METRIC_INVARIANT_IDS = {
    "M001", "M002", "M003", *{f"M{index:03d}" for index in range(11, 16)}
}


def population_to_unlabeled_prior(pi: float, c: float) -> float:
    """Return P(Y=1 | S=0) implied by a population prior under SCAR."""
    return float(pi) * (1.0 - float(c)) / (1.0 - float(pi) * float(c))


def _diagnostics(data, pi_true, pi_supplied, c_true, prior_ratio):
    records = []
    for split_name, split in (
        ("train", data.train),
        ("val", data.val),
        ("test", data.test),
    ):
        record = benchmark.split_diagnostics(split, split_name)
        record.update({
            "pi_true": float(pi_true),
            "pi_supplied": float(pi_supplied),
            "pi_u_true": population_to_unlabeled_prior(pi_true, c_true),
            "pi_u_supplied": population_to_unlabeled_prior(pi_supplied, c_true),
            "c_true": float(c_true),
            "c_used_in_methods": float(data.c),
            "prior_ratio": float(prior_ratio),
            "seed": int(data.seed),
            "delta": float(data.delta),
            "delta_sq": float(data.delta_sq),
            "signal_per_active_coordinate": float(data.signal),
            "rho": float(data.rho),
        })
        records.append(record)
    return pd.DataFrame(records)


def _print_diagnostics(data):
    frame = _diagnostics(
        data,
        data.pi_true,
        data.pi_supplied,
        data.c_true,
        data.prior_ratio,
    )
    print("\n=== ONE-SAMPLE PRIOR-MISSPECIFICATION DIAGNOSTICS ===")
    print(
        f"pi_true={data.pi_true:.6f}; pi_supplied={data.pi_supplied:.6f}; "
        f"c_true=c_used={data.c_true:.6f}; ratio={data.prior_ratio:.2f}; "
        f"pi_u_true={data.pi_u_true:.6f}; pi_u_supplied={data.pi_u_supplied:.6f}"
    )
    print(frame.to_string(index=False))


def run_condition(
    *,
    rep,
    seed,
    pi_true,
    pi_supplied,
    c_true,
    prior_ratio,
    structural_type,
    transform_strength=0.0,
    main_effect_shift=0.0,
    positive_variance_ratio=1.0,
    n_train=5000,
    n_val=5000,
    n_test=10000,
    p=10,
    delta=0.90,
    n_signal=3,
    rho=0.0,
):
    """Generate at true (pi, c), then expose only pi_supplied to methods."""
    if not 0.0 < float(pi_true) < 1.0:
        raise ValueError("pi_true must lie in (0, 1)")
    if not 0.0 < float(pi_supplied) < 1.0:
        raise ValueError("pi_supplied must lie in (0, 1)")
    if not 0.0 < float(c_true) <= 1.0:
        raise ValueError("c_true must lie in (0, 1]")
    if not np.isclose(
        float(pi_supplied), float(pi_true) * float(prior_ratio), rtol=0.0, atol=1e-15
    ):
        raise ValueError("pi_supplied must equal pi_true * prior_ratio")

    data = benchmark.generate_structural_one_sample_scar(
        n_train=int(n_train),
        n_val=int(n_val),
        n_test=int(n_test),
        p=int(p),
        pi=float(pi_true),
        c=float(c_true),
        delta=float(delta),
        n_signal=int(n_signal),
        rho=float(rho),
        seed=int(seed),
        structural_type=str(structural_type),
        transform_strength=float(transform_strength),
        main_effect_shift=float(main_effect_shift),
        positive_variance_ratio=float(positive_variance_ratio),
    )
    frozen_fingerprint = benchmark.experiment_fingerprint(data)
    pi_u_true = population_to_unlabeled_prior(pi_true, c_true)
    pi_u_supplied = population_to_unlabeled_prior(pi_supplied, c_true)
    data.pi_true = float(pi_true)
    data.pi_supplied = float(pi_supplied)
    data.c_true = float(c_true)
    data.prior_ratio = float(prior_ratio)
    data.pi_u_true = pi_u_true
    data.pi_u_supplied = pi_u_supplied

    # Mutation occurs only after data generation. All fitting and threshold
    # routines read data.pi, while the frozen X/Y/S arrays remain unchanged.
    data.pi = float(pi_supplied)
    data.c = float(c_true)

    original_generator = benchmark.base.generate_one_sample_scar
    original_printer = benchmark.base.print_experiment_diagnostics

    def frozen_generator(**_ignored):
        return data

    benchmark.base.generate_one_sample_scar = frozen_generator
    benchmark.base.print_experiment_diagnostics = _print_diagnostics
    try:
        fitted_data, results = benchmark.base.run_full_benchmark(
            n_train=int(n_train),
            n_val=int(n_val),
            n_test=int(n_test),
            p=int(p),
            pi=float(pi_supplied),
            c=float(c_true),
            delta=float(delta),
            n_signal=int(n_signal),
            rho=float(rho),
            seed=int(seed),
        )
    finally:
        benchmark.base.generate_one_sample_scar = original_generator
        benchmark.base.print_experiment_diagnostics = original_printer

    if fitted_data is not data:
        raise RuntimeError("Frozen generated dataset was not used")
    if benchmark.experiment_fingerprint(data) != frozen_fingerprint:
        raise RuntimeError("A method modified the frozen dataset")
    if not np.isclose(data.c, c_true, rtol=0.0, atol=0.0):
        raise RuntimeError("The true SCAR labeling probability c was modified")

    results = results.copy()
    results.insert(0, "rep", int(rep))
    results.insert(1, "seed", int(seed))
    results["pi_true"] = float(pi_true)
    results["pi_supplied"] = float(pi_supplied)
    results["prior_ratio"] = float(prior_ratio)
    results["relative_prior_error"] = float(prior_ratio) - 1.0
    results["prior_is_correct"] = bool(np.isclose(prior_ratio, 1.0))
    results["c_true"] = float(c_true)
    results["c_used_in_methods"] = float(data.c)
    results["c_is_fixed_true"] = True
    results["pi_u_true"] = pi_u_true
    results["pi_u_supplied"] = pi_u_supplied
    results["data_fingerprint"] = frozen_fingerprint
    results["prior_used_in_model_fit"] = results["method_id"].isin(PRIOR_FIT_IDS)
    results["prior_used_in_probability"] = results["method_id"].isin(
        PRIOR_PROBABILITY_IDS
    )
    results["prior_used_in_threshold"] = results["method_id"].isin(
        PRIOR_THRESHOLD_IDS
    )
    results["probability_correction_uses_c_only"] = results["method_id"].isin(
        {"M003", *{f"M{index:03d}" for index in range(11, 16)}}
    )
    results["expected_ranking_invariant"] = results["method_id"].isin(
        EXPECTED_RANKING_INVARIANT_IDS
    )
    results["expected_full_metric_invariant"] = results["method_id"].isin(
        EXPECTED_FULL_METRIC_INVARIANT_IDS
    )
    results["expected_probability_metric_invariant"] = results["method_id"].isin(
        EXPECTED_PROBABILITY_METRIC_INVARIANT_IDS
    )
    results["n_train"] = int(n_train)
    results["n_val"] = int(n_val)
    results["n_test"] = int(n_test)
    results["p"] = int(p)
    results["delta"] = float(data.delta)
    results["delta_sq"] = float(data.delta_sq)
    results["signal_per_active_coordinate"] = float(data.signal)
    results["n_signal"] = int(n_signal)
    results["rho"] = float(rho)
    results["sampling_scheme"] = "one_sample_scar"
    results["structural_type"] = str(structural_type)
    results["transform_strength"] = float(transform_strength)
    results["main_effect_shift"] = float(main_effect_shift)
    results["positive_variance_ratio"] = float(positive_variance_ratio)
    results["dgp_implementation"] = "phase3_frozen_one_sample_scar_v1"

    diagnostics = _diagnostics(data, pi_true, pi_supplied, c_true, prior_ratio)
    diagnostics.insert(0, "rep", int(rep))
    diagnostics["data_fingerprint"] = frozen_fingerprint
    return results, diagnostics, data


EXPECTED_METHOD_COUNT = 27
