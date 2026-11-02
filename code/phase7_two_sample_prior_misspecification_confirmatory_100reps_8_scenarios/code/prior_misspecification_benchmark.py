"""Run all 27 frozen methods with a supplied prior on data from a true prior."""

from __future__ import annotations

import numpy as np
import pandas as pd

import two_sample_benchmark as benchmark


PRIOR_FIT_IDS = {"TM004", "TM005", "TM026", "TM027"}
PRIOR_PROBABILITY_IDS = {
    "TM003", "TM004", "TM005", "TM011", "TM012", "TM013", "TM014", "TM015"
}
PRIOR_THRESHOLD_IDS = {f"TM{index:03d}" for index in range(3, 28)}
EXPECTED_RANKING_INVARIANT_IDS = (
    {f"TM{index:03d}" for index in range(1, 28)} - PRIOR_FIT_IDS
)
EXPECTED_FULL_METRIC_INVARIANT_IDS = {"TM001", "TM002"}


def _diagnostics(data, pi_true, pi_supplied, prior_ratio):
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
        data.prior_ratio,
    )
    print("\n=== TWO-SAMPLE PRIOR-MISSPECIFICATION DIAGNOSTICS ===")
    print(
        f"pi_true={data.pi_true:.6f}; pi_supplied={data.pi_supplied:.6f}; "
        f"ratio={data.prior_ratio:.2f}; "
        f"nP/nU={data.n_p_train}/{data.n_u_train}"
    )
    print(frame.to_string(index=False))


def run_condition(
    *,
    rep,
    seed,
    pi_true,
    pi_supplied,
    prior_ratio,
    structural_type,
    transform_strength=0.0,
    main_effect_shift=0.0,
    positive_variance_ratio=1.0,
    n_p_train=300,
    n_u_train=4700,
    n_p_val=300,
    n_u_val=4700,
    n_test=10000,
    p=10,
    delta=0.90,
    n_signal=3,
    rho=0.0,
):
    """Generate at ``pi_true`` once, then expose only ``pi_supplied`` to methods."""
    if not 0.0 < float(pi_true) < 1.0:
        raise ValueError("pi_true must lie in (0, 1)")
    if not 0.0 < float(pi_supplied) < 1.0:
        raise ValueError("pi_supplied must lie in (0, 1)")
    if not np.isclose(float(pi_supplied), float(pi_true) * float(prior_ratio)):
        raise ValueError("pi_supplied must equal pi_true * prior_ratio")

    data = benchmark.generate_two_sample_scar(
        n_p_train=int(n_p_train),
        n_u_train=int(n_u_train),
        n_p_val=int(n_p_val),
        n_u_val=int(n_u_val),
        n_test=int(n_test),
        p=int(p),
        pi=float(pi_true),
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
    data.pi_true = float(pi_true)
    data.pi_supplied = float(pi_supplied)
    data.prior_ratio = float(prior_ratio)
    # All fitting, posterior conversion and PU threshold code reads data.pi.
    # Mutation happens only after data generation; the arrays and true labels
    # remain those generated under pi_true.
    data.pi = float(pi_supplied)

    original_generator = benchmark.base.generate_one_sample_scar
    original_printer = benchmark.base.print_experiment_diagnostics

    def frozen_generator(**_ignored):
        return data

    benchmark.base.generate_one_sample_scar = frozen_generator
    benchmark.base.print_experiment_diagnostics = _print_diagnostics
    try:
        fitted_data, results = benchmark.base.run_full_benchmark(
            n_train=int(n_p_train) + int(n_u_train),
            n_val=int(n_p_val) + int(n_u_val),
            n_test=int(n_test),
            p=int(p),
            pi=float(pi_supplied),
            c=0.0,
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

    results = results.copy()
    results.insert(0, "rep", int(rep))
    results.insert(1, "seed", int(seed))
    results["pi_true"] = float(pi_true)
    results["pi_supplied"] = float(pi_supplied)
    results["prior_ratio"] = float(prior_ratio)
    results["relative_prior_error"] = float(prior_ratio) - 1.0
    results["prior_is_correct"] = bool(np.isclose(prior_ratio, 1.0))
    results["pi_u_theoretical_true"] = float(pi_true)
    results["pi_u_supplied"] = float(pi_supplied)
    results["data_fingerprint"] = frozen_fingerprint
    results["prior_used_in_model_fit"] = results["method_id"].isin(PRIOR_FIT_IDS)
    results["prior_used_in_probability"] = results["method_id"].isin(
        PRIOR_PROBABILITY_IDS
    )
    results["prior_used_in_threshold"] = results["method_id"].isin(
        PRIOR_THRESHOLD_IDS
    )
    results["expected_ranking_invariant"] = results["method_id"].isin(
        EXPECTED_RANKING_INVARIANT_IDS
    )
    results["expected_full_metric_invariant"] = results["method_id"].isin(
        EXPECTED_FULL_METRIC_INVARIANT_IDS
    )
    results["n_p_train"] = int(n_p_train)
    results["n_u_train"] = int(n_u_train)
    results["n_p_val"] = int(n_p_val)
    results["n_u_val"] = int(n_u_val)
    results["n_test"] = int(n_test)
    results["p"] = int(p)
    results["delta"] = float(data.delta)
    results["delta_sq"] = float(data.delta_sq)
    results["signal_per_active_coordinate"] = float(data.signal)
    results["n_signal"] = int(n_signal)
    results["rho"] = float(rho)
    results["c_applicable"] = False
    results["sampling_scheme"] = benchmark.SAMPLING_SCHEME
    results["structural_type"] = str(structural_type)
    results["transform_strength"] = float(transform_strength)
    results["main_effect_shift"] = float(main_effect_shift)
    results["positive_variance_ratio"] = float(positive_variance_ratio)
    results["dgp_implementation"] = benchmark.DGP_IMPLEMENTATION
    diagnostics = _diagnostics(data, pi_true, pi_supplied, prior_ratio)
    diagnostics.insert(0, "rep", int(rep))
    diagnostics["data_fingerprint"] = frozen_fingerprint
    return results, diagnostics, data


EXPECTED_METHOD_COUNT = 27

