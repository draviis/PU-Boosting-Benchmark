"""Fast validation tests for the Phase 1 method-repair pilot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
ADAPU_SOURCE = PROJECT_DIR / "Ada-PU" / "src" / "decisionstump.pyx"

sys.path.insert(0, str(CODE_DIR))
import repaired_method_library as benchmark  # noqa: E402


def assert_close(left, right, tolerance=1e-8, message="values differ"):
    if not np.allclose(left, right, rtol=tolerance, atol=tolerance, equal_nan=True):
        raise AssertionError(f"{message}: {left!r} != {right!r}")


def test_adapu_source_repair() -> None:
    source = ADAPU_SOURCE.read_text(encoding="utf-8")
    assert "datetime.datetime.now" not in source
    assert "np.random.seed(time)" not in source
    assert (
        "feature_min, feature_max, lower_bound, higher_bound, step)" in source
    )
    print("PASS: AdaPU source contains no wall-clock reseeding and uses current step")


def test_pu_gradient_and_hessian() -> None:
    F_pos = np.array([-0.4, 0.2, 0.8], dtype=float)
    F_unl = np.array([-1.0, -0.1, 0.5, 1.1], dtype=float)
    pi = 0.2
    c = 0.3
    grad_p, grad_u, hess_p, hess_u = (
        benchmark.pu_gb_standard_gradient_hessian_numpy(
            F_pos, F_unl, pi, c
        )
    )
    epsilon = 1e-5

    def risk(pos, unl):
        return benchmark.pu_gb_risk_components_numpy(pos, unl, pi, c)[
            "uPU_risk"
        ]

    numerical_grad_p = []
    numerical_hess_p = []
    for index in range(F_pos.size):
        plus = F_pos.copy()
        minus = F_pos.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        numerical_grad_p.append(
            (risk(plus, F_unl) - risk(minus, F_unl)) / (2.0 * epsilon)
        )
        numerical_hess_p.append(
            (risk(plus, F_unl) - 2.0 * risk(F_pos, F_unl) + risk(minus, F_unl))
            / (epsilon**2)
        )

    numerical_grad_u = []
    numerical_hess_u = []
    for index in range(F_unl.size):
        plus = F_unl.copy()
        minus = F_unl.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        numerical_grad_u.append(
            (risk(F_pos, plus) - risk(F_pos, minus)) / (2.0 * epsilon)
        )
        numerical_hess_u.append(
            (risk(F_pos, plus) - 2.0 * risk(F_pos, F_unl) + risk(F_pos, minus))
            / (epsilon**2)
        )

    assert_close(grad_p, numerical_grad_p, 1e-6, "P gradients")
    assert_close(grad_u, numerical_grad_u, 1e-6, "U gradients")
    assert_close(hess_p, numerical_hess_p, 2e-4, "P Hessians")
    assert_close(hess_u, numerical_hess_u, 2e-4, "U Hessians")
    print("PASS: repaired PU-GB gradients and Hessians match finite differences")


def test_pu_gradient_boosting_determinism() -> None:
    rng = np.random.default_rng(20260820)
    X_pos = rng.normal(loc=0.6, size=(80, 5))
    X_unl = rng.normal(size=(320, 5))
    X_test = rng.normal(size=(60, 5))
    parameters = dict(
        n_estimators=20,
        learning_rate=0.05,
        max_depth=1,
        min_samples_leaf=10,
        random_state=77,
    )
    first = benchmark.PUGradientBoosting(**parameters).fit(
        X_pos, X_unl, pi=0.2, c=0.3
    )
    second = benchmark.PUGradientBoosting(**parameters).fit(
        X_pos, X_unl, pi=0.2, c=0.3
    )
    assert_close(
        first.decision_function(X_test),
        second.decision_function(X_test),
        0.0,
        "PU-GB predictions",
    )
    assert np.isfinite(first.minimum_negative_component_)
    print("PASS: repaired PU-GB is deterministic and records risk diagnostics")


def small_experiment():
    return benchmark.generate_one_sample_scar(
        n_train=600,
        n_val=600,
        n_test=1000,
        p=5,
        pi=0.2,
        c=0.3,
        delta=0.9,
        n_signal=3,
        rho=0.0,
        seed=9917,
    )


def test_linear_upu_nnpu_diagnostic() -> None:
    data = small_experiment()
    common = dict(
        l2_grid=(1e-3,),
        max_epochs=60,
        check_every=10,
        patience_checks=4,
    )
    upu = benchmark.fit_pu_logistic_baseline(
        data, non_negative=False, **common
    )
    nnpu = benchmark.fit_pu_logistic_baseline(
        data, non_negative=True, beta=0.0, gamma=1.0, **common
    )
    if nnpu["nnpu_correction_count"] == 0:
        assert_close(upu["AUROC"], nnpu["AUROC"], 0.0, "uPU/nnPU AUROC")
    assert "minimum_negative_component_train" in nnpu
    assert "minimum_negative_component_tuning" in nnpu
    print("PASS: linear uPU/nnPU equivalence is explicitly diagnosed")


def test_adapu_determinism() -> None:
    if not benchmark.ADAPU_AVAILABLE:
        raise AssertionError(
            "Repaired AdaPU extension is unavailable; compile Ada-PU/setup.py first"
        )
    data = small_experiment()
    arguments = dict(
        beta_grid=(1e-2,),
        n_estimators_grid=(8,),
        random_selection=1,
    )
    first = benchmark.fit_adapu_one_sample(data, **arguments)
    second = benchmark.fit_adapu_one_sample(data, **arguments)
    for key in ["AUROC", "AUPRC", "threshold", "val_metric"]:
        assert_close(first[key], second[key], 0.0, f"AdaPU {key}")
    assert first["adapu_rng_mode"] == "seeded_numpy_no_wall_clock_reseed"
    print("PASS: repaired AdaPU produces identical results for identical seeds")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--skip-adapu", action="store_true")
    args = parser.parse_args()

    test_adapu_source_repair()
    test_pu_gradient_and_hessian()
    if not args.static_only:
        test_pu_gradient_boosting_determinism()
        test_linear_upu_nnpu_diagnostic()
        if not args.skip_adapu:
            test_adapu_determinism()
    print("ALL METHOD-REPAIR TESTS PASSED")


if __name__ == "__main__":
    main()
