"""Fast integration smoke for all 27 methods under a misspecified prior."""

from __future__ import annotations

import numpy as np

import prior_misspecification_benchmark as pilot
import prior_misspecification_runner as runner


def main():
    benchmark = pilot.benchmark
    base = benchmark.base
    repaired = benchmark.repaired
    for engine in list(base.ENGINE_GRIDS):
        base.ENGINE_GRIDS[engine] = base.ENGINE_GRIDS[engine][:1]
    base.STRATEGY_GRIDS["RN"] = base.STRATEGY_GRIDS["RN"][:1]
    base.STRATEGY_GRIDS["CostSensitive"] = base.STRATEGY_GRIDS["CostSensitive"][:1]
    base.fit_pu_logistic_baseline = lambda data, non_negative=False, beta=0.0, gamma=1.0: repaired.fit_pu_logistic_baseline(
        data,
        non_negative=non_negative,
        beta=beta,
        gamma=gamma,
        l2_grid=(1e-4,),
        max_epochs=80,
        check_every=10,
        patience_checks=3,
    )
    base.fit_adapu_one_sample = lambda data: repaired.fit_adapu_one_sample(
        data, beta_grid=(1e-2,), n_estimators_grid=(20,)
    )
    tiny_gb_grid = [{
        "n_estimators": 20,
        "learning_rate": 0.05,
        "max_depth": 1,
        "min_samples_leaf": 10,
    }]
    base.fit_pu_gradient_boosting = lambda data: benchmark.fit_pu_gradient_boosting(
        data, param_grid=tiny_gb_grid
    )
    raw, diagnostics, _ = pilot.run_condition(
        rep=1,
        seed=777000,
        pi_true=0.2,
        pi_supplied=0.16,
        prior_ratio=0.8,
        structural_type="linear_gaussian",
        n_p_train=80,
        n_u_train=320,
        n_p_val=80,
        n_u_val=320,
        n_test=600,
        p=10,
        delta=0.9,
        n_signal=3,
        rho=0.0,
    )
    if len(raw) != 27 or raw["method_id"].nunique() != 27:
        raise RuntimeError("smoke did not return all 27 methods")
    if raw["data_fingerprint"].nunique() != 1:
        raise RuntimeError("methods did not share one dataset")
    if np.isfinite(raw["AUROC"]).sum() != 27:
        raise RuntimeError("one or more methods has missing AUROC")
    if len(diagnostics) != 3:
        raise RuntimeError("expected three split diagnostics")
    issues = runner.repaired_method_issues(raw)
    issues.extend(runner._direct_prior_routing_issues(raw, 0.16))
    if issues:
        raise RuntimeError("; ".join(issues))
    if not raw["pi_true"].eq(0.2).all() or not raw["pi_supplied"].eq(0.16).all():
        raise RuntimeError("true and supplied prior tags are incorrect")
    print("PASS: all 27 methods received one shared dataset and the intended supplied prior")


if __name__ == "__main__":
    main()
