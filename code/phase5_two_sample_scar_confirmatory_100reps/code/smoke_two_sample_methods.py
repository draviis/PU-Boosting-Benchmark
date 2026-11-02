"""Fast local integration smoke for all 27 methods (reduced tuning grids only)."""

from __future__ import annotations

import numpy as np

import two_sample_benchmark as benchmark
import two_sample_runner as runner


def main():
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
    raw, _, diagnostics, failures = benchmark.run_repeated_benchmark(
        structural_type="linear_gaussian",
        n_reps=1,
        rep_ids=[1],
        base_seed=777000,
        n_p_train=80,
        n_u_train=320,
        n_p_val=80,
        n_u_val=320,
        n_test=600,
        p=10,
        pi=0.2,
        delta=0.9,
        n_signal=3,
        rho=0.0,
    )
    if not failures.empty:
        raise RuntimeError(failures.to_dict("records"))
    if len(raw) != 27 or raw["method_id"].nunique() != 27:
        raise RuntimeError("smoke did not return all 27 methods")
    if raw["data_fingerprint"].nunique() != 1:
        raise RuntimeError("methods did not share one dataset")
    if np.isfinite(raw["AUROC"]).sum() != 27:
        raise RuntimeError(raw.loc[~np.isfinite(raw["AUROC"]), ["method_id", "method", "best_params"]].to_dict("records"))
    if len(diagnostics) != 3:
        raise RuntimeError("expected three split diagnostics")
    for naive_id, corrected_id in zip(
        [f"TM{i:03d}" for i in range(6, 11)],
        [f"TM{i:03d}" for i in range(11, 16)],
    ):
        naive = raw.loc[raw["method_id"].eq(naive_id)].iloc[0]
        corrected = raw.loc[raw["method_id"].eq(corrected_id)].iloc[0]
        for metric in ["AUROC", "AUPRC", "F1", "BalancedAccuracy", "MCC"]:
            if not np.isclose(naive[metric], corrected[metric], rtol=0.0, atol=1e-12):
                raise RuntimeError(
                    f"{naive_id}/{corrected_id} ranking/classification mismatch for {metric}"
                )
        if naive["best_params"] != corrected["best_params"]:
            raise RuntimeError(f"{naive_id}/{corrected_id} selected different parameters")
    tm003 = raw.loc[raw["method_id"].eq("TM003")].iloc[0]
    if tm003["ranking_score_type"] != "unclipped_log_case_control_density_ratio":
        raise RuntimeError("TM003 did not use the repaired ranking score")
    tm027 = raw.loc[raw["method_id"].eq("TM027")].iloc[0]
    if tm027["pu_gb_objective"] != "first_order_canonical_two_sample_nnpu":
        raise RuntimeError("TM027 did not use the explicit first-order objective")
    if not tm027["minimum_positive_pseudo_residual"] > 0.0:
        raise RuntimeError("TM027 positive observations had no first-order influence")
    release_issues = runner.repaired_method_issues(raw)
    if release_issues:
        raise RuntimeError("task-level repaired-method gate failed: " + "; ".join(release_issues))
    print("PASS: all 27 methods completed on one shared two-sample dataset")


if __name__ == "__main__":
    main()
