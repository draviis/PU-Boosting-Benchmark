"""Fail-closed analysis for the 100-repetition confirmatory prior study."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import prior_misspecification_benchmark as pilot


PROJECT_DIR = Path(__file__).resolve().parent.parent
COMBINED = PROJECT_DIR / "results" / "prior_misspecification_combined"
OUTPUT = PROJECT_DIR / "results" / "prior_misspecification_analysis"
RAW = COMBINED / "prior_misspecification_raw.csv"
DIAGNOSTICS = COMBINED / "prior_misspecification_diagnostics.csv"
N_REPETITIONS = 100
EXPECTED_TASKS = 8 * 5 * N_REPETITIONS
EXPECTED_ROWS = EXPECTED_TASKS * 27
METRICS = ["AUROC", "AUPRC", "F1", "BalancedAccuracy", "MCC", "Brier", "LogLoss"]
RANKING_METRICS = ["AUROC", "AUPRC"]
HIGHER_IS_BETTER = {"AUROC", "AUPRC", "F1", "BalancedAccuracy", "MCC"}
INVARIANCE_TOLERANCE = 5e-5
REPAIRED_RANKING_IDS = ["TM003", *[f"TM{i:03d}" for i in range(11, 16)]]
NAIVE_CORRECTED_PAIRS = list(zip(
    [f"TM{i:03d}" for i in range(6, 11)],
    [f"TM{i:03d}" for i in range(11, 16)],
))


def atomic_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def atomic_json(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _metric_summary(raw):
    grouping = [
        "scenario_id", "source_scenario_id", "scenario_label", "prior_ratio",
        "pi_true", "pi_supplied", "method_id", "method", "method_block",
        "comparison_role", "prior_used_in_model_fit",
        "prior_used_in_probability", "prior_used_in_threshold",
    ]
    summary = raw.groupby(grouping, as_index=False).agg(
        n=("rep", "nunique"),
        **{
            f"{metric}_{stat}": (metric, stat)
            for metric in METRICS
            for stat in ("mean", "std")
        },
    )
    for metric in METRICS:
        summary[f"{metric}_se"] = summary[f"{metric}_std"] / np.sqrt(summary["n"])
        summary[f"{metric}_ci95_low"] = (
            summary[f"{metric}_mean"] - 1.96 * summary[f"{metric}_se"]
        )
        summary[f"{metric}_ci95_high"] = (
            summary[f"{metric}_mean"] + 1.96 * summary[f"{metric}_se"]
        )
    return summary


def _paired_changes(raw):
    key = ["scenario_id", "rep", "method_id"]
    identity = [
        "source_scenario_id", "scenario_label", "method", "method_block",
        "comparison_role", "pi_true", "prior_used_in_model_fit",
        "prior_used_in_probability", "prior_used_in_threshold",
    ]
    baseline = raw.loc[np.isclose(raw["prior_ratio"], 1.0), key + METRICS].copy()
    baseline = baseline.rename(columns={metric: f"baseline_{metric}" for metric in METRICS})
    alternatives = raw.loc[~np.isclose(raw["prior_ratio"], 1.0)].copy()
    paired = alternatives.merge(baseline, on=key, how="left", validate="many_to_one")
    records = []
    for metric in METRICS:
        current = pd.to_numeric(paired[metric], errors="coerce")
        correct = pd.to_numeric(paired[f"baseline_{metric}"], errors="coerce")
        change = current - correct
        loss = -change if metric in HIGHER_IS_BETTER else change
        frame = paired[key + identity + ["prior_ratio", "pi_supplied"]].copy()
        frame["metric"] = metric
        frame["correct_prior_value"] = correct
        frame["misspecified_prior_value"] = current
        frame["change_misspecified_minus_correct"] = change
        frame["performance_loss"] = loss
        records.append(frame)
    paired_long = pd.concat(records, ignore_index=True)
    def nonmissing_rate(values, predicate):
        usable = values.dropna()
        if usable.empty:
            return np.nan
        return float(predicate(usable).mean())

    summary = paired_long.groupby(
        [
            "scenario_id", "source_scenario_id", "scenario_label", "prior_ratio",
            "pi_true", "pi_supplied", "method_id", "method", "method_block",
            "comparison_role", "prior_used_in_model_fit",
            "prior_used_in_probability", "prior_used_in_threshold", "metric",
        ],
        as_index=False,
        dropna=False,
    ).agg(
        n=("performance_loss", "count"),
        mean_change=("change_misspecified_minus_correct", "mean"),
        mean_performance_loss=("performance_loss", "mean"),
        sd_performance_loss=("performance_loss", "std"),
        median_performance_loss=("performance_loss", "median"),
        harmed_rate=(
            "performance_loss", lambda values: nonmissing_rate(values, lambda x: x > 0)
        ),
        improved_rate=(
            "performance_loss", lambda values: nonmissing_rate(values, lambda x: x < 0)
        ),
        tie_rate=(
            "performance_loss", lambda values: nonmissing_rate(values, lambda x: x == 0)
        ),
    )
    summary["se_performance_loss"] = summary["sd_performance_loss"] / np.sqrt(summary["n"])
    summary["ci95_low"] = (
        summary["mean_performance_loss"] - 1.96 * summary["se_performance_loss"]
    )
    summary["ci95_high"] = (
        summary["mean_performance_loss"] + 1.96 * summary["se_performance_loss"]
    )
    return paired_long, summary


def main():
    raw = pd.read_csv(RAW, low_memory=False)
    diagnostics = pd.read_csv(DIAGNOSTICS, low_memory=False)
    issues = []
    if len(raw) != EXPECTED_ROWS:
        issues.append(f"expected {EXPECTED_ROWS} method rows, found {len(raw)}")
    task = raw.groupby(["condition_id", "rep"], as_index=False).agg(
        rows=("method_id", "size"),
        methods=("method_id", "nunique"),
        fingerprints=("data_fingerprint", "nunique"),
        seeds=("seed", "nunique"),
        primary_metrics=("AUROC", "count"),
    )
    task["pass"] = (
        task["rows"].eq(27)
        & task["methods"].eq(27)
        & task["fingerprints"].eq(1)
        & task["seeds"].eq(1)
        & task["primary_metrics"].eq(27)
    )
    if len(task) != EXPECTED_TASKS or not task["pass"].all():
        issues.append(f"complete tasks: {int(task['pass'].sum())}/{len(task)}")
    if set(raw["method_id"]) != {f"TM{i:03d}" for i in range(1, 28)}:
        issues.append("method catalog is not TM001--TM027")
    if raw.duplicated(["condition_id", "rep", "method_id"]).any():
        issues.append("duplicate condition/repetition/method rows")
    if not raw["pi_supplied"].between(0.0, 1.0, inclusive="neither").all():
        issues.append("invalid pi_supplied in raw results")
    if not np.allclose(
        raw["pi_supplied"], raw["pi_true"] * raw["prior_ratio"],
        rtol=0.0, atol=1e-12,
    ):
        issues.append("pi_supplied does not equal pi_true times prior_ratio")

    cross_ratio = raw.groupby(["scenario_id", "rep"], as_index=False).agg(
        rows=("method_id", "size"),
        prior_ratios=("prior_ratio", "nunique"),
        conditions=("condition_id", "nunique"),
        seeds=("seed", "nunique"),
        fingerprints=("data_fingerprint", "nunique"),
        true_priors=("pi_true", "nunique"),
    )
    cross_ratio["pass"] = (
        cross_ratio["rows"].eq(5 * 27)
        & cross_ratio["prior_ratios"].eq(5)
        & cross_ratio["conditions"].eq(5)
        & cross_ratio["seeds"].eq(1)
        & cross_ratio["fingerprints"].eq(1)
        & cross_ratio["true_priors"].eq(1)
    )
    if len(cross_ratio) != 8 * N_REPETITIONS or not cross_ratio["pass"].all():
        issues.append("cross-ratio shared-data fingerprint gate failed")

    diagnostic_columns = [
        "data_fingerprint", "n", "n_positive_sample", "n_unlabeled_sample",
        "n_hidden_positive_in_U", "n_hidden_negative_in_U", "empirical_pi_in_U",
        "n_true_positive", "n_true_negative",
    ]
    diagnostic_audit = diagnostics.groupby(
        ["scenario_id", "rep", "split"], as_index=False
    ).agg(
        prior_ratios=("prior_ratio", "nunique"),
        **{f"unique_{column}": (column, "nunique") for column in diagnostic_columns},
    )
    unique_columns = [f"unique_{column}" for column in diagnostic_columns]
    diagnostic_audit["pass"] = (
        diagnostic_audit["prior_ratios"].eq(5)
        & diagnostic_audit[unique_columns].eq(1).all(axis=1)
    )
    if (
        len(diagnostic_audit) != 8 * N_REPETITIONS * 3
        or not diagnostic_audit["pass"].all()
    ):
        issues.append("cross-ratio DGP diagnostic equality failed")

    ranking_records = []
    expected_rank = raw.loc[raw["expected_ranking_invariant"].astype(bool)]
    for keys, frame in expected_rank.groupby(["scenario_id", "rep", "method_id"]):
        for metric in RANKING_METRICS:
            values = pd.to_numeric(frame[metric], errors="coerce")
            span = float(values.max() - values.min())
            ranking_records.append({
                "scenario_id": keys[0],
                "rep": keys[1],
                "method_id": keys[2],
                "metric": metric,
                "prior_ratios": int(frame["prior_ratio"].nunique()),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "span": span,
                "exact_pass": bool(span == 0.0),
                "tolerance_pass": bool(span <= INVARIANCE_TOLERANCE),
            })
    ranking_audit = pd.DataFrame(ranking_records)
    expected_ranking_rows = (
        8 * N_REPETITIONS * len(pilot.EXPECTED_RANKING_INVARIANT_IDS) * 2
    )
    if len(ranking_audit) != expected_ranking_rows or not ranking_audit["tolerance_pass"].all():
        issues.append("expected ranking-invariance negative control failed")

    oracle_records = []
    oracle = raw.loc[raw["method_id"].isin(pilot.EXPECTED_FULL_METRIC_INVARIANT_IDS)]
    for keys, frame in oracle.groupby(["scenario_id", "rep", "method_id"]):
        for metric in METRICS:
            values = pd.to_numeric(frame[metric], errors="coerce")
            span = float(values.max() - values.min())
            oracle_records.append({
                "scenario_id": keys[0],
                "rep": keys[1],
                "method_id": keys[2],
                "metric": metric,
                "span": span,
                "exact_pass": bool(span == 0.0),
                "tolerance_pass": bool(span <= INVARIANCE_TOLERANCE),
            })
    oracle_audit = pd.DataFrame(oracle_records)
    if (
        len(oracle_audit) != 8 * N_REPETITIONS * 2 * len(METRICS)
        or not oracle_audit["tolerance_pass"].all()
    ):
        issues.append("oracle full-metric negative control failed")

    routing_map = {
        "TM003": "known_population_prior",
        "TM004": "pi_u",
        "TM005": "pi_u",
        "TM026": "prior_given_to_AdaPU",
        "TM027": "pi_u",
    }
    routing_records = []
    for method_id, column in routing_map.items():
        for _, row in raw.loc[raw["method_id"].eq(method_id)].iterrows():
            actual = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
            routing_records.append({
                "condition_id": row["condition_id"],
                "scenario_id": row["scenario_id"],
                "rep": row["rep"],
                "method_id": method_id,
                "routing_column": column,
                "pi_supplied": row["pi_supplied"],
                "recorded_prior": actual,
                "pass": bool(np.isclose(actual, row["pi_supplied"], rtol=0.0, atol=1e-12)),
            })
    routing_audit = pd.DataFrame(routing_records)
    if len(routing_audit) != EXPECTED_TASKS * 5 or not routing_audit["pass"].all():
        issues.append("direct supplied-prior routing audit failed")

    pair_records = []
    for (condition_id, rep), frame in raw.groupby(["condition_id", "rep"]):
        by_id = frame.set_index("method_id")
        for naive_id, corrected_id in NAIVE_CORRECTED_PAIRS:
            differences = {
                metric: abs(float(by_id.loc[naive_id, metric]) - float(by_id.loc[corrected_id, metric]))
                for metric in ["AUROC", "AUPRC", "F1", "BalancedAccuracy", "MCC"]
            }
            pair_records.append({
                "condition_id": condition_id,
                "rep": rep,
                "naive_method_id": naive_id,
                "corrected_method_id": corrected_id,
                "best_params_equal": str(by_id.loc[naive_id, "best_params"]) == str(by_id.loc[corrected_id, "best_params"]),
                **{f"abs_diff_{metric}": value for metric, value in differences.items()},
                "pass": bool(
                    str(by_id.loc[naive_id, "best_params"]) == str(by_id.loc[corrected_id, "best_params"])
                    and max(differences.values()) <= 1e-12
                ),
            })
    pair_audit = pd.DataFrame(pair_records)
    if len(pair_audit) != EXPECTED_TASKS * 5 or not pair_audit["pass"].all():
        issues.append("naive/probability-correction paired invariant failed")

    tm027 = raw.loc[raw["method_id"].eq("TM027")].copy()
    tm027["pass"] = (
        tm027["pu_gb_objective"].eq("first_order_canonical_two_sample_nnpu")
        & tm027["pu_gb_optimizer"].eq("functional_gradient_equal_projection_weights")
        & pd.to_numeric(tm027["minimum_positive_pseudo_residual"], errors="coerce").gt(0.0)
    )
    if len(tm027) != EXPECTED_TASKS or not tm027["pass"].all():
        issues.append("TM027 repaired-objective gate failed")

    repaired_probability = raw.loc[raw["method_id"].isin(REPAIRED_RANKING_IDS)].copy()
    repaired_probability["pass"] = repaired_probability["ranking_score_type"].eq(
        "unclipped_log_case_control_density_ratio"
    )
    if len(repaired_probability) != EXPECTED_TASKS * len(REPAIRED_RANKING_IDS) or not repaired_probability["pass"].all():
        issues.append("repaired probability-ranking gate failed")

    metric_summary = _metric_summary(raw)
    paired_raw, paired_summary = _paired_changes(raw)
    extreme_summary = paired_summary.loc[
        paired_summary["prior_ratio"].isin([0.8, 1.2])
    ].copy()
    asymmetry = extreme_summary.pivot_table(
        index=["scenario_id", "method_id", "metric"],
        columns="prior_ratio",
        values="mean_performance_loss",
    ).reset_index().rename(columns={0.8: "underestimate_20pct_loss", 1.2: "overestimate_20pct_loss"})
    asymmetry["over_minus_under_loss"] = (
        asymmetry["overestimate_20pct_loss"] - asymmetry["underestimate_20pct_loss"]
    )

    atomic_csv(task, OUTPUT / "task_integrity_audit.csv")
    atomic_csv(cross_ratio, OUTPUT / "cross_ratio_shared_data_audit.csv")
    atomic_csv(diagnostic_audit, OUTPUT / "cross_ratio_diagnostic_audit.csv")
    atomic_csv(ranking_audit, OUTPUT / "ranking_invariance_negative_control.csv")
    atomic_csv(oracle_audit, OUTPUT / "oracle_full_metric_negative_control.csv")
    atomic_csv(routing_audit, OUTPUT / "supplied_prior_routing_audit.csv")
    atomic_csv(pair_audit, OUTPUT / "naive_probability_pair_audit.csv")
    atomic_csv(tm027, OUTPUT / "tm027_objective_audit.csv")
    atomic_csv(repaired_probability, OUTPUT / "repaired_probability_ranking_audit.csv")
    atomic_csv(metric_summary, OUTPUT / "method_metric_summary.csv")
    atomic_csv(paired_raw, OUTPUT / "paired_changes_raw.csv")
    atomic_csv(paired_summary, OUTPUT / "paired_changes_summary.csv")
    atomic_csv(asymmetry, OUTPUT / "under_over_asymmetry_summary.csv")
    metadata = {
        "analysis_phase": (
            "phase7_two_sample_prior_misspecification_confirmatory_100reps"
        ),
        "study_role": "class_prior_misspecification_confirmatory",
        "base_scenarios": 8,
        "prior_ratios": [0.8, 0.9, 1.0, 1.1, 1.2],
        "repetitions_per_condition": N_REPETITIONS,
        "expected_tasks": EXPECTED_TASKS,
        "observed_tasks": int(len(task)),
        "expected_method_rows": EXPECTED_ROWS,
        "observed_method_rows": int(len(raw)),
        "cross_ratio_scenario_repetitions_passing": int(cross_ratio["pass"].sum()),
        "cross_ratio_scenario_repetitions_expected": 8 * N_REPETITIONS,
        "ranking_invariance_tolerance": INVARIANCE_TOLERANCE,
        "ranking_negative_controls_passing": int(ranking_audit["tolerance_pass"].sum()),
        "ranking_negative_controls_expected": expected_ranking_rows,
        "oracle_negative_controls_passing": int(oracle_audit["tolerance_pass"].sum()),
        "oracle_negative_controls_expected": (
            8 * N_REPETITIONS * 2 * len(METRICS)
        ),
        "supplied_prior_routing_rows_passing": int(routing_audit["pass"].sum()),
        "supplied_prior_routing_rows_expected": EXPECTED_TASKS * 5,
        "naive_probability_pairs_passing": int(pair_audit["pass"].sum()),
        "naive_probability_pairs_expected": EXPECTED_TASKS * 5,
        "tm027_rows_passing": int(tm027["pass"].sum()),
        "tm027_rows_expected": EXPECTED_TASKS,
        "integrity_pass": not issues,
        "issues": issues,
        "confirmatory_interpretation": (
            "One hundred fresh repetitions support the pre-specified paired "
            "class-prior sensitivity analysis."
        ),
    }
    atomic_json(
        metadata,
        OUTPUT / "prior_misspecification_confirmatory_metadata.json",
    )
    if issues:
        raise RuntimeError("; ".join(issues))
    print(
        "PASS: 4000 tasks, exact shared data across ratios, and all method "
        "gates verified"
    )
    print(f"Analysis written to {OUTPUT}")


if __name__ == "__main__":
    main()
