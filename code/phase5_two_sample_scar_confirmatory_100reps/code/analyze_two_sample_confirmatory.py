"""Pre-specified integrity and method analysis for the 100-rep study."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
COMBINED = PROJECT_DIR / "results" / "two_sample_combined"
OUTPUT = PROJECT_DIR / "results" / "two_sample_confirmatory_analysis"
RAW = COMBINED / "two_sample_raw_all_scenarios.csv"
DIAGNOSTICS = COMBINED / "two_sample_diagnostics_all_scenarios.csv"
REPAIRED_PROBABILITY_IDS = ["TM003", *[f"TM{i:03d}" for i in range(11, 16)]]
NAIVE_CORRECTED_PAIRS = list(zip(
    [f"TM{i:03d}" for i in range(6, 11)],
    [f"TM{i:03d}" for i in range(11, 16)],
))
PAIR_METRICS = ["AUROC", "AUPRC", "F1", "BalancedAccuracy", "MCC"]
RANKING_EQUIVALENT_CORRECTED_IDS = {f"TM{i:03d}" for i in range(11, 16)}
PRIMARY_CONTRAST_TARGETS = ["TM026", "TM027"]
PRIMARY_CONTRAST_REFERENCES = ["TM003", "TM004", "TM005", "TM020"]


def atomic_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def main():
    raw = pd.read_csv(RAW)
    diagnostics = pd.read_csv(DIAGNOSTICS)
    expected_tasks = 40 * 100
    expected_rows = expected_tasks * 27
    issues = []
    if len(raw) != expected_rows:
        issues.append(f"expected {expected_rows} method rows, found {len(raw)}")
    task_counts = raw.groupby(["scenario_id", "rep"]).agg(
        rows=("method", "size"),
        methods=("method_id", "nunique"),
        fingerprints=("data_fingerprint", "nunique"),
        auroc_available=("AUROC", "count"),
    ).reset_index()
    task_counts["complete"] = (
        task_counts["rows"].eq(27)
        & task_counts["methods"].eq(27)
        & task_counts["fingerprints"].eq(1)
        & task_counts["auroc_available"].eq(27)
    )
    if len(task_counts) != expected_tasks or not task_counts["complete"].all():
        issues.append(f"complete tasks: {int(task_counts['complete'].sum())}/{len(task_counts)}")
    expected_ids = {f"TM{i:03d}" for i in range(1, 28)}
    if set(raw["method_id"]) != expected_ids:
        issues.append("method catalog is not TM001--TM027")
    if not raw["sampling_scheme"].eq("two_sample_case_control_scar").all():
        issues.append("unexpected sampling scheme")
    if not raw["dgp_implementation"].eq("deterministic_toeplitz_ar1_v1").all():
        issues.append("unexpected DGP implementation")
    if raw["confirmatory_seed_reuse"].astype(str).str.lower().ne("false").any():
        issues.append("confirmatory seeds are incorrectly tagged as reused")

    required_repair_columns = {
        "best_params", "ranking_score_type", "probability_score_type",
        "pu_gb_objective", "pu_gb_optimizer", "minimum_positive_pseudo_residual",
    }
    missing_repair_columns = sorted(required_repair_columns.difference(raw.columns))
    if missing_repair_columns:
        issues.append("missing repair-audit columns: " + ", ".join(missing_repair_columns))
        for column in missing_repair_columns:
            raw[column] = np.nan

    probability_audit = raw.loc[
        raw["method_id"].isin(REPAIRED_PROBABILITY_IDS),
        [
            "scenario_id", "rep", "method_id", "method", "ranking_score_type",
            "probability_score_type",
        ],
    ].copy()
    probability_audit["unclipped_ranking_pass"] = probability_audit[
        "ranking_score_type"
    ].eq("unclipped_log_case_control_density_ratio")
    expected_probability_rows = expected_tasks * len(REPAIRED_PROBABILITY_IDS)
    if len(probability_audit) != expected_probability_rows:
        issues.append(
            f"expected {expected_probability_rows} repaired probability rows, "
            f"found {len(probability_audit)}"
        )
    if not probability_audit["unclipped_ranking_pass"].all():
        issues.append("TM003/TM011--TM015 unclipped-ranking invariant failed")

    pair_records = []
    for (scenario_id, rep), task in raw.groupby(["scenario_id", "rep"], sort=True):
        by_id = task.set_index("method_id", drop=False)
        for naive_id, corrected_id in NAIVE_CORRECTED_PAIRS:
            record = {
                "scenario_id": scenario_id,
                "rep": rep,
                "naive_method_id": naive_id,
                "corrected_method_id": corrected_id,
                "pair_present": naive_id in by_id.index and corrected_id in by_id.index,
            }
            if record["pair_present"]:
                naive = by_id.loc[naive_id]
                corrected = by_id.loc[corrected_id]
                record["best_params_equal"] = str(naive["best_params"]) == str(corrected["best_params"])
                for metric in PAIR_METRICS:
                    record[f"abs_diff_{metric}"] = abs(float(naive[metric]) - float(corrected[metric]))
                record["pair_pass"] = bool(
                    record["best_params_equal"]
                    and all(record[f"abs_diff_{metric}"] <= 1e-12 for metric in PAIR_METRICS)
                )
            else:
                record["best_params_equal"] = False
                for metric in PAIR_METRICS:
                    record[f"abs_diff_{metric}"] = np.nan
                record["pair_pass"] = False
            pair_records.append(record)
    pair_audit = pd.DataFrame(pair_records)
    if (
        len(pair_audit) != expected_tasks * len(NAIVE_CORRECTED_PAIRS)
        or not pair_audit["pair_pass"].all()
    ):
        issues.append(
            "Naive/Probability-correction paired ranking, tuning, or classification invariant failed"
        )

    tm027_audit = raw.loc[
        raw["method_id"].eq("TM027"),
        [
            "scenario_id", "rep", "method_id", "pu_gb_objective", "pu_gb_optimizer",
            "minimum_positive_pseudo_residual", "nnpu_correction_count",
            "minimum_negative_component_train", "final_negative_component",
        ],
    ].copy()
    tm027_audit["objective_pass"] = tm027_audit["pu_gb_objective"].eq(
        "first_order_canonical_two_sample_nnpu"
    )
    tm027_audit["optimizer_pass"] = tm027_audit["pu_gb_optimizer"].eq(
        "functional_gradient_equal_projection_weights"
    )
    tm027_audit["positive_influence_pass"] = pd.to_numeric(
        tm027_audit["minimum_positive_pseudo_residual"], errors="coerce"
    ).gt(0.0)
    tm027_audit["tm027_pass"] = tm027_audit[
        ["objective_pass", "optimizer_pass", "positive_influence_pass"]
    ].all(axis=1)
    if len(tm027_audit) != expected_tasks or not tm027_audit["tm027_pass"].all():
        issues.append("TM027 explicit two-sample first-order objective invariant failed")

    metric_summary = raw.groupby(
        ["scenario_id", "scenario_label", "method_id", "method", "method_block", "comparison_role"],
        as_index=False,
    ).agg(
        n=("rep", "nunique"),
        AUROC_mean=("AUROC", "mean"),
        AUROC_sd=("AUROC", "std"),
        AUPRC_mean=("AUPRC", "mean"),
        AUPRC_sd=("AUPRC", "std"),
        F1_mean=("F1", "mean"),
        F1_sd=("F1", "std"),
        BalancedAccuracy_mean=("BalancedAccuracy", "mean"),
        BalancedAccuracy_sd=("BalancedAccuracy", "std"),
        MCC_mean=("MCC", "mean"),
        MCC_sd=("MCC", "std"),
    )
    for metric in PAIR_METRICS:
        metric_summary[f"{metric}_se"] = (
            metric_summary[f"{metric}_sd"] / np.sqrt(metric_summary["n"])
        )
        metric_summary[f"{metric}_ci95_low"] = (
            metric_summary[f"{metric}_mean"]
            - 1.96 * metric_summary[f"{metric}_se"]
        )
        metric_summary[f"{metric}_ci95_high"] = (
            metric_summary[f"{metric}_mean"]
            + 1.96 * metric_summary[f"{metric}_se"]
        )

    # Corrected twins are retained for calibration metrics but excluded from
    # winner/rank counting because their rankings exactly duplicate TM006--10.
    pu = raw.loc[
        raw["comparison_role"].eq("pu_method")
        & ~raw["method_id"].isin(RANKING_EQUIVALENT_CORRECTED_IDS)
    ].copy()
    for metric in ["AUROC", "AUPRC"]:
        pu[f"{metric}_rank"] = pu.groupby(["scenario_id", "rep"])[metric].rank(
            ascending=False, method="average"
        )
    ranks = pu.groupby(["method_id", "method", "method_block"], as_index=False).agg(
        mean_AUROC_rank=("AUROC_rank", "mean"),
        mean_AUPRC_rank=("AUPRC_rank", "mean"),
        mean_AUROC=("AUROC", "mean"),
        mean_AUPRC=("AUPRC", "mean"),
    ).sort_values("mean_AUROC_rank")
    winners = []
    winner_candidates = metric_summary.loc[
        metric_summary["comparison_role"].eq("pu_method")
        & ~metric_summary["method_id"].isin(RANKING_EQUIVALENT_CORRECTED_IDS)
    ]
    for scenario_id, frame in winner_candidates.groupby("scenario_id"):
        for metric in ["AUROC", "AUPRC"]:
            best = frame.sort_values(f"{metric}_mean", ascending=False).iloc[0]
            winners.append({
                "scenario_id": scenario_id,
                "metric": metric,
                "winner_method_id": best["method_id"],
                "winner_method": best["method"],
                "winner_mean": best[f"{metric}_mean"],
            })
    winners = pd.DataFrame(winners)

    primary_contrast_rows = []
    indexed = raw.set_index(["scenario_id", "rep", "method_id"], drop=False)
    for scenario_id in sorted(raw["scenario_id"].unique()):
        for rep in range(1, 101):
            for target in PRIMARY_CONTRAST_TARGETS:
                for reference in PRIMARY_CONTRAST_REFERENCES:
                    for metric in ["AUROC", "AUPRC"]:
                        target_value = float(indexed.loc[(scenario_id, rep, target), metric])
                        reference_value = float(indexed.loc[(scenario_id, rep, reference), metric])
                        primary_contrast_rows.append({
                            "scenario_id": scenario_id,
                            "rep": rep,
                            "metric": metric,
                            "target_method_id": target,
                            "reference_method_id": reference,
                            "paired_difference": target_value - reference_value,
                        })
    primary_contrasts = pd.DataFrame(primary_contrast_rows)
    primary_contrast_summary = primary_contrasts.groupby(
        ["scenario_id", "metric", "target_method_id", "reference_method_id"],
        as_index=False,
    ).agg(
        n=("paired_difference", "size"),
        mean_difference=("paired_difference", "mean"),
        sd_difference=("paired_difference", "std"),
        median_difference=("paired_difference", "median"),
        win_rate=("paired_difference", lambda values: float((values > 0).mean())),
        tie_rate=("paired_difference", lambda values: float((values == 0).mean())),
    )
    primary_contrast_summary["se_difference"] = (
        primary_contrast_summary["sd_difference"]
        / np.sqrt(primary_contrast_summary["n"])
    )
    primary_contrast_summary["ci95_low"] = (
        primary_contrast_summary["mean_difference"]
        - 1.96 * primary_contrast_summary["se_difference"]
    )
    primary_contrast_summary["ci95_high"] = (
        primary_contrast_summary["mean_difference"]
        + 1.96 * primary_contrast_summary["se_difference"]
    )

    diagnostic_summary = diagnostics.loc[diagnostics["split"].isin(["train", "val"])].groupby(
        ["scenario_id", "split"], as_index=False
    ).agg(
        repetitions=("rep", "nunique"),
        n_positive_sample_min=("n_positive_sample", "min"),
        n_positive_sample_max=("n_positive_sample", "max"),
        n_unlabeled_sample_min=("n_unlabeled_sample", "min"),
        n_unlabeled_sample_max=("n_unlabeled_sample", "max"),
        empirical_pi_U_mean=("empirical_pi_in_U", "mean"),
        empirical_pi_U_sd=("empirical_pi_in_U", "std"),
    )

    atomic_csv(task_counts, OUTPUT / "task_integrity_audit.csv")
    atomic_csv(metric_summary, OUTPUT / "scenario_method_metric_summary.csv")
    atomic_csv(ranks, OUTPUT / "overall_pu_method_ranks.csv")
    atomic_csv(winners, OUTPUT / "scenario_pu_winners.csv")
    atomic_csv(diagnostic_summary, OUTPUT / "sampling_diagnostic_summary.csv")
    atomic_csv(probability_audit, OUTPUT / "repaired_probability_method_audit.csv")
    atomic_csv(pair_audit, OUTPUT / "naive_probability_pair_invariant_audit.csv")
    atomic_csv(tm027_audit, OUTPUT / "tm027_objective_audit.csv")
    atomic_csv(primary_contrasts, OUTPUT / "prespecified_paired_contrasts_raw.csv")
    atomic_csv(
        primary_contrast_summary,
        OUTPUT / "prespecified_paired_contrasts_summary.csv",
    )
    metadata = {
        "analysis_phase": "phase5_two_sample_scar_confirmatory_100reps",
        "study_role": "confirmatory_monte_carlo",
        "frozen_method_source_phase": "phase4b_two_sample_method_repair_pilot_10reps",
        "dgp_reproducibility_source_phase": "phase4c_deterministic_correlated_dgp_cross_node",
        "confirmatory_seed_reuse": False,
        "dgp_implementation": "deterministic_toeplitz_ar1_v1",
        "repair_scope": REPAIRED_PROBABILITY_IDS + ["TM027"],
        "ranking_equivalent_corrected_ids_excluded_from_winner_counts": sorted(
            RANKING_EQUIVALENT_CORRECTED_IDS
        ),
        "prespecified_contrast_targets": PRIMARY_CONTRAST_TARGETS,
        "prespecified_contrast_references": PRIMARY_CONTRAST_REFERENCES,
        "expected_tasks": expected_tasks,
        "observed_tasks": int(len(task_counts)),
        "expected_method_rows": expected_rows,
        "observed_method_rows": int(len(raw)),
        "complete_tasks": int(task_counts["complete"].sum()),
        "repaired_probability_rows_passing": int(probability_audit["unclipped_ranking_pass"].sum()),
        "repaired_probability_rows_expected": expected_probability_rows,
        "naive_probability_pairs_passing": int(pair_audit["pair_pass"].sum()),
        "naive_probability_pairs_expected": expected_tasks * len(NAIVE_CORRECTED_PAIRS),
        "tm027_rows_passing": int(tm027_audit["tm027_pass"].sum()),
        "tm027_rows_expected": expected_tasks,
        "integrity_pass": not issues,
        "issues": issues,
        "interpretation_guardrail": "Naive and Probability-correction twins are one ranking-equivalence group per engine; do not count them as independent ranking evidence.",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "two_sample_confirmatory_analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    if issues:
        raise RuntimeError("; ".join(issues))
    print(
        "PASS: 4000/4000 confirmatory tasks and all repaired-method "
        "invariants verified"
    )
    print(f"Analysis written to {OUTPUT}")


if __name__ == "__main__":
    main()
