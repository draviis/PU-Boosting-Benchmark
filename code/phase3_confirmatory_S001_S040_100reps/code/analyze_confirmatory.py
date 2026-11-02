"""Pre-specified integrity audit and analysis for the confirmatory study."""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PROJECT_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_DIR / "config/confirmatory_scenarios_S001_S040_100reps.csv"
COMBINED_DIR = PROJECT_DIR / "results/confirmatory_combined"
RAW_PATH = COMBINED_DIR / "confirmatory_raw_all_scenarios.csv"
DIAGNOSTICS_PATH = COMBINED_DIR / "confirmatory_diagnostics_all_scenarios.csv"
OUTPUT_DIR = PROJECT_DIR / "results/confirmatory_analysis"
METRICS = ["AUROC", "AUPRC", "F1", "BalancedAccuracy", "MCC", "Brier", "LogLoss"]
PRIMARY_METRICS = ["AUROC", "AUPRC"]
REPAIRED_METHODS = ["M026", "M027"]
COMPARATORS = ["M003", "M004", "M005"]
EXPECTED_METHOD_IDS = [f"M{index:03d}" for index in range(1, 28)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def finite_count(values: pd.Series) -> int:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return int(np.isfinite(array).sum())


def integrity_audit(
    raw: pd.DataFrame, diagnostics: pd.DataFrame, manifest: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    manifest_by_id = manifest.set_index("scenario_id")
    rows = []
    for (scenario_id, rep), group in raw.groupby(["scenario_id", "rep"], sort=True):
        definition = manifest_by_id.loc[scenario_id]
        expected_seed = int(definition["scenario_base_seed"]) + int(rep) - 1
        method_ids = sorted(group["method_id"].dropna().astype(str).unique())
        roles = group["comparison_role"].value_counts()
        issues = []
        if len(group) != 27:
            issues.append(f"row_count={len(group)}")
        if method_ids != EXPECTED_METHOD_IDS:
            issues.append("method_catalog_mismatch")
        if int(roles.get("supervised_reference", 0)) != 2:
            issues.append("oracle_count_mismatch")
        if int(roles.get("pu_method", 0)) != 25:
            issues.append("pu_count_mismatch")
        if group["data_fingerprint"].nunique(dropna=False) != 1:
            issues.append("shared_data_fingerprint_mismatch")
        if set(pd.to_numeric(group["seed"], errors="coerce")) != {expected_seed}:
            issues.append("seed_mismatch")
        for metric in PRIMARY_METRICS:
            if finite_count(group[metric]) != 27:
                issues.append(f"nonfinite_{metric}")
        rows.append(
            {
                "scenario_id": scenario_id,
                "rep": int(rep),
                "expected_seed": expected_seed,
                "row_count": len(group),
                "method_count": len(method_ids),
                "oracle_count": int(roles.get("supervised_reference", 0)),
                "pu_method_count": int(roles.get("pu_method", 0)),
                "data_fingerprint_count": group["data_fingerprint"].nunique(dropna=False),
                "complete": not issues,
                "issues": ";".join(issues),
            }
        )
    by_run = pd.DataFrame(rows)

    scenario_rows = []
    for definition in manifest.itertuples(index=False):
        group = by_run.loc[by_run["scenario_id"].eq(definition.scenario_id)]
        raw_group = raw.loc[raw["scenario_id"].eq(definition.scenario_id)]
        diag_group = diagnostics.loc[
            diagnostics["scenario_id"].eq(definition.scenario_id)
        ]
        m026 = raw_group.loc[raw_group["method_id"].eq("M026")]
        m027 = raw_group.loc[raw_group["method_id"].eq("M027")]
        prior = pd.to_numeric(m026["prior_given_to_AdaPU"], errors="coerce")
        prior_ok = len(prior) == 100 and np.allclose(
            prior, float(definition.pi_u_theoretical), rtol=1e-10, atol=1e-10
        )
        rng_ok = (
            len(m026) == 100
            and m026["adapu_rng_mode"].astype(str).eq(
                "seeded_numpy_no_wall_clock_reseed"
            ).all()
        )
        objective_ok = (
            len(m027) == 100
            and m027["pu_gb_objective"].astype(str).eq(
                "newton_one_sample_scar_with_nnpu_correction"
            ).all()
        )
        split_counts = diag_group.groupby("rep")["split"].nunique()
        scenario_rows.append(
            {
                "scenario_index": int(definition.scenario_index),
                "scenario_id": definition.scenario_id,
                "complete_runs": int(group["complete"].sum()),
                "expected_runs": 100,
                "raw_rows": len(raw_group),
                "expected_raw_rows": 2700,
                "diagnostic_rows": len(diag_group),
                "expected_diagnostic_rows": 300,
                "three_diagnostic_splits_each_run": (
                    len(split_counts) == 100 and split_counts.eq(3).all()
                ),
                "M026_theoretical_prior_pass": bool(prior_ok),
                "M026_rng_mode_pass": bool(rng_ok),
                "M027_objective_pass": bool(objective_ok),
            }
        )
    by_scenario = pd.DataFrame(scenario_rows)
    boolean_columns = [
        "three_diagnostic_splits_each_run",
        "M026_theoretical_prior_pass",
        "M026_rng_mode_pass",
        "M027_objective_pass",
    ]
    problems = []
    if len(raw) != 108000:
        problems.append(f"expected 108000 raw rows, found {len(raw)}")
    if len(by_run) != 4000 or not by_run["complete"].all():
        problems.append(
            f"expected 4000 complete scenario-repetition runs, found {int(by_run['complete'].sum())}/{len(by_run)}"
        )
    if len(by_scenario) != 40:
        problems.append(f"expected 40 scenario audits, found {len(by_scenario)}")
    if not by_scenario[boolean_columns].all(axis=None):
        problems.append("one or more repaired-method/diagnostic integrity gates failed")
    if not by_scenario["raw_rows"].eq(2700).all():
        problems.append("one or more scenarios do not have 2700 raw rows")
    if not by_scenario["diagnostic_rows"].eq(300).all():
        problems.append("one or more scenarios do not have 300 diagnostic rows")
    return by_run, by_scenario, problems


def descriptive_summary(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = [
        "scenario_index",
        "scenario_id",
        "scenario_label",
        "scenario_family",
        "method_id",
        "method",
        "method_block",
        "comparison_role",
    ]
    for key, group in raw.groupby(keys, dropna=False, sort=True):
        base = dict(zip(keys, key))
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy()
            n = len(values)
            if n == 0:
                # Several score-only boosting methods do not expose calibrated
                # probabilities, so Brier score and log loss are intentionally
                # unavailable. Keep an explicit n=0 row instead of failing or
                # silently dropping the method-metric combination.
                rows.append(
                    {
                        **base,
                        "metric": metric,
                        "n": 0,
                        "available": False,
                        "mean": math.nan,
                        "sd": math.nan,
                        "mcse": math.nan,
                        "ci95_low": math.nan,
                        "ci95_high": math.nan,
                        "median": math.nan,
                        "q025": math.nan,
                        "q975": math.nan,
                    }
                )
                continue
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1)) if n > 1 else math.nan
            mcse = sd / math.sqrt(n) if n > 1 else math.nan
            critical = float(stats.t.ppf(0.975, n - 1)) if n > 1 else math.nan
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "n": n,
                    "available": True,
                    "mean": mean,
                    "sd": sd,
                    "mcse": mcse,
                    "ci95_low": mean - critical * mcse,
                    "ci95_high": mean + critical * mcse,
                    "median": float(np.median(values)),
                    "q025": float(np.quantile(values, 0.025)),
                    "q975": float(np.quantile(values, 0.975)),
                }
            )
    return pd.DataFrame(rows)


def pu_ranks_and_winners(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pu = raw.loc[raw["comparison_role"].eq("pu_method")].copy()
    long_rows = []
    for metric in PRIMARY_METRICS:
        ranked = pu[["scenario_index", "scenario_id", "rep", "method_id", "method", metric]].copy()
        ranked["rank"] = ranked.groupby(["scenario_id", "rep"])[metric].rank(
            method="average", ascending=False
        )
        ranked["metric"] = metric
        ranked["value"] = ranked[metric]
        long_rows.append(
            ranked.drop(columns=[metric])[
                ["scenario_index", "scenario_id", "rep", "method_id", "method", "metric", "value", "rank"]
            ]
        )
    ranks = pd.concat(long_rows, ignore_index=True)
    group_keys = ["scenario_id", "rep", "metric"]
    ranks["best_value_in_run"] = ranks.groupby(group_keys)["value"].transform("max")
    ranks["winner_including_ties"] = np.isclose(
        ranks["value"], ranks["best_value_in_run"], rtol=0.0, atol=1e-15
    )
    ranks["number_of_tied_winners"] = ranks.groupby(group_keys)[
        "winner_including_ties"
    ].transform("sum")
    ranks["fractional_win_credit"] = np.where(
        ranks["winner_including_ties"],
        1.0 / ranks["number_of_tied_winners"],
        0.0,
    )
    winners = (
        ranks.groupby(
            ["scenario_index", "scenario_id", "metric", "method_id", "method"],
            as_index=False,
        )
        .agg(
            mean_rank=("rank", "mean"),
            median_rank=("rank", "median"),
            winner_repetitions_including_ties=("winner_including_ties", "sum"),
            fractional_wins=("fractional_win_credit", "sum"),
            repetitions=("rep", "nunique"),
        )
    )
    winners["winner_rate_including_ties"] = (
        winners["winner_repetitions_including_ties"] / winners["repetitions"]
    )
    winners["fractional_win_rate"] = winners["fractional_wins"] / winners["repetitions"]
    return ranks, winners


def holm_adjust(pvalues: pd.Series) -> np.ndarray:
    values = pd.to_numeric(pvalues, errors="coerce").to_numpy(dtype=float)
    adjusted = np.full(values.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        return adjusted
    order = valid[np.argsort(values[valid])]
    running = 0.0
    m = len(order)
    for position, index in enumerate(order):
        candidate = min(1.0, (m - position) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def paired_repaired_contrasts(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    lookup = raw.set_index(["scenario_id", "rep", "method_id"])
    detail_rows = []
    summary_rows = []
    method_names = raw.drop_duplicates("method_id").set_index("method_id")["method"]
    for scenario_id in sorted(raw["scenario_id"].unique()):
        reps = sorted(raw.loc[raw["scenario_id"].eq(scenario_id), "rep"].unique())
        for repaired in REPAIRED_METHODS:
            for comparator in COMPARATORS:
                for metric in PRIMARY_METRICS:
                    differences = []
                    for rep in reps:
                        repaired_value = float(lookup.loc[(scenario_id, rep, repaired), metric])
                        comparator_value = float(lookup.loc[(scenario_id, rep, comparator), metric])
                        difference = repaired_value - comparator_value
                        differences.append(difference)
                        detail_rows.append(
                            {
                                "scenario_id": scenario_id,
                                "rep": int(rep),
                                "repaired_method_id": repaired,
                                "repaired_method": method_names[repaired],
                                "comparator_method_id": comparator,
                                "comparator_method": method_names[comparator],
                                "metric": metric,
                                "difference_repaired_minus_comparator": difference,
                            }
                        )
                    values = np.asarray(differences, dtype=float)
                    n = len(values)
                    mean = float(values.mean())
                    sd = float(values.std(ddof=1))
                    mcse = sd / math.sqrt(n)
                    critical = float(stats.t.ppf(0.975, n - 1))
                    test = stats.ttest_1samp(values, popmean=0.0)
                    summary_rows.append(
                        {
                            "scenario_id": scenario_id,
                            "repaired_method_id": repaired,
                            "repaired_method": method_names[repaired],
                            "comparator_method_id": comparator,
                            "comparator_method": method_names[comparator],
                            "metric": metric,
                            "n_pairs": n,
                            "mean_difference": mean,
                            "sd_difference": sd,
                            "mcse_difference": mcse,
                            "ci95_low": mean - critical * mcse,
                            "ci95_high": mean + critical * mcse,
                            "paired_t_pvalue": float(test.pvalue),
                            "positive_favors_repaired": True,
                        }
                    )
    summary = pd.DataFrame(summary_rows)
    summary["holm_pvalue_all_480_primary_contrasts"] = holm_adjust(
        summary["paired_t_pvalue"]
    )
    return pd.DataFrame(detail_rows), summary


def scenario_contrasts(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    reference = raw.loc[raw["scenario_id"].eq("S001")]
    for scenario_id in sorted(set(raw["scenario_id"]) - {"S001"}):
        current = raw.loc[raw["scenario_id"].eq(scenario_id)]
        for method_id in EXPECTED_METHOD_IDS:
            method_name = current.loc[current["method_id"].eq(method_id), "method"].iloc[0]
            for metric in PRIMARY_METRICS:
                left = pd.to_numeric(
                    current.loc[current["method_id"].eq(method_id), metric], errors="coerce"
                ).dropna().to_numpy()
                right = pd.to_numeric(
                    reference.loc[reference["method_id"].eq(method_id), metric], errors="coerce"
                ).dropna().to_numpy()
                difference = float(left.mean() - right.mean())
                se = math.sqrt(left.var(ddof=1) / len(left) + right.var(ddof=1) / len(right))
                denominator = (
                    (left.var(ddof=1) / len(left)) ** 2 / (len(left) - 1)
                    + (right.var(ddof=1) / len(right)) ** 2 / (len(right) - 1)
                )
                df = (se**4) / denominator if denominator > 0 else math.inf
                critical = float(stats.t.ppf(0.975, df)) if math.isfinite(df) else 1.9599639845
                test = stats.ttest_ind(left, right, equal_var=False)
                rows.append(
                    {
                        "scenario_id": scenario_id,
                        "reference_scenario_id": "S001",
                        "method_id": method_id,
                        "method": method_name,
                        "metric": metric,
                        "n_scenario": len(left),
                        "n_reference": len(right),
                        "mean_difference_scenario_minus_S001": difference,
                        "ci95_low": difference - critical * se,
                        "ci95_high": difference + critical * se,
                        "welch_t_pvalue": float(test.pvalue),
                    }
                )
    return pd.DataFrame(rows)


def oracle_capacity_gaps(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    structural_ids = ["S038", "S039", "S040"]
    lookup = raw.set_index(["scenario_id", "rep", "method_id"])
    details = []
    summaries = []
    for scenario_id in structural_ids:
        values = []
        for rep in range(1, 101):
            logistic = float(lookup.loc[(scenario_id, rep, "M001"), "AUROC"])
            catboost = float(lookup.loc[(scenario_id, rep, "M002"), "AUROC"])
            gap = catboost - logistic
            values.append(gap)
            details.append(
                {
                    "scenario_id": scenario_id,
                    "rep": rep,
                    "oracle_catboost_minus_oracle_logistic_AUROC": gap,
                }
            )
        array = np.asarray(values)
        mcse = float(array.std(ddof=1) / math.sqrt(len(array)))
        critical = float(stats.t.ppf(0.975, len(array) - 1))
        summaries.append(
            {
                "scenario_id": scenario_id,
                "n_pairs": len(array),
                "mean_gap": float(array.mean()),
                "sd_gap": float(array.std(ddof=1)),
                "mcse_gap": mcse,
                "ci95_low": float(array.mean()) - critical * mcse,
                "ci95_high": float(array.mean()) + critical * mcse,
            }
        )
    return pd.DataFrame(details), pd.DataFrame(summaries)


def risk_correction_summary(raw: pd.DataFrame) -> pd.DataFrame:
    selected = raw.loc[raw["method_id"].isin(["M004", "M005", "M027"])].copy()
    numeric_columns = [
        "nnpu_correction_count",
        "tuning_correction_count",
        "minimum_negative_component_train",
        "minimum_negative_component_tuning",
        "final_negative_component",
        "final_raw_negative_component",
        "n_estimators_fitted",
    ]
    rows = []
    for (scenario_id, method_id, method), group in selected.groupby(
        ["scenario_id", "method_id", "method"], sort=True
    ):
        row = {"scenario_id": scenario_id, "method_id": method_id, "method": method, "n": len(group)}
        for column in numeric_columns:
            if column in group.columns:
                values = pd.to_numeric(group[column], errors="coerce")
                row[f"{column}_available"] = int(values.notna().sum())
                row[f"{column}_mean"] = float(values.mean()) if values.notna().any() else math.nan
                row[f"{column}_max"] = float(values.max()) if values.notna().any() else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    manifest = pd.read_csv(MANIFEST_PATH)
    raw = pd.read_csv(RAW_PATH, low_memory=False)
    diagnostics = pd.read_csv(DIAGNOSTICS_PATH, low_memory=False)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    by_run, by_scenario, problems = integrity_audit(raw, diagnostics, manifest)
    atomic_csv(by_run, OUTPUT_DIR / "integrity_by_run.csv")
    atomic_csv(by_scenario, OUTPUT_DIR / "integrity_by_scenario.csv")
    if problems:
        payload = {
            "status": "FAIL",
            "problems": problems,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "raw_sha256": sha256_file(RAW_PATH),
            "diagnostics_sha256": sha256_file(DIAGNOSTICS_PATH),
        }
        atomic_json(payload, OUTPUT_DIR / "confirmatory_analysis_metadata.json")
        raise SystemExit("Confirmatory integrity audit failed: " + "; ".join(problems))

    method_catalog = (
        raw[["method_id", "method", "method_block", "comparison_role", "uses_true_y_for_development"]]
        .drop_duplicates()
        .sort_values("method_id")
    )
    summary = descriptive_summary(raw)
    ranks, winners = pu_ranks_and_winners(raw)
    pair_details, pair_summary = paired_repaired_contrasts(raw)
    reference_contrasts = scenario_contrasts(raw)
    oracle_details, oracle_summary = oracle_capacity_gaps(raw)
    risk = risk_correction_summary(raw)

    outputs = {
        "method_catalog.csv": method_catalog,
        "scenario_method_metric_summary.csv": summary,
        "pu_ranks_by_run.csv": ranks,
        "pu_rank_and_winner_summary.csv": winners,
        "repaired_boosting_paired_differences.csv": pair_details,
        "repaired_boosting_primary_contrasts.csv": pair_summary,
        "scenario_vs_S001_contrasts.csv": reference_contrasts,
        "structural_oracle_capacity_gaps_by_run.csv": oracle_details,
        "structural_oracle_capacity_gap_summary.csv": oracle_summary,
        "risk_correction_summary.csv": risk,
    }
    for name, frame in outputs.items():
        atomic_csv(frame, OUTPUT_DIR / name)

    payload = {
        "status": "PASS",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scenario_count": 40,
        "repetitions_per_scenario": 100,
        "complete_runs": 4000,
        "raw_method_rows": 108000,
        "methods": 27,
        "pu_methods": 25,
        "oracle_references": 2,
        "primary_metrics": PRIMARY_METRICS,
        "primary_contrast_count": len(pair_summary),
        "primary_multiplicity_rule": "Holm across all 480 repaired-vs-Block1 primary contrasts",
        "frozen_unchanged_method_numerical_audit_tolerance": 5e-5,
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "raw_sha256": sha256_file(RAW_PATH),
        "diagnostics_sha256": sha256_file(DIAGNOSTICS_PATH),
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "output_rows": {name: len(frame) for name, frame in outputs.items()},
    }
    atomic_json(payload, OUTPUT_DIR / "confirmatory_analysis_metadata.json")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
