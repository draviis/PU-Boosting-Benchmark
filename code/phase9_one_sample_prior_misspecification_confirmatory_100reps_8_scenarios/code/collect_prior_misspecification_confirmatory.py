"""Memory-safe collector that avoids importing the numerical method stack.

The original collector launched 40 fresh Python processes through the full
benchmark runner. On the cluster, a native-library import crashed with SIGSEGV
even though the affected condition's 100 result files were complete. This
version performs the same file validation, ranking, and summaries directly
with pandas/numpy, one condition at a time.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT_DIR / "config" / "prior_misspecification_confirmatory_manifest.csv"
RESULTS = PROJECT_DIR / "results"
COMBINED = RESULTS / "prior_misspecification_combined"
EXPECTED_REPETITIONS = 100
EXPECTED_METHODS = 27
TEST_METRICS = [
    "AUROC", "AUPRC", "F1", "BalancedAccuracy", "MCC", "Brier", "LogLoss"
]


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False, lineterminator="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def manifest_rows() -> list[dict[str, str]]:
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 40:
        raise RuntimeError(f"expected 40 manifest conditions, found {len(rows)}")
    return sorted(rows, key=lambda row: int(row["condition_index"]))


def condition_base(row: dict[str, str]) -> Path:
    return (
        RESULTS
        / "scenarios"
        / row["scenario_slug"]
        / "prior_conditions"
        / row["condition_slug"]
    )


def repetition_paths(row: dict[str, str], rep: int) -> dict[str, Path]:
    root = condition_base(row)
    stem = f"{row['condition_id']}_rep_{rep:03d}"
    return {
        "results": root / "repetitions" / f"{stem}_results.csv",
        "diagnostics": root / "repetitions" / f"{stem}_diagnostics.csv",
        "metadata": root / "run_metadata" / f"{stem}_metadata.json",
    }


def rank_and_summarize(raw_results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_results = raw_results.copy()
    raw_results["comparison_role"] = np.where(
        raw_results["method"].str.startswith("Block0 |"),
        "supervised_reference",
        "pu_method",
    )
    raw_results["method_block"] = raw_results["method"].str.extract(r"^(Block\d+)")[0]
    for metric in TEST_METRICS:
        raw_results[metric] = pd.to_numeric(raw_results[metric], errors="coerce")

    raw_results["AUROC_rank_all_27"] = raw_results.groupby("rep")["AUROC"].rank(
        ascending=False, method="average"
    )
    raw_results["AUPRC_rank_all_27"] = raw_results.groupby("rep")["AUPRC"].rank(
        ascending=False, method="average"
    )
    raw_results["AUROC_rank_PU_25"] = np.nan
    raw_results["AUPRC_rank_PU_25"] = np.nan
    pu_mask = raw_results["comparison_role"].eq("pu_method")
    raw_results.loc[pu_mask, "AUROC_rank_PU_25"] = (
        raw_results.loc[pu_mask].groupby("rep")["AUROC"].rank(
            ascending=False, method="average"
        )
    )
    raw_results.loc[pu_mask, "AUPRC_rank_PU_25"] = (
        raw_results.loc[pu_mask].groupby("rep")["AUPRC"].rank(
            ascending=False, method="average"
        )
    )
    raw_results["AUROC_rank"] = raw_results["AUROC_rank_PU_25"]
    raw_results["AUPRC_rank"] = raw_results["AUPRC_rank_PU_25"]

    summary_rows: list[dict] = []
    for method, frame in raw_results.groupby("method"):
        record = {
            "method": method,
            "method_block": frame["method_block"].iloc[0],
            "comparison_role": frame["comparison_role"].iloc[0],
            "n_repetitions_available": frame["rep"].nunique(),
        }
        for metric in TEST_METRICS:
            values = frame[metric].dropna()
            n = len(values)
            record[f"{metric}_n"] = n
            if n == 0:
                mean = sd = median = se = ci_low = ci_high = np.nan
            else:
                mean = values.mean()
                median = values.median()
                if n > 1:
                    sd = values.std(ddof=1)
                    se = sd / np.sqrt(n)
                    ci_low = mean - 1.96 * se
                    ci_high = mean + 1.96 * se
                else:
                    sd = se = ci_low = ci_high = np.nan
            record[f"{metric}_mean"] = mean
            record[f"{metric}_sd"] = sd
            record[f"{metric}_median"] = median
            record[f"{metric}_se"] = se
            record[f"{metric}_ci95_low"] = ci_low
            record[f"{metric}_ci95_high"] = ci_high
        record["mean_AUROC_rank_PU_25"] = frame["AUROC_rank_PU_25"].mean()
        record["mean_AUPRC_rank_PU_25"] = frame["AUPRC_rank_PU_25"].mean()
        record["mean_AUROC_rank_all_27"] = frame["AUROC_rank_all_27"].mean()
        record["mean_AUPRC_rank_all_27"] = frame["AUPRC_rank_all_27"].mean()
        summary_rows.append(record)
    summary = pd.DataFrame(summary_rows).sort_values(
        "AUROC_mean", ascending=False, na_position="last"
    )
    return raw_results, summary


def aggregate_condition(row: dict[str, str]) -> tuple[Path, Path, Path]:
    results_frames: list[pd.DataFrame] = []
    diagnostic_frames: list[pd.DataFrame] = []
    audit: list[dict] = []
    expected_seed_base = int(row["scenario_base_seed"])
    expected_ids = {f"M{index:03d}" for index in range(1, EXPECTED_METHODS + 1)}

    for rep in range(1, EXPECTED_REPETITIONS + 1):
        paths = repetition_paths(row, rep)
        issues: list[str] = []
        frame = None
        if not paths["results"].is_file():
            issues.append("missing results")
        else:
            try:
                frame = pd.read_csv(paths["results"], low_memory=False)
                if len(frame) != EXPECTED_METHODS:
                    issues.append(f"expected 27 rows, found {len(frame)}")
                if set(frame["method_id"].astype(str)) != expected_ids:
                    issues.append("method catalog is incomplete")
                if frame["method_id"].nunique() != EXPECTED_METHODS:
                    issues.append("method IDs are not unique")
                if frame["data_fingerprint"].nunique() != 1:
                    issues.append("methods do not share one data fingerprint")
                if pd.to_numeric(frame["AUROC"], errors="coerce").notna().sum() != EXPECTED_METHODS:
                    issues.append("one or more AUROC values are missing")
                observed_seeds = set(
                    pd.to_numeric(frame["seed"], errors="raise").astype(int)
                )
                if observed_seeds != {expected_seed_base + rep - 1}:
                    issues.append("seed differs from manifest")
            except Exception as error:
                issues.append(f"unreadable results: {error!r}")
        if not paths["diagnostics"].is_file():
            issues.append("missing diagnostics")
        if not paths["metadata"].is_file():
            issues.append("missing metadata")
        else:
            try:
                json.loads(paths["metadata"].read_text(encoding="utf-8"))
            except Exception as error:
                issues.append(f"unreadable metadata: {error!r}")
        if not issues and frame is not None:
            results_frames.append(frame)
            diagnostic_frames.append(
                pd.read_csv(paths["diagnostics"], low_memory=False)
            )
        audit.append({
            "condition_index": int(row["condition_index"]),
            "condition_id": row["condition_id"],
            "scenario_id": row["scenario_id"],
            "rep": rep,
            "expected_seed": expected_seed_base + rep - 1,
            "complete": not issues,
            "issues": "; ".join(issues),
        })

    report = pd.DataFrame(audit)
    aggregate = condition_base(row) / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    report_path = aggregate / f"{row['condition_id']}_collection_report.csv"
    atomic_csv(report, report_path)
    complete = int(report["complete"].sum())
    if complete != EXPECTED_REPETITIONS:
        raise RuntimeError(
            f"{row['condition_id']}: only {complete}/{EXPECTED_REPETITIONS} repetitions complete"
        )

    raw = pd.concat(results_frames, ignore_index=True)
    ranked, summary = rank_and_summarize(raw)
    diagnostics = pd.concat(diagnostic_frames, ignore_index=True)
    raw_path = aggregate / f"{row['condition_id']}_raw_100reps.csv"
    summary_path = aggregate / f"{row['condition_id']}_summary_100reps.csv"
    diagnostics_path = aggregate / f"{row['condition_id']}_diagnostics_100reps.csv"
    atomic_csv(ranked, raw_path)
    atomic_csv(summary, summary_path)
    atomic_csv(diagnostics, diagnostics_path)
    return raw_path, summary_path, diagnostics_path


def concatenate(sources: list[Path], destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_header = None
    try:
        with temporary.open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output, lineterminator="\n")
            for source in sources:
                with source.open("r", encoding="utf-8-sig", newline="") as handle:
                    reader = csv.reader(handle)
                    header = next(reader)
                    if expected_header is None:
                        expected_header = header
                        writer.writerow(header)
                    elif header != expected_header:
                        raise RuntimeError(f"header mismatch in {source}")
                    writer.writerows(reader)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    raw_paths: list[Path] = []
    summary_paths: list[Path] = []
    diagnostic_paths: list[Path] = []
    overview: list[dict] = []
    for row in manifest_rows():
        raw_path, summary_path, diagnostics_path = aggregate_condition(row)
        raw_paths.append(raw_path)
        summary_paths.append(summary_path)
        diagnostic_paths.append(diagnostics_path)
        overview.append({
            "condition_index": int(row["condition_index"]),
            "condition_id": row["condition_id"],
            "scenario_id": row["scenario_id"],
            "source_scenario_id": row["source_scenario_id"],
            "prior_ratio": row["prior_ratio"],
            "pi_true": row["pi_true"],
            "pi_supplied": row["pi_supplied"],
            "complete_repetitions": EXPECTED_REPETITIONS,
            "expected_repetitions": EXPECTED_REPETITIONS,
            "complete": True,
        })
        print(f"Validated {row['condition_id']}: 100/100")

    COMBINED.mkdir(parents=True, exist_ok=True)
    atomic_csv(
        pd.DataFrame(overview),
        COMBINED / "prior_condition_collection_overview.csv",
    )
    concatenate(raw_paths, COMBINED / "prior_misspecification_raw.csv")
    concatenate(
        summary_paths,
        COMBINED / "prior_misspecification_condition_summaries.csv",
    )
    concatenate(
        diagnostic_paths,
        COMBINED / "prior_misspecification_diagnostics.csv",
    )
    print(f"Complete combined confirmatory results: {COMBINED}")


if __name__ == "__main__":
    main()
