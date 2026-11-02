"""Traceable runner and collector for S001--S040 confirmatory simulations.

One Slurm task runs exactly one (scenario, repetition) pair.  The manifest is
the source of truth.  All 27 methods are run by confirmatory_benchmark.py on the same
PUExperiment object and therefore share one data_fingerprint per repetition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import build_confirmatory_manifest as frozen_design
import confirmatory_benchmark as benchmark


CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
DEFAULT_MANIFEST = (
    PROJECT_DIR / "config" / "confirmatory_scenarios_S001_S040_100reps.csv"
)
DEFAULT_RESULTS_ROOT = PROJECT_DIR / "results"
EXPECTED_SCENARIO_COUNT = 40
EXPECTED_METHOD_COUNT = 27
EXPECTED_PU_METHOD_COUNT = 25
EXPECTED_ORACLE_COUNT = 2

PARAMETER_COLUMNS = [
    "n_train",
    "n_val",
    "n_test",
    "p",
    "pi",
    "c",
    "delta",
    "n_signal",
    "rho",
    "covariance_type",
    "boundary_type",
    "structural_type",
    "transform_strength",
    "main_effect_shift",
    "positive_variance_ratio",
]

REQUIRED_MANIFEST_COLUMNS = {
    "scenario_index",
    "global_scenario_number",
    "structural_test_id",
    "scenario_id",
    "scenario_slug",
    "scenario_label",
    "scenario_family",
    "varied_factor",
    "is_reference",
    "reference_scenario_id",
    *PARAMETER_COLUMNS,
    "n_repetitions",
    "n_methods",
    "expected_method_rows",
    "global_base_seed",
    "scenario_base_seed",
    "seed_start",
    "seed_stop",
    "seed_scheme",
    "expected_labeled_train",
    "expected_labeled_val",
    "expected_positive_test",
    "pi_u_theoretical",
    "interpretation",
    "target_bayes_AUROC",
    "calibration_bayes_AUROC_mean",
    "calibration_oracle_logistic_AUROC_mean",
    "calibration_oracle_catboost_AUROC_mean",
    "pilot_source",
    "pilot_source_manifest",
    "pilot_source_manifest_sha256",
    "pilot_source_scenario_index",
    "pilot_scenario_base_seed",
    "pilot_n_repetitions",
    "confirmatory_phase",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_manifest(path: Path) -> pd.DataFrame:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Scenario manifest not found: {path}")
    manifest = pd.read_csv(path)
    manifest.columns = [str(column).strip() for column in manifest.columns]
    return manifest


def _same_value(left, right) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    if pd.isna(left) or pd.isna(right):
        return False
    try:
        left_number = float(left)
        right_number = float(right)
        return bool(np.isclose(left_number, right_number, rtol=1e-12, atol=1e-12))
    except (TypeError, ValueError):
        return str(left) == str(right)


def validate_manifest(manifest: pd.DataFrame) -> list[str]:
    """Return all manifest problems; an empty list means the design is valid."""
    issues: list[str] = []
    missing = sorted(REQUIRED_MANIFEST_COLUMNS.difference(manifest.columns))
    if missing:
        return ["missing columns: " + ", ".join(missing)]

    if len(manifest) != EXPECTED_SCENARIO_COUNT:
        issues.append(
            f"expected {EXPECTED_SCENARIO_COUNT} scenarios, found {len(manifest)}"
        )

    indices = pd.to_numeric(manifest["scenario_index"], errors="coerce")
    if indices.isna().any() or indices.astype("Int64").tolist() != list(
        range(1, len(manifest) + 1)
    ):
        issues.append("scenario_index must be consecutive integers starting at 1")

    for column in ["scenario_id", "scenario_slug"]:
        if manifest[column].isna().any() or manifest[column].duplicated().any():
            issues.append(f"{column} must be nonempty and unique")

    expected_ids = [f"S{index:03d}" for index in range(1, 41)]
    if manifest["scenario_id"].astype(str).tolist() != expected_ids:
        issues.append(f"scenario_id order must be {expected_ids}")
    test_map = {
        str(row.scenario_id): str(row.structural_test_id)
        for row in manifest.fillna("").itertuples()
        if str(row.structural_test_id)
    }
    expected_tests = {
        "S026": "ST04",
        "S038": "ST01",
        "S039": "ST02",
        "S040": "ST03",
    }
    if test_map != expected_tests:
        issues.append(f"structural test map must be {expected_tests}; found {test_map}")
    if not manifest["reference_scenario_id"].astype(str).eq("S001").all():
        issues.append("all scenarios must reference S001")
    reference_ids = manifest.loc[
        manifest["is_reference"].astype(str).str.upper().eq("TRUE"), "scenario_id"
    ].astype(str).tolist()
    if reference_ids != ["S001"]:
        issues.append("S001 must be the only reference scenario")

    repetitions = pd.to_numeric(manifest["n_repetitions"], errors="coerce")
    if repetitions.isna().any() or not repetitions.eq(100).all():
        issues.append("all scenarios must specify 100 confirmatory repetitions")

    # Rebuild the design from immutable pilot manifests and compare every
    # traceability/design field.  This prevents accidental scenario drift.
    expected_manifest = frozen_design.build()
    for column in expected_manifest.columns:
        if column not in manifest.columns:
            continue
        for row_index, (observed, expected) in enumerate(
            zip(manifest[column], expected_manifest[column]), start=1
        ):
            observed_blank = "" if pd.isna(observed) else observed
            expected_blank = "" if pd.isna(expected) else expected
            if not _same_value(observed_blank, expected_blank):
                issues.append(
                    f"row {row_index} column {column} differs from frozen design: "
                    f"observed={observed!r}, expected={expected!r}"
                )

    for _, row in manifest.iterrows():
        scenario_id = str(row["scenario_id"])

        pi = float(row["pi"])
        c = float(row["c"])
        n_train = int(row["n_train"])
        n_val = int(row["n_val"])
        n_test = int(row["n_test"])
        checks = {
            "expected_labeled_train": n_train * pi * c,
            "expected_labeled_val": n_val * pi * c,
            "expected_positive_test": n_test * pi,
            "pi_u_theoretical": pi * (1.0 - c) / (1.0 - pi * c),
        }
        for column, expected in checks.items():
            if not np.isclose(float(row[column]), expected, rtol=1e-8, atol=1e-8):
                issues.append(
                    f"{scenario_id} {column}={row[column]} but expected {expected}"
                )

        if int(row["n_methods"]) != EXPECTED_METHOD_COUNT:
            issues.append(f"{scenario_id} n_methods must be 27")
        if int(row["expected_method_rows"]) != 2700:
            issues.append(f"{scenario_id} expected_method_rows must be 2700")
        start = int(row["scenario_base_seed"])
        if int(row["seed_start"]) != start or int(row["seed_stop"]) != start + 99:
            issues.append(f"{scenario_id} seed_start/seed_stop violate the frozen rule")
        pilot_start = int(row["pilot_scenario_base_seed"])
        pilot_stop = pilot_start + int(row["pilot_n_repetitions"]) - 1
        if set(range(start, start + 100)).intersection(
            range(pilot_start, pilot_stop + 1)
        ):
            issues.append(f"{scenario_id} confirmatory and pilot seed ranges overlap")

    seed_ranges = []
    for _, row in manifest.iterrows():
        start = int(row["scenario_base_seed"])
        stop = start + int(row["n_repetitions"]) - 1
        seed_ranges.append((str(row["scenario_id"]), set(range(start, stop + 1))))
    for index, (left_id, left) in enumerate(seed_ranges):
        for right_id, right in seed_ranges[index + 1 :]:
            if left.intersection(right):
                issues.append(f"seed ranges overlap for {left_id} and {right_id}")

    return list(dict.fromkeys(issues))


def load_validated_manifest(path: Path) -> pd.DataFrame:
    manifest = read_manifest(path)
    issues = validate_manifest(manifest)
    if issues:
        raise ValueError("Invalid scenario manifest:\n- " + "\n- ".join(issues))
    return manifest


def scenario_row(manifest: pd.DataFrame, scenario_index: int) -> pd.Series:
    match = manifest.loc[
        pd.to_numeric(manifest["scenario_index"]) == int(scenario_index)
    ]
    if len(match) != 1:
        raise ValueError(f"Scenario index {scenario_index} was not found exactly once.")
    return match.iloc[0]


def scenario_directory(results_root: Path, row: pd.Series) -> Path:
    return Path(results_root) / "scenarios" / str(row["scenario_slug"])


def manifest_metadata(row: pd.Series) -> dict:
    return {
        key: (value.item() if isinstance(value, np.generic) else value)
        for key, value in row.to_dict().items()
    }


def prepare_results_tree(manifest: pd.DataFrame, manifest_path: Path, results_root: Path) -> None:
    """Create the fixed scenario directory tree and copy each definition into it."""
    manifest_hash = sha256_file(Path(manifest_path))
    for _, row in manifest.iterrows():
        root = scenario_directory(results_root, row)
        for child in ["repetitions", "aggregate", "run_metadata"]:
            (root / child).mkdir(parents=True, exist_ok=True)

        definition = pd.DataFrame([manifest_metadata(row)])
        benchmark.atomic_to_csv(definition, root / "scenario_definition.csv")
        atomic_json(
            {
                "manifest_sha256": manifest_hash,
                "scenario": manifest_metadata(row),
            },
            root / "scenario_definition.json",
        )
        readme = (
            f"# {row['scenario_id']} — {row['scenario_label']}\n\n"
            f"Folder: `{row['scenario_slug']}`  \n"
            f"Family: `{row['scenario_family']}`  \n"
            f"Varied factor: `{row['varied_factor']}`  \n"
            f"Reference: `{row['reference_scenario_id']}`\n\n"
            f"{row['interpretation']}\n\n"
            "- `repetitions/`: one result and diagnostics file per repetition.\n"
            "- `run_metadata/`: seeds and source/manifest hashes per repetition.\n"
            "- `aggregate/`: scenario-level collection audit and summaries.\n"
            "- `scenario_definition.csv/json`: immutable scenario parameters.\n"
        )
        (root / "README.md").write_text(readme, encoding="utf-8")

    registry = manifest.copy()
    registry.insert(0, "manifest_sha256", manifest_hash)
    benchmark.atomic_to_csv(
        registry, Path(results_root) / "confirmatory_scenario_registry.csv"
    )


def task_to_scenario_rep(task_id: int, manifest: pd.DataFrame) -> tuple[int, int]:
    repetitions = int(manifest["n_repetitions"].iloc[0])
    total_tasks = len(manifest) * repetitions
    if not (1 <= task_id <= total_tasks):
        raise ValueError(f"task_id must be between 1 and {total_tasks}")
    scenario_index = (task_id - 1) // repetitions + 1
    rep = (task_id - 1) % repetitions + 1
    return scenario_index, rep


def repetition_paths(root: Path, scenario_id: str, rep: int) -> dict[str, Path]:
    stem = f"{scenario_id}_rep_{rep:03d}"
    return {
        "results": root / "repetitions" / f"{stem}_results.csv",
        "diagnostics": root / "repetitions" / f"{stem}_diagnostics.csv",
        "failure": root / "repetitions" / f"{stem}_failure.csv",
        "metadata": root / "run_metadata" / f"{stem}_metadata.json",
    }


def _tag_frame(frame: pd.DataFrame, row: pd.Series, manifest_hash: str) -> pd.DataFrame:
    tagged = frame.copy()
    trace_columns = [
        "scenario_index",
        "global_scenario_number",
        "structural_test_id",
        "scenario_id",
        "scenario_slug",
        "scenario_label",
        "scenario_family",
        "varied_factor",
        "is_reference",
        "reference_scenario_id",
        "covariance_type",
        "boundary_type",
        "structural_type",
        "transform_strength",
        "main_effect_shift",
        "positive_variance_ratio",
        "pilot_source",
        "pilot_source_scenario_index",
        "pilot_scenario_base_seed",
        "confirmatory_phase",
    ]
    for column in reversed(trace_columns):
        tagged.insert(0, column, row[column])
    tagged["manifest_sha256"] = manifest_hash
    return tagged


def run_one_repetition(
    manifest: pd.DataFrame,
    manifest_path: Path,
    results_root: Path,
    scenario_index: int,
    rep: int,
    overwrite: bool = False,
) -> Path:
    row = scenario_row(manifest, scenario_index)
    n_repetitions = int(row["n_repetitions"])
    if not (1 <= rep <= n_repetitions):
        raise ValueError(f"rep must be between 1 and {n_repetitions}")

    root = scenario_directory(results_root, row)
    if not (root / "scenario_definition.csv").is_file():
        raise RuntimeError(
            "Scenario result folders are not prepared. Run "
            "`python code/structural_runner.py --prepare-results` once before submission."
        )
    paths = repetition_paths(root, str(row["scenario_id"]), rep)
    if paths["results"].is_file() and not overwrite:
        print(f"Complete result already exists; leaving unchanged: {paths['results']}")
        return paths["results"]

    manifest_hash = sha256_file(Path(manifest_path))
    scenario_seed = int(row["scenario_base_seed"]) + rep - 1
    benchmark_hash = sha256_file(CODE_DIR / "confirmatory_benchmark.py")
    runner_hash = sha256_file(Path(__file__).resolve())
    started_at = datetime.now(timezone.utc).isoformat()

    raw, _, diagnostics, failures = benchmark.run_repeated_benchmark(
        structural_type=str(row["structural_type"]),
        transform_strength=float(row["transform_strength"]),
        main_effect_shift=float(row["main_effect_shift"]),
        positive_variance_ratio=float(row["positive_variance_ratio"]),
        n_reps=n_repetitions,
        base_seed=int(row["scenario_base_seed"]),
        rep_ids=[rep],
        n_train=int(row["n_train"]),
        n_val=int(row["n_val"]),
        n_test=int(row["n_test"]),
        p=int(row["p"]),
        pi=float(row["pi"]),
        c=float(row["c"]),
        delta=float(row["delta"]),
        n_signal=int(row["n_signal"]),
        rho=float(row["rho"]),
    )

    if raw.empty:
        failure = failures.copy()
        failure = _tag_frame(failure, row, manifest_hash)
        benchmark.atomic_to_csv(failure, paths["failure"])
        raise RuntimeError(f"{row['scenario_id']} repetition {rep} produced no results")

    raw = _tag_frame(raw, row, manifest_hash)
    diagnostics = _tag_frame(diagnostics, row, manifest_hash)
    raw["phase"] = "phase3_confirmatory_100reps"
    diagnostics["phase"] = "phase3_confirmatory_100reps"

    method_count = raw["method"].nunique()
    oracle_count = raw["comparison_role"].eq("supervised_reference").sum()
    pu_count = raw["comparison_role"].eq("pu_method").sum()
    fingerprint_count = raw["data_fingerprint"].nunique()
    auroc_count = pd.to_numeric(raw["AUROC"], errors="coerce").notna().sum()
    validation_issues = []
    if len(raw) != EXPECTED_METHOD_COUNT or method_count != EXPECTED_METHOD_COUNT:
        validation_issues.append(
            f"expected {EXPECTED_METHOD_COUNT} method rows, found {len(raw)} "
            f"rows/{method_count} unique methods"
        )
    if pu_count != EXPECTED_PU_METHOD_COUNT:
        validation_issues.append(
            f"expected {EXPECTED_PU_METHOD_COUNT} PU methods, found {pu_count}"
        )
    if oracle_count != EXPECTED_ORACLE_COUNT:
        validation_issues.append(
            f"expected {EXPECTED_ORACLE_COUNT} oracle references, found {oracle_count}"
        )
    if fingerprint_count != 1:
        validation_issues.append(
            f"expected one shared data fingerprint, found {fingerprint_count}"
        )
    if auroc_count != EXPECTED_METHOD_COUNT:
        validation_issues.append(
            f"AUROC available for {auroc_count}/{EXPECTED_METHOD_COUNT} methods"
        )
    if set(pd.to_numeric(raw["seed"], errors="coerce").dropna()) != {scenario_seed}:
        validation_issues.append("result seed does not match the manifest seed rule")

    if validation_issues:
        problem = pd.DataFrame(
            [{
                "scenario_id": row["scenario_id"],
                "rep": rep,
                "seed": scenario_seed,
                "issues": "; ".join(validation_issues),
            }]
        )
        benchmark.atomic_to_csv(problem, paths["failure"])
        raise RuntimeError("; ".join(validation_issues))

    benchmark.atomic_to_csv(raw, paths["results"])
    benchmark.atomic_to_csv(diagnostics, paths["diagnostics"])
    finished_at = datetime.now(timezone.utc).isoformat()
    atomic_json(
        {
            "phase": "phase3_confirmatory_100reps",
            "scenario_index": int(row["scenario_index"]),
            "scenario_id": str(row["scenario_id"]),
            "scenario_slug": str(row["scenario_slug"]),
            "rep": rep,
            "seed": scenario_seed,
            "data_fingerprint": str(raw["data_fingerprint"].iloc[0]),
            "method_count": method_count,
            "pu_method_count": int(pu_count),
            "oracle_reference_count": int(oracle_count),
            "manifest_sha256": manifest_hash,
            "confirmatory_benchmark_sha256": benchmark_hash,
            "confirmatory_runner_sha256": runner_hash,
            "python_version": sys.version,
            "started_utc": started_at,
            "finished_utc": finished_at,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
        paths["metadata"],
    )
    print(
        f"Saved {row['scenario_id']} repetition {rep}: {paths['results']} "
        f"| seed={scenario_seed} | fingerprint={raw['data_fingerprint'].iloc[0]}"
    )
    return paths["results"]


def aggregate_scenario(
    manifest: pd.DataFrame,
    results_root: Path,
    scenario_index: int,
    allow_incomplete: bool,
) -> dict:
    row = scenario_row(manifest, scenario_index)
    scenario_id = str(row["scenario_id"])
    n_repetitions = int(row["n_repetitions"])
    root = scenario_directory(results_root, row)
    aggregate_dir = root / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    result_frames = []
    diagnostic_frames = []
    report_rows = []

    for rep in range(1, n_repetitions + 1):
        paths = repetition_paths(root, scenario_id, rep)
        expected_seed = int(row["scenario_base_seed"]) + rep - 1
        issues = []
        method_count = oracle_count = pu_count = fingerprint_count = auroc_count = 0
        if not paths["results"].is_file():
            issues.append("missing result file")
        else:
            try:
                frame = pd.read_csv(paths["results"])
                required = {
                    "scenario_id", "rep", "seed", "method", "comparison_role",
                    "AUROC", "data_fingerprint", "manifest_sha256",
                }
                missing = sorted(required.difference(frame.columns))
                if missing:
                    issues.append("missing columns: " + ", ".join(missing))
                else:
                    method_count = frame["method"].nunique()
                    oracle_count = frame["comparison_role"].eq(
                        "supervised_reference"
                    ).sum()
                    pu_count = frame["comparison_role"].eq("pu_method").sum()
                    fingerprint_count = frame["data_fingerprint"].nunique()
                    auroc_count = pd.to_numeric(
                        frame["AUROC"], errors="coerce"
                    ).notna().sum()
                    if len(frame) != EXPECTED_METHOD_COUNT:
                        issues.append(f"expected 27 rows, found {len(frame)}")
                    if method_count != EXPECTED_METHOD_COUNT:
                        issues.append(f"expected 27 methods, found {method_count}")
                    if oracle_count != EXPECTED_ORACLE_COUNT:
                        issues.append(f"expected 2 oracle rows, found {oracle_count}")
                    if pu_count != EXPECTED_PU_METHOD_COUNT:
                        issues.append(f"expected 25 PU rows, found {pu_count}")
                    if fingerprint_count != 1:
                        issues.append(
                            f"expected one fingerprint, found {fingerprint_count}"
                        )
                    if auroc_count != EXPECTED_METHOD_COUNT:
                        issues.append(f"AUROC available for {auroc_count}/27 methods")
                    if set(frame["scenario_id"].astype(str)) != {scenario_id}:
                        issues.append("unexpected scenario_id")
                    if set(pd.to_numeric(frame["rep"], errors="coerce")) != {rep}:
                        issues.append("unexpected repetition number")
                    if set(pd.to_numeric(frame["seed"], errors="coerce")) != {
                        expected_seed
                    }:
                        issues.append("unexpected seed")
                    if not issues:
                        result_frames.append(frame)
            except Exception as error:
                issues.append(f"could not read result file: {error!r}")

        if not paths["diagnostics"].is_file():
            issues.append("missing diagnostics file")
        elif not issues:
            try:
                diagnostic_frames.append(pd.read_csv(paths["diagnostics"]))
            except Exception as error:
                issues.append(f"could not read diagnostics file: {error!r}")
        if not paths["metadata"].is_file():
            issues.append("missing run metadata file")

        report_rows.append(
            {
                "scenario_index": int(row["scenario_index"]),
                "scenario_id": scenario_id,
                "scenario_slug": row["scenario_slug"],
                "rep": rep,
                "expected_seed": expected_seed,
                "method_count": method_count,
                "pu_method_count": pu_count,
                "oracle_reference_count": oracle_count,
                "methods_with_AUROC": auroc_count,
                "data_fingerprint_count": fingerprint_count,
                "complete": len(issues) == 0,
                "issues": "; ".join(issues),
            }
        )

    report = pd.DataFrame(report_rows)
    benchmark.atomic_to_csv(report, aggregate_dir / f"{scenario_id}_collection_report.csv")
    complete_count = int(report["complete"].sum())
    if complete_count != n_repetitions and not allow_incomplete:
        raise RuntimeError(
            f"{scenario_id} has {complete_count}/{n_repetitions} complete repetitions; "
            f"inspect {aggregate_dir / f'{scenario_id}_collection_report.csv'}"
        )
    if not result_frames:
        raise RuntimeError(f"{scenario_id} has no complete repetition files to aggregate")

    raw = pd.concat(result_frames, ignore_index=True)
    raw, summary = benchmark.rank_and_summarize(raw)
    diagnostics = (
        pd.concat(diagnostic_frames, ignore_index=True)
        if diagnostic_frames
        else pd.DataFrame()
    )
    summary.insert(0, "scenario_label", row["scenario_label"])
    summary.insert(0, "scenario_slug", row["scenario_slug"])
    summary.insert(0, "scenario_id", scenario_id)
    summary.insert(0, "scenario_index", int(row["scenario_index"]))

    raw_path = aggregate_dir / f"{scenario_id}_raw_{complete_count}reps.csv"
    summary_path = aggregate_dir / f"{scenario_id}_summary_{complete_count}reps.csv"
    diagnostics_path = aggregate_dir / f"{scenario_id}_diagnostics_{complete_count}reps.csv"
    benchmark.atomic_to_csv(raw, raw_path)
    benchmark.atomic_to_csv(summary, summary_path)
    benchmark.atomic_to_csv(diagnostics, diagnostics_path)
    print(f"Aggregated {scenario_id}: {complete_count}/{n_repetitions} complete")
    return {
        "scenario_index": int(row["scenario_index"]),
        "scenario_id": scenario_id,
        "scenario_slug": str(row["scenario_slug"]),
        "complete_repetitions": complete_count,
        "expected_repetitions": n_repetitions,
        "complete": complete_count == n_repetitions,
        "raw_path": str(raw_path),
        "summary_path": str(summary_path),
        "diagnostics_path": str(diagnostics_path),
    }


def aggregate_all(
    manifest: pd.DataFrame,
    results_root: Path,
    allow_incomplete: bool,
) -> None:
    combined_dir = Path(results_root) / "confirmatory_combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    registry_rows = []
    raw_frames = []
    summary_frames = []
    diagnostic_frames = []

    for scenario_index in manifest["scenario_index"].astype(int):
        try:
            result = aggregate_scenario(
                manifest,
                results_root,
                scenario_index,
                allow_incomplete=allow_incomplete,
            )
            registry_rows.append(result)
            raw_frames.append(pd.read_csv(result["raw_path"]))
            summary_frames.append(pd.read_csv(result["summary_path"]))
            diagnostic_frames.append(pd.read_csv(result["diagnostics_path"]))
        except Exception as error:
            row = scenario_row(manifest, scenario_index)
            registry_rows.append(
                {
                    "scenario_index": scenario_index,
                    "scenario_id": row["scenario_id"],
                    "scenario_slug": row["scenario_slug"],
                    "complete_repetitions": 0,
                    "expected_repetitions": int(row["n_repetitions"]),
                    "complete": False,
                    "error": repr(error),
                }
            )
            if not allow_incomplete:
                raise

    registry = pd.DataFrame(registry_rows).sort_values("scenario_index")
    benchmark.atomic_to_csv(
        registry,
        combined_dir / "confirmatory_collection_overview.csv",
    )
    if raw_frames:
        benchmark.atomic_to_csv(
            pd.concat(raw_frames, ignore_index=True),
            combined_dir / "confirmatory_raw_all_scenarios.csv",
        )
    if summary_frames:
        benchmark.atomic_to_csv(
            pd.concat(summary_frames, ignore_index=True),
            combined_dir / "confirmatory_method_scenario_summary.csv",
        )
    if diagnostic_frames:
        benchmark.atomic_to_csv(
            pd.concat(diagnostic_frames, ignore_index=True),
            combined_dir / "confirmatory_diagnostics_all_scenarios.csv",
        )
    print(f"Confirmatory collection written to {combined_dir}")


def print_scenarios(manifest: pd.DataFrame) -> None:
    columns = [
        "scenario_index",
        "scenario_id",
        "scenario_label",
        "scenario_family",
        "varied_factor",
        "pi",
        "c",
        "p",
        "delta",
        "rho",
        "scenario_base_seed",
    ]
    print(manifest[columns].to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--rep", type=int, help="One-based repetition for --run-scenario")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-manifest", action="store_true")
    mode.add_argument("--prepare-results", action="store_true")
    mode.add_argument("--list-scenarios", action="store_true")
    mode.add_argument("--check-dependencies", action="store_true")
    mode.add_argument(
        "--task-id",
        type=int,
        help="One-based Slurm array task; maps across 40 scenarios x 100 reps.",
    )
    mode.add_argument("--run-scenario", type=int, metavar="INDEX")
    mode.add_argument("--aggregate-scenario", type=int, metavar="INDEX")
    mode.add_argument("--aggregate-all", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = load_validated_manifest(args.manifest)

    if args.validate_manifest:
        print(
            f"VALID: {len(manifest)} scenarios, "
            f"{int(manifest['n_repetitions'].iloc[0])} repetitions each, "
            f"{len(manifest) * int(manifest['n_repetitions'].iloc[0])} tasks."
        )
        return
    if args.prepare_results:
        prepare_results_tree(manifest, args.manifest, args.results_root)
        print(f"Prepared scenario folders below {args.results_root / 'scenarios'}")
        return
    if args.list_scenarios:
        print_scenarios(manifest)
        return
    if args.check_dependencies:
        raise SystemExit(0 if benchmark.dependency_status() else 1)
    if args.task_id is not None:
        scenario_index, rep = task_to_scenario_rep(args.task_id, manifest)
        run_one_repetition(
            manifest,
            args.manifest,
            args.results_root,
            scenario_index,
            rep,
            overwrite=args.overwrite,
        )
        return
    if args.run_scenario is not None:
        if args.rep is None:
            raise SystemExit("--run-scenario requires --rep")
        run_one_repetition(
            manifest,
            args.manifest,
            args.results_root,
            args.run_scenario,
            args.rep,
            overwrite=args.overwrite,
        )
        return
    if args.aggregate_scenario is not None:
        aggregate_scenario(
            manifest,
            args.results_root,
            args.aggregate_scenario,
            allow_incomplete=args.allow_incomplete,
        )
        return
    if args.aggregate_all:
        aggregate_all(
            manifest,
            args.results_root,
            allow_incomplete=args.allow_incomplete,
        )
        return


if __name__ == "__main__":
    main()
