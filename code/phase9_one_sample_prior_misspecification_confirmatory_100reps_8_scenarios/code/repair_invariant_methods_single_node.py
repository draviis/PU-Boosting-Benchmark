"""Stage, validate, and atomically promote the targeted Phase 9 repair."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import one_sample_benchmark as benchmark
import prior_misspecification_benchmark as prior


PROJECT_DIR = Path(__file__).resolve().parent.parent
STUDY_MANIFEST = (
    PROJECT_DIR / "config" / "prior_misspecification_confirmatory_manifest.csv"
)
RESULTS_ROOT = PROJECT_DIR / "results"
REPAIR_ROOT = PROJECT_DIR / "repair" / "invariant_methods_single_node_4groups"
GROUP_MANIFEST = REPAIR_ROOT / "repair_group_manifest.csv"
STAGED_ROOT = REPAIR_ROOT / "staged_results"
BACKUP_ROOT = REPAIR_ROOT / "original_backup"
AUDIT_PATH = REPAIR_ROOT / "invariant_repair_audit.csv"
METADATA_PATH = REPAIR_ROOT / "invariant_repair_metadata.json"
FINAL_ANALYSIS_METADATA = (
    RESULTS_ROOT
    / "prior_misspecification_analysis"
    / "prior_misspecification_confirmatory_metadata.json"
)

EXPECTED_METHODS = 27
EXPECTED_CONDITIONS = 40
EXPECTED_REPETITIONS = 100
EXPECTED_GROUPS = 4
EXPECTED_RESULT_FILES = 20
EXPECTED_METHOD_ROWS = 25
STRICT_TOLERANCE = 1e-12
ANALYSIS_TOLERANCE = 5e-5
RATIOS = {0.8, 0.9, 1.0, 1.1, 1.2}
METHOD_SPECS = {
    "M006": ("Naive", "AdaBoost", "Block2 | Naive + AdaBoost"),
    "M011": (
        "ProbCorrection",
        "AdaBoost",
        "Block2 | ProbCorrection + AdaBoost",
    ),
    "M016": ("RN", "AdaBoost", "Block2 | RN + AdaBoost"),
}
INVARIANT_METRICS = {
    "M006": ("AUROC", "AUPRC"),
    "M011": ("AUROC", "AUPRC", "Brier", "LogLoss"),
    "M016": ("AUROC", "AUPRC"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False, lineterminator="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_study_manifest() -> pd.DataFrame:
    manifest = pd.read_csv(STUDY_MANIFEST)
    if len(manifest) != EXPECTED_CONDITIONS:
        raise RuntimeError(f"expected 40 conditions, found {len(manifest)}")
    if manifest["condition_index"].nunique() != EXPECTED_CONDITIONS:
        raise RuntimeError("condition indexes are not unique")
    return manifest


def load_group_manifest() -> pd.DataFrame:
    groups = pd.read_csv(GROUP_MANIFEST, dtype={"scenario_id": str})
    required = {"group_id", "scenario_id", "rep", "method_ids", "reason"}
    if not required.issubset(groups.columns):
        raise RuntimeError(
            f"repair group manifest lacks {sorted(required - set(groups.columns))}"
        )
    if len(groups) != EXPECTED_GROUPS or groups["group_id"].nunique() != EXPECTED_GROUPS:
        raise RuntimeError("repair group manifest must contain four unique groups")
    observed = {
        (str(row.scenario_id), int(row.rep), tuple(str(row.method_ids).split(";")))
        for row in groups.itertuples(index=False)
    }
    expected = {
        ("OPM002", 8, ("M016",)),
        ("OPM002", 44, ("M016",)),
        ("OPM003", 51, ("M016",)),
        ("OPM005", 35, ("M006", "M011")),
    }
    if observed != expected:
        raise RuntimeError(f"repair scope changed: {sorted(observed)}")
    return groups


def result_path(root: Path, row: pd.Series, rep: int) -> Path:
    base = (
        root
        / "scenarios"
        / str(row["scenario_slug"])
        / "prior_conditions"
        / str(row["condition_slug"])
    )
    stem = f"{row['condition_id']}_rep_{int(rep):03d}"
    return base / "repetitions" / f"{stem}_results.csv"


def resolve_targets() -> list[dict]:
    manifest = load_study_manifest()
    groups = load_group_manifest()
    targets: list[dict] = []
    for group in groups.itertuples(index=False):
        rows = manifest.loc[manifest["scenario_id"].astype(str).eq(str(group.scenario_id))]
        if len(rows) != 5 or set(rows["prior_ratio"].astype(float)) != RATIOS:
            raise RuntimeError(f"{group.group_id} does not map to a complete five-ratio grid")
        methods = tuple(str(group.method_ids).split(";"))
        for method_id in methods:
            if method_id not in METHOD_SPECS:
                raise RuntimeError(f"unsupported repair method {method_id}")
        for _, row in rows.sort_values("prior_ratio").iterrows():
            original = result_path(RESULTS_ROOT, row, int(group.rep))
            staged = result_path(STAGED_ROOT, row, int(group.rep))
            task_id = (int(row["condition_index"]) - 1) * EXPECTED_REPETITIONS + int(group.rep)
            targets.append({
                "group_id": str(group.group_id),
                "scenario_id": str(group.scenario_id),
                "rep": int(group.rep),
                "method_ids": methods,
                "reason": str(group.reason),
                "task_id": task_id,
                "condition_index": int(row["condition_index"]),
                "condition_id": str(row["condition_id"]),
                "prior_ratio": float(row["prior_ratio"]),
                "seed": int(row["scenario_base_seed"]) + int(group.rep) - 1,
                "row": row,
                "original_path": original,
                "staged_path": staged,
            })
    if len(targets) != EXPECTED_RESULT_FILES:
        raise RuntimeError(f"expected 20 result files, resolved {len(targets)}")
    if sum(len(item["method_ids"]) for item in targets) != EXPECTED_METHOD_ROWS:
        raise RuntimeError("repair must replace exactly 25 method rows")
    return sorted(targets, key=lambda item: item["task_id"])


def validate_result(frame: pd.DataFrame, item: dict, label: str) -> None:
    expected_ids = {f"M{index:03d}" for index in range(1, EXPECTED_METHODS + 1)}
    if len(frame) != EXPECTED_METHODS or set(frame["method_id"].astype(str)) != expected_ids:
        raise RuntimeError(f"{label} task {item['task_id']} is not a complete 27-method file")
    if frame["method_id"].nunique() != EXPECTED_METHODS:
        raise RuntimeError(f"{label} task {item['task_id']} has duplicate method IDs")
    if frame["data_fingerprint"].nunique() != 1:
        raise RuntimeError(f"{label} task {item['task_id']} has multiple fingerprints")
    if set(pd.to_numeric(frame["seed"], errors="raise").astype(int)) != {item["seed"]}:
        raise RuntimeError(f"{label} task {item['task_id']} seed changed")
    if set(frame["scenario_id"].astype(str)) != {item["scenario_id"]}:
        raise RuntimeError(f"{label} task {item['task_id']} scenario changed")
    if set(pd.to_numeric(frame["rep"], errors="raise").astype(int)) != {item["rep"]}:
        raise RuntimeError(f"{label} task {item['task_id']} repetition changed")
    if not np.allclose(
        pd.to_numeric(frame["prior_ratio"], errors="raise"),
        item["prior_ratio"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(f"{label} task {item['task_id']} prior ratio changed")


def target_rows(targets: list[dict], root_key: str) -> pd.DataFrame:
    records: list[dict] = []
    for item in targets:
        path = item[root_key]
        if not path.is_file():
            raise RuntimeError(f"missing {root_key}: {path}")
        frame = pd.read_csv(path, low_memory=False)
        validate_result(frame, item, root_key)
        for method_id in item["method_ids"]:
            match = frame.loc[frame["method_id"].eq(method_id)]
            if len(match) != 1:
                raise RuntimeError(f"task {item['task_id']} lacks one {method_id} row")
            row = match.iloc[0]
            records.append({
                "group_id": item["group_id"],
                "scenario_id": item["scenario_id"],
                "rep": item["rep"],
                "task_id": item["task_id"],
                "condition_id": item["condition_id"],
                "prior_ratio": item["prior_ratio"],
                "method_id": method_id,
                "data_fingerprint": str(row["data_fingerprint"]),
                "AUROC": float(row["AUROC"]),
                "AUPRC": float(row["AUPRC"]),
                "Brier": pd.to_numeric(pd.Series([row.get("Brier")]), errors="coerce").iloc[0],
                "LogLoss": pd.to_numeric(pd.Series([row.get("LogLoss")]), errors="coerce").iloc[0],
                "F1": float(row["F1"]),
                "BalancedAccuracy": float(row["BalancedAccuracy"]),
                "MCC": float(row["MCC"]),
                "threshold": float(row["threshold"]),
                "val_metric": float(row["val_metric"]),
                "best_params": str(row["best_params"]),
            })
    return pd.DataFrame(records)


def invariance_summary(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for (group_id, scenario_id, rep, method_id), frame in rows.groupby(
        ["group_id", "scenario_id", "rep", "method_id"]
    ):
        if len(frame) != 5 or set(frame["prior_ratio"].astype(float)) != RATIOS:
            raise RuntimeError(f"incomplete five-ratio group {(group_id, method_id)}")
        if frame["data_fingerprint"].nunique() != 1:
            raise RuntimeError(f"data fingerprint changed in {(group_id, method_id)}")
        if frame["best_params"].nunique() != 1:
            raise RuntimeError(f"best parameters differ in {(group_id, method_id)}")
        val_span = float(frame["val_metric"].max() - frame["val_metric"].min())
        if val_span > STRICT_TOLERANCE:
            raise RuntimeError(
                f"validation metric span {val_span:.12g} in {(group_id, method_id)}"
            )
        for metric in INVARIANT_METRICS[method_id]:
            values = pd.to_numeric(frame[metric], errors="coerce")
            if values.isna().any():
                raise RuntimeError(f"missing {metric} in {(group_id, method_id)}")
            span = float(values.max() - values.min())
            records.append({
                "group_id": group_id,
                "scenario_id": scenario_id,
                "rep": int(rep),
                "method_id": method_id,
                "metric": metric,
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "span": span,
                "strict_tolerance": STRICT_TOLERANCE,
                "strict_pass": bool(span <= STRICT_TOLERANCE),
                "publication_tolerance": ANALYSIS_TOLERANCE,
                "publication_pass": bool(span <= ANALYSIS_TOLERANCE),
            })
    summary = pd.DataFrame(records)
    if not summary["strict_pass"].all():
        failed = summary.loc[~summary["strict_pass"]]
        raise RuntimeError(f"strict staged invariance failed:\n{failed.to_string(index=False)}")
    return summary


def validate_targets() -> None:
    targets = resolve_targets()
    rows = target_rows(targets, "original_path")
    failures = []
    for keys, frame in rows.groupby(["group_id", "method_id"]):
        for metric in INVARIANT_METRICS[keys[1]]:
            values = pd.to_numeric(frame[metric], errors="coerce")
            span = float(values.max() - values.min())
            if span > ANALYSIS_TOLERANCE:
                failures.append((keys[0], keys[1], metric, span))
    print(
        f"VALID: {EXPECTED_GROUPS} groups, {EXPECTED_RESULT_FILES} files, "
        f"{EXPECTED_METHOD_ROWS} method rows"
    )
    print(f"Current publication-tolerance failures in repair scope: {len(failures)}")
    for group_id, method_id, metric, span in failures:
        print(f"  {group_id} {method_id} {metric}: span={span:.12g}")


def generate_group_data(row: pd.Series, rep: int):
    seed = int(row["scenario_base_seed"]) + int(rep) - 1
    return benchmark.generate_structural_one_sample_scar(
        n_train=int(row["n_train"]),
        n_val=int(row["n_val"]),
        n_test=int(row["n_test"]),
        p=int(row["p"]),
        pi=float(row["pi_true"]),
        c=float(row["c_true"]),
        delta=float(row["delta"]),
        n_signal=int(row["n_signal"]),
        rho=float(row["rho"]),
        seed=seed,
        structural_type=str(row["structural_type"]),
        transform_strength=float(row["transform_strength"]),
        main_effect_shift=float(row["main_effect_shift"]),
        positive_variance_ratio=float(row["positive_variance_ratio"]),
    )


def expose_supplied_prior(data, row: pd.Series) -> None:
    pi_true = float(row["pi_true"])
    pi_supplied = float(row["pi_supplied"])
    c_true = float(row["c_true"])
    ratio = float(row["prior_ratio"])
    data.pi_true = pi_true
    data.pi_supplied = pi_supplied
    data.c_true = c_true
    data.prior_ratio = ratio
    data.pi_u_true = prior.population_to_unlabeled_prior(pi_true, c_true)
    data.pi_u_supplied = prior.population_to_unlabeled_prior(pi_supplied, c_true)
    data.pi = pi_supplied
    data.c = c_true


def stage_repair() -> None:
    targets = resolve_targets()
    manifest = load_study_manifest()
    for group_id, group_targets in pd.DataFrame(targets).groupby("group_id", sort=True):
        items = group_targets.to_dict("records")
        first = items[0]
        scenario_rows = manifest.loc[
            manifest["scenario_id"].astype(str).eq(first["scenario_id"])
        ].sort_values("prior_ratio")
        data = generate_group_data(scenario_rows.iloc[0], first["rep"])
        fingerprint = benchmark.experiment_fingerprint(data)
        original_fingerprints = set()
        for item in items:
            original = pd.read_csv(item["original_path"], low_memory=False)
            validate_result(original, item, "original")
            original_fingerprints.add(str(original["data_fingerprint"].iloc[0]))
        if original_fingerprints != {fingerprint}:
            raise RuntimeError(
                f"{group_id}: regenerated fingerprint {fingerprint} differs from originals "
                f"{sorted(original_fingerprints)}"
            )
        print(
            f"{group_id}: regenerated {first['scenario_id']} rep {first['rep']} "
            f"once; fingerprint={fingerprint}"
        )

        for item in items:
            row = item["row"]
            expose_supplied_prior(data, row)
            original = pd.read_csv(item["original_path"], low_memory=False)
            repaired = original.copy()
            for method_id in item["method_ids"]:
                strategy, engine, expected_name = METHOD_SPECS[method_id]
                print(
                    f"  fitting {item['condition_id']} rep {item['rep']} "
                    f"ratio={item['prior_ratio']:.1f} {method_id}"
                )
                result = benchmark.base.fit_one_boosting_cell(data, strategy, engine)
                if result.get("method") != expected_name:
                    raise RuntimeError(
                        f"{method_id}: unexpected method name {result.get('method')!r}"
                    )
                target_index = repaired.index[repaired["method_id"].eq(method_id)]
                if len(target_index) != 1:
                    raise RuntimeError(f"{item['task_id']}: cannot locate one {method_id} row")
                index = target_index[0]
                for column, value in result.items():
                    if column not in repaired.columns:
                        raise RuntimeError(f"result column {column} is absent from source CSV")
                    repaired.at[index, column] = value
                if benchmark.experiment_fingerprint(data) != fingerprint:
                    raise RuntimeError(f"{method_id} modified the frozen data arrays")
            atomic_csv(repaired, item["staged_path"])

    staged_rows = target_rows(targets, "staged_path")
    strict = invariance_summary(staged_rows)
    atomic_csv(strict, REPAIR_ROOT / "staged_invariance_audit.csv")
    atomic_json({
        "repair_phase": "phase9_invariant_methods_single_node_repair",
        "status": "staged_validated_pending_promotion",
        "groups": EXPECTED_GROUPS,
        "result_files": EXPECTED_RESULT_FILES,
        "method_rows": EXPECTED_METHOD_ROWS,
        "single_node": True,
        "single_thread": True,
        "strict_tolerance": STRICT_TOLERANCE,
        "repair_host": socket.gethostname(),
        "repair_slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "python_version": sys.version,
        "repair_script_sha256": sha256_file(Path(__file__)),
        "repair_group_manifest_sha256": sha256_file(GROUP_MANIFEST),
        "one_sample_benchmark_sha256": sha256_file(PROJECT_DIR / "code" / "one_sample_benchmark.py"),
        "base_benchmark_sha256": sha256_file(PROJECT_DIR / "code" / "base_phase1_benchmark.py"),
        "staged_utc": datetime.now(timezone.utc).isoformat(),
    }, METADATA_PATH)
    print("PASS: all 20 staged files satisfy strict cross-ratio invariance")


def compare_unaffected(original: pd.DataFrame, staged: pd.DataFrame, item: dict) -> None:
    unaffected = sorted(
        set(original["method_id"].astype(str)) - set(item["method_ids"])
    )
    left = original.loc[original["method_id"].isin(unaffected)].sort_values("method_id")
    right = staged.loc[staged["method_id"].isin(unaffected)].sort_values("method_id")
    left = left.reset_index(drop=True)
    right = right.reset_index(drop=True)
    if list(left.columns) != list(right.columns):
        raise RuntimeError(f"task {item['task_id']}: staged columns changed")
    for column in left.columns:
        if pd.api.types.is_numeric_dtype(left[column]) and pd.api.types.is_numeric_dtype(right[column]):
            a = pd.to_numeric(left[column], errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(right[column], errors="coerce").to_numpy(dtype=float)
            if not np.array_equal(a, b, equal_nan=True):
                raise RuntimeError(
                    f"task {item['task_id']}: unaffected numeric column {column} changed"
                )
        else:
            if not left[column].fillna("<NA>").astype(str).equals(
                right[column].fillna("<NA>").astype(str)
            ):
                raise RuntimeError(
                    f"task {item['task_id']}: unaffected text column {column} changed"
                )


def preserve_unaffected_csv_fields(item: dict) -> None:
    """Restore every non-target CSV field from the original file verbatim.

    pandas intentionally shortens some decimal strings when it rewrites a CSV.
    Although those changes are only at machine precision, a targeted repair
    should not alter any non-target field. This merge retains the staged target
    rows and copies all other field strings directly from the source CSV.
    """
    with item["original_path"].open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        original_reader = csv.DictReader(handle)
        original_header = list(original_reader.fieldnames or [])
        original_rows = list(original_reader)
    with item["staged_path"].open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        staged_reader = csv.DictReader(handle)
        staged_header = list(staged_reader.fieldnames or [])
        staged_rows = list(staged_reader)
    if original_header != staged_header:
        raise RuntimeError(f"task {item['task_id']}: staged header changed")
    staged_by_method = {row["method_id"]: row for row in staged_rows}
    if len(staged_by_method) != EXPECTED_METHODS or len(original_rows) != EXPECTED_METHODS:
        raise RuntimeError(f"task {item['task_id']}: staged/original row count changed")
    merged_rows = [
        staged_by_method[row["method_id"]]
        if row["method_id"] in item["method_ids"]
        else row
        for row in original_rows
    ]
    temporary = item["staged_path"].with_name(
        f".{item['staged_path'].name}.preserve.{os.getpid()}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=original_header, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(merged_rows)
        os.replace(temporary, item["staged_path"])
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_and_promote() -> None:
    targets = resolve_targets()
    for item in targets:
        preserve_unaffected_csv_fields(item)
    staged_rows = target_rows(targets, "staged_path")
    strict = invariance_summary(staged_rows)
    audit_records: list[dict] = []
    prepared: list[tuple[Path, Path, Path]] = []
    for item in targets:
        original = pd.read_csv(item["original_path"], low_memory=False)
        staged = pd.read_csv(item["staged_path"], low_memory=False)
        validate_result(original, item, "original")
        validate_result(staged, item, "staged")
        compare_unaffected(original, staged, item)
        if str(original["data_fingerprint"].iloc[0]) != str(staged["data_fingerprint"].iloc[0]):
            raise RuntimeError(f"task {item['task_id']}: fingerprint changed")

        relative = item["original_path"].relative_to(RESULTS_ROOT)
        backup = BACKUP_ROOT / relative
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            shutil.copy2(item["original_path"], backup)
        temporary = item["original_path"].with_name(
            f".{item['original_path'].name}.repair.{os.getpid()}.tmp"
        )
        shutil.copy2(item["staged_path"], temporary)
        prepared.append((item["original_path"], backup, temporary))
        for method_id in item["method_ids"]:
            before = original.loc[original["method_id"].eq(method_id)].iloc[0]
            after = staged.loc[staged["method_id"].eq(method_id)].iloc[0]
            audit_records.append({
                "group_id": item["group_id"],
                "scenario_id": item["scenario_id"],
                "rep": item["rep"],
                "task_id": item["task_id"],
                "condition_id": item["condition_id"],
                "prior_ratio": item["prior_ratio"],
                "method_id": method_id,
                "reason": item["reason"],
                "data_fingerprint": str(after["data_fingerprint"]),
                "original_AUROC": float(before["AUROC"]),
                "repaired_AUROC": float(after["AUROC"]),
                "original_AUPRC": float(before["AUPRC"]),
                "repaired_AUPRC": float(after["AUPRC"]),
                "original_Brier": pd.to_numeric(pd.Series([before.get("Brier")]), errors="coerce").iloc[0],
                "repaired_Brier": pd.to_numeric(pd.Series([after.get("Brier")]), errors="coerce").iloc[0],
                "original_LogLoss": pd.to_numeric(pd.Series([before.get("LogLoss")]), errors="coerce").iloc[0],
                "repaired_LogLoss": pd.to_numeric(pd.Series([after.get("LogLoss")]), errors="coerce").iloc[0],
                "original_best_params": str(before["best_params"]),
                "repaired_best_params": str(after["best_params"]),
                "original_result_sha256": sha256_file(item["original_path"]),
                "staged_result_sha256": sha256_file(item["staged_path"]),
                "backup_path": str(backup),
                "backup_sha256": sha256_file(backup),
            })

    promoted: list[tuple[Path, Path]] = []
    try:
        for original_path, backup, temporary in prepared:
            os.replace(temporary, original_path)
            promoted.append((original_path, backup))
    except Exception:
        for original_path, backup in promoted:
            shutil.copy2(backup, original_path)
        raise
    finally:
        for _, _, temporary in prepared:
            if temporary.exists():
                temporary.unlink()

    final_rows = target_rows(targets, "original_path")
    final_strict = invariance_summary(final_rows)
    if len(final_strict) != len(strict):
        raise RuntimeError("post-promotion invariance audit size changed")
    audit = pd.DataFrame(audit_records)
    audit["repaired_result_sha256"] = audit.apply(
        lambda row: sha256_file(
            next(
                item["original_path"]
                for item in targets
                if item["task_id"] == int(row["task_id"])
            )
        ),
        axis=1,
    )
    audit["repair_host"] = socket.gethostname()
    audit["repair_slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
    audit["promoted_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_csv(audit, AUDIT_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    metadata.update({
        "status": "promoted_pending_full_reanalysis",
        "promoted_files": EXPECTED_RESULT_FILES,
        "promoted_method_rows": EXPECTED_METHOD_ROWS,
        "unaffected_method_rows_preserved": EXPECTED_RESULT_FILES * EXPECTED_METHODS - EXPECTED_METHOD_ROWS,
        "post_promotion_strict_checks": int(len(final_strict)),
        "post_promotion_strict_checks_passing": int(final_strict["strict_pass"].sum()),
        "promoted_utc": datetime.now(timezone.utc).isoformat(),
    })
    atomic_json(metadata, METADATA_PATH)
    print("PASS: originals backed up and exactly 25 method rows promoted in 20 files")


def verify_final() -> None:
    if not FINAL_ANALYSIS_METADATA.is_file():
        raise RuntimeError("final analysis metadata was not regenerated")
    final = json.loads(FINAL_ANALYSIS_METADATA.read_text(encoding="utf-8"))
    required = {
        "integrity_pass": True,
        "expected_tasks": 4000,
        "observed_tasks": 4000,
        "expected_method_rows": 108000,
        "observed_method_rows": 108000,
        "ranking_negative_controls_passing": 36800,
        "ranking_negative_controls_expected": 36800,
        "c_only_probability_metrics_passing": 25600,
        "c_only_probability_metrics_expected": 25600,
    }
    mismatches = {
        key: {"expected": expected, "observed": final.get(key)}
        for key, expected in required.items()
        if final.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"final publication gate failed: {mismatches}")
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    metadata.update({
        "status": "complete",
        "final_integrity_pass": True,
        "final_analysis_metadata_sha256": sha256_file(FINAL_ANALYSIS_METADATA),
        "verified_utc": datetime.now(timezone.utc).isoformat(),
    })
    atomic_json(metadata, METADATA_PATH)
    print("PASS: targeted repair and full Phase 9 publication gate are complete")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-targets", action="store_true")
    parser.add_argument("--stage", action="store_true")
    parser.add_argument("--validate-and-promote", action="store_true")
    parser.add_argument("--verify-final", action="store_true")
    args = parser.parse_args()
    actions = sum(
        [args.validate_targets, args.stage, args.validate_and_promote, args.verify_final]
    )
    if actions != 1:
        parser.error("choose exactly one repair action")
    if args.validate_targets:
        validate_targets()
    elif args.stage:
        stage_repair()
    elif args.validate_and_promote:
        validate_and_promote()
    else:
        verify_final()


if __name__ == "__main__":
    main()
