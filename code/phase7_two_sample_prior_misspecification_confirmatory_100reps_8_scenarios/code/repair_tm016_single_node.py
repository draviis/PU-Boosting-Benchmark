"""Validate, merge, and verify the targeted TM016 single-node repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
STUDY_MANIFEST = (
    PROJECT_DIR / "config" / "prior_misspecification_confirmatory_manifest.csv"
)
MAIN_RESULTS = PROJECT_DIR / "results"
REPAIR_ROOT = PROJECT_DIR / "repair" / "tm016_single_node_3pairs"
TASK_MANIFEST = REPAIR_ROOT / "repair_task_manifest.csv"
STAGED_RESULTS = REPAIR_ROOT / "staged_results"
BACKUP_ROOT = REPAIR_ROOT / "original_backup"
REPAIR_AUDIT = REPAIR_ROOT / "tm016_repair_audit.csv"
REPAIR_METADATA = REPAIR_ROOT / "tm016_repair_metadata.json"
FINAL_ANALYSIS_METADATA = (
    MAIN_RESULTS
    / "prior_misspecification_analysis"
    / "prior_misspecification_confirmatory_metadata.json"
)
TOLERANCE = 5e-5
EXPECTED_METHODS = 27
EXPECTED_TASKS = 15
EXPECTED_CONDITIONS = 40
EXPECTED_REPETITIONS = 100


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


def load_task_manifest() -> pd.DataFrame:
    tasks = pd.read_csv(TASK_MANIFEST)
    required = {"task_id", "condition_id", "scenario_id", "rep", "prior_ratio"}
    if not required.issubset(tasks.columns):
        raise RuntimeError(f"repair manifest is missing {sorted(required - set(tasks.columns))}")
    if len(tasks) != EXPECTED_TASKS or tasks["task_id"].nunique() != EXPECTED_TASKS:
        raise RuntimeError("repair manifest must contain 15 unique task IDs")
    group_sizes = tasks.groupby(["scenario_id", "rep"]).size()
    if len(group_sizes) != 3 or not group_sizes.eq(5).all():
        raise RuntimeError("repair manifest must contain three complete five-ratio groups")
    expected_ratios = {0.8, 0.9, 1.0, 1.1, 1.2}
    for _, group in tasks.groupby(["scenario_id", "rep"]):
        if set(group["prior_ratio"].astype(float)) != expected_ratios:
            raise RuntimeError("each repair group must contain all five prior ratios")
    return tasks.sort_values("task_id").reset_index(drop=True)


def load_study_manifest() -> pd.DataFrame:
    manifest = pd.read_csv(STUDY_MANIFEST)
    if len(manifest) != EXPECTED_CONDITIONS:
        raise RuntimeError("confirmatory manifest must contain 40 conditions")
    if manifest["condition_index"].nunique() != EXPECTED_CONDITIONS:
        raise RuntimeError("confirmatory condition indexes are not unique")
    return manifest


def task_to_condition_rep(task_id: int) -> tuple[int, int]:
    total = EXPECTED_CONDITIONS * EXPECTED_REPETITIONS
    if not 1 <= int(task_id) <= total:
        raise RuntimeError(f"task ID must be in 1..{total}")
    condition_index = (int(task_id) - 1) // EXPECTED_REPETITIONS + 1
    rep = (int(task_id) - 1) % EXPECTED_REPETITIONS + 1
    return condition_index, rep


def condition_row(manifest: pd.DataFrame, condition_index: int) -> pd.Series:
    match = manifest.loc[
        pd.to_numeric(manifest["condition_index"]) == int(condition_index)
    ]
    if len(match) != 1:
        raise RuntimeError(f"condition index {condition_index} not found exactly once")
    return match.iloc[0]


def result_path(
    results_root: Path, row: pd.Series, rep: int
) -> Path:
    condition_root = (
        Path(results_root)
        / "scenarios"
        / str(row["scenario_slug"])
        / "prior_conditions"
        / str(row["condition_slug"])
    )
    stem = f"{row['condition_id']}_rep_{int(rep):03d}"
    return condition_root / "repetitions" / f"{stem}_results.csv"


def resolved_tasks(tasks: pd.DataFrame) -> list[dict]:
    manifest = load_study_manifest()
    resolved = []
    for task in tasks.itertuples(index=False):
        condition_index, rep = task_to_condition_rep(int(task.task_id))
        row = condition_row(manifest, condition_index)
        if str(row["condition_id"]) != str(task.condition_id):
            raise RuntimeError(f"task {task.task_id} condition mapping changed")
        if str(row["scenario_id"]) != str(task.scenario_id) or int(rep) != int(task.rep):
            raise RuntimeError(f"task {task.task_id} scenario/repetition mapping changed")
        if not np.isclose(float(row["prior_ratio"]), float(task.prior_ratio), atol=1e-12):
            raise RuntimeError(f"task {task.task_id} prior-ratio mapping changed")
        resolved.append({
            "task_id": int(task.task_id),
            "condition_index": int(condition_index),
            "condition_id": str(row["condition_id"]),
            "scenario_id": str(row["scenario_id"]),
            "rep": int(rep),
            "prior_ratio": float(row["prior_ratio"]),
            "seed": int(row["scenario_base_seed"]) + int(rep) - 1,
            "staged_path": result_path(STAGED_RESULTS, row, rep),
            "original_path": result_path(MAIN_RESULTS, row, rep),
        })
    return resolved


def validate_complete_result(frame: pd.DataFrame, item: dict, label: str) -> pd.Series:
    if len(frame) != EXPECTED_METHODS or frame["method_id"].nunique() != EXPECTED_METHODS:
        raise RuntimeError(f"{label} {item['task_id']}: expected 27 complete method rows")
    if set(frame["method_id"].astype(str)) != {
        f"TM{i:03d}" for i in range(1, EXPECTED_METHODS + 1)
    }:
        raise RuntimeError(f"{label} {item['task_id']}: method catalog changed")
    if frame["data_fingerprint"].nunique() != 1:
        raise RuntimeError(f"{label} {item['task_id']}: methods did not share one dataset")
    if set(pd.to_numeric(frame["seed"], errors="raise").astype(int)) != {item["seed"]}:
        raise RuntimeError(f"{label} {item['task_id']}: seed changed")
    if set(frame["scenario_id"].astype(str)) != {item["scenario_id"]}:
        raise RuntimeError(f"{label} {item['task_id']}: scenario tag changed")
    if set(pd.to_numeric(frame["rep"], errors="raise").astype(int)) != {item["rep"]}:
        raise RuntimeError(f"{label} {item['task_id']}: repetition tag changed")
    if not np.allclose(
        pd.to_numeric(frame["prior_ratio"], errors="raise"),
        item["prior_ratio"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(f"{label} {item['task_id']}: prior ratio changed")
    tm016 = frame.loc[frame["method_id"].eq("TM016")]
    if len(tm016) != 1:
        raise RuntimeError(f"{label} {item['task_id']}: TM016 row is incomplete")
    row = tm016.iloc[0]
    if not np.isfinite(float(row["AUROC"])) or not np.isfinite(float(row["AUPRC"])):
        raise RuntimeError(f"{label} {item['task_id']}: TM016 ranking metric is not finite")
    return row


def validate_staged(resolved: list[dict]) -> tuple[list[dict], pd.DataFrame]:
    audit_rows = []
    group_rows = []
    for item in resolved:
        if not item["staged_path"].is_file():
            raise RuntimeError(f"missing staged result {item['staged_path']}")
        if not item["original_path"].is_file():
            raise RuntimeError(f"missing original result {item['original_path']}")
        staged = pd.read_csv(item["staged_path"])
        original = pd.read_csv(item["original_path"])
        staged_tm016 = validate_complete_result(staged, item, "staged")
        original_tm016 = validate_complete_result(original, item, "original")
        if str(staged_tm016["data_fingerprint"]) != str(original_tm016["data_fingerprint"]):
            raise RuntimeError(f"task {item['task_id']}: staged data differ from original data")
        record = {
            **item,
            "data_fingerprint": str(staged_tm016["data_fingerprint"]),
            "original_sha256": sha256_file(item["original_path"]),
            "staged_sha256": sha256_file(item["staged_path"]),
            "original_AUROC": float(original_tm016["AUROC"]),
            "staged_AUROC": float(staged_tm016["AUROC"]),
            "original_AUPRC": float(original_tm016["AUPRC"]),
            "staged_AUPRC": float(staged_tm016["AUPRC"]),
            "original_best_params": str(original_tm016["best_params"]),
            "staged_best_params": str(staged_tm016["best_params"]),
        }
        audit_rows.append(record)
        group_rows.append({
            "scenario_id": item["scenario_id"],
            "rep": item["rep"],
            "prior_ratio": item["prior_ratio"],
            "AUROC": float(staged_tm016["AUROC"]),
            "AUPRC": float(staged_tm016["AUPRC"]),
            "best_params": str(staged_tm016["best_params"]),
            "data_fingerprint": str(staged_tm016["data_fingerprint"]),
        })
    groups = pd.DataFrame(group_rows)
    for key, group in groups.groupby(["scenario_id", "rep"]):
        if len(group) != 5 or group["prior_ratio"].nunique() != 5:
            raise RuntimeError(f"staged group {key} is incomplete")
        if group["data_fingerprint"].nunique() != 1:
            raise RuntimeError(f"staged group {key} did not reuse identical data")
        if group["best_params"].nunique() != 1:
            raise RuntimeError(f"staged group {key} selected different TM016 parameters")
        for metric in ["AUROC", "AUPRC"]:
            span = float(group[metric].max() - group[metric].min())
            if span > TOLERANCE:
                raise RuntimeError(
                    f"staged group {key} TM016 {metric} span {span:.12g} "
                    f"exceeds {TOLERANCE:.12g}"
                )
    return audit_rows, groups


def merge_tm016(audit_rows: list[dict]) -> pd.DataFrame:
    prepared = []
    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
    for record in audit_rows:
        original_path = Path(record["original_path"])
        staged_path = Path(record["staged_path"])
        backup_path = BACKUP_ROOT / original_path.name
        if backup_path.exists():
            if sha256_file(backup_path) != record["original_sha256"]:
                raise RuntimeError(f"existing backup hash mismatch for {backup_path.name}")
        else:
            shutil.copy2(original_path, backup_path)
        original = pd.read_csv(original_path)
        staged = pd.read_csv(staged_path)
        replacement = staged.loc[staged["method_id"].eq("TM016")].iloc[0]
        repaired = original.copy()
        target_index = repaired.index[repaired["method_id"].eq("TM016")]
        repaired.loc[target_index, repaired.columns] = replacement.reindex(repaired.columns).values
        for method_id in [f"TM{i:03d}" for i in range(1, EXPECTED_METHODS + 1) if i != 16]:
            before = original.loc[original["method_id"].eq(method_id)].reset_index(drop=True)
            after = repaired.loc[repaired["method_id"].eq(method_id)].reset_index(drop=True)
            if not before.equals(after):
                raise RuntimeError(f"task {record['task_id']}: non-TM016 row {method_id} changed")
        temporary = original_path.with_name(f".{original_path.name}.repair.{os.getpid()}.tmp")
        repaired.to_csv(temporary, index=False, lineterminator="\n")
        prepared.append((record, original_path, backup_path, temporary))
    try:
        for record, original_path, backup_path, temporary in prepared:
            os.replace(temporary, original_path)
            record["backup_path"] = str(backup_path)
            record["repaired_sha256"] = sha256_file(original_path)
            record["merged_utc"] = datetime.now(timezone.utc).isoformat()
            record["repair_slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
            record["repair_host"] = socket.gethostname()
    finally:
        for _, _, _, temporary in prepared:
            if temporary.exists():
                temporary.unlink()
    audit = pd.DataFrame(audit_rows)
    atomic_csv(audit, REPAIR_AUDIT)
    atomic_json({
        "repair_phase": "phase7_tm016_single_node_repair",
        "status": "merged_pending_full_reanalysis",
        "tasks_rerun": EXPECTED_TASKS,
        "scenario_repetition_groups": 3,
        "method_rows_replaced": EXPECTED_TASKS,
        "method_rows_preserved_per_task": EXPECTED_METHODS - 1,
        "ranking_tolerance": TOLERANCE,
        "single_node": True,
        "single_thread": True,
        "repair_slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "repair_host": socket.gethostname(),
        "repair_script_sha256": sha256_file(Path(__file__)),
        "task_manifest_sha256": sha256_file(TASK_MANIFEST),
        "merged_utc": datetime.now(timezone.utc).isoformat(),
    }, REPAIR_METADATA)
    return audit


def verify_final() -> None:
    if not FINAL_ANALYSIS_METADATA.is_file():
        raise RuntimeError("final analysis metadata was not regenerated")
    metadata = json.loads(FINAL_ANALYSIS_METADATA.read_text(encoding="utf-8"))
    required = {
        "integrity_pass": True,
        "expected_tasks": 4000,
        "observed_tasks": 4000,
        "expected_method_rows": 108000,
        "observed_method_rows": 108000,
        "ranking_negative_controls_passing": 36800,
        "ranking_negative_controls_expected": 36800,
    }
    mismatches = {
        key: {"expected": value, "observed": metadata.get(key)}
        for key, value in required.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"final confirmatory gate failed: {mismatches}")
    repair_metadata = json.loads(REPAIR_METADATA.read_text(encoding="utf-8"))
    repair_metadata.update({
        "status": "complete",
        "final_integrity_pass": True,
        "final_analysis_metadata_sha256": sha256_file(FINAL_ANALYSIS_METADATA),
        "verified_utc": datetime.now(timezone.utc).isoformat(),
    })
    atomic_json(repair_metadata, REPAIR_METADATA)
    print("PASS: TM016 repair complete; final confirmatory integrity gate is clean")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-manifest", action="store_true")
    parser.add_argument("--validate-and-merge", action="store_true")
    parser.add_argument("--verify-final", action="store_true")
    args = parser.parse_args()
    tasks = load_task_manifest()
    resolved = resolved_tasks(tasks)
    if args.validate_manifest:
        print("VALID: 15 tasks covering three complete five-ratio TM016 repair groups")
        return
    if args.validate_and_merge:
        audit_rows, groups = validate_staged(resolved)
        for key, group in groups.groupby(["scenario_id", "rep"]):
            print(
                f"PASS staged {key}: AUROC span={group['AUROC'].max()-group['AUROC'].min():.12g}; "
                f"AUPRC span={group['AUPRC'].max()-group['AUPRC'].min():.12g}"
            )
        merge_tm016(audit_rows)
        print("PASS: backed up originals and atomically replaced exactly 15 TM016 rows")
        return
    if args.verify_final:
        verify_final()
        return
    parser.error("choose one repair action")


if __name__ == "__main__":
    main()
