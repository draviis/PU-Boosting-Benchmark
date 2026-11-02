"""Fail-closed collector for the Phase 4c cross-node DGP diagnostic."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_DIR / "results" / "node_runs"
OUTPUT_DIR = PROJECT_DIR / "results" / "analysis"
EXPECTED_NODES = {f"node{index}" for index in range(19, 27)}
EXPECTED_MODES = {"single", "eight"}
EXPECTED_CASES = {"TS027_rep005", "TS027_rep008", "TS028_rep001", "TS028_rep010"}


def atomic_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def atomic_json(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main():
    files = sorted(INPUT_DIR.glob("*.json"))
    records = []
    run_records = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        run_records.append({
            "file": path.name,
            "hostname": payload["hostname"],
            "requested_node": payload["requested_node"],
            "thread_mode": payload["thread_mode"],
            "benchmark_sha256": payload["benchmark_sha256"],
            "runner_sha256": payload["runner_sha256"],
        })
        for case in payload["cases"]:
            records.append({
                "file": path.name,
                "hostname": payload["hostname"],
                "thread_mode": payload["thread_mode"],
                **case,
            })
    runs = pd.DataFrame(run_records)
    frame = pd.DataFrame(records)
    issues = []
    if len(files) != 16:
        issues.append(f"expected 16 node/mode files, found {len(files)}")
    if runs.empty:
        issues.append("no readable diagnostic runs")
    else:
        if set(runs["hostname"]) != EXPECTED_NODES:
            issues.append(f"nodes differ: {sorted(set(runs['hostname']))}")
        if set(runs["thread_mode"]) != EXPECTED_MODES:
            issues.append("single/eight thread modes are incomplete")
        if runs.duplicated(["hostname", "thread_mode"]).any():
            issues.append("duplicate node/thread run")
        if runs["benchmark_sha256"].nunique() != 1 or runs["runner_sha256"].nunique() != 1:
            issues.append("source hashes differ across runs")
    expected_rows = 16 * 4
    if len(frame) != expected_rows:
        issues.append(f"expected {expected_rows} case rows, found {len(frame)}")
    if not frame.empty:
        if set(frame["case_id"]) != EXPECTED_CASES:
            issues.append("case catalog is incomplete")
        if not frame["internal_repeat_exact"].all():
            issues.append("one or more in-process repeat fingerprints differ")
        if not frame["covariance_pass"].all():
            issues.append("empirical covariance calibration failed")
        if not np.allclose(frame["delta_sq"], frame["target_delta_sq"], rtol=0.0, atol=1e-14):
            issues.append("Mahalanobis separation calibration failed")
        if not frame["dgp_implementation"].eq("deterministic_toeplitz_ar1_v1").all():
            issues.append("unexpected DGP implementation tag")
        for case_id, case_frame in frame.groupby("case_id"):
            if case_frame["fingerprint_first"].nunique() != 1:
                issues.append(f"{case_id}: cross-node/thread fingerprints differ")
            for column in ["hidden_positive_train", "hidden_positive_val", "test_positive"]:
                if case_frame[column].nunique() != 1:
                    issues.append(f"{case_id}: {column} differs")

    case_summary = frame.groupby("case_id", as_index=False).agg(
        runs=("file", "size"),
        nodes=("hostname", "nunique"),
        thread_modes=("thread_mode", "nunique"),
        fingerprints=("fingerprint_first", "nunique"),
        max_covariance_error=("covariance_max_abs_error", "max"),
        max_delta_sq_abs_error=(
            "delta_sq", lambda values: float(np.max(np.abs(values - 0.90**2)))
        ),
    ) if not frame.empty else pd.DataFrame()
    release = {
        "diagnostic": "phase4c_deterministic_correlated_dgp_cross_node",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "expected_nodes": sorted(EXPECTED_NODES),
        "observed_nodes": sorted(set(runs["hostname"])) if not runs.empty else [],
        "expected_run_files": 16,
        "observed_run_files": len(files),
        "expected_case_rows": expected_rows,
        "observed_case_rows": len(frame),
        "exact_cross_node_fingerprint_pass": not frame.empty and all(
            group["fingerprint_first"].nunique() == 1
            for _, group in frame.groupby("case_id")
        ),
        "covariance_calibration_pass": not frame.empty and bool(frame["covariance_pass"].all()),
        "separation_calibration_pass": not frame.empty and bool(np.allclose(
            frame["delta_sq"], frame["target_delta_sq"], rtol=0.0, atol=1e-14
        )),
        "release_pass": not issues,
        "issues": issues,
    }
    atomic_csv(runs, OUTPUT_DIR / "diagnostic_run_registry.csv")
    atomic_csv(frame, OUTPUT_DIR / "diagnostic_case_audit.csv")
    atomic_csv(case_summary, OUTPUT_DIR / "diagnostic_case_summary.csv")
    atomic_json(release, OUTPUT_DIR / "dgp_reproducibility_release_gate.json")
    if issues:
        raise RuntimeError("; ".join(issues))
    print("PASS: exact fingerprints across 8 nodes and both thread modes")


if __name__ == "__main__":
    main()
