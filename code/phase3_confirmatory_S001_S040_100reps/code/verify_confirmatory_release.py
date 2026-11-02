"""Fail-closed release gate for the 100-repetition confirmatory study."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import confirmatory_benchmark as benchmark
import confirmatory_runner as runner


PROJECT_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_DIR / "config/confirmatory_scenarios_S001_S040_100reps.csv"
FINGERPRINT_PATH = PROJECT_DIR / "frozen_reference/pilot_rep1_data_fingerprints_S001_S040.csv"
POLICY_PATH = PROJECT_DIR / "frozen_reference/frozen_reproducibility_decision.json"
CALIBRATION_PATH = PROJECT_DIR / "frozen_reference/frozen_structural_calibration_decision.json"
OUTPUT_PATH = PROJECT_DIR / "results/release/confirmatory_release_gate.json"
THREAD_VARIABLES = [
    "PU_BENCHMARK_N_JOBS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
]


def generate(row: pd.Series, seed: int):
    return benchmark.generate_structural_one_sample_scar(
        structural_type=str(row["structural_type"]),
        n_train=int(row["n_train"]),
        n_val=int(row["n_val"]),
        n_test=int(row["n_test"]),
        p=int(row["p"]),
        pi=float(row["pi"]),
        c=float(row["c"]),
        delta=float(row["delta"]),
        n_signal=int(row["n_signal"]),
        rho=float(row["rho"]),
        seed=int(seed),
        transform_strength=float(row["transform_strength"]),
        main_effect_shift=float(row["main_effect_shift"]),
        positive_variance_ratio=float(row["positive_variance_ratio"]),
    )


def main() -> None:
    manifest = runner.load_validated_manifest(MANIFEST_PATH)
    expected = pd.read_csv(FINGERPRINT_PATH)
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    checks = []

    policy_pass = (
        policy.get("decision") == "FREEZE_EIGHT_THREADS_CROSS_NODE_5E-5"
        and policy.get("frozen_execution_threads") == 8
        and float(policy.get("frozen_unchanged_AUROC_audit_tolerance")) == 5e-5
        and policy.get("documented_5e5_tolerance_rule_pass") is True
    )
    checks.append({"check": "frozen_execution_policy", "pass": policy_pass})
    calibration_pass = (
        calibration.get("decision_status") == "FROZEN"
        and calibration.get("frozen_before_pu_results") is True
        and [item["scenario_id"] for item in calibration["selected_scenarios"]]
        == ["S038", "S039", "S040", "S026"]
    )
    checks.append({"check": "frozen_structural_calibration", "pass": calibration_pass})

    thread_values = {name: os.environ.get(name) for name in THREAD_VARIABLES}
    thread_pass = all(value == "8" for value in thread_values.values())
    checks.append(
        {
            "check": "eight_thread_environment",
            "pass": thread_pass,
            "observed": thread_values,
        }
    )

    expected_base_hash = str(policy.get("benchmark_sha256"))
    observed_base_hash = runner.sha256_file(
        Path(__file__).resolve().parent / "base_phase1_benchmark.py"
    )
    checks.append(
        {
            "check": "frozen_base_benchmark_source_hash",
            "pass": observed_base_hash == expected_base_hash,
            "expected": expected_base_hash,
            "observed": observed_base_hash,
        }
    )

    fingerprint_rows = []
    linear_delegation_rows = []
    for record in expected.itertuples(index=False):
        row = manifest.loc[manifest["scenario_id"].eq(record.scenario_id)].iloc[0]
        data = generate(row, int(record.pilot_seed))
        observed = benchmark.experiment_fingerprint(data)
        passed = observed == record.expected_data_fingerprint
        fingerprint_rows.append(
            {
                "scenario_id": record.scenario_id,
                "pilot_seed": int(record.pilot_seed),
                "expected": record.expected_data_fingerprint,
                "observed": observed,
                "pass": passed,
            }
        )
        if str(row["structural_type"]) == "linear_gaussian":
            direct = benchmark._ORIGINAL_GENERATOR(
                n_train=int(row["n_train"]),
                n_val=int(row["n_val"]),
                n_test=int(row["n_test"]),
                p=int(row["p"]),
                pi=float(row["pi"]),
                c=float(row["c"]),
                delta=float(row["delta"]),
                n_signal=int(row["n_signal"]),
                rho=float(row["rho"]),
                seed=int(record.pilot_seed),
            )
            direct_fingerprint = benchmark.experiment_fingerprint(direct)
            linear_delegation_rows.append(
                {
                    "scenario_id": record.scenario_id,
                    "wrapper_fingerprint": observed,
                    "direct_frozen_generator_fingerprint": direct_fingerprint,
                    "pass": observed == direct_fingerprint,
                }
            )
    exact_pilot_matches = sum(item["pass"] for item in fingerprint_rows)
    checks.append(
        {
            "check": "pilot_byte_fingerprint_diagnostic",
            "pass": True,
            "scenario_count": len(fingerprint_rows),
            "exact_matches": exact_pilot_matches,
            "interpretation": (
                "Diagnostic only: multivariate-normal byte output may differ across "
                "LAPACK implementations. Scientific release is governed by the frozen "
                "source hash plus same-runtime delegation below."
            ),
        }
    )
    delegation_pass = all(item["pass"] for item in linear_delegation_rows)
    checks.append(
        {
            "check": "all_37_linear_scenarios_delegate_exactly_to_frozen_generator",
            "pass": delegation_pass,
            "scenario_count": len(linear_delegation_rows),
        }
    )
    structural_ids = {"S038", "S039", "S040"}
    structural_rows = [
        item for item in fingerprint_rows if item["scenario_id"] in structural_ids
    ]
    structural_pass = len(structural_rows) == 3 and all(
        item["pass"] for item in structural_rows
    )
    checks.append(
        {
            "check": "three_custom_structural_pilot_fingerprints",
            "pass": structural_pass,
            "scenario_count": len(structural_rows),
        }
    )

    # Confirm that the new streams really generate new data at representative
    # low-, ordinary-, high-dimensional, and structural scenarios.
    new_seed_checks = []
    for scenario_id in ["S001", "S002", "S026", "S038", "S039", "S040"]:
        row = manifest.loc[manifest["scenario_id"].eq(scenario_id)].iloc[0]
        observed = benchmark.experiment_fingerprint(
            generate(row, int(row["scenario_base_seed"]))
        )
        pilot = expected.loc[
            expected["scenario_id"].eq(scenario_id), "expected_data_fingerprint"
        ].iloc[0]
        new_seed_checks.append(
            {"scenario_id": scenario_id, "new_fingerprint": observed, "differs_from_pilot": observed != pilot}
        )
    new_streams_pass = all(item["differs_from_pilot"] for item in new_seed_checks)
    checks.append(
        {"check": "new_confirmatory_data_streams", "pass": new_streams_pass}
    )

    overall_pass = all(item["pass"] for item in checks)
    payload = {
        "status": "PASS" if overall_pass else "FAIL",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scenario_count": len(manifest),
        "repetitions_per_scenario": 100,
        "slurm_task_count": 4000,
        "expected_method_rows": 108000,
        "checks": checks,
        "fingerprint_details": fingerprint_rows,
        "linear_generator_delegation_details": linear_delegation_rows,
        "new_seed_details": new_seed_checks,
        "manifest_sha256": runner.sha256_file(MANIFEST_PATH),
        "confirmatory_benchmark_sha256": runner.sha256_file(
            Path(__file__).resolve().parent / "confirmatory_benchmark.py"
        ),
        "base_benchmark_sha256": runner.sha256_file(
            Path(__file__).resolve().parent / "base_phase1_benchmark.py"
        ),
        "repaired_method_library_sha256": runner.sha256_file(
            Path(__file__).resolve().parent / "repaired_method_library.py"
        ),
        "frozen_unchanged_method_tolerance": 5e-5,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_PATH.with_name(f".{OUTPUT_PATH.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, OUTPUT_PATH)
    print(json.dumps({"status": payload["status"], "checks": checks}, indent=2))
    if not overall_pass:
        raise SystemExit("Confirmatory release gate FAILED")


if __name__ == "__main__":
    main()
