"""Fail-closed Slurm runner for the confirmatory prior-sensitivity study."""

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

import build_prior_misspecification_manifest as frozen_design
import prior_misspecification_benchmark as pilot


CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
DEFAULT_MANIFEST = (
    PROJECT_DIR / "config" / "prior_misspecification_confirmatory_manifest.csv"
)
DEFAULT_RESULTS_ROOT = PROJECT_DIR / "results"
FROZEN_DGP_GATE = PROJECT_DIR / "frozen_reference" / "dgp_reproducibility_release_gate.json"
FROZEN_PILOT_GATE = (
    PROJECT_DIR / "frozen_reference" / "phase6_pilot_release_metadata.json"
)
EXPECTED_SCENARIOS = 8
EXPECTED_RATIOS = 5
EXPECTED_CONDITIONS = 40
EXPECTED_REPETITIONS = 100
EXPECTED_METHODS = 27
REPAIRED_RANKING_IDS = {"TM003", *{f"TM{i:03d}" for i in range(11, 16)}}
NAIVE_CORRECTED_PAIRS = list(zip(
    [f"TM{i:03d}" for i in range(6, 11)],
    [f"TM{i:03d}" for i in range(11, 16)],
))
PAIR_METRICS = ["AUROC", "AUPRC", "F1", "BalancedAccuracy", "MCC"]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _same_value(left, right):
    if pd.isna(left) and pd.isna(right):
        return True
    if pd.isna(left) or pd.isna(right):
        return False
    try:
        return bool(np.isclose(float(left), float(right), rtol=1e-12, atol=1e-12))
    except (ValueError, TypeError):
        return str(left) == str(right)


def load_validated_manifest(path=DEFAULT_MANIFEST):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = pd.read_csv(path)
    expected = frozen_design.build()
    issues = []
    if len(observed) != EXPECTED_CONDITIONS:
        issues.append(f"expected {EXPECTED_CONDITIONS} rows, found {len(observed)}")
    missing = sorted(set(expected.columns).difference(observed.columns))
    if missing:
        issues.append("missing columns: " + ", ".join(missing))
    if not missing and len(observed) == len(expected):
        for column in expected.columns:
            for row_number, (left, right) in enumerate(
                zip(observed[column], expected[column]), start=1
            ):
                if not _same_value(left, right):
                    issues.append(
                        f"row {row_number} column {column}: "
                        f"observed={left!r}, expected={right!r}"
                    )
    expected_ids = [
        f"CPM{scenario:03d}_R{ratio:03d}"
        for scenario in range(1, EXPECTED_SCENARIOS + 1)
        for ratio in (80, 90, 100, 110, 120)
    ]
    if observed.get("condition_id", pd.Series(dtype=str)).astype(str).tolist() != expected_ids:
        issues.append("condition IDs or ordering differ from the frozen design")
    if observed.get("scenario_id", pd.Series(dtype=str)).nunique() != EXPECTED_SCENARIOS:
        issues.append("expected eight base scenarios")
    if observed.groupby("scenario_id")["prior_ratio"].nunique().ne(EXPECTED_RATIOS).any():
        issues.append("each base scenario must contain five prior ratios")
    if not observed.get("n_repetitions", pd.Series(dtype=int)).eq(EXPECTED_REPETITIONS).all():
        issues.append("every condition must have 100 repetitions")
    if not observed.get("n_methods", pd.Series(dtype=int)).eq(EXPECTED_METHODS).all():
        issues.append("every condition must have 27 methods")
    if not np.allclose(
        observed["pi_supplied"],
        observed["pi_true"] * observed["prior_ratio"],
        rtol=0.0,
        atol=1e-15,
    ):
        issues.append("pi_supplied is not pi_true times prior_ratio")
    if not observed["pi_supplied"].between(0.0, 1.0, inclusive="neither").all():
        issues.append("one or more supplied priors are invalid")
    seed_counts = observed.groupby("scenario_id")["scenario_base_seed"].nunique()
    if not seed_counts.eq(1).all():
        issues.append("scenario seeds differ across prior ratios")
    if pilot.benchmark.DGP_IMPLEMENTATION != "deterministic_toeplitz_ar1_v1":
        issues.append("benchmark does not use the released deterministic DGP")
    if not FROZEN_PILOT_GATE.is_file():
        issues.append(f"missing frozen Phase 6 pilot gate: {FROZEN_PILOT_GATE}")
    else:
        try:
            pilot_release = json.loads(FROZEN_PILOT_GATE.read_text(encoding="utf-8"))
            if pilot_release.get("integrity_pass") is not True:
                issues.append("the Phase 6 pilot did not pass its integrity gate")
            if pilot_release.get("observed_tasks") != 400:
                issues.append("the frozen Phase 6 pilot does not contain 400 tasks")
            if pilot_release.get("observed_method_rows") != 10800:
                issues.append("the frozen Phase 6 pilot does not contain 10,800 rows")
        except Exception as error:
            issues.append(f"unreadable frozen Phase 6 pilot gate: {error!r}")
    if not FROZEN_DGP_GATE.is_file():
        issues.append(f"missing frozen DGP release gate: {FROZEN_DGP_GATE}")
    else:
        try:
            release = json.loads(FROZEN_DGP_GATE.read_text(encoding="utf-8"))
            if release.get("release_pass") is not True:
                issues.append("frozen DGP release gate did not pass")
            if release.get("exact_cross_node_fingerprint_pass") is not True:
                issues.append("frozen DGP gate lacks exact cross-node fingerprints")
            if release.get("dgp_benchmark_sha256") != sha256_file(
                CODE_DIR / "two_sample_benchmark.py"
            ):
                issues.append("frozen DGP benchmark hash does not match current code")
        except Exception as error:
            issues.append(f"unreadable frozen DGP gate: {error!r}")
    if issues:
        raise ValueError("Invalid confirmatory manifest:\n- " + "\n- ".join(issues[:50]))
    return observed


def condition_row(manifest, condition_index):
    match = manifest.loc[
        pd.to_numeric(manifest["condition_index"]) == int(condition_index)
    ]
    if len(match) != 1:
        raise ValueError(f"condition index {condition_index} not found exactly once")
    return match.iloc[0]


def condition_directory(results_root, row):
    return (
        Path(results_root)
        / "scenarios"
        / str(row["scenario_slug"])
        / "prior_conditions"
        / str(row["condition_slug"])
    )


def prepare_results_tree(manifest, manifest_path, results_root):
    manifest_hash = sha256_file(manifest_path)
    for scenario_id, scenario_rows in manifest.groupby("scenario_id", sort=False):
        first = scenario_rows.iloc[0]
        scenario_root = Path(results_root) / "scenarios" / str(first["scenario_slug"])
        scenario_root.mkdir(parents=True, exist_ok=True)
        pilot.benchmark.atomic_to_csv(
            scenario_rows,
            scenario_root / "scenario_prior_grid.csv",
        )
        (scenario_root / "README.md").write_text(
            f"# {scenario_id} — {first['scenario_label']}\n\n"
            f"Source Phase 5 scenario: `{first['source_scenario_id']}`  \n"
            f"True prevalence: `{first['pi_true']}`  \n"
            "Five condition folders change only the prior supplied to methods. "
            "The generated P/U/validation/test data must be identical across "
            "conditions within a repetition.\n",
            encoding="utf-8",
        )
        for _, row in scenario_rows.iterrows():
            root = condition_directory(results_root, row)
            for child in ["repetitions", "aggregate", "run_metadata"]:
                (root / child).mkdir(parents=True, exist_ok=True)
            definition = pd.DataFrame([row.to_dict()])
            pilot.benchmark.atomic_to_csv(definition, root / "condition_definition.csv")
            atomic_json(
                {"manifest_sha256": manifest_hash, "condition": row.to_dict()},
                root / "condition_definition.json",
            )
            (root / "README.md").write_text(
                f"# {row['condition_id']}\n\n"
                f"True pi: `{row['pi_true']}`  \n"
                f"Supplied pi: `{row['pi_supplied']}`  \n"
                f"Prior ratio: `{row['prior_ratio']}`  \n"
                f"Repetitions: `{EXPECTED_REPETITIONS}`  \n"
                "All 27 methods run in every repetition.\n",
                encoding="utf-8",
            )
    registry = manifest.copy()
    registry.insert(0, "manifest_sha256", manifest_hash)
    pilot.benchmark.atomic_to_csv(
        registry,
        Path(results_root) / "prior_misspecification_condition_registry.csv",
    )


def task_to_condition_rep(task_id, manifest):
    total = len(manifest) * EXPECTED_REPETITIONS
    if not 1 <= int(task_id) <= total:
        raise ValueError(f"task_id must be in 1..{total}")
    condition_index = (int(task_id) - 1) // EXPECTED_REPETITIONS + 1
    rep = (int(task_id) - 1) % EXPECTED_REPETITIONS + 1
    return condition_index, rep


def repetition_paths(root, condition_id, rep):
    stem = f"{condition_id}_rep_{rep:03d}"
    return {
        "results": root / "repetitions" / f"{stem}_results.csv",
        "diagnostics": root / "repetitions" / f"{stem}_diagnostics.csv",
        "failure": root / "repetitions" / f"{stem}_failure.csv",
        "metadata": root / "run_metadata" / f"{stem}_metadata.json",
    }


def _tag_frame(frame, row, manifest_hash):
    tagged = frame.copy()
    columns = [
        "condition_index", "condition_id", "condition_slug", "scenario_index",
        "scenario_id", "scenario_slug", "source_scenario_id", "scenario_family",
        "scenario_label", "study_phase", "study_role", "pilot_source_phase",
        "frozen_method_source_phase", "dgp_reproducibility_source_phase",
        "cross_ratio_shared_data_required",
    ]
    for column in reversed(columns):
        if column in tagged.columns:
            tagged = tagged.drop(columns=[column])
        tagged.insert(0, column, row[column])
    tagged["manifest_sha256"] = manifest_hash
    return tagged


def repaired_method_issues(raw):
    issues = []
    repaired_rows = raw.loc[raw["method_id"].isin(REPAIRED_RANKING_IDS)]
    if len(repaired_rows) != len(REPAIRED_RANKING_IDS):
        issues.append("repaired probability-method rows are incomplete")
    elif not repaired_rows["ranking_score_type"].eq(
        "unclipped_log_case_control_density_ratio"
    ).all():
        issues.append("TM003/TM011--TM015 did not use unclipped ranking scores")
    for naive_id, corrected_id in NAIVE_CORRECTED_PAIRS:
        naive = raw.loc[raw["method_id"].eq(naive_id)]
        corrected = raw.loc[raw["method_id"].eq(corrected_id)]
        if len(naive) != 1 or len(corrected) != 1:
            issues.append(f"{naive_id}/{corrected_id} pair is incomplete")
            continue
        naive, corrected = naive.iloc[0], corrected.iloc[0]
        for metric in PAIR_METRICS:
            if not np.isclose(
                float(naive[metric]), float(corrected[metric]), rtol=0.0, atol=1e-12
            ):
                issues.append(f"{naive_id}/{corrected_id} mismatch for {metric}")
        if naive["best_params"] != corrected["best_params"]:
            issues.append(f"{naive_id}/{corrected_id} selected different parameters")
    tm027 = raw.loc[raw["method_id"].eq("TM027")]
    if len(tm027) != 1:
        issues.append("TM027 row is missing")
    else:
        tm027 = tm027.iloc[0]
        if tm027.get("pu_gb_objective") != "first_order_canonical_two_sample_nnpu":
            issues.append("TM027 objective tag is incorrect")
        if tm027.get("pu_gb_optimizer") != "functional_gradient_equal_projection_weights":
            issues.append("TM027 optimizer tag is incorrect")
        if not float(tm027.get("minimum_positive_pseudo_residual", np.nan)) > 0.0:
            issues.append("TM027 positive examples had no first-order influence")
    return issues


def _direct_prior_routing_issues(raw, pi_supplied):
    routing = {
        "TM003": "known_population_prior",
        "TM004": "pi_u",
        "TM005": "pi_u",
        "TM026": "prior_given_to_AdaPU",
        "TM027": "pi_u",
    }
    issues = []
    for method_id, column in routing.items():
        match = raw.loc[raw["method_id"].eq(method_id), column]
        if len(match) != 1 or not np.isclose(
            float(match.iloc[0]), float(pi_supplied), rtol=0.0, atol=1e-12
        ):
            issues.append(f"{method_id} did not receive pi_supplied through {column}")
    return issues


def run_one_task(manifest, manifest_path, results_root, condition_index, rep, overwrite=False):
    row = condition_row(manifest, condition_index)
    root = condition_directory(results_root, row)
    paths = repetition_paths(root, str(row["condition_id"]), rep)
    if paths["results"].is_file() and not overwrite:
        print(f"Already complete: {paths['results']}")
        return paths["results"]
    if not (root / "condition_definition.csv").is_file():
        raise RuntimeError("Prepare result folders before running the array")
    manifest_hash = sha256_file(manifest_path)
    seed = int(row["scenario_base_seed"]) + int(rep) - 1
    started = datetime.now(timezone.utc).isoformat()
    try:
        raw, diagnostics, _ = pilot.run_condition(
            rep=rep,
            seed=seed,
            pi_true=float(row["pi_true"]),
            pi_supplied=float(row["pi_supplied"]),
            prior_ratio=float(row["prior_ratio"]),
            structural_type=str(row["structural_type"]),
            transform_strength=float(row["transform_strength"]),
            main_effect_shift=float(row["main_effect_shift"]),
            positive_variance_ratio=float(row["positive_variance_ratio"]),
            n_p_train=int(row["n_p_train"]),
            n_u_train=int(row["n_u_train"]),
            n_p_val=int(row["n_p_val"]),
            n_u_val=int(row["n_u_val"]),
            n_test=int(row["n_test"]),
            p=int(row["p"]),
            delta=float(row["delta"]),
            n_signal=int(row["n_signal"]),
            rho=float(row["rho"]),
        )
        raw = _tag_frame(raw, row, manifest_hash)
        diagnostics = _tag_frame(diagnostics, row, manifest_hash)
        checks = []
        if len(raw) != EXPECTED_METHODS or raw["method_id"].nunique() != EXPECTED_METHODS:
            checks.append("expected exactly 27 method rows")
        if set(raw["method_id"].astype(str)) != {
            f"TM{i:03d}" for i in range(1, EXPECTED_METHODS + 1)
        }:
            checks.append("method catalog is not TM001--TM027")
        if raw["data_fingerprint"].nunique() != 1:
            checks.append("methods did not share one exact dataset")
        if pd.to_numeric(raw["AUROC"], errors="coerce").notna().sum() != EXPECTED_METHODS:
            checks.append("one or more methods has missing AUROC")
        if set(pd.to_numeric(raw["seed"], errors="coerce")) != {seed}:
            checks.append("seed does not match the frozen manifest")
        if not np.allclose(raw["pi_true"], float(row["pi_true"])):
            checks.append("pi_true changed during fitting")
        if not np.allclose(raw["pi_supplied"], float(row["pi_supplied"])):
            checks.append("pi_supplied was not recorded correctly")
        if not raw["dgp_implementation"].eq("deterministic_toeplitz_ar1_v1").all():
            checks.append("wrong deterministic DGP tag")
        checks.extend(repaired_method_issues(raw))
        checks.extend(_direct_prior_routing_issues(raw, float(row["pi_supplied"])))
        if checks:
            raise RuntimeError("; ".join(checks))
    except Exception as error:
        pilot.benchmark.atomic_to_csv(
            pd.DataFrame([{
                "condition_id": row["condition_id"],
                "scenario_id": row["scenario_id"],
                "rep": rep,
                "seed": seed,
                "error": repr(error),
            }]),
            paths["failure"],
        )
        raise
    pilot.benchmark.atomic_to_csv(raw, paths["results"])
    pilot.benchmark.atomic_to_csv(diagnostics, paths["diagnostics"])
    atomic_json({
        "phase": "phase7_two_sample_prior_misspecification_confirmatory_100reps",
        "study_role": "class_prior_misspecification_confirmatory",
        "condition_id": str(row["condition_id"]),
        "condition_index": int(row["condition_index"]),
        "scenario_id": str(row["scenario_id"]),
        "source_scenario_id": str(row["source_scenario_id"]),
        "rep": int(rep),
        "seed": int(seed),
        "pi_true": float(row["pi_true"]),
        "pi_supplied": float(row["pi_supplied"]),
        "prior_ratio": float(row["prior_ratio"]),
        "data_fingerprint": str(raw["data_fingerprint"].iloc[0]),
        "method_count": EXPECTED_METHODS,
        "cross_ratio_shared_data_required": True,
        "manifest_sha256": manifest_hash,
        "dgp_benchmark_sha256": sha256_file(CODE_DIR / "two_sample_benchmark.py"),
        "phase6_pilot_gate_sha256": sha256_file(FROZEN_PILOT_GATE),
        "prior_wrapper_sha256": sha256_file(CODE_DIR / "prior_misspecification_benchmark.py"),
        "runner_sha256": sha256_file(Path(__file__)),
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }, paths["metadata"])
    print(
        f"Saved {row['condition_id']} rep {rep}; seed={seed}; "
        f"fingerprint={raw['data_fingerprint'].iloc[0]}"
    )
    return paths["results"]


def aggregate_condition(manifest, results_root, condition_index, allow_incomplete=False):
    row = condition_row(manifest, condition_index)
    condition_id = str(row["condition_id"])
    root = condition_directory(results_root, row)
    results, diagnostics, audit = [], [], []
    for rep in range(1, EXPECTED_REPETITIONS + 1):
        paths = repetition_paths(root, condition_id, rep)
        issues = []
        frame = None
        if not paths["results"].is_file():
            issues.append("missing results")
        else:
            try:
                frame = pd.read_csv(paths["results"])
                if len(frame) != EXPECTED_METHODS or frame["method_id"].nunique() != EXPECTED_METHODS:
                    issues.append("not 27 unique method rows")
                if frame["data_fingerprint"].nunique() != 1:
                    issues.append("not one shared fingerprint")
                if pd.to_numeric(frame["AUROC"], errors="coerce").notna().sum() != EXPECTED_METHODS:
                    issues.append("missing AUROC")
            except Exception as error:
                issues.append(f"unreadable results: {error!r}")
        if not paths["diagnostics"].is_file():
            issues.append("missing diagnostics")
        if not paths["metadata"].is_file():
            issues.append("missing metadata")
        if not issues and frame is not None:
            results.append(frame)
            diagnostics.append(pd.read_csv(paths["diagnostics"]))
        audit.append({
            "condition_index": int(condition_index),
            "condition_id": condition_id,
            "scenario_id": str(row["scenario_id"]),
            "rep": rep,
            "expected_seed": int(row["scenario_base_seed"]) + rep - 1,
            "complete": not issues,
            "issues": "; ".join(issues),
        })
    report = pd.DataFrame(audit)
    aggregate = root / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    pilot.benchmark.atomic_to_csv(
        report,
        aggregate / f"{condition_id}_collection_report.csv",
    )
    n_complete = int(report["complete"].sum())
    if n_complete != EXPECTED_REPETITIONS and not allow_incomplete:
        raise RuntimeError(
            f"{condition_id}: only {n_complete}/{EXPECTED_REPETITIONS} repetitions complete"
        )
    if not results:
        raise RuntimeError(f"{condition_id}: no complete repetitions")
    raw = pd.concat(results, ignore_index=True)
    ranked, summary = pilot.benchmark.rank_and_summarize(raw)
    diagnostics_frame = pd.concat(diagnostics, ignore_index=True)
    pilot.benchmark.atomic_to_csv(
        ranked,
        aggregate / f"{condition_id}_raw_{n_complete}reps.csv",
    )
    pilot.benchmark.atomic_to_csv(
        summary,
        aggregate / f"{condition_id}_summary_{n_complete}reps.csv",
    )
    pilot.benchmark.atomic_to_csv(
        diagnostics_frame,
        aggregate / f"{condition_id}_diagnostics_{n_complete}reps.csv",
    )
    return {
        "condition_index": int(condition_index),
        "condition_id": condition_id,
        "scenario_id": str(row["scenario_id"]),
        "prior_ratio": float(row["prior_ratio"]),
        "complete_repetitions": n_complete,
        "expected_repetitions": EXPECTED_REPETITIONS,
        "complete": n_complete == EXPECTED_REPETITIONS,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--validate-manifest", action="store_true")
    parser.add_argument("--prepare-results", action="store_true")
    parser.add_argument("--check-dependencies", action="store_true")
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--condition-index", type=int)
    parser.add_argument("--rep", type=int)
    parser.add_argument("--aggregate-condition", type=int)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    results_root = Path(args.results_root).resolve()
    manifest = load_validated_manifest(manifest_path)
    if args.validate_manifest:
        print(
            "VALID: 8 scenarios x 5 supplied-prior ratios x 100 repetitions "
            "= 4000 tasks"
        )
    if args.prepare_results:
        prepare_results_tree(manifest, manifest_path, results_root)
        print(f"Prepared traceable result tree at {results_root}")
    if args.check_dependencies:
        if not pilot.benchmark.dependency_status():
            raise SystemExit(1)
    if args.task_id is not None:
        condition_index, rep = task_to_condition_rep(args.task_id, manifest)
        run_one_task(
            manifest, manifest_path, results_root, condition_index, rep, args.overwrite
        )
    elif args.condition_index is not None or args.rep is not None:
        if args.condition_index is None or args.rep is None:
            parser.error("--condition-index and --rep must be supplied together")
        run_one_task(
            manifest, manifest_path, results_root,
            args.condition_index, args.rep, args.overwrite,
        )
    if args.aggregate_condition is not None:
        aggregate_condition(
            manifest, results_root, args.aggregate_condition, args.allow_incomplete
        )


if __name__ == "__main__":
    main()
