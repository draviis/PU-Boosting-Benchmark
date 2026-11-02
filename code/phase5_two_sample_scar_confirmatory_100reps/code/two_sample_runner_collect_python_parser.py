"""Traceable Slurm runner/collector for the two-sample confirmatory study."""

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

import build_two_sample_manifest as frozen_design
import two_sample_benchmark as benchmark


CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
DEFAULT_MANIFEST = PROJECT_DIR / "config" / "two_sample_scenarios_TS001_TS040_100reps.csv"
DEFAULT_RESULTS_ROOT = PROJECT_DIR / "results"
FROZEN_DGP_GATE = PROJECT_DIR / "frozen_reference" / "dgp_reproducibility_release_gate.json"
EXPECTED_SCENARIOS = 40
EXPECTED_REPETITIONS = 100
EXPECTED_METHODS = 27
REPAIRED_RANKING_IDS = {"TM003", *{f"TM{i:03d}" for i in range(11, 16)}}
NAIVE_CORRECTED_PAIRS = list(zip(
    [f"TM{i:03d}" for i in range(6, 11)],
    [f"TM{i:03d}" for i in range(11, 16)],
))
RANKING_CLASSIFICATION_METRICS = ["AUROC", "AUPRC", "F1", "BalancedAccuracy", "MCC"]


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
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _same_value(left, right):
    left_blank = pd.isna(left) or (isinstance(left, str) and not left.strip())
    right_blank = pd.isna(right) or (isinstance(right, str) and not right.strip())
    if left_blank and right_blank:
        return True
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
    if len(observed) != EXPECTED_SCENARIOS:
        issues.append(f"expected 40 rows, found {len(observed)}")
    missing = sorted(set(expected.columns).difference(observed.columns))
    if missing:
        issues.append("missing columns: " + ", ".join(missing))
    if not missing and len(observed) == len(expected):
        for column in expected.columns:
            for row_number, (left, right) in enumerate(zip(observed[column], expected[column]), start=1):
                if not _same_value(left, right):
                    issues.append(
                        f"row {row_number} column {column}: observed={left!r}, expected={right!r}"
                    )
    expected_ids = [f"TS{i:03d}" for i in range(1, 41)]
    if observed.get("scenario_id", pd.Series(dtype=str)).astype(str).tolist() != expected_ids:
        issues.append("scenario_id must run from TS001 through TS040")
    if not observed.get("n_repetitions", pd.Series(dtype=int)).eq(EXPECTED_REPETITIONS).all():
        issues.append("every scenario must have 100 repetitions")
    if not observed.get("n_methods", pd.Series(dtype=int)).eq(EXPECTED_METHODS).all():
        issues.append("every scenario must have 27 methods")
    if "sampling_scheme" in observed and not observed["sampling_scheme"].eq("two_sample_case_control_scar").all():
        issues.append("sampling_scheme is not canonical two-sample SCAR")
    if benchmark.DGP_IMPLEMENTATION != "deterministic_toeplitz_ar1_v1":
        issues.append("benchmark does not use the released deterministic DGP")
    if not FROZEN_DGP_GATE.is_file():
        issues.append(f"missing frozen DGP release gate: {FROZEN_DGP_GATE}")
    else:
        try:
            release = json.loads(FROZEN_DGP_GATE.read_text(encoding="utf-8"))
            if release.get("release_pass") is not True:
                issues.append("frozen DGP release gate did not pass")
            if release.get("exact_cross_node_fingerprint_pass") is not True:
                issues.append("frozen DGP gate lacks exact cross-node fingerprints")
            if release.get("covariance_calibration_pass") is not True:
                issues.append("frozen DGP gate lacks covariance calibration")
            if release.get("separation_calibration_pass") is not True:
                issues.append("frozen DGP gate lacks separation calibration")
            current_benchmark_hash = sha256_file(CODE_DIR / "two_sample_benchmark.py")
            if release.get("dgp_benchmark_sha256") != current_benchmark_hash:
                issues.append("frozen DGP benchmark hash does not match current code")
        except Exception as error:
            issues.append(f"unreadable frozen DGP release gate: {error!r}")
    if issues:
        raise ValueError("Invalid two-sample manifest:\n- " + "\n- ".join(issues[:50]))
    return observed


def scenario_row(manifest, index):
    match = manifest.loc[pd.to_numeric(manifest["scenario_index"]) == int(index)]
    if len(match) != 1:
        raise ValueError(f"scenario index {index} not found exactly once")
    return match.iloc[0]


def scenario_directory(results_root, row):
    return Path(results_root) / "scenarios" / str(row["scenario_slug"])


def prepare_results_tree(manifest, manifest_path, results_root):
    manifest_hash = sha256_file(manifest_path)
    for _, row in manifest.iterrows():
        root = scenario_directory(results_root, row)
        for child in ["repetitions", "aggregate", "run_metadata"]:
            (root / child).mkdir(parents=True, exist_ok=True)
        definition = pd.DataFrame([row.to_dict()])
        benchmark.atomic_to_csv(definition, root / "scenario_definition.csv")
        atomic_json({"manifest_sha256": manifest_hash, "scenario": row.to_dict()}, root / "scenario_definition.json")
        (root / "README.md").write_text(
            f"# {row['scenario_id']} — {row['scenario_label']}\n\n"
            f"Sampling: `{row['sampling_scheme']}`  \n"
            f"P/U train: `{row['n_p_train']}/{row['n_u_train']}`  \n"
            f"P/U validation: `{row['n_p_val']}/{row['n_u_val']}`  \n"
            f"Population prevalence: `{row['pi']}`\n\n{row['interpretation']}\n\n"
            "- `repetitions/`: one 27-method result and diagnostics pair per repetition.\n"
            "- `run_metadata/`: seeds, fingerprints and source hashes.\n"
            "- `aggregate/`: completeness audit and scenario summaries.\n",
            encoding="utf-8",
        )
    registry = manifest.copy()
    registry.insert(0, "manifest_sha256", manifest_hash)
    benchmark.atomic_to_csv(registry, Path(results_root) / "two_sample_scenario_registry.csv")


def task_to_scenario_rep(task_id, manifest):
    total = len(manifest) * EXPECTED_REPETITIONS
    if not 1 <= int(task_id) <= total:
        raise ValueError(f"task_id must be in 1..{total}")
    return (int(task_id) - 1) // EXPECTED_REPETITIONS + 1, (int(task_id) - 1) % EXPECTED_REPETITIONS + 1


def repetition_paths(root, scenario_id, rep):
    stem = f"{scenario_id}_rep_{rep:03d}"
    return {
        "results": root / "repetitions" / f"{stem}_results.csv",
        "diagnostics": root / "repetitions" / f"{stem}_diagnostics.csv",
        "failure": root / "repetitions" / f"{stem}_failure.csv",
        "metadata": root / "run_metadata" / f"{stem}_metadata.json",
    }


def _tag_frame(frame, row, manifest_hash):
    tagged = frame.copy()
    columns = [
        "scenario_index", "scenario_id", "structural_test_id", "scenario_slug",
        "scenario_label", "scenario_family", "varied_factor", "is_reference",
        "reference_scenario_id", "sampling_scheme", "covariance_type", "boundary_type",
        "structural_type", "pilot_phase", "study_role", "frozen_method_source_phase",
        "frozen_method_source_folder", "dgp_reproducibility_source_phase",
        "confirmatory_seed_reuse", "dgp_implementation", "repair_scope",
    ]
    for column in reversed(columns):
        if column in tagged.columns:
            tagged = tagged.drop(columns=[column])
        tagged.insert(0, column, row[column])
    tagged["manifest_sha256"] = manifest_hash
    return tagged


def repaired_method_issues(raw):
    """Return fail-closed invariant violations for the seven repaired methods."""
    issues = []
    repaired_rows = raw.loc[raw["method_id"].isin(REPAIRED_RANKING_IDS)]
    if len(repaired_rows) != len(REPAIRED_RANKING_IDS):
        issues.append("repaired probability-method rows are incomplete")
    elif not repaired_rows["ranking_score_type"].eq(
        "unclipped_log_case_control_density_ratio"
    ).all():
        issues.append("TM003/TM011--TM015 did not use the unclipped ranking score")

    for naive_id, corrected_id in NAIVE_CORRECTED_PAIRS:
        naive_rows = raw.loc[raw["method_id"].eq(naive_id)]
        corrected_rows = raw.loc[raw["method_id"].eq(corrected_id)]
        if len(naive_rows) != 1 or len(corrected_rows) != 1:
            issues.append(f"{naive_id}/{corrected_id} pair is incomplete")
            continue
        naive, corrected = naive_rows.iloc[0], corrected_rows.iloc[0]
        for metric in RANKING_CLASSIFICATION_METRICS:
            if not np.isclose(
                float(naive[metric]), float(corrected[metric]), rtol=0.0, atol=1e-12
            ):
                issues.append(f"{naive_id}/{corrected_id} mismatch for {metric}")
        if naive["best_params"] != corrected["best_params"]:
            issues.append(f"{naive_id}/{corrected_id} selected different parameters")

    tm027_rows = raw.loc[raw["method_id"].eq("TM027")]
    if len(tm027_rows) != 1:
        issues.append("TM027 row is missing")
    else:
        tm027 = tm027_rows.iloc[0]
        if tm027.get("pu_gb_objective") != "first_order_canonical_two_sample_nnpu":
            issues.append("TM027 did not use the explicit first-order two-sample objective")
        if tm027.get("pu_gb_optimizer") != "functional_gradient_equal_projection_weights":
            issues.append("TM027 did not use the repaired optimizer")
        if not float(tm027.get("minimum_positive_pseudo_residual", np.nan)) > 0.0:
            issues.append("TM027 positive examples had no first-order influence")
    return issues


def run_one_repetition(manifest, manifest_path, results_root, index, rep, overwrite=False):
    row = scenario_row(manifest, index)
    root = scenario_directory(results_root, row)
    if not (root / "scenario_definition.csv").is_file():
        raise RuntimeError("Prepare result folders before running the array")
    paths = repetition_paths(root, str(row["scenario_id"]), rep)
    if paths["results"].is_file() and not overwrite:
        print(f"Already complete: {paths['results']}")
        return paths["results"]
    manifest_hash = sha256_file(manifest_path)
    seed = int(row["scenario_base_seed"]) + rep - 1
    started = datetime.now(timezone.utc).isoformat()
    raw, _, diagnostics, failures = benchmark.run_repeated_benchmark(
        structural_type=str(row["structural_type"]),
        transform_strength=float(row["transform_strength"]),
        main_effect_shift=float(row["main_effect_shift"]),
        positive_variance_ratio=float(row["positive_variance_ratio"]),
        n_reps=EXPECTED_REPETITIONS,
        base_seed=int(row["scenario_base_seed"]),
        rep_ids=[rep],
        n_p_train=int(row["n_p_train"]),
        n_u_train=int(row["n_u_train"]),
        n_p_val=int(row["n_p_val"]),
        n_u_val=int(row["n_u_val"]),
        n_test=int(row["n_test"]),
        p=int(row["p"]),
        pi=float(row["pi"]),
        delta=float(row["delta"]),
        n_signal=int(row["n_signal"]),
        rho=float(row["rho"]),
    )
    if raw.empty:
        benchmark.atomic_to_csv(_tag_frame(failures, row, manifest_hash), paths["failure"])
        raise RuntimeError(f"{row['scenario_id']} repetition {rep} produced no results")
    raw = _tag_frame(raw, row, manifest_hash)
    diagnostics = _tag_frame(diagnostics, row, manifest_hash)
    checks = []
    if len(raw) != 27 or raw["method"].nunique() != 27:
        checks.append(f"expected 27 method rows, got {len(raw)}/{raw['method'].nunique()}")
    if set(raw["method_id"].astype(str)) != {f"TM{i:03d}" for i in range(1, 28)}:
        checks.append("two-sample method catalog TM001--TM027 is incomplete")
    if raw["data_fingerprint"].nunique() != 1:
        checks.append("methods did not share exactly one data fingerprint")
    if raw["comparison_role"].eq("supervised_reference").sum() != 2:
        checks.append("expected two oracle reference rows")
    if raw["comparison_role"].eq("pu_method").sum() != 25:
        checks.append("expected 25 PU method rows")
    if pd.to_numeric(raw["AUROC"], errors="coerce").notna().sum() != 27:
        checks.append("one or more methods have missing AUROC")
    if set(pd.to_numeric(raw["seed"], errors="coerce")) != {seed}:
        checks.append("seed does not match manifest")
    if not raw["sampling_scheme"].eq("two_sample_case_control_scar").all():
        checks.append("wrong sampling scheme")
    if not raw["dgp_implementation"].eq("deterministic_toeplitz_ar1_v1").all():
        checks.append("wrong DGP implementation")
    checks.extend(repaired_method_issues(raw))
    if checks:
        benchmark.atomic_to_csv(pd.DataFrame([{"scenario_id": row["scenario_id"], "rep": rep, "issues": "; ".join(checks)}]), paths["failure"])
        raise RuntimeError("; ".join(checks))
    benchmark.atomic_to_csv(raw, paths["results"])
    benchmark.atomic_to_csv(diagnostics, paths["diagnostics"])
    atomic_json({
        "phase": "phase5_two_sample_scar_confirmatory_100reps",
        "study_role": "confirmatory_monte_carlo",
        "frozen_method_source_phase": "phase4b_two_sample_method_repair_pilot_10reps",
        "dgp_reproducibility_source_phase": "phase4c_deterministic_correlated_dgp_cross_node",
        "confirmatory_seed_reuse": False,
        "dgp_implementation": benchmark.DGP_IMPLEMENTATION,
        "repair_scope": sorted(REPAIRED_RANKING_IDS | {"TM027"}),
        "scenario_id": str(row["scenario_id"]),
        "scenario_index": int(row["scenario_index"]),
        "rep": rep,
        "seed": seed,
        "sampling_scheme": "two_sample_case_control_scar",
        "data_fingerprint": str(raw["data_fingerprint"].iloc[0]),
        "method_count": 27,
        "manifest_sha256": manifest_hash,
        "benchmark_sha256": sha256_file(CODE_DIR / "two_sample_benchmark.py"),
        "runner_sha256": sha256_file(Path(__file__)),
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }, paths["metadata"])
    print(f"Saved {row['scenario_id']} rep {rep}; seed={seed}; fingerprint={raw['data_fingerprint'].iloc[0]}")
    return paths["results"]


def aggregate_scenario(manifest, results_root, index, allow_incomplete=False):
    row = scenario_row(manifest, index)
    scenario_id = str(row["scenario_id"])
    root = scenario_directory(results_root, row)
    aggregate = root / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    results, diagnostics, audit = [], [], []
    for rep in range(1, EXPECTED_REPETITIONS + 1):
        paths = repetition_paths(root, scenario_id, rep)
        issues = []
        frame = None
        if not paths["results"].is_file():
            issues.append("missing results")
        else:
            try:
                frame = pd.read_csv(
                    paths["results"],
                    engine="python",
                    dtype=str,
                    keep_default_na=False,
                )
                if len(frame) != 27 or frame["method_id"].nunique() != 27:
                    issues.append("not 27 unique method rows")
                if frame["data_fingerprint"].nunique() != 1:
                    issues.append("not one shared fingerprint")
                if pd.to_numeric(frame["AUROC"], errors="coerce").notna().sum() != 27:
                    issues.append("missing AUROC")
                if set(pd.to_numeric(frame["rep"], errors="coerce")) != {rep}:
                    issues.append("wrong rep")
            except Exception as error:
                issues.append(f"unreadable results: {error!r}")
        if not paths["diagnostics"].is_file():
            issues.append("missing diagnostics")
        if not paths["metadata"].is_file():
            issues.append("missing metadata")
        if not issues and frame is not None:
            results.append(frame)
            diagnostics.append(pd.read_csv(
                paths["diagnostics"],
                engine="python",
                dtype=str,
                keep_default_na=False,
            ))
        audit.append({
            "scenario_index": index,
            "scenario_id": scenario_id,
            "rep": rep,
            "expected_seed": int(row["scenario_base_seed"]) + rep - 1,
            "complete": not issues,
            "issues": "; ".join(issues),
        })
    report = pd.DataFrame(audit)
    benchmark.atomic_to_csv(report, aggregate / f"{scenario_id}_collection_report.csv")
    n_complete = int(report["complete"].sum())
    if n_complete != EXPECTED_REPETITIONS and not allow_incomplete:
        raise RuntimeError(
            f"{scenario_id}: only {n_complete}/{EXPECTED_REPETITIONS} complete"
        )
    if not results:
        raise RuntimeError(f"{scenario_id}: no complete repetitions")
    raw, summary = benchmark.rank_and_summarize(pd.concat(results, ignore_index=True))
    summary.insert(0, "scenario_label", row["scenario_label"])
    summary.insert(0, "scenario_id", scenario_id)
    diagnostics_frame = pd.concat(diagnostics, ignore_index=True)
    benchmark.atomic_to_csv(raw, aggregate / f"{scenario_id}_raw_{n_complete}reps.csv")
    benchmark.atomic_to_csv(summary, aggregate / f"{scenario_id}_summary_{n_complete}reps.csv")
    benchmark.atomic_to_csv(diagnostics_frame, aggregate / f"{scenario_id}_diagnostics_{n_complete}reps.csv")
    return {
        "scenario_index": index,
        "scenario_id": scenario_id,
        "scenario_slug": str(row["scenario_slug"]),
        "complete_repetitions": n_complete,
        "expected_repetitions": EXPECTED_REPETITIONS,
        "complete": n_complete == EXPECTED_REPETITIONS,
    }


def aggregate_all(manifest, results_root, allow_incomplete=False):
    combined = Path(results_root) / "two_sample_combined"
    combined.mkdir(parents=True, exist_ok=True)
    overview, raw_frames, summary_frames, diagnostics_frames = [], [], [], []
    for index in manifest["scenario_index"].astype(int):
        row = scenario_row(manifest, index)
        info = aggregate_scenario(manifest, results_root, index, allow_incomplete)
        overview.append(info)
        aggregate = scenario_directory(results_root, row) / "aggregate"
        n = info["complete_repetitions"]
        raw_frames.append(pd.read_csv(aggregate / f"{row['scenario_id']}_raw_{n}reps.csv"))
        summary_frames.append(pd.read_csv(aggregate / f"{row['scenario_id']}_summary_{n}reps.csv"))
        diagnostics_frames.append(pd.read_csv(aggregate / f"{row['scenario_id']}_diagnostics_{n}reps.csv"))
    benchmark.atomic_to_csv(pd.DataFrame(overview), combined / "two_sample_collection_overview.csv")
    benchmark.atomic_to_csv(pd.concat(raw_frames, ignore_index=True), combined / "two_sample_raw_all_scenarios.csv")
    benchmark.atomic_to_csv(pd.concat(summary_frames, ignore_index=True), combined / "two_sample_method_scenario_summary.csv")
    benchmark.atomic_to_csv(pd.concat(diagnostics_frames, ignore_index=True), combined / "two_sample_diagnostics_all_scenarios.csv")
    print(f"Combined pilot written to {combined}")


def parser():
    out = argparse.ArgumentParser(description=__doc__)
    out.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    out.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    out.add_argument("--rep", type=int)
    out.add_argument("--overwrite", action="store_true")
    out.add_argument("--allow-incomplete", action="store_true")
    mode = out.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-manifest", action="store_true")
    mode.add_argument("--prepare-results", action="store_true")
    mode.add_argument("--list-scenarios", action="store_true")
    mode.add_argument("--check-dependencies", action="store_true")
    mode.add_argument("--task-id", type=int)
    mode.add_argument("--run-scenario", type=int)
    mode.add_argument("--aggregate-scenario", type=int)
    mode.add_argument("--aggregate-all", action="store_true")
    return out


def main():
    args = parser().parse_args()
    manifest = load_validated_manifest(args.manifest)
    if args.validate_manifest:
        print("VALID: 40 scenarios x 100 repetitions = 4000 tasks; 27 methods per task")
    elif args.prepare_results:
        prepare_results_tree(manifest, args.manifest, args.results_root)
        print(f"Prepared {len(manifest)} scenario folders")
    elif args.list_scenarios:
        print(manifest[["scenario_index", "scenario_id", "scenario_label", "pi", "n_p_train", "n_u_train", "p", "delta", "structural_type"]].to_string(index=False))
    elif args.check_dependencies:
        raise SystemExit(0 if benchmark.dependency_status() else 1)
    elif args.task_id is not None:
        index, rep = task_to_scenario_rep(args.task_id, manifest)
        run_one_repetition(manifest, args.manifest, args.results_root, index, rep, args.overwrite)
    elif args.run_scenario is not None:
        if args.rep is None:
            raise SystemExit("--run-scenario requires --rep")
        run_one_repetition(manifest, args.manifest, args.results_root, args.run_scenario, args.rep, args.overwrite)
    elif args.aggregate_scenario is not None:
        print(aggregate_scenario(manifest, args.results_root, args.aggregate_scenario, args.allow_incomplete))
    else:
        aggregate_all(manifest, args.results_root, args.allow_incomplete)


if __name__ == "__main__":
    main()
