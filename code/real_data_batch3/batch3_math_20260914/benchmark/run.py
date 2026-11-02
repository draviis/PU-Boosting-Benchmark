"""Run the existing 27 methods on the selected dataset without editing frozen studies.

Examples:
    python benchmark/run.py --repetitions 1
    python benchmark/run.py --repetitions 20 --output results_20reps
Each sampling design/repetition runs in a fresh process to isolate legacy hooks.
"""

import argparse
import copy
import functools
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
METRICS = ["AUROC", "AUPRC", "F1", "BalancedAccuracy", "MCC", "Brier", "LogLoss"]


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def package_versions():
    """Record distribution versions, including the cluster's CPU-only XGBoost."""
    versions = {}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "xgboost", "lightgbm", "catboost"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            if package != "xgboost":
                raise
            versions["xgboost-cpu"] = importlib.metadata.version("xgboost-cpu")
    return versions


def load_methods(scheme):
    phase = ("phase3_confirmatory_S001_S040_100reps" if scheme == "one_sample"
             else "phase5_two_sample_scar_confirmatory_100reps")
    sys.path.insert(0, str(REPO / phase / "code"))
    module = importlib.import_module("confirmatory_benchmark" if scheme == "one_sample" else "two_sample_benchmark")
    base, repaired = module.base, module.repaired
    if not repaired.ADAPU_AVAILABLE or any(getattr(base, name) is None for name in ("XGBClassifier", "LGBMClassifier", "CatBoostClassifier")):
        raise RuntimeError("All benchmark dependencies, including compiled AdaPU, are required")
    # Suppress auxiliary CatBoost files; the model and tuning parameters are unchanged.
    base.CatBoostClassifier = functools.partial(base.CatBoostClassifier, allow_writing_files=False)
    repaired.CatBoostClassifier = functools.partial(repaired.CatBoostClassifier, allow_writing_files=False)
    if scheme == "one_sample":
        from ranking_repair.correction import fit_corrected
        def cell(data, strategy, engine_name):
            if strategy == "ProbCorrection":
                return fit_corrected(base, data, engine_name)
            return base.fit_one_boosting_cell(data, strategy, engine_name)
    else:
        cell = module.fit_one_boosting_cell
    methods = [
        ("Block0", "Oracle logistic regression", base.fit_oracle_logistic),
        ("Block0", "Oracle CatBoost", base.fit_oracle_catboost),
        ("Block1", "Probability correction", base.fit_elkan_noto_known_c),
        ("Block1", "uPU logistic", functools.partial(base.fit_pu_logistic_baseline, non_negative=False)),
        ("Block1", "nnPU logistic", functools.partial(base.fit_pu_logistic_baseline, non_negative=True, beta=0.0, gamma=1.0)),
    ]
    for strategy in ("Naive", "ProbCorrection", "RN", "CostSensitive"):
        for engine in ("AdaBoost", "GBM", "XGBoost", "LightGBM", "CatBoost"):
            methods.append(("Block2", f"{strategy} + {engine}", functools.partial(cell, strategy=strategy, engine_name=engine)))
    methods.extend([
        ("Block3", "AdaPU", base.fit_adapu_one_sample),
        ("Block3", "PU Gradient Boosting", base.fit_pu_gradient_boosting),
    ])
    return module, methods


def worker(args, directory):
    import numpy as np
    import pandas as pd
    from data import construct_data

    module, methods = load_methods(args.worker)
    base, repaired = module.base, module.repaired
    splits, assignments, metadata = construct_data(args.worker, args.seed, args.c)
    experiment = base.PUExperiment(
        **{name: base.PUSplit(X=s["X"], y_true=s["y"], s=s["s"]) for name, s in splits.items()},
        pi=metadata["population_prior_supplied"], c=args.c if args.worker == "one_sample" else 0.0,
        p=splits["train"]["X"].shape[1], delta=float("nan"), delta_sq=float("nan"), signal=float("nan"),
        n_signal=0, rho=float("nan"), seed=args.seed,
    )
    if args.worker == "two_sample":
        module._ACTIVE_VALIDATION_U_MASK = experiment.val.s == 0
    assignments.to_csv(directory / "assignments_audit_only.csv", index=False)
    np.savez_compressed(directory / "data_audit_only.npz", **{
        f"{name}_{key}": value for name, split in splits.items() for key, value in split.items()
    })
    fingerprint = base.experiment_fingerprint(experiment)
    metadata.update(data_fingerprint=fingerprint, tuning="unchanged frozen full grids",
                    worker_threads=base.N_JOBS, python=sys.version,
                    packages=package_versions())
    metadata["source_sha256"] = {str(Path(m.__file__).relative_to(REPO)): hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                                 for m in (module, base, repaired)}
    metadata["adapu_source_directory"] = repaired.ADAPU_SRC
    if args.worker == "one_sample":
        correction_source = ROOT / "ranking_repair" / "correction.py"
        metadata["correction_revision"] = "ranking_v1"
        metadata["correction_source_sha256"] = hashlib.sha256(correction_source.read_bytes()).hexdigest()
    metadata["adapu_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in Path(repaired.ADAPU_SRC).iterdir() if p.suffix in {".py", ".pyx", ".pyd", ".so"}}
    save_json(directory / "manifest.json", metadata)
    diagnostics = []
    for name, split in splits.items():
        u = split["s"] == 0
        diagnostics.append({"split": name, "n": len(u), "n_P": int((~u).sum()),
                            "n_U": int(u.sum()), "hidden_positive_U": int(split["y"][u].sum()),
                            "empirical_pi_U": float(split["y"][u].mean())})
    pd.DataFrame(diagnostics).to_csv(directory / "data_counts.csv", index=False)
    print(pd.DataFrame(diagnostics).to_string(index=False), flush=True)

    # Capture final test predictions through the shared evaluator, preserving all metrics.
    original_evaluate = base.evaluate_test
    current_prediction_file = None
    def evaluate(method, y_test, score_test, threshold, probability=None, extra=None):
        score_test = np.asarray(score_test, dtype=float)
        result = original_evaluate(method, y_test, score_test, threshold, probability, extra)
        pd.DataFrame({"source_row": splits["test"]["source_row"], "y_true": y_test,
                      "score": score_test, "prediction": (np.asarray(score_test) >= threshold).astype(int),
                      "probability": np.nan if probability is None else base.clip_prob(probability)}).to_csv(current_prediction_file, index=False)
        return result
    base.evaluate_test = repaired.evaluate_test = evaluate
    rows = []
    for number, (block, label, fit) in enumerate(methods, start=1):
        method_id = f"{'M' if args.worker == 'one_sample' else 'TM'}{number:03d}"
        current_prediction_file = directory / "predictions" / f"{method_id}.csv"
        print(f"[{number}/27] {block} | {label}", flush=True)
        # PU methods cannot access hidden training or validation labels.
        model_data = copy.deepcopy(experiment)
        if block != "Block0":
            model_data.train.y_true[:] = -1
            model_data.val.y_true[:] = -1
        start = time.perf_counter()
        try:
            result = fit(model_data)
            if not all(np.isfinite(result.get(metric, np.nan)) for metric in METRICS[:5]):
                raise RuntimeError(f"Missing ranking/classification metrics: {result}")
            result["status"] = "ok"
        except Exception:
            result = {"method": f"{block} | {label}", "status": "failed", "error": traceback.format_exc()}
            print(result["error"], flush=True)
        result.update(method_id=method_id, method_block=block, seed=args.seed, sampling_scheme=args.worker,
                      comparison_role="supervised_reference" if block == "Block0" else "pu_method",
                      fit_and_tune_seconds=time.perf_counter() - start, data_fingerprint=fingerprint)
        if isinstance(result.get("best_params"), dict):
            result["best_params"] = json.dumps(result["best_params"], default=str)
        rows.append(result)
        pd.DataFrame(rows).to_csv(directory / "results.csv", index=False)
        print(f"  {result['status']} in {result['fit_and_tune_seconds']:.1f}s", flush=True)
    if base.experiment_fingerprint(experiment) != fingerprint:
        raise RuntimeError("Shared data changed during fitting")
    if any(r["status"] != "ok" for r in rows):
        raise RuntimeError("Some methods failed; inspect results.csv and worker.log")
    save_json(directory / "complete.json", {"methods": len(rows), "data_fingerprint": fingerprint})


def summarize(output):
    import pandas as pd
    frames = [pd.read_csv(path) for path in sorted(output.glob("*/results.csv"))]
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output / "all_results.csv", index=False)
    good = combined[combined.status == "ok"]
    metrics = [m for m in METRICS + ["fit_and_tune_seconds"] if m in good]
    summary = good.groupby(["sampling_scheme", "method_id", "method", "comparison_role"])[metrics].agg(["count", "mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.reset_index().to_csv(output / "summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--seed", type=int, default=913000)
    parser.add_argument("--c", type=float, default=0.2)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", default="pilot")
    parser.add_argument("--worker", choices=["one_sample", "two_sample"])
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["PU_BENCHMARK_N_JOBS"] = str(args.threads)
    if args.worker:
        (output / "predictions").mkdir(exist_ok=True)
        worker(args, output)
        return
    failures = []
    for repetition in range(args.repetitions):
        for scheme in ("one_sample", "two_sample"):
            seed = args.seed + repetition * 100
            directory = output / f"{scheme}_seed{seed}"
            directory.mkdir(exist_ok=True)
            if (directory / "complete.json").exists():
                manifest = json.loads((directory / "manifest.json").read_text())
                if scheme == "one_sample" and manifest.get("correction_revision") != "ranking_v1":
                    raise ValueError("Existing results predate the ranking repair; use a new --output")
                if manifest["labeling_probability_one_sample"] != args.c or manifest["worker_threads"] != args.threads:
                    raise ValueError("Existing output has different settings; use a new --output")
                continue
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", scheme,
                       "--seed", str(seed), "--c", str(args.c), "--threads", str(args.threads), "--output", str(directory)]
            print(f"Running {scheme}, seed={seed}; log: {directory / 'worker.log'}", flush=True)
            with (directory / "worker.log").open("w", encoding="utf-8") as log:
                status = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=directory).returncode
            if status:
                failures.append(str(directory))
            summarize(output)
    if failures:
        raise SystemExit("Failed runs: " + ", ".join(failures))
    print(f"Completed. Summary: {output / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
