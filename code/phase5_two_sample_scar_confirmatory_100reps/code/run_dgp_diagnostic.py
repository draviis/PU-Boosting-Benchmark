"""Run deterministic two-sample DGP fingerprints for four frozen cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import two_sample_benchmark as benchmark


PROJECT_DIR = Path(__file__).resolve().parent.parent
CASES = [
    {"case_id": "TS027_rep005", "scenario_id": "TS027", "rho": 0.30, "seed": 410027004},
    {"case_id": "TS027_rep008", "scenario_id": "TS027", "rho": 0.30, "seed": 410027007},
    {"case_id": "TS028_rep001", "scenario_id": "TS028", "rho": 0.60, "seed": 410028000},
    {"case_id": "TS028_rep010", "scenario_id": "TS028", "rho": 0.60, "seed": 410028009},
]


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
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def run_case(specification):
    arguments = dict(
        n_p_train=300,
        n_u_train=4700,
        n_p_val=300,
        n_u_val=4700,
        n_test=10000,
        p=10,
        pi=0.20,
        delta=0.90,
        n_signal=3,
        rho=specification["rho"],
        seed=specification["seed"],
        structural_type="linear_gaussian",
    )
    first = benchmark.generate_two_sample_scar(**arguments)
    first_fingerprint = benchmark.experiment_fingerprint(first)
    second = benchmark.generate_two_sample_scar(**arguments)
    second_fingerprint = benchmark.experiment_fingerprint(second)

    calibration_rng = np.random.default_rng(99117 + int(specification["rho"] * 100))
    calibration = benchmark._deterministic_toeplitz_standard_normal(
        100000, 10, specification["rho"], calibration_rng
    )
    observed_covariance = np.cov(calibration, rowvar=False, ddof=0)
    target_covariance = benchmark.base.toeplitz_covariance(10, specification["rho"])
    covariance_max_abs_error = float(
        np.max(np.abs(observed_covariance - target_covariance))
    )
    return {
        **specification,
        "fingerprint_first": first_fingerprint,
        "fingerprint_second": second_fingerprint,
        "internal_repeat_exact": first_fingerprint == second_fingerprint,
        "delta_sq": float(first.delta_sq),
        "target_delta_sq": 0.90**2,
        "covariance_max_abs_error": covariance_max_abs_error,
        "covariance_tolerance": 0.015,
        "covariance_pass": covariance_max_abs_error <= 0.015,
        "n_p_train": int(first.n_p_train),
        "n_u_train": int(first.n_u_train),
        "n_p_val": int(first.n_p_val),
        "n_u_val": int(first.n_u_val),
        "hidden_positive_train": int(first.train.y_true[first.train.s == 0].sum()),
        "hidden_positive_val": int(first.val.y_true[first.val.s == 0].sum()),
        "test_positive": int(first.test.y_true.sum()),
        "dgp_implementation": first.dgp_implementation,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thread-mode", choices=["single", "eight"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected_threads = 1 if args.thread_mode == "single" else 8
    configured_threads = {
        name: os.environ.get(name)
        for name in [
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "PU_BENCHMARK_N_JOBS",
        ]
    }
    payload = {
        "diagnostic": "phase4c_deterministic_correlated_dgp_cross_node",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "requested_node": os.environ.get("DIAGNOSTIC_NODE"),
        "thread_mode": args.thread_mode,
        "expected_threads": expected_threads,
        "configured_threads": configured_threads,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "benchmark_sha256": sha256_file(Path(benchmark.__file__)),
        "runner_sha256": sha256_file(Path(__file__)),
        "cases": [run_case(specification) for specification in CASES],
    }
    atomic_json(payload, args.output)
    print(f"Saved {args.thread_mode} diagnostic on {payload['hostname']} to {args.output}")


if __name__ == "__main__":
    main()
