"""Build the immutable S001--S040, 100-repetition confirmatory manifest.

The three pilot manifests in ``frozen_reference/source_manifests`` are the
only design inputs.  S026 is already part of S001--S028 and is annotated as
structural stress test ST04; it is not duplicated as a 41st scenario.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_DIR / "frozen_reference" / "source_manifests"
OUTPUT = PROJECT_DIR / "config" / "confirmatory_scenarios_S001_S040_100reps.csv"
GLOBAL_CONFIRMATORY_SEED = 310_000_000
N_REPETITIONS = 100
N_METHODS = 27


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_or(value, default=""):
    return default if pd.isna(value) else value


def read_source(name: str) -> tuple[pd.DataFrame, Path, str]:
    path = SOURCE_DIR / name
    frame = pd.read_csv(path)
    return frame, path, sha256_file(path)


def normalized_row(
    source_row: pd.Series,
    *,
    scenario_index: int,
    source_name: str,
    source_path: Path,
    source_hash: str,
) -> dict:
    scenario_id = str(source_row["scenario_id"])
    pi = float(source_row["pi"])
    c = float(source_row["c"])
    delta = float(source_row["delta"])
    n_train = int(source_row["n_train"])
    n_val = int(source_row["n_val"])
    n_test = int(source_row["n_test"])
    structural = scenario_id in {"S038", "S039", "S040"}
    structural_test_id = {
        "S026": "ST04",
        "S038": "ST01",
        "S039": "ST02",
        "S040": "ST03",
    }.get(scenario_id, "")
    scenario_seed = GLOBAL_CONFIRMATORY_SEED + scenario_index * 1_000

    return {
        "scenario_index": scenario_index,
        "global_scenario_number": scenario_index,
        "structural_test_id": structural_test_id,
        "scenario_id": scenario_id,
        "scenario_slug": str(source_row["scenario_slug"]),
        "scenario_label": str(source_row["scenario_label"]),
        "scenario_family": str(source_row["scenario_family"]),
        "varied_factor": str(source_row["varied_factor"]),
        "is_reference": str(source_row["is_reference"]),
        "reference_scenario_id": str(source_row["reference_scenario_id"]),
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "p": int(source_row["p"]),
        "pi": pi,
        "c": c,
        "delta": delta,
        "n_signal": int(source_row["n_signal"]),
        "rho": float(source_row["rho"]),
        "covariance_type": str(source_row["covariance_type"]),
        "boundary_type": str(source_row["boundary_type"]),
        "structural_type": (
            str(source_row["structural_type"]) if structural else "linear_gaussian"
        ),
        "transform_strength": float(text_or(source_row.get("transform_strength"), 0.0)),
        "main_effect_shift": float(text_or(source_row.get("main_effect_shift"), 0.0)),
        "positive_variance_ratio": float(
            text_or(source_row.get("positive_variance_ratio"), 1.0)
        ),
        "n_repetitions": N_REPETITIONS,
        "n_methods": N_METHODS,
        "expected_method_rows": N_REPETITIONS * N_METHODS,
        "global_base_seed": GLOBAL_CONFIRMATORY_SEED,
        "scenario_base_seed": scenario_seed,
        "seed_start": scenario_seed,
        "seed_stop": scenario_seed + N_REPETITIONS - 1,
        "seed_scheme": "310000000 + 1000*scenario_index + (rep-1)",
        "expected_labeled_train": n_train * pi * c,
        "expected_labeled_val": n_val * pi * c,
        "expected_positive_test": n_test * pi,
        "pi_u_theoretical": pi * (1.0 - c) / (1.0 - pi * c),
        "target_bayes_AUROC": float(
            text_or(source_row.get("target_bayes_AUROC"), 0.5 * (1.0 + math.erf(delta / 2.0)))
        ),
        "calibration_bayes_AUROC_mean": text_or(
            source_row.get("calibration_bayes_AUROC_mean")
        ),
        "calibration_oracle_logistic_AUROC_mean": text_or(
            source_row.get("calibration_oracle_logistic_AUROC_mean")
        ),
        "calibration_oracle_catboost_AUROC_mean": text_or(
            source_row.get("calibration_oracle_catboost_AUROC_mean")
        ),
        "calibration_catboost_minus_logistic_AUROC_mean": text_or(
            source_row.get("calibration_catboost_minus_logistic_AUROC_mean")
        ),
        "active_features": text_or(source_row.get("active_features")),
        "n_noise": text_or(source_row.get("n_noise")),
        "design_source": text_or(
            source_row.get("design_source"),
            "reused_S026" if scenario_id == "S026" else source_name,
        ),
        "interpretation": str(source_row["interpretation"]),
        "pilot_source": source_name,
        "pilot_source_manifest": source_path.name,
        "pilot_source_manifest_sha256": source_hash,
        "pilot_source_scenario_index": int(source_row["scenario_index"]),
        "pilot_scenario_base_seed": int(source_row["scenario_base_seed"]),
        "pilot_n_repetitions": int(source_row["n_repetitions"]),
        "confirmatory_phase": "phase3_confirmatory_100reps",
    }


def build() -> pd.DataFrame:
    phase1, path1, hash1 = read_source("phase1_scenarios_28.csv")
    phase1b, path1b, hash1b = read_source("phase1b_pi_c_scenarios_9.csv")
    structural, path2, hash2 = read_source("structural_scenarios_4.csv")

    sources = []
    sources.extend((row, "phase1_pilot_28", path1, hash1) for _, row in phase1.iterrows())
    sources.extend((row, "phase1b_pi_c_pilot_9", path1b, hash1b) for _, row in phase1b.iterrows())
    sources.extend(
        (row, "phase2_structural_pilot", path2, hash2)
        for _, row in structural.loc[structural["scenario_id"].isin(["S038", "S039", "S040"])].iterrows()
    )

    rows = [
        normalized_row(
            row,
            scenario_index=index,
            source_name=name,
            source_path=path,
            source_hash=source_hash,
        )
        for index, (row, name, path, source_hash) in enumerate(sources, start=1)
    ]
    manifest = pd.DataFrame(rows)
    expected_ids = [f"S{index:03d}" for index in range(1, 41)]
    observed_ids = manifest["scenario_id"].tolist()
    if observed_ids != expected_ids:
        raise RuntimeError(f"Expected {expected_ids}; found {observed_ids}")
    if manifest["scenario_id"].duplicated().any():
        raise RuntimeError("Scenario IDs are not unique")
    if manifest["scenario_base_seed"].duplicated().any():
        raise RuntimeError("Confirmatory scenario seed starts are not unique")
    if manifest["seed_start"].min() <= 20299838:
        raise RuntimeError("Confirmatory seeds overlap the frozen pilot seed range")
    return manifest


def main() -> None:
    manifest = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUTPUT, index=False, lineterminator="\n")
    print(f"Wrote {len(manifest)} scenarios to {OUTPUT}")
    print(f"Tasks: {len(manifest) * N_REPETITIONS}; method rows: {len(manifest) * N_REPETITIONS * N_METHODS}")


if __name__ == "__main__":
    main()
