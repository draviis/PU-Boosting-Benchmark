"""Build the frozen one-sample SCAR class-prior confirmatory manifest."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
SOURCE = (
    PROJECT_DIR
    / "frozen_reference"
    / "source_confirmatory_scenarios_S001_S040.csv"
)
OUTPUT = PROJECT_DIR / "config" / "prior_misspecification_confirmatory_manifest.csv"
SELECTED_SCENARIOS = ("S001", "S002", "S003", "S011", "S026", "S038", "S039", "S040")
N_REPETITIONS = 100
N_METHODS = 27
GLOBAL_BASE_SEED = 910_000_000
PRIOR_RATIOS = (0.80, 0.90, 1.00, 1.10, 1.20)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_source_rows() -> pd.DataFrame:
    source = pd.read_csv(SOURCE)
    selected = source.loc[source["scenario_id"].isin(SELECTED_SCENARIOS)].copy()
    selected["scenario_id"] = pd.Categorical(
        selected["scenario_id"], categories=SELECTED_SCENARIOS, ordered=True
    )
    selected = selected.sort_values("scenario_id").reset_index(drop=True)
    if selected["scenario_id"].astype(str).tolist() != list(SELECTED_SCENARIOS):
        raise RuntimeError("Frozen source does not contain the eight selected scenarios exactly once")
    return selected


def build() -> pd.DataFrame:
    rows = []
    condition_index = 0
    source_hash = sha256_file(SOURCE)
    for study_scenario_index, source in enumerate(
        _selected_source_rows().itertuples(index=False), start=1
    ):
        prior_true = float(source.pi)
        c_true = float(source.c)
        scenario_id = f"OPM{study_scenario_index:03d}"
        scenario_seed = GLOBAL_BASE_SEED + 1000 * study_scenario_index
        scenario_slug = f"{scenario_id}_{source.scenario_slug}"
        pi_u_true = prior_true * (1.0 - c_true) / (1.0 - prior_true * c_true)
        for prior_index, prior_ratio in enumerate(PRIOR_RATIOS, start=1):
            condition_index += 1
            pi_supplied = prior_true * prior_ratio
            if not 0.0 < pi_supplied < 1.0:
                raise RuntimeError(
                    f"Invalid supplied prior for {source.scenario_id}: {pi_supplied}"
                )
            pi_u_supplied = (
                pi_supplied * (1.0 - c_true) / (1.0 - pi_supplied * c_true)
            )
            ratio_code = f"R{round(100 * prior_ratio):03d}"
            condition_id = f"{scenario_id}_{ratio_code}"
            condition_slug = (
                f"{condition_id}_piTrue{prior_true:.3f}_piHat{pi_supplied:.3f}_"
                f"c{c_true:.3f}"
            ).replace(".", "p")
            rows.append({
                "condition_index": condition_index,
                "condition_id": condition_id,
                "condition_slug": condition_slug,
                "scenario_index": study_scenario_index,
                "scenario_id": scenario_id,
                "scenario_slug": scenario_slug,
                "source_scenario_index": int(source.scenario_index),
                "source_scenario_id": str(source.scenario_id),
                "source_scenario_slug": str(source.scenario_slug),
                "scenario_family": str(source.scenario_family),
                "scenario_label": str(source.scenario_label),
                "sampling_scheme": "one_sample_scar",
                "n_train": int(source.n_train),
                "n_val": int(source.n_val),
                "n_test": int(source.n_test),
                "p": int(source.p),
                "pi_true": prior_true,
                "c_true": c_true,
                "delta": float(source.delta),
                "n_signal": int(source.n_signal),
                "rho": float(source.rho),
                "covariance_type": str(source.covariance_type),
                "boundary_type": str(source.boundary_type),
                "structural_type": str(source.structural_type),
                "transform_strength": float(source.transform_strength),
                "main_effect_shift": float(source.main_effect_shift),
                "positive_variance_ratio": float(source.positive_variance_ratio),
                "prior_index": prior_index,
                "prior_ratio": prior_ratio,
                "relative_prior_error": prior_ratio - 1.0,
                "pi_supplied": pi_supplied,
                "pi_u_true": pi_u_true,
                "pi_u_supplied": pi_u_supplied,
                "prior_is_correct": bool(np.isclose(prior_ratio, 1.0)),
                "c_is_fixed_true": True,
                "n_repetitions": N_REPETITIONS,
                "n_methods": N_METHODS,
                "expected_method_rows": N_REPETITIONS * N_METHODS,
                "global_base_seed": GLOBAL_BASE_SEED,
                "scenario_base_seed": scenario_seed,
                "seed_start": scenario_seed,
                "seed_stop": scenario_seed + N_REPETITIONS - 1,
                "seed_scheme": (
                    "910000000 + 1000*study_scenario_index + (rep-1); "
                    "the same seed is reused across all five supplied-prior ratios"
                ),
                "cross_ratio_shared_data_required": True,
                "study_phase": "phase9_one_sample_prior_misspecification_confirmatory_100reps",
                "study_role": "class_prior_misspecification_confirmatory",
                "pilot_source_phase": "phase8_one_sample_prior_misspecification_pilot",
                "frozen_method_source_phase": "phase3_confirmatory_S001_S040_100reps",
                "reproducibility_source_phase": (
                    "phase2_S026_rep8_M021_reproducibility_diagnostic"
                ),
                "execution_policy": "eight_threads_cross_node_tolerance_5e-5",
                "dgp_implementation": "phase3_frozen_one_sample_scar_v1",
                "source_manifest_sha256": source_hash,
                "interpretation": str(source.interpretation),
            })

    manifest = pd.DataFrame(rows)
    if len(manifest) != 40 or manifest["condition_id"].nunique() != 40:
        raise RuntimeError("Expected 40 unique one-sample prior conditions")
    if manifest.groupby("scenario_id")["prior_ratio"].nunique().ne(5).any():
        raise RuntimeError("Every base scenario must contain five prior ratios")
    if manifest.groupby("scenario_id")["scenario_base_seed"].nunique().ne(1).any():
        raise RuntimeError("Seeds must be identical across prior ratios")
    if not np.allclose(
        manifest["pi_supplied"], manifest["pi_true"] * manifest["prior_ratio"]
    ):
        raise RuntimeError("Supplied population priors are inconsistent")
    expected_pi_u = (
        manifest["pi_supplied"] * (1.0 - manifest["c_true"])
        / (1.0 - manifest["pi_supplied"] * manifest["c_true"])
    )
    if not np.allclose(manifest["pi_u_supplied"], expected_pi_u):
        raise RuntimeError("Supplied unlabeled priors are inconsistent")
    return manifest


def main() -> None:
    manifest = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUTPUT, index=False, lineterminator="\n")
    print(f"Wrote {len(manifest)} conditions to {OUTPUT}")
    print("Confirmatory tasks: 4000; expected method rows: 108000")


if __name__ == "__main__":
    main()
