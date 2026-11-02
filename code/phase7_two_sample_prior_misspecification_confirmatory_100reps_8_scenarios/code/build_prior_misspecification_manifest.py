"""Build the frozen two-sample class-prior confirmatory manifest."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT = PROJECT_DIR / "config" / "prior_misspecification_confirmatory_manifest.csv"
N_REPETITIONS = 100
N_METHODS = 27
GLOBAL_BASE_SEED = 720_000_000
PRIOR_RATIOS = (0.80, 0.90, 1.00, 1.10, 1.20)

REFERENCE = {
    "n_p_train": 300,
    "n_u_train": 4700,
    "n_p_val": 300,
    "n_u_val": 4700,
    "n_test": 10000,
    "p": 10,
    "pi_true": 0.20,
    "delta": 0.90,
    "n_signal": 3,
    "rho": 0.0,
    "covariance_type": "equal_toeplitz",
    "boundary_type": "linear",
    "structural_type": "linear_gaussian",
    "transform_strength": 0.0,
    "main_effect_shift": 0.0,
    "positive_variance_ratio": 1.0,
}


def _base_scenarios():
    """Return four prevalence settings and all four frozen stress tests."""
    return [
        (
            "CPM001", "TS001", "reference", "Reference, true pi=0.20", {},
            "Central linear setting used to measure ordinary prior sensitivity.",
        ),
        (
            "CPM002", "TS002", "prevalence", "Rare positives, true pi=0.01",
            {"pi_true": 0.01},
            "Extreme rare-positive setting; relative errors are small in absolute probability.",
        ),
        (
            "CPM003", "TS003", "prevalence", "Low prevalence, true pi=0.05",
            {"pi_true": 0.05},
            "Low-prevalence setting between the rare and reference designs.",
        ),
        (
            "CPM004", "TS011", "prevalence", "High prevalence, true pi=0.50",
            {"pi_true": 0.50},
            "Largest prevalence in the frozen benchmark; supplied priors remain below one.",
        ),
        (
            "CPM005", "TS026", "structural_stress", "Many irrelevant variables, p=500",
            {"p": 500, "structural_type": "many_irrelevant_variables"},
            "Tests whether prior error compounds high-dimensional estimation difficulty.",
        ),
        (
            "CPM006", "TS038", "structural_stress", "Curved boundary",
            {
                "structural_type": "curved_boundary",
                "boundary_type": "curved",
                "transform_strength": 1.5,
            },
            "Tests prior sensitivity when linear ranking models are misspecified.",
        ),
        (
            "CPM007", "TS039", "structural_stress", "Multiplicative interaction",
            {
                "structural_type": "multiplicative_interaction",
                "boundary_type": "interaction",
                "transform_strength": 0.67,
                "main_effect_shift": 0.60,
            },
            "Tests prior sensitivity under an XOR-like interaction plus a main effect.",
        ),
        (
            "CPM008", "TS040", "structural_stress", "Unequal class covariance",
            {
                "structural_type": "unequal_covariance",
                "covariance_type": "class_specific",
                "boundary_type": "quadratic",
                "positive_variance_ratio": 1.5,
            },
            "Tests prior sensitivity under a quadratic Bayes boundary.",
        ),
    ]


def build():
    rows = []
    condition_index = 0
    for scenario_index, (
        scenario_id,
        source_scenario_id,
        scenario_family,
        scenario_label,
        changes,
        interpretation,
    ) in enumerate(_base_scenarios(), start=1):
        values = {**REFERENCE, **changes}
        scenario_seed = GLOBAL_BASE_SEED + 1000 * scenario_index
        scenario_slug = (
            f"{scenario_id}_{source_scenario_id}_{scenario_family}_"
            f"pi{values['pi_true']:.2f}_p{values['p']}"
        ).replace(".", "p")
        for prior_index, prior_ratio in enumerate(PRIOR_RATIOS, start=1):
            condition_index += 1
            pi_supplied = values["pi_true"] * prior_ratio
            if not 0.0 < pi_supplied < 1.0:
                raise RuntimeError(
                    f"Invalid supplied prior for {scenario_id}: {pi_supplied}"
                )
            ratio_code = f"R{round(100 * prior_ratio):03d}"
            condition_id = f"{scenario_id}_{ratio_code}"
            condition_slug = (
                f"{condition_id}_piTrue{values['pi_true']:.3f}_"
                f"piHat{pi_supplied:.3f}"
            ).replace(".", "p")
            rows.append({
                "condition_index": condition_index,
                "condition_id": condition_id,
                "condition_slug": condition_slug,
                "scenario_index": scenario_index,
                "scenario_id": scenario_id,
                "scenario_slug": scenario_slug,
                "source_scenario_id": source_scenario_id,
                "scenario_family": scenario_family,
                "scenario_label": scenario_label,
                "sampling_scheme": "two_sample_case_control_scar",
                **values,
                "prior_index": prior_index,
                "prior_ratio": prior_ratio,
                "relative_prior_error": prior_ratio - 1.0,
                "pi_supplied": pi_supplied,
                "prior_is_correct": prior_ratio == 1.0,
                "c_applicable": False,
                "n_repetitions": N_REPETITIONS,
                "n_methods": N_METHODS,
                "expected_method_rows": N_REPETITIONS * N_METHODS,
                "global_base_seed": GLOBAL_BASE_SEED,
                "scenario_base_seed": scenario_seed,
                "seed_start": scenario_seed,
                "seed_stop": scenario_seed + N_REPETITIONS - 1,
                "seed_scheme": (
                    "720000000 + 1000*scenario_index + (rep-1); "
                    "the same seed is reused deliberately across the five prior ratios"
                ),
                "cross_ratio_shared_data_required": True,
                "study_phase": (
                    "phase7_two_sample_prior_misspecification_confirmatory_100reps"
                ),
                "study_role": "class_prior_misspecification_confirmatory",
                "pilot_source_phase": (
                    "phase6_two_sample_prior_misspecification_pilot"
                ),
                "frozen_method_source_phase": (
                    "phase5_two_sample_scar_confirmatory_100reps"
                ),
                "dgp_reproducibility_source_phase": (
                    "phase4c_deterministic_correlated_dgp_cross_node"
                ),
                "dgp_implementation": "deterministic_toeplitz_ar1_v1",
                "interpretation": interpretation,
            })
    manifest = pd.DataFrame(rows)
    if len(manifest) != 40:
        raise RuntimeError(f"Expected 40 prior conditions, found {len(manifest)}")
    if manifest["condition_id"].nunique() != 40:
        raise RuntimeError("Condition IDs are not unique")
    if manifest.groupby("scenario_id")["prior_ratio"].nunique().ne(5).any():
        raise RuntimeError("Every base scenario must contain five prior ratios")
    if manifest.groupby("scenario_id")["scenario_base_seed"].nunique().ne(1).any():
        raise RuntimeError("Seeds must be shared across prior ratios")
    return manifest


def main():
    manifest = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUTPUT, index=False, lineterminator="\n")
    print(f"Wrote {len(manifest)} conditions to {OUTPUT}")
    print(
        f"Confirmatory tasks: {len(manifest) * N_REPETITIONS}; "
        f"method rows: {len(manifest) * N_REPETITIONS * N_METHODS}"
    )


if __name__ == "__main__":
    main()
