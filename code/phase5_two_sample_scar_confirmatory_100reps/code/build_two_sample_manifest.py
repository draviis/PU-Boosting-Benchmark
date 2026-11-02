"""Build the frozen TS001--TS040 two-sample confirmatory manifest."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT = PROJECT_DIR / "config" / "two_sample_scenarios_TS001_TS040_100reps.csv"
N_REPETITIONS = 100
N_METHODS = 27
GLOBAL_BASE_SEED = 520_000_000

REFERENCE = {
    "n_p_train": 300,
    "n_u_train": 4700,
    "n_p_val": 300,
    "n_u_val": 4700,
    "n_test": 10000,
    "p": 10,
    "pi": 0.20,
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


def _designs():
    designs = [
        ("reference", "reference", "Reference two-sample design", {}, "Baseline for every one-factor comparison."),
    ]
    for pi in [0.01, 0.05, 0.10, 0.15, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        designs.append((
            "prevalence",
            "pi",
            f"Population prevalence pi={pi:.2f}",
            {"pi": pi},
            "Changes positive prevalence in U while P remains a pure f1 sample.",
        ))
    for n_p in [50, 100, 150, 200, 250, 350, 400, 450, 500]:
        designs.append((
            "positive_sample_size",
            "n_p",
            f"Positive sample size nP={n_p}",
            {"n_p_train": n_p, "n_p_val": n_p},
            "Changes the independent pure-positive sample size at fixed nU=4700.",
        ))
    for delta in [0.50, 0.70, 1.20]:
        designs.append((
            "separation",
            "delta",
            f"Mahalanobis separation delta={delta:.2f}",
            {"delta": delta},
            "Changes class overlap while preserving the active direction.",
        ))
    for p in [50, 100, 500]:
        structural = "many_irrelevant_variables" if p == 500 else "linear_gaussian"
        designs.append((
            "dimension",
            "p",
            f"Ambient dimension p={p}",
            {"p": p, "structural_type": structural},
            "Adds irrelevant coordinates while retaining three active variables.",
        ))
    for rho in [0.30, 0.60]:
        designs.append((
            "correlation",
            "rho",
            f"Toeplitz correlation rho={rho:.2f}",
            {"rho": rho},
            "Changes feature correlation at fixed Mahalanobis separation.",
        ))
    for pi in [0.01, 0.10, 0.50]:
        for n_p in [25, 100, 500]:
            designs.append((
                "pi_by_positive_sample_size",
                "pi_x_n_p",
                f"Interaction pi={pi:.2f}, nP={n_p}",
                {"pi": pi, "n_p_train": n_p, "n_p_val": n_p},
                "Reduced interaction grid for prevalence and positive-sample information.",
            ))
    designs.extend([
        (
            "structural_stress",
            "boundary",
            "Curved boundary",
            {"structural_type": "curved_boundary", "boundary_type": "curved", "transform_strength": 1.5},
            "Invertible nonlinear transformation creates a curved boundary in observed X.",
        ),
        (
            "structural_stress",
            "interaction",
            "Multiplicative interaction",
            {"structural_type": "multiplicative_interaction", "boundary_type": "interaction", "transform_strength": 0.67, "main_effect_shift": 0.60},
            "XOR-like two-feature mixture plus a smaller main effect.",
        ),
        (
            "structural_stress",
            "covariance",
            "Unequal class covariance",
            {"structural_type": "unequal_covariance", "covariance_type": "class_specific", "boundary_type": "quadratic", "positive_variance_ratio": 1.5},
            "Positive active coordinates have 1.5 times the negative variance.",
        ),
    ])
    if len(designs) != 40:
        raise RuntimeError(f"Expected 40 designs, found {len(designs)}")
    return designs


def build():
    rows = []
    for index, (family, factor, label, changes, interpretation) in enumerate(_designs(), start=1):
        values = {**REFERENCE, **changes}
        scenario_id = f"TS{index:03d}"
        scenario_seed = GLOBAL_BASE_SEED + 1000 * index
        expected_p_train = values["n_p_train"]
        expected_u_positive = values["n_u_train"] * values["pi"]
        slug = (
            f"{scenario_id}_{family}_{factor}_"
            f"pi{values['pi']:.2f}_nP{values['n_p_train']}_p{values['p']}"
        ).replace(".", "p")
        structural_test_id = {
            "TS026": "TST04",
            "TS038": "TST01",
            "TS039": "TST02",
            "TS040": "TST03",
        }.get(scenario_id, "")
        rows.append({
            "scenario_index": index,
            "scenario_id": scenario_id,
            "structural_test_id": structural_test_id,
            "scenario_slug": slug,
            "scenario_label": label,
            "scenario_family": family,
            "varied_factor": factor,
            "is_reference": index == 1,
            "reference_scenario_id": "TS001",
            "sampling_scheme": "two_sample_case_control_scar",
            **values,
            "c_applicable": False,
            "n_repetitions": N_REPETITIONS,
            "n_methods": N_METHODS,
            "expected_method_rows": N_REPETITIONS * N_METHODS,
            "global_base_seed": GLOBAL_BASE_SEED,
            "scenario_base_seed": scenario_seed,
            "seed_start": scenario_seed,
            "seed_stop": scenario_seed + N_REPETITIONS - 1,
            "seed_scheme": "520000000 + 1000*scenario_index + (rep-1)",
            "expected_pure_positive_train": expected_p_train,
            "expected_hidden_positive_in_u_train": expected_u_positive,
            "expected_hidden_negative_in_u_train": values["n_u_train"] - expected_u_positive,
            "case_control_positive_fraction_train": expected_p_train / (expected_p_train + values["n_u_train"]),
            "target_linear_bayes_AUROC": (
                0.5 * (1.0 + math.erf(values["delta"] / 2.0))
                if values["structural_type"] in {"linear_gaussian", "many_irrelevant_variables"}
                else ""
            ),
            "interpretation": interpretation,
            "pilot_phase": "phase5_two_sample_scar_confirmatory_100reps",
            "study_role": "confirmatory_monte_carlo",
            "frozen_method_source_phase": "phase4b_two_sample_method_repair_pilot_10reps",
            "frozen_method_source_folder": "phase4b_two_sample_method_repair_pilot_40_scenarios",
            "dgp_reproducibility_source_phase": "phase4c_deterministic_correlated_dgp_cross_node",
            "confirmatory_seed_reuse": False,
            "dgp_implementation": "deterministic_toeplitz_ar1_v1",
            "repair_scope": "TM003_TM011_TM012_TM013_TM014_TM015_TM027",
        })
    manifest = pd.DataFrame(rows)
    expected_ids = [f"TS{i:03d}" for i in range(1, 41)]
    if manifest["scenario_id"].tolist() != expected_ids:
        raise RuntimeError("Scenario IDs are not consecutive")
    return manifest


def main():
    manifest = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUTPUT, index=False, lineterminator="\n")
    print(f"Wrote {len(manifest)} scenarios to {OUTPUT}")
    print(f"Confirmatory tasks: {len(manifest) * N_REPETITIONS}")


if __name__ == "__main__":
    main()
