"""Extract one immutable data fingerprint per pilot scenario for release tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = PROJECT_DIR.parent
OUTPUT = PROJECT_DIR / "frozen_reference" / "pilot_rep1_data_fingerprints_S001_S040.csv"
SOURCES = [
    (
        "phase1_pilot_28",
        WORKSPACE_DIR / "phase1_pilot_28_scenarios/results/phase1_combined/phase1_raw_all_scenarios.csv",
        {f"S{index:03d}" for index in range(1, 29)},
    ),
    (
        "phase1b_pi_c_pilot_9",
        WORKSPACE_DIR / "phase1b_pi_c_interaction_9_scenarios/results/phase1b_combined/phase1b_raw_all_scenarios.csv",
        {f"S{index:03d}" for index in range(29, 38)},
    ),
    (
        "phase2_structural_pilot",
        WORKSPACE_DIR / "phase2_structural_pilot_4_scenarios/results/structural_combined/structural_raw_all_scenarios.csv",
        {"S038", "S039", "S040"},
    ),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    rows = []
    for source_name, source_path, wanted in SOURCES:
        frame = pd.read_csv(
            source_path,
            usecols=["scenario_id", "rep", "seed", "data_fingerprint"],
        )
        frame = frame.loc[
            frame["scenario_id"].isin(wanted) & frame["rep"].eq(1)
        ]
        for scenario_id, group in frame.groupby("scenario_id", sort=True):
            fingerprints = group["data_fingerprint"].dropna().astype(str).unique()
            seeds = pd.to_numeric(group["seed"], errors="raise").unique()
            if len(fingerprints) != 1 or len(seeds) != 1:
                raise RuntimeError(f"{scenario_id}: pilot rows do not share one seed/fingerprint")
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "pilot_rep": 1,
                    "pilot_seed": int(seeds[0]),
                    "expected_data_fingerprint": fingerprints[0],
                    "pilot_source": source_name,
                    "pilot_combined_csv": source_path.name,
                    "pilot_combined_csv_sha256": sha256_file(source_path),
                }
            )
    result = pd.DataFrame(rows).sort_values("scenario_id").reset_index(drop=True)
    expected = [f"S{index:03d}" for index in range(1, 41)]
    if result["scenario_id"].tolist() != expected:
        raise RuntimeError("Frozen fingerprint table does not contain S001--S040 exactly")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False, lineterminator="\n")
    print(f"Wrote {len(result)} frozen pilot fingerprints to {OUTPUT}")


if __name__ == "__main__":
    main()
