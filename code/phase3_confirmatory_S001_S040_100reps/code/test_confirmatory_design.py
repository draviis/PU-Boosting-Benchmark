"""Fast, deterministic tests of the confirmatory manifest and task mapping."""

from __future__ import annotations

import pandas as pd

import confirmatory_runner as runner


def main() -> None:
    manifest = runner.load_validated_manifest(runner.DEFAULT_MANIFEST)
    assert len(manifest) == 40
    assert manifest["scenario_id"].tolist() == [f"S{i:03d}" for i in range(1, 41)]
    assert runner.task_to_scenario_rep(1, manifest) == (1, 1)
    assert runner.task_to_scenario_rep(100, manifest) == (1, 100)
    assert runner.task_to_scenario_rep(101, manifest) == (2, 1)
    assert runner.task_to_scenario_rep(1000, manifest) == (10, 100)
    assert runner.task_to_scenario_rep(1001, manifest) == (11, 1)
    assert runner.task_to_scenario_rep(4000, manifest) == (40, 100)

    confirmatory_seeds = set()
    pilot_seeds = set()
    for row in manifest.itertuples(index=False):
        confirmatory_seeds.update(range(row.scenario_base_seed, row.scenario_base_seed + 100))
        pilot_seeds.update(
            range(row.pilot_scenario_base_seed, row.pilot_scenario_base_seed + row.pilot_n_repetitions)
        )
    assert len(confirmatory_seeds) == 4000
    assert confirmatory_seeds.isdisjoint(pilot_seeds)

    structural = manifest.fillna("").set_index("scenario_id")["structural_test_id"]
    assert structural.loc[["S026", "S038", "S039", "S040"]].to_dict() == {
        "S026": "ST04",
        "S038": "ST01",
        "S039": "ST02",
        "S040": "ST03",
    }
    assert pd.to_numeric(manifest["expected_method_rows"]).sum() == 108000
    print("ALL CONFIRMATORY DESIGN TESTS PASSED")


if __name__ == "__main__":
    main()
