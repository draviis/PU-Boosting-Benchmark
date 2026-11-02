"""Memory-safe collector: aggregate each TS scenario in a fresh process."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT_DIR / "config" / "two_sample_scenarios_TS001_TS040_100reps.csv"
RUNNER = PROJECT_DIR / "code" / "two_sample_runner.py"
RESULTS = PROJECT_DIR / "results"
COMBINED = RESULTS / "two_sample_combined"
EXPECTED_REPETITIONS = 100


def rows():
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        out = list(csv.DictReader(handle))
    return sorted(out, key=lambda row: int(row["scenario_index"]))


def concatenate(sources, destination):
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_header = None
    try:
        with temporary.open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output, lineterminator="\n")
            for source in sources:
                with source.open("r", encoding="utf-8-sig", newline="") as handle:
                    reader = csv.reader(handle)
                    header = next(reader)
                    if expected_header is None:
                        expected_header = header
                        writer.writerow(header)
                    elif header != expected_header:
                        raise RuntimeError(f"header mismatch in {source}")
                    writer.writerows(reader)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    manifest = rows()
    raw_paths, summary_paths, diagnostic_paths, overview = [], [], [], []
    for row in manifest:
        index = int(row["scenario_index"])
        subprocess.run(
            [sys.executable, str(RUNNER), "--aggregate-scenario", str(index)],
            cwd=PROJECT_DIR,
            check=True,
        )
        scenario_id = row["scenario_id"]
        aggregate = RESULTS / "scenarios" / row["scenario_slug"] / "aggregate"
        report = aggregate / f"{scenario_id}_collection_report.csv"
        with report.open("r", encoding="utf-8-sig", newline="") as handle:
            report_rows = list(csv.DictReader(handle))
        complete = sum(str(item["complete"]).lower() == "true" for item in report_rows)
        if len(report_rows) != EXPECTED_REPETITIONS or complete != EXPECTED_REPETITIONS:
            raise RuntimeError(f"{scenario_id}: {complete}/{len(report_rows)} complete")
        raw_paths.append(aggregate / f"{scenario_id}_raw_{EXPECTED_REPETITIONS}reps.csv")
        summary_paths.append(aggregate / f"{scenario_id}_summary_{EXPECTED_REPETITIONS}reps.csv")
        diagnostic_paths.append(aggregate / f"{scenario_id}_diagnostics_{EXPECTED_REPETITIONS}reps.csv")
        overview.append({
            "scenario_index": index,
            "scenario_id": scenario_id,
            "scenario_slug": row["scenario_slug"],
            "complete_repetitions": EXPECTED_REPETITIONS,
            "expected_repetitions": EXPECTED_REPETITIONS,
            "complete": True,
        })
        print(
            f"Validated {scenario_id}: "
            f"{EXPECTED_REPETITIONS}/{EXPECTED_REPETITIONS}"
        )
    COMBINED.mkdir(parents=True, exist_ok=True)
    with (COMBINED / "two_sample_collection_overview.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(overview[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(overview)
    concatenate(raw_paths, COMBINED / "two_sample_raw_all_scenarios.csv")
    concatenate(summary_paths, COMBINED / "two_sample_method_scenario_summary.csv")
    concatenate(diagnostic_paths, COMBINED / "two_sample_diagnostics_all_scenarios.csv")
    print(f"Complete combined results: {COMBINED}")


if __name__ == "__main__":
    main()
