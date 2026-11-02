"""Memory-safe fail-closed collector for the confirmatory prior study."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
MANIFEST = (
    PROJECT_DIR / "config" / "prior_misspecification_confirmatory_manifest.csv"
)
RUNNER = PROJECT_DIR / "code" / "prior_misspecification_runner.py"
RESULTS = PROJECT_DIR / "results"
COMBINED = RESULTS / "prior_misspecification_combined"
EXPECTED_REPETITIONS = 100


def manifest_rows():
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return sorted(rows, key=lambda row: int(row["condition_index"]))


def condition_root(row):
    return (
        RESULTS
        / "scenarios"
        / row["scenario_slug"]
        / "prior_conditions"
        / row["condition_slug"]
        / "aggregate"
    )


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
    raw_paths, summary_paths, diagnostic_paths, overview = [], [], [], []
    for row in manifest_rows():
        condition_index = int(row["condition_index"])
        subprocess.run(
            [sys.executable, str(RUNNER), "--aggregate-condition", str(condition_index)],
            cwd=PROJECT_DIR,
            check=True,
        )
        aggregate = condition_root(row)
        condition_id = row["condition_id"]
        report = aggregate / f"{condition_id}_collection_report.csv"
        with report.open("r", encoding="utf-8-sig", newline="") as handle:
            report_rows = list(csv.DictReader(handle))
        complete = sum(str(item["complete"]).lower() == "true" for item in report_rows)
        if len(report_rows) != EXPECTED_REPETITIONS or complete != EXPECTED_REPETITIONS:
            raise RuntimeError(f"{condition_id}: {complete}/{len(report_rows)} complete")
        raw_paths.append(aggregate / f"{condition_id}_raw_100reps.csv")
        summary_paths.append(aggregate / f"{condition_id}_summary_100reps.csv")
        diagnostic_paths.append(aggregate / f"{condition_id}_diagnostics_100reps.csv")
        overview.append({
            "condition_index": condition_index,
            "condition_id": condition_id,
            "scenario_id": row["scenario_id"],
            "source_scenario_id": row["source_scenario_id"],
            "prior_ratio": row["prior_ratio"],
            "pi_true": row["pi_true"],
            "pi_supplied": row["pi_supplied"],
            "complete_repetitions": complete,
            "expected_repetitions": EXPECTED_REPETITIONS,
            "complete": True,
        })
        print(f"Validated {condition_id}: 100/100")
    COMBINED.mkdir(parents=True, exist_ok=True)
    with (COMBINED / "prior_condition_collection_overview.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(overview[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(overview)
    concatenate(raw_paths, COMBINED / "prior_misspecification_raw.csv")
    concatenate(summary_paths, COMBINED / "prior_misspecification_condition_summaries.csv")
    concatenate(diagnostic_paths, COMBINED / "prior_misspecification_diagnostics.csv")
    print(f"Complete combined confirmatory results: {COMBINED}")


if __name__ == "__main__":
    main()
