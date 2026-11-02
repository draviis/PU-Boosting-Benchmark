"""Memory-safe S001--S040 confirmatory collector.

Each scenario is aggregated in a fresh Python process.  The final CSV files
are then combined with the standard library one row at a time.  This preserves
the source-hashed simulation runner while avoiding the native-library
segmentation fault observed when one process retained all scenario frames.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
MANIFEST = PROJECT_DIR / "config" / "confirmatory_scenarios_S001_S040_100reps.csv"
RUNNER = CODE_DIR / "confirmatory_runner.py"
RESULTS_ROOT = PROJECT_DIR / "results"
COMBINED_DIR = RESULTS_ROOT / "confirmatory_combined"


def read_manifest() -> list[dict[str, str]]:
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["scenario_index"]))
    return rows


def atomic_write_rows(
    destination: Path,
    fieldnames: list[str],
    rows: list[dict[str, object]],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_concatenate_csv(
    source_paths: list[Path],
    destination: Path,
) -> None:
    if not source_paths:
        raise RuntimeError(f"No source CSV files for {destination.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    expected_header = None
    try:
        with temporary.open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output, lineterminator="\n")
            for source_path in source_paths:
                with source_path.open(
                    "r", encoding="utf-8-sig", newline=""
                ) as source:
                    reader = csv.reader(source)
                    try:
                        header = next(reader)
                    except StopIteration as error:
                        raise RuntimeError(
                            f"Cannot combine empty CSV: {source_path}"
                        ) from error
                    if expected_header is None:
                        expected_header = header
                        writer.writerow(header)
                    elif header != expected_header:
                        raise RuntimeError(
                            f"CSV header mismatch: {source_path}"
                        )
                    writer.writerows(reader)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_collection_report(path: Path, expected: int) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    complete = sum(
        str(row.get("complete", "")).strip().lower() == "true"
        for row in rows
    )
    if len(rows) != expected or complete != expected:
        raise RuntimeError(
            f"{path}: expected {expected}/{expected} complete rows, "
            f"found {complete}/{len(rows)}"
        )
    return complete


def collect(*, combine_only: bool) -> None:
    manifest = read_manifest()
    registry_rows = []
    raw_paths = []
    summary_paths = []
    diagnostic_paths = []

    for row in manifest:
        index = int(row["scenario_index"])
        scenario_id = row["scenario_id"]
        slug = row["scenario_slug"]
        repetitions = int(row["n_repetitions"])
        scenario_root = RESULTS_ROOT / "scenarios" / slug
        aggregate_dir = scenario_root / "aggregate"

        if not combine_only:
            subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "--aggregate-scenario",
                    str(index),
                ],
                cwd=PROJECT_DIR,
                check=True,
            )

        report_path = aggregate_dir / f"{scenario_id}_collection_report.csv"
        raw_path = aggregate_dir / f"{scenario_id}_raw_{repetitions}reps.csv"
        summary_path = (
            aggregate_dir / f"{scenario_id}_summary_{repetitions}reps.csv"
        )
        diagnostics_path = (
            aggregate_dir / f"{scenario_id}_diagnostics_{repetitions}reps.csv"
        )
        for required in [report_path, raw_path, summary_path, diagnostics_path]:
            if not required.is_file():
                raise FileNotFoundError(f"Missing aggregate file: {required}")
        complete = validate_collection_report(report_path, repetitions)
        raw_paths.append(raw_path)
        summary_paths.append(summary_path)
        diagnostic_paths.append(diagnostics_path)
        registry_rows.append(
            {
                "scenario_index": index,
                "scenario_id": scenario_id,
                "scenario_slug": slug,
                "complete_repetitions": complete,
                "expected_repetitions": repetitions,
                "complete": True,
                "raw_path": str(raw_path),
                "summary_path": str(summary_path),
                "diagnostics_path": str(diagnostics_path),
            }
        )
        print(f"Validated aggregate {scenario_id}: {complete}/{repetitions}")

    atomic_write_rows(
        COMBINED_DIR / "confirmatory_collection_overview.csv",
        list(registry_rows[0]),
        registry_rows,
    )
    atomic_concatenate_csv(
        raw_paths,
        COMBINED_DIR / "confirmatory_raw_all_scenarios.csv",
    )
    atomic_concatenate_csv(
        summary_paths,
        COMBINED_DIR / "confirmatory_method_scenario_summary.csv",
    )
    atomic_concatenate_csv(
        diagnostic_paths,
        COMBINED_DIR / "confirmatory_diagnostics_all_scenarios.csv",
    )
    print(f"Safe confirmatory collection written to {COMBINED_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--combine-only",
        action="store_true",
        help="Use existing per-scenario aggregates without refreshing them.",
    )
    args = parser.parse_args()
    collect(combine_only=args.combine_only)


if __name__ == "__main__":
    main()
