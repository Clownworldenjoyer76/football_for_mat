#!/usr/bin/env python3
"""Combine all Stage 2 NFL prop CSV files into one Stage 3 CSV per week."""

from __future__ import annotations

import csv
from pathlib import Path


PROP_ENGINE_ROOT = Path("docs/win/football/nfl/prop_engine")
FINAL_ROOT = PROP_ENGINE_ROOT / "prop_picks_final"


def discover_stage_2_weeks() -> list[tuple[str, Path]]:
    discovered: list[tuple[str, Path]] = []

    if not FINAL_ROOT.is_dir():
        return discovered

    for season_path in sorted(FINAL_ROOT.iterdir()):
        if not season_path.is_dir():
            continue

        if not season_path.name.isdigit():
            continue

        stage_2_path = season_path / "stage_2"

        if not stage_2_path.is_dir():
            continue

        for week_path in sorted(stage_2_path.iterdir()):
            if not week_path.is_dir():
                continue

            if not week_path.name.startswith("week_"):
                continue

            week_number = week_path.name.removeprefix("week_")

            if not week_number.isdigit():
                continue

            discovered.append((season_path.name, week_path))

    return discovered


def discover_csv_files(week_path: Path) -> list[Path]:
    return sorted(
        path
        for path in week_path.rglob("*")
        if path.is_file() and path.suffix.lower() == ".csv"
    )


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]

    return fieldnames, rows


def combine_csv_files(csv_files: list[Path]) -> tuple[list[str], list[dict[str, str]]]:
    combined_fieldnames: list[str] = []
    combined_rows: list[dict[str, str]] = []

    for csv_path in csv_files:
        fieldnames, rows = read_csv(csv_path)

        for fieldname in fieldnames:
            if fieldname not in combined_fieldnames:
                combined_fieldnames.append(fieldname)

        combined_rows.extend(rows)

    return combined_fieldnames, combined_rows


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    fieldname: row.get(fieldname, "")
                    for fieldname in fieldnames
                }
            )


def process_week(season: str, week_path: Path) -> None:
    week_number = week_path.name.removeprefix("week_")
    csv_files = discover_csv_files(week_path)

    if not csv_files:
        return

    fieldnames, rows = combine_csv_files(csv_files)

    if not fieldnames:
        return

    output_path = (
        FINAL_ROOT
        / season
        / "stage_3"
        / f"{season}_{week_number}_all_props.csv"
    )

    write_csv(output_path, fieldnames, rows)


def main() -> None:
    stage_2_weeks = discover_stage_2_weeks()

    if not stage_2_weeks:
        raise FileNotFoundError(
            f"No Stage 2 week folders found under {FINAL_ROOT}"
        )

    for season, week_path in stage_2_weeks:
        process_week(season, week_path)


if __name__ == "__main__":
    main()
