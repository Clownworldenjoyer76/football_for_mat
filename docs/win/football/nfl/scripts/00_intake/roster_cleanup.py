#!/usr/bin/env python3
"""
roster_cleanup.py

Reads the hardened raw ESPN roster pull and writes roster_master.csv containing
only KEEP_COLUMNS, in exactly the existing order.
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
NFL_ROOT = SCRIPT_PATH.parents[2]

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_reporter import PipelineReporter

INPUT_PATH = NFL_ROOT / "data" / "raw" / "raw_roster.csv"
OUTPUT_PATH = NFL_ROOT / "data" / "master" / "roster_master.csv"
TEAM_MAP_PATH = NFL_ROOT / "config" / "mapping" / "team_map.csv"
REPORT_ROOT = NFL_ROOT / "errors"

KEEP_COLUMNS = [
    "age",
    "alternateIds.sdr",
    "birthPlace.city",
    "birthPlace.country",
    "birthPlace.state",
    "college.abbrev",
    "college.guid",
    "college.id",
    "college.name",
    "college.shortName",
    "contract.active",
    "contract.bonus",
    "contract.optionType",
    "contract.salary",
    "contract.salaryRemaining",
    "contract.season.endDate",
    "contract.season.startDate",
    "contract.season.year",
    "contract.signedThrough",
    "dateOfBirth",
    "debutYear",
    "displayHeight",
    "displayName",
    "displayWeight",
    "experience.years",
    "firstName",
    "fullName",
    "guid",
    "hand.abbreviation",
    "hand.displayValue",
    "hand.type",
    "headshot.alt",
    "headshot.href",
    "height",
    "id",
    "injuries.0.date",
    "injuries.0.status",
    "jersey",
    "lastName",
    "position.abbreviation",
    "position.displayName",
    "position.id",
    "position.leaf",
    "position.name",
    "position.parent.abbreviation",
    "position.parent.displayName",
    "position.parent.id",
    "position.parent.leaf",
    "position.parent.name",
    "shortName",
    "slug",
    "status.abbreviation",
    "status.id",
    "status.name",
    "status.type",
    "team_id",
    "uid",
    "weight",
]

CORE_REQUIRED_FIELDS = [
    "id",
    "displayName",
    "position.id",
    "position.abbreviation",
    "team_id",
]


class RosterCleanupError(RuntimeError):
    pass


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def fail(message: str) -> None:
    raise RosterCleanupError(message)


def read_csv(
    path: Path,
    *,
    label: str,
    required_columns: list[str] | None = None,
    exact_columns: list[str] | None = None,
) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        fail(f"Missing {label}: {path}")

    if path.stat().st_size == 0:
        fail(f"Zero-byte {label}: {path}")

    try:
        with path.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            rows = list(reader)
    except Exception as exc:
        fail(
            f"Could not read {label} {path}: "
            f"{type(exc).__name__}: {exc}"
        )

    if not fieldnames:
        fail(f"{label} has no CSV header: {path}")

    if required_columns:
        missing = [
            column
            for column in required_columns
            if column not in fieldnames
        ]
        if missing:
            fail(
                f"{label} missing expected columns: {missing}"
            )

    if exact_columns is not None and fieldnames != exact_columns:
        fail(
            f"{label} has unexpected column order/schema. "
            f"Expected={exact_columns} actual={fieldnames}"
        )

    if not rows:
        fail(f"{label} contains no data rows: {path}")

    return fieldnames, rows


def load_canonical_team_ids() -> set[str]:
    _, rows = read_csv(
        TEAM_MAP_PATH,
        label="NFL team map",
        required_columns=[
            "sport",
            "league",
            "team_id",
        ],
    )

    team_ids: set[str] = set()

    for line_number, row in enumerate(rows, start=2):
        sport = clean(row.get("sport")).casefold()
        league = clean(row.get("league")).casefold()

        if sport not in {"", "football"}:
            continue
        if league not in {"", "nfl"}:
            continue

        team_id = clean(row.get("team_id"))
        if not team_id:
            continue

        team_ids.add(team_id)

    if len(team_ids) != 32:
        fail(
            f"{TEAM_MAP_PATH} must resolve exactly 32 NFL teams; "
            f"found {len(team_ids)}"
        )

    return team_ids


def validate_rows(
    rows: list[dict[str, str]],
    *,
    label: str,
    canonical_team_ids: set[str],
) -> None:
    seen_ids: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    represented_teams: set[str] = set()

    for line_number, row in enumerate(rows, start=2):
        for field in CORE_REQUIRED_FIELDS:
            if not clean(row.get(field)):
                fail(
                    f"{label} line {line_number} "
                    f"has blank {field}"
                )

        athlete_id = clean(row.get("id"))
        team_id = clean(row.get("team_id"))

        if team_id not in canonical_team_ids:
            fail(
                f"{label} line {line_number} "
                f"has unknown team_id={team_id!r}"
            )

        if athlete_id in seen_ids:
            fail(
                f"{label} contains duplicate athlete "
                f"id={athlete_id}"
            )

        pair = (athlete_id, team_id)
        if pair in seen_pairs:
            fail(
                f"{label} contains duplicate athlete/team "
                f"pair={pair}"
            )

        seen_ids.add(athlete_id)
        seen_pairs.add(pair)
        represented_teams.add(team_id)

    if represented_teams != canonical_team_ids:
        fail(
            f"{label} team IDs do not exactly match "
            "the canonical 32-team NFL universe"
        )


def project_rows(
    raw_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    return [
        {
            column: row.get(column, "")
            for column in KEEP_COLUMNS
        }
        for row in raw_rows
    ]


def write_csv(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=KEEP_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def validate_round_trip(
    staged_rows: list[dict[str, str]],
    projected_rows: list[dict[str, str]],
) -> None:
    if len(staged_rows) != len(projected_rows):
        fail(
            "Staged roster row count changed during "
            f"round-trip validation: staged={len(staged_rows)} "
            f"expected={len(projected_rows)}"
        )

    for index, (staged, expected) in enumerate(
        zip(staged_rows, projected_rows),
        start=2,
    ):
        for column in KEEP_COLUMNS:
            if staged.get(column, "") != expected.get(column, ""):
                fail(
                    "Staged roster projection mismatch "
                    f"line={index} column={column!r}"
                )


def publish(
    rows: list[dict[str, str]],
    *,
    canonical_team_ids: set[str],
) -> None:
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory(
        prefix=".roster_master_stage_",
        dir=OUTPUT_PATH.parent,
    ) as staging_dir:
        staged_path = Path(staging_dir) / OUTPUT_PATH.name

        write_csv(
            staged_path,
            rows,
        )

        staged_columns, staged_rows = read_csv(
            staged_path,
            label="staged roster master CSV",
            exact_columns=KEEP_COLUMNS,
        )

        if staged_columns != KEEP_COLUMNS:
            fail(
                "Staged roster master schema changed "
                "during round-trip validation"
            )

        validate_rows(
            staged_rows,
            label="staged roster master CSV",
            canonical_team_ids=canonical_team_ids,
        )

        validate_round_trip(
            staged_rows,
            rows,
        )

        os.replace(
            staged_path,
            OUTPUT_PATH,
        )


def run(
    reporter: PipelineReporter,
) -> None:
    reporter.add_input(INPUT_PATH)
    reporter.add_input(TEAM_MAP_PATH)
    reporter.update_details(
        {
            "input_path": str(INPUT_PATH),
            "output_path": str(OUTPUT_PATH),
            "expected_output_columns": len(KEEP_COLUMNS),
            "publication_completed": False,
            "staged_roundtrip_verified": False,
        }
    )

    canonical_team_ids = load_canonical_team_ids()

    raw_columns, raw_rows = read_csv(
        INPUT_PATH,
        label="raw roster CSV",
        required_columns=KEEP_COLUMNS,
    )

    validate_rows(
        raw_rows,
        label="raw roster CSV",
        canonical_team_ids=canonical_team_ids,
    )

    projected_rows = project_rows(
        raw_rows
    )

    validate_rows(
        projected_rows,
        label="projected roster master rows",
        canonical_team_ids=canonical_team_ids,
    )

    reporter.set_rows(
        rows_in=len(raw_rows),
        rows_out=0,
    )
    reporter.update_details(
        {
            "raw_rows": len(raw_rows),
            "raw_columns": len(raw_columns),
            "canonical_team_count": len(canonical_team_ids),
            "projected_rows": len(projected_rows),
        }
    )

    publish(
        projected_rows,
        canonical_team_ids=canonical_team_ids,
    )

    reporter.add_output(OUTPUT_PATH)
    reporter.set_rows(
        rows_in=len(raw_rows),
        rows_out=len(projected_rows),
    )
    reporter.update_details(
        {
            "rows_published": len(projected_rows),
            "output_columns": len(KEEP_COLUMNS),
            "staged_roundtrip_verified": True,
            "publication_completed": True,
        }
    )

    print(
        f"rows={len(projected_rows)} "
        f"columns={len(KEEP_COLUMNS)} "
        f"output={OUTPUT_PATH}"
    )


def main() -> int:
    try:
        with PipelineReporter(
            script=SCRIPT_PATH,
            stage="00_intake",
            report_root=REPORT_ROOT,
            pipeline="NFL",
            league="NFL",
            extra_context={
                "component": "roster cleanup",
            },
        ) as reporter:
            run(reporter)

        return 0

    except Exception as exc:
        print(
            f"ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
