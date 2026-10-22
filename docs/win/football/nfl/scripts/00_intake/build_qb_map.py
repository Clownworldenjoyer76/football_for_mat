#!/usr/bin/env python3
"""
Build the current NFL QB mapping from roster_master.csv, team_master.csv,
and the validated per-team cleaned depth charts.

Every roster QB is preserved exactly once. A roster QB that is legitimately
absent from its team's QB depth rows keeps blank depth-derived fields.
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

ROSTER_MASTER = NFL_ROOT / "data" / "master" / "roster_master.csv"
TEAM_MASTER = NFL_ROOT / "data" / "master" / "team_master.csv"
DEPTH_CHART_DIR = NFL_ROOT / "data" / "master" / "depth_charts"
OUTPUT_FILE = NFL_ROOT / "config" / "mapping" / "qb_map_nfl.csv"
REPORT_ROOT = NFL_ROOT / "errors"

OUTPUT_HEADERS = [
    "sport",
    "league",
    "player_id",
    "qb_name",
    "team_abbr",
    "depth_chart_rank",
    "starter_flag",
    "backup_flag",
    "injury",
    "position_abb",
    "position.id",
    "team_id",
]

ROSTER_REQUIRED_COLUMNS = [
    "id",
    "displayName",
    "position.id",
    "position.abbreviation",
    "team_id",
]

TEAM_REQUIRED_COLUMNS = [
    "sport",
    "league",
    "team_id",
    "team_abbr",
]

DEPTH_HEADERS = [
    "sport",
    "league",
    "player_id",
    "name",
    "team",
    "position_abb",
    "position",
    "injury",
    "depth_chart_rank",
    "starter_flag",
    "backup_flag",
    "team_id",
    "season",
    "guid",
    "uid",
]

DEPTH_DERIVED_FIELDS = [
    "depth_chart_rank",
    "starter_flag",
    "backup_flag",
    "injury",
    "position_abb",
]


class QBMapError(RuntimeError):
    pass


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def fail(message: str) -> None:
    raise QBMapError(message)


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

    if len(fieldnames) != len(set(fieldnames)):
        fail(f"{label} contains duplicate CSV columns: {path}")

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


def load_team_master() -> dict[str, str]:
    _, rows = read_csv(
        TEAM_MASTER,
        label="NFL team master",
        required_columns=TEAM_REQUIRED_COLUMNS,
    )

    by_id: dict[str, str] = {}
    by_abbr: dict[str, str] = {}

    for line_number, row in enumerate(rows, start=2):
        sport = clean(row.get("sport")).casefold()
        league = clean(row.get("league")).casefold()

        if sport != "football" or league != "nfl":
            continue

        team_id = clean(row.get("team_id"))
        team_abbr = clean(row.get("team_abbr")).upper()

        if not team_id or not team_abbr:
            fail(
                f"{TEAM_MASTER} line {line_number} has "
                "blank team_id/team_abbr"
            )

        previous_abbr = by_id.get(team_id)
        if previous_abbr and previous_abbr != team_abbr:
            fail(
                f"{TEAM_MASTER} has conflicting team_abbr "
                f"for team_id={team_id}: "
                f"{previous_abbr!r} vs {team_abbr!r}"
            )

        previous_id = by_abbr.get(team_abbr)
        if previous_id and previous_id != team_id:
            fail(
                f"{TEAM_MASTER} has conflicting team_id "
                f"for team_abbr={team_abbr}: "
                f"{previous_id!r} vs {team_id!r}"
            )

        by_id[team_id] = team_abbr
        by_abbr[team_abbr] = team_id

    if len(by_id) != 32 or len(by_abbr) != 32:
        fail(
            "NFL team master must resolve exactly 32 teams; "
            f"ids={len(by_id)} abbrs={len(by_abbr)}"
        )

    return by_id


def load_roster_qbs(
    team_map: dict[str, str],
) -> tuple[int, list[dict[str, str]]]:
    _, rows = read_csv(
        ROSTER_MASTER,
        label="roster master",
        required_columns=ROSTER_REQUIRED_COLUMNS,
    )

    qb_rows: list[dict[str, str]] = []
    seen_qb_ids: set[str] = set()

    for line_number, row in enumerate(rows, start=2):
        if clean(row.get("position.id")) != "8":
            continue

        player_id = clean(row.get("id"))
        qb_name = clean(row.get("displayName"))
        team_id = clean(row.get("team_id"))

        if not player_id:
            fail(
                f"Roster QB line {line_number} has blank id"
            )

        if not qb_name:
            fail(
                f"Roster QB line {line_number} has blank displayName"
            )

        if not team_id:
            fail(
                f"Roster QB line {line_number} has blank team_id"
            )

        if team_id not in team_map:
            fail(
                f"Roster QB line {line_number} has unknown "
                f"team_id={team_id!r}"
            )

        if player_id in seen_qb_ids:
            fail(
                "Roster contains duplicate QB player_id="
                f"{player_id}"
            )

        position_abbr = clean(
            row.get("position.abbreviation")
        ).upper()
        if position_abbr and position_abbr != "QB":
            fail(
                "Roster QB position mismatch "
                f"player_id={player_id} "
                f"position.id=8 "
                f"position.abbreviation={position_abbr!r}"
            )

        seen_qb_ids.add(player_id)
        qb_rows.append(row)

    if not qb_rows:
        fail("Roster master contains no position.id=8 QB rows")

    return len(rows), qb_rows


def validate_depth_row(
    row: dict[str, str],
    *,
    team_id: str,
    team_abbr: str,
    line_number: int,
    path: Path,
) -> None:
    for field in (
        "player_id",
        "name",
        "team",
        "position_abb",
        "injury",
        "depth_chart_rank",
        "starter_flag",
        "backup_flag",
        "team_id",
        "season",
    ):
        if not clean(row.get(field)):
            fail(
                f"{path} line {line_number} has blank {field}"
            )

    if clean(row.get("sport")) != "football":
        fail(
            f"{path} line {line_number} has invalid sport"
        )

    if clean(row.get("league")) != "nfl":
        fail(
            f"{path} line {line_number} has invalid league"
        )

    if clean(row.get("team")).upper() != team_abbr:
        fail(
            f"{path} line {line_number} team mismatch "
            f"expected={team_abbr} "
            f"received={row.get('team')!r}"
        )

    if clean(row.get("team_id")) != team_id:
        fail(
            f"{path} line {line_number} team_id mismatch "
            f"expected={team_id} "
            f"received={row.get('team_id')!r}"
        )

    try:
        rank = int(clean(row.get("depth_chart_rank")))
        starter = int(clean(row.get("starter_flag")))
        backup = int(clean(row.get("backup_flag")))
    except ValueError:
        fail(
            f"{path} line {line_number} has non-integer "
            "rank/starter/backup value"
        )

    if rank < 1:
        fail(
            f"{path} line {line_number} has invalid "
            f"depth_chart_rank={rank}"
        )

    expected_starter = 1 if rank == 1 else 0
    expected_backup = 1 if rank > 1 else 0

    if (
        starter != expected_starter
        or backup != expected_backup
    ):
        fail(
            f"{path} line {line_number} rank/flag mismatch "
            f"rank={rank} starter={starter} backup={backup}"
        )


def load_depth_charts(
    team_map: dict[str, str],
) -> tuple[
    dict[str, dict[str, dict[str, str]]],
    str,
    int,
]:
    qb_depth_by_team: dict[
        str,
        dict[str, dict[str, str]],
    ] = {}

    seasons: set[str] = set()
    total_depth_rows = 0

    for team_id, team_abbr in sorted(
        team_map.items(),
        key=lambda item: item[1],
    ):
        path = (
            DEPTH_CHART_DIR
            / team_abbr
            / f"{team_abbr}_depth.csv"
        )

        _, rows = read_csv(
            path,
            label=f"{team_abbr} depth chart",
            exact_columns=DEPTH_HEADERS,
        )

        team_qbs: dict[str, dict[str, str]] = {}

        for line_number, row in enumerate(
            rows,
            start=2,
        ):
            validate_depth_row(
                row,
                team_id=team_id,
                team_abbr=team_abbr,
                line_number=line_number,
                path=path,
            )

            total_depth_rows += 1
            seasons.add(clean(row.get("season")))

            if clean(row.get("position_abb")).upper() != "QB":
                continue

            player_id = clean(row.get("player_id"))

            if player_id in team_qbs:
                fail(
                    f"{path} contains duplicate QB player_id="
                    f"{player_id}"
                )

            team_qbs[player_id] = row

        if not team_qbs:
            fail(
                f"{path} contains no QB depth-chart rows"
            )

        qb_depth_by_team[team_abbr] = team_qbs

    if len(qb_depth_by_team) != 32:
        fail(
            "Expected exactly 32 loaded team depth charts; "
            f"received={len(qb_depth_by_team)}"
        )

    if len(seasons) != 1:
        fail(
            "Depth charts must contain exactly one season; "
            f"received={sorted(seasons)}"
        )

    return (
        qb_depth_by_team,
        next(iter(seasons)),
        total_depth_rows,
    )


def build_output_rows(
    roster_qbs: list[dict[str, str]],
    *,
    team_map: dict[str, str],
    qb_depth_by_team: dict[
        str,
        dict[str, dict[str, str]],
    ],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows_out: list[dict[str, str]] = []
    unmatched_qbs: list[dict[str, str]] = []

    for roster_row in roster_qbs:
        player_id = clean(roster_row.get("id"))
        qb_name = clean(roster_row.get("displayName"))
        team_id = clean(roster_row.get("team_id"))
        team_abbr = team_map[team_id]

        if team_abbr not in qb_depth_by_team:
            fail(
                "Missing loaded team depth chart "
                f"team={team_abbr}"
            )

        depth_row = qb_depth_by_team[
            team_abbr
        ].get(player_id)

        if depth_row is None:
            unmatched_qbs.append(
                {
                    "player_id": player_id,
                    "qb_name": qb_name,
                    "team_abbr": team_abbr,
                    "team_id": team_id,
                }
            )

        rows_out.append(
            {
                "sport": "football",
                "league": "nfl",
                "player_id": player_id,
                "qb_name": qb_name,
                "team_abbr": team_abbr,
                "depth_chart_rank": (
                    ""
                    if depth_row is None
                    else clean(
                        depth_row.get("depth_chart_rank")
                    )
                ),
                "starter_flag": (
                    ""
                    if depth_row is None
                    else clean(
                        depth_row.get("starter_flag")
                    )
                ),
                "backup_flag": (
                    ""
                    if depth_row is None
                    else clean(
                        depth_row.get("backup_flag")
                    )
                ),
                "injury": (
                    ""
                    if depth_row is None
                    else clean(depth_row.get("injury"))
                ),
                "position_abb": (
                    ""
                    if depth_row is None
                    else clean(
                        depth_row.get("position_abb")
                    ).upper()
                ),
                "position.id": "8",
                "team_id": team_id,
            }
        )

    return rows_out, unmatched_qbs


def validate_output_rows(
    rows_out: list[dict[str, str]],
    *,
    roster_qbs: list[dict[str, str]],
    team_map: dict[str, str],
) -> tuple[int, int]:
    if len(rows_out) != len(roster_qbs):
        fail(
            "QB map row count does not match roster QB count "
            f"output={len(rows_out)} "
            f"roster_qbs={len(roster_qbs)}"
        )

    roster_by_id = {
        clean(row.get("id")): row
        for row in roster_qbs
    }

    if len(roster_by_id) != len(roster_qbs):
        fail("Roster QB IDs are not unique")

    seen_ids: set[str] = set()
    matched = 0
    unmatched = 0

    for line_number, row in enumerate(
        rows_out,
        start=2,
    ):
        player_id = clean(row.get("player_id"))
        qb_name = clean(row.get("qb_name"))
        team_id = clean(row.get("team_id"))
        team_abbr = clean(
            row.get("team_abbr")
        ).upper()

        for field in (
            "sport",
            "league",
            "player_id",
            "qb_name",
            "team_abbr",
            "position.id",
            "team_id",
        ):
            if not clean(row.get(field)):
                fail(
                    "QB map row has blank required field "
                    f"line={line_number} field={field}"
                )

        if clean(row.get("sport")) != "football":
            fail(
                f"QB map line {line_number} has invalid sport"
            )

        if clean(row.get("league")) != "nfl":
            fail(
                f"QB map line {line_number} has invalid league"
            )

        if clean(row.get("position.id")) != "8":
            fail(
                f"QB map line {line_number} has invalid position.id"
            )

        if player_id in seen_ids:
            fail(
                "QB map contains duplicate player_id="
                f"{player_id}"
            )
        seen_ids.add(player_id)

        roster_row = roster_by_id.get(player_id)
        if roster_row is None:
            fail(
                "QB map contains player absent from roster QB "
                f"universe player_id={player_id}"
            )

        expected_name = clean(
            roster_row.get("displayName")
        )
        expected_team_id = clean(
            roster_row.get("team_id")
        )
        expected_abbr = team_map.get(
            expected_team_id,
            "",
        )

        if qb_name != expected_name:
            fail(
                "QB map name mismatch "
                f"player_id={player_id} "
                f"expected={expected_name!r} "
                f"received={qb_name!r}"
            )

        if team_id != expected_team_id:
            fail(
                "QB map team_id mismatch "
                f"player_id={player_id} "
                f"expected={expected_team_id!r} "
                f"received={team_id!r}"
            )

        if team_abbr != expected_abbr:
            fail(
                "QB map team_abbr mismatch "
                f"player_id={player_id} "
                f"expected={expected_abbr!r} "
                f"received={team_abbr!r}"
            )

        depth_values = [
            clean(row.get(field))
            for field in DEPTH_DERIVED_FIELDS
        ]

        if not any(depth_values):
            unmatched += 1
            continue

        if not all(depth_values):
            fail(
                "QB map contains partially populated depth data "
                f"player_id={player_id}"
            )

        if clean(row.get("position_abb")).upper() != "QB":
            fail(
                "Matched QB depth row has non-QB position "
                f"player_id={player_id} "
                f"position_abb={row.get('position_abb')!r}"
            )

        try:
            rank = int(
                clean(row.get("depth_chart_rank"))
            )
            starter = int(
                clean(row.get("starter_flag"))
            )
            backup = int(
                clean(row.get("backup_flag"))
            )
        except ValueError:
            fail(
                "Matched QB depth fields must contain integer "
                f"rank/flags player_id={player_id}"
            )

        if rank < 1:
            fail(
                "Matched QB has invalid depth rank "
                f"player_id={player_id} rank={rank}"
            )

        if starter != (1 if rank == 1 else 0):
            fail(
                "Matched QB starter flag disagrees with rank "
                f"player_id={player_id}"
            )

        if backup != (1 if rank > 1 else 0):
            fail(
                "Matched QB backup flag disagrees with rank "
                f"player_id={player_id}"
            )

        matched += 1

    if seen_ids != set(roster_by_id):
        missing = sorted(
            set(roster_by_id) - seen_ids
        )
        extra = sorted(
            seen_ids - set(roster_by_id)
        )
        fail(
            "QB map player universe mismatch "
            f"missing={missing[:10]} extra={extra[:10]}"
        )

    if matched + unmatched != len(rows_out):
        fail("QB map matched/unmatched accounting mismatch")

    return matched, unmatched


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
            fieldnames=OUTPUT_HEADERS,
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def normalize_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    return [
        {
            column: clean(row.get(column))
            for column in OUTPUT_HEADERS
        }
        for row in rows
    ]


def publish(
    rows_out: list[dict[str, str]],
    *,
    roster_qbs: list[dict[str, str]],
    team_map: dict[str, str],
) -> None:
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory(
        prefix=".qb_map_stage_",
        dir=OUTPUT_FILE.parent,
    ) as staging_dir:
        staged_path = (
            Path(staging_dir)
            / OUTPUT_FILE.name
        )

        write_csv(
            staged_path,
            rows_out,
        )

        staged_columns, staged_rows = read_csv(
            staged_path,
            label="staged QB map",
            exact_columns=OUTPUT_HEADERS,
        )

        if staged_columns != OUTPUT_HEADERS:
            fail(
                "Staged QB map schema changed during "
                "round-trip validation"
            )

        validate_output_rows(
            staged_rows,
            roster_qbs=roster_qbs,
            team_map=team_map,
        )

        if staged_rows != normalize_rows(rows_out):
            fail(
                "Staged QB map differs from validated "
                "in-memory projection"
            )

        os.replace(
            staged_path,
            OUTPUT_FILE,
        )


def run(
    reporter: PipelineReporter,
) -> None:
    reporter.add_input(ROSTER_MASTER)
    reporter.add_input(TEAM_MASTER)
    reporter.update_details(
        {
            "roster_master": str(ROSTER_MASTER),
            "team_master": str(TEAM_MASTER),
            "depth_chart_root": str(DEPTH_CHART_DIR),
            "output_file": str(OUTPUT_FILE),
            "expected_output_columns": len(
                OUTPUT_HEADERS
            ),
            "publication_mode": "staged_atomic_replace",
            "publication_completed": False,
            "staged_roundtrip_verified": False,
        }
    )

    team_map = load_team_master()

    roster_row_count, roster_qbs = (
        load_roster_qbs(team_map)
    )

    (
        qb_depth_by_team,
        depth_season,
        depth_row_count,
    ) = load_depth_charts(team_map)

    for team_abbr in sorted(
        qb_depth_by_team
    ):
        reporter.add_input(
            DEPTH_CHART_DIR
            / team_abbr
            / f"{team_abbr}_depth.csv"
        )

    rows_out, unmatched_qbs = build_output_rows(
        roster_qbs,
        team_map=team_map,
        qb_depth_by_team=qb_depth_by_team,
    )

    matched_count, unmatched_count = (
        validate_output_rows(
            rows_out,
            roster_qbs=roster_qbs,
            team_map=team_map,
        )
    )

    reporter.set_rows(
        rows_in=len(roster_qbs),
        rows_out=0,
    )
    reporter.update_details(
        {
            "roster_rows": roster_row_count,
            "roster_qb_rows": len(roster_qbs),
            "team_count": len(team_map),
            "depth_team_count": len(
                qb_depth_by_team
            ),
            "depth_rows": depth_row_count,
            "depth_season": depth_season,
            "matched_qbs": matched_count,
            "unmatched_qbs": unmatched_count,
        }
    )

    if unmatched_qbs:
        reporter.warning(
            "Roster QBs absent from their team's QB depth rows; "
            "depth-derived fields left blank",
            count=len(unmatched_qbs),
            examples=unmatched_qbs[:10],
        )

    publish(
        rows_out,
        roster_qbs=roster_qbs,
        team_map=team_map,
    )

    reporter.add_output(OUTPUT_FILE)
    reporter.set_rows(
        rows_in=len(roster_qbs),
        rows_out=len(rows_out),
    )
    reporter.update_details(
        {
            "rows_published": len(rows_out),
            "output_columns": len(
                OUTPUT_HEADERS
            ),
            "staged_roundtrip_verified": True,
            "publication_completed": True,
        }
    )

    print(
        f"rows={len(rows_out)} "
        f"matched_qbs={matched_count} "
        f"unmatched_qbs={unmatched_count} "
        f"depth_season={depth_season} "
        f"output={OUTPUT_FILE}"
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
                "component": "QB map builder",
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
