#!/usr/bin/env python3
"""
Read the hardened raw NFL depth-chart snapshot and publish one validated
15-column cleaned depth-chart CSV per canonical NFL team.
"""

from __future__ import annotations

import csv
import os
import re
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
NFL_ROOT = SCRIPT_PATH.parents[2]

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_reporter import PipelineReporter

INPUT_PATH = NFL_ROOT / "data" / "raw" / "raw_depth.csv"
OUTPUT_ROOT = NFL_ROOT / "data" / "master" / "depth_charts"
TEAM_MAP_PATH = NFL_ROOT / "config" / "mapping" / "team_map.csv"
REPORT_ROOT = NFL_ROOT / "errors"

OUT_HEADER = [
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

ATHLETE_ID_PATTERN = re.compile(
    r"^depthchart\.(\d+)\.positions\.([a-z0-9]+)\.athletes\.(\d+)\.id$"
)

RAW_REQUIRED_COLUMNS = [
    "team.id",
    "team.abbreviation",
    "team_id",
    "season.year",
]


class DepthCleanupError(RuntimeError):
    pass


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def fail(message: str) -> None:
    raise DepthCleanupError(message)


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


def load_canonical_teams() -> dict[str, str]:
    _, rows = read_csv(
        TEAM_MAP_PATH,
        label="NFL team map",
        required_columns=[
            "sport",
            "league",
            "team_id",
            "team_abbr",
        ],
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
                f"{TEAM_MAP_PATH} line {line_number} has "
                "blank team_id/team_abbr"
            )

        previous_abbr = by_id.get(team_id)
        if previous_abbr and previous_abbr != team_abbr:
            fail(
                f"{TEAM_MAP_PATH} has conflicting abbreviations "
                f"for team_id={team_id}: "
                f"{previous_abbr!r} vs {team_abbr!r}"
            )

        previous_id = by_abbr.get(team_abbr)
        if previous_id and previous_id != team_id:
            fail(
                f"{TEAM_MAP_PATH} has conflicting team IDs "
                f"for team_abbr={team_abbr}: "
                f"{previous_id!r} vs {team_id!r}"
            )

        by_id[team_id] = team_abbr
        by_abbr[team_abbr] = team_id

    if len(by_id) != 32 or len(by_abbr) != 32:
        fail(
            "Canonical NFL team map must resolve exactly 32 teams; "
            f"ids={len(by_id)} abbrs={len(by_abbr)}"
        )

    return by_id


def validate_raw_input(
    fieldnames: list[str],
    rows: list[dict[str, str]],
    *,
    canonical_teams: dict[str, str],
) -> tuple[list[str], str]:
    if len(rows) != 32:
        fail(
            "Raw depth chart must contain exactly 32 team rows; "
            f"received={len(rows)}"
        )

    athlete_id_columns = [
        column
        for column in fieldnames
        if ATHLETE_ID_PATTERN.fullmatch(column)
    ]

    if not athlete_id_columns:
        fail(
            "Raw depth chart contains no compatible athlete ID columns"
        )

    seen_team_ids: set[str] = set()
    seen_team_abbrs: set[str] = set()
    seasons: set[str] = set()

    for line_number, row in enumerate(rows, start=2):
        team_id = clean(row.get("team_id"))
        team_field_id = clean(row.get("team.id"))
        team_abbr = clean(
            row.get("team.abbreviation")
        ).upper()
        season = clean(row.get("season.year"))

        if not team_id:
            fail(
                f"Raw depth chart line {line_number} "
                "has blank team_id"
            )

        if not team_abbr:
            fail(
                f"Raw depth chart line {line_number} "
                "has blank team.abbreviation"
            )

        if not season:
            fail(
                f"Raw depth chart line {line_number} "
                "has blank season.year"
            )

        if team_id in seen_team_ids:
            fail(
                "Raw depth chart contains duplicate team row "
                f"team_id={team_id}"
            )

        if team_abbr in seen_team_abbrs:
            fail(
                "Raw depth chart contains duplicate team abbreviation "
                f"team_abbr={team_abbr}"
            )

        expected_abbr = canonical_teams.get(team_id)
        if expected_abbr is None:
            fail(
                f"Raw depth chart line {line_number} "
                f"has unknown team_id={team_id!r}"
            )

        if team_field_id != team_id:
            fail(
                "Raw depth chart team.id/team_id mismatch "
                f"line={line_number} "
                f"team.id={team_field_id!r} "
                f"team_id={team_id!r}"
            )

        if team_abbr != expected_abbr:
            fail(
                "Raw depth chart team abbreviation mismatch "
                f"line={line_number} "
                f"team_id={team_id} "
                f"expected={expected_abbr} "
                f"received={team_abbr}"
            )

        team_athlete_count = 0

        for id_column in athlete_id_columns:
            player_id = clean(row.get(id_column))
            if not player_id:
                continue

            match = ATHLETE_ID_PATTERN.fullmatch(
                id_column
            )
            if match is None:
                continue

            depth_num, position_key, athlete_idx = (
                match.groups()
            )
            prefix = (
                f"depthchart.{depth_num}."
                f"positions.{position_key}"
            )
            athlete_prefix = (
                f"{prefix}.athletes.{athlete_idx}"
            )

            companion_columns = [
                f"{athlete_prefix}.displayName",
                f"{prefix}.position.abbreviation",
                f"{prefix}.position.displayName",
            ]
            missing = [
                column
                for column in companion_columns
                if column not in fieldnames
            ]
            if missing:
                fail(
                    "Raw depth chart missing companion columns "
                    f"for {id_column}: {missing}"
                )

            if not clean(
                row.get(
                    f"{athlete_prefix}.displayName"
                )
            ):
                fail(
                    "Raw depth chart athlete missing displayName "
                    f"line={line_number} "
                    f"player_id={player_id}"
                )

            if not clean(
                row.get(
                    f"{prefix}.position.abbreviation"
                )
            ):
                fail(
                    "Raw depth chart athlete missing position "
                    f"abbreviation line={line_number} "
                    f"player_id={player_id}"
                )

            team_athlete_count += 1

        if team_athlete_count == 0:
            fail(
                "Raw depth chart team row contains no athletes "
                f"team_id={team_id} team_abbr={team_abbr}"
            )

        seen_team_ids.add(team_id)
        seen_team_abbrs.add(team_abbr)
        seasons.add(season)

    if seen_team_ids != set(canonical_teams):
        missing = sorted(
            set(canonical_teams) - seen_team_ids
        )
        extra = sorted(
            seen_team_ids - set(canonical_teams)
        )
        fail(
            "Raw depth chart team universe mismatch "
            f"missing={missing} extra={extra}"
        )

    if seen_team_abbrs != set(
        canonical_teams.values()
    ):
        fail(
            "Raw depth chart abbreviation universe does not "
            "match canonical NFL teams"
        )

    if len(seasons) != 1:
        fail(
            "Raw depth chart must contain exactly one season; "
            f"received={sorted(seasons)}"
        )

    return athlete_id_columns, next(iter(seasons))


def first_injury_status(
    row: dict[str, str],
    *,
    athlete_prefix: str,
) -> str:
    injury_index = 0

    while True:
        injury_column = (
            f"{athlete_prefix}.injuries."
            f"{injury_index}.status"
        )

        if injury_column not in row:
            break

        injury = clean(row.get(injury_column))
        if injury:
            return injury

        injury_index += 1

    return "healthy"


def build_output_rows(
    raw_rows: list[dict[str, str]],
    athlete_id_columns: list[str],
) -> dict[str, list[dict[str, str | int]]]:
    output_rows_by_team: dict[
        str,
        list[dict[str, str | int]],
    ] = {}

    for raw_row in raw_rows:
        team_abbr = clean(
            raw_row.get("team.abbreviation")
        ).upper()
        team_id = clean(raw_row.get("team_id"))
        season = clean(raw_row.get("season.year"))

        team_rows = output_rows_by_team.setdefault(
            team_abbr,
            [],
        )

        for id_column in athlete_id_columns:
            player_id = clean(
                raw_row.get(id_column)
            )
            if not player_id:
                continue

            match = ATHLETE_ID_PATTERN.fullmatch(
                id_column
            )
            if match is None:
                fail(
                    f"Internal athlete-column mismatch: {id_column}"
                )

            depth_num, position_key, athlete_idx = (
                match.groups()
            )
            prefix = (
                f"depthchart.{depth_num}."
                f"positions.{position_key}"
            )
            athlete_prefix = (
                f"{prefix}.athletes.{athlete_idx}"
            )

            rank = int(athlete_idx) + 1
            starter_flag = 1 if rank == 1 else 0
            backup_flag = 1 if rank > 1 else 0

            team_rows.append(
                {
                    "sport": "football",
                    "league": "nfl",
                    "player_id": player_id,
                    "name": clean(
                        raw_row.get(
                            f"{athlete_prefix}.displayName"
                        )
                    ),
                    "team": team_abbr,
                    "position_abb": clean(
                        raw_row.get(
                            f"{prefix}.position.abbreviation"
                        )
                    ),
                    "position": clean(
                        raw_row.get(
                            f"{prefix}.position.displayName"
                        )
                    ),
                    "injury": first_injury_status(
                        raw_row,
                        athlete_prefix=athlete_prefix,
                    ),
                    "depth_chart_rank": rank,
                    "starter_flag": starter_flag,
                    "backup_flag": backup_flag,
                    "team_id": team_id,
                    "season": season,
                    "guid": clean(
                        raw_row.get(
                            f"{athlete_prefix}.guid"
                        )
                    ),
                    "uid": clean(
                        raw_row.get(
                            f"{athlete_prefix}.uid"
                        )
                    ),
                }
            )

    return output_rows_by_team


def validate_output_rows(
    output_rows_by_team: dict[
        str,
        list[dict[str, str | int]],
    ],
    *,
    canonical_teams: dict[str, str],
    expected_season: str,
) -> int:
    canonical_abbrs = set(
        canonical_teams.values()
    )
    actual_abbrs = set(output_rows_by_team)

    if actual_abbrs != canonical_abbrs:
        fail(
            "Cleaned depth-chart team universe mismatch "
            f"missing={sorted(canonical_abbrs - actual_abbrs)} "
            f"extra={sorted(actual_abbrs - canonical_abbrs)}"
        )

    total_rows = 0

    for team_id, expected_abbr in sorted(
        canonical_teams.items(),
        key=lambda item: item[1],
    ):
        rows = output_rows_by_team.get(
            expected_abbr,
            [],
        )
        if not rows:
            fail(
                "Cleaned depth chart contains no rows "
                f"team={expected_abbr}"
            )

        seen_keys: set[
            tuple[str, str, str, str]
        ] = set()

        for row_number, row in enumerate(
            rows,
            start=2,
        ):
            for field in (
                "player_id",
                "name",
                "team",
                "position_abb",
                "team_id",
                "season",
            ):
                if not clean(row.get(field)):
                    fail(
                        "Cleaned depth-chart row has blank "
                        f"{field} team={expected_abbr} "
                        f"line={row_number}"
                    )

            if clean(row.get("sport")) != "football":
                fail(
                    f"Invalid sport for team={expected_abbr}"
                )

            if clean(row.get("league")) != "nfl":
                fail(
                    f"Invalid league for team={expected_abbr}"
                )

            if clean(row.get("team")) != expected_abbr:
                fail(
                    "Cleaned team mismatch "
                    f"expected={expected_abbr} "
                    f"received={row.get('team')!r}"
                )

            if clean(row.get("team_id")) != team_id:
                fail(
                    "Cleaned team_id mismatch "
                    f"team={expected_abbr} "
                    f"expected={team_id} "
                    f"received={row.get('team_id')!r}"
                )

            if clean(row.get("season")) != expected_season:
                fail(
                    "Cleaned season mismatch "
                    f"team={expected_abbr} "
                    f"expected={expected_season} "
                    f"received={row.get('season')!r}"
                )

            try:
                rank = int(clean(row.get("depth_chart_rank")))
                starter = int(clean(row.get("starter_flag")))
                backup = int(clean(row.get("backup_flag")))
            except ValueError:
                fail(
                    "Cleaned depth rank/flags must be integers "
                    f"team={expected_abbr} line={row_number}"
                )

            if rank < 1:
                fail(
                    "Cleaned depth_chart_rank must be >= 1 "
                    f"team={expected_abbr} line={row_number}"
                )

            expected_starter = 1 if rank == 1 else 0
            expected_backup = 1 if rank > 1 else 0

            if (
                starter != expected_starter
                or backup != expected_backup
            ):
                fail(
                    "Cleaned depth rank/flag mismatch "
                    f"team={expected_abbr} line={row_number} "
                    f"rank={rank} starter={starter} "
                    f"backup={backup}"
                )

            key = (
                clean(row.get("team")),
                clean(row.get("player_id")),
                clean(row.get("position_abb")),
                str(rank),
            )
            if key in seen_keys:
                fail(
                    "Duplicate cleaned depth-chart key "
                    f"team={expected_abbr} key={key}"
                )

            seen_keys.add(key)
            total_rows += 1

    return total_rows


def normalize_rows(
    rows: list[dict[str, str | int]],
) -> list[dict[str, str]]:
    return [
        {
            column: clean(row.get(column))
            for column in OUT_HEADER
        }
        for row in rows
    ]


def write_team_csv(
    path: Path,
    rows: list[dict[str, str | int]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUT_HEADER,
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def validate_staged_tree(
    stage_root: Path,
    output_rows_by_team: dict[
        str,
        list[dict[str, str | int]],
    ],
    *,
    canonical_teams: dict[str, str],
    expected_season: str,
) -> int:
    expected_abbrs = set(
        canonical_teams.values()
    )

    stage_dirs = {
        path.name
        for path in stage_root.iterdir()
        if path.is_dir()
    }

    if stage_dirs != expected_abbrs:
        fail(
            "Staged depth-chart directory universe mismatch "
            f"missing={sorted(expected_abbrs - stage_dirs)} "
            f"extra={sorted(stage_dirs - expected_abbrs)}"
        )

    staged_files = sorted(
        stage_root.glob("*/*_depth.csv")
    )

    if len(staged_files) != 32:
        fail(
            "Staged depth tree must contain exactly 32 CSV files; "
            f"received={len(staged_files)}"
        )

    reread_by_team: dict[
        str,
        list[dict[str, str]],
    ] = {}

    for team_abbr in sorted(expected_abbrs):
        expected_path = (
            stage_root
            / team_abbr
            / f"{team_abbr}_depth.csv"
        )

        fieldnames, rows = read_csv(
            expected_path,
            label=f"staged {team_abbr} depth CSV",
            exact_columns=OUT_HEADER,
        )

        if fieldnames != OUT_HEADER:
            fail(
                f"Staged schema mismatch team={team_abbr}"
            )

        expected_rows = normalize_rows(
            output_rows_by_team[team_abbr]
        )

        if rows != expected_rows:
            fail(
                "Staged depth-chart projection mismatch "
                f"team={team_abbr}"
            )

        reread_by_team[team_abbr] = rows

    staged_total = validate_output_rows(
        reread_by_team,
        canonical_teams=canonical_teams,
        expected_season=expected_season,
    )

    return staged_total


def publish_tree(
    output_rows_by_team: dict[
        str,
        list[dict[str, str | int]],
    ],
    *,
    canonical_teams: dict[str, str],
    expected_season: str,
) -> tuple[int, list[Path], str | None]:
    OUTPUT_ROOT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    backup_path = (
        OUTPUT_ROOT.parent
        / f".depth_charts_backup_{uuid.uuid4().hex}"
    )
    published_paths: list[Path] = []
    cleanup_warning: str | None = None

    with tempfile.TemporaryDirectory(
        prefix=".depth_charts_stage_",
        dir=OUTPUT_ROOT.parent,
    ) as temp_dir:
        stage_root = (
            Path(temp_dir)
            / "depth_charts"
        )
        stage_root.mkdir()

        for team_abbr in sorted(
            canonical_teams.values()
        ):
            staged_path = (
                stage_root
                / team_abbr
                / f"{team_abbr}_depth.csv"
            )
            write_team_csv(
                staged_path,
                output_rows_by_team[team_abbr],
            )

        staged_total = validate_staged_tree(
            stage_root,
            output_rows_by_team,
            canonical_teams=canonical_teams,
            expected_season=expected_season,
        )

        old_moved = False
        new_moved = False

        try:
            if OUTPUT_ROOT.exists():
                os.replace(
                    OUTPUT_ROOT,
                    backup_path,
                )
                old_moved = True

            os.replace(
                stage_root,
                OUTPUT_ROOT,
            )
            new_moved = True

        except Exception:
            if new_moved and OUTPUT_ROOT.exists():
                shutil.rmtree(
                    OUTPUT_ROOT,
                    ignore_errors=True,
                )

            if old_moved and backup_path.exists():
                os.replace(
                    backup_path,
                    OUTPUT_ROOT,
                )

            raise

        if backup_path.exists():
            try:
                shutil.rmtree(backup_path)
            except Exception as exc:
                cleanup_warning = (
                    "Published new depth tree but could not remove "
                    f"old backup {backup_path}: {exc}"
                )

        published_paths = [
            OUTPUT_ROOT
            / team_abbr
            / f"{team_abbr}_depth.csv"
            for team_abbr in sorted(
                canonical_teams.values()
            )
        ]

    return staged_total, published_paths, cleanup_warning


def run(
    reporter: PipelineReporter,
) -> None:
    reporter.add_input(INPUT_PATH)
    reporter.add_input(TEAM_MAP_PATH)
    reporter.update_details(
        {
            "input_path": str(INPUT_PATH),
            "output_root": str(OUTPUT_ROOT),
            "expected_team_count": 32,
            "expected_output_columns": len(OUT_HEADER),
            "publication_mode": "validated_tree_swap_with_rollback",
            "publication_completed": False,
            "staged_roundtrip_verified": False,
        }
    )

    canonical_teams = load_canonical_teams()

    fieldnames, raw_rows = read_csv(
        INPUT_PATH,
        label="raw depth CSV",
        required_columns=RAW_REQUIRED_COLUMNS,
    )

    athlete_id_columns, season = (
        validate_raw_input(
            fieldnames,
            raw_rows,
            canonical_teams=canonical_teams,
        )
    )

    output_rows_by_team = build_output_rows(
        raw_rows,
        athlete_id_columns,
    )

    projected_total = validate_output_rows(
        output_rows_by_team,
        canonical_teams=canonical_teams,
        expected_season=season,
    )

    reporter.set_rows(
        rows_in=len(raw_rows),
        rows_out=0,
    )
    reporter.update_details(
        {
            "season": season,
            "raw_team_rows": len(raw_rows),
            "raw_columns": len(fieldnames),
            "athlete_id_columns": len(
                athlete_id_columns
            ),
            "canonical_team_count": len(
                canonical_teams
            ),
            "projected_player_rows": projected_total,
        }
    )

    (
        staged_total,
        published_paths,
        cleanup_warning,
    ) = publish_tree(
        output_rows_by_team,
        canonical_teams=canonical_teams,
        expected_season=season,
    )

    if staged_total != projected_total:
        fail(
            "Published depth row count differs from "
            f"projected count staged={staged_total} "
            f"projected={projected_total}"
        )

    if cleanup_warning:
        reporter.warning(cleanup_warning)

    for path in published_paths:
        reporter.add_output(path)

    reporter.set_rows(
        rows_in=len(raw_rows),
        rows_out=projected_total,
    )
    reporter.update_details(
        {
            "rows_published": projected_total,
            "files_published": len(published_paths),
            "staged_roundtrip_verified": True,
            "publication_completed": True,
        }
    )

    for team_abbr in sorted(
        canonical_teams.values()
    ):
        print(
            f"team={team_abbr} "
            f"rows={len(output_rows_by_team[team_abbr])} "
            f"output={OUTPUT_ROOT / team_abbr / (team_abbr + '_depth.csv')}"
        )

    print(
        f"total_rows_written={projected_total} "
        f"teams={len(published_paths)} "
        f"season={season}"
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
                "component": "depth cleanup",
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
