#!/usr/bin/env python3
"""
Validate the configured NFL daily workflow outputs against the contracts
enforced by the hardened intake producers.

This validator is intentionally scoped to the configured season/season type/
week for daily artifacts, while ESPN predictions remain a season-wide
contract.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
NFL_ROOT = SCRIPT_PATH.parents[2]
INTAKE_SCRIPTS_DIR = SCRIPTS_DIR / "00_intake"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_reporter import PipelineReporter

REPORT_ROOT = NFL_ROOT / "errors"
TEAM_MAP_PATH = NFL_ROOT / "config" / "mapping" / "team_map.csv"
ROSTER_PATH = NFL_ROOT / "data" / "master" / "roster_master.csv"
DEPTH_ROOT = NFL_ROOT / "data" / "master" / "depth_charts"
QB_MAP_PATH = NFL_ROOT / "config" / "mapping" / "qb_map_nfl.csv"
TEAM_MASTER_PATH = NFL_ROOT / "data" / "master" / "team_master.csv"

SEASON_TYPE_NUMERIC = {
    "pre": 1,
    "reg": 2,
    "post": 3,
}

_REPORTER: PipelineReporter | None = None
_TRACKED_INPUTS: set[str] = set()


class DailyValidationError(RuntimeError):
    pass


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def fail(message: str) -> None:
    print(f"VALIDATION FAILED: {message}", file=sys.stderr)
    raise DailyValidationError(message)


def passed(message: str) -> None:
    print(f"PASS: {message}")


def track_input(path: Path) -> None:
    key = str(path)
    if key in _TRACKED_INPUTS:
        return

    _TRACKED_INPUTS.add(key)

    if _REPORTER is not None:
        _REPORTER.add_input(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate configured NFL daily workflow outputs."
    )
    parser.add_argument(
        "--season",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--season-type",
        required=True,
        choices=tuple(SEASON_TYPE_NUMERIC),
    )
    parser.add_argument(
        "--week",
        required=True,
        type=int,
    )
    args = parser.parse_args()

    if args.season < 2000 or args.season > 2100:
        parser.error("--season must be between 2000 and 2100")

    if args.week < 1 or args.week > 25:
        parser.error("--week must be between 1 and 25")

    return args


def load_contract_module(
    module_name: str,
    filename: str,
) -> ModuleType:
    path = INTAKE_SCRIPTS_DIR / filename

    if not path.is_file():
        fail(f"Missing producer contract module: {path}")

    spec = importlib.util.spec_from_file_location(
        f"_daily_validation_{module_name}",
        path,
    )

    if spec is None or spec.loader is None:
        fail(f"Could not load producer contract module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv(
    path: Path,
    *,
    label: str,
    required_columns: list[str] | None = None,
    exact_columns: list[str] | None = None,
    allow_empty: bool = False,
    unique_by: list[str] | None = None,
) -> tuple[list[str], list[dict[str, str]]]:
    track_input(path)

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

    if not rows and not allow_empty:
        fail(f"{label} contains no data rows: {path}")

    if unique_by and rows:
        seen: set[tuple[str, ...]] = set()

        for line_number, row in enumerate(rows, start=2):
            key = tuple(
                clean(row.get(column))
                for column in unique_by
            )

            if not all(key):
                fail(
                    f"{label} line {line_number} has blank "
                    f"unique-key value columns={unique_by} "
                    f"key={key}"
                )

            if key in seen:
                fail(
                    f"{label} contains duplicate key "
                    f"columns={unique_by} key={key}"
                )

            seen.add(key)

    passed(f"{path} | rows={len(rows)}")
    return fieldnames, rows


def load_team_universe() -> tuple[
    dict[str, str],
    dict[str, str],
    dict[str, str],
]:
    _, rows = read_csv(
        TEAM_MAP_PATH,
        label="NFL team map",
        required_columns=[
            "sport",
            "league",
            "team_id",
            "canonical_team",
            "team_abbr",
        ],
    )

    by_id: dict[str, str] = {}
    by_abbr: dict[str, str] = {}
    by_name: dict[str, str] = {}

    for line_number, row in enumerate(rows, start=2):
        if clean(row.get("sport")).casefold() != "football":
            continue
        if clean(row.get("league")).casefold() != "nfl":
            continue

        team_id = clean(row.get("team_id"))
        team_abbr = clean(row.get("team_abbr")).upper()
        canonical_name = clean(row.get("canonical_team"))

        if not team_id or not team_abbr or not canonical_name:
            fail(
                f"{TEAM_MAP_PATH} line {line_number} has blank "
                "team_id/team_abbr/canonical_team"
            )

        old_abbr = by_id.get(team_id)
        if old_abbr and old_abbr != team_abbr:
            fail(
                f"{TEAM_MAP_PATH} has conflicting abbreviations "
                f"for team_id={team_id}"
            )

        old_id = by_abbr.get(team_abbr)
        if old_id and old_id != team_id:
            fail(
                f"{TEAM_MAP_PATH} has conflicting team IDs "
                f"for team_abbr={team_abbr}"
            )

        old_name_id = by_name.get(canonical_name)
        if old_name_id and old_name_id != team_id:
            fail(
                f"{TEAM_MAP_PATH} has conflicting team IDs "
                f"for canonical_team={canonical_name!r}"
            )

        by_id[team_id] = team_abbr
        by_abbr[team_abbr] = team_id
        by_name[canonical_name] = team_id

    if (
        len(by_id) != 32
        or len(by_abbr) != 32
        or len(by_name) != 32
    ):
        fail(
            "NFL team map must resolve exactly 32 canonical teams "
            f"ids={len(by_id)} abbrs={len(by_abbr)} "
            f"names={len(by_name)}"
        )

    passed("canonical NFL team universe | teams=32")
    return by_id, by_abbr, by_name


def validate_full_schedule(
    *,
    season: int,
    canonical_name_to_id: dict[str, str],
) -> tuple[Path, list[dict[str, str]]]:
    path = (
        NFL_ROOT
        / "00_intake"
        / "schedule"
        / f"{season}_schedule.csv"
    )

    _, rows = read_csv(
        path,
        label="season schedule",
        required_columns=[
            "season",
            "season_type",
            "week",
            "game_id",
            "game_date",
            "game_time",
            "away_team",
            "home_team",
        ],
        unique_by=["game_id"],
    )

    for line_number, row in enumerate(rows, start=2):
        if clean(row.get("season")) != str(season):
            fail(
                f"{path} line {line_number} has wrong season "
                f"{row.get('season')!r}"
            )

        game_id = clean(row.get("game_id"))
        if not game_id.isdigit():
            fail(
                f"{path} line {line_number} has invalid "
                f"game_id={game_id!r}"
            )

        season_type = clean(row.get("season_type"))
        if not season_type:
            fail(
                f"{path} line {line_number} has blank season_type"
            )

        week_text = clean(row.get("week"))
        try:
            week = int(week_text)
        except ValueError:
            fail(
                f"{path} line {line_number} has invalid "
                f"week={week_text!r}"
            )

        if week < 1 or week > 25:
            fail(
                f"{path} line {line_number} has out-of-range "
                f"week={week}"
            )

        away_team = clean(row.get("away_team"))
        home_team = clean(row.get("home_team"))

        if away_team not in canonical_name_to_id:
            fail(
                f"{path} line {line_number} has unknown "
                f"away_team={away_team!r}"
            )

        if home_team not in canonical_name_to_id:
            fail(
                f"{path} line {line_number} has unknown "
                f"home_team={home_team!r}"
            )

        if away_team == home_team:
            fail(
                f"{path} line {line_number} has identical "
                "away/home teams"
            )

    passed(
        f"season schedule semantic validation | rows={len(rows)}"
    )
    return path, rows


def validate_current_odds_and_weekly_schedule(
    *,
    args: argparse.Namespace,
    full_schedule_rows: list[dict[str, str]],
    weekly_module: ModuleType,
    odds_module: ModuleType,
) -> tuple[
    list[dict[str, str]],
    dict[str, Any],
    list[dict[str, str]],
]:
    target_schedule_rows = weekly_module.validate_schedule(
        full_schedule_rows,
        season=args.season,
        season_type=args.season_type,
        week=args.week,
    )

    (
        raw_path,
        odds_csv_path,
        raw_payload,
        odds_rows,
    ) = weekly_module.select_odds_capture(
        season=args.season,
        season_type_numeric=SEASON_TYPE_NUMERIC[
            args.season_type
        ],
        week=args.week,
    )

    track_input(raw_path)
    track_input(odds_csv_path)

    if not isinstance(raw_payload, dict):
        fail(f"Selected odds raw payload is not an object: {raw_path}")

    events = raw_payload.get("events")
    raw_odds = raw_payload.get("odds")

    if not isinstance(events, list):
        fail(f"{raw_path} events is not a list")
    if not isinstance(raw_odds, list):
        fail(f"{raw_path} odds is not a list")

    event_records: list[dict[str, str]] = []
    for index, item in enumerate(events, start=1):
        if not isinstance(item, dict):
            fail(
                f"{raw_path} events item {index} is not an object"
            )
        event_id = clean(item.get("id"))
        if not event_id:
            fail(
                f"{raw_path} events item {index} has blank id"
            )
        event_records.append({"event_id": event_id})

    odds_records: list[dict[str, str]] = []
    for index, item in enumerate(raw_odds, start=1):
        if not isinstance(item, dict):
            fail(
                f"{raw_path} odds item {index} is not an object"
            )
        event_id = clean(item.get("id"))
        if not event_id:
            fail(
                f"{raw_path} odds item {index} has blank id"
            )
        odds_records.append({"event_id": event_id})

    snapshot_id = clean(raw_payload.get("snapshot_id"))
    snapshot_fetched_at = clean(
        raw_payload.get("fetched_at")
    )

    if not snapshot_id or not snapshot_fetched_at:
        fail(
            f"{raw_path} has blank snapshot_id/fetched_at"
        )

    odds_module.validate_raw_payload(
        raw_payload,
        season=args.season,
        season_type=SEASON_TYPE_NUMERIC[
            args.season_type
        ],
        week=args.week,
        snapshot_id=snapshot_id,
        snapshot_fetched_at=snapshot_fetched_at,
        event_records=event_records,
        odds_records=odds_records,
    )

    odds_module.validate_normalized_rows(
        odds_rows,
        odds_records,
        snapshot_id,
        snapshot_fetched_at,
    )

    weekly_path = (
        NFL_ROOT
        / "00_intake"
        / "schedule"
        / "weekly"
        / f"week_{args.week}_NFL_weekly_schedule.csv"
    )

    _, weekly_rows = read_csv(
        weekly_path,
        label="configured weekly schedule",
        exact_columns=list(weekly_module.OUTPUT_COLUMNS),
    )

    raw_event_ids = {
        record["event_id"]
        for record in event_records
    }
    odds_summary_ids = {
        clean(row.get("game_id"))
        for row in odds_rows
    }

    weekly_module.validate_output_rows(
        weekly_rows,
        target_schedule_rows=target_schedule_rows,
        season=args.season,
        season_type=args.season_type,
        week=args.week,
        raw_event_ids=raw_event_ids,
        odds_summary_ids=odds_summary_ids,
    )

    passed(
        "configured current odds/raw odds/weekly schedule "
        f"| week={args.week} games={len(weekly_rows)}"
    )

    return weekly_rows, raw_payload, odds_rows


def validate_openers(
    *,
    args: argparse.Namespace,
    weekly_rows: list[dict[str, str]],
    opener_module: ModuleType,
) -> int:
    path = (
        NFL_ROOT
        / "00_intake"
        / "odds"
        / "openers"
        / f"{args.season}_NFL_openers.csv"
    )

    _, rows = read_csv(
        path,
        label="cumulative opening odds",
        exact_columns=list(opener_module.OUTPUT_COLUMNS),
    )

    opener_module.validate_opener_rows(
        rows,
        label="daily validation cumulative opener CSV",
    )

    weekly_by_id = {
        clean(row.get("game_id")): row
        for row in weekly_rows
    }
    by_game: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        game_id = clean(row.get("game_id"))
        by_game[game_id].append(row)

    for game_id, weekly_row in weekly_by_id.items():
        game_rows = by_game.get(game_id, [])
        odds_available = clean(
            weekly_row.get("odds_available")
        )

        if odds_available == "1":
            if len(game_rows) != 6:
                fail(
                    f"{path} configured game_id={game_id} "
                    f"requires six opener rows; found={len(game_rows)}"
                )
        else:
            if len(game_rows) not in {0, 6}:
                fail(
                    f"{path} unavailable-odds game_id={game_id} "
                    "must have zero preserved rows or a complete "
                    f"six-row opener set; found={len(game_rows)}"
                )

        if not game_rows:
            continue

        weekly_provider_id = clean(
            weekly_row.get("odds_provider_game_id")
        )
        provider_ids = {
            clean(row.get("odds_provider_game_id"))
            for row in game_rows
        }

        if (
            weekly_provider_id
            and provider_ids != {weekly_provider_id}
        ):
            fail(
                f"{path} game_id={game_id} opener provider IDs "
                "do not match configured weekly schedule"
            )

    passed(
        f"opening odds | rows={len(rows)} "
        f"configured_games={len(weekly_by_id)}"
    )
    return len(rows)


def validate_travel(
    *,
    args: argparse.Namespace,
    weekly_rows: list[dict[str, str]],
    travel_module: ModuleType,
) -> int:
    path = (
        NFL_ROOT
        / "data"
        / "travel"
        / f"{args.season}_week_{args.week}_travel.csv"
    )

    _, rows = read_csv(
        path,
        label="configured travel output",
        exact_columns=list(travel_module.OUTPUT_HEADERS),
    )

    travel_module.validate_output(
        rows,
        weekly_rows,
    )

    passed(
        f"travel | week={args.week} rows={len(rows)}"
    )
    return len(rows)


def validate_roster(
    *,
    canonical_team_ids: set[str],
    roster_module: ModuleType,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    _, rows = read_csv(
        ROSTER_PATH,
        label="roster master",
        exact_columns=list(roster_module.KEEP_COLUMNS),
    )

    seen_ids: set[str] = set()
    represented_teams: set[str] = set()
    roster_qbs: list[dict[str, str]] = []

    for line_number, row in enumerate(rows, start=2):
        for field in roster_module.CORE_REQUIRED_FIELDS:
            if not clean(row.get(field)):
                fail(
                    f"{ROSTER_PATH} line {line_number} has "
                    f"blank {field}"
                )

        player_id = clean(row.get("id"))
        team_id = clean(row.get("team_id"))

        if player_id in seen_ids:
            fail(
                f"{ROSTER_PATH} contains duplicate athlete "
                f"id={player_id}"
            )

        if team_id not in canonical_team_ids:
            fail(
                f"{ROSTER_PATH} line {line_number} has "
                f"unknown team_id={team_id!r}"
            )

        seen_ids.add(player_id)
        represented_teams.add(team_id)

        if clean(row.get("position.id")) == "8":
            position_abbr = clean(
                row.get("position.abbreviation")
            ).upper()

            if position_abbr and position_abbr != "QB":
                fail(
                    f"{ROSTER_PATH} player_id={player_id} has "
                    "position.id=8 with non-QB abbreviation"
                )

            roster_qbs.append(row)

    if represented_teams != canonical_team_ids:
        fail(
            f"{ROSTER_PATH} team universe does not exactly "
            "match the canonical 32 NFL team IDs"
        )

    if not roster_qbs:
        fail(f"{ROSTER_PATH} contains no position.id=8 QBs")

    passed(
        f"roster master | rows={len(rows)} qbs={len(roster_qbs)}"
    )
    return rows, roster_qbs


def validate_depth_charts(
    *,
    season: int,
    canonical_id_to_abbr: dict[str, str],
    depth_module: ModuleType,
) -> tuple[int, dict[str, list[dict[str, str]]]]:
    module_teams = depth_module.load_canonical_teams()

    if module_teams != canonical_id_to_abbr:
        fail(
            "Depth-chart producer canonical team universe "
            "does not match daily validator team universe"
        )

    expected_abbrs = set(
        canonical_id_to_abbr.values()
    )

    actual_files = sorted(
        DEPTH_ROOT.glob("*/*_depth.csv")
    )
    expected_files = {
        DEPTH_ROOT / abbr / f"{abbr}_depth.csv"
        for abbr in expected_abbrs
    }

    if set(actual_files) != expected_files:
        fail(
            "Depth-chart file universe mismatch "
            f"missing={sorted(str(p) for p in expected_files - set(actual_files))[:10]} "
            f"extra={sorted(str(p) for p in set(actual_files) - expected_files)[:10]}"
        )

    rows_by_team: dict[
        str,
        list[dict[str, str]],
    ] = {}

    for abbr in sorted(expected_abbrs):
        path = DEPTH_ROOT / abbr / f"{abbr}_depth.csv"
        _, rows = read_csv(
            path,
            label=f"{abbr} cleaned depth chart",
            exact_columns=list(depth_module.OUT_HEADER),
        )
        rows_by_team[abbr] = rows

    total_rows = depth_module.validate_output_rows(
        rows_by_team,
        canonical_teams=canonical_id_to_abbr,
        expected_season=str(season),
    )

    passed(
        f"depth charts | teams=32 rows={total_rows}"
    )
    return total_rows, rows_by_team


def validate_qb_map(
    *,
    roster_qbs: list[dict[str, str]],
    canonical_id_to_abbr: dict[str, str],
    qb_module: ModuleType,
) -> tuple[int, int, int]:
    track_input(TEAM_MASTER_PATH)
    team_master = qb_module.load_team_master()

    if team_master != canonical_id_to_abbr:
        fail(
            "team_master canonical team mapping does not match "
            "team_map canonical team mapping"
        )

    _, rows = read_csv(
        QB_MAP_PATH,
        label="NFL QB map",
        exact_columns=list(qb_module.OUTPUT_HEADERS),
    )

    matched, unmatched = qb_module.validate_output_rows(
        rows,
        roster_qbs=roster_qbs,
        team_map=team_master,
    )

    passed(
        f"QB map | rows={len(rows)} matched={matched} "
        f"unmatched={unmatched}"
    )
    return len(rows), matched, unmatched


def validate_injuries(
    *,
    args: argparse.Namespace,
    canonical_team_names: set[str],
    injuries_module: ModuleType,
) -> int:
    path = (
        NFL_ROOT
        / "00_intake"
        / "injuries"
        / f"{args.season}_injuries.csv"
    )

    _, rows = read_csv(
        path,
        label="season injury output",
        exact_columns=list(injuries_module.OUTPUT_HEADERS),
    )

    injuries_module.validate_rows(
        rows,
        requested_season=args.season,
        canonical_teams=canonical_team_names,
    )

    passed(
        f"injuries | season={args.season} rows={len(rows)}"
    )
    return len(rows)


def validate_weather(
    *,
    args: argparse.Namespace,
    weekly_rows: list[dict[str, str]],
    weather_module: ModuleType,
) -> int:
    path = (
        NFL_ROOT
        / "data"
        / "weather"
        / f"week_{args.week}_NFL_weekly_weather.csv"
    )

    _, rows = read_csv(
        path,
        label="configured weekly weather",
        exact_columns=list(weather_module.OUTPUT_HEADERS),
    )

    track_input(weather_module.STADIUM_MAP_PATH)
    stadium_lookup = weather_module.load_stadium_map()

    preserved_past_ids: set[str] = set()

    for game in weekly_rows:
        game_id = clean(game.get("game_id"))
        game_dt = weather_module.parse_game_datetime(game)

        if not weather_module.is_future_game(game_dt):
            preserved_past_ids.add(game_id)

    weather_module.validate_output_rows(
        rows,
        schedule_rows=weekly_rows,
        stadium_lookup=stadium_lookup,
        preserved_past_ids=preserved_past_ids,
    )

    passed(
        f"weather | week={args.week} rows={len(rows)} "
        f"past_rows={len(preserved_past_ids)}"
    )
    return len(rows)


def validate_predictions(
    *,
    args: argparse.Namespace,
    full_schedule_rows: list[dict[str, str]],
    canonical_name_to_id: dict[str, str],
    predictions_module: ModuleType,
) -> tuple[int, int]:
    expected_paths = predictions_module.expected_output_paths(
        full_schedule_rows
    )

    expected_names = {
        path.name
        for path in expected_paths.values()
    }

    actual_paths = sorted(
        predictions_module.OUTPUT_DIR.glob(
            f"{args.season}_*_e_predictions.csv"
        )
    )
    actual_names = {
        path.name
        for path in actual_paths
    }

    if actual_names != expected_names:
        fail(
            "ESPN prediction season file set mismatch "
            f"missing={sorted(expected_names - actual_names)} "
            f"extra={sorted(actual_names - expected_names)}"
        )

    all_rows: list[dict[str, str]] = []

    for path in sorted(
        expected_paths.values(),
        key=lambda value: value.name,
    ):
        _, rows = read_csv(
            path,
            label="ESPN prediction output",
            exact_columns=list(predictions_module.OUTPUT_HEADER),
        )
        all_rows.extend(rows)

    predictions_module.validate_generation(
        all_rows,
        schedule_rows=full_schedule_rows,
        team_map=canonical_name_to_id,
    )

    passed(
        f"ESPN predictions | files={len(expected_paths)} "
        f"rows={len(all_rows)}"
    )
    return len(expected_paths), len(all_rows)


def run(
    args: argparse.Namespace,
    reporter: PipelineReporter,
) -> None:
    global _REPORTER
    _REPORTER = reporter

    reporter.update_details(
        {
            "configured_season": args.season,
            "configured_season_type": args.season_type,
            "configured_week": args.week,
            "validation_completed": False,
        }
    )

    print(
        "Validating NFL daily workflow outputs "
        f"season={args.season} "
        f"season_type={args.season_type} "
        f"week={args.week}"
    )

    weekly_module = load_contract_module(
        "build_weekly_schedule",
        "build_weekly_schedule.py",
    )
    odds_module = load_contract_module(
        "pull_odds",
        "pull_odds.py",
    )
    opener_module = load_contract_module(
        "pull_opening_odds",
        "pull_opening_odds.py",
    )
    travel_module = load_contract_module(
        "build_travel",
        "build_travel.py",
    )
    roster_module = load_contract_module(
        "roster_cleanup",
        "roster_cleanup.py",
    )
    depth_module = load_contract_module(
        "depth_cleanup",
        "depth_cleanup.py",
    )
    qb_module = load_contract_module(
        "build_qb_map",
        "build_qb_map.py",
    )
    injuries_module = load_contract_module(
        "pull_injuries",
        "pull_injuries.py",
    )
    weather_module = load_contract_module(
        "fetch_weather",
        "fetch_weather.py",
    )
    predictions_module = load_contract_module(
        "pull_e_predictions",
        "pull_e_predictions.py",
    )

    (
        canonical_id_to_abbr,
        _canonical_abbr_to_id,
        canonical_name_to_id,
    ) = load_team_universe()

    (
        schedule_path,
        full_schedule_rows,
    ) = validate_full_schedule(
        season=args.season,
        canonical_name_to_id=canonical_name_to_id,
    )

    (
        weekly_rows,
        raw_odds_payload,
        normalized_odds_rows,
    ) = validate_current_odds_and_weekly_schedule(
        args=args,
        full_schedule_rows=full_schedule_rows,
        weekly_module=weekly_module,
        odds_module=odds_module,
    )

    opener_rows = validate_openers(
        args=args,
        weekly_rows=weekly_rows,
        opener_module=opener_module,
    )

    travel_rows = validate_travel(
        args=args,
        weekly_rows=weekly_rows,
        travel_module=travel_module,
    )

    roster_rows, roster_qbs = validate_roster(
        canonical_team_ids=set(
            canonical_id_to_abbr
        ),
        roster_module=roster_module,
    )

    depth_rows, _depth_by_team = (
        validate_depth_charts(
            season=args.season,
            canonical_id_to_abbr=canonical_id_to_abbr,
            depth_module=depth_module,
        )
    )

    (
        qb_rows,
        qb_matched,
        qb_unmatched,
    ) = validate_qb_map(
        roster_qbs=roster_qbs,
        canonical_id_to_abbr=canonical_id_to_abbr,
        qb_module=qb_module,
    )

    injury_rows = validate_injuries(
        args=args,
        canonical_team_names=set(
            canonical_name_to_id
        ),
        injuries_module=injuries_module,
    )

    weather_rows = validate_weather(
        args=args,
        weekly_rows=weekly_rows,
        weather_module=weather_module,
    )

    (
        prediction_files,
        prediction_rows,
    ) = validate_predictions(
        args=args,
        full_schedule_rows=full_schedule_rows,
        canonical_name_to_id=canonical_name_to_id,
        predictions_module=predictions_module,
    )

    raw_events = raw_odds_payload.get("events")
    raw_odds = raw_odds_payload.get("odds")
    raw_event_count = (
        len(raw_events)
        if isinstance(raw_events, list)
        else 0
    )
    raw_odds_count = (
        len(raw_odds)
        if isinstance(raw_odds, list)
        else 0
    )

    total_rows_checked = (
        len(full_schedule_rows)
        + len(normalized_odds_rows)
        + len(weekly_rows)
        + opener_rows
        + travel_rows
        + len(roster_rows)
        + depth_rows
        + qb_rows
        + injury_rows
        + weather_rows
        + prediction_rows
    )

    reporter.set_rows(
        rows_in=total_rows_checked,
        rows_out=0,
    )
    reporter.update_details(
        {
            "schedule_path": str(schedule_path),
            "season_schedule_rows": len(
                full_schedule_rows
            ),
            "configured_week_schedule_rows": len(
                weekly_rows
            ),
            "raw_odds_events": raw_event_count,
            "raw_odds_events_with_odds": raw_odds_count,
            "normalized_odds_rows": len(
                normalized_odds_rows
            ),
            "opener_rows": opener_rows,
            "travel_rows": travel_rows,
            "roster_rows": len(roster_rows),
            "roster_qbs": len(roster_qbs),
            "depth_rows": depth_rows,
            "qb_map_rows": qb_rows,
            "qb_map_matched": qb_matched,
            "qb_map_unmatched": qb_unmatched,
            "injury_rows": injury_rows,
            "weather_rows": weather_rows,
            "prediction_files": prediction_files,
            "prediction_rows": prediction_rows,
            "tracked_input_files": len(
                _TRACKED_INPUTS
            ),
            "validation_completed": True,
        }
    )

    print("DAILY VALIDATION PASSED")


def main() -> int:
    args = parse_args()

    try:
        with PipelineReporter(
            script=SCRIPT_PATH,
            stage="validation",
            report_root=REPORT_ROOT,
            pipeline="NFL",
            league="NFL",
            season=args.season,
            week=args.week,
            extra_context={
                "component": "daily validation",
                "season_type": args.season_type,
            },
        ) as reporter:
            run(args, reporter)

        return 0

    except Exception as exc:
        print(
            f"ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
