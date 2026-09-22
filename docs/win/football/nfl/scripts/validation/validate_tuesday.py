#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
NFL_ROOT = SCRIPT_PATH.parents[2]

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_reporter import PipelineReporter

REPORT_ROOT = NFL_ROOT / "errors"

TEAM_ABBR_ALIASES = {
    "WAS": "WSH",
    "LA": "LAR",
    "JAC": "JAX",
}

_REPORTER: PipelineReporter | None = None
_TRACKED_INPUTS: set[str] = set()


class TuesdayValidationError(RuntimeError):
    pass


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def fail(message: str) -> None:
    print(f"VALIDATION FAILED: {message}", file=sys.stderr)
    raise TuesdayValidationError(message)


def passed(message: str) -> None:
    print(f"PASS: {message}")


def warning(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)
    if _REPORTER is not None:
        _REPORTER.warning(message)


def track_input(path: Path) -> None:
    key = str(path)
    if key in _TRACKED_INPUTS:
        return
    _TRACKED_INPUTS.add(key)
    if _REPORTER is not None:
        _REPORTER.add_input(path)


def normalize_team_abbr(value: Any) -> str:
    text = clean(value).upper()
    return TEAM_ABBR_ALIASES.get(text, text)


def normalize_team_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", clean(value).casefold())


def parse_int(value: Any, label: str) -> int:
    text = clean(value)
    try:
        number = int(float(text))
    except ValueError:
        fail(f"{label} is not an integer: {text!r}")
    return number


def require_finite_number(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    text = clean(value)
    if not text:
        fail(f"{label} is blank")

    try:
        number = float(text)
    except ValueError:
        fail(f"{label} is not numeric: {text!r}")

    if not math.isfinite(number):
        fail(f"{label} is not finite: {text!r}")

    if minimum is not None and number < minimum:
        fail(f"{label}={number} is below minimum {minimum}")

    if maximum is not None and number > maximum:
        fail(f"{label}={number} exceeds maximum {maximum}")

    return number


def optional_finite_number(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    if not clean(value):
        return None

    return require_finite_number(
        value,
        label,
        minimum=minimum,
        maximum=maximum,
    )


def read_csv(
    path: Path,
    required_columns: list[str],
    *,
    allow_empty: bool = False,
    unique_by: list[str] | None = None,
) -> list[dict[str, str]]:
    track_input(path)

    if not path.is_file():
        fail(f"Missing file: {path}")

    if path.stat().st_size == 0:
        fail(f"Zero-byte file: {path}")

    try:
        with path.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames or []
            rows = list(reader)
    except Exception as exc:
        fail(
            f"Could not read {path}: "
            f"{type(exc).__name__}: {exc}"
        )

    missing = [
        column
        for column in required_columns
        if column not in columns
    ]

    if missing:
        fail(f"{path} is missing columns: {missing}")

    if not rows and not allow_empty:
        fail(f"{path} has a header but no data rows")

    if unique_by and rows:
        seen: set[tuple[str, ...]] = set()
        duplicates: list[tuple[str, ...]] = []

        for line_number, row in enumerate(rows, start=2):
            key = tuple(
                clean(row.get(column))
                for column in unique_by
            )

            if not all(key):
                fail(
                    f"{path} line {line_number} has a blank "
                    f"unique-key value for columns "
                    f"{unique_by}: {key}"
                )

            if key in seen:
                duplicates.append(key)

            seen.add(key)

        if duplicates:
            fail(
                f"{path} contains duplicate keys for "
                f"{unique_by}. Examples: {duplicates[:5]}"
            )

    passed(f"{path} | rows={len(rows)}")
    return rows


def load_team_universe() -> tuple[
    set[str],
    set[str],
    dict[str, str],
    dict[str, str],
]:
    path = NFL_ROOT / "config/mapping/team_map.csv"

    rows = read_csv(
        path,
        [
            "sport",
            "league",
            "team_id",
            "canonical_team",
            "team_abbr",
        ],
    )

    team_id_to_abbr: dict[str, str] = {}
    name_to_abbr: dict[str, str] = {}

    for line_number, row in enumerate(rows, start=2):
        sport = clean(row.get("sport")).casefold()
        league = clean(row.get("league")).casefold()

        if sport not in {"", "football"}:
            continue
        if league not in {"", "nfl"}:
            continue

        team_id = clean(row.get("team_id"))
        canonical = clean(row.get("canonical_team"))
        abbr = normalize_team_abbr(row.get("team_abbr"))

        if not team_id or not canonical or not abbr:
            continue

        previous_abbr = team_id_to_abbr.get(team_id)
        if previous_abbr is not None and previous_abbr != abbr:
            fail(
                f"{path} line {line_number} maps team_id={team_id!r} "
                f"to conflicting abbreviations "
                f"{previous_abbr!r} and {abbr!r}"
            )

        team_id_to_abbr[team_id] = abbr

        name_key = normalize_team_name(canonical)
        previous_name_abbr = name_to_abbr.get(name_key)
        if (
            previous_name_abbr is not None
            and previous_name_abbr != abbr
        ):
            fail(
                f"{path} maps canonical team {canonical!r} "
                "to multiple abbreviations"
            )
        name_to_abbr[name_key] = abbr

    valid_ids = set(team_id_to_abbr)
    valid_abbrs = set(team_id_to_abbr.values())

    if len(valid_ids) != 32 or len(valid_abbrs) != 32:
        fail(
            f"{path} must resolve exactly 32 NFL team IDs "
            f"and abbreviations; ids={len(valid_ids)} "
            f"abbrs={len(valid_abbrs)}"
        )

    return (
        valid_ids,
        valid_abbrs,
        team_id_to_abbr,
        name_to_abbr,
    )


def resolve_team_abbr(
    value: Any,
    *,
    valid_abbrs: set[str],
    name_to_abbr: dict[str, str],
    label: str,
) -> str:
    text = clean(value)
    if not text:
        fail(f"{label} is blank")

    direct = normalize_team_abbr(text)
    if direct in valid_abbrs:
        return direct

    key = normalize_team_name(text)
    if key in name_to_abbr:
        return name_to_abbr[key]

    fail(f"{label} could not be mapped to an NFL team: {text!r}")


def schedule_game_identity(
    row: dict[str, str],
    name_to_abbr: dict[str, str],
    valid_abbrs: set[str],
) -> tuple[int, int, str, str]:
    game_id = clean(row.get("game_id"))
    season = parse_int(
        row.get("season"),
        f"schedule game_id={game_id} season",
    )
    week = parse_int(
        row.get("week"),
        f"schedule game_id={game_id} week",
    )

    away_abbr = resolve_team_abbr(
        row.get("away_team"),
        valid_abbrs=valid_abbrs,
        name_to_abbr=name_to_abbr,
        label=f"schedule game_id={game_id} away_team",
    )
    home_abbr = resolve_team_abbr(
        row.get("home_team"),
        valid_abbrs=valid_abbrs,
        name_to_abbr=name_to_abbr,
        label=f"schedule game_id={game_id} home_team",
    )

    if away_abbr == home_abbr:
        fail(
            f"Schedule game_id={game_id} has identical "
            f"away/home team {away_abbr}"
        )

    return season, week, away_abbr, home_abbr


def pbp_game_identity(game_id: Any) -> tuple[int, int, str, str]:
    text = clean(game_id)

    match = re.fullmatch(
        r"(\d{4})_(\d{1,2})_([A-Za-z0-9]+)_([A-Za-z0-9]+)",
        text,
    )

    if not match:
        fail(f"Unrecognized nflverse PBP game_id format: {text!r}")

    return (
        int(match.group(1)),
        int(match.group(2)),
        normalize_team_abbr(match.group(3)),
        normalize_team_abbr(match.group(4)),
    )


def validate_schedule(
    rows: list[dict[str, str]],
    *,
    season: int,
    valid_abbrs: set[str],
    name_to_abbr: dict[str, str],
) -> tuple[
    dict[str, dict[str, str]],
    dict[str, tuple[int, int, str, str]],
    set[tuple[int, int, str, str]],
    set[int],
]:
    schedule_by_id: dict[str, dict[str, str]] = {}
    identity_by_id: dict[str, tuple[int, int, str, str]] = {}
    identities: set[tuple[int, int, str, str]] = set()
    weeks: set[int] = set()

    for line_number, row in enumerate(rows, start=2):
        game_id = clean(row.get("game_id"))

        row_season = parse_int(
            row.get("season"),
            f"schedule line {line_number} season",
        )
        week = parse_int(
            row.get("week"),
            f"schedule game_id={game_id} week",
        )

        if row_season != season:
            fail(
                f"Schedule game_id={game_id} has season={row_season}; "
                f"expected {season}"
            )

        if week < 1 or week > 25:
            fail(
                f"Schedule game_id={game_id} has invalid week={week}"
            )

        if not clean(row.get("season_type")):
            fail(
                f"Schedule game_id={game_id} has blank season_type"
            )

        for field in (
            "game_date",
            "game_time",
            "away_team",
            "home_team",
        ):
            if not clean(row.get(field)):
                fail(
                    f"Schedule game_id={game_id} has blank {field}"
                )

        identity = schedule_game_identity(
            row,
            name_to_abbr,
            valid_abbrs,
        )

        if identity in identities:
            fail(
                "Schedule contains duplicate season/week/team "
                f"identity: {identity}"
            )

        schedule_by_id[game_id] = row
        identity_by_id[game_id] = identity
        identities.add(identity)
        weeks.add(week)

    return schedule_by_id, identity_by_id, identities, weeks


def validate_result_rows(
    rows: list[dict[str, str]],
    *,
    path: Path,
    season: int,
    schedule_by_id: dict[str, dict[str, str]],
) -> set[str]:
    completed: set[str] = set()

    for line_number, row in enumerate(rows, start=2):
        game_id = clean(row.get("game_id"))
        scheduled = schedule_by_id.get(game_id)

        if scheduled is None:
            fail(
                f"{path} line {line_number} game_id={game_id} "
                "is absent from the season schedule"
            )

        comparisons = {
            "season": str(season),
            "season_type": clean(scheduled.get("season_type")),
            "week": clean(scheduled.get("week")),
            "away_team": clean(scheduled.get("away_team")),
            "home_team": clean(scheduled.get("home_team")),
        }

        for field, expected in comparisons.items():
            actual = clean(row.get(field))
            if field in {"away_team", "home_team"}:
                if normalize_team_name(actual) != normalize_team_name(expected):
                    fail(
                        f"{path} line {line_number} game_id={game_id} "
                        f"{field}={actual!r}; expected {expected!r}"
                    )
            elif actual != expected:
                fail(
                    f"{path} line {line_number} game_id={game_id} "
                    f"{field}={actual!r}; expected {expected!r}"
                )

        status = clean(row.get("status")).casefold()

        if "final" in status or "completed" in status:
            completed.add(game_id)

            away_score = require_finite_number(
                row.get("away_score"),
                f"{path} game_id={game_id} away_score",
                minimum=0,
            )
            home_score = require_finite_number(
                row.get("home_score"),
                f"{path} game_id={game_id} home_score",
                minimum=0,
            )

            if not away_score.is_integer() or not home_score.is_integer():
                fail(
                    f"{path} game_id={game_id} has "
                    "non-integer completed score"
                )

    return completed


def read_pbp(
    path: Path,
    required_columns: list[str],
    *,
    require_rows: bool,
    season: int,
    schedule_identities: set[tuple[int, int, str, str]],
    completed_identities: set[tuple[int, int, str, str]],
) -> pd.DataFrame:
    track_input(path)

    if not path.is_file():
        if require_rows:
            fail(f"Missing file: {path}")
        passed(f"{path} not required before completed games exist")
        return pd.DataFrame()

    if path.stat().st_size == 0:
        if require_rows:
            fail(f"Zero-byte file: {path}")
        passed(f"{path} is empty and allowed before completed games exist")
        return pd.DataFrame()

    try:
        frame = pd.read_csv(
            path,
            compression="gzip",
            low_memory=False,
        )
    except pd.errors.EmptyDataError:
        if require_rows:
            fail(f"{path} contains no PBP data")
        passed(
            f"{path} contains no PBP data and is allowed "
            "before completed games exist"
        )
        return pd.DataFrame()
    except Exception as exc:
        fail(f"Could not read compressed PBP file {path}: {exc}")

    if frame.empty:
        if require_rows:
            fail(
                f"{path} contains no PBP rows even though "
                "completed games exist"
            )
        passed(
            f"{path} has no PBP rows and is allowed "
            "before completed games exist"
        )
        return frame

    missing = [
        column
        for column in required_columns
        if column not in frame.columns
    ]
    if missing:
        fail(f"{path} is missing columns: {missing}")

    game_text = frame["game_id"].fillna("").astype(str).str.strip()
    play_text = frame["play_id"].fillna("").astype(str).str.strip()

    blank_mask = game_text.eq("") | play_text.eq("")
    if blank_mask.any():
        examples = (
            frame.loc[
                blank_mask,
                ["game_id", "play_id"],
            ]
            .head(5)
            .to_dict("records")
        )
        fail(
            f"{path} has blank game_id/play_id values: {examples}"
        )

    keys = pd.DataFrame({
        "game_id": game_text,
        "play_id": play_text,
    })
    if keys.duplicated().any():
        examples = (
            keys[keys.duplicated(keep=False)]
            .head(5)
            .to_dict("records")
        )
        fail(
            f"{path} has duplicate game_id/play_id rows: "
            f"{examples}"
        )

    pbp_game_ids = sorted(set(game_text.tolist()))
    identity_by_game_id = {
        game_id: pbp_game_identity(game_id)
        for game_id in pbp_game_ids
    }

    invalid_teams = [
        game_id
        for game_id, identity in identity_by_game_id.items()
        if identity not in schedule_identities
    ]
    if invalid_teams:
        fail(
            f"{path} contains games absent from the schedule: "
            f"{invalid_teams[:5]}"
        )

    season_numeric = pd.to_numeric(
        frame["season"],
        errors="coerce",
    )
    week_numeric = pd.to_numeric(
        frame["week"],
        errors="coerce",
    )

    if season_numeric.isna().any() or week_numeric.isna().any():
        fail(f"{path} contains invalid season/week values")

    expected_season = game_text.map(
        lambda game_id: identity_by_game_id[game_id][0]
    )
    expected_week = game_text.map(
        lambda game_id: identity_by_game_id[game_id][1]
    )

    if not (season_numeric == expected_season).all():
        fail(
            f"{path} season column does not match nflverse "
            "game_id identities"
        )

    if not (week_numeric == expected_week).all():
        fail(
            f"{path} week column does not match nflverse "
            "game_id identities"
        )

    if not (season_numeric == season).all():
        fail(
            f"{path} contains PBP rows outside requested "
            f"season {season}"
        )

    pbp_identities = set(identity_by_game_id.values())
    missing_completed = completed_identities - pbp_identities

    if missing_completed:
        fail(
            f"{path} is missing completed schedule identities: "
            f"{sorted(missing_completed)[:5]}"
        )

    passed(
        f"{path} | rows={len(frame)} "
        f"columns={len(frame.columns)}"
    )
    return frame


def validate_team_stats(
    rows: list[dict[str, str]],
    *,
    path: Path,
    season: int,
    schedule_weeks: set[int],
    valid_abbrs: set[str],
    name_to_abbr: dict[str, str],
) -> None:
    metric_columns = [
        "off_epa_per_play",
        "def_epa_per_play",
        "off_success_rate",
        "def_success_rate",
        "yards_per_play",
        "yards_per_play_allowed",
        "points_per_drive",
        "points_per_drive_allowed",
        "red_zone_td_rate",
        "red_zone_td_rate_allowed",
        "early_down_epa",
        "third_down_conversion_rate",
    ]
    rate_columns = {
        "off_success_rate",
        "def_success_rate",
        "red_zone_td_rate",
        "red_zone_td_rate_allowed",
        "third_down_conversion_rate",
    }

    for line_number, row in enumerate(rows, start=2):
        row_season = parse_int(
            row.get("season"),
            f"{path} line {line_number} season",
        )
        week = parse_int(
            row.get("week"),
            f"{path} line {line_number} week",
        )

        if row_season != season:
            fail(
                f"{path} line {line_number} has season={row_season}; "
                f"expected {season}"
            )
        if week not in schedule_weeks:
            fail(
                f"{path} line {line_number} has week={week} "
                "absent from season schedule"
            )

        resolve_team_abbr(
            row.get("team"),
            valid_abbrs=valid_abbrs,
            name_to_abbr=name_to_abbr,
            label=f"{path} line {line_number} team",
        )

        for column in metric_columns:
            optional_finite_number(
                row.get(column),
                f"{path} line {line_number} {column}",
                minimum=(0.0 if column in rate_columns else None),
                maximum=(1.0 if column in rate_columns else None),
            )


def validate_qb_stats(
    rows: list[dict[str, str]],
    *,
    path: Path,
    season: int,
    schedule_weeks: set[int],
    valid_abbrs: set[str],
    name_to_abbr: dict[str, str],
    completed_games_exist: bool,
) -> None:
    optional_numeric_columns: list[str] = []
    rate_columns = {
        "sack_rate",
        "interception_rate",
        "fumble_rate",
    }

    if completed_games_exist:
        optional_numeric_columns.extend([
            "epa_per_play",
            "cpoe",
            "air_yards",
            "sack_rate",
            "interception_rate",
            "fumble_rate",
        ])

    for line_number, row in enumerate(rows, start=2):
        row_season = parse_int(
            row.get("season"),
            f"{path} line {line_number} season",
        )
        week = parse_int(
            row.get("week"),
            f"{path} line {line_number} week",
        )

        if row_season != season:
            fail(
                f"{path} line {line_number} has season={row_season}; "
                f"expected {season}"
            )
        if week not in schedule_weeks:
            fail(
                f"{path} line {line_number} has week={week} "
                "absent from season schedule"
            )

        resolve_team_abbr(
            row.get("team"),
            valid_abbrs=valid_abbrs,
            name_to_abbr=name_to_abbr,
            label=f"{path} line {line_number} team",
        )

        if not clean(row.get("player_id")):
            fail(f"{path} line {line_number} has blank player_id")
        if not clean(row.get("qb_name")):
            fail(f"{path} line {line_number} has blank qb_name")

        require_finite_number(
            row.get("dropbacks"),
            f"{path} line {line_number} dropbacks",
        )

        for column in optional_numeric_columns:
            optional_finite_number(
                row.get(column),
                f"{path} line {line_number} {column}",
                minimum=(0.0 if column in rate_columns else None),
                maximum=(1.0 if column in rate_columns else None),
            )


def validate_league_master(
    rows: list[dict[str, str]],
    *,
    path: Path,
    season: int,
    valid_team_ids: set[str],
    canonical_id_to_abbr: dict[str, str],
) -> dict[str, str]:
    if len(rows) != 32:
        fail(
            f"{path} must contain exactly 32 teams; "
            f"found {len(rows)}"
        )

    observed: dict[str, str] = {}

    for line_number, row in enumerate(rows, start=2):
        team_id = clean(row.get("team_id"))
        abbr = normalize_team_abbr(row.get("team_abbr"))

        if clean(row.get("season")) != str(season):
            fail(
                f"{path} line {line_number} has wrong season"
            )

        if team_id not in valid_team_ids:
            fail(
                f"{path} line {line_number} has unknown "
                f"team_id={team_id!r}"
            )

        expected_abbr = canonical_id_to_abbr[team_id]
        if abbr != expected_abbr:
            fail(
                f"{path} line {line_number} team_id={team_id} "
                f"has team_abbr={abbr!r}; "
                f"expected {expected_abbr!r}"
            )

        for field in (
            "conference",
            "conference_abbr",
            "division",
            "division_abbr",
        ):
            if not clean(row.get(field)):
                fail(
                    f"{path} line {line_number} has blank {field}"
                )

        observed[team_id] = abbr

    if set(observed) != valid_team_ids:
        fail(f"{path} does not exactly cover the 32 NFL teams")

    return observed


def validate_standings(
    rows: list[dict[str, str]],
    *,
    path: Path,
    season: int,
    league_id_to_abbr: dict[str, str],
) -> None:
    observed_team_ids: set[str] = set()

    for line_number, row in enumerate(rows, start=2):
        team_id = clean(row.get("team_id"))
        abbr = normalize_team_abbr(row.get("team_abbr"))
        standings_type = clean(row.get("standings_type"))
        stat_name = clean(row.get("stat_name"))

        if clean(row.get("season")) != str(season):
            fail(
                f"{path} line {line_number} has wrong season"
            )

        expected_abbr = league_id_to_abbr.get(team_id)
        if expected_abbr is None:
            fail(
                f"{path} line {line_number} has unknown "
                f"team_id={team_id!r}"
            )

        if abbr != expected_abbr:
            fail(
                f"{path} line {line_number} has abbreviation "
                f"mismatch for team_id={team_id}"
            )

        for field in (
            "conference",
            "division",
            "standings_type",
            "stat_name",
        ):
            if not clean(row.get(field)):
                fail(
                    f"{path} line {line_number} has blank {field}"
                )

        observed_team_ids.add(team_id)

    if observed_team_ids != set(league_id_to_abbr):
        fail(
            f"{path} does not contain standings rows for "
            "all 32 NFL teams"
        )


def validate_coaches(
    rows: list[dict[str, str]],
    *,
    path: Path,
    valid_team_ids: set[str],
) -> None:
    if len(rows) != 32:
        fail(
            f"{path} must contain exactly 32 head coaches; "
            f"found {len(rows)}"
        )

    observed = {clean(row.get("team_id")) for row in rows}

    if observed != valid_team_ids:
        fail(
            f"{path} team IDs do not exactly match "
            "the canonical NFL team universe"
        )

    for line_number, row in enumerate(rows, start=2):
        for field in ("name", "team", "team_id", "id", "uid"):
            if not clean(row.get(field)):
                fail(
                    f"{path} line {line_number} has blank {field}"
                )


def qbr_filename_week(path: Path) -> int:
    patterns = (
        r"qbr_week(\d+)\.csv",
        r"qbr_playoffs_week(\d+)\.csv",
        r"qbr_type\d+_week(\d+)\.csv",
    )

    for pattern in patterns:
        match = re.fullmatch(pattern, path.name)
        if match:
            return int(match.group(1))

    fail(f"Unrecognized QBR filename: {path.name}")


def validate_qbr(
    rows: list[dict[str, str]],
    *,
    path: Path,
    season: int,
    valid_team_ids: set[str],
) -> None:
    expected_week = qbr_filename_week(path)

    for line_number, row in enumerate(rows, start=2):
        if clean(row.get("season")) != str(season):
            fail(
                f"{path} line {line_number} has wrong season"
            )

        row_week = parse_int(
            row.get("week"),
            f"{path} line {line_number} week",
        )
        if row_week != expected_week:
            fail(
                f"{path} line {line_number} has week={row_week}; "
                f"filename requires week={expected_week}"
            )

        if not clean(row.get("athlete_id")):
            fail(
                f"{path} line {line_number} has blank athlete_id"
            )

        team_id = clean(row.get("team_id"))
        if team_id not in valid_team_ids:
            fail(
                f"{path} line {line_number} has unknown "
                f"team_id={team_id!r}"
            )


def validate_fpi(
    rows: list[dict[str, str]],
    *,
    path: Path,
    season: int,
    valid_team_ids: set[str],
) -> None:
    if len(rows) != 32:
        fail(
            f"{path} must contain exactly 32 teams; "
            f"found {len(rows)}"
        )

    observed: set[str] = set()

    for line_number, row in enumerate(rows, start=2):
        if clean(row.get("season")) != str(season):
            fail(
                f"{path} line {line_number} has wrong season"
            )

        team_id = clean(row.get("team_id"))
        if team_id not in valid_team_ids:
            fail(
                f"{path} line {line_number} has unknown "
                f"team_id={team_id!r}"
            )

        if not clean(row.get("lastUpdated")):
            fail(
                f"{path} line {line_number} has blank lastUpdated"
            )

        observed.add(team_id)

    if observed != valid_team_ids:
        fail(
            f"{path} team IDs do not exactly match "
            "the canonical NFL team universe"
        )


def validate_leaders(
    rows: list[dict[str, str]],
    *,
    path: Path,
    season: int,
    valid_team_ids: set[str],
) -> None:
    seen: set[tuple[str, str, str]] = set()

    for line_number, row in enumerate(rows, start=2):
        row_season = clean(row.get("season"))
        category = clean(row.get("category"))
        rank = clean(row.get("rank"))
        athlete_id = clean(row.get("athlete_id"))
        team_id = clean(row.get("team_id"))

        if row_season != str(season):
            fail(
                f"{path} line {line_number} has wrong season"
            )

        for field, value in (
            ("category", category),
            ("rank", rank),
            ("athlete_id", athlete_id),
            ("team_id", team_id),
        ):
            if not value:
                fail(
                    f"{path} line {line_number} has blank {field}"
                )

        if team_id not in valid_team_ids:
            fail(
                f"{path} line {line_number} has unknown "
                f"team_id={team_id!r}"
            )

        key = (row_season, category, rank)
        if key in seen:
            fail(f"{path} contains duplicate leader key: {key}")
        seen.add(key)


def validate_market_futures(
    rows: list[dict[str, str]],
    *,
    path: Path,
    season: int,
    valid_team_ids: set[str],
) -> None:
    seen: set[tuple[str, str, str, str, str]] = set()

    for line_number, row in enumerate(rows, start=2):
        row_season = clean(row.get("season"))
        future_id = clean(row.get("future_id"))
        future_name = clean(row.get("future_name"))
        provider_id = clean(row.get("provider_id"))
        provider_name = clean(row.get("provider_name"))
        athlete_id = clean(row.get("athlete_id"))
        team_id = clean(row.get("team_id"))
        value = clean(row.get("value"))

        if row_season != str(season):
            fail(
                f"{path} line {line_number} has wrong season"
            )

        for field, field_value in (
            ("future_id", future_id),
            ("future_name", future_name),
            ("provider_id", provider_id),
            ("provider_name", provider_name),
            ("value", value),
        ):
            if not field_value:
                fail(
                    f"{path} line {line_number} has blank {field}"
                )

        if bool(athlete_id) == bool(team_id):
            fail(
                f"{path} line {line_number} must contain "
                "exactly one of athlete_id or team_id"
            )

        if team_id and team_id not in valid_team_ids:
            fail(
                f"{path} line {line_number} has unknown "
                f"team_id={team_id!r}"
            )

        key = (
            row_season,
            future_id,
            provider_id,
            athlete_id,
            team_id,
        )

        if key in seen:
            fail(
                f"{path} contains duplicate market-futures "
                f"key: {key}"
            )

        seen.add(key)


def validate_weekly_and_travel(
    *,
    season: int,
    schedule_by_id: dict[str, dict[str, str]],
) -> int:
    weekly_files = sorted(
        (
            NFL_ROOT
            / "00_intake/schedule/weekly"
        ).glob("week_*_NFL_weekly_schedule.csv")
    )

    current_weekly_files = 0

    for schedule_week_path in weekly_files:
        match = re.fullmatch(
            r"week_(\d+)_NFL_weekly_schedule\.csv",
            schedule_week_path.name,
        )
        if not match:
            continue

        filename_week = int(match.group(1))

        week_rows = read_csv(
            schedule_week_path,
            [
                "season",
                "week",
                "game_id",
                "away_team",
                "home_team",
                "neutral_site",
            ],
            unique_by=["game_id"],
        )

        file_seasons = {
            clean(row.get("season"))
            for row in week_rows
        }

        if file_seasons == {str(season)}:
            pass
        elif str(season) in file_seasons:
            fail(
                f"{schedule_week_path} mixes requested season "
                f"{season} with other seasons: "
                f"{sorted(file_seasons)}"
            )
        else:
            continue

        current_weekly_files += 1
        weekly_by_id: dict[str, dict[str, str]] = {}

        for line_number, row in enumerate(week_rows, start=2):
            row_week = parse_int(
                row.get("week"),
                f"{schedule_week_path} line {line_number} week",
            )

            if row_week != filename_week:
                fail(
                    f"{schedule_week_path} line {line_number} "
                    f"has week={row_week}; filename requires "
                    f"week={filename_week}"
                )

            game_id = clean(row.get("game_id"))
            canonical = schedule_by_id.get(game_id)
            if canonical is None:
                fail(
                    f"{schedule_week_path} game_id={game_id} "
                    "is absent from season schedule"
                )

            if clean(canonical.get("week")) != str(filename_week):
                fail(
                    f"{schedule_week_path} game_id={game_id} "
                    "belongs to a different canonical week"
                )

            for field in ("away_team", "home_team"):
                if normalize_team_name(row.get(field)) != normalize_team_name(
                    canonical.get(field)
                ):
                    fail(
                        f"{schedule_week_path} game_id={game_id} "
                        f"{field} does not match season schedule"
                    )

            if clean(row.get("neutral_site")) != clean(
                canonical.get("neutral_site")
            ):
                fail(
                    f"{schedule_week_path} game_id={game_id} "
                    "neutral_site does not match season schedule"
                )

            weekly_by_id[game_id] = row

        travel_path = (
            NFL_ROOT
            / "data/travel"
            / f"{season}_week_{filename_week}_travel.csv"
        )

        travel_rows = read_csv(
            travel_path,
            [
                "game_id",
                "away_team",
                "home_team",
                "away_lat",
                "away_lon",
                "home_lat",
                "home_lon",
                "miles_traveled",
                "time_zones_crossed",
                "east_to_west",
                "west_to_east",
                "international_flag",
                "neutral_site_flag",
            ],
            unique_by=["game_id"],
        )

        travel_ids: set[str] = set()

        for line_number, row in enumerate(travel_rows, start=2):
            game_id = clean(row.get("game_id"))
            schedule_row = weekly_by_id.get(game_id)

            if schedule_row is None:
                fail(
                    f"{travel_path} line {line_number} "
                    f"game_id={game_id} absent from weekly schedule"
                )

            travel_ids.add(game_id)

            for field in ("away_team", "home_team"):
                if normalize_team_name(row.get(field)) != normalize_team_name(
                    schedule_row.get(field)
                ):
                    fail(
                        f"{travel_path} game_id={game_id} "
                        f"{field} does not match weekly schedule"
                    )

            if clean(row.get("neutral_site_flag")) != clean(
                schedule_row.get("neutral_site")
            ):
                fail(
                    f"{travel_path} game_id={game_id} "
                    "neutral_site_flag does not match weekly schedule"
                )

            require_finite_number(
                row.get("away_lat"),
                f"{travel_path} game_id={game_id} away_lat",
                minimum=-90,
                maximum=90,
            )
            require_finite_number(
                row.get("away_lon"),
                f"{travel_path} game_id={game_id} away_lon",
                minimum=-180,
                maximum=180,
            )
            require_finite_number(
                row.get("home_lat"),
                f"{travel_path} game_id={game_id} home_lat",
                minimum=-90,
                maximum=90,
            )
            require_finite_number(
                row.get("home_lon"),
                f"{travel_path} game_id={game_id} home_lon",
                minimum=-180,
                maximum=180,
            )
            require_finite_number(
                row.get("miles_traveled"),
                f"{travel_path} game_id={game_id} miles_traveled",
                minimum=0,
            )
            require_finite_number(
                row.get("time_zones_crossed"),
                f"{travel_path} game_id={game_id} time_zones_crossed",
                minimum=0,
            )

            east = clean(row.get("east_to_west"))
            west = clean(row.get("west_to_east"))
            international = clean(row.get("international_flag"))
            neutral = clean(row.get("neutral_site_flag"))

            if east not in {"0", "1"} or west not in {"0", "1"}:
                fail(
                    f"{travel_path} game_id={game_id} has "
                    "invalid direction flags"
                )
            if east == "1" and west == "1":
                fail(
                    f"{travel_path} game_id={game_id} has both "
                    "direction flags set"
                )
            if international not in {"0", "1"}:
                fail(
                    f"{travel_path} game_id={game_id} has "
                    "invalid international_flag"
                )
            if neutral not in {"0", "1"}:
                fail(
                    f"{travel_path} game_id={game_id} has "
                    "invalid neutral_site_flag"
                )

        if travel_ids != set(weekly_by_id):
            fail(
                f"{travel_path} game IDs do not exactly match "
                f"week {filename_week} schedule game IDs"
            )

    if current_weekly_files == 0:
        fail(
            f"No weekly schedule files found for season {season}"
        )

    return current_weekly_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--season",
        required=True,
        type=int,
    )
    args = parser.parse_args()

    if args.season < 2000 or args.season > 2100:
        parser.error("--season must be between 2000 and 2100")

    return args


def run(
    args: argparse.Namespace,
    reporter: PipelineReporter,
) -> None:
    global _REPORTER
    _REPORTER = reporter

    season = args.season

    reporter.update_details({
        "validated_season": season,
        "validation_completed": False,
    })

    print(
        f"Validating NFL Tuesday workflow outputs "
        f"for season {season}"
    )

    (
        valid_team_ids,
        valid_abbrs,
        canonical_id_to_abbr,
        name_to_abbr,
    ) = load_team_universe()

    schedule_path = (
        NFL_ROOT
        / "00_intake/schedule"
        / f"{season}_schedule.csv"
    )

    schedule_rows = read_csv(
        schedule_path,
        [
            "season",
            "season_type",
            "week",
            "game_id",
            "game_date",
            "game_time",
            "away_team",
            "home_team",
            "neutral_site",
            "stadium",
            "game_timezone",
        ],
        unique_by=["game_id"],
    )

    (
        schedule_by_id,
        schedule_identity_by_game_id,
        schedule_identities,
        schedule_weeks,
    ) = validate_schedule(
        schedule_rows,
        season=season,
        valid_abbrs=valid_abbrs,
        name_to_abbr=name_to_abbr,
    )

    reporter.update_details({
        "schedule_rows": len(schedule_rows),
        "schedule_weeks": sorted(schedule_weeks),
    })

    result_files = sorted(
        (
            NFL_ROOT
            / "04_final_results/results"
        ).glob(f"{season}_*.csv")
    )

    if not result_files:
        warning(
            f"No final-score files found for season {season}. "
            "Continuing because final-score data may not yet "
            "be available."
        )

    result_game_ids: set[str] = set()
    completed_game_ids: set[str] = set()
    result_rows_count = 0

    for path in result_files:
        rows = read_csv(
            path,
            [
                "season",
                "season_type",
                "week",
                "game_id",
                "game_date",
                "game_time",
                "away_team",
                "home_team",
                "away_score",
                "home_score",
                "status",
            ],
            unique_by=["game_id"],
        )

        for row in rows:
            game_id = clean(row.get("game_id"))
            if game_id in result_game_ids:
                fail(
                    "Duplicate game_id across final-score "
                    f"files: {game_id}"
                )
            result_game_ids.add(game_id)

        completed_game_ids.update(
            validate_result_rows(
                rows,
                path=path,
                season=season,
                schedule_by_id=schedule_by_id,
            )
        )
        result_rows_count += len(rows)

    completed_games_exist = bool(completed_game_ids)

    reporter.update_details({
        "final_score_files": len(result_files),
        "final_score_rows": result_rows_count,
        "completed_games": len(completed_game_ids),
    })

    print(
        f"Completed games detected: "
        f"{len(completed_game_ids)}"
    )

    completed_identities = {
        schedule_identity_by_game_id[game_id]
        for game_id in completed_game_ids
    }

    pbp_path = (
        NFL_ROOT
        / "00_intake/pbp"
        / f"{season}_pbp.csv.gz"
    )

    pbp = read_pbp(
        pbp_path,
        [
            "season",
            "week",
            "game_id",
            "play_id",
            "posteam",
            "defteam",
            "play_type",
            "yards_gained",
            "epa",
            "success",
            "down",
            "yardline_100",
            "posteam_score",
            "posteam_score_post",
            "touchdown",
            "pass_touchdown",
            "rush_touchdown",
            "third_down_converted",
            "third_down_failed",
            "passer_player_id",
            "passer_player_name",
            "qb_dropback",
            "pass_attempt",
            "qb_epa",
            "cpoe",
            "air_yards",
            "sack",
            "interception",
            "fumbled_1_player_id",
        ],
        require_rows=completed_games_exist,
        season=season,
        schedule_identities=schedule_identities,
        completed_identities=completed_identities,
    )

    reporter.set_detail(
        "pbp_rows",
        int(len(pbp)),
    )

    team_stats_path = (
        NFL_ROOT
        / "00_intake/team_stats"
        / f"{season}_team_stats.csv"
    )

    team_stats_rows = read_csv(
        team_stats_path,
        [
            "season",
            "week",
            "team",
            "off_epa_per_play",
            "def_epa_per_play",
            "off_success_rate",
            "def_success_rate",
            "yards_per_play",
            "yards_per_play_allowed",
            "points_per_drive",
            "points_per_drive_allowed",
            "red_zone_td_rate",
            "red_zone_td_rate_allowed",
            "early_down_epa",
            "third_down_conversion_rate",
        ],
        allow_empty=not completed_games_exist,
        unique_by=["season", "week", "team"],
    )

    validate_team_stats(
        team_stats_rows,
        path=team_stats_path,
        season=season,
        schedule_weeks=schedule_weeks,
        valid_abbrs=valid_abbrs,
        name_to_abbr=name_to_abbr,
    )

    qb_stats_path = (
        NFL_ROOT
        / "00_intake/qb"
        / f"{season}_qb_stats.csv"
    )

    qb_stats_rows: list[dict[str, str]] = []

    if completed_games_exist:
        qb_stats_rows = read_csv(
            qb_stats_path,
            [
                "season",
                "week",
                "team",
                "player_id",
                "qb_name",
                "dropbacks",
                "epa_per_play",
                "cpoe",
                "air_yards",
                "sack_rate",
                "interception_rate",
                "fumble_rate",
            ],
            unique_by=[
                "season",
                "week",
                "team",
                "player_id",
            ],
        )
    elif qb_stats_path.exists():
        qb_stats_rows = read_csv(
            qb_stats_path,
            [
                "season",
                "week",
                "team",
                "player_id",
                "qb_name",
                "dropbacks",
            ],
            allow_empty=True,
            unique_by=[
                "season",
                "week",
                "team",
                "player_id",
            ],
        )
    else:
        passed(
            f"{qb_stats_path} not required "
            "before completed games exist"
        )

    validate_qb_stats(
        qb_stats_rows,
        path=qb_stats_path,
        season=season,
        schedule_weeks=schedule_weeks,
        valid_abbrs=valid_abbrs,
        name_to_abbr=name_to_abbr,
        completed_games_exist=completed_games_exist,
    )

    league_master_path = (
        NFL_ROOT
        / "data/master/league_master.csv"
    )

    league_master_rows = read_csv(
        league_master_path,
        [
            "team_id",
            "team_abbr",
            "conference",
            "conference_abbr",
            "division",
            "division_abbr",
            "season",
        ],
        unique_by=["team_id"],
    )

    league_id_to_abbr = validate_league_master(
        league_master_rows,
        path=league_master_path,
        season=season,
        valid_team_ids=valid_team_ids,
        canonical_id_to_abbr=canonical_id_to_abbr,
    )

    standings_path = (
        NFL_ROOT
        / "data/master/league_standings.csv"
    )

    standings_rows = read_csv(
        standings_path,
        [
            "team_id",
            "team_abbr",
            "conference",
            "conference_abbr",
            "division",
            "division_abbr",
            "standings_type",
            "stat_name",
            "stat_value",
            "season",
        ],
    )

    validate_standings(
        standings_rows,
        path=standings_path,
        season=season,
        league_id_to_abbr=league_id_to_abbr,
    )

    coaches_path = (
        NFL_ROOT
        / "data/master/coaches_master.csv"
    )

    coaches_rows = read_csv(
        coaches_path,
        [
            "name",
            "team",
            "team_id",
            "experience",
            "id",
            "uid",
        ],
        unique_by=["team_id"],
    )

    validate_coaches(
        coaches_rows,
        path=coaches_path,
        valid_team_ids=valid_team_ids,
    )

    qbr_files = sorted(
        (
            NFL_ROOT
            / "data/qb_data/qbr_data"
            / str(season)
        ).glob("*.csv")
    )

    if completed_games_exist and not qbr_files:
        fail(f"No QBR files found for season {season}")

    qbr_rows_count = 0

    for path in qbr_files:
        rows = read_csv(
            path,
            [
                "season",
                "week",
                "athlete_id",
                "team_id",
            ],
            unique_by=[
                "season",
                "week",
                "athlete_id",
                "team_id",
            ],
        )

        validate_qbr(
            rows,
            path=path,
            season=season,
            valid_team_ids=valid_team_ids,
        )
        qbr_rows_count += len(rows)

    fpi_path = (
        NFL_ROOT
        / "data/team_power_index"
        / f"team_power_index_{season}.csv"
    )

    fpi_rows = read_csv(
        fpi_path,
        [
            "season",
            "team_id",
            "lastUpdated",
        ],
        unique_by=["team_id"],
    )

    validate_fpi(
        fpi_rows,
        path=fpi_path,
        season=season,
        valid_team_ids=valid_team_ids,
    )

    leaders_path = (
        NFL_ROOT
        / "data/league_leaders"
        / f"league_leaders_{season}.csv"
    )

    leaders_rows: list[dict[str, str]] = []

    if completed_games_exist:
        leaders_rows = read_csv(
            leaders_path,
            [
                "season",
                "category",
                "rank",
                "athlete_id",
                "team_id",
                "value",
                "displayValue",
            ],
            unique_by=[
                "season",
                "category",
                "rank",
            ],
        )
    elif leaders_path.exists():
        leaders_rows = read_csv(
            leaders_path,
            [
                "season",
                "category",
                "rank",
                "athlete_id",
                "team_id",
                "value",
                "displayValue",
            ],
            allow_empty=True,
            unique_by=[
                "season",
                "category",
                "rank",
            ],
        )
    else:
        passed(
            f"{leaders_path} not required "
            "before completed games exist"
        )

    validate_leaders(
        leaders_rows,
        path=leaders_path,
        season=season,
        valid_team_ids=valid_team_ids,
    )

    futures_path = (
        NFL_ROOT
        / "data/market_futures"
        / f"market_futures_{season}.csv"
    )

    futures_rows = read_csv(
        futures_path,
        [
            "season",
            "future_id",
            "future_name",
            "provider_id",
            "provider_name",
            "athlete_id",
            "team_id",
            "value",
        ],
    )

    validate_market_futures(
        futures_rows,
        path=futures_path,
        season=season,
        valid_team_ids=valid_team_ids,
    )

    current_weekly_files = validate_weekly_and_travel(
        season=season,
        schedule_by_id=schedule_by_id,
    )

    reporter.set_rows(
        rows_in=(
            len(schedule_rows)
            + result_rows_count
            + len(pbp)
            + len(team_stats_rows)
            + len(qb_stats_rows)
            + len(league_master_rows)
            + len(standings_rows)
            + len(coaches_rows)
            + qbr_rows_count
            + len(fpi_rows)
            + len(leaders_rows)
            + len(futures_rows)
        ),
        rows_out=0,
    )

    reporter.update_details({
        "team_universe_size": len(valid_team_ids),
        "team_stats_rows": len(team_stats_rows),
        "qb_stats_rows": len(qb_stats_rows),
        "league_master_rows": len(league_master_rows),
        "standings_rows": len(standings_rows),
        "coaches_rows": len(coaches_rows),
        "qbr_files": len(qbr_files),
        "qbr_rows": qbr_rows_count,
        "fpi_rows": len(fpi_rows),
        "league_leader_rows": len(leaders_rows),
        "market_futures_rows": len(futures_rows),
        "current_season_weekly_files": current_weekly_files,
        "validation_completed": True,
    })

    print("TUESDAY VALIDATION PASSED")


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
            extra_context={
                "component": "Tuesday validation",
            },
        ) as reporter:
            run(args, reporter)

        return 0

    except Exception:
        return 1


if __name__ == "__main__":
    sys.exit(main())
