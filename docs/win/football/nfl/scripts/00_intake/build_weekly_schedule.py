#!/usr/bin/env python3
"""Build the configured NFL weekly schedule from schedule and current odds."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tempfile
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
NFL_ROOT = SCRIPT_PATH.parents[2]

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_reporter import PipelineReporter

SCHEDULE_DIR = NFL_ROOT / "00_intake" / "schedule"
WEEKLY_DIR = SCHEDULE_DIR / "weekly"
ODDS_DIR = NFL_ROOT / "00_intake" / "odds"
RAW_ODDS_DIR = ODDS_DIR / "raw"
TEAM_MAP_PATH = NFL_ROOT / "config" / "mapping" / "team_map.csv"
REPORT_ROOT = NFL_ROOT / "errors"
LOG_FILE = REPORT_ROOT / "00_intake" / "build_weekly_schedule.txt"

SEASON_TYPE_TO_ESPN = {"pre": 1, "reg": 2, "post": 3}

OUTPUT_COLUMNS = [
    "season",
    "season_type",
    "week",
    "game_id",
    "odds_provider_game_id",
    "game_date",
    "game_time",
    "commence_time",
    "away_team",
    "home_team",
    "odds_away_team",
    "odds_home_team",
    "neutral_site",
    "stadium",
    "roof",
    "surface",
    "home_timezone",
    "away_timezone",
    "game_timezone",
    "bookmaker",
    "home_moneyline_american",
    "away_moneyline_american",
    "home_spread",
    "away_spread",
    "home_spread_american",
    "away_spread_american",
    "total",
    "over_american",
    "under_american",
    "odds_last_update",
    "odds_available",
    "odds_missing_reason",
]

SCHEDULE_REQUIRED_COLUMNS = [
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
    "roof",
    "surface",
    "home_timezone",
    "away_timezone",
    "game_timezone",
]

ODDS_REQUIRED_COLUMNS = [
    "snapshot_id",
    "snapshot_fetched_at",
    "game_id",
    "commence_time",
    "home_team",
    "away_team",
    "bookmaker",
    "market_type",
    "bet_side",
    "line",
    "odds_american",
    "odds_decimal",
    "last_update",
    "home_moneyline_american",
    "away_moneyline_american",
    "home_spread",
    "away_spread",
    "home_spread_american",
    "away_spread_american",
    "total",
    "over_american",
    "under_american",
]

EXPECTED_MARKET_SIDES = {
    ("h2h", "home"),
    ("h2h", "away"),
    ("spreads", "home"),
    ("spreads", "away"),
    ("totals", "over"),
    ("totals", "under"),
}

OBSERVED_ODDS_TEAM_MAP = {
    "Arizona": "Arizona Cardinals",
    "Atlanta": "Atlanta Falcons",
    "Baltimore": "Baltimore Ravens",
    "Buffalo": "Buffalo Bills",
    "Carolina": "Carolina Panthers",
    "Chicago": "Chicago Bears",
    "Cincinnati": "Cincinnati Bengals",
    "Cleveland": "Cleveland Browns",
    "Dallas": "Dallas Cowboys",
    "Denver": "Denver Broncos",
    "Detroit": "Detroit Lions",
    "Green Bay": "Green Bay Packers",
    "Houston": "Houston Texans",
    "Indianapolis": "Indianapolis Colts",
    "Jacksonville": "Jacksonville Jaguars",
    "Kansas City": "Kansas City Chiefs",
    "LA Chargers": "Los Angeles Chargers",
    "LA Rams": "Los Angeles Rams",
    "Las Vegas": "Las Vegas Raiders",
    "Miami": "Miami Dolphins",
    "Minnesota": "Minnesota Vikings",
    "New England": "New England Patriots",
    "New Orleans": "New Orleans Saints",
    "NY Giants": "New York Giants",
    "NY Jets": "New York Jets",
    "Philadelphia": "Philadelphia Eagles",
    "Pittsburgh": "Pittsburgh Steelers",
    "San Francisco": "San Francisco 49ers",
    "Seattle": "Seattle Seahawks",
    "Tampa Bay": "Tampa Bay Buccaneers",
    "Tennessee": "Tennessee Titans",
    "Washington": "Washington Commanders",
}


class WeeklyScheduleError(RuntimeError):
    pass


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(f"[{utc_now_iso()}] {message}\n")


def fail(message: str) -> None:
    log(f"ERROR: {message}")
    raise WeeklyScheduleError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", required=True, type=int)
    parser.add_argument(
        "--season-type",
        required=True,
        choices=sorted(SEASON_TYPE_TO_ESPN),
    )
    parser.add_argument("--week", required=True, type=int)
    args = parser.parse_args()

    if not 2000 <= args.season <= 2100:
        parser.error("--season must be between 2000 and 2100")
    if not 1 <= args.week <= 25:
        parser.error("--week must be between 1 and 25")

    return args


def read_csv(
    path: Path,
    required_columns: list[str],
    label: str,
    *,
    allow_empty: bool = False,
) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        fail(f"Missing {label}: {path}")
    if path.stat().st_size == 0:
        fail(f"Zero-byte {label}: {path}")

    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            rows = list(reader)
    except Exception as exc:
        fail(
            f"Could not read {label} {path}: "
            f"{type(exc).__name__}: {exc}"
        )

    missing = [
        column for column in required_columns
        if column not in fieldnames
    ]
    if missing:
        fail(f"{label} missing columns: {missing}")
    if not rows and not allow_empty:
        fail(f"{label} contains no data rows: {path}")

    return fieldnames, rows


def parse_iso_datetime(value: Any, label: str) -> datetime:
    text = clean(value)
    if not text:
        fail(f"{label} is blank")

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        fail(f"{label} is not a valid ISO timestamp: {value!r}")

    if parsed.tzinfo is None:
        fail(f"{label} has no timezone: {value!r}")

    return parsed


def parse_date(value: Any):
    text = clean(value)
    if not text:
        return None

    for fmt in ("%Y-%m-%d", "%Y_%m_%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        return None


def normalize_key(value: Any) -> str:
    text = clean(value).lower().replace("&", "and")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def add_team_mapping(
    mapping: dict[str, str],
    source_name: Any,
    canonical_team: Any,
    *,
    source: str,
) -> None:
    source_text = clean(source_name)
    canonical_text = clean(canonical_team)
    if not source_text or not canonical_text:
        return

    key = normalize_key(source_text)
    if not key:
        return

    existing = mapping.get(key)
    if existing is not None and existing != canonical_text:
        fail(
            f"Conflicting team mapping for {source_text!r}: "
            f"{existing!r} vs {canonical_text!r} from {source}"
        )

    mapping[key] = canonical_text


def load_team_map() -> dict[str, str]:
    mapping: dict[str, str] = {}

    for source_name, canonical_team in OBSERVED_ODDS_TEAM_MAP.items():
        add_team_mapping(
            mapping,
            source_name,
            canonical_team,
            source="observed map",
        )
        add_team_mapping(
            mapping,
            canonical_team,
            canonical_team,
            source="observed map",
        )

    _, rows = read_csv(
        TEAM_MAP_PATH,
        ["canonical_team", "alias"],
        "team map",
    )

    for row in rows:
        if (
            clean(row.get("sport")).lower()
            not in {"", "football"}
        ):
            continue
        if (
            clean(row.get("league")).lower()
            not in {"", "nfl"}
        ):
            continue

        canonical_team = clean(row.get("canonical_team"))
        alias = clean(row.get("alias"))

        add_team_mapping(
            mapping,
            canonical_team,
            canonical_team,
            source=str(TEAM_MAP_PATH),
        )
        add_team_mapping(
            mapping,
            alias,
            canonical_team,
            source=str(TEAM_MAP_PATH),
        )

    if not mapping:
        fail("Team mapping resolved no usable entries")

    return mapping


def canonical_team(value: Any, team_map: dict[str, str]) -> str:
    raw = clean(value)
    if not raw:
        return ""
    return team_map.get(normalize_key(raw), raw)


def validate_schedule(
    rows: list[dict[str, str]],
    *,
    season: int,
    season_type: str,
    week: int,
) -> list[dict[str, str]]:
    seen_game_ids: set[str] = set()
    target_rows: list[dict[str, str]] = []

    for line_number, row in enumerate(rows, start=2):
        game_id = clean(row.get("game_id"))
        if not game_id:
            fail(f"Schedule row {line_number} has blank game_id")
        if game_id in seen_game_ids:
            fail(f"Duplicate schedule game_id: {game_id}")
        seen_game_ids.add(game_id)

        if clean(row.get("season")) != str(season):
            fail(
                f"Schedule row {line_number} has "
                f"season={row.get('season')!r}; expected {season}"
            )

        if (
            clean(row.get("season_type")) != season_type
            or clean(row.get("week")) != str(week)
        ):
            continue

        for field in ("game_date", "away_team", "home_team"):
            if not clean(row.get(field)):
                fail(
                    f"Target schedule row {line_number} "
                    f"has blank {field}"
                )

        if parse_date(row.get("game_date")) is None:
            fail(
                f"Target schedule row {line_number} "
                "has invalid game_date"
            )

        target_rows.append(row)

    if not target_rows:
        fail(
            "Configured target contains no schedule rows: "
            f"season={season} season_type={season_type} week={week}"
        )

    return target_rows


def raw_capture_matches(
    payload: dict[str, Any],
    *,
    season: int,
    season_type_numeric: int,
    week: int,
) -> bool:
    return (
        payload.get("season") == season
        and payload.get("season_type") == season_type_numeric
        and payload.get("target_week") == week
    )


def select_odds_capture(
    *,
    season: int,
    season_type_numeric: int,
    week: int,
) -> tuple[Path, Path, dict[str, Any], list[dict[str, str]]]:
    candidates = []

    for raw_path in sorted(RAW_ODDS_DIR.glob("*_nfl_odds.json")):
        match = re.fullmatch(
            r"(\d{4}_\d{2}_\d{2})_nfl_odds\.json",
            raw_path.name,
        )
        if not match:
            continue

        csv_path = ODDS_DIR / f"{match.group(1)}_NFL_odds.csv"
        if not csv_path.is_file():
            continue

        try:
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        if not isinstance(payload, dict):
            continue
        if not raw_capture_matches(
            payload,
            season=season,
            season_type_numeric=season_type_numeric,
            week=week,
        ):
            continue

        snapshot_id = clean(payload.get("snapshot_id"))
        fetched_at = clean(payload.get("fetched_at"))
        if not snapshot_id or not fetched_at:
            continue

        try:
            timestamp = parse_iso_datetime(
                fetched_at,
                f"{raw_path} fetched_at",
            )
        except WeeklyScheduleError:
            continue

        candidates.append(
            (
                timestamp,
                snapshot_id,
                raw_path,
                csv_path,
                payload,
            )
        )

    if not candidates:
        fail(
            "No paired odds capture matches configured target "
            f"season={season} season_type={season_type_numeric} "
            f"week={week}"
        )

    _, _, raw_path, csv_path, payload = max(
        candidates,
        key=lambda item: (item[0], item[1]),
    )

    _, odds_rows = read_csv(
        csv_path,
        ODDS_REQUIRED_COLUMNS,
        "normalized odds CSV",
        allow_empty=True,
    )

    return raw_path, csv_path, payload, odds_rows


def validate_raw_odds(
    payload: dict[str, Any],
    *,
    season: int,
    season_type_numeric: int,
    week: int,
) -> tuple[str, str, list[dict[str, Any]], list[dict[str, Any]]]:
    if not raw_capture_matches(
        payload,
        season=season,
        season_type_numeric=season_type_numeric,
        week=week,
    ):
        fail("Selected raw odds capture does not match configured target")

    snapshot_id = clean(payload.get("snapshot_id"))
    fetched_at = clean(payload.get("fetched_at"))
    if not snapshot_id:
        fail("Raw odds snapshot_id is blank")
    parse_iso_datetime(fetched_at, "raw odds fetched_at")

    events = payload.get("events")
    odds = payload.get("odds")

    if not isinstance(events, list):
        fail("Raw odds events is not a list")
    if not isinstance(odds, list):
        fail("Raw odds odds is not a list")
    if not events:
        fail("Raw odds contains no configured-week events")

    if payload.get("events_count") != len(events):
        fail("Raw odds events_count does not match events")
    if payload.get("odds_events_count") != len(odds):
        fail("Raw odds odds_events_count does not match odds")

    event_ids: list[str] = []

    for index, event in enumerate(events, start=1):
        if not isinstance(event, dict):
            fail(f"Raw odds event {index} is not an object")

        event_id = clean(event.get("id"))
        if not event_id:
            fail(f"Raw odds event {index} has blank id")

        if (
            not clean(event.get("home"))
            or not clean(event.get("away"))
            or parse_date(event.get("date")) is None
        ):
            fail(
                f"Raw odds event {event_id} "
                "has invalid matching metadata"
            )

        event_week = clean(event.get("week"))
        if event_week and event_week != str(week):
            fail(
                f"Raw odds event {event_id} has "
                f"week={event_week}; expected {week}"
            )

        event_ids.append(event_id)

    if len(event_ids) != len(set(event_ids)):
        fail("Raw odds events contain duplicate IDs")

    odds_ids: list[str] = []

    for index, odds_item in enumerate(odds, start=1):
        if not isinstance(odds_item, dict):
            fail(f"Raw odds object {index} is not an object")

        odds_id = clean(odds_item.get("id"))
        if not odds_id:
            fail(f"Raw odds object {index} has blank id")
        odds_ids.append(odds_id)

    if len(odds_ids) != len(set(odds_ids)):
        fail("Raw odds objects contain duplicate IDs")

    if not set(odds_ids).issubset(set(event_ids)):
        fail("Raw odds contains odds IDs absent from events")

    return snapshot_id, fetched_at, events, odds


def validate_normalized_odds(
    rows: list[dict[str, str]],
    *,
    snapshot_id: str,
    fetched_at: str,
    raw_odds: list[dict[str, Any]],
) -> dict[str, dict[str, str]]:
    expected_ids = {
        clean(item.get("id"))
        for item in raw_odds
        if isinstance(item, dict) and clean(item.get("id"))
    }

    expected_row_count = len(expected_ids) * len(EXPECTED_MARKET_SIDES)
    if len(rows) != expected_row_count:
        fail(
            "Normalized odds row count does not match raw odds events: "
            f"rows={len(rows)} expected={expected_row_count}"
        )

    grouped: dict[str, list[dict[str, str]]] = {}
    seen_keys: set[tuple[str, str, str, str]] = set()

    for line_number, row in enumerate(rows, start=2):
        game_id = clean(row.get("game_id"))
        bookmaker = clean(row.get("bookmaker"))
        market_type = clean(row.get("market_type"))
        bet_side = clean(row.get("bet_side"))

        if not game_id or not bookmaker:
            fail(
                f"Normalized odds row {line_number} "
                "has blank game_id or bookmaker"
            )

        if clean(row.get("snapshot_id")) != snapshot_id:
            fail(
                f"Normalized odds row {line_number} "
                "has wrong snapshot_id"
            )
        if clean(row.get("snapshot_fetched_at")) != fetched_at:
            fail(
                f"Normalized odds row {line_number} "
                "has wrong snapshot_fetched_at"
            )

        pair = (market_type, bet_side)
        if pair not in EXPECTED_MARKET_SIDES:
            fail(
                f"Normalized odds row {line_number} "
                f"has invalid market/side {pair}"
            )

        key = (game_id, market_type, bet_side, bookmaker)
        if key in seen_keys:
            fail(f"Duplicate normalized odds key: {key}")

        seen_keys.add(key)
        grouped.setdefault(game_id, []).append(row)

    if set(grouped) != expected_ids:
        fail("Normalized odds game IDs do not match raw odds IDs")

    consistent_fields = [
        "commence_time",
        "home_team",
        "away_team",
        "bookmaker",
        "home_moneyline_american",
        "away_moneyline_american",
        "home_spread",
        "away_spread",
        "home_spread_american",
        "away_spread_american",
        "total",
        "over_american",
        "under_american",
        "last_update",
    ]

    summaries: dict[str, dict[str, str]] = {}

    for game_id, game_rows in grouped.items():
        pairs = {
            (
                clean(row.get("market_type")),
                clean(row.get("bet_side")),
            )
            for row in game_rows
        }
        if pairs != EXPECTED_MARKET_SIDES:
            fail(
                f"Normalized odds game {game_id} "
                "does not have the expected six market rows"
            )

        values_by_field: dict[str, str] = {}

        for field in consistent_fields:
            values = {clean(row.get(field)) for row in game_rows}
            if len(values) != 1:
                fail(
                    f"Normalized odds game {game_id} "
                    f"has inconsistent {field}"
                )
            values_by_field[field] = next(iter(values))

        summaries[game_id] = {
            "bookmaker": values_by_field["bookmaker"],
            "home_moneyline_american": (
                values_by_field["home_moneyline_american"]
            ),
            "away_moneyline_american": (
                values_by_field["away_moneyline_american"]
            ),
            "home_spread": values_by_field["home_spread"],
            "away_spread": values_by_field["away_spread"],
            "home_spread_american": (
                values_by_field["home_spread_american"]
            ),
            "away_spread_american": (
                values_by_field["away_spread_american"]
            ),
            "total": values_by_field["total"],
            "over_american": values_by_field["over_american"],
            "under_american": values_by_field["under_american"],
            "odds_last_update": values_by_field["last_update"],
        }

    return summaries


def build_schedule_index(
    schedule_rows: list[dict[str, str]],
    team_map: dict[str, str],
) -> dict[tuple[str, str, str], dict[str, str]]:
    index = {}

    for row in schedule_rows:
        home = canonical_team(row.get("home_team"), team_map)
        away = canonical_team(row.get("away_team"), team_map)
        game_date = parse_date(row.get("game_date"))

        if not home or not away or game_date is None:
            fail(
                "Target schedule has invalid team/date for "
                f"game_id={row.get('game_id')!r}"
            )

        key = (
            game_date.isoformat(),
            normalize_key(home),
            normalize_key(away),
        )

        if key in index:
            fail(f"Ambiguous target schedule identity for key={key}")

        index[key] = row

    return index


def schedule_candidate_keys(
    raw_event: dict[str, Any],
    team_map: dict[str, str],
) -> list[tuple[str, str, str]]:
    odds_home = canonical_team(raw_event.get("home"), team_map)
    odds_away = canonical_team(raw_event.get("away"), team_map)
    odds_date = parse_date(raw_event.get("date"))

    if not odds_home or not odds_away or odds_date is None:
        return []

    return [
        (
            candidate_date.isoformat(),
            normalize_key(odds_home),
            normalize_key(odds_away),
        )
        for candidate_date in (
            odds_date,
            odds_date - timedelta(days=1),
        )
    ]


def match_raw_events_to_schedule(
    raw_events: list[dict[str, Any]],
    schedule_index: dict[tuple[str, str, str], dict[str, str]],
    team_map: dict[str, str],
) -> tuple[dict[str, dict[str, str]], list[dict[str, Any]]]:
    matches: dict[str, dict[str, str]] = {}
    unmatched_events: list[dict[str, Any]] = []

    for event in raw_events:
        event_id = clean(event.get("id"))
        candidates_by_id: dict[str, dict[str, str]] = {}

        for key in schedule_candidate_keys(event, team_map):
            candidate = schedule_index.get(key)
            if candidate is None:
                continue
            candidate_id = clean(candidate.get("game_id"))
            candidates_by_id[candidate_id] = candidate

        if len(candidates_by_id) > 1:
            fail(
                f"Raw event {event_id} matches multiple "
                f"schedule games: {sorted(candidates_by_id)}"
            )

        if not candidates_by_id:
            unmatched_events.append(event)
            continue

        matched_schedule = next(iter(candidates_by_id.values()))
        schedule_game_id = clean(matched_schedule.get("game_id"))

        if schedule_game_id in matches:
            fail(
                "Multiple raw events matched schedule "
                f"game_id={schedule_game_id}"
            )

        matches[schedule_game_id] = {
            "odds_provider_game_id": event_id,
            "commence_time": clean(event.get("date")),
            "odds_home_team": clean(event.get("home")),
            "odds_away_team": clean(event.get("away")),
        }

    return matches, unmatched_events


def build_output_rows(
    schedule_rows: list[dict[str, str]],
    schedule_matches: dict[str, dict[str, str]],
    odds_summary: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    output_rows = []

    for schedule_row in schedule_rows:
        schedule_game_id = clean(schedule_row.get("game_id"))
        match = schedule_matches.get(schedule_game_id, {})
        odds_provider_game_id = clean(
            match.get("odds_provider_game_id")
        )
        odds = odds_summary.get(odds_provider_game_id, {})

        row = {
            "season": schedule_row.get("season", ""),
            "season_type": schedule_row.get("season_type", ""),
            "week": schedule_row.get("week", ""),
            "game_id": schedule_game_id,
            "odds_provider_game_id": odds_provider_game_id,
            "game_date": schedule_row.get("game_date", ""),
            "game_time": schedule_row.get("game_time", ""),
            "commence_time": match.get("commence_time", ""),
            "away_team": schedule_row.get("away_team", ""),
            "home_team": schedule_row.get("home_team", ""),
            "odds_away_team": match.get("odds_away_team", ""),
            "odds_home_team": match.get("odds_home_team", ""),
            "neutral_site": schedule_row.get("neutral_site", ""),
            "stadium": schedule_row.get("stadium", ""),
            "roof": schedule_row.get("roof", ""),
            "surface": schedule_row.get("surface", ""),
            "home_timezone": schedule_row.get("home_timezone", ""),
            "away_timezone": schedule_row.get("away_timezone", ""),
            "game_timezone": schedule_row.get("game_timezone", ""),
            "bookmaker": odds.get("bookmaker", ""),
            "home_moneyline_american": odds.get(
                "home_moneyline_american", ""
            ),
            "away_moneyline_american": odds.get(
                "away_moneyline_american", ""
            ),
            "home_spread": odds.get("home_spread", ""),
            "away_spread": odds.get("away_spread", ""),
            "home_spread_american": odds.get(
                "home_spread_american", ""
            ),
            "away_spread_american": odds.get(
                "away_spread_american", ""
            ),
            "total": odds.get("total", ""),
            "over_american": odds.get("over_american", ""),
            "under_american": odds.get("under_american", ""),
            "odds_last_update": odds.get("odds_last_update", ""),
            "odds_available": "",
            "odds_missing_reason": "",
        }

        if odds_provider_game_id and odds:
            row["odds_available"] = "1"
        elif odds_provider_game_id:
            row["odds_available"] = "0"
            row["odds_missing_reason"] = "no_odds_returned"
        else:
            row["odds_available"] = "0"
            row["odds_missing_reason"] = "no_odds_event_match"

        output_rows.append(row)

    output_rows.sort(
        key=lambda row: (
            clean(row.get("game_date")),
            clean(row.get("game_time")),
            clean(row.get("away_team")),
            clean(row.get("home_team")),
        )
    )

    return output_rows


def validate_output_rows(
    rows: list[dict[str, str]],
    *,
    target_schedule_rows: list[dict[str, str]],
    season: int,
    season_type: str,
    week: int,
    raw_event_ids: set[str],
    odds_summary_ids: set[str],
) -> None:
    if not rows:
        fail("Weekly schedule output contains no rows")

    expected_schedule_ids = {
        clean(row.get("game_id"))
        for row in target_schedule_rows
    }
    actual_ids: set[str] = set()

    for line_number, row in enumerate(rows, start=2):
        actual = (
            clean(row.get("season")),
            clean(row.get("season_type")),
            clean(row.get("week")),
        )
        expected = (str(season), season_type, str(week))

        if actual != expected:
            fail(
                f"Weekly schedule row {line_number} has "
                f"target={actual}; expected={expected}"
            )

        game_id = clean(row.get("game_id"))
        if not game_id:
            fail(
                f"Weekly schedule row {line_number} "
                "has blank game_id"
            )
        if game_id in actual_ids:
            fail(f"Duplicate weekly schedule game_id: {game_id}")
        actual_ids.add(game_id)

        odds_available = clean(row.get("odds_available"))
        missing_reason = clean(row.get("odds_missing_reason"))
        provider_game_id = clean(row.get("odds_provider_game_id"))

        if odds_available not in {"0", "1"}:
            fail(
                f"Weekly schedule row {line_number} "
                "has invalid odds_available"
            )

        if odds_available == "1":
            if (
                not provider_game_id
                or missing_reason
                or provider_game_id not in raw_event_ids
                or provider_game_id not in odds_summary_ids
                or not clean(row.get("bookmaker"))
            ):
                fail(
                    f"Weekly schedule row {line_number} "
                    "has invalid available-odds state"
                )

        elif missing_reason == "no_odds_returned":
            if (
                not provider_game_id
                or provider_game_id not in raw_event_ids
                or provider_game_id in odds_summary_ids
            ):
                fail(
                    f"Weekly schedule row {line_number} "
                    "has invalid no_odds_returned state"
                )

        elif missing_reason == "no_odds_event_match":
            if provider_game_id:
                fail(
                    f"Weekly schedule row {line_number} "
                    "has provider ID with no_odds_event_match"
                )

        else:
            fail(
                f"Weekly schedule row {line_number} has "
                f"invalid missing-odds reason={missing_reason!r}"
            )

    if actual_ids != expected_schedule_ids:
        fail(
            "Weekly schedule game IDs do not exactly match "
            "configured schedule week"
        )


def publish(
    output_path: Path,
    rows: list[dict[str, str]],
    *,
    target_schedule_rows: list[dict[str, str]],
    season: int,
    season_type: str,
    week: int,
    raw_event_ids: set[str],
    odds_summary_ids: set[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=".weekly_schedule_stage_",
        dir=output_path.parent,
    ) as staging_dir:
        staged_path = Path(staging_dir) / output_path.name

        with staged_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=OUTPUT_COLUMNS,
                lineterminator="\n",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        column: row.get(column, "")
                        for column in OUTPUT_COLUMNS
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())

        if staged_path.stat().st_size == 0:
            fail("Staged weekly schedule is zero bytes")

        with staged_path.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            reader = csv.DictReader(handle)
            staged_header = reader.fieldnames or []
            staged_rows = list(reader)

        if staged_header != OUTPUT_COLUMNS:
            fail("Staged weekly schedule headers changed")
        if len(staged_rows) != len(rows):
            fail("Staged weekly schedule row count changed")

        validate_output_rows(
            staged_rows,
            target_schedule_rows=target_schedule_rows,
            season=season,
            season_type=season_type,
            week=week,
            raw_event_ids=raw_event_ids,
            odds_summary_ids=odds_summary_ids,
        )

        os.replace(staged_path, output_path)


def run(
    args: argparse.Namespace,
    reporter: PipelineReporter,
) -> None:
    season_type_numeric = SEASON_TYPE_TO_ESPN[args.season_type]
    schedule_path = SCHEDULE_DIR / f"{args.season}_schedule.csv"
    output_path = (
        WEEKLY_DIR
        / f"week_{args.week}_NFL_weekly_schedule.csv"
    )

    reporter.add_input(schedule_path)
    reporter.add_input(TEAM_MAP_PATH)
    reporter.update_details(
        {
            "configured_season": args.season,
            "configured_season_type": args.season_type,
            "configured_season_type_numeric": season_type_numeric,
            "configured_week": args.week,
            "publication_completed": False,
            "staged_roundtrip_verified": False,
        }
    )

    _, schedule_rows = read_csv(
        schedule_path,
        SCHEDULE_REQUIRED_COLUMNS,
        "season schedule CSV",
    )

    target_schedule_rows = validate_schedule(
        schedule_rows,
        season=args.season,
        season_type=args.season_type,
        week=args.week,
    )

    team_map = load_team_map()

    (
        raw_odds_path,
        odds_csv_path,
        raw_payload,
        odds_rows,
    ) = select_odds_capture(
        season=args.season,
        season_type_numeric=season_type_numeric,
        week=args.week,
    )

    reporter.add_input(raw_odds_path)
    reporter.add_input(odds_csv_path)

    (
        snapshot_id,
        snapshot_fetched_at,
        raw_events,
        raw_odds,
    ) = validate_raw_odds(
        raw_payload,
        season=args.season,
        season_type_numeric=season_type_numeric,
        week=args.week,
    )

    odds_summary = validate_normalized_odds(
        odds_rows,
        snapshot_id=snapshot_id,
        fetched_at=snapshot_fetched_at,
        raw_odds=raw_odds,
    )

    schedule_index = build_schedule_index(
        target_schedule_rows,
        team_map,
    )
    schedule_matches, unmatched_events = (
        match_raw_events_to_schedule(
            raw_events,
            schedule_index,
            team_map,
        )
    )

    output_rows = build_output_rows(
        target_schedule_rows,
        schedule_matches,
        odds_summary,
    )

    raw_event_ids = {
        clean(event.get("id"))
        for event in raw_events
    }
    odds_summary_ids = set(odds_summary)

    validate_output_rows(
        output_rows,
        target_schedule_rows=target_schedule_rows,
        season=args.season,
        season_type=args.season_type,
        week=args.week,
        raw_event_ids=raw_event_ids,
        odds_summary_ids=odds_summary_ids,
    )

    matched_with_odds = sum(
        row.get("odds_available") == "1"
        for row in output_rows
    )
    matched_without_odds = sum(
        row.get("odds_missing_reason") == "no_odds_returned"
        for row in output_rows
    )
    no_event_match = sum(
        row.get("odds_missing_reason") == "no_odds_event_match"
        for row in output_rows
    )

    reporter.set_rows(
        rows_in=len(target_schedule_rows),
        rows_out=0,
    )
    reporter.update_details(
        {
            "schedule_path": str(schedule_path),
            "raw_odds_path": str(raw_odds_path),
            "odds_csv_path": str(odds_csv_path),
            "output_path": str(output_path),
            "snapshot_id": snapshot_id,
            "snapshot_fetched_at": snapshot_fetched_at,
            "season_schedule_rows": len(schedule_rows),
            "target_schedule_rows": len(target_schedule_rows),
            "raw_events": len(raw_events),
            "raw_odds_events": len(raw_odds),
            "normalized_odds_rows": len(odds_rows),
            "schedule_matches": len(schedule_matches),
            "unmatched_raw_events": len(unmatched_events),
            "rows_with_odds": matched_with_odds,
            "rows_with_event_no_odds": matched_without_odds,
            "rows_no_event_match": no_event_match,
        }
    )

    if unmatched_events:
        reporter.warning(
            "Some configured-week raw events did not match "
            "the season schedule",
            event_ids=[
                clean(event.get("id"))
                for event in unmatched_events
            ],
        )

    if matched_without_odds:
        reporter.warning(
            "Some schedule games have an ESPN event "
            "but no usable odds",
            count=matched_without_odds,
        )

    if no_event_match:
        reporter.warning(
            "Some configured schedule games have no "
            "matched raw odds event",
            count=no_event_match,
        )

    publish(
        output_path,
        output_rows,
        target_schedule_rows=target_schedule_rows,
        season=args.season,
        season_type=args.season_type,
        week=args.week,
        raw_event_ids=raw_event_ids,
        odds_summary_ids=odds_summary_ids,
    )

    reporter.add_output(output_path)
    reporter.add_output(LOG_FILE)
    reporter.set_rows(
        rows_in=len(target_schedule_rows),
        rows_out=len(output_rows),
    )
    reporter.update_details(
        {
            "rows_published": len(output_rows),
            "staged_roundtrip_verified": True,
            "publication_completed": True,
        }
    )

    log(f"Schedule input: {schedule_path}")
    log(f"Odds CSV input: {odds_csv_path}")
    log(f"Raw odds input: {raw_odds_path}")
    log(f"Odds snapshot: {snapshot_id}")
    log(f"Schedule rows loaded: {len(schedule_rows)}")
    log(f"Configured-week schedule rows: {len(target_schedule_rows)}")
    log(f"Raw odds events loaded: {len(raw_events)}")
    log(f"Raw odds objects loaded: {len(raw_odds)}")
    log(f"Odds CSV rows loaded: {len(odds_rows)}")
    log(f"Schedule matches from raw events: {len(schedule_matches)}")
    log(f"Unmatched raw odds events: {len(unmatched_events)}")
    log(
        "Target week: "
        f"season={args.season}, "
        f"season_type={args.season_type}, "
        f"week={args.week}"
    )
    log(f"Weekly schedule rows written: {len(output_rows)}")
    log(f"Rows with odds: {matched_with_odds}")
    log(f"Rows with event but no odds: {matched_without_odds}")
    log(f"Rows with no odds event match: {no_event_match}")
    log(f"Output written: {output_path}")

    for event in unmatched_events:
        log(
            "UNMATCHED_RAW_EVENT "
            f"id={event.get('id', '')} "
            f"date={event.get('date', '')} "
            f"away={event.get('away', '')} "
            f"home={event.get('home', '')}"
        )

    print(
        f"rows={len(output_rows)} "
        f"with_odds={matched_with_odds} "
        f"event_no_odds={matched_without_odds} "
        f"no_event_match={no_event_match} "
        f"snapshot_id={snapshot_id}"
    )


def main() -> int:
    args = parse_args()

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("", encoding="utf-8")

    try:
        with PipelineReporter(
            script=SCRIPT_PATH,
            stage="00_intake",
            report_root=REPORT_ROOT,
            pipeline="NFL",
            league="NFL",
            season=args.season,
            week=args.week,
            extra_context={
                "component": "weekly schedule",
                "season_type": args.season_type,
            },
        ) as reporter:
            run(args, reporter)

        return 0

    except Exception as exc:
        log(traceback.format_exc())
        print(
            f"ERROR: {type(exc).__name__}: {exc}; "
            f"see {LOG_FILE}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
