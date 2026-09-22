#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
docs/win/football/nfl/scripts/00_intake/pull_schedule.py

Pulls an NFL regular-season schedule from the ESPN Core API.

Source:
  https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{SEASON}/types/2/weeks/{WEEK}/events?limit=100

Inputs:
  docs/win/football/nfl/config/mapping/team_map.csv
  docs/win/football/nfl/config/mapping/stadium_map_nfl.csv

Main output:
  docs/win/football/nfl/00_intake/schedule/{SEASON}_schedule.csv

Per-run pulled output:
  docs/win/football/nfl/00_intake/schedule/updates/{SEASON}_schedule_YYYYMMDD_HHMMSS.csv

Standard report:
  docs/win/football/nfl/errors/00_intake/pull_schedule.json

Legacy summary / warnings:
  docs/win/football/nfl/errors/00_intake/pull_schedule.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import tempfile
import traceback
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
NFL_DIR = SCRIPT_PATH.parents[2]

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_reporter import PipelineReporter


REGULAR_SEASON_WEEKS = tuple(range(1, 19))


OUTPUT_COLUMNS = [
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

TEAM_ID_COLUMN = "team_id"
CANONICAL_TEAM_COLUMN = "canonical_team"

TEAM_MAP_FILE = NFL_DIR / "config" / "mapping" / "team_map.csv"
STADIUM_MAP_FILE = NFL_DIR / "config" / "mapping" / "stadium_map_nfl.csv"

OUTPUT_DIR = NFL_DIR / "00_intake" / "schedule"
UPDATES_DIR = OUTPUT_DIR / "updates"

REPORT_ROOT = NFL_DIR / "errors"
LEGACY_LOG_FILE = REPORT_ROOT / "00_intake" / "pull_schedule.txt"


class ScheduleError(RuntimeError):
    pass


class RunLog:
    def __init__(self, reporter: PipelineReporter) -> None:
        self.reporter = reporter
        self.lines: list[str] = []
        self.warning_count = 0
        self.error_count = 0

    def info(self, message: str) -> None:
        text = str(message).rstrip()
        self.lines.append(text)
        print(f"INFO: {text}")

    def warning(self, message: str, **details: Any) -> None:
        text = str(message).rstrip()
        self.warning_count += 1
        self.lines.append(f"WARNING: {text}")
        self.reporter.warning(text, **details)
        print(f"WARNING: {text}")

    def error(self, message: str, **details: Any) -> None:
        text = str(message).rstrip()
        self.error_count += 1
        self.lines.append(f"ERROR: {text}")
        self.reporter.error(text, **details)
        print(f"ERROR: {text}", file=sys.stderr)

    @property
    def has_errors(self) -> bool:
        return self.error_count > 0

    def write_legacy(self) -> None:
        LEGACY_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = LEGACY_LOG_FILE.with_name(
            f".{LEGACY_LOG_FILE.name}.tmp"
        )

        payload = "\n".join(self.lines).rstrip() + "\n"

        try:
            with temporary_path.open("w", encoding="utf-8", newline="") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())

            os.replace(temporary_path, LEGACY_LOG_FILE)
        finally:
            temporary_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull the configured NFL season schedule from ESPN."
    )
    parser.add_argument(
        "--season",
        required=True,
        type=int,
        help="NFL season year to pull, for example 2026.",
    )
    args = parser.parse_args()

    if args.season < 2000 or args.season > 2100:
        parser.error("--season must be between 2000 and 2100")

    return args


def clean(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip()

    if text.lower() in {"none", "nan", "null"}:
        return ""

    return text


def key(value: Any) -> str:
    return clean(value).casefold()


def venue_key(value: Any) -> str:
    text = clean(value)

    if not text:
        return ""

    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.casefold()

    text = re.sub(
        r"\b(stadium|stadion|estadio|arena)\b",
        " ",
        text,
    )
    return re.sub(r"[^a-z0-9]+", "", text)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise ScheduleError(f"Missing required file: {path}")

    rows: list[dict[str, str]] = []

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ScheduleError(f"Missing header row: {path}")

        for row in reader:
            rows.append(
                {
                    clean(column): clean(value)
                    for column, value in row.items()
                }
            )

    return rows


def read_existing_output(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    rows: list[dict[str, str]] = []

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ScheduleError(f"Missing header row: {path}")

        missing = [
            column
            for column in OUTPUT_COLUMNS
            if column not in reader.fieldnames
        ]

        if missing:
            raise ScheduleError(
                f"{path} missing required columns: {missing}"
            )

        for row in reader:
            rows.append(
                {
                    column: clean(row.get(column))
                    for column in OUTPUT_COLUMNS
                }
            )

    return rows


def require_columns(
    rows: list[dict[str, str]],
    required_cols: list[str],
    file_label: str,
) -> None:
    if not rows:
        raise ScheduleError(f"{file_label} has no data rows")

    available = set(rows[0].keys())
    missing = [
        column
        for column in required_cols
        if column not in available
    ]

    if missing:
        raise ScheduleError(
            f"{file_label} missing required columns: {missing}"
        )


def build_team_maps(
    team_rows: list[dict[str, str]],
    log: RunLog,
) -> tuple[list[str], dict[str, str]]:
    require_columns(
        rows=team_rows,
        required_cols=[TEAM_ID_COLUMN, CANONICAL_TEAM_COLUMN],
        file_label=str(TEAM_MAP_FILE),
    )

    team_ids: list[str] = []
    seen_team_ids: set[str] = set()
    team_id_to_canonical: dict[str, str] = {}
    team_lookup: dict[str, str] = {}

    optional_lookup_columns = [
        TEAM_ID_COLUMN,
        "team_abbr",
        "source_name",
        "canonical_team",
    ]

    for row_number, row in enumerate(team_rows, start=2):
        team_id = clean(row.get(TEAM_ID_COLUMN))
        canonical_team = clean(row.get(CANONICAL_TEAM_COLUMN))

        if not team_id:
            log.error(
                f"team_map row {row_number} missing {TEAM_ID_COLUMN}",
                row_number=row_number,
            )

        if not canonical_team:
            log.error(
                f"team_map row {row_number} missing "
                f"{CANONICAL_TEAM_COLUMN}",
                row_number=row_number,
            )
            continue

        if team_id:
            prior_canonical = team_id_to_canonical.get(team_id)

            if (
                prior_canonical is not None
                and prior_canonical != canonical_team
            ):
                log.error(
                    f"team_map TEAM_ID={team_id} maps to multiple "
                    f"canonical teams: {prior_canonical!r} and "
                    f"{canonical_team!r}",
                    row_number=row_number,
                    team_id=team_id,
                )
            else:
                team_id_to_canonical[team_id] = canonical_team

            if team_id not in seen_team_ids:
                team_ids.append(team_id)
                seen_team_ids.add(team_id)

        for column in optional_lookup_columns:
            value = clean(row.get(column))

            if value:
                lookup_key = key(value)
                prior_lookup = team_lookup.get(lookup_key)

                if (
                    prior_lookup is not None
                    and prior_lookup != canonical_team
                ):
                    log.error(
                        f"team_map value {value!r} maps to multiple "
                        f"canonical teams: {prior_lookup!r} and "
                        f"{canonical_team!r}",
                        row_number=row_number,
                        lookup_value=value,
                    )
                else:
                    team_lookup[lookup_key] = canonical_team

    if not team_ids:
        raise ScheduleError(
            f"No TEAM_ID values found in {TEAM_MAP_FILE} "
            f"column {TEAM_ID_COLUMN}"
        )

    return team_ids, team_lookup


def build_stadium_maps(
    stadium_rows: list[dict[str, str]],
    log: RunLog,
) -> tuple[
    dict[str, dict[str, str]],
    dict[str, dict[str, str]],
]:
    require_columns(
        rows=stadium_rows,
        required_cols=[
            "team",
            "stadium",
            "timezone",
            "surface",
            "roof_type",
        ],
        file_label=str(STADIUM_MAP_FILE),
    )

    by_team: dict[str, dict[str, str]] = {}
    by_stadium: dict[str, dict[str, str]] = {}

    for row_number, row in enumerate(stadium_rows, start=2):
        team_value = clean(row.get("team"))
        stadium_value = clean(row.get("stadium"))
        venue_full_name = clean(row.get("venue_full_name"))

        if team_value:
            team_lookup_key = key(team_value)

            if team_lookup_key in by_team:
                log.error(
                    f"stadium_map row {row_number} duplicates "
                    f"team={team_value!r}",
                    row_number=row_number,
                    team=team_value,
                )
            else:
                by_team[team_lookup_key] = row

        if not stadium_value:
            log.error(
                f"stadium_map row {row_number} missing stadium",
                row_number=row_number,
            )
            continue

        for candidate in (stadium_value, venue_full_name):
            stadium_lookup_key = venue_key(candidate)

            if not stadium_lookup_key:
                continue

            existing = by_stadium.get(stadium_lookup_key)

            if existing is not None and existing is not row:
                comparable_columns = (
                    "stadium",
                    "timezone",
                    "surface",
                    "roof_type",
                )
                conflicts = [
                    column
                    for column in comparable_columns
                    if clean(existing.get(column))
                    != clean(row.get(column))
                ]

                if conflicts:
                    log.error(
                        f"stadium_map row {row_number} conflicts with "
                        f"an existing venue key for {candidate!r}; "
                        f"columns={conflicts}",
                        row_number=row_number,
                        venue=candidate,
                        conflicting_columns=conflicts,
                    )

                continue

            by_stadium[stadium_lookup_key] = row

    return by_team, by_stadium


def normalize_core_ref(
    value: Any,
    *,
    label: str,
) -> str:
    ref = clean(value)

    if not ref:
        raise ScheduleError(f"{label} is blank")

    parsed = urllib.parse.urlsplit(ref)

    if parsed.scheme not in {"http", "https"}:
        raise ScheduleError(
            f"{label} has unsupported scheme={parsed.scheme!r}"
        )

    if parsed.netloc.casefold() != "sports.core.api.espn.com":
        raise ScheduleError(
            f"{label} has unexpected host={parsed.netloc!r}"
        )

    return urllib.parse.urlunsplit(
        (
            "https",
            "sports.core.api.espn.com",
            parsed.path,
            parsed.query,
            "",
        )
    )


def get_ref_segment(
    value: Any,
    segment: str,
) -> str:
    if not isinstance(value, dict):
        return ""

    ref = clean(value.get("$ref"))

    if not ref:
        return ""

    try:
        path = urllib.parse.urlsplit(ref).path
    except Exception:
        return ""

    parts = [
        part
        for part in path.split("/")
        if part
    ]

    try:
        index = parts.index(segment)
    except ValueError:
        return ""

    if index + 1 >= len(parts):
        return ""

    return clean(parts[index + 1])


def fetch_core_json(
    url: str,
    *,
    label: str,
    log: RunLog,
) -> dict[str, Any] | None:
    request = urllib.request.Request(
        url=url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
        },
        method="GET",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=30,
        ) as response:
            body = response.read().decode("utf-8")

        data = json.loads(body)

    except urllib.error.HTTPError as exc:
        log.error(
            f"HTTP error for {label}: "
            f"{exc.code} {exc.reason}",
            source_label=label,
            http_status=exc.code,
            url=url,
        )
        return None

    except urllib.error.URLError as exc:
        log.error(
            f"URL error for {label}: {exc.reason}",
            source_label=label,
            url=url,
        )
        return None

    except Exception as exc:
        log.error(
            f"Fetch failed for {label}: "
            f"{type(exc).__name__}: {exc}",
            source_label=label,
            error_type=type(exc).__name__,
            url=url,
        )
        return None

    if not isinstance(data, dict):
        log.error(
            f"{label} response is not a JSON object",
            source_label=label,
            url=url,
        )
        return None

    return data


def fetch_core_week_event_refs(
    week: int,
    season: int,
    log: RunLog,
) -> list[str] | None:
    url = (
        "https://sports.core.api.espn.com/v2/sports/"
        "football/leagues/nfl/"
        f"seasons/{season}/types/2/weeks/{week}/"
        "events?limit=100"
    )

    data = fetch_core_json(
        url,
        label=f"WEEK={week}",
        log=log,
    )

    if data is None:
        return None

    items = data.get("items")

    if not isinstance(items, list):
        log.error(
            f"WEEK={week} response missing items list",
            week=week,
            url=url,
        )
        return None

    if not items:
        log.error(
            f"WEEK={week} response contains zero event refs",
            week=week,
            url=url,
        )
        return None

    try:
        count = int(data.get("count"))
        page_count = int(data.get("pageCount"))
    except (TypeError, ValueError):
        log.error(
            f"WEEK={week} response has invalid pagination metadata",
            week=week,
            url=url,
        )
        return None

    if page_count != 1:
        log.error(
            f"WEEK={week} response unexpectedly spans "
            f"{page_count} pages",
            week=week,
            page_count=page_count,
            url=url,
        )
        return None

    if count != len(items):
        log.error(
            f"WEEK={week} response count mismatch "
            f"count={count} items={len(items)}",
            week=week,
            response_count=count,
            item_count=len(items),
            url=url,
        )
        return None

    refs: list[str] = []
    seen_refs: set[str] = set()

    for item_number, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            log.error(
                f"WEEK={week} item {item_number} "
                "is not an object",
                week=week,
                item_number=item_number,
            )
            continue

        try:
            ref = normalize_core_ref(
                item.get("$ref"),
                label=(
                    f"WEEK={week} item {item_number} $ref"
                ),
            )
        except ScheduleError as exc:
            log.error(
                str(exc),
                week=week,
                item_number=item_number,
            )
            continue

        if ref in seen_refs:
            log.error(
                f"WEEK={week} contains duplicate event ref={ref}",
                week=week,
                event_ref=ref,
            )
            continue

        seen_refs.add(ref)
        refs.append(ref)

    if len(refs) != len(items):
        return None

    return refs


def fetch_core_event(
    ref: str,
    *,
    week: int,
    log: RunLog,
) -> dict[str, Any] | None:
    try:
        url = normalize_core_ref(
            ref,
            label=f"WEEK={week} event ref",
        )
    except ScheduleError as exc:
        log.error(
            str(exc),
            week=week,
        )
        return None

    event = fetch_core_json(
        url,
        label=f"WEEK={week} EVENT",
        log=log,
    )

    if event is None:
        return None

    event_id = clean(event.get("id"))

    if not event_id:
        log.error(
            f"WEEK={week} event response missing id",
            week=week,
            url=url,
        )
        return None

    competitions = event.get("competitions")

    if (
        not isinstance(competitions, list)
        or len(competitions) != 1
        or not isinstance(competitions[0], dict)
    ):
        log.error(
            f"WEEK={week} EVENT={event_id} must contain "
            "exactly one competition object",
            week=week,
            game_id=event_id,
            url=url,
        )
        return None

    return event

def get_first_competition(
    event: dict[str, Any],
) -> dict[str, Any]:
    competitions = event.get("competitions")

    if isinstance(competitions, list) and competitions:
        first = competitions[0]

        if isinstance(first, dict):
            return first

    return {}


def get_team_by_home_away(
    competition: dict[str, Any],
    home_away: str,
) -> dict[str, Any]:
    competitors = competition.get("competitors")

    if not isinstance(competitors, list):
        return {}

    for competitor in competitors:
        if not isinstance(competitor, dict):
            continue

        if (
            clean(competitor.get("homeAway")).casefold()
            == home_away.casefold()
        ):
            team = competitor.get("team")

            if isinstance(team, dict):
                resolved_team = dict(team)

                if not clean(resolved_team.get("id")):
                    resolved_team["id"] = (
                        clean(competitor.get("id"))
                        or get_ref_segment(
                            resolved_team,
                            "teams",
                        )
                    )

                return resolved_team

    return {}


def map_team_name(
    team: dict[str, Any],
    team_lookup: dict[str, str],
    game_id: str,
    side: str,
    log: RunLog,
) -> str:
    candidates = [
        team.get("id"),
        team.get("displayName"),
        team.get("abbreviation"),
        team.get("shortDisplayName"),
        team.get("name"),
        team.get("location"),
        team.get("nickname"),
    ]

    for candidate in candidates:
        mapped = team_lookup.get(key(candidate))

        if mapped:
            return mapped

    log.error(
        "unmapped team "
        f"game_id={game_id} side={side} "
        f"id={clean(team.get('id'))} "
        f"displayName={clean(team.get('displayName'))} "
        f"abbreviation={clean(team.get('abbreviation'))}",
        game_id=game_id,
        side=side,
    )

    return ""


def get_bool_text(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"

    text = clean(value).casefold()

    if text in {"true", "1", "yes", "y"}:
        return "1"

    if text in {"false", "0", "no", "n"}:
        return "0"

    return ""


def get_stadium_row(
    *,
    home_team: str,
    espn_stadium: str,
    neutral_site: str,
    stadium_by_team: dict[str, dict[str, str]],
    stadium_by_stadium: dict[str, dict[str, str]],
    game_id: str,
    log: RunLog,
) -> dict[str, str]:
    if neutral_site == "1":
        stadium_match = stadium_by_stadium.get(
            venue_key(espn_stadium)
        )

        if stadium_match:
            return stadium_match

        log.error(
            "neutral-site stadium not mapped "
            f"game_id={game_id} stadium={espn_stadium!r}",
            game_id=game_id,
            stadium=espn_stadium,
        )
        return {}

    home_match = stadium_by_team.get(key(home_team))

    if home_match:
        return home_match

    log.error(
        "home team stadium row not mapped "
        f"game_id={game_id} home_team={home_team!r}",
        game_id=game_id,
        home_team=home_team,
    )

    return {}


def get_team_timezone(
    team: str,
    stadium_by_team: dict[str, dict[str, str]],
    game_id: str,
    side: str,
    log: RunLog,
) -> str:
    row = stadium_by_team.get(key(team), {})
    timezone_value = clean(row.get("timezone"))

    if not timezone_value:
        log.error(
            f"missing {side}_timezone "
            f"game_id={game_id} team={team!r}",
            game_id=game_id,
            side=side,
            team=team,
        )

    return timezone_value


def parse_event_datetime(
    *,
    raw_date: str,
    game_timezone: str,
    game_id: str,
    log: RunLog,
) -> tuple[str, str]:
    if not raw_date:
        log.error(
            f"missing event.date game_id={game_id}",
            game_id=game_id,
        )
        return "", ""

    try:
        dt_utc = datetime.fromisoformat(
            raw_date.replace("Z", "+00:00")
        )

        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=timezone.utc)

    except Exception as exc:
        log.error(
            f"could not parse event.date game_id={game_id} "
            f"date={raw_date!r}: {exc}",
            game_id=game_id,
            raw_date=raw_date,
        )
        return "", ""

    if not game_timezone:
        log.error(
            "missing game_timezone; UTC fallback is not permitted "
            f"game_id={game_id}",
            game_id=game_id,
        )
        return "", ""

    try:
        dt_local = dt_utc.astimezone(ZoneInfo(game_timezone))
    except Exception as exc:
        log.error(
            "invalid game_timezone; UTC fallback is not permitted "
            f"game_id={game_id} "
            f"game_timezone={game_timezone!r}: {exc}",
            game_id=game_id,
            game_timezone=game_timezone,
        )
        return "", ""

    return (
        dt_local.strftime("%Y-%m-%d"),
        dt_local.strftime("%H:%M"),
    )


def build_row(
    *,
    event: dict[str, Any],
    expected_season: int,
    team_lookup: dict[str, str],
    stadium_by_team: dict[str, dict[str, str]],
    stadium_by_stadium: dict[str, dict[str, str]],
    log: RunLog,
) -> dict[str, str] | None:
    game_id = clean(event.get("id"))

    if not game_id:
        log.error("skipped event with missing id")
        return None

    competition = get_first_competition(event)

    if not competition:
        log.error(
            f"missing competition data game_id={game_id}",
            game_id=game_id,
        )
        return None

    home_team_obj = get_team_by_home_away(
        competition,
        "home",
    )
    away_team_obj = get_team_by_home_away(
        competition,
        "away",
    )

    home_team = (
        map_team_name(
            home_team_obj,
            team_lookup,
            game_id,
            "home",
            log,
        )
        if home_team_obj
        else ""
    )
    away_team = (
        map_team_name(
            away_team_obj,
            team_lookup,
            game_id,
            "away",
            log,
        )
        if away_team_obj
        else ""
    )

    if not home_team_obj:
        log.error(
            f"missing home competitor game_id={game_id}",
            game_id=game_id,
        )

    if not away_team_obj:
        log.error(
            f"missing away competitor game_id={game_id}",
            game_id=game_id,
        )

    neutral_site = get_bool_text(
        competition.get("neutralSite")
    )

    if neutral_site not in {"0", "1"}:
        log.error(
            f"missing or invalid neutral_site game_id={game_id}",
            game_id=game_id,
        )

    venue = competition.get("venue")

    if not isinstance(venue, dict):
        venue = {}

    espn_stadium = clean(venue.get("fullName"))

    if not espn_stadium:
        log.error(
            f"missing venue.fullName game_id={game_id}",
            game_id=game_id,
        )

    stadium_row = get_stadium_row(
        home_team=home_team,
        espn_stadium=espn_stadium,
        neutral_site=neutral_site,
        stadium_by_team=stadium_by_team,
        stadium_by_stadium=stadium_by_stadium,
        game_id=game_id,
        log=log,
    )

    stadium = clean(stadium_row.get("stadium"))
    roof = clean(stadium_row.get("roof_type"))
    surface = clean(stadium_row.get("surface"))

    if not stadium:
        log.error(
            f"missing stadium game_id={game_id}",
            game_id=game_id,
        )

    if not roof:
        log.error(
            f"missing roof game_id={game_id}",
            game_id=game_id,
        )

    if not surface:
        log.error(
            f"missing surface game_id={game_id}",
            game_id=game_id,
        )

    home_timezone = get_team_timezone(
        home_team,
        stadium_by_team,
        game_id,
        "home",
        log,
    )
    away_timezone = get_team_timezone(
        away_team,
        stadium_by_team,
        game_id,
        "away",
        log,
    )

    game_timezone = clean(stadium_row.get("timezone"))

    if not game_timezone:
        log.error(
            f"missing game_timezone game_id={game_id}",
            game_id=game_id,
        )

    game_date, game_time = parse_event_datetime(
        raw_date=clean(event.get("date")),
        game_timezone=game_timezone,
        game_id=game_id,
        log=log,
    )

    season = ""
    season_obj = event.get("season")

    if isinstance(season_obj, dict):
        season = (
            clean(season_obj.get("year"))
            or get_ref_segment(
                season_obj,
                "seasons",
            )
        )

    if season != str(expected_season):
        log.error(
            f"event season mismatch game_id={game_id}: "
            f"expected={expected_season} actual={season!r}",
            game_id=game_id,
            expected_season=expected_season,
            actual_season=season,
        )

    season_type = ""
    season_type_obj = event.get("seasonType")

    if isinstance(season_type_obj, dict):
        season_type = clean(
            season_type_obj.get("abbreviation")
        )

    if not season_type:
        season_type_code = (
            get_ref_segment(
                season_type_obj,
                "types",
            )
            or (
                clean(season_obj.get("type"))
                if isinstance(season_obj, dict)
                else ""
            )
        )
        season_slug = (
            clean(
                season_obj.get("slug")
            ).casefold()
            if isinstance(season_obj, dict)
            else ""
        )

        season_type = {
            "1": "pre",
            "2": "reg",
            "3": "post",
        }.get(season_type_code, "")

        if not season_type:
            season_type = {
                "preseason": "pre",
                "regular-season": "reg",
                "postseason": "post",
                "post-season": "post",
            }.get(season_slug, "")

    if not season_type:
        log.error(
            f"missing season_type game_id={game_id}",
            game_id=game_id,
        )

    week = ""
    week_obj = event.get("week")

    if isinstance(week_obj, dict):
        week = (
            clean(week_obj.get("number"))
            or get_ref_segment(
                week_obj,
                "weeks",
            )
        )

    if not week:
        log.error(
            f"missing week game_id={game_id}",
            game_id=game_id,
        )

    if not game_date:
        log.error(
            f"missing game_date game_id={game_id}",
            game_id=game_id,
        )

    if not game_time:
        log.error(
            f"missing game_time game_id={game_id}",
            game_id=game_id,
        )

    return {
        "season": season,
        "season_type": season_type,
        "week": week,
        "game_id": game_id,
        "game_date": game_date,
        "game_time": game_time,
        "away_team": away_team,
        "home_team": home_team,
        "neutral_site": neutral_site,
        "stadium": stadium,
        "roof": roof,
        "surface": surface,
        "home_timezone": home_timezone,
        "away_timezone": away_timezone,
        "game_timezone": game_timezone,
    }


def rows_equal(
    first: dict[str, str],
    second: dict[str, str],
) -> bool:
    return all(
        clean(first.get(column))
        == clean(second.get(column))
        for column in OUTPUT_COLUMNS
    )


def changed_columns(
    first: dict[str, str],
    second: dict[str, str],
) -> list[str]:
    return [
        column
        for column in OUTPUT_COLUMNS
        if clean(first.get(column))
        != clean(second.get(column))
    ]


def validate_pulled_rows(
    rows: list[dict[str, str]],
    *,
    season: int,
    log: RunLog,
) -> None:
    if not rows:
        log.error(
            "ESPN schedule collection produced zero unique games."
        )
        return

    required_nonblank = [
        column
        for column in OUTPUT_COLUMNS
        if column != "neutral_site"
    ]

    game_ids: set[str] = set()

    for row_number, row in enumerate(rows, start=2):
        game_id = clean(row.get("game_id"))

        if not game_id:
            log.error(
                f"pulled row {row_number} has blank game_id",
                row_number=row_number,
            )
            continue

        if game_id in game_ids:
            log.error(
                f"pulled schedule contains duplicate game_id={game_id}",
                row_number=row_number,
                game_id=game_id,
            )
        else:
            game_ids.add(game_id)

        if clean(row.get("season")) != str(season):
            log.error(
                f"pulled row {row_number} has wrong season "
                f"game_id={game_id}",
                row_number=row_number,
                game_id=game_id,
            )

        if clean(row.get("neutral_site")) not in {"0", "1"}:
            log.error(
                f"pulled row {row_number} has invalid neutral_site "
                f"game_id={game_id}",
                row_number=row_number,
                game_id=game_id,
            )

        missing = [
            column
            for column in required_nonblank
            if not clean(row.get(column))
        ]

        if missing:
            log.error(
                f"pulled row {row_number} has blank required "
                f"field(s) game_id={game_id}: {missing}",
                row_number=row_number,
                game_id=game_id,
                missing_columns=missing,
            )


def write_csv_staged(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
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
                    column: clean(row.get(column))
                    for column in OUTPUT_COLUMNS
                }
            )

        handle.flush()
        os.fsync(handle.fileno())


def get_updates_file(
    season: int,
) -> Path:
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    return (
        UPDATES_DIR
        / f"{season}_schedule_{timestamp}.csv"
    )


def publish_schedule_outputs(
    *,
    staged_canonical: Path,
    staged_update: Path,
    output_file: Path,
    updates_file: Path,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    UPDATES_DIR.mkdir(parents=True, exist_ok=True)

    if updates_file.exists():
        raise ScheduleError(
            f"Per-run updates file already exists: {updates_file}"
        )

    backup_dir = Path(
        tempfile.mkdtemp(
            prefix=".pull_schedule_backup_",
            dir=OUTPUT_DIR,
        )
    )
    canonical_backup = backup_dir / output_file.name
    update_published = False
    canonical_published = False
    canonical_backed_up = False

    try:
        if output_file.exists():
            os.replace(
                output_file,
                canonical_backup,
            )
            canonical_backed_up = True

        try:
            os.replace(
                staged_update,
                updates_file,
            )
            update_published = True

            os.replace(
                staged_canonical,
                output_file,
            )
            canonical_published = True

        except Exception:
            if canonical_published:
                output_file.unlink(missing_ok=True)

            if update_published:
                updates_file.unlink(missing_ok=True)

            if canonical_backed_up:
                os.replace(
                    canonical_backup,
                    output_file,
                )

            raise

    finally:
        shutil.rmtree(
            backup_dir,
            ignore_errors=True,
        )


def run(season: int) -> int:
    output_file = OUTPUT_DIR / f"{season}_schedule.csv"
    updates_file = get_updates_file(season)

    with PipelineReporter(
        script=SCRIPT_PATH,
        stage="00_intake",
        report_root=REPORT_ROOT,
        pipeline="NFL",
        league="NFL",
        season=season,
        extra_context={
            "component": "ESPN season schedule pull",
        },
    ) as reporter:
        log = RunLog(reporter)

        reporter.add_input(TEAM_MAP_FILE)
        reporter.add_input(STADIUM_MAP_FILE)

        if output_file.exists():
            reporter.add_input(output_file)

        reporter.add_output(LEGACY_LOG_FILE)

        api_calls_attempted = 0
        api_calls_succeeded = 0
        week_calls_attempted = 0
        week_calls_succeeded = 0
        event_calls_attempted = 0
        event_calls_succeeded = 0
        events_seen = 0
        duplicate_events_seen = 0
        duplicate_events_changed = 0
        existing_rows_count = 0
        pulled_unique_games = 0
        removed_existing_games = 0
        added_rows = 0
        updated_rows = 0
        unchanged_rows = 0
        publication_completed = False
        published_outputs: list[Path] = []

        log.info("pull_schedule.py started")
        log.info(f"season={season}")
        log.info(f"TEAM_MAP_FILE={TEAM_MAP_FILE}")
        log.info(f"STADIUM_MAP_FILE={STADIUM_MAP_FILE}")
        log.info(f"OUTPUT_FILE={output_file}")
        log.info(f"UPDATES_FILE={updates_file}")

        try:
            team_rows = read_csv(TEAM_MAP_FILE)
            stadium_rows = read_csv(STADIUM_MAP_FILE)
            existing_rows = read_existing_output(
                output_file
            )

            existing_rows_count = len(existing_rows)

            (
                team_ids,
                team_lookup,
            ) = build_team_maps(
                team_rows,
                log,
            )
            (
                stadium_by_team,
                stadium_by_stadium,
            ) = build_stadium_maps(
                stadium_rows,
                log,
            )

            log.info(
                f"team_ids_found={len(team_ids)}"
            )
            log.info(
                f"existing_rows_found="
                f"{existing_rows_count}"
            )

            pulled_rows_by_game_id: dict[
                str,
                dict[str, str],
            ] = {}

            week_calls_attempted = 0
            week_calls_succeeded = 0
            event_calls_attempted = 0
            event_calls_succeeded = 0

            for week in REGULAR_SEASON_WEEKS:
                week_calls_attempted += 1
                api_calls_attempted += 1

                event_refs = fetch_core_week_event_refs(
                    week,
                    season,
                    log,
                )

                if event_refs is None:
                    continue

                week_calls_succeeded += 1
                api_calls_succeeded += 1

                log.info(
                    f"WEEK={week} "
                    f"event_refs_returned={len(event_refs)}"
                )

                for event_ref in event_refs:
                    event_calls_attempted += 1
                    api_calls_attempted += 1

                    event = fetch_core_event(
                        event_ref,
                        week=week,
                        log=log,
                    )

                    if event is None:
                        continue

                    event_calls_succeeded += 1
                    api_calls_succeeded += 1
                    events_seen += 1

                    row = build_row(
                        event=event,
                        expected_season=season,
                        team_lookup=team_lookup,
                        stadium_by_team=stadium_by_team,
                        stadium_by_stadium=(
                            stadium_by_stadium
                        ),
                        log=log,
                    )

                    if row is None:
                        continue

                    game_id = clean(
                        row.get("game_id")
                    )

                    previous_row = (
                        pulled_rows_by_game_id.get(
                            game_id
                        )
                    )

                    if previous_row is not None:
                        duplicate_events_seen += 1

                        if not rows_equal(
                            previous_row,
                            row,
                        ):
                            duplicate_events_changed += 1
                            log.error(
                                "duplicate game_id pulled "
                                "with conflicting rows "
                                f"game_id={game_id} "
                                f"WEEK={week} "
                                f"changed_columns="
                                f"{changed_columns(previous_row, row)}",
                                game_id=game_id,
                                week=week,
                            )

                        continue

                    pulled_rows_by_game_id[
                        game_id
                    ] = row

            if (
                week_calls_succeeded
                != len(REGULAR_SEASON_WEEKS)
            ):
                log.error(
                    "Incomplete ESPN Core week source: "
                    f"{week_calls_succeeded}/"
                    f"{len(REGULAR_SEASON_WEEKS)} regular-season "
                    "week collections returned valid event refs. "
                    "No schedule output will be published.",
                    week_calls_attempted=(
                        week_calls_attempted
                    ),
                    week_calls_succeeded=(
                        week_calls_succeeded
                    ),
                    configured_week_count=(
                        len(REGULAR_SEASON_WEEKS)
                    ),
                )

            if (
                event_calls_attempted == 0
                or event_calls_succeeded
                != event_calls_attempted
            ):
                log.error(
                    "Incomplete ESPN Core event source: "
                    f"{event_calls_succeeded}/"
                    f"{event_calls_attempted} event refs "
                    "resolved to valid event objects. "
                    "No schedule output will be published.",
                    event_calls_attempted=(
                        event_calls_attempted
                    ),
                    event_calls_succeeded=(
                        event_calls_succeeded
                    ),
                )

            pulled_rows = list(
                pulled_rows_by_game_id.values()
            )
            pulled_unique_games = len(pulled_rows)

            validate_pulled_rows(
                pulled_rows,
                season=season,
                log=log,
            )

            existing_by_id: dict[
                str,
                dict[str, str],
            ] = {}

            for row in existing_rows:
                game_id = clean(
                    row.get("game_id")
                )

                if not game_id:
                    log.error(
                        "existing canonical schedule "
                        "contains blank game_id"
                    )
                    continue

                if game_id in existing_by_id:
                    log.error(
                        "existing canonical schedule "
                        "contains duplicate game_id="
                        f"{game_id}",
                        game_id=game_id,
                    )
                    continue

                existing_by_id[game_id] = row

            for game_id, row in (
                pulled_rows_by_game_id.items()
            ):
                existing_row = existing_by_id.get(
                    game_id
                )

                if existing_row is None:
                    added_rows += 1
                elif rows_equal(
                    existing_row,
                    row,
                ):
                    unchanged_rows += 1
                else:
                    updated_rows += 1

            removed_existing_games = sum(
                1
                for game_id in existing_by_id
                if game_id
                not in pulled_rows_by_game_id
            )

            if not log.has_errors:
                OUTPUT_DIR.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                with tempfile.TemporaryDirectory(
                    prefix=".pull_schedule_stage_",
                    dir=OUTPUT_DIR,
                ) as staging_name:
                    staging_dir = Path(staging_name)
                    staged_canonical = (
                        staging_dir
                        / output_file.name
                    )
                    staged_update = (
                        staging_dir
                        / updates_file.name
                    )

                    write_csv_staged(
                        staged_canonical,
                        pulled_rows,
                    )
                    write_csv_staged(
                        staged_update,
                        pulled_rows,
                    )

                    publish_schedule_outputs(
                        staged_canonical=(
                            staged_canonical
                        ),
                        staged_update=staged_update,
                        output_file=output_file,
                        updates_file=updates_file,
                    )

                publication_completed = True
                published_outputs = [
                    output_file,
                    updates_file,
                ]

                for path in published_outputs:
                    reporter.add_output(path)

                log.info(
                    f"Wrote {pulled_unique_games} "
                    f"authoritative rows to "
                    f"{output_file}"
                )
                log.info(
                    f"Wrote {pulled_unique_games} "
                    f"pulled rows to "
                    f"{updates_file}"
                )

        except ScheduleError as exc:
            log.error(
                str(exc),
                error_type=type(exc).__name__,
            )

        except Exception as exc:
            log.error(
                f"Unexpected fatal error: "
                f"{type(exc).__name__}: {exc}",
                error_type=type(exc).__name__,
                traceback_text=traceback.format_exc(),
            )

        finally:
            reporter.set_rows(
                rows_in=events_seen,
                rows_out=(
                    pulled_unique_games
                    if publication_completed
                    else 0
                ),
            )
            reporter.update_details(
                {
                    "requested_season": season,
                    "source_url_template": (
                        "https://sports.core.api.espn.com/"
                        "v2/sports/football/leagues/nfl/"
                        f"seasons/{season}/types/2/"
                        "weeks/{WEEK}/events?limit=100"
                    ),
                    "source_event_ref_type": (
                        "sports.core.api.espn.com event $ref"
                    ),
                    "source_week_count": (
                        len(REGULAR_SEASON_WEEKS)
                    ),
                    "week_calls_attempted": (
                        week_calls_attempted
                    ),
                    "week_calls_succeeded": (
                        week_calls_succeeded
                    ),
                    "event_calls_attempted": (
                        event_calls_attempted
                    ),
                    "event_calls_succeeded": (
                        event_calls_succeeded
                    ),
                    "api_calls_attempted": (
                        api_calls_attempted
                    ),
                    "api_calls_succeeded": (
                        api_calls_succeeded
                    ),
                    "events_seen": events_seen,
                    "duplicate_events_seen": (
                        duplicate_events_seen
                    ),
                    "duplicate_events_changed": (
                        duplicate_events_changed
                    ),
                    "existing_rows_found": (
                        existing_rows_count
                    ),
                    "pulled_unique_games": (
                        pulled_unique_games
                    ),
                    "added_rows": added_rows,
                    "updated_rows": updated_rows,
                    "unchanged_rows": unchanged_rows,
                    "removed_existing_games": (
                        removed_existing_games
                    ),
                    "publication_completed": (
                        publication_completed
                    ),
                    "published_outputs": [
                        str(path)
                        for path in published_outputs
                    ],
                }
            )

            log.info(
                f"api_calls_attempted="
                f"{api_calls_attempted}"
            )
            log.info(
                f"api_calls_succeeded="
                f"{api_calls_succeeded}"
            )
            log.info(
                f"events_seen={events_seen}"
            )
            log.info(
                f"duplicate_events_seen="
                f"{duplicate_events_seen}"
            )
            log.info(
                f"duplicate_events_changed="
                f"{duplicate_events_changed}"
            )
            log.info(
                f"existing_rows_found="
                f"{existing_rows_count}"
            )
            log.info(
                f"pulled_unique_games="
                f"{pulled_unique_games}"
            )
            log.info(
                f"added_rows={added_rows}"
            )
            log.info(
                f"updated_rows={updated_rows}"
            )
            log.info(
                f"unchanged_rows={unchanged_rows}"
            )
            log.info(
                f"removed_existing_games="
                f"{removed_existing_games}"
            )
            log.info(
                "publication_completed="
                f"{publication_completed}"
            )
            log.info("pull_schedule.py finished")
            log.write_legacy()

        return 1 if log.has_errors else 0


def main() -> int:
    args = parse_args()
    return run(args.season)


if __name__ == "__main__":
    sys.exit(main())
