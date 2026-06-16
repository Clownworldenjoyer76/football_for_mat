#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
docs/win/football/nfl/scripts/00_intake/pull_schedule.py

Pulls 2026 NFL schedule from ESPN team schedule API.

Source:
  https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{TEAM_ID}/schedule?season=2026

Inputs:
  docs/win/football/nfl/config/mapping/team_map_nfl.csv
  docs/win/football/nfl/config/mapping/stadium_map_nfl.csv

Output:
  docs/win/football/nfl/00_intake/schedule/2026_schedule.csv

Summary / warnings:
  docs/win/football/nfl/errors/00_intake/pull_schedule.txt
"""

from __future__ import annotations

import csv
import json
import sys
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


YEAR = 2026

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

NFL_DIR = Path(__file__).resolve().parents[2]

TEAM_MAP_FILE = NFL_DIR / "config" / "mapping" / "team_map_nfl.csv"
STADIUM_MAP_FILE = NFL_DIR / "config" / "mapping" / "stadium_map_nfl.csv"

OUTPUT_DIR = NFL_DIR / "00_intake" / "schedule"
OUTPUT_FILE = OUTPUT_DIR / f"{YEAR}_schedule.csv"

ERROR_DIR = NFL_DIR / "errors" / "00_intake"
LOG_FILE = ERROR_DIR / "pull_schedule.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)


def clean(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip()

    if text.lower() in {"none", "nan", "null"}:
        return ""

    return text


def key(value: Any) -> str:
    return clean(value).casefold()


def reset_log() -> None:
    LOG_FILE.write_text("", encoding="utf-8")


def log(message: str) -> None:
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")


def fatal(message: str) -> None:
    log(f"ERROR: {message}")
    sys.exit(f"ERROR: {message}")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        fatal(f"Missing required file: {path}")

    rows: list[dict[str, str]] = []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            fatal(f"Missing header row: {path}")

        for row in reader:
            rows.append({clean(k): clean(v) for k, v in row.items()})

    return rows


def require_columns(rows: list[dict[str, str]], required_cols: list[str], file_label: str) -> None:
    if not rows:
        fatal(f"{file_label} has no data rows")

    available = set(rows[0].keys())
    missing = [col for col in required_cols if col not in available]

    if missing:
        fatal(f"{file_label} missing required columns: {missing}")


def build_team_maps(team_rows: list[dict[str, str]]) -> tuple[list[str], dict[str, str]]:
    require_columns(
        rows=team_rows,
        required_cols=[TEAM_ID_COLUMN, CANONICAL_TEAM_COLUMN],
        file_label=str(TEAM_MAP_FILE),
    )

    team_ids: list[str] = []
    seen_team_ids: set[str] = set()
    team_lookup: dict[str, str] = {}

    optional_lookup_columns = [
        TEAM_ID_COLUMN,
        "team_id",
        "team_abbr",
        "source_name",
        "canonical_team",
        "sports.leagues.teams.team.abbreviation",
        "sports.leagues.teams.team.displayName",
        "sports.leagues.teams.team.shortDisplayName",
        "sports.leagues.teams.team.name",
        "sports.leagues.teams.team.location",
        "sports.leagues.teams.team.nickname",
    ]

    for row_number, row in enumerate(team_rows, start=2):
        team_id = clean(row.get(TEAM_ID_COLUMN))
        canonical_team = clean(row.get(CANONICAL_TEAM_COLUMN))

        if not team_id:
            log(f"WARNING: team_map row {row_number} missing {TEAM_ID_COLUMN}")
        elif team_id not in seen_team_ids:
            team_ids.append(team_id)
            seen_team_ids.add(team_id)

        if not canonical_team:
            log(f"WARNING: team_map row {row_number} missing {CANONICAL_TEAM_COLUMN}")
            continue

        for col in optional_lookup_columns:
            value = clean(row.get(col))
            if value:
                team_lookup[key(value)] = canonical_team

    if not team_ids:
        fatal(f"No TEAM_ID values found in {TEAM_MAP_FILE} column {TEAM_ID_COLUMN}")

    return team_ids, team_lookup


def build_stadium_maps(stadium_rows: list[dict[str, str]]) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    require_columns(
        rows=stadium_rows,
        required_cols=["team", "stadium", "timezone", "surface", "roof_type"],
        file_label=str(STADIUM_MAP_FILE),
    )

    by_team: dict[str, dict[str, str]] = {}
    by_stadium: dict[str, dict[str, str]] = {}

    for row_number, row in enumerate(stadium_rows, start=2):
        team_value = clean(row.get("team"))
        stadium_value = clean(row.get("stadium"))

        if team_value:
            by_team[key(team_value)] = row
        else:
            log(f"WARNING: stadium_map row {row_number} missing team")

        if stadium_value:
            by_stadium[key(stadium_value)] = row
        else:
            log(f"WARNING: stadium_map row {row_number} missing stadium")

    return by_team, by_stadium


def fetch_team_schedule(team_id: str) -> dict[str, Any] | None:
    url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team_id}/schedule?season={YEAR}"

    request = urllib.request.Request(
        url=url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
        },
        method="GET",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
            return json.loads(body)

    except urllib.error.HTTPError as e:
        log(f"WARNING: HTTP error for TEAM_ID={team_id}: {e.code} {e.reason}")
        return None

    except urllib.error.URLError as e:
        log(f"WARNING: URL error for TEAM_ID={team_id}: {e.reason}")
        return None

    except Exception as e:
        log(f"WARNING: Fetch failed for TEAM_ID={team_id}: {e}")
        return None


def get_first_competition(event: dict[str, Any]) -> dict[str, Any]:
    competitions = event.get("competitions")

    if isinstance(competitions, list) and competitions:
        first = competitions[0]
        if isinstance(first, dict):
            return first

    return {}


def get_team_by_home_away(competition: dict[str, Any], home_away: str) -> dict[str, Any]:
    competitors = competition.get("competitors")

    if not isinstance(competitors, list):
        return {}

    for competitor in competitors:
        if not isinstance(competitor, dict):
            continue

        if clean(competitor.get("homeAway")).casefold() == home_away.casefold():
            team = competitor.get("team")
            if isinstance(team, dict):
                return team

    return {}


def map_team_name(team: dict[str, Any], team_lookup: dict[str, str], game_id: str, side: str) -> str:
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

    log(
        "WARNING: unmapped team "
        f"game_id={game_id} side={side} "
        f"id={clean(team.get('id'))} "
        f"displayName={clean(team.get('displayName'))} "
        f"abbreviation={clean(team.get('abbreviation'))}"
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
    home_team: str,
    espn_stadium: str,
    neutral_site: str,
    stadium_by_team: dict[str, dict[str, str]],
    stadium_by_stadium: dict[str, dict[str, str]],
    game_id: str,
) -> dict[str, str]:
    if neutral_site == "1":
        stadium_match = stadium_by_stadium.get(key(espn_stadium))
        if stadium_match:
            return stadium_match

        log(
            "WARNING: neutral-site stadium not mapped "
            f"game_id={game_id} stadium={espn_stadium}"
        )
        return {}

    home_match = stadium_by_team.get(key(home_team))
    if home_match:
        return home_match

    log(
        "WARNING: home team stadium row not mapped "
        f"game_id={game_id} home_team={home_team}"
    )

    return {}


def get_team_timezone(team: str, stadium_by_team: dict[str, dict[str, str]], game_id: str, side: str) -> str:
    row = stadium_by_team.get(key(team), {})

    timezone_value = clean(row.get("timezone"))

    if not timezone_value:
        log(f"WARNING: missing {side}_timezone game_id={game_id} team={team}")

    return timezone_value


def parse_event_datetime(
    raw_date: str,
    game_timezone: str,
    game_id: str,
) -> tuple[str, str]:
    if not raw_date:
        log(f"WARNING: missing event.date game_id={game_id}")
        return "", ""

    try:
        dt_utc = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))

        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=timezone.utc)

    except Exception as e:
        log(f"WARNING: could not parse event.date game_id={game_id} date={raw_date} error={e}")
        return "", ""

    if game_timezone:
        try:
            dt_local = dt_utc.astimezone(ZoneInfo(game_timezone))
        except Exception as e:
            log(
                "WARNING: invalid game_timezone; using UTC datetime "
                f"game_id={game_id} game_timezone={game_timezone} error={e}"
            )
            dt_local = dt_utc.astimezone(timezone.utc)
    else:
        log(f"WARNING: missing game_timezone; using UTC datetime game_id={game_id}")
        dt_local = dt_utc.astimezone(timezone.utc)

    return dt_local.strftime("%Y-%m-%d"), dt_local.strftime("%H:%M")


def build_row(
    event: dict[str, Any],
    team_lookup: dict[str, str],
    stadium_by_team: dict[str, dict[str, str]],
    stadium_by_stadium: dict[str, dict[str, str]],
) -> dict[str, str] | None:
    game_id = clean(event.get("id"))

    if not game_id:
        log("WARNING: skipped event with missing id")
        return None

    competition = get_first_competition(event)

    home_team_obj = get_team_by_home_away(competition, "home")
    away_team_obj = get_team_by_home_away(competition, "away")

    home_team = map_team_name(home_team_obj, team_lookup, game_id, "home") if home_team_obj else ""
    away_team = map_team_name(away_team_obj, team_lookup, game_id, "away") if away_team_obj else ""

    if not home_team:
        log(f"WARNING: missing mapped home_team game_id={game_id}")

    if not away_team:
        log(f"WARNING: missing mapped away_team game_id={game_id}")

    neutral_site = get_bool_text(competition.get("neutralSite"))

    if neutral_site == "":
        log(f"WARNING: missing neutral_site game_id={game_id}")

    venue = competition.get("venue")
    if not isinstance(venue, dict):
        venue = {}

    espn_stadium = clean(venue.get("fullName"))

    stadium_row = get_stadium_row(
        home_team=home_team,
        espn_stadium=espn_stadium,
        neutral_site=neutral_site,
        stadium_by_team=stadium_by_team,
        stadium_by_stadium=stadium_by_stadium,
        game_id=game_id,
    )

    stadium = clean(stadium_row.get("stadium"))
    roof = clean(stadium_row.get("roof_type"))
    surface = clean(stadium_row.get("surface"))

    if not stadium:
        log(f"WARNING: missing stadium game_id={game_id}")

    if not roof:
        log(f"WARNING: missing roof game_id={game_id}")

    if not surface:
        log(f"WARNING: missing surface game_id={game_id}")

    home_timezone = get_team_timezone(home_team, stadium_by_team, game_id, "home")
    away_timezone = get_team_timezone(away_team, stadium_by_team, game_id, "away")

    game_timezone = clean(stadium_row.get("timezone"))

    if not game_timezone:
        log(f"WARNING: missing game_timezone game_id={game_id}")

    game_date, game_time = parse_event_datetime(
        raw_date=clean(event.get("date")),
        game_timezone=game_timezone,
        game_id=game_id,
    )

    season = ""
    season_obj = event.get("season")
    if isinstance(season_obj, dict):
        season = clean(season_obj.get("year"))

    if not season:
        log(f"WARNING: missing season game_id={game_id}")

    season_type = ""
    season_type_obj = event.get("seasonType")
    if isinstance(season_type_obj, dict):
        season_type = clean(season_type_obj.get("abbreviation"))

    if not season_type:
        log(f"WARNING: missing season_type game_id={game_id}")

    week = ""
    week_obj = event.get("week")
    if isinstance(week_obj, dict):
        week = clean(week_obj.get("number"))

    if not week:
        log(f"WARNING: missing week game_id={game_id}")

    if not game_date:
        log(f"WARNING: missing game_date game_id={game_id}")

    if not game_time:
        log(f"WARNING: missing game_time game_id={game_id}")

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


def write_output(rows: list[dict[str, str]]) -> None:
    with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for row in rows:
            writer.writerow({col: clean(row.get(col)) for col in OUTPUT_COLUMNS})


def main() -> None:
    reset_log()

    log("pull_schedule.py started")
    log(f"YEAR={YEAR}")
    log(f"TEAM_MAP_FILE={TEAM_MAP_FILE}")
    log(f"STADIUM_MAP_FILE={STADIUM_MAP_FILE}")
    log(f"OUTPUT_FILE={OUTPUT_FILE}")

    try:
        team_rows = read_csv(TEAM_MAP_FILE)
        stadium_rows = read_csv(STADIUM_MAP_FILE)

        team_ids, team_lookup = build_team_maps(team_rows)
        stadium_by_team, stadium_by_stadium = build_stadium_maps(stadium_rows)

        log(f"team_ids_found={len(team_ids)}")

        rows_by_game_id: dict[str, dict[str, str]] = {}
        api_calls_attempted = 0
        api_calls_succeeded = 0
        events_seen = 0
        duplicate_events_seen = 0

        for team_id in team_ids:
            api_calls_attempted += 1

            data = fetch_team_schedule(team_id)

            if not data:
                continue

            api_calls_succeeded += 1

            events = data.get("events")

            if not isinstance(events, list):
                log(f"WARNING: TEAM_ID={team_id} response missing events list")
                continue

            log(f"TEAM_ID={team_id} events_returned={len(events)}")

            for event in events:
                if not isinstance(event, dict):
                    log(f"WARNING: TEAM_ID={team_id} skipped non-dict event")
                    continue

                events_seen += 1

                game_id = clean(event.get("id"))

                if not game_id:
                    log(f"WARNING: TEAM_ID={team_id} skipped event missing id")
                    continue

                if game_id in rows_by_game_id:
                    duplicate_events_seen += 1
                    log(f"WARNING: duplicate game_id seen and skipped game_id={game_id} TEAM_ID={team_id}")
                    continue

                row = build_row(
                    event=event,
                    team_lookup=team_lookup,
                    stadium_by_team=stadium_by_team,
                    stadium_by_stadium=stadium_by_stadium,
                )

                if row is None:
                    continue

                rows_by_game_id[game_id] = row

        output_rows = list(rows_by_game_id.values())

        write_output(output_rows)

        log(f"api_calls_attempted={api_calls_attempted}")
        log(f"api_calls_succeeded={api_calls_succeeded}")
        log(f"events_seen={events_seen}")
        log(f"duplicate_events_seen={duplicate_events_seen}")
        log(f"unique_games_written={len(output_rows)}")
        log("pull_schedule.py finished")

        print(f"Wrote {len(output_rows)} rows to {OUTPUT_FILE}")
        print(f"Summary/warnings written to {LOG_FILE}")

    except SystemExit:
        raise

    except Exception:
        log("ERROR: unhandled exception")
        log(traceback.format_exc())
        sys.exit(f"ERROR: pull_schedule.py failed. See {LOG_FILE}")


if __name__ == "__main__":
    main()
