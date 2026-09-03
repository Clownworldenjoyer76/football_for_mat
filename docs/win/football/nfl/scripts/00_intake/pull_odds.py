#!/usr/bin/env python3
# docs/win/football/nfl/scripts/00_intake/pull_odds.py
"""Pull current NFL odds from ESPN Core and preserve point-in-time snapshots.

Compatibility outputs remain unchanged:
  docs/win/football/nfl/00_intake/odds/YYYY_MM_DD_NFL_odds.csv
  docs/win/football/nfl/00_intake/odds/raw/YYYY_MM_DD_nfl_odds.json

Every run also writes immutable timestamped copies under:
  docs/win/football/nfl/00_intake/odds/snapshots/
  docs/win/football/nfl/00_intake/odds/raw/snapshots/

The normalized CSV keeps the existing downstream schema. ESPN event IDs are used
as game_id values inside the odds artifact; build_weekly_schedule.py maps them to
the canonical schedule game_id through the raw compatibility events list.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE_DIR = Path("docs/win/football/nfl")
ODDS_DIR = BASE_DIR / "00_intake" / "odds"
RAW_ODDS_DIR = ODDS_DIR / "raw"
SNAPSHOT_DIR = ODDS_DIR / "snapshots"
RAW_SNAPSHOT_DIR = RAW_ODDS_DIR / "snapshots"
ERROR_DIR = BASE_DIR / "errors" / "00_intake"

for directory in (RAW_ODDS_DIR, SNAPSHOT_DIR, RAW_SNAPSHOT_DIR, ODDS_DIR, ERROR_DIR):
    directory.mkdir(parents=True, exist_ok=True)

LOG_FILE = ERROR_DIR / "pull_odds.txt"

ESPN_CORE_BASE = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"
SEASON_TYPE = int(os.getenv("NFL_SEASON_TYPE", "2"))
EXPLICIT_SEASON = os.getenv("NFL_SEASON", "").strip()
EXPLICIT_WEEK = os.getenv("NFL_WEEK", "").strip()
MAX_REGULAR_WEEKS = 18
PREFERRED_BOOKMAKERS = ["DraftKings"]
HTTP_RETRIES = 4
HTTP_TIMEOUT = 45
WORKERS = max(1, min(int(os.getenv("NFL_ODDS_WORKERS", "8")), 16))
TARGET_WEEK_GRACE = timedelta(hours=5)

OUTPUT_COLUMNS = [
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


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat()


def log(message: str) -> None:
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"[{utc_now_iso()}] {message}\n")


def fail(message: str) -> None:
    log(f"ERROR: {message}")
    raise RuntimeError(message)


def resolve_season(now: datetime) -> int:
    if EXPLICIT_SEASON:
        return int(EXPLICIT_SEASON)
    return now.year if now.month >= 7 else now.year - 1


def secure_ref(value: object) -> str:
    return str(value or "").strip().replace("http://", "https://", 1)


def http_get_json(url: str) -> object:
    last_error = None
    for attempt in range(1, HTTP_RETRIES + 1):
        request = Request(url, headers={"User-Agent": "nfl-pull-odds-espn/1.0"})
        try:
            with urlopen(request, timeout=HTTP_TIMEOUT) as response:
                body = response.read().decode("utf-8")
                return json.loads(body)
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {exc.code}: {body[:500]}"
            if exc.code not in {408, 425, 429, 500, 502, 503, 504}:
                raise RuntimeError(last_error) from exc
        except URLError as exc:
            last_error = f"network error: {exc.reason}"
        except json.JSONDecodeError as exc:
            last_error = f"JSON parse error: {exc}"
        except Exception as exc:
            last_error = str(exc)

        if attempt < HTTP_RETRIES:
            time.sleep(min(2 ** (attempt - 1), 8))

    raise RuntimeError(f"ESPN request failed after {HTTP_RETRIES} attempts: {last_error}")


def parse_espn_datetime(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def to_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def clean_number(value: object) -> str:
    number = to_float(value)
    if number is None:
        return ""
    if number.is_integer():
        return str(int(number))
    return str(number)


def clean_american(value: object) -> str:
    number = to_float(value)
    if number is None:
        return ""
    return str(int(round(number)))


def nested(data: object, *keys: str) -> object:
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def value_from_display_object(value: object) -> object:
    if isinstance(value, dict):
        for key in ("american", "alternateDisplayValue", "value"):
            if value.get(key) not in (None, ""):
                return value.get(key)
    return value


def decimal_from_price(value: object) -> str:
    decimal = nested(value, "decimal") if isinstance(value, dict) else None
    return clean_number(decimal)


def team_id_from_ref(ref: object) -> str:
    text = secure_ref(ref)
    marker = "/teams/"
    if marker not in text:
        return ""
    return text.split(marker, 1)[1].split("?", 1)[0].split("/", 1)[0]


def week_events_url(season: int, week: int) -> str:
    return (
        f"{ESPN_CORE_BASE}/seasons/{season}/types/{SEASON_TYPE}/weeks/{week}/events"
        "?limit=100&lang=en&region=us"
    )


def odds_url(event_id: str) -> str:
    return (
        f"{ESPN_CORE_BASE}/events/{event_id}/competitions/{event_id}/odds"
        "?lang=en&region=us"
    )


def get_event_id(event_ref: str) -> str:
    marker = "/events/"
    if marker not in event_ref:
        return ""
    return event_ref.split(marker, 1)[1].split("?", 1)[0].split("/", 1)[0]


def discover_event_refs(season: int) -> dict[int, list[str]]:
    weeks: dict[int, list[str]] = {}
    week_numbers = [int(EXPLICIT_WEEK)] if EXPLICIT_WEEK else list(range(1, MAX_REGULAR_WEEKS + 1))

    for week in week_numbers:
        payload = http_get_json(week_events_url(season, week))
        items = payload.get("items", []) if isinstance(payload, dict) else []
        refs = [secure_ref(item.get("$ref")) for item in items if isinstance(item, dict) and item.get("$ref")]
        if refs:
            weeks[week] = refs

    if not weeks:
        fail(f"No ESPN NFL event references returned for season={season}, season_type={SEASON_TYPE}")

    return weeks


def fetch_event_record(week: int, event_ref: str) -> dict:
    event = http_get_json(event_ref)
    if not isinstance(event, dict):
        raise RuntimeError(f"Invalid ESPN event response: {event_ref}")

    event_id = str(event.get("id") or get_event_id(event_ref)).strip()
    event_date = str(event.get("date", "")).strip()

    return {
        "week": week,
        "event_id": event_id,
        "event_ref": event_ref,
        "date": event_date,
        "event": event,
    }


def fetch_all_event_records(week_refs: dict[int, list[str]]) -> list[dict]:
    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        future_map = {
            executor.submit(fetch_event_record, week, ref): (week, ref)
            for week, refs in week_refs.items()
            for ref in refs
        }
        for future in as_completed(future_map):
            week, ref = future_map[future]
            try:
                records.append(future.result())
            except Exception as exc:
                log(f"EVENT_FETCH_FAILED week={week} ref={ref} error={exc}")

    if not records:
        fail("No ESPN NFL event records could be fetched")

    records.sort(key=lambda r: (r["week"], r.get("date", ""), r["event_id"]))
    return records


def choose_target_week(records: list[dict], now: datetime) -> int:
    if EXPLICIT_WEEK:
        return int(EXPLICIT_WEEK)

    threshold = now - TARGET_WEEK_GRACE
    candidate_weeks = sorted(
        {
            int(record["week"])
            for record in records
            if (parse_espn_datetime(record.get("date")) or datetime.min.replace(tzinfo=timezone.utc)) >= threshold
        }
    )
    if candidate_weeks:
        return candidate_weeks[0]

    return max(int(record["week"]) for record in records)


def select_bookmaker_item(items: object) -> dict | None:
    if not isinstance(items, list):
        return None
    valid = [item for item in items if isinstance(item, dict)]
    for bookmaker in PREFERRED_BOOKMAKERS:
        for item in valid:
            provider = item.get("provider")
            if isinstance(provider, dict) and str(provider.get("name", "")).strip() == bookmaker:
                return item
    return valid[0] if valid else None


def fetch_odds_for_record(record: dict) -> dict | None:
    event_id = record["event_id"]
    payload = http_get_json(odds_url(event_id))
    if not isinstance(payload, dict):
        return None
    selected = select_bookmaker_item(payload.get("items"))
    if not selected:
        return None
    return {**record, "odds_collection": payload, "odds_item": selected}


def fetch_future_odds(records: list[dict], now: datetime) -> list[dict]:
    threshold = now - TARGET_WEEK_GRACE
    eligible = [
        record
        for record in records
        if (parse_espn_datetime(record.get("date")) or datetime.min.replace(tzinfo=timezone.utc)) >= threshold
    ]

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        future_map = {executor.submit(fetch_odds_for_record, record): record for record in eligible}
        for future in as_completed(future_map):
            record = future_map[future]
            try:
                value = future.result()
                if value:
                    results.append(value)
            except Exception as exc:
                log(f"ODDS_FETCH_FAILED event_id={record['event_id']} week={record['week']} error={exc}")

    results.sort(key=lambda r: (r["week"], r.get("date", ""), r["event_id"]))
    return results


def team_name_from_payload(team_payload: object) -> str:
    if not isinstance(team_payload, dict):
        return ""
    for key in ("displayName", "name", "shortDisplayName", "location"):
        text = str(team_payload.get(key, "")).strip()
        if text:
            return text
    return ""


def fetch_team_names(odds_records: list[dict]) -> dict[str, str]:
    refs: dict[str, str] = {}
    for record in odds_records:
        item = record["odds_item"]
        for side in ("homeTeamOdds", "awayTeamOdds"):
            ref = secure_ref(nested(item, side, "team", "$ref"))
            team_id = team_id_from_ref(ref)
            if team_id and ref:
                refs[team_id] = ref

    names: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        future_map = {executor.submit(http_get_json, ref): team_id for team_id, ref in refs.items()}
        for future in as_completed(future_map):
            team_id = future_map[future]
            try:
                names[team_id] = team_name_from_payload(future.result())
            except Exception as exc:
                log(f"TEAM_FETCH_FAILED team_id={team_id} error={exc}")
    return names


def fallback_team_names(event: dict) -> tuple[str, str]:
    name = str(event.get("name", "")).strip()
    if " at " in name:
        away, home = name.split(" at ", 1)
        return home.strip(), away.strip()
    if " vs " in name:
        away, home = name.split(" vs ", 1)
        return home.strip(), away.strip()
    return "", ""


def enrich_names(record: dict, team_names: dict[str, str]) -> tuple[str, str]:
    item = record["odds_item"]
    home_ref = nested(item, "homeTeamOdds", "team", "$ref")
    away_ref = nested(item, "awayTeamOdds", "team", "$ref")
    home = team_names.get(team_id_from_ref(home_ref), "")
    away = team_names.get(team_id_from_ref(away_ref), "")
    if home and away:
        return home, away
    fallback_home, fallback_away = fallback_team_names(record.get("event", {}))
    return home or fallback_home, away or fallback_away


def market_values(item: dict) -> dict[str, str]:
    home_current_spread = value_from_display_object(nested(item, "homeTeamOdds", "current", "pointSpread"))
    away_current_spread = value_from_display_object(nested(item, "awayTeamOdds", "current", "pointSpread"))
    current_total = value_from_display_object(nested(item, "current", "total"))

    return {
        "home_moneyline_american": clean_american(nested(item, "homeTeamOdds", "moneyLine")),
        "away_moneyline_american": clean_american(nested(item, "awayTeamOdds", "moneyLine")),
        "home_spread": clean_number(home_current_spread if home_current_spread not in (None, "") else item.get("spread")),
        "away_spread": clean_number(away_current_spread),
        "home_spread_american": clean_american(nested(item, "homeTeamOdds", "spreadOdds")),
        "away_spread_american": clean_american(nested(item, "awayTeamOdds", "spreadOdds")),
        "total": clean_number(current_total if current_total not in (None, "") else item.get("overUnder")),
        "over_american": clean_american(item.get("overOdds")),
        "under_american": clean_american(item.get("underOdds")),
    }


def add_row(rows: list[dict], *, record: dict, home: str, away: str, bookmaker: str,
            market_type: str, bet_side: str, line: object, american: object, decimal: object,
            current_fields: dict[str, str], snapshot_id: str, snapshot_fetched_at: str) -> None:
    row = {
        "snapshot_id": snapshot_id,
        "snapshot_fetched_at": snapshot_fetched_at,
        "game_id": record["event_id"],
        "commence_time": record.get("date", ""),
        "home_team": home,
        "away_team": away,
        "bookmaker": bookmaker,
        "market_type": market_type,
        "bet_side": bet_side,
        "line": clean_number(line),
        "odds_american": clean_american(american),
        "odds_decimal": clean_number(decimal),
        "last_update": snapshot_fetched_at,
    }
    row.update(current_fields)
    rows.append(row)


def normalize_record(record: dict, team_names: dict[str, str], snapshot_id: str, snapshot_fetched_at: str) -> list[dict]:
    item = record["odds_item"]
    provider = item.get("provider") if isinstance(item.get("provider"), dict) else {}
    bookmaker = str(provider.get("name", "")).strip()
    home, away = enrich_names(record, team_names)
    current_fields = market_values(item)
    rows: list[dict] = []

    add_row(
        rows,
        record=record,
        home=home,
        away=away,
        bookmaker=bookmaker,
        market_type="h2h",
        bet_side="home",
        line="",
        american=nested(item, "homeTeamOdds", "moneyLine"),
        decimal=nested(item, "homeTeamOdds", "current", "moneyLine", "decimal"),
        current_fields=current_fields,
        snapshot_id=snapshot_id,
        snapshot_fetched_at=snapshot_fetched_at,
    )
    add_row(
        rows,
        record=record,
        home=home,
        away=away,
        bookmaker=bookmaker,
        market_type="h2h",
        bet_side="away",
        line="",
        american=nested(item, "awayTeamOdds", "moneyLine"),
        decimal=nested(item, "awayTeamOdds", "current", "moneyLine", "decimal"),
        current_fields=current_fields,
        snapshot_id=snapshot_id,
        snapshot_fetched_at=snapshot_fetched_at,
    )

    add_row(
        rows,
        record=record,
        home=home,
        away=away,
        bookmaker=bookmaker,
        market_type="spreads",
        bet_side="home",
        line=current_fields["home_spread"],
        american=nested(item, "homeTeamOdds", "spreadOdds"),
        decimal=nested(item, "homeTeamOdds", "current", "spread", "decimal"),
        current_fields=current_fields,
        snapshot_id=snapshot_id,
        snapshot_fetched_at=snapshot_fetched_at,
    )
    add_row(
        rows,
        record=record,
        home=home,
        away=away,
        bookmaker=bookmaker,
        market_type="spreads",
        bet_side="away",
        line=current_fields["away_spread"],
        american=nested(item, "awayTeamOdds", "spreadOdds"),
        decimal=nested(item, "awayTeamOdds", "current", "spread", "decimal"),
        current_fields=current_fields,
        snapshot_id=snapshot_id,
        snapshot_fetched_at=snapshot_fetched_at,
    )

    add_row(
        rows,
        record=record,
        home=home,
        away=away,
        bookmaker=bookmaker,
        market_type="totals",
        bet_side="over",
        line=current_fields["total"],
        american=item.get("overOdds"),
        decimal=nested(item, "current", "over", "decimal"),
        current_fields=current_fields,
        snapshot_id=snapshot_id,
        snapshot_fetched_at=snapshot_fetched_at,
    )
    add_row(
        rows,
        record=record,
        home=home,
        away=away,
        bookmaker=bookmaker,
        market_type="totals",
        bet_side="under",
        line=current_fields["total"],
        american=item.get("underOdds"),
        decimal=nested(item, "current", "under", "decimal"),
        current_fields=current_fields,
        snapshot_id=snapshot_id,
        snapshot_fetched_at=snapshot_fetched_at,
    )

    return rows


def compatibility_event(record: dict, team_names: dict[str, str]) -> dict:
    home, away = enrich_names(record, team_names)
    return {
        "id": record["event_id"],
        "date": record.get("date", ""),
        "home": home,
        "away": away,
        "week": record.get("week", ""),
        "source": "espn_core",
    }


def raw_odds_record(record: dict, team_names: dict[str, str]) -> dict:
    home, away = enrich_names(record, team_names)
    return {
        "id": record["event_id"],
        "date": record.get("date", ""),
        "week": record.get("week", ""),
        "home": home,
        "away": away,
        "provider": record["odds_item"].get("provider", {}),
        "odds": record["odds_item"],
        "odds_collection": record["odds_collection"],
        "event": record["event"],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in OUTPUT_COLUMNS})


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    LOG_FILE.write_text("", encoding="utf-8")

    captured_at = utc_now()
    season = resolve_season(captured_at)
    run_date = captured_at.strftime("%Y_%m_%d")
    snapshot_id = captured_at.strftime("%Y_%m_%d_%H%M%S_%f")
    snapshot_fetched_at = captured_at.isoformat()

    raw_path = RAW_ODDS_DIR / f"{run_date}_nfl_odds.json"
    csv_path = ODDS_DIR / f"{run_date}_NFL_odds.csv"
    raw_snapshot_path = RAW_SNAPSHOT_DIR / f"{snapshot_id}_nfl_odds.json"
    csv_snapshot_path = SNAPSHOT_DIR / f"{snapshot_id}_NFL_odds.csv"

    log(f"ESPN season={season} season_type={SEASON_TYPE}")
    week_refs = discover_event_refs(season)
    event_records = fetch_all_event_records(week_refs)
    target_week = choose_target_week(event_records, captured_at)
    odds_records = fetch_future_odds(event_records, captured_at)

    if not odds_records:
        fail("No current/future ESPN NFL odds were returned")

    team_names = fetch_team_names(odds_records)

    rows: list[dict] = []
    for record in odds_records:
        rows.extend(normalize_record(record, team_names, snapshot_id, snapshot_fetched_at))

    target_records = [record for record in odds_records if int(record["week"]) == target_week]
    if not target_records:
        fail(f"ESPN returned no odds for target week {target_week}")

    raw_payload = {
        "snapshot_id": snapshot_id,
        "fetched_at": snapshot_fetched_at,
        "source": "espn_core",
        "season": season,
        "season_type": SEASON_TYPE,
        "target_week": target_week,
        "preferred_bookmakers": PREFERRED_BOOKMAKERS,
        "events_count": len(target_records),
        "odds_events_count": len(odds_records),
        "events": [compatibility_event(record, team_names) for record in target_records],
        "all_events": [compatibility_event(record, team_names) for record in odds_records],
        "odds": [raw_odds_record(record, team_names) for record in odds_records],
    }

    write_json(raw_path, raw_payload)
    write_json(raw_snapshot_path, raw_payload)
    write_csv(csv_path, rows)
    write_csv(csv_snapshot_path, rows)

    log(f"Target week: {target_week}")
    log(f"Compatibility events returned: {len(target_records)}")
    log(f"Odds events returned: {len(odds_records)}")
    log(f"CSV rows written: {len(rows)}")
    log(f"Current raw JSON written: {raw_path}")
    log(f"Current normalized CSV written: {csv_path}")
    log(f"Archived raw JSON written: {raw_snapshot_path}")
    log(f"Archived normalized CSV written: {csv_snapshot_path}")

    print(f"Snapshot ID: {snapshot_id}")
    print(f"ESPN season: {season}")
    print(f"Target week: {target_week}")
    print(f"Current raw JSON written: {raw_path}")
    print(f"Current normalized CSV written: {csv_path}")
    print(f"Archived raw JSON written: {raw_snapshot_path}")
    print(f"Archived normalized CSV written: {csv_snapshot_path}")
    print(f"Rows written: {len(rows)}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log(traceback.format_exc())
        print(f"ERROR: see {LOG_FILE}", file=sys.stderr)
        raise
