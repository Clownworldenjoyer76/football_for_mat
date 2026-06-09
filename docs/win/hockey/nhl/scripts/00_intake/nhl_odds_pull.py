#!/usr/bin/env python3
# docs/win/hockey/nhl/scripts/00_intake/nhl_odds_pull.py

import csv
import json
import os
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

API_KEY_ENV = "API_ODDS"
BASE_URL = "https://api.odds-api.io/v3"

SPORT_SLUG = "ice-hockey"
LEAGUE_SLUG = "usa-nhl"
BOOKMAKER = "FanDuel"

ET = ZoneInfo("America/New_York")

JSON_OUT_DIR = Path("docs/win/hockey/nhl/odds")
SPORTSBOOK_OUT_DIR = Path("docs/win/hockey/nhl/00_intake/sportsbook")
LOG_DIR = Path("docs/win/hockey/nhl/errors/00_intake")

JSON_OUT_DIR.mkdir(parents=True, exist_ok=True)
SPORTSBOOK_OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "nhl_odds_pull.txt"

SPORTSBOOK_FIELDS = [
    "game_id",
    "sport",
    "league",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
    "home_dk_moneyline_american",
    "away_dk_moneyline_american",
    "home_puck_line",
    "away_puck_line",
    "total",
    "home_dk_puck_line_american",
    "away_dk_puck_line_american",
    "dk_total_over_american",
    "dk_total_under_american",
    "home_dk_moneyline_decimal",
    "away_dk_moneyline_decimal",
    "home_dk_puck_line_decimal",
    "away_dk_puck_line_decimal",
    "dk_total_over_decimal",
    "dk_total_under_decimal",
]

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write(f"=== nhl_odds_pull RUN {datetime.now(ET).isoformat()} ===\n")


def log(msg: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now(ET).isoformat()} | {msg}\n")


def get_api_key() -> str:
    key = os.environ.get(API_KEY_ENV, "").strip()
    if not key:
        raise RuntimeError(f"{API_KEY_ENV} environment variable is not set")
    return key


def wipe_outputs() -> None:
    removed_json = 0
    removed_sportsbook = 0

    for path in JSON_OUT_DIR.glob("*.json"):
        path.unlink()
        removed_json += 1

    for path in SPORTSBOOK_OUT_DIR.glob("NHL_*.csv"):
        path.unlink()
        removed_sportsbook += 1

    log(f"Removed old odds JSON files: {removed_json}")
    log(f"Removed old sportsbook CSV files: {removed_sportsbook}")


def request_json(path: str, params: dict) -> object:
    response = requests.get(f"{BASE_URL}{path}", params=params, timeout=30)

    if response.status_code != 200:
        raise RuntimeError(
            f"HTTP {response.status_code} for {BASE_URL}{path} | body={response.text[:500]}"
        )

    return response.json()


def decimal_to_american(decimal_value) -> str:
    try:
        dec = float(decimal_value)
        if dec <= 1:
            return ""

        if dec >= 2:
            american = round((dec - 1) * 100)
            return f"+{american}"

        american = round(-100 / (dec - 1))
        return str(american)

    except Exception:
        return ""


def to_et_date_time(date_str: str) -> tuple[str, str]:
    if not date_str:
        return "", ""

    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        dt_et = dt.astimezone(ET)
        return dt_et.strftime("%Y_%m_%d"), dt_et.strftime("%H:%M")
    except Exception:
        return "", ""


def clean_decimal(value) -> str:
    if value in ("", None):
        return ""
    return str(value).strip()


def clean_line(value) -> str:
    if value in ("", None):
        return ""

    try:
        v = float(value)
        if v.is_integer():
            return str(int(v))
        return str(v)
    except Exception:
        return str(value).strip()


def get_markets(odds_payload: dict) -> list:
    books = odds_payload.get("bookmakers", {})
    if not isinstance(books, dict):
        return []

    markets = books.get(BOOKMAKER, [])
    if not isinstance(markets, list):
        return []

    return markets


def find_market(markets: list, name: str) -> dict | None:
    wanted = name.strip().lower()

    for market in markets:
        if str(market.get("name", "")).strip().lower() == wanted:
            return market

    return None


def parse_moneyline(markets: list) -> dict:
    market = find_market(markets, "ML")
    if not market:
        return {
            "home_decimal": "",
            "away_decimal": "",
            "home_american": "",
            "away_american": "",
        }

    odds = market.get("odds", [])
    if not isinstance(odds, list) or not odds:
        return {
            "home_decimal": "",
            "away_decimal": "",
            "home_american": "",
            "away_american": "",
        }

    row = odds[0]
    home_decimal = clean_decimal(row.get("home", ""))
    away_decimal = clean_decimal(row.get("away", ""))

    return {
        "home_decimal": home_decimal,
        "away_decimal": away_decimal,
        "home_american": decimal_to_american(home_decimal),
        "away_american": decimal_to_american(away_decimal),
    }


def pick_puck_line_row(rows: list) -> dict:
    if not rows:
        return {}

    valid = [row for row in rows if isinstance(row, dict)]
    if not valid:
        return {}

    for row in valid:
        try:
            if abs(float(row.get("hdp"))) == 1.5:
                return row
        except Exception:
            continue

    return valid[0]


def parse_spread(markets: list) -> dict:
    market = find_market(markets, "Spread")
    if not market:
        return {
            "home_line": "",
            "away_line": "",
            "home_decimal": "",
            "away_decimal": "",
            "home_american": "",
            "away_american": "",
        }

    odds = market.get("odds", [])
    if not isinstance(odds, list):
        odds = []

    row = pick_puck_line_row(odds)

    home_line_raw = row.get("hdp", "")
    home_decimal = clean_decimal(row.get("home", ""))
    away_decimal = clean_decimal(row.get("away", ""))

    try:
        home_line_float = float(home_line_raw)
        away_line_float = -home_line_float
        home_line = clean_line(home_line_float)
        away_line = clean_line(away_line_float)
    except Exception:
        home_line = clean_line(home_line_raw)
        away_line = ""

    return {
        "home_line": home_line,
        "away_line": away_line,
        "home_decimal": home_decimal,
        "away_decimal": away_decimal,
        "home_american": decimal_to_american(home_decimal),
        "away_american": decimal_to_american(away_decimal),
    }


def pick_total_row(rows: list) -> dict:
    if not rows:
        return {}

    valid = [row for row in rows if isinstance(row, dict)]
    if not valid:
        return {}

    for row in valid:
        try:
            if float(row.get("hdp")) == 5.5:
                return row
        except Exception:
            continue

    return valid[0]


def parse_totals(markets: list) -> dict:
    market = find_market(markets, "Totals")
    if not market:
        return {
            "total": "",
            "over_decimal": "",
            "under_decimal": "",
            "over_american": "",
            "under_american": "",
        }

    odds = market.get("odds", [])
    if not isinstance(odds, list):
        odds = []

    row = pick_total_row(odds)

    total = clean_line(row.get("hdp", ""))
    over_decimal = clean_decimal(row.get("over", ""))
    under_decimal = clean_decimal(row.get("under", ""))

    return {
        "total": total,
        "over_decimal": over_decimal,
        "under_decimal": under_decimal,
        "over_american": decimal_to_american(over_decimal),
        "under_american": decimal_to_american(under_decimal),
    }


def fetch_events(api_key: str) -> list:
    payload = request_json(
        "/events",
        {
            "apiKey": api_key,
            "sport": SPORT_SLUG,
            "league": LEAGUE_SLUG,
            "status": "pending",
            "limit": 100,
        },
    )

    if isinstance(payload, list):
        return payload

    log(f"WARNING: events response was not a list: {payload}")
    return []


def fetch_odds(api_key: str, event_id: str) -> dict:
    payload = request_json(
        "/odds",
        {
            "apiKey": api_key,
            "eventId": event_id,
            "bookmakers": BOOKMAKER,
        },
    )

    if isinstance(payload, dict):
        return payload

    return {}


def build_row(event: dict, odds_payload: dict) -> dict:
    markets = get_markets(odds_payload)

    moneyline = parse_moneyline(markets)
    spread = parse_spread(markets)
    totals = parse_totals(markets)

    game_id = str(odds_payload.get("id") or event.get("id") or "").strip()
    home_team = str(odds_payload.get("home") or event.get("home") or "").strip()
    away_team = str(odds_payload.get("away") or event.get("away") or "").strip()

    game_date, game_time = to_et_date_time(
        str(odds_payload.get("date") or event.get("date") or "")
    )

    return {
        "bookmaker": BOOKMAKER,
        "game_id": game_id,
        "sport": "hockey",
        "league": "nhl",
        "game_date": game_date,
        "game_time": game_time,
        "home_team": home_team,
        "away_team": away_team,
        "home_dk_moneyline_american": moneyline["home_american"],
        "away_dk_moneyline_american": moneyline["away_american"],
        "home_puck_line": spread["home_line"],
        "away_puck_line": spread["away_line"],
        "total": totals["total"],
        "home_dk_puck_line_american": spread["home_american"],
        "away_dk_puck_line_american": spread["away_american"],
        "dk_total_over_american": totals["over_american"],
        "dk_total_under_american": totals["under_american"],
        "home_dk_moneyline_decimal": moneyline["home_decimal"],
        "away_dk_moneyline_decimal": moneyline["away_decimal"],
        "home_dk_puck_line_decimal": spread["home_decimal"],
        "away_dk_puck_line_decimal": spread["away_decimal"],
        "dk_total_over_decimal": totals["over_decimal"],
        "dk_total_under_decimal": totals["under_decimal"],
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SPORTSBOOK_FIELDS, extrasaction="ignore")
        writer.writeheader()

        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SPORTSBOOK_FIELDS})

    log(f"WROTE {path} ({len(rows)} rows)")


def main() -> None:
    api_key = get_api_key()
    run_date = datetime.now(ET).strftime("%Y_%m_%d")
    json_out_path = JSON_OUT_DIR / f"{run_date}.json"

    wipe_outputs()

    events = fetch_events(api_key)
    log(f"EVENTS FOUND: {len(events)}")

    if not events:
        log("--- SUMMARY ---")
        log("Events found: 0")
        log("Normalized rows: 0")
        log("Raw records written: 0")
        log("Sportsbook CSV files written: 0")
        log("STATUS: FAILED")
        raise RuntimeError("No pending NHL events returned by odds API.")

    raw_records = []
    normalized_rows = []
    rows_by_date = defaultdict(list)
    odds_errors = 0

    for event in events:
        event_id = str(event.get("id", "")).strip()

        if not event_id:
            log(f"SKIP event missing id: {event}")
            continue

        try:
            odds_payload = fetch_odds(api_key, event_id)
            normalized = build_row(event, odds_payload)

            raw_records.append(
                {
                    "event": event,
                    "odds": odds_payload,
                    "normalized": normalized,
                }
            )

            normalized_rows.append(normalized)

            game_date = normalized.get("game_date", "")
            if game_date:
                rows_by_date[game_date].append(normalized)

            log(
                "ROW "
                f"game_id={normalized['game_id']} "
                f"date={normalized['game_date']} "
                f"time={normalized['game_time']} "
                f"{normalized['away_team']} at {normalized['home_team']} "
                f"ML away/home=({normalized['away_dk_moneyline_decimal']},"
                f"{normalized['home_dk_moneyline_decimal']}) "
                f"PL away/home=({normalized['away_puck_line']} "
                f"{normalized['away_dk_puck_line_decimal']}, "
                f"{normalized['home_puck_line']} "
                f"{normalized['home_dk_puck_line_decimal']}) "
                f"Total={normalized['total']} "
                f"O/U=({normalized['dk_total_over_decimal']},"
                f"{normalized['dk_total_under_decimal']})"
            )

        except Exception as e:
            odds_errors += 1
            log(f"ERROR event_id={event_id}: {e}")
            log(traceback.format_exc())

        time.sleep(0.2)

    if not normalized_rows:
        log("--- SUMMARY ---")
        log(f"Events found: {len(events)}")
        log("Normalized rows: 0")
        log(f"Odds errors: {odds_errors}")
        log("Sportsbook CSV files written: 0")
        log("STATUS: FAILED")
        raise RuntimeError("Events were found but no normalized odds rows were written.")

    json_payload = {
        "run_date": run_date,
        "generated_at_et": datetime.now(ET).isoformat(),
        "source": "odds-api.io",
        "sport_slug": SPORT_SLUG,
        "league_slug": LEAGUE_SLUG,
        "bookmaker": BOOKMAKER,
        "rows": normalized_rows,
        "raw": raw_records,
    }

    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)

    log(f"WROTE {json_out_path} ({len(normalized_rows)} normalized rows)")

    sportsbook_files_written = 0

    for game_date, rows in sorted(rows_by_date.items()):
        write_csv(SPORTSBOOK_OUT_DIR / f"NHL_{game_date}.csv", rows)
        sportsbook_files_written += 1

    log("--- SUMMARY ---")
    log(f"Events found: {len(events)}")
    log(f"Normalized rows: {len(normalized_rows)}")
    log(f"Raw records written: {len(raw_records)}")
    log(f"Odds errors: {odds_errors}")
    log(f"JSON output: {json_out_path}")
    log(f"Sportsbook CSV files written: {sportsbook_files_written}")
    log("STATUS: SUCCESS")

    print(f"WROTE {json_out_path} rows={len(normalized_rows)}")
    print(f"WROTE {sportsbook_files_written} sportsbook CSV date file(s)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL ERROR: {e}")
        log(traceback.format_exc())
        log("STATUS: FAILED")
        raise
