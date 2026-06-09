#!/usr/bin/env python3
# docs/win/hockey/nhl/scripts/00_intake/nhl_odds_pull.py

import json
import os
import time
import traceback
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
LOG_DIR = Path("docs/win/hockey/nhl/errors/00_intake")

JSON_OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "nhl_odds_pull.txt"

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


def request_json(path: str, params: dict) -> object:
    response = requests.get(f"{BASE_URL}{path}", params=params, timeout=30)

    if response.status_code != 200:
        raise RuntimeError(
            f"HTTP {response.status_code} for {BASE_URL}{path} | body={response.text[:500]}"
        )

    return response.json()


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


def main() -> None:
    api_key = get_api_key()
    run_date = datetime.now(ET).strftime("%Y_%m_%d")
    json_out_path = JSON_OUT_DIR / f"{run_date}.json"

    events = fetch_events(api_key)
    log(f"EVENTS FOUND: {len(events)}")

    raw_records = []
    odds_errors = 0

    for event in events:
        event_id = str(event.get("id", "")).strip()

        if not event_id:
            log(f"SKIP event missing id: {event}")
            continue

        try:
            odds_payload = fetch_odds(api_key, event_id)

            raw_records.append(
                {
                    "event": event,
                    "odds": odds_payload,
                }
            )

            log(
                "RAW "
                f"event_id={event_id} "
                f"home={event.get('home', '')} "
                f"away={event.get('away', '')}"
            )

        except Exception as e:
            odds_errors += 1
            log(f"ERROR event_id={event_id}: {e}")
            log(traceback.format_exc())

        time.sleep(0.2)

    json_payload = {
        "run_date": run_date,
        "generated_at_et": datetime.now(ET).isoformat(),
        "source": "odds-api.io",
        "sport_slug": SPORT_SLUG,
        "league_slug": LEAGUE_SLUG,
        "bookmaker": BOOKMAKER,
        "raw": raw_records,
    }

    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)

    log("--- SUMMARY ---")
    log(f"Events found: {len(events)}")
    log(f"Raw records written: {len(raw_records)}")
    log(f"Odds errors: {odds_errors}")
    log(f"JSON output: {json_out_path}")

    if events and not raw_records:
        log("STATUS: FAILED")
        raise RuntimeError("Events were found but no raw odds records were written")

    log("STATUS: SUCCESS")
    print(f"WROTE {json_out_path} raw_records={len(raw_records)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL ERROR: {e}")
        log(traceback.format_exc())
        log("STATUS: FAILED")
        raise
