#!/usr/bin/env python3
# docs/win/basketball/scripts/00_intake/00_ncaam_wnba_cloudbet_odds_pull.py

import json
import os
import traceback
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from zoneinfo import ZoneInfo

import requests

BASE_URL = "https://sports-api.cloudbet.com/pub/v2/odds"
API_KEY = os.environ.get("CLOUDBET_API_KEY", "").strip()

LEAGUES = {
    "ncaam": {
        "competition_key": "basketball-usa-ncaa",
        "output_label": "ncaam",
        "output_dir": Path("docs/win/basketball/odds/ncaam"),
    },
    "wnba": {
        "competition_key": "basketball-usa-wnba",
        "output_label": "wnba",
        "output_dir": Path("docs/win/basketball/odds/wnba"),
    },
}

for cfg in LEAGUES.values():
    cfg["output_dir"].mkdir(parents=True, exist_ok=True)

ERROR_DIR = Path("docs/win/basketball/errors/00_intake")
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ERROR_DIR / "cloudbet_odds_pull.txt"

NY_TZ = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")


with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write(f"=== cloudbet_odds_pull RUN {datetime.now(NY_TZ).isoformat()} ===\n")


def log(msg: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now(NY_TZ).isoformat()} | {msg}\n")


def get_headers() -> dict:
    if not API_KEY:
        raise RuntimeError("CLOUDBET_API_KEY environment variable is not set")
    return {"X-API-Key": API_KEY}


def get_json(url: str) -> dict:
    resp = requests.get(url, headers=get_headers(), timeout=60)
    resp.raise_for_status()
    return resp.json()


def iso_to_et(iso_str: str):
    if not iso_str:
        return None, "", ""

    dt_utc = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=UTC_TZ)

    dt_et = dt_utc.astimezone(NY_TZ)
    return dt_et, dt_et.strftime("%Y_%m_%d"), dt_et.strftime("%H:%M")


def decimal_to_american(dec):
    if dec is None:
        return None
    try:
        dec = float(dec)
    except Exception:
        return None

    if dec <= 1:
        return None
    if dec >= 2.0:
        return round((dec - 1) * 100)
    return round(-100 / (dec - 1))


def parse_params(params: str) -> dict:
    out = {}
    if not params:
        return out

    for part in params.split("&"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
    return out


def extract_main_submarket(market_obj: dict):
    submarkets = market_obj.get("submarkets", {}) or {}
    preferred_keys = [
        "period=ot.period=ft",
        "period=ot&period=ft",
        "period=ft",
        "",
    ]

    for key in preferred_keys:
        if key in submarkets:
            return key, submarkets[key]

    for key, value in submarkets.items():
        return key, value

    return None, None


def extract_moneyline(markets: dict) -> dict:
    out = {
        "home_price_decimal": None,
        "away_price_decimal": None,
        "home_price_american": None,
        "away_price_american": None,
        "submarket_key": None,
    }

    market_obj = markets.get("basketball.moneyline")
    if not market_obj:
        return out

    submarket_key, submarket = extract_main_submarket(market_obj)
    out["submarket_key"] = submarket_key
    if not submarket:
        return out

    for sel in submarket.get("selections", []) or []:
        outcome = (sel.get("outcome") or "").lower()
        price = sel.get("price")
        if outcome == "home":
            out["home_price_decimal"] = price
            out["home_price_american"] = decimal_to_american(price)
        elif outcome == "away":
            out["away_price_decimal"] = price
            out["away_price_american"] = decimal_to_american(price)

    return out


def extract_spreads(markets: dict) -> dict:
    out = {
        "submarket_key": None,
        "lines": [],
    }

    market_obj = markets.get("basketball.handicap")
    if not market_obj:
        return out

    submarket_key, submarket = extract_main_submarket(market_obj)
    out["submarket_key"] = submarket_key
    if not submarket:
        return out

    grouped = defaultdict(dict)

    for sel in submarket.get("selections", []) or []:
        outcome = (sel.get("outcome") or "").lower()
        params = parse_params(sel.get("params", ""))
        handicap = params.get("handicap")
        price = sel.get("price")

        if handicap is None:
            continue

        grp = grouped[handicap]
        if outcome == "home":
            grp["home_spread"] = handicap
            grp["home_price_decimal"] = price
            grp["home_price_american"] = decimal_to_american(price)
        elif outcome == "away":
            grp["away_spread"] = handicap
            grp["away_price_decimal"] = price
            grp["away_price_american"] = decimal_to_american(price)

    def sort_key(item):
        try:
            return float(item[0])
        except Exception:
            return 999999

    out["lines"] = [v for _, v in sorted(grouped.items(), key=sort_key)]
    return out


def extract_totals(markets: dict) -> dict:
    out = {
        "submarket_key": None,
        "lines": [],
    }

    market_obj = markets.get("basketball.totals")
    if not market_obj:
        return out

    submarket_key, submarket = extract_main_submarket(market_obj)
    out["submarket_key"] = submarket_key
    if not submarket:
        return out

    grouped = defaultdict(dict)

    for sel in submarket.get("selections", []) or []:
        outcome = (sel.get("outcome") or "").lower()
        params = parse_params(sel.get("params", ""))
        total = params.get("total")
        price = sel.get("price")

        if total is None:
            continue

        grp = grouped[total]
        grp["total"] = total

        if outcome == "over":
            grp["over_price_decimal"] = price
            grp["over_price_american"] = decimal_to_american(price)
        elif outcome == "under":
            grp["under_price_decimal"] = price
            grp["under_price_american"] = decimal_to_american(price)

    def sort_key(item):
        try:
            return float(item[0])
        except Exception:
            return 999999

    out["lines"] = [v for _, v in sorted(grouped.items(), key=sort_key)]
    return out


def parse_event(event: dict, league_key: str) -> dict:
    dt_et, game_date, game_time = iso_to_et(event.get("startTime") or event.get("cutoffTime"))

    return {
        "sport": "basketball",
        "league": league_key.upper(),
        "market": league_key,
        "game_date": game_date,
        "game_time": game_time,
        "start_time_utc": event.get("startTime"),
        "cutoff_time_utc": event.get("cutoffTime"),
        "event_id": event.get("id"),
        "sequence": event.get("sequence"),
        "status": event.get("status"),
        "competition_name": (event.get("competition") or {}).get("name"),
        "competition_key": (event.get("competition") or {}).get("key"),
        "home_team": (event.get("home") or {}).get("name"),
        "home_team_key": (event.get("home") or {}).get("key"),
        "away_team": (event.get("away") or {}).get("name"),
        "away_team_key": (event.get("away") or {}).get("key"),
        "event_name": event.get("name"),
        "event_key": event.get("key"),
        "moneyline": extract_moneyline(event.get("markets", {}) or {}),
        "spreads": extract_spreads(event.get("markets", {}) or {}),
        "totals": extract_totals(event.get("markets", {}) or {}),
        "metadata": event.get("metadata", {}),
    }


def fetch_competition_events(competition_key: str) -> list:
    url = f"{BASE_URL}/competitions/{competition_key}"
    payload = get_json(url)
    return payload.get("events", []) or []


def write_grouped_files(output_dir: Path, output_label: str, rows: list) -> list:
    by_date = defaultdict(list)
    for row in rows:
        game_date = row.get("game_date")
        if game_date:
            by_date[game_date].append(row)

    written = []
    for game_date, date_rows in sorted(by_date.items()):
        out_path = output_dir / f"{game_date}_{output_label}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(date_rows, f, indent=2, ensure_ascii=False)
        written.append((str(out_path), len(date_rows)))
        log(f"WROTE {out_path} ({len(date_rows)} events)")

    return written


def main():
    files_written = []
    total_events = 0

    try:
        for league_key, cfg in LEAGUES.items():
            competition_key = cfg["competition_key"]
            output_label = cfg["output_label"]
            output_dir = cfg["output_dir"]

            log(f"FETCH league={league_key} competition={competition_key}")
            try:
                events = fetch_competition_events(competition_key)
                log(f"FOUND {len(events)} raw events for {league_key}")

                parsed_rows = []
                for event in events:
                    try:
                        row = parse_event(event, league_key)
                        if row.get("game_date"):
                            parsed_rows.append(row)
                    except Exception as e:
                        log(
                            f"ERROR parsing event {event.get('id', '?')} for {league_key}: "
                            f"{e}\n{traceback.format_exc()}"
                        )

                total_events += len(parsed_rows)
                files_written.extend(write_grouped_files(output_dir, output_label, parsed_rows))

            except Exception as e:
                log(
                    f"ERROR fetching {league_key} ({competition_key}): "
                    f"{e}\n{traceback.format_exc()}"
                )

        log("--- SUMMARY ---")
        log(f"Total parsed events: {total_events}")
        log(f"Files written: {len(files_written)}")
        for path, count in files_written:
            log(f"FILE: {path} ({count} events)")
        log("STATUS: SUCCESS")

        print("Cloudbet basketball odds pull complete.")

    except Exception as e:
        log(f"FATAL ERROR: {e}\n{traceback.format_exc()}")
        log("STATUS: FAILED")
        raise


if __name__ == "__main__":
    main()