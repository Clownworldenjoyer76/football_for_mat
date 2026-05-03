#!/usr/bin/env python3
# docs/win/soccer/odds_api_soccer_pull.py

import json
import os
import time
import traceback
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

# Requires GitHub secret/env:
#   API_ODDS

API_KEY_ENV = "API_ODDS"
BASE_URL = "https://api.odds-api.io/v3"

ET = ZoneInfo("America/New_York")

LEAGUES = {
    "MLS": {
        "slug": "usa-mls",
        "bookmaker": "FanDuel",
    },
    "EPL": {
        "slug": "england-premier-league",
        "bookmaker": "DraftKings",
    },
    "LIGUE_1": {
        "slug": "france-ligue-1",
        "bookmaker": "DraftKings",
    },
    "LA_LIGA": {
        "slug": "spain-laliga",
        "bookmaker": "DraftKings",
    },
    "SERIE_A": {
        "slug": "italy-serie-a",
        "bookmaker": "DraftKings",
    },
    "BUNDESLIGA": {
        "slug": "germany-bundesliga",
        "bookmaker": "DraftKings",
    },
}

OUT_DIR = Path("docs/win/soccer/00_intake/odds_api_raw")
LOG_DIR = Path("docs/win/soccer/errors/00_intake")

OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "odds_api_soccer_pull.txt"

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write(f"=== odds_api_soccer_pull RUN {datetime.now(ET).isoformat()} ===\n")


def log(msg: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now(ET).isoformat()} | {msg}\n")


def get_api_key() -> str:
    key = os.environ.get(API_KEY_ENV, "").strip()
    if not key:
        raise RuntimeError(f"{API_KEY_ENV} environment variable is not set")
    return key


def request_json(path: str, params: dict) -> object:
    url = f"{BASE_URL}{path}"
    response = requests.get(url, params=params, timeout=30)

    if response.status_code != 200:
        raise RuntimeError(
            f"HTTP {response.status_code} for {url} | body={response.text[:500]}"
        )

    return response.json()


def to_match_date_time(date_str: str) -> tuple[str, str]:
    if not date_str:
        return "", ""

    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        dt_et = dt.astimezone(ET)
        return dt_et.strftime("%Y_%m_%d"), dt_et.strftime("%I:%M %p")
    except Exception:
        return "", ""


def get_bookmaker_markets(odds_payload: dict, bookmaker: str) -> list:
    books = odds_payload.get("bookmakers", {})

    if not isinstance(books, dict):
        return []

    markets = books.get(bookmaker, [])

    if isinstance(markets, list):
        return markets

    return []


def market_name(market: dict) -> str:
    return str(market.get("name", "")).strip()


def find_market(markets: list, wanted_name: str) -> dict | None:
    wanted = wanted_name.strip().lower()

    for market in markets:
        if market_name(market).lower() == wanted:
            return market

    return None


def parse_ml(markets: list) -> tuple[str, str, str]:
    market = find_market(markets, "ML")
    if not market:
        return "", "", ""

    odds = market.get("odds", [])
    if not odds or not isinstance(odds, list):
        return "", "", ""

    row = odds[0]
    if not isinstance(row, dict):
        return "", "", ""

    return (
        str(row.get("home", "") or ""),
        str(row.get("draw", "") or ""),
        str(row.get("away", "") or ""),
    )


def odds_line_value(row: dict) -> str:
    for key in ("line", "hdp", "max", "total", "points", "point"):
        val = row.get(key)
        if val not in ("", None):
            return str(val)
    return ""


def parse_total_line(markets: list, target_line: float) -> tuple[str, str]:
    market = find_market(markets, "Totals")
    if not market:
        return "", ""

    odds = market.get("odds", [])
    if not isinstance(odds, list):
        return "", ""

    for row in odds:
        if not isinstance(row, dict):
            continue

        over = str(row.get("over", "") or row.get("Over", "") or "")
        under = str(row.get("under", "") or row.get("Under", "") or "")

        line_raw = odds_line_value(row)

        if line_raw:
            try:
                if abs(float(line_raw) - float(target_line)) < 0.001:
                    return over, under
            except Exception:
                pass

        label = " ".join(str(v) for v in row.values()).lower()
        if str(target_line) in label:
            return over, under

    return "", ""


def parse_btts(markets: list) -> tuple[str, str]:
    market = find_market(markets, "Both Teams To Score")
    if not market:
        return "", ""

    odds = market.get("odds", [])
    if not isinstance(odds, list):
        return "", ""

    for row in odds:
        if not isinstance(row, dict):
            continue

        yes = (
            row.get("yes")
            or row.get("Yes")
            or row.get("YES")
            or row.get("both_teams_to_score_yes")
            or ""
        )
        no = (
            row.get("no")
            or row.get("No")
            or row.get("NO")
            or row.get("both_teams_to_score_no")
            or ""
        )

        if yes or no:
            return str(yes or ""), str(no or "")

    return "", ""


def fetch_events(api_key: str, league_slug: str, bookmaker: str) -> list:
    payload = request_json(
        "/events",
        {
            "apiKey": api_key,
            "sport": "football",
            "league": league_slug,
            "status": "pending",
            "bookmaker": bookmaker,
            "limit": 100,
        },
    )

    if isinstance(payload, list):
        return payload

    log(f"WARNING: events response was not list for {league_slug}: {payload}")
    return []


def fetch_odds(api_key: str, event_id: str, bookmaker: str) -> dict:
    payload = request_json(
        "/odds",
        {
            "apiKey": api_key,
            "eventId": event_id,
            "bookmakers": bookmaker,
        },
    )

    if isinstance(payload, dict):
        return payload

    return {}


def build_row(league: str, event: dict, odds_payload: dict, bookmaker: str) -> dict:
    markets = get_bookmaker_markets(odds_payload, bookmaker)

    dk_home, dk_draw, dk_away = parse_ml(markets)
    over25, under25 = parse_total_line(markets, 2.5)
    over35, under35 = parse_total_line(markets, 3.5)
    btts_yes, btts_no = parse_btts(markets)

    match_date, match_time = to_match_date_time(
        odds_payload.get("date") or event.get("date") or ""
    )

    return {
        "sport": "soccer",
        "league": league,
        "game_id": str(odds_payload.get("id") or event.get("id") or ""),
        "match_date": match_date,
        "match_time": match_time,
        "home_team": odds_payload.get("home") or event.get("home") or "",
        "away_team": odds_payload.get("away") or event.get("away") or "",
        "dk_home_decimal": dk_home,
        "dk_draw_decimal": dk_draw,
        "dk_away_decimal": dk_away,
        "dk_over25_decimal": over25,
        "dk_under25_decimal": under25,
        "dk_over35_decimal": over35,
        "dk_under35_decimal": under35,
        "btts_yes": btts_yes,
        "btts_no": btts_no,
    }


def main() -> None:
    api_key = get_api_key()
    today = datetime.now(ET).strftime("%Y_%m_%d")
    out_path = OUT_DIR / f"{today}_soccer_odds.json"

    all_rows = []
    league_errors = 0

    for league, cfg in LEAGUES.items():
        slug = cfg["slug"]
        bookmaker = cfg["bookmaker"]

        log(f"START league={league} slug={slug} bookmaker={bookmaker}")

        try:
            events = fetch_events(api_key, slug, bookmaker)
            log(f"EVENTS league={league} count={len(events)}")

            for event in events:
                event_id = str(event.get("id", "")).strip()
                if not event_id:
                    log(f"SKIP league={league}: missing event id")
                    continue

                try:
                    odds_payload = fetch_odds(api_key, event_id, bookmaker)
                    row = build_row(league, event, odds_payload, bookmaker)
                    all_rows.append(row)

                    log(
                        "ROW "
                        f"league={league} "
                        f"game_id={row['game_id']} "
                        f"{row['away_team']} at {row['home_team']} "
                        f"ML=({row['dk_home_decimal']},{row['dk_draw_decimal']},{row['dk_away_decimal']}) "
                        f"O25/U25=({row['dk_over25_decimal']},{row['dk_under25_decimal']}) "
                        f"O35/U35=({row['dk_over35_decimal']},{row['dk_under35_decimal']}) "
                        f"BTTS=({row['btts_yes']},{row['btts_no']})"
                    )

                except Exception as e:
                    log(f"ERROR odds league={league} event_id={event_id}: {e}")
                    log(traceback.format_exc())

                time.sleep(0.2)

        except Exception as e:
            league_errors += 1
            log(f"ERROR league={league}: {e}")
            log(traceback.format_exc())

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2)

    log("--- SUMMARY ---")
    log(f"Rows written: {len(all_rows)}")
    log(f"League errors: {league_errors}")
    log(f"Output: {out_path}")

    if league_errors == len(LEAGUES):
        log("STATUS: FAILED")
        raise RuntimeError("All soccer league pulls failed")

    log("STATUS: SUCCESS")
    print(f"WROTE {out_path} rows={len(all_rows)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL ERROR: {e}")
        log(traceback.format_exc())
        log("STATUS: FAILED")
        raise
