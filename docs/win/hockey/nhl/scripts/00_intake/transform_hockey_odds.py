#!/usr/bin/env python3
# docs/win/hockey/nhl/scripts/00_intake/transform_hockey_odds.py

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

BOOKMAKER = "FanDuel"
ET = ZoneInfo("America/New_York")

ODDS_DIR = Path("docs/win/hockey/nhl/odds")
SPORTSBOOK_DIR = Path("docs/win/hockey/nhl/00_intake/sportsbook")

SPORTSBOOK_DIR.mkdir(parents=True, exist_ok=True)

FIELDS = [
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


def decimal_to_american(value) -> str:
    try:
        dec = float(value)

        if dec <= 1:
            return ""

        if dec >= 2:
            return f"+{round((dec - 1) * 100)}"

        return str(round(-100 / (dec - 1)))

    except Exception:
        return ""


def clean_decimal(value) -> str:
    if value in ("", None, "N/A"):
        return ""
    return str(value).strip()


def clean_line(value) -> str:
    if value in ("", None, "N/A"):
        return ""

    try:
        v = float(value)
        if v.is_integer():
            return str(int(v))
        return str(v)
    except Exception:
        return str(value).strip()


def to_et_date_time(date_str: str) -> tuple[str, str]:
    if not date_str:
        return "", ""

    try:
        dt = datetime.fromisoformat(str(date_str).replace("Z", "+00:00"))
        dt_et = dt.astimezone(ET)
        return dt_et.strftime("%Y_%m_%d"), dt_et.strftime("%H:%M")
    except Exception:
        return "", ""


def get_markets(odds_payload: dict) -> list:
    bookmakers = odds_payload.get("bookmakers", {})
    if not isinstance(bookmakers, dict):
        return []

    markets = bookmakers.get(BOOKMAKER, [])
    if not isinstance(markets, list):
        return []

    return markets


def find_market(markets: list, market_name: str) -> dict:
    wanted = market_name.strip().lower()

    for market in markets:
        if str(market.get("name", "")).strip().lower() == wanted:
            return market

    return {}


def parse_moneyline(markets: list) -> dict:
    market = find_market(markets, "ML")
    odds = market.get("odds", [])

    if not isinstance(odds, list) or not odds or not isinstance(odds[0], dict):
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


def pick_standard_puck_line_row(rows: list) -> dict:
    valid = [row for row in rows if isinstance(row, dict)]

    for row in valid:
        try:
            if abs(float(row.get("hdp"))) == 1.5:
                return row
        except Exception:
            continue

    return {}


def parse_spread(markets: list) -> dict:
    market = find_market(markets, "Spread")
    odds = market.get("odds", [])

    if not isinstance(odds, list):
        odds = []

    row = pick_standard_puck_line_row(odds)

    if not row:
        return {
            "home_line": "",
            "away_line": "",
            "home_decimal": "",
            "away_decimal": "",
            "home_american": "",
            "away_american": "",
        }

    try:
        hdp = float(row.get("hdp"))

        if hdp > 0:
            home_line = -abs(hdp)
            away_line = abs(hdp)
            home_decimal = clean_decimal(row.get("away", ""))
            away_decimal = clean_decimal(row.get("home", ""))
        else:
            home_line = abs(hdp)
            away_line = -abs(hdp)
            home_decimal = clean_decimal(row.get("home", ""))
            away_decimal = clean_decimal(row.get("away", ""))

    except Exception:
        home_line = ""
        away_line = ""
        home_decimal = ""
        away_decimal = ""

    return {
        "home_line": clean_line(home_line),
        "away_line": clean_line(away_line),
        "home_decimal": home_decimal,
        "away_decimal": away_decimal,
        "home_american": decimal_to_american(home_decimal),
        "away_american": decimal_to_american(away_decimal),
    }


def pick_total_row_closest_odds(rows: list) -> dict:
    valid = [row for row in rows if isinstance(row, dict)]
    candidates = []

    for index, row in enumerate(valid):
        try:
            over = float(row.get("over"))
            under = float(row.get("under"))

            if over <= 1 or under <= 1:
                continue

            candidates.append((abs(over - under), index, row))

        except Exception:
            continue

    if not candidates:
        return {}

    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def parse_totals(markets: list) -> dict:
    market = find_market(markets, "Totals")
    odds = market.get("odds", [])

    if not isinstance(odds, list):
        odds = []

    row = pick_total_row_closest_odds(odds)

    if not row:
        return {
            "total": "",
            "over_decimal": "",
            "under_decimal": "",
            "over_american": "",
            "under_american": "",
        }

    over_decimal = clean_decimal(row.get("over", ""))
    under_decimal = clean_decimal(row.get("under", ""))

    return {
        "total": clean_line(row.get("hdp", "")),
        "over_decimal": over_decimal,
        "under_decimal": under_decimal,
        "over_american": decimal_to_american(over_decimal),
        "under_american": decimal_to_american(under_decimal),
    }


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


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def main() -> None:
    sportsbook_by_date = defaultdict(dict)

    json_files = sorted(ODDS_DIR.glob("*.json"))

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            payload = json.load(f)

        events = payload.get("events", [])
        odds = payload.get("odds", [])

        if not isinstance(events, list):
            events = []

        if not isinstance(odds, list):
            odds = []

        events_by_id = {
            str(event.get("id", "")).strip(): event
            for event in events
            if isinstance(event, dict) and event.get("id")
        }

        for odds_payload in odds:
            if not isinstance(odds_payload, dict):
                continue

            game_id = str(odds_payload.get("id", "")).strip()
            if not game_id:
                continue

            event = events_by_id.get(game_id, {})
            row = build_row(event, odds_payload)

            game_date = row.get("game_date", "")
            if not game_date:
                continue

            sportsbook_by_date[game_date][game_id] = row

    for game_date, rows_by_game_id in sorted(sportsbook_by_date.items()):
        rows = list(rows_by_game_id.values())
        write_csv(SPORTSBOOK_DIR / f"NHL_{game_date}.csv", rows)

    print("NHL odds transform complete.")


if __name__ == "__main__":
    main()
