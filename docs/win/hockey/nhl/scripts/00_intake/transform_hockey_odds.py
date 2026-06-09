#!/usr/bin/env python3
# docs/win/hockey/nhl/scripts/00_intake/transform_hockey_odds.py

import argparse
import csv
import json
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

BOOKMAKER = "FanDuel"
ET = ZoneInfo("America/New_York")

JSON_IN_DIR = Path("docs/win/hockey/nhl/odds")
SPORTSBOOK_OUT_DIR = Path("docs/win/hockey/nhl/00_intake/sportsbook")
LOG_DIR = Path("docs/win/hockey/nhl/errors/00_intake")

SPORTSBOOK_OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "transform_hockey_odds.txt"

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write(f"=== transform_hockey_odds RUN {datetime.now(ET).isoformat()} ===\n")


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


def log(msg: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now(ET).isoformat()} | {msg}\n")


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

    valid = [r for r in rows if isinstance(r, dict)]
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

    valid = [r for r in rows if isinstance(r, dict)]
    if not valid:
        return {}

    balanced = []

    for row in valid:
        try:
            over = float(row.get("over"))
            under = float(row.get("under"))
            hdp = float(row.get("hdp"))

            if over <= 1 or under <= 1:
                continue

            balanced.append((abs(over - under), hdp, row))
        except Exception:
            continue

    if balanced:
        balanced.sort(key=lambda x: (x[0], abs(x[1])))
        return balanced[0][2]

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


def build_row(record: dict) -> dict:
    event = record.get("event", {})
    odds_payload = record.get("odds", {})

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
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=SPORTSBOOK_FIELDS)
        writer.writeheader()

        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SPORTSBOOK_FIELDS})

    log(f"WROTE {path} ({len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=datetime.now(ET).strftime("%Y_%m_%d"))
    args = parser.parse_args()

    json_path = JSON_IN_DIR / f"{args.date}.json"

    if not json_path.exists():
        raise FileNotFoundError(f"Missing odds JSON: {json_path}")

    log(f"Input JSON: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    raw_records = payload.get("raw", [])
    log(f"Raw records loaded: {len(raw_records)}")

    sportsbook_by_date = defaultdict(list)
    rows_built = 0

    for record in raw_records:
        row = build_row(record)
        rows_built += 1

        game_date = row.get("game_date", "")
        if game_date:
            sportsbook_by_date[game_date].append(row)

        log(
            "ROW "
            f"game_id={row['game_id']} "
            f"date={row['game_date']} "
            f"time={row['game_time']} "
            f"{row['away_team']} at {row['home_team']} "
            f"ML away/home=({row['away_dk_moneyline_decimal']},"
            f"{row['home_dk_moneyline_decimal']}) "
            f"PL away/home=({row['away_puck_line']} "
            f"{row['away_dk_puck_line_decimal']}, "
            f"{row['home_puck_line']} "
            f"{row['home_dk_puck_line_decimal']}) "
            f"Total={row['total']} "
            f"O/U=({row['dk_total_over_decimal']},"
            f"{row['dk_total_under_decimal']})"
        )

    for game_date, rows in sorted(sportsbook_by_date.items()):
        write_csv(SPORTSBOOK_OUT_DIR / f"NHL_{game_date}.csv", rows)

    log("--- SUMMARY ---")
    log(f"Rows built: {rows_built}")
    log(f"CSV sportsbook dates: {len(sportsbook_by_date)}")
    log("STATUS: SUCCESS")

    print(f"WROTE {len(sportsbook_by_date)} sportsbook CSV date file(s)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL ERROR: {e}")
        log(traceback.format_exc())
        log("STATUS: FAILED")
        raise
