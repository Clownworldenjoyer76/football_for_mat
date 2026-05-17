# docs/win/baseball/scripts/00_intake/odds_parse.py

import sys
import json
import csv
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo  # ✅ handles DST correctly

# -----------------------
# INPUT HANDLING (FIXED)
# -----------------------

if len(sys.argv) > 1:
    INPUT_PATH = Path(sys.argv[1])
else:
    INPUT_PATH = Path("docs/win/baseball/odds")

if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Input path does not exist: {INPUT_PATH}")

# -----------------------
# TIME CONVERSION (UTC -> NY)
# -----------------------
def utc_to_est(utc_str):
    dt = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
    est = dt.astimezone(ZoneInfo("America/New_York"))
    return est.strftime("%Y_%m_%d"), est.strftime("%H:%M:%S")

# -----------------------
# ODDS CONVERSION
# -----------------------
def decimal_to_american(decimal_odds):
    if decimal_odds is None:
        return None
    if decimal_odds >= 2:
        return int((decimal_odds - 1) * 100)
    else:
        return int(-100 / (decimal_odds - 1))

# -----------------------
# PROCESS ONE FILE
# -----------------------
def process_file(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    grouped_rows = {}

    for game in data:
        game_id = game.get("id")

        sport = "baseball"
        league = "mlb"

        game_date, game_time = utc_to_est(game["commence_time"])

        away_team = game["away_team"]
        home_team = game["home_team"]

        away_run_line = None
        home_run_line = None
        total = None

        away_rl_dec = None
        home_rl_dec = None
        over_dec = None
        under_dec = None
        away_ml_dec = None
        home_ml_dec = None

        if not game.get("bookmakers"):
            continue

        markets = game["bookmakers"][0].get("markets", [])

        for market in markets:
            key = market["key"]

            if key == "h2h":
                for o in market["outcomes"]:
                    if o["name"] == away_team:
                        away_ml_dec = o["price"]
                    elif o["name"] == home_team:
                        home_ml_dec = o["price"]

            elif key == "spreads":
                for o in market["outcomes"]:
                    if o["name"] == away_team:
                        away_run_line = o["point"]
                        away_rl_dec = o["price"]
                    elif o["name"] == home_team:
                        home_run_line = o["point"]
                        home_rl_dec = o["price"]

            elif key == "totals":
                if market["outcomes"]:
                    total = market["outcomes"][0]["point"]

                for o in market["outcomes"]:
                    if o["name"] == "Over":
                        over_dec = o["price"]
                    elif o["name"] == "Under":
                        under_dec = o["price"]

        row = [
            game_id, sport, league, game_date, game_time,
            home_team, away_team,
            away_run_line, home_run_line, total,
            decimal_to_american(away_rl_dec),
            decimal_to_american(home_rl_dec),
            decimal_to_american(over_dec),
            decimal_to_american(under_dec),
            decimal_to_american(away_ml_dec),
            decimal_to_american(home_ml_dec),
            away_rl_dec, home_rl_dec,
            over_dec, under_dec,
            away_ml_dec, home_ml_dec
        ]

        grouped_rows.setdefault(game_date, []).append(row)

    base_output_dir = Path("docs/win/baseball/00_intake/sportsbook")
    base_output_dir.mkdir(parents=True, exist_ok=True)

    for game_date, rows in grouped_rows.items():
        output_path = base_output_dir / f"{game_date}_MLB.csv"

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "game_id","sport","league","game_date","game_time","home_team","away_team",
                "away_run_line","home_run_line","total",
                "away_dk_run_line_american","home_dk_run_line_american",
                "dk_total_over_american","dk_total_under_american",
                "away_dk_moneyline_american","home_dk_moneyline_american",
                "away_dk_run_line_decimal","home_dk_run_line_decimal",
                "dk_total_over_decimal","dk_total_under_decimal",
                "away_dk_moneyline_decimal","home_dk_moneyline_decimal"
            ])
            writer.writerows(rows)

        print(f"Saved {output_path}")

# -----------------------
# ENTRY
# -----------------------
if INPUT_PATH.is_file():
    process_file(INPUT_PATH)
elif INPUT_PATH.is_dir():
    files = list(INPUT_PATH.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {INPUT_PATH}")
    for file in sorted(files):
        process_file(file)
else:
    raise ValueError(f"Invalid input path: {INPUT_PATH}")
