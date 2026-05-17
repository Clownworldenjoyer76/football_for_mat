#!/usr/bin/env python3
# docs/win/baseball/scripts/00_intake/transform_baseball.py

import json
import csv
from pathlib import Path
from datetime import datetime

# -------------------------
# PATHS
# -------------------------

RAW_DIR = Path("docs/win/baseball/00_intake/drat_raw")
PRED_DIR = Path("docs/win/baseball/00_intake/predictions")
FINAL_DIR = Path("docs/win/final_scores/results/mlb/final_scores")

SPORTSBOOK_DIR = Path("docs/win/baseball/00_intake/sportsbook")

PRED_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# HELPERS
# -------------------------

def parse_datetime(dt_str):
    dt = datetime.strptime(dt_str.strip(), "%m/%d/%Y %I:%M %p")
    return dt, dt.strftime("%Y_%m_%d"), dt.strftime("%I:%M %p")


def clean_team(team_str):
    return team_str.split("(")[0].strip()


def pct_to_decimal(p):
    return str(round(float(p.replace("%", "")) / 100, 3))


def split_lines(val):
    parts = val.split("\n")
    return parts[0].strip(), parts[1].strip() if len(parts) > 1 else ("", "")


# -------------------------
# LOAD LOOKUPS
# -------------------------

def load_predictions_lookup(date):
    path = PRED_DIR / f"{date}_MLB.csv"
    lookup = {}

    if not path.exists():
        return lookup

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            key = (r["home_team"], r["away_team"])
            lookup[key] = r.get("game_id")

    return lookup


def load_sportsbook_lookup(date):
    path = SPORTSBOOK_DIR / f"{date}_MLB.csv"
    lookup = {}

    if not path.exists():
        return lookup

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            key = (r["home_team"], r["away_team"])
            lookup[key] = {
                "away_run_line": r.get("away_run_line"),
                "home_run_line": r.get("home_run_line"),
                "total": r.get("total"),
            }

    return lookup


# -------------------------
# MAIN PROCESS
# -------------------------

def process_file(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    today = datetime.now().date()

    predictions_by_date = {}
    final_scores_by_date = {}

    for row in data:
        if not row or len(row) < 2:
            continue

        try:
            dt, game_date, game_time = parse_datetime(row[0])
        except:
            continue

        is_past = dt.date() < today

        # -------------------------
        # TEAM PARSE
        # -------------------------
        teams = row[1].split("\n")
        if len(teams) < 2:
            continue

        away_team = clean_team(teams[0])
        home_team = clean_team(teams[1])

        key = (home_team, away_team)

        # -------------------------
        # PREDICTIONS
        # -------------------------
        if not is_past and len(row) >= 8:

            try:
                pitchers = row[2].split("\n")
                home_pitcher = pitchers[0].strip()
                away_pitcher = pitchers[1].strip()

                probs = row[3].split("\n")
                away_prob = pct_to_decimal(probs[0])
                home_prob = pct_to_decimal(probs[1])

                runs = row[6].split("\n")
                away_runs = runs[0]
                home_runs = runs[1]

                total_runs = row[7]

                pred_row = [
                    "",  # game_id blank
                    "baseball",
                    "mlb",
                    game_date,
                    game_time,
                    home_team,
                    away_team,
                    home_pitcher,
                    away_pitcher,
                    home_prob,
                    away_prob,
                    away_runs,
                    home_runs,
                    total_runs
                ]

                predictions_by_date.setdefault(game_date, []).append(pred_row)

            except:
                continue

        # -------------------------
        # FINAL SCORES
        # -------------------------
        if is_past and len(row) >= 6:

            try:
                scores = row[5].split("\n")
                away_score = int(scores[0])
                home_score = int(scores[1])
                final_total = str(away_score + home_score)

                pred_lookup = load_predictions_lookup(game_date)
                book_lookup = load_sportsbook_lookup(game_date)

                game_id = pred_lookup.get(key, "")

                book = book_lookup.get(key, {})

                final_row = [
                    "baseball",
                    "mlb",
                    game_id,
                    game_date,
                    game_time,
                    home_team,
                    away_team,
                    str(away_score),
                    str(home_score),
                    final_total,
                    book.get("away_run_line"),
                    book.get("home_run_line"),
                    book.get("total"),
                ]

                final_scores_by_date.setdefault(game_date, []).append(final_row)

            except:
                continue

    # -------------------------
    # WRITE PREDICTIONS
    # -------------------------
    for date, rows in predictions_by_date.items():
        out = PRED_DIR / f"{date}_MLB.csv"

        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "game_id","sport","league","game_date","game_time",
                "home_team","away_team","home_pitcher","away_pitcher",
                "home_prob","away_prob",
                "away_projected_runs","home_projected_runs","total_projected_runs"
            ])
            writer.writerows(rows)

    # -------------------------
    # WRITE FINAL SCORES
    # -------------------------
    for date, rows in final_scores_by_date.items():
        out = FINAL_DIR / f"{date}_final_scores_MLB.csv"

        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "sport","league","game_id","game_date","game_time",
                "home_team","away_team",
                "final_away_score","final_home_score","final_total",
                "away_run_line","home_run_line","total"
            ])
            writer.writerows(rows)


# -------------------------
# ENTRY
# -------------------------

if __name__ == "__main__":
    for file in sorted(RAW_DIR.glob("*_mlb_raw.json")):
        process_file(file)
