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
# SETTINGS
# -------------------------

DOUBLEHEADER_TIME_TOLERANCE_MINUTES = 90

# -------------------------
# HELPERS
# -------------------------

def parse_datetime(dt_str):
    dt = datetime.strptime(dt_str.strip(), "%m/%d/%Y %I:%M %p")
    return dt, dt.strftime("%Y_%m_%d"), dt.strftime("%I:%M %p")


def parse_time_value(value):
    value = str(value).strip()

    if not value:
        return None

    formats = [
        "%I:%M %p",
        "%H:%M:%S",
        "%H:%M",
    ]

    for fmt in formats:
        try:
            parsed = datetime.strptime(value, fmt)
            return parsed.hour * 60 + parsed.minute
        except ValueError:
            continue

    return None


def clean_team(team_str):
    return team_str.split("(")[0].strip()


def pct_to_decimal(p):
    return str(round(float(p.replace("%", "")) / 100, 3))


def split_lines(val):
    parts = val.split("\n")
    if len(parts) > 1:
        return parts[0].strip(), parts[1].strip()
    return parts[0].strip(), ""


def closest_time_match(candidates, target_game_time, value_field):
    if not candidates:
        return ""

    if len(candidates) == 1:
        return candidates[0].get(value_field, "")

    target_minutes = parse_time_value(target_game_time)

    if target_minutes is None:
        return ""

    best_candidate = None
    best_diff = None

    for candidate in candidates:
        candidate_minutes = parse_time_value(candidate.get("game_time", ""))

        if candidate_minutes is None:
            continue

        diff = abs(candidate_minutes - target_minutes)

        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_candidate = candidate

    if best_candidate is None:
        return ""

    if best_diff > DOUBLEHEADER_TIME_TOLERANCE_MINUTES:
        return ""

    return best_candidate.get(value_field, "")


def closest_time_book_match(candidates, target_game_time):
    if not candidates:
        return {}

    if len(candidates) == 1:
        return candidates[0]

    target_minutes = parse_time_value(target_game_time)

    if target_minutes is None:
        return {}

    best_candidate = None
    best_diff = None

    for candidate in candidates:
        candidate_minutes = parse_time_value(candidate.get("game_time", ""))

        if candidate_minutes is None:
            continue

        diff = abs(candidate_minutes - target_minutes)

        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_candidate = candidate

    if best_candidate is None:
        return {}

    if best_diff > DOUBLEHEADER_TIME_TOLERANCE_MINUTES:
        return {}

    return best_candidate


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
            key = (
                r.get("home_team", "").strip(),
                r.get("away_team", "").strip(),
            )

            lookup.setdefault(key, []).append({
                "game_id": r.get("game_id", ""),
                "game_time": r.get("game_time", ""),
                "home_team": r.get("home_team", ""),
                "away_team": r.get("away_team", ""),
            })

    return lookup


def load_sportsbook_lookup(date):
    path = SPORTSBOOK_DIR / f"{date}_MLB.csv"
    lookup = {}

    if not path.exists():
        return lookup

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for r in reader:
            key = (
                r.get("home_team", "").strip(),
                r.get("away_team", "").strip(),
            )

            lookup.setdefault(key, []).append({
                "game_time": r.get("game_time", ""),
                "away_run_line": r.get("away_run_line"),
                "home_run_line": r.get("home_run_line"),
                "total": r.get("total"),
            })

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
    predictions_lookup_cache = {}
    sportsbook_lookup_cache = {}

    for row in data:
        if not row or len(row) < 2:
            continue

        try:
            dt, game_date, game_time = parse_datetime(row[0])
        except Exception:
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
                    "",
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
                    total_runs,
                ]

                predictions_by_date.setdefault(game_date, []).append(pred_row)

            except Exception:
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

                if game_date not in predictions_lookup_cache:
                    predictions_lookup_cache[game_date] = load_predictions_lookup(game_date)

                if game_date not in sportsbook_lookup_cache:
                    sportsbook_lookup_cache[game_date] = load_sportsbook_lookup(game_date)

                pred_lookup = predictions_lookup_cache[game_date]
                book_lookup = sportsbook_lookup_cache[game_date]

                pred_candidates = pred_lookup.get(key, [])
                game_id = closest_time_match(pred_candidates, game_time, "game_id")

                book_candidates = book_lookup.get(key, [])
                book = closest_time_book_match(book_candidates, game_time)

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

            except Exception:
                continue

    # -------------------------
    # WRITE PREDICTIONS
    # -------------------------
    for date, rows in predictions_by_date.items():
        out = PRED_DIR / f"{date}_MLB.csv"

        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "game_id",
                "sport",
                "league",
                "game_date",
                "game_time",
                "home_team",
                "away_team",
                "home_pitcher",
                "away_pitcher",
                "home_prob",
                "away_prob",
                "away_projected_runs",
                "home_projected_runs",
                "total_projected_runs",
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
                "sport",
                "league",
                "game_id",
                "game_date",
                "game_time",
                "home_team",
                "away_team",
                "final_away_score",
                "final_home_score",
                "final_total",
                "away_run_line",
                "home_run_line",
                "total",
            ])
            writer.writerows(rows)


# -------------------------
# ENTRY
# -------------------------

if __name__ == "__main__":
    for file in sorted(RAW_DIR.glob("*_mlb_raw.json")):
        process_file(file)
