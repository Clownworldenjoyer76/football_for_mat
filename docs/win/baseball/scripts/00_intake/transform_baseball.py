#!/usr/bin/env python3
# docs/win/baseball/scripts/00_intake/transform_baseball.py

import json
import csv
import traceback
from pathlib import Path
from datetime import datetime

ERROR_DIR = Path("docs/win/baseball/errors/00_intake")
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ERROR_DIR / "transform_baseball.txt"

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write(f"=== transform_baseball RUN {datetime.now().isoformat()} ===\n")


def log(msg: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} | {msg}\n")


# -------------------------
# PATHS
# -------------------------

RAW_DIR             = Path("docs/win/baseball/00_intake/drat_raw")
PRED_DIR            = Path("docs/win/baseball/00_intake/predictions")
FINAL_DIR           = Path("docs/win/final_scores/results/mlb/final_scores")
BASEBALL_FINAL_DIR  = Path("docs/win/baseball/05_final_scores/results/final_scores")
SPORTSBOOK_DIR      = Path("docs/win/baseball/00_intake/sportsbook")

PRED_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)
BASEBALL_FINAL_DIR.mkdir(parents=True, exist_ok=True)


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
# ROW DETECTION
# Cell count is the reliable differentiator:
#   11 cells = future game
#    8 cells = completed game
# -------------------------

def is_future_game(row):
    return len(row) == 11


def is_completed_game(row):
    return len(row) == 8


SUMMARY_ROW_PREFIXES = {"Sportsbooks", "DRatings"}


def is_summary_row(row):
    return row and str(row[0]).strip() in SUMMARY_ROW_PREFIXES


# -------------------------
# WRITE HELPERS
# -------------------------

def write_csv(path, header, rows, files_written, label):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    files_written.append((str(path), len(rows)))
    log(f"WROTE {label} -> {path} ({len(rows)} rows)")


# -------------------------
# PROCESS
# -------------------------

def process_file(file_path, files_written, seen_final_keys):
    log(f"Processing {file_path.name}")

    with open(file_path, "r") as f:
        data = json.load(f)

    predictions_by_date = {}
    final_scores_by_date = {}
    parse_errors = 0
    skipped_summary = 0
    skipped_duplicate = 0

    for row in data:
        if not row or len(row) < 2:
            continue

        if is_summary_row(row):
            skipped_summary += 1
            continue

        try:
            dt, game_date, game_time = parse_datetime(row[0])
        except Exception:
            parse_errors += 1
            continue

        teams = row[1].split("\n")
        if len(teams) < 2:
            continue

        away_team = clean_team(teams[0])
        home_team = clean_team(teams[1])
        key = (home_team, away_team)

        if is_future_game(row):
            # Cell layout (11 cells):
            #   0: date/time
            #   1: teams (with records)
            #   2: pitchers (away\nhome)
            #   3: win probs (away%\nhome%)
            #   4: moneyline (away\nhome)
            #   5: run line (away\nhome)
            #   6: projected runs (away\nhome)
            #   7: total projected runs
            #   8: over/under lines
            #   9: bet value label
            #  10: (empty)
            try:
                pitchers = row[2].split("\n")
                away_pitcher = pitchers[0].strip()
                home_pitcher = pitchers[1].strip() if len(pitchers) > 1 else ""

                probs = row[3].split("\n")
                away_prob = pct_to_decimal(probs[0])
                home_prob = pct_to_decimal(probs[1]) if len(probs) > 1 else ""

                runs = row[6].split("\n")
                away_runs = runs[0].strip()
                home_runs = runs[1].strip() if len(runs) > 1 else ""

                total_runs = row[7]

                pred_row = [
                    "", "baseball", "mlb", game_date, game_time,
                    home_team, away_team,
                    home_pitcher, away_pitcher,
                    home_prob, away_prob,
                    away_runs, home_runs, total_runs,
                ]

                predictions_by_date.setdefault(game_date, []).append(pred_row)

            except Exception:
                parse_errors += 1
                continue

        elif is_completed_game(row):
            # Cell layout (8 cells):
            #   0: date/time
            #   1: teams (no records)
            #   2: win probs
            #   3: moneyline
            #   4: run line
            #   5: score (away\nhome)
            #   6: rating
            #   7: rating
            try:
                dedup_key = (game_date, home_team, away_team)

                if dedup_key in seen_final_keys:
                    skipped_duplicate += 1
                    continue

                seen_final_keys.add(dedup_key)

                scores = row[5].split("\n")
                away_score = int(scores[0].strip())
                home_score = int(scores[1].strip()) if len(scores) > 1 else 0
                final_total = str(away_score + home_score)

                pred_lookup = load_predictions_lookup(game_date)
                book_lookup = load_sportsbook_lookup(game_date)

                game_id = pred_lookup.get(key, "")
                book = book_lookup.get(key, {})

                final_row = [
                    "baseball", "mlb", game_id,
                    game_date, game_time,
                    home_team, away_team,
                    str(away_score), str(home_score), final_total,
                    book.get("away_run_line"),
                    book.get("home_run_line"),
                    book.get("total"),
                ]

                final_scores_by_date.setdefault(game_date, []).append(final_row)

            except Exception:
                parse_errors += 1
                continue

        else:
            log(f"  SKIPPED unknown row ({len(row)} cells): {row[0]} | {row[1]}")

    prediction_header = [
        "game_id", "sport", "league", "game_date", "game_time",
        "home_team", "away_team", "home_pitcher", "away_pitcher",
        "home_prob", "away_prob",
        "away_projected_runs", "home_projected_runs", "total_projected_runs",
    ]

    for date, rows in predictions_by_date.items():
        out = PRED_DIR / f"{date}_MLB.csv"
        write_csv(out, prediction_header, rows, files_written, "predictions")

    final_header = [
        "sport", "league", "game_id", "game_date", "game_time",
        "home_team", "away_team",
        "final_away_score", "final_home_score", "final_total",
        "away_run_line", "home_run_line", "total",
    ]

    for date, rows in final_scores_by_date.items():
        output_paths = [
            FINAL_DIR / f"{date}_final_scores_MLB.csv",
            BASEBALL_FINAL_DIR / f"{date}_final_scores_MLB.csv",
        ]

        for out in output_paths:
            write_csv(out, final_header, rows, files_written, "final scores")

    log(
        f"  parse_errors={parse_errors}, "
        f"skipped_summary={skipped_summary}, "
        f"skipped_duplicate={skipped_duplicate}, "
        f"predictions_dates={len(predictions_by_date)}, "
        f"final_score_dates={len(final_scores_by_date)}"
    )


# -------------------------
# ENTRY
# -------------------------

def main():
    files_written = []
    seen_final_keys = set()

    try:
        raw_files = sorted(RAW_DIR.glob("*_mlb_raw.json"))
        log(f"Raw files found: {len(raw_files)}")

        for file in raw_files:
            process_file(file, files_written, seen_final_keys)

        log("--- SUMMARY ---")
        log(f"Raw files processed: {len(raw_files)}")
        log(f"Files written: {len(files_written)}")

        for path, count in files_written:
            log(f"  FILE: {path} ({count} rows)")

        log("STATUS: SUCCESS")

    except Exception as e:
        log(f"FATAL ERROR: {e}\n{traceback.format_exc()}")
        log("STATUS: FAILED")
        raise

    print("Baseball transform complete.")


if __name__ == "__main__":
    main()
