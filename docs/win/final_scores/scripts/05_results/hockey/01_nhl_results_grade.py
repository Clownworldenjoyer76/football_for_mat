#!/usr/bin/env python3
# docs/win/final_scores/scripts/05_results/01_nhl_results_grade.py

import glob
from datetime import datetime, UTC
from pathlib import Path

import pandas as pd

###############################################################
######################## PATH CONFIG ##########################
###############################################################

BASE = Path("docs/win/hockey")
SELECT_DIR = BASE / "04_select"

NHL_SCORE_DIR = Path("docs/win/final_scores/results/nhl/final_scores")
NHL_OUTPUT    = Path("docs/win/final_scores/results/nhl/graded")

INTERMEDIATE_DIR = Path("docs/win/final_scores/intermediate")
INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

ERROR_DIR = Path("docs/win/final_scores/errors")
ERROR_DIR.mkdir(parents=True, exist_ok=True)

GRADE_ERROR_LOG   = ERROR_DIR / "nhl_results_grade_errors.txt"
GRADE_SUMMARY_LOG = ERROR_DIR / "nhl_results_grade_summary.txt"

###############################################################
######################## LOGGING ##############################
###############################################################

def reset_logs():
    GRADE_ERROR_LOG.write_text("", encoding="utf-8")
    GRADE_SUMMARY_LOG.write_text("", encoding="utf-8")


def log_error(msg):
    with open(GRADE_ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now(UTC).isoformat()}] {msg}\n")


def log_summary(msg):
    with open(GRADE_SUMMARY_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now(UTC).isoformat()}] {msg}\n")


###############################################################
######################## HELPERS ##############################
###############################################################

def safe_read(path):
    try:
        path = Path(path)
        if not path.exists():
            log_error(f"MISSING FILE | {path}")
            return pd.DataFrame()
        df = pd.read_csv(path)
        if df is None or df.empty:
            log_error(f"EMPTY FILE | {path}")
            return pd.DataFrame()
        return df
    except Exception as e:
        log_error(f"READ ERROR | {path} | {e}")
        return pd.DataFrame()


def clear_old_outputs():
    try:
        NHL_OUTPUT.mkdir(parents=True, exist_ok=True)
        for f in NHL_OUTPUT.glob("*_results_NHL.csv"):
            f.unlink(missing_ok=True)
        for f in NHL_OUTPUT.glob("NHL_final.csv"):
            f.unlink(missing_ok=True)
        log_summary("CLEARED OLD NHL GRADED OUTPUTS")
    except Exception as e:
        log_error(f"CLEAR OUTPUT ERROR | {e}")


###############################################################
######################## OUTCOME LOGIC ########################
###############################################################

def determine_outcome(row):
    try:
        market = str(row.get("market_type", "")).strip().lower()
        side   = str(row.get("bet_side",    "")).strip().lower()

        away = float(row["away_score"])
        home = float(row["home_score"])

        if market == "moneyline":
            if away == home:
                return "Push"
            if side == "home":
                return "Win" if home > away else "Loss"
            if side == "away":
                return "Win" if away > home else "Loss"

        if market == "puck_line":
            line = float(row.get("line", 0))
            diff = (home + line) - away if side == "home" else (away + line) - home
            if abs(diff) < 1e-9:
                return "Push"
            return "Win" if diff > 0 else "Loss"

        if market == "total":
            line  = float(row.get("line", 0))
            total = away + home
            if abs(total - line) < 1e-9:
                return "Push"
            if side == "over":
                return "Win" if total > line else "Loss"
            if side == "under":
                return "Win" if total < line else "Loss"

    except Exception as e:
        log_error(f"DETERMINE OUTCOME ERROR | {e}")

    return "Unknown"


###############################################################
######################## GRADING ##############################
###############################################################

def grade_league():
    NHL_OUTPUT.mkdir(parents=True, exist_ok=True)

    # Load all select files and normalise game_date to underscore format
    select_files = sorted(SELECT_DIR.glob("*_NHL.csv"))

    if not select_files:
        log_error(f"NO SELECT FILES FOUND IN {SELECT_DIR}")
        return

    select_parts = []
    for f in select_files:
        df = safe_read(f)
        if not df.empty:
            df["game_date"] = df["game_date"].astype(str).str.strip().str.replace("-", "_")
            select_parts.append(df)

    if not select_parts:
        log_error("ALL SELECT FILES EMPTY OR UNREADABLE")
        return

    all_bets = pd.concat(select_parts, ignore_index=True)

    # Iterate over existing score files — source of truth for graded dates
    score_files = sorted(NHL_SCORE_DIR.glob("*_final_scores_NHL.csv"))

    if not score_files:
        log_error(f"NO SCORE FILES FOUND IN {NHL_SCORE_DIR}")
        return

    for score_file in score_files:
        try:
            scores = safe_read(score_file)
            if scores.empty:
                continue

            # Normalise game_date in scores
            scores["game_date"] = scores["game_date"].astype(str).str.strip().str.replace("-", "_")

            score_dates = scores["game_date"].unique().tolist()

            # Filter select rows to dates present in this score file
            bets = all_bets[all_bets["game_date"].isin(score_dates)].copy()

            if bets.empty:
                log_error(f"NO SELECT ROWS FOR DATES {score_dates} | {score_file.name}")
                continue

            df = pd.merge(
                bets,
                scores,
                on=["away_team", "home_team", "game_date"],
                how="inner",
            )

            if df.empty:
                log_error(f"MERGE EMPTY | {score_file.name} — check team name consistency")
                continue

            # Drop duplicate _x/_y columns from merge
            for col in list(df.columns):
                if col.endswith("_x"):
                    df[col[:-2]] = df[col]
                elif col.endswith("_y"):
                    base = col[:-2]
                    if base not in df.columns:
                        df[base] = df[col]
            df = df.drop(columns=[c for c in df.columns if c.endswith("_x") or c.endswith("_y")], errors="ignore")

            df["bet_result"] = df.apply(determine_outcome, axis=1)

            # Write per-date graded file using the date from the score filename
            date_str = score_file.name.split("_final_scores_")[0]
            outfile  = NHL_OUTPUT / f"{date_str}_results_NHL.csv"
            df.to_csv(outfile, index=False)

            result_counts = df["bet_result"].astype(str).value_counts().to_dict()
            log_summary(f"NHL GRADED | DATE={date_str} | ROWS={len(df)} | RESULTS={result_counts}")

        except Exception as e:
            log_error(f"GRADE LOOP ERROR | {score_file.name} | {e}")


###############################################################
######################## MASTER BUILD #########################
###############################################################

def build_master():
    try:
        files = sorted(NHL_OUTPUT.glob("*_results_NHL.csv"))
        parts = [safe_read(f) for f in files]
        parts = [d for d in parts if not d.empty]

        if not parts:
            log_error("NO GRADED FILES FOR MASTER")
            return

        df = pd.concat(parts, ignore_index=True)

        sort_cols = [c for c in ["game_date", "away_team", "home_team", "market_type", "bet_side"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols, kind="mergesort")

        # Deduplicate
        key_cols = [c for c in ["game_date", "away_team", "home_team", "market_type", "bet_side"] if c in df.columns]
        df = df.drop_duplicates(subset=key_cols, keep="last")

        master = NHL_OUTPUT / "NHL_final.csv"
        df.to_csv(master, index=False)
        log_summary(f"NHL MASTER BUILT | ROWS={len(df)} | OUT={master}")

    except Exception as e:
        log_error(f"BUILD MASTER ERROR | {e}")


###############################################################
######################## MAIN #################################
###############################################################

def main():
    reset_logs()
    log_summary("START nhl_results_grade.py")
    clear_old_outputs()
    grade_league()
    build_master()
    log_summary("END nhl_results_grade.py")
    print("NHL grading complete.")


if __name__ == "__main__":
    main()
