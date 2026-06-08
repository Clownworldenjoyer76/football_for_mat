#!/usr/bin/env python3
# docs/win/final_scores/scripts/05_results/mlb/01_mlb_results_grade.py

from datetime import datetime, UTC
from pathlib import Path

import pandas as pd

###############################################################
######################## PATH CONFIG ##########################
###############################################################

SELECT_DIR = Path("docs/win/baseball/04_select")
SCORE_DIR  = Path("docs/win/final_scores/results/mlb/final_scores")
OUTPUT_DIR = Path("docs/win/final_scores/results/mlb/graded")

ERROR_DIR = Path("docs/win/final_scores/errors")
ERROR_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GRADE_ERROR_LOG   = ERROR_DIR / "mlb_results_grade_errors.txt"
GRADE_SUMMARY_LOG = ERROR_DIR / "mlb_results_grade_summary.txt"

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


def clean_merge_columns(df):
    for col in list(df.columns):
        if col.endswith("_x"):
            df[col[:-2]] = df[col]
        elif col.endswith("_y"):
            base = col[:-2]
            if base not in df.columns:
                df[base] = df[col]
    df = df.drop(columns=[c for c in df.columns if c.endswith("_x") or c.endswith("_y")], errors="ignore")
    return df


###############################################################
######################## OUTCOME LOGIC ########################
###############################################################

def determine_outcome(row):
    try:
        market = str(row.get("market_type", "")).strip().lower()
        side   = str(row.get("bet_side",    "")).strip().lower()

        away = float(row["final_away_score"])
        home = float(row["final_home_score"])

        if market == "moneyline":
            if away == home:
                return "Push"
            if side == "home":
                return "Win" if home > away else "Loss"
            if side == "away":
                return "Win" if away > home else "Loss"

        if market == "run_line":
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
######################## EDGE REBUILD #########################
###############################################################

def rebuild_selected_edge(df):
    has_decimal  = "home_edge_decimal_moneyline" in df.columns
    has_standard = "home_ml_edge" in df.columns

    def pick_edge(row):
        market = row["market_type"]
        side   = row["bet_side"]

        if market == "moneyline":
            if has_decimal:
                return row.get("home_edge_decimal_moneyline") if side == "home" else row.get("away_edge_decimal_moneyline")
            if has_standard:
                return row.get("home_ml_edge") if side == "home" else row.get("away_ml_edge")

        if market == "run_line":
            if "home_edge_decimal_run_line" in row.index:
                return row.get("home_edge_decimal_run_line") if side == "home" else row.get("away_edge_decimal_run_line")
            if "home_rl_edge" in row.index:
                return row.get("home_rl_edge") if side == "home" else row.get("away_rl_edge")

        if market == "total":
            if "over_edge_decimal_total" in row.index:
                return row.get("over_edge_decimal_total") if side == "over" else row.get("under_edge_decimal_total")
            if "over_edge" in row.index:
                return row.get("over_edge") if side == "over" else row.get("under_edge")

        return None

    df["selected_edge"] = df.apply(pick_edge, axis=1)
    return df


###############################################################
######################## GRADING ##############################
###############################################################

def grade_league():
    # Load all select files upfront and normalise game_date
    select_files = sorted(SELECT_DIR.glob("*MLB*.csv"))

    if not select_files:
        log_error(f"NO SELECT FILES FOUND IN {SELECT_DIR}")
        return

    parts = []
    for f in select_files:
        df = safe_read(f)
        if not df.empty:
            df["game_date"] = df["game_date"].astype(str).str.strip().str.replace("-", "_")
            parts.append(df)

    if not parts:
        log_error("ALL SELECT FILES EMPTY OR UNREADABLE")
        return

    all_bets = pd.concat(parts, ignore_index=True)

    # Iterate over existing score files — source of truth for graded dates
    score_files = sorted(SCORE_DIR.glob("*_final_scores_MLB.csv"))

    if not score_files:
        log_error(f"NO SCORE FILES FOUND IN {SCORE_DIR}")
        return

    all_results = []

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

            merged = pd.merge(
                bets,
                scores,
                on=["away_team", "home_team", "game_date"],
                how="inner"
            )

            if merged.empty:
                log_error(f"MERGE EMPTY | {score_file.name} — check team name consistency")
                continue

            merged = clean_merge_columns(merged)
            merged = rebuild_selected_edge(merged)
            merged["bet_result"] = merged.apply(determine_outcome, axis=1)

            all_results.append(merged)

            result_counts = merged["bet_result"].astype(str).value_counts().to_dict()
            date_str = score_file.name.split("_final_scores_")[0]
            log_summary(f"MLB GRADED | DATE={date_str} | ROWS={len(merged)} | RESULTS={result_counts}")

        except Exception as e:
            log_error(f"GRADE LOOP ERROR | {score_file.name} | {e}")

    if all_results:
        final = pd.concat(all_results, ignore_index=True)

        # Deduplicate
        key_cols = [c for c in ["game_date", "away_team", "home_team", "market_type", "bet_side"] if c in final.columns]
        final = final.drop_duplicates(subset=key_cols, keep="last")

        out_path = OUTPUT_DIR / "MLB_final.csv"
        final.to_csv(out_path, index=False)
        log_summary(f"MLB MASTER BUILT | ROWS={len(final)} | OUT={out_path}")
    else:
        log_error("NO RESULTS TO SAVE")


###############################################################
######################## MAIN #################################
###############################################################

def main():
    reset_logs()
    log_summary("START mlb_results_grade.py")
    grade_league()
    log_summary("END mlb_results_grade.py")
    print("MLB grading complete.")


if __name__ == "__main__":
    main()
