#!/usr/bin/env python3
# docs/win/baseball/scripts/05_final_scores/01_mlb_results_grade.py

from datetime import datetime, UTC
from pathlib import Path
import pandas as pd

SELECT_DIR  = Path("docs/win/baseball/04_select")
SCORE_DIR   = Path("docs/win/baseball/05_final_scores/results/final_scores")
OUTPUT_DIR  = Path("docs/win/baseball/05_final_scores/results/graded")
DAILY_DIR   = OUTPUT_DIR / "daily"
ERROR_DIR   = Path("docs/win/baseball/errors/05_final_scores")

ERROR_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DAILY_DIR.mkdir(parents=True, exist_ok=True)

GRADE_ERROR_LOG   = ERROR_DIR / "mlb_results_grade_errors.txt"
GRADE_SUMMARY_LOG = ERROR_DIR / "mlb_results_grade_summary.txt"

OUTPUT_COLS = [
    "game_id", "sport", "league", "game_date", "game_time",
    "home_team", "away_team", "market_type", "bet_side", "line",
    "take_bet", "dk_odds_american", "model_prob", "ev", "kelly",
    "low_confidence", "final_home_score", "final_away_score",
    "final_total", "home_run_line", "away_run_line", "total", "bet_result",
]

def reset_logs():
    GRADE_ERROR_LOG.write_text("", encoding="utf-8")
    GRADE_SUMMARY_LOG.write_text("", encoding="utf-8")

def log_error(msg):
    with open(GRADE_ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now(UTC).isoformat()}] {msg}\n")

def log_summary(msg):
    with open(GRADE_SUMMARY_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now(UTC).isoformat()}] {msg}\n")

def safe_read(path):
    try:
        path = Path(path)
        if not path.exists():
            log_error(f"MISSING FILE | {path}")
            return pd.DataFrame()
        df = pd.read_csv(path, dtype=str)
        if df is None or df.empty:
            log_error(f"EMPTY FILE | {path}")
            return pd.DataFrame()
        df = df.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))
        return df
    except Exception as e:
        log_error(f"READ ERROR | {path} | {e}")
        return pd.DataFrame()

def normalize_date(val):
    return str(val).strip().replace("-", "_")

def clean_game_id(series):
    return series.fillna("").astype(str).str.strip().str.split(".").str[0]

def enforce_output_cols(df):
    for col in OUTPUT_COLS:
        if col not in df.columns:
            df[col] = ""
    return df[OUTPUT_COLS].copy()

def determine_outcome(row):
    try:
        market = str(row.get("market_type", "")).strip().lower()
        side   = str(row.get("bet_side",    "")).strip().lower()
        away   = float(row["final_away_score"])
        home   = float(row["final_home_score"])

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
        log_error(f"DETERMINE OUTCOME ERROR | row={dict(row.get(['market_type','bet_side','final_away_score','final_home_score'], {})) if hasattr(row,'get') else row} | {e}")

    return "Unknown"

def grade_league():
    # Load select files
    select_files = sorted(SELECT_DIR.glob("*MLB*.csv"))
    if not select_files:
        log_error(f"NO SELECT FILES FOUND IN {SELECT_DIR}")
        return

    parts = []
    for f in select_files:
        df = safe_read(f)
        if not df.empty:
            df["game_date"] = df["game_date"].apply(normalize_date)
            parts.append(df)

    if not parts:
        log_error("ALL SELECT FILES EMPTY OR UNREADABLE")
        return

    all_bets = pd.concat(parts, ignore_index=True)
    all_bets["game_id"] = clean_game_id(all_bets.get("game_id", pd.Series(dtype=str)))

    # Load score files
    score_files = sorted(SCORE_DIR.glob("*_final_scores_MLB.csv"))
    if not score_files:
        log_error(f"NO SCORE FILES FOUND IN {SCORE_DIR}")
        return

    score_parts = []
    for sf in score_files:
        df = safe_read(sf)
        if not df.empty:
            df["game_date"] = df["game_date"].apply(normalize_date)
            score_parts.append(df)

    if not score_parts:
        log_error("ALL SCORE FILES EMPTY OR UNREADABLE")
        return

    all_scores = pd.concat(score_parts, ignore_index=True)
    all_scores["game_id"] = clean_game_id(all_scores.get("game_id", pd.Series(dtype=str)))

    log_summary(f"BET cols: {list(all_bets.columns)}")
    log_summary(f"SCORE cols: {list(all_scores.columns)}")
    log_summary(f"BET game_id sample: {all_bets['game_id'].head(3).tolist()}")
    log_summary(f"SCORE game_id sample: {all_scores['game_id'].head(3).tolist()}")

    # Merge on game_id
    merged = pd.merge(
        all_bets,
        all_scores,
        on="game_id",
        how="inner",
        suffixes=("_bet", "_score")
    )

    if merged.empty:
        log_error("MERGE EMPTY")
        return

    log_summary(f"MERGED cols: {list(merged.columns)}")
    log_summary(f"MERGED ON game_id | rows={len(merged)}")

    # Resolve duplicate columns - score wins for score fields, bet wins for bet fields
    score_fields = {"game_date", "game_time", "home_team", "away_team", "sport", "league",
                    "final_home_score", "final_away_score", "final_total",
                    "home_run_line", "away_run_line", "total"}

    for base in score_fields:
        score_col = f"{base}_score"
        bet_col   = f"{base}_bet"
        if score_col in merged.columns:
            merged[base] = merged[score_col]
        elif base not in merged.columns and bet_col in merged.columns:
            merged[base] = merged[bet_col]

    # For bet-origin fields, prefer _bet suffix
    for col in list(merged.columns):
        if col.endswith("_bet"):
            base = col[:-4]
            if base not in merged.columns:
                merged[base] = merged[col]

    # Drop suffixed cols — but ONLY true suffixed duplicates, not natural _score names
    protected = {"final_home_score", "final_away_score", "final_total",
                 "home_run_line", "away_run_line"}
    to_drop = []
    for col in merged.columns:
        if col in protected:
            continue
        if col.endswith("_bet") or col.endswith("_score"):
            to_drop.append(col)
    merged = merged.drop(columns=to_drop, errors="ignore")

    log_summary(f"POST-RESOLVE cols: {list(merged.columns)}")
    log_summary(f"final_away_score sample: {merged['final_away_score'].head(3).tolist() if 'final_away_score' in merged.columns else 'MISSING'}")
    log_summary(f"final_home_score sample: {merged['final_home_score'].head(3).tolist() if 'final_home_score' in merged.columns else 'MISSING'}")
    log_summary(f"market_type sample: {merged['market_type'].head(3).tolist() if 'market_type' in merged.columns else 'MISSING'}")

    merged["bet_result"] = merged.apply(determine_outcome, axis=1)

    key_cols = [c for c in ["game_id", "market_type", "bet_side"] if c in merged.columns]
    merged = merged.drop_duplicates(subset=key_cols, keep="last")

    final = enforce_output_cols(merged)
    master_path = OUTPUT_DIR / "MLB_final.csv"
    final.to_csv(master_path, index=False)
    log_summary(f"MLB MASTER BUILT | ROWS={len(final)} | OUT={master_path}")

    for date_val, group in merged.groupby("game_date"):
        date_str   = normalize_date(date_val)
        daily_df   = enforce_output_cols(group.copy())
        daily_path = DAILY_DIR / f"{date_str}_MLB_final.csv"
        daily_df.to_csv(daily_path, index=False)
        result_counts = group["bet_result"].value_counts().to_dict()
        log_summary(f"MLB DAILY | DATE={date_str} | ROWS={len(daily_df)} | RESULTS={result_counts}")

def main():
    reset_logs()
    log_summary("START 01_mlb_results_grade.py")
    grade_league()
    log_summary("END 01_mlb_results_grade.py")
    print("MLB grading complete.")

if __name__ == "__main__":
    main()
