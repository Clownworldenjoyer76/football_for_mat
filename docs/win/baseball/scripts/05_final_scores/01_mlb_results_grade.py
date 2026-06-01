#!/usr/bin/env python3
# docs/win/baseball/scripts/05_final_scores/01_mlb_results_grade.py

from datetime import datetime, UTC
from pathlib import Path
import csv
import sys
import pandas as pd

SELECT_DIR = Path("docs/win/baseball/04_select")
SCORE_DIR = Path("docs/win/baseball/05_final_scores/results/final_scores")
OUTPUT_DIR = Path("docs/win/baseball/05_final_scores/results/graded")
DAILY_DIR = OUTPUT_DIR / "daily"
UNMATCHED_DIR = Path("docs/win/baseball/05_final_scores/results/unmatched")
ERROR_DIR = Path("docs/win/baseball/errors/05_final_scores")

ERROR_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DAILY_DIR.mkdir(parents=True, exist_ok=True)
UNMATCHED_DIR.mkdir(parents=True, exist_ok=True)

GRADE_ERROR_LOG = ERROR_DIR / "mlb_results_grade_errors.txt"
GRADE_SUMMARY_LOG = ERROR_DIR / "mlb_results_grade_summary.txt"

OUTPUT_COLS = [
    "game_id", "sport", "league", "game_date", "game_time",
    "home_team", "away_team", "market_type", "bet_side", "line",
    "take_bet", "dk_odds_american", "model_prob", "ev", "kelly",
    "low_confidence", "final_home_score", "final_away_score",
    "final_total", "home_run_line", "away_run_line", "total", "bet_result",
]

UNMATCHED_COLS = [
    "unmatched_reason", "game_id", "sport", "league", "game_date", "game_time",
    "home_team", "away_team", "market_type", "bet_side", "line",
    "take_bet", "dk_odds_american", "model_prob", "ev", "kelly",
    "low_confidence", "source_file",
]

REQUIRED_SELECTED_COLUMNS = [
    "game_id", "game_date", "market_type", "bet_side", "line",
]

REQUIRED_SCORE_COLUMNS = [
    "game_id", "game_date", "final_home_score", "final_away_score",
]


def _now():
    return datetime.now(UTC).isoformat()


def reset_logs():
    GRADE_ERROR_LOG.write_text("", encoding="utf-8")
    GRADE_SUMMARY_LOG.write_text("", encoding="utf-8")


def log_error(msg):
    with open(GRADE_ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{_now()}] {msg}\n")


def log_summary(msg):
    with open(GRADE_SUMMARY_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{_now()}] {msg}\n")


def duplicate_columns(columns):
    seen = set()
    duplicates = []
    for col in columns:
        if col in seen and col not in duplicates:
            duplicates.append(col)
        seen.add(col)
    return duplicates


def read_header_columns(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        return next(reader, [])


def validate_no_duplicate_columns(df, label):
    dupes = duplicate_columns(list(df.columns))
    if dupes:
        raise ValueError(f"{label} has duplicate columns: {dupes}")


def validate_no_duplicate_header(path, label):
    header = read_header_columns(path)
    dupes = duplicate_columns(header)
    if dupes:
        raise ValueError(f"{label} has duplicate header columns: {dupes}")


def validate_required_columns(df, required_columns, label):
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def write_csv_validated(df, path, label):
    validate_no_duplicate_columns(df, label)
    df.to_csv(path, index=False)


def safe_read(path, required_columns=None, label=None):
    try:
        path = Path(path)
        read_label = label or str(path)

        if not path.exists():
            log_error(f"MISSING FILE | {path}")
            return pd.DataFrame()

        validate_no_duplicate_header(path, read_label)

        df = pd.read_csv(path, dtype=str)
        if df is None or df.empty:
            log_error(f"EMPTY FILE | {path}")
            return pd.DataFrame()

        validate_no_duplicate_columns(df, read_label)

        if required_columns:
            validate_required_columns(df, required_columns, read_label)

        df = df.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))
        return df

    except Exception as e:
        log_error(f"READ/SCHEMA ERROR | {path} | {e}")
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


def enforce_unmatched_cols(df):
    for col in UNMATCHED_COLS:
        if col not in df.columns:
            df[col] = ""
    return df[UNMATCHED_COLS].copy()


def validate_score_game_ids(all_scores):
    blank_scores = all_scores[all_scores["game_id"].fillna("").astype(str).str.strip() == ""]
    if not blank_scores.empty:
        raise ValueError(f"final-score rows with blank game_id: {len(blank_scores)}")

    duplicated = all_scores[all_scores["game_id"].duplicated(keep=False)].copy()
    if not duplicated.empty:
        duplicate_path = UNMATCHED_DIR / "duplicate_final_score_game_ids_MLB.csv"
        write_csv_validated(
            duplicated.sort_values(["game_id", "game_date"], na_position="last"),
            duplicate_path,
            "duplicate final-score game_id audit",
        )
        raise ValueError(
            f"duplicate final-score game_id rows: {len(duplicated)} | audit={duplicate_path}"
        )


def determine_outcome(row):
    try:
        market = str(row.get("market_type", "")).strip().lower()
        side = str(row.get("bet_side", "")).strip().lower()
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
            line = float(row.get("line", 0))
            total = away + home
            if abs(total - line) < 1e-9:
                return "Push"
            if side == "over":
                return "Win" if total > line else "Loss"
            if side == "under":
                return "Win" if total < line else "Loss"

    except Exception as e:
        log_error(
            "DETERMINE OUTCOME ERROR | "
            f"game_id={row.get('game_id', '')} market_type={row.get('market_type', '')} "
            f"bet_side={row.get('bet_side', '')} | {e}"
        )

    return "Unknown"


def write_unmatched(unmatched):
    if unmatched.empty:
        return None

    unmatched = enforce_unmatched_cols(unmatched.copy())
    unmatched_path = UNMATCHED_DIR / "MLB_unmatched_selected_bets.csv"
    write_csv_validated(unmatched, unmatched_path, "unmatched selected bets output")
    return unmatched_path


def grade_league():
    select_files = sorted(SELECT_DIR.glob("*MLB*.csv"))
    if not select_files:
        log_error(f"NO SELECT FILES FOUND IN {SELECT_DIR}")
        return False

    parts = []
    for f in select_files:
        df = safe_read(f, REQUIRED_SELECTED_COLUMNS, f"selected file {f.name}")
        if not df.empty:
            df["source_file"] = f.name
            df["game_date"] = df["game_date"].apply(normalize_date)
            parts.append(df)

    if not parts:
        log_error("ALL SELECT FILES EMPTY, UNREADABLE, OR SCHEMA-INVALID")
        return False

    all_bets = pd.concat(parts, ignore_index=True)
    validate_no_duplicate_columns(all_bets, "combined selected bets")
    all_bets["game_id"] = clean_game_id(all_bets.get("game_id", pd.Series(dtype=str)))
    all_bets["selected_row_id"] = range(len(all_bets))

    score_files = sorted(SCORE_DIR.glob("*_final_scores_MLB.csv"))
    if not score_files:
        log_error(f"NO SCORE FILES FOUND IN {SCORE_DIR}")
        return False

    score_parts = []
    for sf in score_files:
        df = safe_read(sf, REQUIRED_SCORE_COLUMNS, f"score file {sf.name}")
        if not df.empty:
            df["game_date"] = df["game_date"].apply(normalize_date)
            score_parts.append(df)

    if not score_parts:
        log_error("ALL SCORE FILES EMPTY, UNREADABLE, OR SCHEMA-INVALID")
        return False

    all_scores = pd.concat(score_parts, ignore_index=True)
    validate_no_duplicate_columns(all_scores, "combined final scores")
    all_scores["game_id"] = clean_game_id(all_scores.get("game_id", pd.Series(dtype=str)))

    try:
        validate_score_game_ids(all_scores)
    except Exception as e:
        log_error(f"FINAL SCORE GAME_ID VALIDATION FAILED | {e}")
        return False

    selected_blank_id = all_bets[all_bets["game_id"].fillna("").astype(str).str.strip() == ""].copy()
    selected_valid = all_bets[all_bets["game_id"].fillna("").astype(str).str.strip() != ""].copy()

    if not selected_blank_id.empty:
        selected_blank_id["unmatched_reason"] = "missing_game_id"

    log_summary(f"BET cols: {list(all_bets.columns)}")
    log_summary(f"SCORE cols: {list(all_scores.columns)}")
    log_summary(f"SELECTED ROWS: {len(all_bets)}")
    log_summary(f"SELECTED BLANK GAME_ID ROWS: {len(selected_blank_id)}")
    log_summary(f"SCORE ROWS: {len(all_scores)}")
    log_summary(f"BET game_id sample: {all_bets['game_id'].head(3).tolist()}")
    log_summary(f"SCORE game_id sample: {all_scores['game_id'].head(3).tolist()}")

    merged_all = pd.merge(
        selected_valid,
        all_scores,
        on="game_id",
        how="left",
        suffixes=("_bet", "_score"),
        indicator=True,
    )

    missing_scores = merged_all[merged_all["_merge"] == "left_only"].copy()
    if not missing_scores.empty:
        missing_scores["unmatched_reason"] = "missing_final_score"

    unmatched_parts = []
    if not selected_blank_id.empty:
        unmatched_parts.append(selected_blank_id)
    if not missing_scores.empty:
        unmatched_parts.append(missing_scores)

    unmatched = pd.concat(unmatched_parts, ignore_index=True) if unmatched_parts else pd.DataFrame()
    unmatched_path = write_unmatched(unmatched)

    merged = merged_all[merged_all["_merge"] == "both"].drop(columns=["_merge"], errors="ignore").copy()

    if merged.empty:
        log_error("MERGE EMPTY")
        if unmatched_path:
            log_error(f"UNMATCHED SELECTED BETS WRITTEN | {unmatched_path}")
        return False

    log_summary(f"MERGED cols: {list(merged.columns)}")
    log_summary(f"MERGED ON game_id | rows={len(merged)}")
    log_summary(f"UNMATCHED SELECTED ROWS: {len(unmatched)}")
    if unmatched_path:
        log_error(f"UNMATCHED SELECTED BETS WRITTEN | rows={len(unmatched)} | out={unmatched_path}")

    score_fields = {
        "game_date", "game_time", "home_team", "away_team", "sport", "league",
        "final_home_score", "final_away_score", "final_total",
        "home_run_line", "away_run_line", "total",
    }

    for base in score_fields:
        score_col = f"{base}_score"
        bet_col = f"{base}_bet"
        if score_col in merged.columns:
            merged[base] = merged[score_col]
        elif base not in merged.columns and bet_col in merged.columns:
            merged[base] = merged[bet_col]

    for col in list(merged.columns):
        if col.endswith("_bet"):
            base = col[:-4]
            if base not in merged.columns:
                merged[base] = merged[col]

    protected = {
        "final_home_score", "final_away_score", "final_total",
        "home_run_line", "away_run_line",
    }
    to_drop = []
    for col in merged.columns:
        if col in protected:
            continue
        if col.endswith("_bet") or col.endswith("_score"):
            to_drop.append(col)
    merged = merged.drop(columns=to_drop, errors="ignore")
    validate_no_duplicate_columns(merged, "post-resolve graded rows")

    log_summary(f"POST-RESOLVE cols: {list(merged.columns)}")
    log_summary(
        "final_away_score sample: "
        f"{merged['final_away_score'].head(3).tolist() if 'final_away_score' in merged.columns else 'MISSING'}"
    )
    log_summary(
        "final_home_score sample: "
        f"{merged['final_home_score'].head(3).tolist() if 'final_home_score' in merged.columns else 'MISSING'}"
    )
    log_summary(
        "market_type sample: "
        f"{merged['market_type'].head(3).tolist() if 'market_type' in merged.columns else 'MISSING'}"
    )

    merged["bet_result"] = merged.apply(determine_outcome, axis=1)

    final = enforce_output_cols(merged)
    master_path = OUTPUT_DIR / "MLB_final.csv"
    write_csv_validated(final, master_path, "MLB graded master output")

    selected_count = len(all_bets)
    graded_count = len(final)
    unmatched_count = len(unmatched)

    log_summary(f"MLB MASTER BUILT | ROWS={graded_count} | OUT={master_path}")
    log_summary(
        f"SELECTED VS GRADED | selected={selected_count} graded={graded_count} "
        f"unmatched={unmatched_count}"
    )

    if selected_count > graded_count:
        log_error(
            f"SELECTED COUNT EXCEEDS GRADED COUNT | selected={selected_count} "
            f"graded={graded_count} unmatched={unmatched_count}"
        )

    for date_val, group in merged.groupby("game_date"):
        date_str = normalize_date(date_val)
        daily_df = enforce_output_cols(group.copy())
        daily_path = DAILY_DIR / f"{date_str}_MLB_final.csv"
        write_csv_validated(daily_df, daily_path, f"MLB graded daily output {date_str}")
        result_counts = group["bet_result"].value_counts().to_dict()
        log_summary(f"MLB DAILY | DATE={date_str} | ROWS={len(daily_df)} | RESULTS={result_counts}")

    return True


def main():
    reset_logs()
    log_summary("START 01_mlb_results_grade.py")
    success = grade_league()
    log_summary("END 01_mlb_results_grade.py")

    if not success:
        print("MLB grading completed with errors. Check logs.")
        sys.exit(1)

    print("MLB grading complete.")


if __name__ == "__main__":
    main()
