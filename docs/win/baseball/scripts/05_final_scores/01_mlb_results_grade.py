#!/usr/bin/env python3
# docs/win/baseball/scripts/05_final_scores/01_mlb_results_grade.py

from datetime import datetime, UTC
from pathlib import Path
import csv
import sys
import traceback

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

FAIL_ON_UNMATCHED_SELECTED_BETS = True

OUTPUT_COLS = [
    "game_id",
    "sport",
    "league",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
    "market_type",
    "bet_side",
    "line",
    "take_bet",
    "dk_odds_american",
    "model_prob",
    "ev",
    "kelly",
    "low_confidence",
    "final_home_score",
    "final_away_score",
    "final_total",
    "home_run_line",
    "away_run_line",
    "total",
    "bet_result",
]

SELECT_REQUIRED_COLS = [
    "game_id",
    "sport",
    "league",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
    "market_type",
    "bet_side",
    "line",
    "take_bet",
    "dk_odds_american",
    "model_prob",
    "ev",
    "kelly",
    "low_confidence",
]

SCORE_REQUIRED_COLS = [
    "game_id",
    "sport",
    "league",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
    "final_home_score",
    "final_away_score",
    "final_total",
    "home_run_line",
    "away_run_line",
    "total",
]

UNMATCHED_OUTPUT_COLS = [
    "unmatched_reason",
    "game_id",
    "sport",
    "league",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
    "market_type",
    "bet_side",
    "line",
    "take_bet",
    "dk_odds_american",
    "model_prob",
    "ev",
    "kelly",
    "low_confidence",
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

<<<<<<< HEAD
def clear_output_files():
    deleted = 0

    master_path = OUTPUT_DIR / "MLB_final.csv"
    if master_path.exists():
        master_path.unlink()
        deleted += 1
        log_summary(f"DELETED OLD OUTPUT | {master_path}")

    for old_file in sorted(DAILY_DIR.glob("*_MLB_final.csv")):
        old_file.unlink()
        deleted += 1
        log_summary(f"DELETED OLD OUTPUT | {old_file}")

    log_summary(f"OLD GRADED OUTPUT FILES DELETED | count={deleted}")

def safe_read(path):
=======

def duplicate_columns(cols):
    seen = set()
    duplicates = []

    for col in cols:
        if col in seen and col not in duplicates:
            duplicates.append(col)
        seen.add(col)

    return duplicates


def read_raw_header(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration:
            return []


def validate_no_duplicate_columns_from_header(path, label):
    header = read_raw_header(path)
    dupes = duplicate_columns(header)

    if dupes:
        raise ValueError(f"{label} has duplicate columns in {path}: {dupes}")


def validate_no_duplicate_columns_df(df, label):
    dupes = duplicate_columns(list(df.columns))

    if dupes:
        raise ValueError(f"{label} has duplicate columns: {dupes}")


def validate_required_columns(df, required_cols, label):
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def safe_read(path, required_cols, label):
    path = Path(path)

    if not path.exists():
        log_error(f"MISSING FILE | {path}")
        return pd.DataFrame()

>>>>>>> 67f12db62f4d2ba562c5ffa0eebb8972e2235186
    try:
        validate_no_duplicate_columns_from_header(path, label)

        df = pd.read_csv(path, dtype=str)

        validate_no_duplicate_columns_df(df, label)
        validate_required_columns(df, required_cols, label)

        if df is None or df.empty:
            log_error(f"EMPTY FILE | {path}")
            return pd.DataFrame()

        df = df.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))

        return df

    except Exception as e:
        log_error(f"READ/SCHEMA ERROR | {path} | {e}")
        raise


def normalize_date(val):
    return str(val).strip().replace("-", "_")


def clean_game_id(series):
    return series.fillna("").astype(str).str.strip().str.split(".").str[0]


def enforce_output_cols(df):
    validate_no_duplicate_columns_df(df, "pre-output dataframe")

    for col in OUTPUT_COLS:
        if col not in df.columns:
            df[col] = ""

    out = df[OUTPUT_COLS].copy()
    validate_no_duplicate_columns_df(out, "final output dataframe")

    return out


def enforce_unmatched_output_cols(df):
    validate_no_duplicate_columns_df(df, "pre-unmatched dataframe")

    for col in UNMATCHED_OUTPUT_COLS:
        if col not in df.columns:
            df[col] = ""

    out = df[UNMATCHED_OUTPUT_COLS].copy()
    validate_no_duplicate_columns_df(out, "unmatched output dataframe")

    return out


def write_csv_checked(df, path, label):
    validate_no_duplicate_columns_df(df, label)
    df.to_csv(path, index=False)


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
            f"game_id={row.get('game_id', '')} "
            f"market_type={row.get('market_type', '')} "
            f"bet_side={row.get('bet_side', '')} "
            f"final_away_score={row.get('final_away_score', '')} "
            f"final_home_score={row.get('final_home_score', '')} | {e}"
        )

    return "Unknown"


def validate_no_blank_game_ids(df, label):
    blank_count = int((df["game_id"].fillna("").astype(str).str.strip() == "").sum())

    if blank_count > 0:
        raise ValueError(f"{label} has blank game_id rows: {blank_count}")


def validate_score_game_id_uniqueness(all_scores):
    duplicated_mask = all_scores["game_id"].duplicated(keep=False)
    duplicates = all_scores.loc[duplicated_mask].copy()

    if duplicates.empty:
        return

    sample_cols = [
        col for col in [
            "game_id",
            "game_date",
            "game_time",
            "away_team",
            "home_team",
            "final_away_score",
            "final_home_score",
        ]
        if col in duplicates.columns
    ]

    sample = duplicates[sample_cols].head(25).to_dict("records")

    raise ValueError(
        f"final-score game_id uniqueness check failed. "
        f"duplicate_rows={len(duplicates)} sample={sample}"
    )


def selected_bet_key_columns(df):
    preferred = [
        "game_id",
        "market_type",
        "bet_side",
        "line",
        "take_bet",
        "dk_odds_american",
    ]

    return [col for col in preferred if col in df.columns]


def write_unmatched_selected_bets(unmatched):
    if unmatched.empty:
        return None

    if "game_date" in unmatched.columns and unmatched["game_date"].nunique(dropna=True) == 1:
        date_str = normalize_date(unmatched["game_date"].iloc[0])
        out_path = UNMATCHED_DIR / f"{date_str}_MLB_unmatched_selected_bets.csv"
    else:
        out_path = UNMATCHED_DIR / "MLB_unmatched_selected_bets.csv"

    out = enforce_unmatched_output_cols(unmatched.copy())
    write_csv_checked(out, out_path, "unmatched selected bets output")
    log_error(f"UNMATCHED SELECTED BETS WRITTEN | rows={len(out)} | OUT={out_path}")
    log_summary(f"UNMATCHED SELECTED BETS WRITTEN | rows={len(out)} | OUT={out_path}")

    return out_path


def grade_league():
    for old in OUTPUT_DIR.glob("*.csv"):
        old.unlink()

    for old in DAILY_DIR.glob("*.csv"):
        old.unlink()

    for old in UNMATCHED_DIR.glob("*.csv"):
        old.unlink()

    select_files = sorted(SELECT_DIR.glob("*MLB*.csv"))

    if not select_files:
        raise RuntimeError(f"NO SELECT FILES FOUND IN {SELECT_DIR}")

    parts = []

    for f in select_files:
        df = safe_read(f, SELECT_REQUIRED_COLS, f"selected bets input {f.name}")

        if not df.empty:
            df["source_select_file"] = f.name
            df["game_date"] = df["game_date"].apply(normalize_date)
            df["game_id"] = clean_game_id(df["game_id"])
            parts.append(df)

    if not parts:
        raise RuntimeError("ALL SELECT FILES EMPTY OR UNREADABLE")

    all_bets = pd.concat(parts, ignore_index=True)
    all_bets["selected_row_id"] = range(1, len(all_bets) + 1)

    validate_no_duplicate_columns_df(all_bets, "combined selected bets")
    validate_no_blank_game_ids(all_bets, "combined selected bets")

    score_files = sorted(SCORE_DIR.glob("*_final_scores_MLB.csv"))

    if not score_files:
        raise RuntimeError(f"NO SCORE FILES FOUND IN {SCORE_DIR}")

    score_parts = []

    for sf in score_files:
        df = safe_read(sf, SCORE_REQUIRED_COLS, f"final scores input {sf.name}")

        if not df.empty:
            df["source_score_file"] = sf.name
            df["game_date"] = df["game_date"].apply(normalize_date)
            df["game_id"] = clean_game_id(df["game_id"])
            score_parts.append(df)

    if not score_parts:
        raise RuntimeError("ALL SCORE FILES EMPTY OR UNREADABLE")

    all_scores = pd.concat(score_parts, ignore_index=True)

    validate_no_duplicate_columns_df(all_scores, "combined final scores")
    validate_no_blank_game_ids(all_scores, "combined final scores")
    validate_score_game_id_uniqueness(all_scores)

    selected_count = len(all_bets)

    log_summary(f"BET cols: {list(all_bets.columns)}")
    log_summary(f"SCORE cols: {list(all_scores.columns)}")
    log_summary(f"BET game_id sample: {all_bets['game_id'].head(3).tolist()}")
    log_summary(f"SCORE game_id sample: {all_scores['game_id'].head(3).tolist()}")
    log_summary(f"SELECTED BET ROWS: {selected_count}")
    log_summary(f"FINAL SCORE ROWS: {len(all_scores)}")

    merged = pd.merge(
        all_bets,
        all_scores,
        on="game_id",
        how="left",
        suffixes=("_bet", "_score"),
        indicator=True,
    )

    validate_no_duplicate_columns_df(merged, "merged selected bets to final scores")

    unmatched = merged[merged["_merge"] != "both"].copy()
    matched = merged[merged["_merge"] == "both"].copy()

    unmatched_count = len(unmatched)
    matched_count = len(matched)

    log_summary(
        f"MERGED ON game_id | selected_rows={selected_count} "
        f"matched_rows={matched_count} unmatched_rows={unmatched_count}"
    )

    if unmatched_count > 0:
        unmatched["unmatched_reason"] = "missing_final_score"
        unmatched_path = write_unmatched_selected_bets(unmatched)

        msg = (
            f"SELECTED BETS MERGED TO FEWER FINAL-SCORE ROWS THAN SELECTED. "
            f"selected_rows={selected_count} matched_rows={matched_count} "
            f"unmatched_rows={unmatched_count} unmatched_file={unmatched_path}"
        )

        log_error(msg)

        if FAIL_ON_UNMATCHED_SELECTED_BETS:
            raise RuntimeError(msg)

    if matched.empty:
        raise RuntimeError("MERGE EMPTY AFTER MATCH FILTER")

    matched = matched.drop(columns=["_merge"], errors="ignore")

    log_summary(f"MERGED cols: {list(matched.columns)}")
    log_summary(f"MERGED MATCHED ROWS: {len(matched)}")

    score_fields = {
        "game_date",
        "game_time",
        "home_team",
        "away_team",
        "sport",
        "league",
        "final_home_score",
        "final_away_score",
        "final_total",
        "home_run_line",
        "away_run_line",
        "total",
    }

    for base in score_fields:
        score_col = f"{base}_score"
        bet_col = f"{base}_bet"

        if score_col in matched.columns:
            matched[base] = matched[score_col]
        elif base not in matched.columns and bet_col in matched.columns:
            matched[base] = matched[bet_col]

    for col in list(matched.columns):
        if col.endswith("_bet"):
            base = col[:-4]

            if base not in matched.columns:
                matched[base] = matched[col]

    protected = {
        "final_home_score",
        "final_away_score",
        "final_total",
        "home_run_line",
        "away_run_line",
    }

    to_drop = []

    for col in matched.columns:
        if col in protected:
            continue

        if col.endswith("_bet") or col.endswith("_score"):
            to_drop.append(col)

    matched = matched.drop(columns=to_drop, errors="ignore")
    validate_no_duplicate_columns_df(matched, "post-resolve merged dataframe")

    log_summary(f"POST-RESOLVE cols: {list(matched.columns)}")
    log_summary(
        f"final_away_score sample: "
        f"{matched['final_away_score'].head(3).tolist() if 'final_away_score' in matched.columns else 'MISSING'}"
    )
    log_summary(
        f"final_home_score sample: "
        f"{matched['final_home_score'].head(3).tolist() if 'final_home_score' in matched.columns else 'MISSING'}"
    )
    log_summary(
        f"market_type sample: "
        f"{matched['market_type'].head(3).tolist() if 'market_type' in matched.columns else 'MISSING'}"
    )

    matched["bet_result"] = matched.apply(determine_outcome, axis=1)

    key_cols = selected_bet_key_columns(matched)

    if key_cols:
        before_dedupe = len(matched)
        matched = matched.drop_duplicates(subset=key_cols, keep="last")
        removed = before_dedupe - len(matched)

        if removed > 0:
            log_error(
                f"DUPLICATE GRADED BET ROWS REMOVED | removed={removed} "
                f"key_cols={key_cols}"
            )
            log_summary(
                f"DUPLICATE GRADED BET ROWS REMOVED | removed={removed} "
                f"key_cols={key_cols}"
            )

    final = enforce_output_cols(matched)

    graded_count = len(final)

    if graded_count != matched_count:
        log_error(
            f"GRADED ROW COUNT CHANGED AFTER DEDUPE/OUTPUT ENFORCEMENT | "
            f"matched_rows={matched_count} graded_rows={graded_count}"
        )

    master_path = OUTPUT_DIR / "MLB_final.csv"
    write_csv_checked(final, master_path, "MLB master graded output")
    result_counts = final["bet_result"].value_counts().to_dict()

    log_summary(
        f"MLB MASTER BUILT | ROWS={len(final)} | RESULTS={result_counts} | OUT={master_path}"
    )

    for date_val, group in matched.groupby("game_date"):
        date_str = normalize_date(date_val)
        daily_df = enforce_output_cols(group.copy())
        daily_path = DAILY_DIR / f"{date_str}_MLB_final.csv"

        write_csv_checked(daily_df, daily_path, f"MLB daily graded output {date_str}")

        daily_result_counts = daily_df["bet_result"].value_counts().to_dict()

        log_summary(
            f"MLB DAILY | DATE={date_str} | ROWS={len(daily_df)} "
            f"| RESULTS={daily_result_counts} | OUT={daily_path}"
        )

    log_summary(
        f"GRADING COMPLETE | selected_rows={selected_count} "
        f"matched_rows={matched_count} graded_rows={graded_count} "
        f"unmatched_rows={unmatched_count}"
    )


def main():
    reset_logs()
    log_summary("START 01_mlb_results_grade.py")
<<<<<<< HEAD
    clear_output_files()
    grade_league()
    log_summary("END 01_mlb_results_grade.py")
    print("MLB grading complete.")
=======

    try:
        grade_league()
        log_summary("STATUS: SUCCESS")
        log_summary("END 01_mlb_results_grade.py")
        print("MLB grading complete. Status: SUCCESS")

    except Exception as e:
        log_error(f"FATAL ERROR | {e}\n{traceback.format_exc()}")
        log_summary("STATUS: FAILED")
        log_summary("END 01_mlb_results_grade.py")
        print(f"MLB grading failed. {e}")
        sys.exit(1)

>>>>>>> 67f12db62f4d2ba562c5ffa0eebb8972e2235186

if __name__ == "__main__":
    main()
