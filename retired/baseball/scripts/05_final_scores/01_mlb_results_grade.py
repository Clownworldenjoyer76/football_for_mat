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
AUDIT_DIR = Path("docs/win/baseball/05_final_scores/results/audit")
ERROR_DIR = Path("docs/win/baseball/errors/05_final_scores")

ERROR_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DAILY_DIR.mkdir(parents=True, exist_ok=True)
UNMATCHED_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

GRADE_ERROR_LOG = ERROR_DIR / "mlb_results_grade_errors.txt"
GRADE_SUMMARY_LOG = ERROR_DIR / "mlb_results_grade_summary.txt"

UNMATCHED_SELECTED_FILE = UNMATCHED_DIR / "MLB_unmatched_selected_bets.csv"
NOT_FINAL_SELECTED_FILE = UNMATCHED_DIR / "MLB_not_final_selected_bets.csv"
POSTPONED_CANCELED_FILE = UNMATCHED_DIR / "MLB_postponed_canceled_games.csv"
BLANK_SCORE_GAME_ID_FILE = UNMATCHED_DIR / "blank_final_score_game_ids_MLB.csv"

RECONCILIATION_AUDIT_FILE = AUDIT_DIR / "selected_vs_graded_reconciliation.csv"
DUPLICATE_AUDIT_FILE = AUDIT_DIR / "grading_duplicate_audit.csv"
VALIDATION_AUDIT_FILE = AUDIT_DIR / "graded_output_validation_audit.csv"
RESULT_COUNTS_FILE = AUDIT_DIR / "grading_result_counts.csv"
SPOT_CHECK_FILE = AUDIT_DIR / "grading_spot_check.csv"

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
    "gamePk",
    "gameNumber",
    "game_status",
    "final_scores_generated_at",
    "final_home_score",
    "final_away_score",
    "final_total",
    "home_run_line",
    "away_run_line",
    "total",
    "bet_result",
]

UNMATCHED_COLS = [
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
    "source_file",
]

REQUIRED_SELECTED_COLUMNS = [
    "game_id",
    "game_date",
    "market_type",
    "bet_side",
    "line",
    "dk_odds_american",
]

REQUIRED_SCORE_COLUMNS = [
    "game_id",
    "game_date",
    "final_home_score",
    "final_away_score",
]

SELECTED_DUP_KEY = ["game_id", "market_type", "bet_side", "line"]
SCORE_DUP_KEY = ["game_id"]
VALID_RESULTS = {"Win", "Loss", "Push"}


def now_utc():
    return datetime.now(UTC).isoformat()


def reset_logs():
    GRADE_ERROR_LOG.write_text("", encoding="utf-8")
    GRADE_SUMMARY_LOG.write_text("", encoding="utf-8")


def log_error(msg):
    with open(GRADE_ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{now_utc()}] {msg}\n")


def log_summary(msg):
    with open(GRADE_SUMMARY_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{now_utc()}] {msg}\n")


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


def validate_no_duplicate_header(path, label):
    header = read_header_columns(path)
    duplicates = duplicate_columns(header)
    if duplicates:
        raise ValueError(f"{label} has duplicate header columns: {duplicates}")


def validate_no_duplicate_columns(df, label):
    duplicates = duplicate_columns(list(df.columns))
    if duplicates:
        raise ValueError(f"{label} has duplicate columns: {duplicates}")


def validate_required_columns(df, required_columns, label):
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def write_csv_checked(df, path, label):
    validate_no_duplicate_columns(df, label)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def safe_read(path, required_columns=None, label=None):
    path = Path(path)
    read_label = label or str(path)
    try:
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


def blank_mask(series):
    return series.fillna("").astype(str).str.strip() == ""


def normalize_game_status(value):
    raw = str(value or "").strip().lower()
    if raw in {"final", "game over", "completed", "complete"}:
        return "final"
    if raw in {"postponed", "ppd"}:
        return "postponed"
    if raw in {"canceled", "cancelled"}:
        return "canceled"
    if raw == "suspended":
        return "suspended"
    if raw == "delayed":
        return "delayed"
    if raw in {"in progress", "in_progress", "live", "active"}:
        return "in_progress"
    if raw in {"scheduled", "pre-game", "pregame", "preview"}:
        return "scheduled"
    if raw in {"", "nan", "none"}:
        return "unknown"
    return raw.replace(" ", "_")


def make_empty_csv(path, columns, label):
    write_csv_checked(pd.DataFrame(columns=columns), path, label)


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


def normalize_unmatched_selected_rows(unmatched):
    if unmatched.empty:
        return unmatched
    out = unmatched.copy()
    selected_preferred = ["sport", "league", "game_date", "game_time", "home_team", "away_team", "source_file", "take_bet"]
    for base in selected_preferred:
        bet_col = f"{base}_bet"
        score_col = f"{base}_score"
        if bet_col in out.columns:
            out[base] = out[bet_col]
        elif base not in out.columns and score_col in out.columns:
            out[base] = out[score_col]
    if "game_date" in out.columns:
        out["game_date"] = out["game_date"].apply(normalize_date)
    return out


def write_unmatched(unmatched):
    if unmatched.empty:
        make_empty_csv(UNMATCHED_SELECTED_FILE, UNMATCHED_COLS, "empty unmatched selected bets output")
        return None
    unmatched = normalize_unmatched_selected_rows(unmatched.copy())
    unmatched = enforce_unmatched_cols(unmatched)
    write_csv_checked(unmatched, UNMATCHED_SELECTED_FILE, "unmatched selected bets output")
    return UNMATCHED_SELECTED_FILE


def duplicate_audit_row(scope, group, action_taken, failure_reason, source_files):
    first = group.iloc[0] if not group.empty else {}
    return {
        "duplicate_scope": scope,
        "game_date": first.get("game_date", ""),
        "game_id": first.get("game_id", ""),
        "market_type": first.get("market_type", ""),
        "bet_side": first.get("bet_side", ""),
        "line": first.get("line", ""),
        "duplicate_count": len(group),
        "identical_duplicate": str(group.drop(columns=["selected_row_id"], errors="ignore").drop_duplicates().shape[0] == 1),
        "action_taken": action_taken,
        "failure_reason": failure_reason,
        "source_files": source_files,
    }


def validate_and_collapse_duplicates(df, key_cols, scope, compare_cols=None):
    if df.empty:
        return df, [], True
    missing_key_cols = [col for col in key_cols if col not in df.columns]
    if missing_key_cols:
        log_error(f"DUPLICATE VALIDATION SKIPPED | scope={scope} missing_key_cols={missing_key_cols}")
        return df, [], False
    audit_rows = []
    cleaned_parts = []
    ok = True
    duplicate_mask = df.duplicated(subset=key_cols, keep=False)
    duplicates = df[duplicate_mask].copy()
    non_duplicates = df[~duplicate_mask].copy()
    if not non_duplicates.empty:
        cleaned_parts.append(non_duplicates)
    if duplicates.empty:
        return df.copy(), audit_rows, ok
    for _key, group in duplicates.groupby(key_cols, dropna=False):
        source_files = ",".join(sorted(set(group.get("source_file", pd.Series(dtype=str)).fillna("").astype(str))))
        available_compare_cols = [col for col in (compare_cols or list(group.columns)) if col in group.columns]
        comparable = group[available_compare_cols].fillna("").astype(str).copy()
        identical = comparable.drop_duplicates().shape[0] == 1
        if identical:
            cleaned_parts.append(group.head(1).copy())
            audit_rows.append(duplicate_audit_row(scope, group, "collapsed_identical_duplicate", "", source_files))
        else:
            ok = False
            audit_rows.append(duplicate_audit_row(scope, group, "hard_fail", "conflicting_duplicate_rows", source_files))
    cleaned = pd.concat(cleaned_parts, ignore_index=True) if cleaned_parts else pd.DataFrame(columns=df.columns)
    return cleaned, audit_rows, ok


def write_duplicate_audit(rows):
    columns = [
        "duplicate_scope", "game_date", "game_id", "market_type", "bet_side", "line",
        "duplicate_count", "identical_duplicate", "action_taken", "failure_reason", "source_files",
    ]
    audit_df = pd.DataFrame(rows, columns=columns)
    write_csv_checked(audit_df, DUPLICATE_AUDIT_FILE, "grading duplicate audit")


def audit_and_drop_blank_score_game_ids(all_scores):
    blank = blank_mask(all_scores["game_id"])
    blank_scores = all_scores[blank].copy()
    clean_scores = all_scores[~blank].copy()
    if not blank_scores.empty:
        write_csv_checked(blank_scores.sort_values(["game_date"], na_position="last"), BLANK_SCORE_GAME_ID_FILE, "blank final-score game_id audit")
        log_summary(f"FINAL SCORE BLANK GAME_ID ROWS DROPPED | rows={len(blank_scores)} | audit={BLANK_SCORE_GAME_ID_FILE}")
    else:
        make_empty_csv(BLANK_SCORE_GAME_ID_FILE, list(all_scores.columns), "empty blank final-score game_id audit")
    return clean_scores, len(blank_scores)


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
        log_error(f"DETERMINE OUTCOME ERROR | game_id={row.get('game_id', '')} market_type={row.get('market_type', '')} bet_side={row.get('bet_side', '')} | {e}")
    return ""


def build_calculation(row):
    try:
        market = str(row.get("market_type", "")).strip().lower()
        side = str(row.get("bet_side", "")).strip().lower()
        away = float(row.get("final_away_score", ""))
        home = float(row.get("final_home_score", ""))
        result = str(row.get("bet_result", "")).strip()
        line_raw = row.get("line", "")
        if market == "moneyline":
            if side == "away":
                return f"moneyline away: away_score={away:g}, home_score={home:g} => {result.lower()}"
            if side == "home":
                return f"moneyline home: home_score={home:g}, away_score={away:g} => {result.lower()}"
        if market == "run_line":
            line = float(line_raw)
            if side == "home":
                adjusted = home + line
                return f"run_line home {line:g}: home_score={home:g}, away_score={away:g}, adjusted_home_score={adjusted:g} => {result.lower()}"
            if side == "away":
                adjusted = away + line
                return f"run_line away {line:g}: away_score={away:g}, home_score={home:g}, adjusted_away_score={adjusted:g} => {result.lower()}"
        if market == "total":
            line = float(line_raw)
            total = away + home
            return f"total {side} {line:g}: final_total={total:g} vs line={line:g} => {result.lower()}"
    except Exception as e:
        return f"calculation_error: {e}"
    return ""


def resolve_merge_columns(merged):
    score_fields = {
        "game_date", "game_time", "home_team", "away_team", "sport", "league",
        "final_home_score", "final_away_score", "final_total", "home_run_line", "away_run_line", "total",
        "gamePk", "gameNumber", "game_status", "final_scores_generated_at",
    }
    selected_fields = {"sport", "league", "game_date", "game_time", "home_team", "away_team", "source_file"}
    for base in score_fields:
        score_col = f"{base}_score"
        bet_col = f"{base}_bet"
        if score_col in merged.columns:
            merged[base] = merged[score_col]
        elif base not in merged.columns and bet_col in merged.columns:
            merged[base] = merged[bet_col]
    for base in selected_fields:
        bet_col = f"{base}_bet"
        score_col = f"{base}_score"
        if base not in merged.columns and bet_col in merged.columns:
            merged[base] = merged[bet_col]
        elif base not in merged.columns and score_col in merged.columns:
            merged[base] = merged[score_col]
    to_drop = []
    for col in merged.columns:
        if col == "take_bet":
            continue
        if col.endswith("_bet") or col.endswith("_score"):
            base = col[:-4] if col.endswith("_bet") else col[:-6]
            if base in selected_fields or base in score_fields:
                to_drop.append(col)
    merged = merged.drop(columns=to_drop, errors="ignore")
    validate_no_duplicate_columns(merged, "post-resolve graded rows")
    return merged


def load_selected_bets():
    select_files = sorted(SELECT_DIR.glob("*MLB*.csv"))
    if not select_files:
        log_error(f"NO SELECT FILES FOUND IN {SELECT_DIR}")
        return pd.DataFrame(), []
    parts = []
    duplicate_audit_rows = []
    for path in select_files:
        df = safe_read(path, REQUIRED_SELECTED_COLUMNS, f"selected file {path.name}")
        if df.empty:
            continue
        df["source_file"] = path.name
        df["game_date"] = df["game_date"].apply(normalize_date)
        df["game_id"] = clean_game_id(df.get("game_id", pd.Series(dtype=str)))
        df, daily_dup_rows, daily_ok = validate_and_collapse_duplicates(df, SELECTED_DUP_KEY, "daily_selected_bet_key", compare_cols=[c for c in df.columns if c != "selected_row_id"])
        duplicate_audit_rows.extend(daily_dup_rows)
        if not daily_ok:
            return pd.DataFrame(), duplicate_audit_rows
        parts.append(df)
    if not parts:
        log_error("ALL SELECT FILES EMPTY, UNREADABLE, OR SCHEMA-INVALID")
        return pd.DataFrame(), duplicate_audit_rows
    all_bets = pd.concat(parts, ignore_index=True)
    validate_no_duplicate_columns(all_bets, "combined selected bets")
    all_bets["game_id"] = clean_game_id(all_bets.get("game_id", pd.Series(dtype=str)))
    all_bets["selected_row_id"] = range(len(all_bets))
    compare_cols = ["game_id", "game_date", "market_type", "bet_side", "line", "take_bet", "dk_odds_american", "model_prob", "ev", "kelly", "prob_for_ev", "prob_for_kelly"]
    all_bets, combined_dup_rows, combined_ok = validate_and_collapse_duplicates(all_bets, SELECTED_DUP_KEY, "combined_selected_bet_key", compare_cols=compare_cols)
    duplicate_audit_rows.extend(combined_dup_rows)
    if not combined_ok:
        return pd.DataFrame(), duplicate_audit_rows
    all_bets["selected_row_id"] = range(len(all_bets))
    return all_bets, duplicate_audit_rows


def load_final_scores():
    score_files = sorted(SCORE_DIR.glob("*_final_scores_MLB.csv"))
    if not score_files:
        log_error(f"NO SCORE FILES FOUND IN {SCORE_DIR}")
        return pd.DataFrame(), []
    parts = []
    duplicate_audit_rows = []
    for path in score_files:
        df = safe_read(path, REQUIRED_SCORE_COLUMNS, f"score file {path.name}")
        if df.empty:
            continue
        df["source_file"] = path.name
        df["game_date"] = df["game_date"].apply(normalize_date)
        parts.append(df)
    if not parts:
        log_error("ALL SCORE FILES EMPTY, UNREADABLE, OR SCHEMA-INVALID")
        return pd.DataFrame(), duplicate_audit_rows
    all_scores = pd.concat(parts, ignore_index=True)
    validate_no_duplicate_columns(all_scores, "combined final scores")
    all_scores["game_id"] = clean_game_id(all_scores.get("game_id", pd.Series(dtype=str)))
    if "game_status" not in all_scores.columns:
        all_scores["game_status"] = "unknown"
    all_scores["game_status"] = all_scores["game_status"].apply(normalize_game_status)
    all_scores, blank_score_count = audit_and_drop_blank_score_game_ids(all_scores)
    score_compare_cols = ["game_id", "game_date", "game_time", "home_team", "away_team", "final_home_score", "final_away_score", "final_total", "gamePk", "gameNumber", "game_status"]
    all_scores, score_dup_rows, score_ok = validate_and_collapse_duplicates(all_scores, SCORE_DUP_KEY, "final_score_game_id", compare_cols=score_compare_cols)
    duplicate_audit_rows.extend(score_dup_rows)
    if not score_ok:
        log_error(f"FINAL SCORE CONFLICTING DUPLICATES | audit={DUPLICATE_AUDIT_FILE}")
        return pd.DataFrame(), duplicate_audit_rows
    log_summary(f"SCORE BLANK GAME_ID ROWS DROPPED: {blank_score_count}")
    return all_scores, duplicate_audit_rows


def build_non_final_reports(merged_both):
    columns = ["game_date", "game_id", "gamePk", "gameNumber", "away_team", "home_team", "market_type", "bet_side", "line", "game_status", "unmatched_reason"]
    pc_cols = ["game_date", "game_id", "gamePk", "gameNumber", "away_team", "home_team", "game_status", "selected_rows_affected"]
    if merged_both.empty or "game_status" not in merged_both.columns:
        make_empty_csv(NOT_FINAL_SELECTED_FILE, columns, "empty not-final selected bets output")
        make_empty_csv(POSTPONED_CANCELED_FILE, pc_cols, "empty postponed/canceled games output")
        return pd.DataFrame(columns=columns)
    non_final = merged_both[merged_both["game_status"].apply(normalize_game_status) != "final"].copy()
    if non_final.empty:
        make_empty_csv(NOT_FINAL_SELECTED_FILE, columns, "empty not-final selected bets output")
        make_empty_csv(POSTPONED_CANCELED_FILE, pc_cols, "empty postponed/canceled games output")
        return pd.DataFrame(columns=columns)
    non_final = resolve_merge_columns(non_final.drop(columns=["_merge"], errors="ignore"))
    def reason_for_status(status):
        status = normalize_game_status(status)
        if status == "postponed":
            return "postponed"
        if status == "canceled":
            return "canceled"
        if status == "unknown":
            return "unknown_game_status"
        return "game_not_final"
    non_final["unmatched_reason"] = non_final["game_status"].apply(reason_for_status)
    not_final_out = non_final.copy()
    for col in columns:
        if col not in not_final_out.columns:
            not_final_out[col] = ""
    write_csv_checked(not_final_out[columns], NOT_FINAL_SELECTED_FILE, "not-final selected bets output")
    pc = non_final[non_final["game_status"].isin(["postponed", "canceled"])].copy()
    if pc.empty:
        make_empty_csv(POSTPONED_CANCELED_FILE, pc_cols, "empty postponed/canceled games output")
    else:
        grouped = pc.groupby(["game_date", "game_id", "gamePk", "gameNumber", "away_team", "home_team", "game_status"], dropna=False).size().reset_index(name="selected_rows_affected")
        write_csv_checked(grouped[pc_cols], POSTPONED_CANCELED_FILE, "postponed/canceled games output")
    return non_final


def write_reconciliation(all_bets, final, unmatched, non_final):
    all_dates = sorted(set(list(all_bets.get("game_date", pd.Series(dtype=str)).dropna().astype(str)) + list(final.get("game_date", pd.Series(dtype=str)).dropna().astype(str)) + list(unmatched.get("game_date", pd.Series(dtype=str)).dropna().astype(str)) + list(non_final.get("game_date", pd.Series(dtype=str)).dropna().astype(str))))
    rows = []
    for date in all_dates:
        selected_rows = int((all_bets["game_date"].astype(str) == date).sum()) if not all_bets.empty else 0
        graded_rows = int((final["game_date"].astype(str) == date).sum()) if not final.empty else 0
        if not unmatched.empty and "game_date" in unmatched.columns:
            unmatched_date = unmatched[unmatched["game_date"].astype(str) == date].copy()
        else:
            unmatched_date = pd.DataFrame()
        def reason_count(reason):
            if unmatched_date.empty or "unmatched_reason" not in unmatched_date.columns:
                return 0
            return int((unmatched_date["unmatched_reason"].astype(str) == reason).sum())
        unmatched_rows = len(unmatched_date)
        allowed_unmatched = reason_count("missing_final_score") + reason_count("future_game") + reason_count("postponed") + reason_count("canceled") + reason_count("game_not_final") + reason_count("unknown_game_status")
        status = "ok" if unmatched_rows == 0 or unmatched_rows == allowed_unmatched else "review"
        rows.append({
            "game_date": date,
            "selected_rows": selected_rows,
            "graded_rows": graded_rows,
            "unmatched_rows": unmatched_rows,
            "missing_final_score_rows": reason_count("missing_final_score"),
            "missing_game_id_rows": reason_count("missing_game_id"),
            "future_game_rows": reason_count("future_game"),
            "postponed_rows": reason_count("postponed"),
            "canceled_rows": reason_count("canceled"),
            "game_not_final_rows": reason_count("game_not_final"),
            "unknown_game_status_rows": reason_count("unknown_game_status"),
            "other_unmatched_rows": reason_count("other"),
            "status": status,
        })
    recon = pd.DataFrame(rows)
    write_csv_checked(recon, RECONCILIATION_AUDIT_FILE, "selected vs graded reconciliation audit")
    return recon


def write_result_counts(final):
    columns = ["market_type", "wins", "losses", "pushes", "blank_results", "total_rows"]
    if final.empty:
        make_empty_csv(RESULT_COUNTS_FILE, columns, "empty grading result counts")
        return pd.DataFrame(columns=columns)
    rows = []
    for market, group in final.groupby("market_type", dropna=False):
        results = group["bet_result"].fillna("").astype(str).str.strip()
        rows.append({"market_type": market, "wins": int((results == "Win").sum()), "losses": int((results == "Loss").sum()), "pushes": int((results == "Push").sum()), "blank_results": int((results == "").sum()), "total_rows": len(group)})
    out = pd.DataFrame(rows, columns=columns)
    write_csv_checked(out, RESULT_COUNTS_FILE, "grading result counts")
    return out


def write_spot_check(final):
    columns = ["game_date", "game_id", "market_type", "bet_side", "line", "final_away_score", "final_home_score", "final_total", "result", "calculation"]
    if final.empty:
        make_empty_csv(SPOT_CHECK_FILE, columns, "empty grading spot check")
        return
    out = final.copy()
    out["result"] = out["bet_result"]
    out["calculation"] = out.apply(build_calculation, axis=1)
    for col in columns:
        if col not in out.columns:
            out[col] = ""
    write_csv_checked(out[columns], SPOT_CHECK_FILE, "grading spot check")


def validate_graded_output(final):
    required_nonblank = ["game_id", "market_type", "bet_side", "dk_odds_american", "bet_result"]
    audit_rows = []
    if "take_bet" not in final.columns:
        audit_rows.append({"validation": "required_column_exists", "column": "take_bet", "bad_rows": "", "status": "fail", "notes": "take_bet column missing after suffix cleanup"})
    else:
        audit_rows.append({"validation": "required_column_exists", "column": "take_bet", "bad_rows": 0, "status": "ok", "notes": "take_bet column preserved"})
    for col in required_nonblank:
        if col not in final.columns:
            audit_rows.append({"validation": "required_nonblank", "column": col, "bad_rows": "", "status": "fail", "notes": "column missing"})
            continue
        bad = final[blank_mask(final[col])].copy()
        audit_rows.append({"validation": "required_nonblank", "column": col, "bad_rows": len(bad), "status": "fail" if len(bad) else "ok", "notes": "blank values are not allowed"})
    invalid_results = final[~final["bet_result"].fillna("").astype(str).str.strip().isin(VALID_RESULTS)].copy() if "bet_result" in final.columns else final.copy()
    audit_rows.append({"validation": "valid_bet_result", "column": "bet_result", "bad_rows": len(invalid_results), "status": "fail" if len(invalid_results) else "ok", "notes": "allowed values are Win, Loss, Push"})
    audit = pd.DataFrame(audit_rows)
    write_csv_checked(audit, VALIDATION_AUDIT_FILE, "graded output validation audit")
    failures = audit[audit["status"] == "fail"].copy()
    if not failures.empty:
        log_error(f"GRADED OUTPUT VALIDATION FAILED | audit={VALIDATION_AUDIT_FILE}")
        for _, row in failures.iterrows():
            log_error(f"VALIDATION FAILURE | validation={row.get('validation')} column={row.get('column')} bad_rows={row.get('bad_rows')} notes={row.get('notes')}")
        return False
    return True


def grade_league():
    duplicate_audit_rows = []
    all_bets, selected_duplicate_rows = load_selected_bets()
    duplicate_audit_rows.extend(selected_duplicate_rows)
    if all_bets.empty:
        write_duplicate_audit(duplicate_audit_rows)
        return False
    all_scores, score_duplicate_rows = load_final_scores()
    duplicate_audit_rows.extend(score_duplicate_rows)
    if all_scores.empty:
        write_duplicate_audit(duplicate_audit_rows)
        return False
    write_duplicate_audit(duplicate_audit_rows)
    selected_blank_id = all_bets[blank_mask(all_bets["game_id"])].copy()
    selected_valid = all_bets[~blank_mask(all_bets["game_id"])].copy()
    if not selected_blank_id.empty:
        selected_blank_id["unmatched_reason"] = "missing_game_id"
    log_summary(f"BET cols: {list(all_bets.columns)}")
    log_summary(f"SCORE cols: {list(all_scores.columns)}")
    log_summary(f"SELECTED ROWS: {len(all_bets)}")
    log_summary(f"SELECTED BLANK GAME_ID ROWS: {len(selected_blank_id)}")
    log_summary(f"SCORE ROWS AFTER BLANK GAME_ID DROP: {len(all_scores)}")
    log_summary(f"BET game_id sample: {all_bets['game_id'].head(3).tolist()}")
    log_summary(f"SCORE game_id sample: {all_scores['game_id'].head(3).tolist()}")
    merged_all = pd.merge(selected_valid, all_scores, on="game_id", how="left", suffixes=("_bet", "_score"), indicator=True)
    matched_all = merged_all[merged_all["_merge"] == "both"].copy()
    non_final = build_non_final_reports(matched_all)
    matched_final = matched_all[matched_all["game_status"].apply(normalize_game_status) == "final"].copy() if "game_status" in matched_all.columns else matched_all.copy()
    missing_scores = merged_all[merged_all["_merge"] == "left_only"].copy()
    if not missing_scores.empty:
        missing_scores["unmatched_reason"] = "missing_final_score"
    unmatched_parts = []
    if not selected_blank_id.empty:
        unmatched_parts.append(selected_blank_id)
    if not missing_scores.empty:
        unmatched_parts.append(missing_scores)
    if not non_final.empty:
        unmatched_parts.append(non_final)
    unmatched = pd.concat(unmatched_parts, ignore_index=True) if unmatched_parts else pd.DataFrame()
    unmatched_path = write_unmatched(unmatched)
    merged = matched_final.drop(columns=["_merge"], errors="ignore").copy()
    if merged.empty:
        log_error("MERGE EMPTY")
        if unmatched_path:
            log_summary(f"UNMATCHED SELECTED BETS WRITTEN | {unmatched_path}")
        return False
    log_summary(f"MERGED cols: {list(merged.columns)}")
    log_summary(f"MERGED ON game_id | rows={len(merged)}")
    log_summary(f"UNMATCHED SELECTED ROWS: {len(unmatched)}")
    if unmatched_path:
        log_summary(f"UNMATCHED SELECTED BETS WRITTEN | rows={len(unmatched)} | out={unmatched_path}")
    merged = resolve_merge_columns(merged)
    log_summary(f"POST-RESOLVE cols: {list(merged.columns)}")
    log_summary(f"final_away_score sample: {merged['final_away_score'].head(3).tolist() if 'final_away_score' in merged.columns else 'MISSING'}")
    log_summary(f"final_home_score sample: {merged['final_home_score'].head(3).tolist() if 'final_home_score' in merged.columns else 'MISSING'}")
    log_summary(f"market_type sample: {merged['market_type'].head(3).tolist() if 'market_type' in merged.columns else 'MISSING'}")
    log_summary(f"take_bet present after resolve: {'take_bet' in merged.columns}")
    merged["bet_result"] = merged.apply(determine_outcome, axis=1)
    final = enforce_output_cols(merged)
    if not validate_graded_output(final):
        return False
    result_counts = write_result_counts(final)
    write_spot_check(final)
    master_path = OUTPUT_DIR / "MLB_final.csv"
    write_csv_checked(final, master_path, "MLB graded master output")
    selected_count = len(all_bets)
    graded_count = len(final)
    unmatched_count = len(unmatched)
    log_summary(f"MLB MASTER BUILT | ROWS={graded_count} | OUT={master_path}")
    log_summary(f"SELECTED VS GRADED | selected={selected_count} graded={graded_count} unmatched={unmatched_count}")
    if selected_count > graded_count:
        reason_counts = unmatched["unmatched_reason"].value_counts().to_dict() if not unmatched.empty and "unmatched_reason" in unmatched.columns else {}
        log_summary(f"SELECTED COUNT EXCEEDS GRADED COUNT | selected={selected_count} graded={graded_count} unmatched={unmatched_count} reasons={reason_counts}")
    reconciliation = write_reconciliation(all_bets, final, unmatched, non_final)
    review_rows = reconciliation[reconciliation["status"] == "review"].copy()
    if not review_rows.empty:
        log_error(f"SELECTED VS GRADED RECONCILIATION HAS REVIEW ROWS | audit={RECONCILIATION_AUDIT_FILE}")
        return False
    if not result_counts.empty:
        blank_results = int(pd.to_numeric(result_counts["blank_results"], errors="coerce").fillna(0).sum())
        if blank_results > 0:
            log_error(f"BLANK BET RESULTS FOUND | rows={blank_results} | audit={RESULT_COUNTS_FILE}")
            return False
    for date_val, group in final.groupby("game_date"):
        date_str = normalize_date(date_val)
        daily_df = enforce_output_cols(group.copy())
        daily_path = DAILY_DIR / f"{date_str}_MLB_final.csv"
        write_csv_checked(daily_df, daily_path, f"MLB graded daily output {date_str}")
        daily_counts = group["bet_result"].value_counts().to_dict()
        log_summary(f"MLB DAILY | DATE={date_str} | ROWS={len(daily_df)} | RESULTS={daily_counts}")
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
