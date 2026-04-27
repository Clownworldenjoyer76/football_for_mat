#!/usr/bin/env python3
# docs/win/basketball/scripts/03_edges/compute_edges.py

import re
import traceback
from datetime import datetime, UTC
from pathlib import Path

import pandas as pd

INPUT_DIR  = Path("docs/win/basketball/02_juice")
OUTPUT_DIR = Path("docs/win/basketball/03_edges")
ERROR_DIR  = Path("docs/win/basketball/errors/03_edges")
LOG_FILE   = ERROR_DIR / "compute_edges.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# LOGGING
# =========================

def _now():
    return datetime.now(UTC).isoformat()


def _log(msg: str, level: str = "INFO"):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{_now()} | {level:<5} | {msg.rstrip()}\n")


def _write_summary(summary: dict, per_file: list) -> None:
    lines = [
        "",
        "=" * 60,
        f"SUMMARY  {_now()}",
        "=" * 60,
        f"  files_processed  : {summary['files_processed']}",
        f"  rows_processed   : {summary['rows_processed']}",
        f"  nba_files        : {summary['nba_files']}",
        f"  ncaab_files      : {summary['ncaab_files']}",
        f"  skipped          : {summary['skipped']}",
        f"  null_edges       : {summary['null_edges']}",
        f"  schema_errors    : {summary['schema_errors']}",
        f"  errors           : {summary['errors']}",
        "",
        f"  {'file':<52} {'league':>6} {'market':<10} {'rows':>5} {'null_edges':>10} {'status':>10}",
    ]
    for pf in per_file:
        lines.append(
            f"  {pf['name']:<52} {pf['league']:>6} {pf['market']:<10} {pf['rows']:>5} "
            f"{pf['null_edges']:>10} {pf['status']:>10}"
        )
    status = "SUCCESS" if summary["errors"] == 0 and summary["schema_errors"] == 0 \
        else "COMPLETED WITH ERRORS"
    lines += ["", f"STATUS: {status}", "=" * 60]
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# =========================
# HELPERS
# =========================

def american_to_decimal(odds):
    try:
        odds = float(odds)
    except Exception:
        return None
    if odds > 0:
        return 1 + (odds / 100)
    return 1 + (100 / abs(odds))


def calculate_edge(model_prob, book_decimal):
    p = pd.to_numeric(model_prob,   errors="coerce")
    d = pd.to_numeric(book_decimal, errors="coerce")
    return (p * d) - 1


def validate_columns(df, required_cols):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def extract_date_from_filename(filename):
    match = re.search(r"\d{4}_\d{2}_\d{2}", filename)
    if not match:
        raise ValueError(f"No date found in filename: {filename}")
    return match.group(0)


def count_null_edges(df, cols):
    return sum(df[c].isna().sum() for c in cols if c in df.columns)


def atomic_write_csv(df, output_path):
    tmp = output_path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(output_path)


def ensure_decimal_columns(df):
    pairs = [
        ("dk_total_over_decimal",     "dk_total_over_american"),
        ("dk_total_under_decimal",    "dk_total_under_american"),
        ("home_dk_decimal_moneyline", "home_dk_moneyline_american"),
        ("away_dk_decimal_moneyline", "away_dk_moneyline_american"),
        ("home_dk_spread_decimal",    "home_dk_spread_american"),
        ("away_dk_spread_decimal",    "away_dk_spread_american"),
    ]
    for dec_col, amer_col in pairs:
        if dec_col not in df.columns and amer_col in df.columns:
            df[dec_col] = df[amer_col].apply(american_to_decimal)
    return df


# =========================
# MARKET PROCESSORS
# =========================

def compute_moneyline_edges(df):
    df = ensure_decimal_columns(df)
    validate_columns(df, ["home_prob", "away_prob",
                           "home_dk_decimal_moneyline", "away_dk_decimal_moneyline"])
    df["home_ml_edge"] = calculate_edge(df["home_prob"], df["home_dk_decimal_moneyline"])
    df["away_ml_edge"] = calculate_edge(df["away_prob"], df["away_dk_decimal_moneyline"])
    return df, count_null_edges(df, ["home_ml_edge", "away_ml_edge"])


def compute_spread_edges(df):
    df = ensure_decimal_columns(df)
    validate_columns(df, ["home_prob", "away_prob",
                           "home_dk_spread_decimal", "away_dk_spread_decimal"])
    df["home_spread_edge"] = calculate_edge(df["home_prob"], df["home_dk_spread_decimal"])
    df["away_spread_edge"] = calculate_edge(df["away_prob"], df["away_dk_spread_decimal"])
    return df, count_null_edges(df, ["home_spread_edge", "away_spread_edge"])


def compute_total_edges(df):
    df = ensure_decimal_columns(df)
    validate_columns(df, ["fair_over", "fair_under",
                           "dk_total_over_decimal", "dk_total_under_decimal"])
    df["over_prob"]   = 1 / pd.to_numeric(df["fair_over"],  errors="coerce")
    df["under_prob"]  = 1 / pd.to_numeric(df["fair_under"], errors="coerce")
    df["over_edge"]   = calculate_edge(df["over_prob"],  df["dk_total_over_decimal"])
    df["under_edge"]  = calculate_edge(df["under_prob"], df["dk_total_under_decimal"])
    return df, count_null_edges(df, ["over_edge", "under_edge"])


# =========================
# LEAGUE PROCESSOR
# =========================

def process_league(league, summary, per_file):
    markets = {
        "moneyline": compute_moneyline_edges,
        "spread":    compute_spread_edges,
        "total":     compute_total_edges,
    }

    for market, compute_fn in markets.items():
        files = sorted(INPUT_DIR.glob(f"*_{league}_{market}.csv"))

        if not files:
            _log(f"No files found: league={league} market={market}", "WARN")
            continue

        for f in files:
            pf = {"name": f.name, "league": league, "market": market,
                  "rows": 0, "null_edges": 0, "status": "ok"}
            _log(f"--- FILE: {f.name}  league={league} market={market}")

            try:
                df = pd.read_csv(f)

                if df.empty:
                    _log(f"{f.name} empty — skipping")
                    pf["status"] = "empty"
                    summary["skipped"] += 1
                    per_file.append(pf)
                    continue

                pf["rows"] = len(df)
                summary["rows_processed"] += len(df)

                date       = extract_date_from_filename(f.name)
                df, null_e = compute_fn(df)

                pf["null_edges"]      = null_e
                summary["null_edges"] += null_e

                if null_e > 0:
                    _log(f"{f.name} | {null_e} null edges", "WARN")

                out_path = OUTPUT_DIR / f"{date}_basketball_{league}_{market}.csv"
                atomic_write_csv(df, out_path)

                summary["files_processed"]          += 1
                summary[f"{league.lower()}_files"]  += 1
                _log(f"WROTE: {out_path} ({len(df)} rows, {null_e} null edges)")

            except ValueError as e:
                _log(f"{f.name} schema error: {e}", "ERROR")
                pf["status"] = "schema_error"
                summary["schema_errors"] += 1
            except Exception as e:
                _log(f"{f.name} FAILED: {e}\n{traceback.format_exc()}", "ERROR")
                pf["status"] = "error"
                summary["errors"] += 1

            per_file.append(pf)


# =========================
# MAIN
# =========================

def main():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== compute_edges RUN {_now()} ===\n")

    summary = {
        "files_processed": 0, "rows_processed": 0,
        "nba_files": 0, "ncaab_files": 0,
        "skipped": 0, "null_edges": 0, "schema_errors": 0, "errors": 0,
    }
    per_file = []

    # clear old outputs
    for stale in OUTPUT_DIR.glob("*.csv"):
        stale.unlink(missing_ok=True)

    _log(f"INPUT_DIR : {INPUT_DIR}")
    _log(f"OUTPUT_DIR: {OUTPUT_DIR}")

    try:
        process_league("NBA",   summary, per_file)
        process_league("NCAAB", summary, per_file)
    except Exception as e:
        _log(f"FATAL: {e}\n{traceback.format_exc()}", "ERROR")
        _write_summary(summary, per_file)
        raise

    _write_summary(summary, per_file)
    print("compute_edges complete.")


if __name__ == "__main__":
    main()
