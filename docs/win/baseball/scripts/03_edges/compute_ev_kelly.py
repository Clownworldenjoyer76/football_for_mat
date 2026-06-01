#!/usr/bin/env python3
# docs/win/baseball/scripts/03_edges/compute_ev_kelly.py

import traceback
from datetime import datetime, UTC
from pathlib import Path

import numpy as np
import pandas as pd

INPUT_DIR = Path("docs/win/baseball/03_edges")
OUTPUT_DIR = Path("docs/win/baseball/03_edges/ev_kelly")
ERROR_DIR = Path("docs/win/baseball/errors/03_edges")
LOG_FILE = ERROR_DIR / "compute_ev_kelly.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)


MONEYLINE_REQUIRED_COLUMNS = [
    "game_id",
    "sport",
    "league",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
    "home_prob",
    "away_prob",
    "home_dk_decimal_moneyline",
    "away_dk_decimal_moneyline",
    "home_edge_decimal_moneyline",
    "away_edge_decimal_moneyline",
]

RUN_LINE_REQUIRED_COLUMNS = [
    "game_id",
    "sport",
    "league",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
    "home_normalized_prob_run_line",
    "away_normalized_prob_run_line",
    "home_dk_run_line_decimal",
    "away_dk_run_line_decimal",
    "home_edge_decimal_run_line",
    "away_edge_decimal_run_line",
]

TOTAL_REQUIRED_COLUMNS = [
    "game_id",
    "sport",
    "league",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
    "fair_total_over_decimal",
    "fair_total_under_decimal",
    "dk_total_over_decimal",
    "dk_total_under_decimal",
    "over_edge_decimal_total",
    "under_edge_decimal_total",
]


# =========================
# LOGGING
# =========================

def _now():
    return datetime.now(UTC).isoformat()


def _log(msg: str, level: str = "INFO"):
    with open(LOG_FILE, "a", encoding="utf-8") as log_f:
        log_f.write(f"{_now()} | {level:<5} | {msg.rstrip()}\n")


def _write_summary(summary: dict, per_file: list) -> None:
    lines = [
        "",
        "=" * 60,
        f"SUMMARY  {_now()}",
        "=" * 60,
        f"  files_processed  : {summary['files_processed']}",
        f"  rows_processed   : {summary['rows_processed']}",
        f"  moneyline_files  : {summary['moneyline_files']}",
        f"  run_line_files   : {summary['run_line_files']}",
        f"  total_files      : {summary['total_files']}",
        f"  skipped          : {summary['skipped']}",
        f"  schema_errors    : {summary['schema_errors']}",
        f"  neg_kelly_clipped: {summary['neg_kelly_clipped']}",
        f"  missing_adj_ev   : {summary['missing_adj_ev']}",
        f"  errors           : {summary['errors']}",
        "",
        f"  {'file':<48} {'market':<12} {'rows':>5} {'neg_kelly':>10} {'missing_adj':>12} {'status':>14}",
    ]

    for pf in per_file:
        lines.append(
            f"  {pf['name']:<48} {pf['market']:<12} {pf['rows']:>5} "
            f"{pf['neg_kelly']:>10} {pf['missing_adj']:>12} {pf['status']:>14}"
        )

    status = "SUCCESS" if summary["errors"] == 0 and summary["schema_errors"] == 0 else "COMPLETED WITH ERRORS"
    lines += ["", f"STATUS: {status}", "=" * 60]

    with open(LOG_FILE, "a", encoding="utf-8") as log_f:
        log_f.write("\n".join(lines) + "\n")


# =========================
# SCHEMA GUARDS
# =========================

def duplicate_columns(columns) -> list:
    seen = set()
    dupes = []

    for col in columns:
        if col in seen and col not in dupes:
            dupes.append(col)
        seen.add(col)

    return dupes


def assert_no_duplicate_columns(df: pd.DataFrame, label: str) -> None:
    dupes = duplicate_columns(list(df.columns))

    if dupes:
        raise ValueError(f"{label} has duplicate columns: {dupes}")


def assert_required_columns(df: pd.DataFrame, required_columns: list, label: str) -> None:
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def validate_input_schema(df: pd.DataFrame, market: str, file_name: str) -> None:
    assert_no_duplicate_columns(df, f"{file_name} input")

    if market == "moneyline":
        assert_required_columns(df, MONEYLINE_REQUIRED_COLUMNS, f"{file_name} moneyline input")
    elif market == "run_line":
        assert_required_columns(df, RUN_LINE_REQUIRED_COLUMNS, f"{file_name} run_line input")
    elif market == "total":
        assert_required_columns(df, TOTAL_REQUIRED_COLUMNS, f"{file_name} total input")
    else:
        raise ValueError(f"{file_name} unknown market for schema validation: {market}")


def write_csv_checked(df: pd.DataFrame, output_path: Path) -> None:
    assert_no_duplicate_columns(df, f"{output_path} output")
    df.to_csv(output_path, index=False)


# =========================
# HELPERS
# =========================

def compute_ev(p, dec):
    p = pd.to_numeric(p, errors="coerce")
    dec = pd.to_numeric(dec, errors="coerce")
    return (p * dec) - 1


def compute_kelly(p, dec, file_name=""):
    p = pd.to_numeric(p, errors="coerce")
    dec = pd.to_numeric(dec, errors="coerce")

    b = dec - 1
    q = 1 - p

    k = pd.Series(np.nan, index=p.index, dtype="float64")
    valid = b.notna() & (b != 0) & p.notna() & np.isfinite(b) & np.isfinite(p)

    k.loc[valid] = ((b.loc[valid] * p.loc[valid]) - q.loc[valid]) / b.loc[valid]

    neg = k[k.notna() & (k < 0)]

    if not neg.empty:
        _log(
            f"{file_name} | {len(neg)} negative Kelly values clipped to 0 "
            f"(min={neg.min():.4f})",
            "WARN",
        )

    k = k.clip(lower=0)
    return k, len(neg)


def adjusted_ev(df, adjusted_col, fallback_ev, file_name):
    """
    Uses adjusted edge column as EV.
    Required-column validation guarantees adjusted_col exists before this runs.
    Fallback is used only where adjusted column exists but row value is NaN.
    """
    raw = pd.to_numeric(fallback_ev, errors="coerce")
    adj = pd.to_numeric(df[adjusted_col], errors="coerce")

    missing = int(adj.isna().sum())

    if missing > 0:
        _log(
            f"{file_name} | {missing} rows missing adjusted EV in {adjusted_col}; using raw EV fallback for those rows",
            "WARN",
        )

    return adj.where(adj.notna(), raw), missing


# =========================
# MARKET PROCESSORS
# =========================

def process_moneyline(df, file_name):
    raw_home_ev = compute_ev(df["home_prob"], df["home_dk_decimal_moneyline"])
    raw_away_ev = compute_ev(df["away_prob"], df["away_dk_decimal_moneyline"])

    df["home_ml_ev"], h_missing = adjusted_ev(
        df,
        "home_edge_decimal_moneyline",
        raw_home_ev,
        file_name,
    )

    df["away_ml_ev"], a_missing = adjusted_ev(
        df,
        "away_edge_decimal_moneyline",
        raw_away_ev,
        file_name,
    )

    home_kelly, h_neg = compute_kelly(
        df["home_prob"],
        df["home_dk_decimal_moneyline"],
        file_name,
    )

    away_kelly, a_neg = compute_kelly(
        df["away_prob"],
        df["away_dk_decimal_moneyline"],
        file_name,
    )

    df["home_ml_kelly"] = home_kelly
    df["away_ml_kelly"] = away_kelly

    return df, h_neg + a_neg, h_missing + a_missing


def process_run_line(df, file_name):
    raw_home_ev = compute_ev(
        df["home_normalized_prob_run_line"],
        df["home_dk_run_line_decimal"],
    )

    raw_away_ev = compute_ev(
        df["away_normalized_prob_run_line"],
        df["away_dk_run_line_decimal"],
    )

    df["home_rl_ev"], h_missing = adjusted_ev(
        df,
        "home_edge_decimal_run_line",
        raw_home_ev,
        file_name,
    )

    df["away_rl_ev"], a_missing = adjusted_ev(
        df,
        "away_edge_decimal_run_line",
        raw_away_ev,
        file_name,
    )

    home_kelly, h_neg = compute_kelly(
        df["home_normalized_prob_run_line"],
        df["home_dk_run_line_decimal"],
        file_name,
    )

    away_kelly, a_neg = compute_kelly(
        df["away_normalized_prob_run_line"],
        df["away_dk_run_line_decimal"],
        file_name,
    )

    df["home_rl_kelly"] = home_kelly
    df["away_rl_kelly"] = away_kelly

    return df, h_neg + a_neg, h_missing + a_missing


def process_total(df, file_name):
    df["over_prob"] = 1 / pd.to_numeric(df["fair_total_over_decimal"], errors="coerce")
    df["under_prob"] = 1 / pd.to_numeric(df["fair_total_under_decimal"], errors="coerce")

    raw_over_ev = compute_ev(df["over_prob"], df["dk_total_over_decimal"])
    raw_under_ev = compute_ev(df["under_prob"], df["dk_total_under_decimal"])

    df["over_ev"], o_missing = adjusted_ev(
        df,
        "over_edge_decimal_total",
        raw_over_ev,
        file_name,
    )

    df["under_ev"], u_missing = adjusted_ev(
        df,
        "under_edge_decimal_total",
        raw_under_ev,
        file_name,
    )

    over_kelly, o_neg = compute_kelly(
        df["over_prob"],
        df["dk_total_over_decimal"],
        file_name,
    )

    under_kelly, u_neg = compute_kelly(
        df["under_prob"],
        df["dk_total_under_decimal"],
        file_name,
    )

    df["over_kelly"] = over_kelly
    df["under_kelly"] = under_kelly

    return df, o_neg + u_neg, o_missing + u_missing


# =========================
# MAIN
# =========================

def main():
    with open(LOG_FILE, "w", encoding="utf-8") as log_f:
        log_f.write(f"=== compute_ev_kelly RUN {_now()} ===\n")

    summary = {
        "files_processed": 0,
        "rows_processed": 0,
        "moneyline_files": 0,
        "run_line_files": 0,
        "total_files": 0,
        "skipped": 0,
        "schema_errors": 0,
        "neg_kelly_clipped": 0,
        "missing_adj_ev": 0,
        "errors": 0,
    }

    per_file = []

    _log(f"INPUT_DIR : {INPUT_DIR}")
    _log(f"OUTPUT_DIR: {OUTPUT_DIR}")

    input_files = sorted(INPUT_DIR.glob("*.csv"))
    _log(f"Files found: {len(input_files)}")

    for out_file in OUTPUT_DIR.glob("*.csv"):
        out_file.unlink()

    for input_file in input_files:
        name = input_file.name.lower()
        market = None

        pf = {
            "name": input_file.name,
            "market": "unknown",
            "rows": 0,
            "neg_kelly": 0,
            "missing_adj": 0,
            "status": "ok",
        }

        if "moneyline" in name:
            market = "moneyline"
        elif "run_line" in name:
            market = "run_line"
        elif "total" in name:
            market = "total"
        else:
            _log(f"SKIP unrecognized file: {input_file.name}")
            pf["status"] = "skipped"
            summary["skipped"] += 1
            per_file.append(pf)
            continue

        pf["market"] = market
        _log(f"--- FILE: {input_file.name}  market={market}")

        try:
            df = pd.read_csv(input_file)

            if df.empty:
                _log(f"{input_file.name} empty — skipping")
                pf["status"] = "empty"
                summary["skipped"] += 1
                per_file.append(pf)
                continue

            try:
                validate_input_schema(df, market, input_file.name)
            except Exception as schema_error:
                _log(f"{input_file.name} SCHEMA FAILED: {schema_error}", "ERROR")
                pf["status"] = "schema_error"
                summary["schema_errors"] += 1
                per_file.append(pf)
                continue

            pf["rows"] = len(df)
            summary["rows_processed"] += len(df)

            if market == "moneyline":
                df, neg_kelly, missing_adj = process_moneyline(df, input_file.name)
                summary["moneyline_files"] += 1
            elif market == "run_line":
                df, neg_kelly, missing_adj = process_run_line(df, input_file.name)
                summary["run_line_files"] += 1
            else:
                df, neg_kelly, missing_adj = process_total(df, input_file.name)
                summary["total_files"] += 1

            pf["neg_kelly"] = neg_kelly
            pf["missing_adj"] = missing_adj

            summary["neg_kelly_clipped"] += neg_kelly
            summary["missing_adj_ev"] += missing_adj

            output_path = OUTPUT_DIR / input_file.name
            write_csv_checked(df, output_path)

            summary["files_processed"] += 1

            _log(
                f"WROTE: {output_path} "
                f"({len(df)} rows, {neg_kelly} kelly clipped, {missing_adj} adjusted EV fallback)"
            )

        except Exception as e:
            _log(
                f"{input_file.name} FAILED: {e}\n{traceback.format_exc()}",
                "ERROR",
            )
            pf["status"] = "error"
            summary["errors"] += 1

        per_file.append(pf)

    _write_summary(summary, per_file)
    print("compute_ev_kelly complete.")


if __name__ == "__main__":
    main()
