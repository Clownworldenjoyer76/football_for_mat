#!/usr/bin/env python3
# docs/win/baseball/scripts/02_juice/apply_total_juice.py

import glob
import math
import sys
import traceback
from datetime import datetime, UTC
from pathlib import Path

import pandas as pd

INPUT_DIR = Path("docs/win/baseball/01_merge/01_merguiced")
OUTPUT_DIR = Path("docs/win/baseball/02_juice")
JUICE_FILE = Path("config/baseball/mlb/mlb_totals_juice.csv")

ERROR_DIR = Path("docs/win/baseball/errors/02_juice")
LOG_FILE = ERROR_DIR / "apply_total_juice.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)

MIN_JUICED_DECIMAL = 1.01

REQUIRED_INPUT_COLUMNS = [
    "game_id",
    "sport",
    "league",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
    "total",
    "dk_total_over_american",
    "dk_total_under_american",
    "dk_total_over_decimal",
    "dk_total_under_decimal",
    "fair_total_over_decimal",
    "fair_total_under_decimal",
]

REQUIRED_JUICE_COLUMNS = [
    "band_min",
    "band_max",
    "side",
    "extra_juice",
]


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
        f"  files_found     : {summary['files_found']}",
        f"  files_written   : {summary['files_written']}",
        f"  total_rows      : {summary['total_rows']}",
        f"  over_applied    : {summary['over_applied']}",
        f"  under_applied   : {summary['under_applied']}",
        f"  clamped_decimal : {summary['clamped_decimal']}",
        f"  skipped_bad     : {summary['skipped_bad']}",
        f"  skipped_noband  : {summary['skipped_noband']}",
        f"  schema_errors   : {summary['schema_errors']}",
        f"  errors          : {summary['errors']}",
        "",
        f"  {'file':<45} {'rows':>5} {'o_applied':>10} {'u_applied':>10} {'clamp':>7} {'bad':>5} {'noband':>7}",
    ]

    for pf in per_file:
        lines.append(
            f"  {pf['name']:<45} {pf['rows']:>5} {pf['over_applied']:>10} "
            f"{pf['under_applied']:>10} {pf['clamped_decimal']:>7} "
            f"{pf['skipped_bad']:>5} {pf['skipped_noband']:>7}"
        )

    status = "SUCCESS" if summary["errors"] == 0 and summary["schema_errors"] == 0 else "COMPLETED WITH ERRORS"
    lines += ["", f"STATUS: {status}", "=" * 60]

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# =========================
# SCHEMA GUARDS
# =========================

def duplicate_columns(columns):
    seen = set()
    duplicates = []

    for col in columns:
        if col in seen and col not in duplicates:
            duplicates.append(col)
        seen.add(col)

    return duplicates


def validate_no_duplicate_columns(df: pd.DataFrame, label: str) -> None:
    dupes = duplicate_columns(list(df.columns))

    if dupes:
        raise ValueError(f"{label} has duplicate columns: {dupes}")


def validate_required_columns(df: pd.DataFrame, required_columns: list, label: str) -> None:
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def read_csv_validated(path: Path, required_columns: list, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    validate_no_duplicate_columns(df, label)
    validate_required_columns(df, required_columns, label)

    return df


def write_csv_validated(df: pd.DataFrame, path: Path, label: str) -> None:
    validate_no_duplicate_columns(df, label)
    df.to_csv(path, index=False)


# =========================
# JUICE LOOKUP
# =========================

def find_band_row(juice_df, total, side):
    band = juice_df[
        (juice_df["band_min"] <= total) &
        (total < juice_df["band_max"]) &
        (juice_df["side"] == side)
    ]

    if band.empty:
        return None

    return float(band.iloc[0]["extra_juice"])


# =========================
# SIDE PROCESSOR
# =========================

def process_side(df, juice_df, side):
    fair_col = f"fair_total_{side}_decimal"
    juiced_dec_col = f"{side}_juiced_decimal_total"
    juiced_prob_col = f"{side}_juiced_prob_total"

    df[juiced_dec_col] = pd.NA
    df[juiced_prob_col] = pd.NA

    applied = 0
    skipped_noband = 0
    skipped_bad = 0
    clamped_decimal = 0

    for idx, row in df.iterrows():
        try:
            total = round(float(row["total"]), 1)
            fair_decimal = float(row[fair_col])
        except Exception:
            _log(f"row={idx} side={side} reason=bad_parse", "SKIP")
            skipped_bad += 1
            continue

        if not math.isfinite(fair_decimal) or fair_decimal <= 1:
            _log(f"row={idx} side={side} reason=bad_fair_decimal val={fair_decimal}", "SKIP")
            skipped_bad += 1
            continue

        extra = find_band_row(juice_df, total, side)

        if extra is None:
            _log(f"row={idx} side={side} reason=no_band total={total}", "SKIP")
            skipped_noband += 1
            continue

        try:
            juiced_decimal = fair_decimal * (1 - extra)

            if not math.isfinite(juiced_decimal):
                _log(
                    f"row={idx} side={side} reason=invalid_juiced_decimal "
                    f"val={juiced_decimal} fair={fair_decimal} extra={extra}",
                    "SKIP",
                )
                skipped_bad += 1
                continue

            if juiced_decimal <= 1:
                _log(
                    f"row={idx} side={side} reason=clamped_juiced_decimal "
                    f"original={juiced_decimal} clamped={MIN_JUICED_DECIMAL} "
                    f"fair={fair_decimal} extra={extra}",
                    "WARN",
                )
                juiced_decimal = MIN_JUICED_DECIMAL
                clamped_decimal += 1

            df.at[idx, juiced_dec_col] = juiced_decimal
            df.at[idx, juiced_prob_col] = 1 / juiced_decimal
            applied += 1

        except Exception:
            _log(f"row={idx} side={side} reason=calc_error", "SKIP")
            skipped_bad += 1

    return df, applied, skipped_noband, skipped_bad, clamped_decimal


# =========================
# NORMALIZATION
# =========================

def apply_normalization(df):
    df["over_normalized_prob_total"] = pd.NA
    df["under_normalized_prob_total"] = pd.NA

    for idx, row in df.iterrows():
        try:
            op = float(row["over_juiced_prob_total"])
            up = float(row["under_juiced_prob_total"])

            if not math.isfinite(op) or not math.isfinite(up):
                continue

            total = op + up

            if total <= 0:
                continue

            df.at[idx, "over_normalized_prob_total"] = op / total
            df.at[idx, "under_normalized_prob_total"] = up / total

        except Exception:
            continue

    return df


# =========================
# MAIN
# =========================

def main():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== apply_total_juice RUN {_now()} ===\n")

    summary = {
        "files_found": 0,
        "files_written": 0,
        "total_rows": 0,
        "over_applied": 0,
        "under_applied": 0,
        "clamped_decimal": 0,
        "skipped_bad": 0,
        "skipped_noband": 0,
        "schema_errors": 0,
        "errors": 0,
    }

    per_file = []

    for f in OUTPUT_DIR.glob("*total.csv"):
        f.unlink()

    try:
        _log(f"INPUT_DIR : {INPUT_DIR}")
        _log(f"JUICE_FILE: {JUICE_FILE}")

        juice_df = read_csv_validated(
            JUICE_FILE,
            REQUIRED_JUICE_COLUMNS,
            f"juice file {JUICE_FILE}",
        )

        juice_df["band_min"] = juice_df["band_min"].astype(float)
        juice_df["band_max"] = juice_df["band_max"].astype(float)
        juice_df["side"] = juice_df["side"].astype(str).str.strip()
        juice_df["extra_juice"] = juice_df["extra_juice"].astype(float)

        files = sorted(glob.glob(str(INPUT_DIR / "*_mlb_total.csv")))
        summary["files_found"] = len(files)

        _log(f"Files found: {len(files)}")

        if not files:
            _log("No total files found — exiting", "WARN")
            _write_summary(summary, per_file)
            return

        for file_path in files:
            in_path = Path(file_path)
            out_path = OUTPUT_DIR / in_path.name

            pf = {
                "name": in_path.name,
                "rows": 0,
                "over_applied": 0,
                "under_applied": 0,
                "clamped_decimal": 0,
                "skipped_bad": 0,
                "skipped_noband": 0,
            }

            _log(f"--- FILE: {in_path.name}")

            try:
                df = read_csv_validated(
                    in_path,
                    REQUIRED_INPUT_COLUMNS,
                    f"total input {in_path.name}",
                )

                if df.empty:
                    _log(f"{in_path.name} empty — skipping")
                    per_file.append(pf)
                    continue

                pf["rows"] = len(df)
                summary["total_rows"] += len(df)

                df, o_applied, o_noband, o_bad, o_clamped = process_side(df, juice_df, "over")
                df, u_applied, u_noband, u_bad, u_clamped = process_side(df, juice_df, "under")
                df = apply_normalization(df)

                pf["over_applied"] = o_applied
                pf["under_applied"] = u_applied
                pf["clamped_decimal"] = o_clamped + u_clamped
                pf["skipped_bad"] = o_bad + u_bad
                pf["skipped_noband"] = o_noband + u_noband

                write_csv_validated(
                    df,
                    out_path,
                    f"total output {out_path.name}",
                )

                summary["files_written"] += 1
                summary["over_applied"] += o_applied
                summary["under_applied"] += u_applied
                summary["clamped_decimal"] += pf["clamped_decimal"]
                summary["skipped_bad"] += pf["skipped_bad"]
                summary["skipped_noband"] += pf["skipped_noband"]

                _log(
                    f"{in_path.name} | rows={pf['rows']} "
                    f"over_applied={o_applied} "
                    f"under_applied={u_applied} "
                    f"clamped_decimal={pf['clamped_decimal']} "
                    f"skipped_bad={pf['skipped_bad']} "
                    f"skipped_noband={pf['skipped_noband']}"
                )

                _log(f"WROTE: {out_path}")

            except ValueError as e:
                _log(f"{in_path.name} SCHEMA FAILED: {e}\n{traceback.format_exc()}", "ERROR")
                summary["schema_errors"] += 1
                summary["errors"] += 1

            except Exception as e:
                _log(f"{in_path.name} FAILED: {e}\n{traceback.format_exc()}", "ERROR")
                summary["errors"] += 1

            per_file.append(pf)

    except Exception as e:
        _log(f"FATAL: {e}\n{traceback.format_exc()}", "ERROR")
        _write_summary(summary, per_file)
        sys.exit(1)

    _write_summary(summary, per_file)

    if summary["errors"] > 0 or summary["schema_errors"] > 0:
        print(
            f"apply_total_juice completed with errors. "
            f"errors={summary['errors']} schema_errors={summary['schema_errors']}"
        )
        sys.exit(1)

    print(
        f"apply_total_juice complete. "
        f"files_written={summary['files_written']} "
        f"total_rows={summary['total_rows']} "
        f"clamped_decimal={summary['clamped_decimal']} "
        f"skipped_bad={summary['skipped_bad']} "
        f"skipped_noband={summary['skipped_noband']} "
        f"errors={summary['errors']}"
    )


if __name__ == "__main__":
    main()
