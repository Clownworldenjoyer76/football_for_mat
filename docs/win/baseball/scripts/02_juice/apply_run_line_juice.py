#!/usr/bin/env python3
# docs/win/baseball/scripts/02_juice/apply_run_line_juice.py

import glob
import math
import sys
import traceback
from datetime import datetime, UTC
from pathlib import Path

import pandas as pd

INPUT_DIR = Path("docs/win/baseball/01_merge/01_merguiced")
OUTPUT_DIR = Path("docs/win/baseball/02_juice")
JUICE_FILE = Path("config/baseball/mlb/mlb_run_line_juice.csv")

ERROR_DIR = Path("docs/win/baseball/errors/02_juice")
LOG_FILE = ERROR_DIR / "apply_run_line_juice.txt"

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
        f"  files_found    : {summary['files_found']}",
        f"  files_written  : {summary['files_written']}",
        f"  total_rows     : {summary['total_rows']}",
        f"  applied        : {summary['applied']}",
        f"  skipped_bad    : {summary['skipped_bad']}",
        f"  skipped_noband : {summary['skipped_noband']}",
        f"  schema_errors  : {summary['schema_errors']}",
        f"  errors         : {summary['errors']}",
        "",
        f"  {'file':<45} {'rows':>5} {'applied':>8} {'skipped_bad':>12} {'skipped_noband':>15} {'schema_errors':>15}",
    ]

    for pf in per_file:
        lines.append(
            f"  {pf['name']:<45} {pf['rows']:>5} {pf['applied']:>8} "
            f"{pf['skipped_bad']:>12} {pf['skipped_noband']:>15} {pf['schema_errors']:>15}"
        )

    status = "SUCCESS" if summary["errors"] == 0 and summary["schema_errors"] == 0 else "COMPLETED WITH ERRORS"
    lines += ["", f"STATUS: {status}", "=" * 60]

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# =========================
# SCHEMA VALIDATION
# =========================

REQUIRED_JUICE_COLUMNS = [
    "band_min",
    "band_max",
    "venue",
    "fav_ud",
    "extra_juice",
]

REQUIRED_RUN_LINE_COLUMNS = [
    "last_run",
    "game_id",
    "sport",
    "league",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
    "away_run_line",
    "home_run_line",
    "total",
    "away_dk_run_line_american",
    "home_dk_run_line_american",
    "away_dk_run_line_decimal",
    "home_dk_run_line_decimal",
    "home_pitcher",
    "away_pitcher",
    "home_prob",
    "away_prob",
    "away_projected_runs",
    "home_projected_runs",
    "total_projected_runs",
    "home_prob_run_line",
    "away_prob_run_line",
]

FORBIDDEN_RUN_LINE_COLUMNS = [
    "home_run_line_prob",
    "away_run_line_prob",
]


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


def validate_forbidden_columns(df: pd.DataFrame, forbidden_columns: list, label: str) -> None:
    present = [col for col in forbidden_columns if col in df.columns]

    if present:
        raise ValueError(
            f"{label} contains forbidden obsolete columns: {present}. "
            f"Use home_prob_run_line and away_prob_run_line instead."
        )


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

def find_band(juice_df, odds, venue, fav_ud):
    band = juice_df[
        (juice_df["band_min"] <= odds) &
        (juice_df["band_max"] > odds) &
        (juice_df["venue"] == venue) &
        (juice_df["fav_ud"] == fav_ud)
    ]

    if band.empty:
        return None

    return float(band.iloc[0]["extra_juice"])


# =========================
# ROW PROCESSOR
# =========================

def process_row(df, juice_df, idx, row):
    try:
        home_base = float(row["home_prob_run_line"])
        away_base = float(row["away_prob_run_line"])
        home_odds = float(row["home_dk_run_line_american"])
        away_odds = float(row["away_dk_run_line_american"])
    except Exception:
        _log(f"row={idx} reason=conversion_failed", "SKIP")
        return df, "bad"

    if not math.isfinite(home_base) or not math.isfinite(away_base):
        _log(f"row={idx} reason=invalid_base_prob home={home_base} away={away_base}", "SKIP")
        return df, "bad"

    if not math.isfinite(home_odds) or not math.isfinite(away_odds):
        _log(f"row={idx} reason=invalid_odds home_odds={home_odds} away_odds={away_odds}", "SKIP")
        return df, "bad"

    home_type = "favorite" if home_odds < 0 else "underdog"
    away_type = "favorite" if away_odds < 0 else "underdog"

    home_extra = find_band(juice_df, home_odds, "home", home_type)
    away_extra = find_band(juice_df, away_odds, "away", away_type)

    if home_extra is None or away_extra is None:
        _log(f"row={idx} reason=no_band home_odds={home_odds} away_odds={away_odds}", "SKIP")
        return df, "noband"

    home_final = max(min(home_base + home_extra, 0.75), 0.05)
    away_final = max(min(away_base + away_extra, 0.75), 0.05)

    df.at[idx, "home_juiced_prob_run_line"] = home_final
    df.at[idx, "away_juiced_prob_run_line"] = away_final
    df.at[idx, "home_normalized_prob_run_line"] = home_final
    df.at[idx, "away_normalized_prob_run_line"] = away_final

    return df, "ok"


# =========================
# MAIN
# =========================

def main():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== apply_run_line_juice RUN {_now()} ===\n")

    summary = {
        "files_found": 0,
        "files_written": 0,
        "total_rows": 0,
        "applied": 0,
        "skipped_bad": 0,
        "skipped_noband": 0,
        "schema_errors": 0,
        "errors": 0,
    }

    per_file = []

    for f in OUTPUT_DIR.glob("*run_line.csv"):
        f.unlink()

    try:
        _log(f"INPUT_DIR : {INPUT_DIR}")
        _log(f"JUICE_FILE: {JUICE_FILE}")

        juice_df = read_csv_validated(
            JUICE_FILE,
            REQUIRED_JUICE_COLUMNS,
            f"juice file {JUICE_FILE}",
        )

        juice_df["band_min"] = pd.to_numeric(juice_df["band_min"], errors="coerce")
        juice_df["band_max"] = pd.to_numeric(juice_df["band_max"], errors="coerce")
        juice_df["extra_juice"] = pd.to_numeric(juice_df["extra_juice"], errors="coerce")
        juice_df["venue"] = juice_df["venue"].astype(str).str.strip()
        juice_df["fav_ud"] = juice_df["fav_ud"].astype(str).str.strip()

        bad_juice = juice_df[
            juice_df["band_min"].isna() |
            juice_df["band_max"].isna() |
            juice_df["extra_juice"].isna() |
            (juice_df["venue"] == "") |
            (juice_df["fav_ud"] == "")
        ]

        if not bad_juice.empty:
            raise ValueError(f"juice file contains invalid rows: {len(bad_juice)}")

        files = sorted(glob.glob(str(INPUT_DIR / "*_mlb_run_line.csv")))
        summary["files_found"] = len(files)
        _log(f"Files found: {len(files)}")

        for file_path in files:
            in_path = Path(file_path)
            out_path = OUTPUT_DIR / in_path.name

            pf = {
                "name": in_path.name,
                "rows": 0,
                "applied": 0,
                "skipped_bad": 0,
                "skipped_noband": 0,
                "schema_errors": 0,
            }

            _log(f"--- FILE: {in_path.name}")

            try:
                df = read_csv_validated(
                    in_path,
                    REQUIRED_RUN_LINE_COLUMNS,
                    f"run-line input {in_path.name}",
                )

                validate_forbidden_columns(
                    df,
                    FORBIDDEN_RUN_LINE_COLUMNS,
                    f"run-line input {in_path.name}",
                )

                if df.empty:
                    _log(f"{in_path.name} empty — skipping")
                    per_file.append(pf)
                    continue

                numeric_cols = [
                    "home_prob_run_line",
                    "away_prob_run_line",
                    "home_dk_run_line_american",
                    "away_dk_run_line_american",
                ]

                for c in numeric_cols:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

                output_prob_cols = [
                    "home_juiced_prob_run_line",
                    "away_juiced_prob_run_line",
                    "home_normalized_prob_run_line",
                    "away_normalized_prob_run_line",
                ]

                for c in output_prob_cols:
                    df[c] = pd.NA

                pf["rows"] = len(df)
                summary["total_rows"] += len(df)

                for idx, row in df.iterrows():
                    df, result = process_row(df, juice_df, idx, row)

                    if result == "ok":
                        pf["applied"] += 1
                    elif result == "noband":
                        pf["skipped_noband"] += 1
                    else:
                        pf["skipped_bad"] += 1

                write_csv_validated(
                    df,
                    out_path,
                    f"run-line output {out_path.name}",
                )

                summary["files_written"] += 1
                summary["applied"] += pf["applied"]
                summary["skipped_bad"] += pf["skipped_bad"]
                summary["skipped_noband"] += pf["skipped_noband"]

                _log(
                    f"{in_path.name} | rows={pf['rows']} applied={pf['applied']} "
                    f"skipped_bad={pf['skipped_bad']} skipped_noband={pf['skipped_noband']}"
                )
                _log(f"WROTE: {out_path}")

            except ValueError as e:
                pf["schema_errors"] += 1
                summary["schema_errors"] += 1
                summary["errors"] += 1
                _log(f"{in_path.name} SCHEMA FAILED: {e}\n{traceback.format_exc()}", "ERROR")

            except Exception as e:
                summary["errors"] += 1
                _log(f"{in_path.name} FAILED: {e}\n{traceback.format_exc()}", "ERROR")

            per_file.append(pf)

    except Exception as e:
        _log(f"FATAL: {e}\n{traceback.format_exc()}", "ERROR")
        summary["errors"] += 1
        _write_summary(summary, per_file)
        sys.exit(1)

    _write_summary(summary, per_file)

    if summary["errors"] > 0 or summary["schema_errors"] > 0:
        print(
            f"apply_run_line_juice completed with errors. "
            f"errors={summary['errors']} schema_errors={summary['schema_errors']}"
        )
        sys.exit(1)

    print(
        f"apply_run_line_juice complete. "
        f"files_written={summary['files_written']} "
        f"applied={summary['applied']} "
        f"skipped_bad={summary['skipped_bad']} "
        f"skipped_noband={summary['skipped_noband']} "
        f"schema_errors={summary['schema_errors']}"
    )


if __name__ == "__main__":
    main()
