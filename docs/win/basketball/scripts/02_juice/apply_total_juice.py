#!/usr/bin/env python3
# docs/win/basketball/scripts/02_juice/apply_total_juice.py

import math
import sys
import traceback
from datetime import datetime, UTC
from pathlib import Path

import pandas as pd

# =========================
# PATHS
# =========================

INPUT_DIR  = Path("docs/win/basketball/01_merge/01_merguiced")
OUTPUT_DIR = Path("docs/win/basketball/02_juice")
ERROR_DIR  = Path("docs/win/basketball/errors/02_juice")

NBA_CONFIG   = Path("config/basketball/nba/nba_totals_juice.csv")
NCAAB_CONFIG = Path("config/basketball/ncaab/ncaab_totals_juice.csv")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = ERROR_DIR / "apply_total_juice.txt"


# =========================
# LOGGING
# =========================

def _now():
    return datetime.now(UTC).isoformat()


def _log(msg: str, level: str = "INFO"):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{_now()} | {level:<5} | {msg.rstrip()}\n")


def _audit(stage: str, status: str, msg: str = "", df: pd.DataFrame = None):
    lines = [f"  [{stage}] {status}"]
    if msg:
        lines.append(f"    MSG  : {msg}")
    if df is not None:
        lines.append(f"    ROWS : {len(df)}  COLS: {len(df.columns)}  NULLS: {df.isnull().sum().sum()}")
        lines.append(f"    SAMPLE:\n{df.head(3).to_string(index=False)}")
    lines.append("  " + "-" * 40)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _write_summary(summary: dict, per_file: list) -> None:
    lines = [
        "",
        "=" * 60,
        f"SUMMARY  {_now()}",
        "=" * 60,
        f"  files_found   : {summary['files_found']}",
        f"  files_written : {summary['files_written']}",
        f"  nba_files     : {summary['nba_files']}",
        f"  ncaab_files   : {summary['ncaab_files']}",
        f"  total_rows    : {summary['total_rows']}",
        f"  errors        : {summary['errors']}",
        "",
        f"  {'file':<50} {'league':>6} {'rows':>6} {'nulls':>6} {'status':>10}",
    ]
    for pf in per_file:
        lines.append(
            f"  {pf['name']:<50} {pf['league']:>6} {pf['rows']:>6} "
            f"{pf['nulls']:>6} {pf['status']:>10}"
        )
    status = "SUCCESS" if summary["errors"] == 0 else "COMPLETED WITH ERRORS"
    lines += ["", f"STATUS: {status}", "=" * 60]
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# =========================
# HELPERS
# =========================

def decimal_to_american(d):
    if d is None or not math.isfinite(d) or d <= 1:
        return ""
    return f"+{int(round((d - 1) * 100))}" if d >= 2 else f"-{int(round(100 / (d - 1)))}"


def validate_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def atomic_write(df, path):
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


# =========================
# LOAD CONFIG
# =========================

try:
    NBA_JUICE   = pd.read_csv(NBA_CONFIG)
    NCAAB_JUICE = pd.read_csv(NCAAB_CONFIG)
except FileNotFoundError as e:
    raise SystemExit(f"ERROR: Missing juice config file — {e}") from e

NBA_JUICE["band_min"]    = pd.to_numeric(NBA_JUICE["band_min"],    errors="coerce")
NBA_JUICE["band_max"]    = pd.to_numeric(NBA_JUICE["band_max"],    errors="coerce")
NBA_JUICE["extra_juice"] = pd.to_numeric(NBA_JUICE["extra_juice"], errors="coerce")

NCAAB_JUICE["over_under"]  = pd.to_numeric(NCAAB_JUICE["over_under"],  errors="coerce")
NCAAB_JUICE["extra_juice"] = pd.to_numeric(NCAAB_JUICE["extra_juice"], errors="coerce")


# =========================
# NBA PROCESSING
# =========================

def apply_nba(df):
    validate_columns(df, ["total", "acceptable_over", "acceptable_under"])

    def process(row, side):
        total = pd.to_numeric(row.get("total"), errors="coerce")
        if pd.isna(total):
            return None, ""
        base_col     = "acceptable_over" if side == "over" else "acceptable_under"
        base_decimal = pd.to_numeric(row.get(base_col), errors="coerce")
        if pd.isna(base_decimal) or not math.isfinite(base_decimal) or base_decimal <= 1:
            return None, ""
        band  = NBA_JUICE[
            (NBA_JUICE["band_min"] <= total) &
            (total <= NBA_JUICE["band_max"]) &
            (NBA_JUICE["side"] == side)
        ]
        extra = float(band.iloc[0]["extra_juice"]) if not band.empty else 0.0
        if band.empty:
            _log(f"no band match: total={total} side={side}", "WARN")
        if pd.isna(extra) or not math.isfinite(extra):
            extra = 0.0
        final = base_decimal * (1 + extra)
        if not math.isfinite(final) or final <= 1:
            return None, ""
        return final, decimal_to_american(final)

    df[["total_over_juice_decimal",  "total_over_juice_odds"]]  = \
        df.apply(lambda r: process(r, "over"),  axis=1, result_type="expand")
    df[["total_under_juice_decimal", "total_under_juice_odds"]] = \
        df.apply(lambda r: process(r, "under"), axis=1, result_type="expand")

    df["acceptable_over"]  = df["total_over_juice_decimal"]
    df["acceptable_under"] = df["total_under_juice_decimal"]

    return df


# =========================
# NCAAB PROCESSING
# =========================

def apply_ncaab(df):
    validate_columns(df, ["total", "acceptable_over", "acceptable_under"])

    def process(row, side):
        total = pd.to_numeric(row.get("total"), errors="coerce")
        if pd.isna(total):
            return None, ""
        base_col     = "acceptable_over" if side == "over" else "acceptable_under"
        base_decimal = pd.to_numeric(row.get(base_col), errors="coerce")
        if pd.isna(base_decimal) or not math.isfinite(base_decimal) or base_decimal <= 1:
            return None, ""
        jt_side = NCAAB_JUICE[NCAAB_JUICE["side"] == side].copy()
        if jt_side.empty:
            extra = 0.0
        else:
            jt_side["diff"] = (jt_side["over_under"] - total).abs()
            nearest = jt_side.loc[jt_side["diff"].idxmin()]
            extra   = float(nearest["extra_juice"])
        if pd.isna(extra) or not math.isfinite(extra):
            extra = 0.0
        final = base_decimal * (1 + extra)
        if not math.isfinite(final) or final <= 1:
            return None, ""
        return final, decimal_to_american(final)

    df[["total_over_juice_decimal",  "total_over_juice_odds"]]  = \
        df.apply(lambda r: process(r, "over"),  axis=1, result_type="expand")
    df[["total_under_juice_decimal", "total_under_juice_odds"]] = \
        df.apply(lambda r: process(r, "under"), axis=1, result_type="expand")

    df["acceptable_over"]  = df["total_over_juice_decimal"]
    df["acceptable_under"] = df["total_under_juice_decimal"]

    return df


# =========================
# MAIN
# =========================

def main():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== apply_total_juice RUN {_now()} ===\n")

    summary = {
        "files_found": 0, "files_written": 0,
        "nba_files": 0, "ncaab_files": 0,
        "total_rows": 0, "errors": 0,
    }
    per_file = []

    # clean old outputs
    for f in OUTPUT_DIR.glob("*_NBA_total.csv"):
        f.unlink(missing_ok=True)
    for f in OUTPUT_DIR.glob("*_NCAAB_total.csv"):
        f.unlink(missing_ok=True)

    try:
        _log(f"INPUT_DIR   : {INPUT_DIR}")
        _log(f"NBA_CONFIG  : {NBA_CONFIG}")
        _log(f"NCAAB_CONFIG: {NCAAB_CONFIG}")

        input_files = sorted(INPUT_DIR.iterdir())
        _log(f"Files in input dir: {len(input_files)}")

        for f in input_files:
            name   = f.name
            league = None

            if name.endswith("_NBA_total.csv"):
                league = "NBA"
            elif name.endswith("_NCAAB_total.csv"):
                league = "NCAAB"
            else:
                continue

            summary["files_found"] += 1
            pf = {"name": name, "league": league, "rows": 0, "nulls": 0, "status": "ok"}
            _log(f"--- FILE: {name}")

            try:
                df = pd.read_csv(f)

                if df.empty:
                    _log(f"{name} empty — skipping")
                    pf["status"] = "empty"
                    per_file.append(pf)
                    continue

                pf["rows"] = len(df)
                summary["total_rows"] += len(df)

                df = apply_nba(df) if league == "NBA" else apply_ncaab(df)

                pf["nulls"] = int(df.isnull().sum().sum())
                out_path    = OUTPUT_DIR / name
                atomic_write(df, out_path)

                summary["files_written"] += 1
                summary[f"{league.lower()}_files"] += 1

                _log(f"WROTE: {out_path} ({len(df)} rows, {pf['nulls']} nulls)")
                _audit(f"JUICE_TOTAL_{league}", "SUCCESS",
                       msg=f"Applied {league} Totals Juice to {name}", df=df)

            except Exception as e:
                _log(f"{name} FAILED: {e}\n{traceback.format_exc()}", "ERROR")
                _audit(f"JUICE_TOTAL_{league}", "FAILED", msg=str(e))
                pf["status"] = "error"
                summary["errors"] += 1

            per_file.append(pf)

    except Exception as e:
        _log(f"FATAL: {e}\n{traceback.format_exc()}", "ERROR")
        _audit("JUICE_TOTAL_CRITICAL", "FAILED", msg=str(e))
        _write_summary(summary, per_file)
        sys.exit(1)

    _write_summary(summary, per_file)
    print("apply_total_juice complete.")


if __name__ == "__main__":
    main()
