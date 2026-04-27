#!/usr/bin/env python3
# docs/win/basketball/scripts/02_juice/apply_spread_juice.py

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

NBA_CONFIG   = Path("config/basketball/nba/nba_spreads_juice.csv")
NCAAB_CONFIG = Path("config/basketball/ncaab/ncaab_spreads_juice.csv")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = ERROR_DIR / "apply_spread_juice.txt"


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

def normalize_american(val):
    if pd.isna(val):
        return None
    text = str(val).strip().replace(",", "").replace("+", "")
    try:
        return float(text)
    except Exception:
        return None


def american_to_decimal(a):
    a = normalize_american(a)
    if a is None:
        return None
    return 1 + (a / 100 if a > 0 else 100 / abs(a))


def decimal_to_american(d):
    try:
        d = float(d)
    except Exception:
        return ""
    if not math.isfinite(d) or d <= 1:
        return ""
    return f"+{int(round((d - 1) * 100))}" if d >= 2 else f"-{int(round(100 / (d - 1)))}"


def safe_decimal(v):
    try:
        v = float(v)
    except Exception:
        return 1.01
    return v if math.isfinite(v) and v > 1 else 1.01


def atomic_write(df, path):
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def validate_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def ensure_american_columns(df):
    for side in ["home", "away"]:
        amer = f"{side}_acceptable_spread_american"
        dec  = f"{side}_acceptable_spread_decimal"
        if amer not in df.columns and dec in df.columns:
            df[amer] = df[dec].apply(decimal_to_american)
    return df


# =========================
# LOAD CONFIG
# =========================

try:
    NBA_JUICE_TABLE   = pd.read_csv(NBA_CONFIG)
    NCAAB_JUICE_TABLE = pd.read_csv(NCAAB_CONFIG)
except FileNotFoundError as e:
    raise SystemExit(f"ERROR: Missing juice config file — {e}") from e


# =========================
# NBA SPREAD JUICE
# =========================

def apply_nba(df):
    df = ensure_american_columns(df)
    validate_columns(df, ["home_spread", "away_spread",
                           "home_acceptable_spread_american",
                           "away_acceptable_spread_american"])
    jt = NBA_JUICE_TABLE

    def process(row, side):
        spread = float(row[f"{side}_spread"])
        odds   = normalize_american(row[f"{side}_acceptable_spread_american"])
        fav_ud = "favorite" if spread < 0 else "underdog"
        band   = jt[
            (jt.band_min <= abs(spread)) &
            (abs(spread) <= jt.band_max) &
            (jt.fav_ud == fav_ud) &
            (jt.venue == side)
        ]
        extra = float(band.iloc[0]["extra_juice"]) if not band.empty else 0.0
        if not math.isfinite(extra):
            extra = 0.0
        if band.empty:
            _log(f"no band match: spread={spread} venue={side} fav_ud={fav_ud}", "WARN")
        base  = safe_decimal(american_to_decimal(odds))
        final = base * (1 + extra)
        return final, decimal_to_american(final)

    for side in ["home", "away"]:
        df[[f"{side}_spread_juice_decimal",
            f"{side}_spread_juice_odds"]] = df.apply(
            lambda r: process(r, side), axis=1, result_type="expand"
        )

    df["home_acceptable_spread_decimal"]  = df["home_spread_juice_decimal"]
    df["away_acceptable_spread_decimal"]  = df["away_spread_juice_decimal"]
    df["home_acceptable_spread_american"] = df["home_spread_juice_odds"]
    df["away_acceptable_spread_american"] = df["away_spread_juice_odds"]

    return df


# =========================
# NCAAB SPREAD JUICE
# =========================

def apply_ncaab(df):
    df = ensure_american_columns(df)
    validate_columns(df, ["home_spread", "away_spread",
                           "home_acceptable_spread_american",
                           "away_acceptable_spread_american"])
    jt = NCAAB_JUICE_TABLE

    def process(row, side):
        spread   = float(row[f"{side}_spread"])
        odds     = normalize_american(row[f"{side}_acceptable_spread_american"])
        jt_temp  = jt.copy()
        jt_temp["diff"] = (jt_temp["spread"] - spread).abs()
        nearest  = jt_temp.loc[jt_temp["diff"].idxmin()]
        extra    = float(nearest["extra_juice"])
        if not math.isfinite(extra):
            extra = 0.0
        base  = safe_decimal(american_to_decimal(odds))
        final = base * (1 + extra)
        return final, decimal_to_american(final)

    for side in ["home", "away"]:
        df[[f"{side}_spread_juice_decimal",
            f"{side}_spread_juice_odds"]] = df.apply(
            lambda r: process(r, side), axis=1, result_type="expand"
        )

    df["home_acceptable_spread_decimal"]  = df["home_spread_juice_decimal"]
    df["away_acceptable_spread_decimal"]  = df["away_spread_juice_decimal"]
    df["home_acceptable_spread_american"] = df["home_spread_juice_odds"]
    df["away_acceptable_spread_american"] = df["away_spread_juice_odds"]

    return df


# =========================
# MAIN
# =========================

def main():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== apply_spread_juice RUN {_now()} ===\n")

    summary = {
        "files_found": 0, "files_written": 0,
        "nba_files": 0, "ncaab_files": 0,
        "total_rows": 0, "errors": 0,
    }
    per_file = []

    # clean old outputs
    for f in OUTPUT_DIR.glob("*_NBA_spread.csv"):
        f.unlink(missing_ok=True)
    for f in OUTPUT_DIR.glob("*_NCAAB_spread.csv"):
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

            if name.endswith("_NBA_spread.csv"):
                league = "NBA"
            elif name.endswith("_NCAAB_spread.csv"):
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
                _audit(f"JUICE_SPREAD_{league}", "SUCCESS",
                       msg=f"Applied {league} Spread Juice to {name}", df=df)

            except Exception as e:
                _log(f"{name} FAILED: {e}\n{traceback.format_exc()}", "ERROR")
                _audit(f"JUICE_SPREAD_{league}", "FAILED", msg=str(e))
                pf["status"] = "error"
                summary["errors"] += 1

            per_file.append(pf)

    except Exception as e:
        _log(f"FATAL: {e}\n{traceback.format_exc()}", "ERROR")
        _audit("JUICE_SPREAD_CRITICAL", "FAILED", msg=str(e))
        _write_summary(summary, per_file)
        sys.exit(1)

    _write_summary(summary, per_file)
    print("apply_spread_juice complete.")


if __name__ == "__main__":
    main()
