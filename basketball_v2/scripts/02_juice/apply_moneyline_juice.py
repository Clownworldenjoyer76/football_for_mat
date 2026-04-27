#!/usr/bin/env python3
# docs/win/basketball/scripts/02_juice/apply_moneyline_juice.py

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

NBA_CONFIG   = Path("config/basketball/nba/nba_ml_juice.csv")
NCAAB_CONFIG = Path("config/basketball/ncaab/ncaab_ml_juice.csv")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = ERROR_DIR / "apply_moneyline_juice.txt"


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

def normalize_american_value(val):
    if pd.isna(val):
        return None
    text = str(val).strip().replace(",", "")
    if text == "":
        return None
    try:
        return float(text)
    except Exception:
        _log(f"invalid American odds value: {val!r}", "WARN")
        return None


def american_to_decimal(a):
    a = normalize_american_value(a)
    if a is None or a == 0:
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


# =========================
# LOAD CONFIG
# =========================

try:
    NBA_JT   = pd.read_csv(NBA_CONFIG)
    NCAAB_JT = pd.read_csv(NCAAB_CONFIG)
except FileNotFoundError as e:
    raise SystemExit(f"ERROR: Missing juice config file — {e}") from e


# =========================
# NBA JUICE
# =========================

def lookup_nba_extra(price, venue):
    price = normalize_american_value(price)
    if price is None:
        return 0.0
    fav_ud = "favorite" if price < 0 else "underdog"
    rows = NBA_JT[
        (NBA_JT["venue"] == venue) &
        (NBA_JT["fav_ud"] == fav_ud) &
        (price >= NBA_JT["band_min"]) &
        (price <= NBA_JT["band_max"])
    ]
    if rows.empty:
        _log(f"no band match: price={price} venue={venue} fav_ud={fav_ud}", "WARN")
        return 0.0
    return float(rows.iloc[0]["extra_juice"])


def apply_nba(df):
    df = df.copy()

    df["home_extra_juice"] = df["home_dk_moneyline_american"].apply(lambda x: lookup_nba_extra(x, "home"))
    df["away_extra_juice"] = df["away_dk_moneyline_american"].apply(lambda x: lookup_nba_extra(x, "away"))

    df["home_juice_decimal_moneyline"] = (
        pd.to_numeric(df["home_acceptable_decimal_moneyline"], errors="coerce").apply(safe_decimal)
        * (1 + df["home_extra_juice"])
    )
    df["away_juice_decimal_moneyline"] = (
        pd.to_numeric(df["away_acceptable_decimal_moneyline"], errors="coerce").apply(safe_decimal)
        * (1 + df["away_extra_juice"])
    )

    df["home_juice_odds"] = df["home_juice_decimal_moneyline"].apply(decimal_to_american)
    df["away_juice_odds"] = df["away_juice_decimal_moneyline"].apply(decimal_to_american)

    df["home_acceptable_decimal_moneyline"]  = df["home_juice_decimal_moneyline"]
    df["away_acceptable_decimal_moneyline"]  = df["away_juice_decimal_moneyline"]
    df["home_acceptable_american_moneyline"] = df["home_juice_odds"]
    df["away_acceptable_american_moneyline"] = df["away_juice_odds"]

    return df


# =========================
# NCAAB JUICE
# =========================

def lookup_ncaab_extra(prob):
    if pd.isna(prob):
        return 0.0
    rows = NCAAB_JT[
        (prob >= NCAAB_JT["prob_bin_min"]) &
        (prob < NCAAB_JT["prob_bin_max"])
    ]
    if rows.empty:
        _log(f"no band match: prob={prob}", "WARN")
        return 0.0
    return float(rows.iloc[0]["extra_juice"])


def apply_ncaab(df):
    df = df.copy()

    df["home_extra_juice"] = df["home_prob"].apply(lookup_ncaab_extra)
    df["away_extra_juice"] = df["away_prob"].apply(lookup_ncaab_extra)

    base_home = df["home_acceptable_american_moneyline"].apply(american_to_decimal).apply(safe_decimal)
    base_away = df["away_acceptable_american_moneyline"].apply(american_to_decimal).apply(safe_decimal)

    df["home_juice_decimal_moneyline"] = base_home * (1 + df["home_extra_juice"])
    df["away_juice_decimal_moneyline"] = base_away * (1 + df["away_extra_juice"])

    df["home_juice_odds"] = df["home_juice_decimal_moneyline"].apply(decimal_to_american)
    df["away_juice_odds"] = df["away_juice_decimal_moneyline"].apply(decimal_to_american)

    df["home_acceptable_decimal_moneyline"]  = df["home_juice_decimal_moneyline"]
    df["away_acceptable_decimal_moneyline"]  = df["away_juice_decimal_moneyline"]
    df["home_acceptable_american_moneyline"] = df["home_juice_odds"]
    df["away_acceptable_american_moneyline"] = df["away_juice_odds"]

    return df


# =========================
# MAIN
# =========================

def main():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== apply_moneyline_juice RUN {_now()} ===\n")

    summary = {
        "files_found": 0, "files_written": 0,
        "nba_files": 0, "ncaab_files": 0,
        "total_rows": 0, "errors": 0,
    }
    per_file = []

    # clean old outputs
    for f in OUTPUT_DIR.glob("*_NBA_moneyline.csv"):
        f.unlink(missing_ok=True)
    for f in OUTPUT_DIR.glob("*_NCAAB_moneyline.csv"):
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

            if name.endswith("_NBA_moneyline.csv"):
                league = "NBA"
            elif name.endswith("_NCAAB_moneyline.csv"):
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
                _audit(f"JUICE_ML_{league}", "SUCCESS",
                       msg=f"Applied {league} Moneyline Juice to {name}", df=df)

            except Exception as e:
                _log(f"{name} FAILED: {e}\n{traceback.format_exc()}", "ERROR")
                _audit(f"JUICE_ML_{league}", "FAILED", msg=str(e))
                pf["status"] = "error"
                summary["errors"] += 1

            per_file.append(pf)

    except Exception as e:
        _log(f"FATAL: {e}\n{traceback.format_exc()}", "ERROR")
        _audit("JUICE_ML_CRITICAL", "FAILED", msg=str(e))
        _write_summary(summary, per_file)
        sys.exit(1)

    _write_summary(summary, per_file)
    print("apply_moneyline_juice complete.")


if __name__ == "__main__":
    main()
