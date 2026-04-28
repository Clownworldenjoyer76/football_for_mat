#!/usr/bin/env python3
# docs/win/basketball/scripts/01_merge/build_juice_files.py

import csv
import traceback
import sys
from pathlib import Path
from datetime import datetime, timezone
from scipy.stats import norm, poisson
import pandas as pd

# ============================================================
# SETTINGS
# ============================================================

LEAGUE_SETTINGS = {
    "NBA": {
        "ML_EDGE":     0.015,
        "TOTAL_EDGE":  0.025,
        "SPREAD_EDGE": 0.035,
        "TOTAL_STD":   17.85,
        "SPREAD_STD":  17.5,
    },
    "NCAAM": {
        "ML_EDGE":     0.015,
        "TOTAL_EDGE":  0.03,
        "SPREAD_EDGE": 0.03,
        "TOTAL_STD":   18.6662,
        "SPREAD_STD":  11.5,
    },
    "WNBA": {
        "ML_EDGE":     0.020,
        "TOTAL_EDGE":  0.030,
        "SPREAD_EDGE": 0.040,
        "TOTAL_STD":   16.5,
        "SPREAD_STD":  16.0,
    },
}

LEAGUES = ["nba", "ncaam", "wnba"]

# ============================================================
# PATHS
# ============================================================

INPUT_DIR  = Path("docs/win/basketball/01_merge")
OUTPUT_DIR = Path("docs/win/basketball/01_merge/01_merguiced")
ERROR_DIR  = Path("docs/win/basketball/errors/01_merge")
LOG_FILE   = ERROR_DIR / "build_juice_files.txt"

ERROR_DIR.mkdir(parents=True, exist_ok=True)

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write(f"=== build_juice_files RUN {datetime.now(timezone.utc).isoformat()} ===\n\n")


def log(msg: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} | {msg}\n")


def audit(stage, status, msg="", df=None):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n[{ts}] [{stage}] {status}\n")
        if msg:
            f.write(f"  MSG: {msg}\n")
        if df is not None:
            f.write(f"  ROWS: {len(df)}\n")
        f.write("-" * 40 + "\n")


# ============================================================
# HELPERS
# ============================================================

def american_to_decimal(odds):
    if pd.isna(odds) or str(odds).strip() == "":
        return ""
    try:
        odds = float(odds)
    except (ValueError, TypeError):
        return ""
    if odds > 0:
        return 1 + (odds / 100)
    return 1 + (100 / abs(odds))


def to_american(dec):
    if dec == "" or pd.isna(dec) or float(dec) <= 1:
        return ""
    dec = float(dec)
    if dec >= 2:
        return f"+{int((dec - 1) * 100)}"
    return f"-{int(100 / (dec - 1))}"


def clamp_probability(p):
    return min(max(p, 0.05), 0.95)


def wipe_outputs():
    for league in LEAGUES:
        for subdir in ["moneyline", "spread", "total"]:
            folder = OUTPUT_DIR / league / subdir
            folder.mkdir(parents=True, exist_ok=True)
            for f in folder.glob("*.csv"):
                f.unlink(missing_ok=True)
    log("Wiped all output folders.")


# ============================================================
# PROCESS MONEYLINE
# ============================================================

def process_moneyline(df: pd.DataFrame, date: str, league_upper: str, settings: dict, league: str) -> tuple:
    ML_EDGE = settings["ML_EDGE"]

    ml_df = df.copy()
    ml_df["away_decimal"]      = ml_df["away_dk_moneyline_american"].apply(american_to_decimal)
    ml_df["home_decimal"]      = ml_df["home_dk_moneyline_american"].apply(american_to_decimal)
    ml_df["away_implied_prob"] = ml_df["away_decimal"].apply(lambda x: 1 / x if x != "" and float(x) > 0 else "")
    ml_df["home_implied_prob"] = ml_df["home_decimal"].apply(lambda x: 1 / x if x != "" and float(x) > 0 else "")

    total_implied = ml_df["away_implied_prob"].apply(lambda x: float(x) if x != "" else 0) + \
                    ml_df["home_implied_prob"].apply(lambda x: float(x) if x != "" else 0)

    ml_df["away_market_prob"] = ml_df.apply(
        lambda r: float(r["away_implied_prob"]) / total_implied[r.name] if r["away_implied_prob"] != "" and total_implied[r.name] > 0 else "", axis=1
    )
    ml_df["home_market_prob"] = ml_df.apply(
        lambda r: float(r["home_implied_prob"]) / total_implied[r.name] if r["home_implied_prob"] != "" and total_implied[r.name] > 0 else "", axis=1
    )

    ml_df["away_fair"] = ml_df["away_prob"].apply(lambda x: 1 / float(x) if str(x).strip() != "" and float(x) > 0 else "")
    ml_df["home_fair"] = ml_df["home_prob"].apply(lambda x: 1 / float(x) if str(x).strip() != "" and float(x) > 0 else "")

    ml_df["away_acceptable_decimal_moneyline"]  = ml_df["away_fair"].apply(lambda x: float(x) * (1 + ML_EDGE) if x != "" else "")
    ml_df["home_acceptable_decimal_moneyline"]  = ml_df["home_fair"].apply(lambda x: float(x) * (1 + ML_EDGE) if x != "" else "")
    ml_df["away_acceptable_american_moneyline"] = ml_df["away_acceptable_decimal_moneyline"].apply(to_american)
    ml_df["home_acceptable_american_moneyline"] = ml_df["home_acceptable_decimal_moneyline"].apply(to_american)

    out_path = OUTPUT_DIR / league / "moneyline" / f"{date}_{league_upper}_moneyline.csv"
    ml_df.to_csv(out_path, index=False)
    return out_path, len(ml_df)


# ============================================================
# PROCESS TOTALS
# ============================================================

def process_totals(df: pd.DataFrame, date: str, league_upper: str, settings: dict, league: str) -> tuple:
    TOTAL_EDGE = settings["TOTAL_EDGE"]
    TOTAL_STD  = settings["TOTAL_STD"]

    total_df   = df.copy()
    fair_over  = []
    fair_under = []
    acc_over   = []
    acc_under  = []

    for _, row in total_df.iterrows():
        try:
            T    = float(row["total"])
            mean = float(row["total_projected_points"])
        except (ValueError, TypeError):
            fair_over.append("")
            fair_under.append("")
            acc_over.append("")
            acc_under.append("")
            continue

        if league_upper == "NCAAM":
            p_under = poisson.cdf(T - 0.5, mean)
        else:
            z       = (T - mean) / TOTAL_STD
            p_under = norm.cdf(z)

        p_under        = clamp_probability(p_under)
        p_over         = 1 - p_under
        fair_under_dec = 1 / p_under
        fair_over_dec  = 1 / p_over
        fair_under.append(fair_under_dec)
        fair_over.append(fair_over_dec)
        acc_under.append(fair_under_dec * (1 + TOTAL_EDGE))
        acc_over.append(fair_over_dec  * (1 + TOTAL_EDGE))

    total_df["fair_over"]        = fair_over
    total_df["fair_under"]       = fair_under
    total_df["acceptable_over"]  = acc_over
    total_df["acceptable_under"] = acc_under

    out_path = OUTPUT_DIR / league / "total" / f"{date}_{league_upper}_total.csv"
    total_df.to_csv(out_path, index=False)
    return out_path, len(total_df)


# ============================================================
# PROCESS SPREAD
# ============================================================

def process_spread(df: pd.DataFrame, date: str, league_upper: str, settings: dict, league: str) -> tuple:
    SPREAD_EDGE = settings["SPREAD_EDGE"]
    SPREAD_STD  = settings["SPREAD_STD"]

    spread_df = df.copy()
    fair_home = []
    fair_away = []
    acc_home  = []
    acc_away  = []

    for _, row in spread_df.iterrows():
        try:
            mean_margin = float(row["home_projected_points"]) - float(row["away_projected_points"])
            home_line   = float(row["home_dk_spread_american"])
        except (ValueError, TypeError):
            fair_home.append("")
            fair_away.append("")
            acc_home.append("")
            acc_away.append("")
            continue

        p_home        = 1 - norm.cdf(home_line, loc=mean_margin, scale=SPREAD_STD)
        p_home        = clamp_probability(p_home)
        p_away        = 1 - p_home
        fair_home_dec = 1 / p_home
        fair_away_dec = 1 / p_away
        fair_home.append(fair_home_dec)
        fair_away.append(fair_away_dec)
        acc_home.append(fair_home_dec * (1 + SPREAD_EDGE))
        acc_away.append(fair_away_dec * (1 + SPREAD_EDGE))

    spread_df["fair_home_spread_decimal"]        = fair_home
    spread_df["fair_away_spread_decimal"]        = fair_away
    spread_df["home_acceptable_spread_decimal"]  = acc_home
    spread_df["away_acceptable_spread_decimal"]  = acc_away
    spread_df["home_acceptable_spread_american"] = spread_df["home_acceptable_spread_decimal"].apply(to_american)
    spread_df["away_acceptable_spread_american"] = spread_df["away_acceptable_spread_decimal"].apply(to_american)

    out_path = OUTPUT_DIR / league / "spread" / f"{date}_{league_upper}_spread.csv"
    spread_df.to_csv(out_path, index=False)
    return out_path, len(spread_df)


# ============================================================
# MAIN
# ============================================================

def main():
    files_written = []
    files_skipped = 0

    try:
        wipe_outputs()

        for league in LEAGUES:
            league_upper = league.upper()
            settings     = LEAGUE_SETTINGS[league_upper]

            for market_type in ["moneyline", "spread", "total"]:
                input_folder = INPUT_DIR / league / market_type
                if not input_folder.exists():
                    log(f"INPUT FOLDER NOT FOUND: {input_folder}")
                    continue

                input_files = sorted(input_folder.glob(f"*_{league_upper}_{market_type}.csv"))

                if not input_files:
                    log(f"NO INPUT FILES: {input_folder}")
                    continue

                for file_path in input_files:
                    try:
                        df = pd.read_csv(file_path)

                        if df.empty:
                            log(f"EMPTY: {file_path.name} — skipping")
                            files_skipped += 1
                            continue

                        # Parse date from filename: {date}_{LEAGUE}_{type}.csv
                        date = file_path.stem.replace(f"_{league_upper}_{market_type}", "")

                        if market_type == "moneyline":
                            out_path, count = process_moneyline(df, date, league_upper, settings, league)
                        elif market_type == "total":
                            out_path, count = process_totals(df, date, league_upper, settings, league)
                        elif market_type == "spread":
                            out_path, count = process_spread(df, date, league_upper, settings, league)

                        files_written.append((str(out_path), count))
                        log(f"WROTE {out_path.name} ({count} rows)")
                        audit(market_type.upper(), "SUCCESS", file_path.name, df)

                    except Exception as e:
                        log(f"ERROR processing {file_path.name}: {e}\n{traceback.format_exc()}")
                        files_skipped += 1

        log("--- SUMMARY ---")
        log(f"Files written: {len(files_written)}")
        log(f"Files skipped: {files_skipped}")
        for path, count in files_written:
            log(f"  FILE: {path} ({count} rows)")
        log("STATUS: SUCCESS")
        print("build_juice_files complete.")

    except Exception as e:
        log(f"FATAL ERROR: {e}\n{traceback.format_exc()}")
        log("STATUS: FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
