#!/usr/bin/env python3
# docs/win/basketball/scripts/01_merge/build_juice_files.py

import pandas as pd
import glob
import sys
import traceback
from pathlib import Path
from datetime import datetime, timezone
from scipy.stats import norm, poisson

# ============================================================
# SETTINGS
# ============================================================

NBA_EDGE        = 0.015
NBA_TOTAL_EDGE  = 0.025
NBA_SPREAD_EDGE = 0.035
NBA_TOTAL_STD   = 17.85
NBA_SPREAD_STD  = 17.5

NCAAB_EDGE        = 0.015
NCAAB_TOTAL_EDGE  = 0.03
NCAAB_SPREAD_EDGE = 0.03
NCAAB_TOTAL_STD   = 18.6662
NCAAB_SPREAD_STD  = 11.5

# ============================================================
# PATHS
# ============================================================

INPUT_DIR  = Path("docs/win/basketball/01_merge")
OUTPUT_DIR = Path("docs/win/basketball/01_merge/01_merguiced")
ERROR_DIR  = Path("docs/win/basketball/errors/01_merge")
LOG_FILE   = ERROR_DIR / "build_juice_files.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)

with open(LOG_FILE, "w") as f:
    f.write(f"=== build_juice_files RUN {datetime.now(timezone.utc).isoformat()} ===\n\n")

def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.now().isoformat()} | {msg}\n")

# ============================================================
# LOGGER UTILITY
# ============================================================

def audit(log_path, stage, status, msg="", df=None):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_path, "a") as f:
        f.write(f"\n[{ts}] [{stage}] {status}\n")
        if msg:
            f.write(f"MSG: {msg}\n")
        if df is not None:
            f.write(f"ROWS: {len(df)}\n")
        f.write("-" * 40 + "\n")

# ============================================================
# HELPERS
# ============================================================

def american_to_decimal(odds):
    if pd.isna(odds) or odds == "":
        return ""
    odds = float(odds)
    if odds > 0:
        return 1 + (odds / 100)
    return 1 + (100 / abs(odds))

def to_american(dec):
    if pd.isna(dec) or dec <= 1:
        return ""
    if dec >= 2:
        return f"+{int((dec - 1) * 100)}"
    return f"-{int(100 / (dec - 1))}"

def clamp_probability(p):
    return min(max(p, 0.05), 0.95)

def get_market_settings(market):
    if market == "NBA":
        return {
            "ML_EDGE":     NBA_EDGE,
            "TOTAL_EDGE":  NBA_TOTAL_EDGE,
            "SPREAD_EDGE": NBA_SPREAD_EDGE,
            "TOTAL_STD":   NBA_TOTAL_STD,
            "SPREAD_STD":  NBA_SPREAD_STD,
        }
    return {
        "ML_EDGE":     NCAAB_EDGE,
        "TOTAL_EDGE":  NCAAB_TOTAL_EDGE,
        "SPREAD_EDGE": NCAAB_SPREAD_EDGE,
        "TOTAL_STD":   NCAAB_TOTAL_STD,
        "SPREAD_STD":  NCAAB_SPREAD_STD,
    }

# ============================================================
# MAIN
# ============================================================

def main():
    files_written = []
    files_skipped = 0
    row_issues    = 0

    try:
        for f in OUTPUT_DIR.glob("*.csv"):
            f.unlink()

        input_files = sorted(glob.glob(str(INPUT_DIR / "basketball_*.csv")))
        log(f"Input files found: {len(input_files)}")

        if not input_files:
            log("No input files found.")
            log("STATUS: SUCCESS (nothing to do)")
            return

        for file_path in input_files:
            df = pd.read_csv(file_path)

            if df.empty:
                log(f"EMPTY: {file_path} — skipping")
                files_skipped += 1
                continue

            if "market" not in df.columns or "game_date" not in df.columns:
                log(f"MISSING COLUMNS: {file_path} — skipping")
                files_skipped += 1
                continue

            market = str(df["market"].iloc[0]).upper()
            game_date = df["game_date"].iloc[0]
            settings  = get_market_settings(market)

            ML_EDGE     = settings["ML_EDGE"]
            TOTAL_EDGE  = settings["TOTAL_EDGE"]
            SPREAD_EDGE = settings["SPREAD_EDGE"]
            TOTAL_STD   = settings["TOTAL_STD"]
            SPREAD_STD  = settings["SPREAD_STD"]

            # =====================================================
            # MONEYLINE
            # =====================================================

            ml_df = df.copy()
            ml_df["away_decimal"]      = ml_df["away_dk_moneyline_american"].apply(american_to_decimal)
            ml_df["home_decimal"]      = ml_df["home_dk_moneyline_american"].apply(american_to_decimal)
            ml_df["away_implied_prob"] = 1 / ml_df["away_decimal"]
            ml_df["home_implied_prob"] = 1 / ml_df["home_decimal"]
            total_implied              = ml_df["away_implied_prob"] + ml_df["home_implied_prob"]
            ml_df["away_market_prob"]  = ml_df["away_implied_prob"] / total_implied
            ml_df["home_market_prob"]  = ml_df["home_implied_prob"] / total_implied
            ml_df["away_fair"]         = 1 / ml_df["away_prob"]
            ml_df["home_fair"]         = 1 / ml_df["home_prob"]
            ml_df["away_acceptable_decimal_moneyline"]  = ml_df["away_fair"] * (1 + ML_EDGE)
            ml_df["home_acceptable_decimal_moneyline"]  = ml_df["home_fair"] * (1 + ML_EDGE)
            ml_df["away_acceptable_american_moneyline"] = ml_df["away_acceptable_decimal_moneyline"].apply(to_american)
            ml_df["home_acceptable_american_moneyline"] = ml_df["home_acceptable_decimal_moneyline"].apply(to_american)

            ml_out = OUTPUT_DIR / f"{game_date}_{market}_moneyline.csv"
            ml_df.to_csv(ml_out, index=False)
            files_written.append((str(ml_out), len(ml_df)))
            log(f"WROTE {ml_out} ({len(ml_df)} rows)")
            audit(LOG_FILE, "ML", "SUCCESS", file_path, ml_df)

            # =====================================================
            # TOTALS
            # =====================================================

            total_df   = df.copy()
            fair_over  = []
            fair_under = []
            acc_over   = []
            acc_under  = []

            for _, row in total_df.iterrows():
                T    = row["total"]
                mean = row["total_projected_points"]

                if pd.isna(T):
                    fair_over.append("")
                    fair_under.append("")
                    acc_over.append("")
                    acc_under.append("")
                    row_issues += 1
                    continue

                if market == "NCAAB":
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

            tot_out = OUTPUT_DIR / f"{game_date}_{market}_total.csv"
            total_df.to_csv(tot_out, index=False)
            files_written.append((str(tot_out), len(total_df)))
            log(f"WROTE {tot_out} ({len(total_df)} rows)")
            audit(LOG_FILE, "TOTAL", "SUCCESS", file_path, total_df)

            # =====================================================
            # SPREAD
            # =====================================================

            spread_df = df.copy()
            fair_home = []
            fair_away = []
            acc_home  = []
            acc_away  = []

            for _, row in spread_df.iterrows():
                mean_margin = row["home_projected_points"] - row["away_projected_points"]
                try:
                    home_line = float(row["home_spread"])
                except:
                    fair_home.append("")
                    fair_away.append("")
                    acc_home.append("")
                    acc_away.append("")
                    row_issues += 1
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

            spread_out = OUTPUT_DIR / f"{game_date}_{market}_spread.csv"
            spread_df.to_csv(spread_out, index=False)
            files_written.append((str(spread_out), len(spread_df)))
            log(f"WROTE {spread_out} ({len(spread_df)} rows)")
            audit(LOG_FILE, "SPREAD", "SUCCESS", file_path, spread_df)

        log("--- SUMMARY ---")
        log(f"Input files: {len(input_files)}")
        log(f"Files skipped: {files_skipped}")
        log(f"Row issues: {row_issues}")
        log(f"Files written: {len(files_written)}")
        for path, count in files_written:
            log(f"  FILE: {path} ({count} rows)")
        log("STATUS: SUCCESS")
        print("Build juice files complete")

    except Exception as e:
        log(f"FATAL ERROR: {e}\n{traceback.format_exc()}")
        log("STATUS: FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
