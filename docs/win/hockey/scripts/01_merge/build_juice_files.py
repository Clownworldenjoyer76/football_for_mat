#!/usr/bin/env python3
# docs/win/hockey/scripts/01_merge/build_juice_files.py

import pandas as pd
import glob
import sys
import traceback
import math
from pathlib import Path
from datetime import datetime, UTC
from scipy.stats import skellam, poisson

INPUT_DIR  = Path("docs/win/hockey/01_merge")
OUTPUT_DIR = INPUT_DIR / "01_merguiced"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ERROR_DIR = Path("docs/win/hockey/errors/01_merge")
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ERROR_DIR / "build_juice_files.txt"

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write(f"=== build_juice_files RUN {datetime.now(UTC).isoformat()} ===\n")

REQUIRED_COLUMNS = [
    "league", "market", "game_date", "game_time",
    "home_team", "away_team", "game_id",
    "home_prob", "away_prob",
    "away_projected_goals", "home_projected_goals", "total_projected_goals",
    "away_puck_line", "home_puck_line", "total",
    "away_dk_puck_line_american", "home_dk_puck_line_american",
    "dk_total_over_american", "dk_total_under_american",
    "away_dk_moneyline_american", "home_dk_moneyline_american",
]


def log(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now(UTC).isoformat()} | {msg}\n")


def american_to_decimal(odds):
    try:
        if pd.isna(odds):
            return None
        odds = float(odds)
        if odds == 0:
            return None
        if odds > 0:
            return 1 + (odds / 100)
        else:
            return 1 + (100 / abs(odds))
    except Exception:
        return None


def poisson_cdf(k, lam):
    return poisson.cdf(k, lam)


def main():
    files_written = []
    schema_errors = 0
    row_issues    = 0
    empty_files   = 0

    for f in OUTPUT_DIR.glob("*.csv"):
        f.unlink()
    
    try:
        files = glob.glob(str(INPUT_DIR / "hockey_NHL_*.csv"))
        log(f"Input files found: {len(files)}")

        for file_path in files:
            df = pd.read_csv(file_path)

            if df.empty:
                log(f"EMPTY: {file_path} — skipping")
                empty_files += 1
                continue

            missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
            if missing_cols:
                log(f"SCHEMA ERROR: {file_path} missing columns: {missing_cols}")
                schema_errors += 1
                continue

            num_cols = [
                "home_prob", "away_prob",
                "total_projected_goals", "home_projected_goals", "away_projected_goals",
                "total", "home_puck_line", "away_puck_line",
            ]
            for col in num_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            for i, r in df.iterrows():
                if pd.isna(r["home_prob"]) or pd.isna(r["away_prob"]):
                    log(f"ROW ISSUE: {file_path} idx={i} bad probs")
                    row_issues += 1

            stem       = Path(file_path).stem
            slate_date = stem.replace("hockey_NHL_", "")
            market     = df["market"].iloc[0]

            # =========================
            # MONEYLINE
            # =========================

            ml = df.copy()
            ml["away_dk_decimal_moneyline"]  = ml["away_dk_moneyline_american"].apply(american_to_decimal)
            ml["home_dk_decimal_moneyline"]  = ml["home_dk_moneyline_american"].apply(american_to_decimal)
            ml["away_fair_decimal_moneyline"] = ml["away_prob"].apply(
                lambda x: 1 / x if pd.notna(x) and x > 0 else None
            )
            ml["home_fair_decimal_moneyline"] = ml["home_prob"].apply(
                lambda x: 1 / x if pd.notna(x) and x > 0 else None
            )
            ml_path = OUTPUT_DIR / f"{slate_date}_NHL_moneyline.csv"
            ml.to_csv(ml_path, index=False)
            files_written.append((str(ml_path), len(ml)))
            log(f"WROTE {ml_path} ({len(ml)} rows)")

            # =========================
            # TOTAL
            # =========================

            tot = df.copy()
            tot["dk_total_over_decimal"]  = tot["dk_total_over_american"].apply(american_to_decimal)
            tot["dk_total_under_decimal"] = tot["dk_total_under_american"].apply(american_to_decimal)

            over  = []
            under = []

            for i, r in tot.iterrows():
                lam = r["total_projected_goals"]
                T   = r["total"]

                if pd.isna(lam) or pd.isna(T) or lam <= 0:
                    log(f"ROW ISSUE: {file_path} idx={i} bad total inputs")
                    row_issues += 1
                    over.append(None)
                    under.append(None)
                    continue

                if T % 1 == 0:
                    log(f"WHOLE NUMBER TOTAL: {file_path} idx={i} total={T} — push not modelled; skipping row")
                    row_issues += 1
                    over.append(None)
                    under.append(None)
                    continue

                k       = math.floor(T)
                p_under = poisson_cdf(k, lam)
                p_over  = 1 - p_under

                under.append(1 / p_under if p_under > 0 else None)
                over.append(1 / p_over   if p_over  > 0 else None)

            tot["fair_total_over_decimal"]  = over
            tot["fair_total_under_decimal"] = under
            tot_path = OUTPUT_DIR / f"{slate_date}_NHL_total.csv"
            tot.to_csv(tot_path, index=False)
            files_written.append((str(tot_path), len(tot)))
            log(f"WROTE {tot_path} ({len(tot)} rows)")

            # =========================
            # PUCK LINE
            # =========================

            pl = df.copy()
            pl["home_dk_puck_line_decimal"] = pl["home_dk_puck_line_american"].apply(american_to_decimal)
            pl["away_dk_puck_line_decimal"] = pl["away_dk_puck_line_american"].apply(american_to_decimal)

            home_vals  = []
            away_vals  = []
            home_probs = []
            away_probs = []

            for i, r in pl.iterrows():
                lambda_home = r["home_projected_goals"]
                lambda_away = r["away_projected_goals"]

                if pd.isna(lambda_home) or pd.isna(lambda_away) or lambda_home <= 0 or lambda_away <= 0:
                    log(f"ROW ISSUE: {file_path} idx={i} puck invalid lambdas")
                    row_issues += 1
                    home_vals.append(None)
                    away_vals.append(None)
                    home_probs.append(None)
                    away_probs.append(None)
                    continue

                home_line = r["home_puck_line"]
                away_line = r["away_puck_line"]

                if home_line == -1.5:
                    p_home = 1 - skellam.cdf(1, lambda_home, lambda_away)
                    p_away = 1 - p_home
                elif away_line == -1.5:
                    p_away = 1 - skellam.cdf(1, lambda_away, lambda_home)
                    p_home = 1 - p_away
                else:
                    log(f"ROW ISSUE: {file_path} idx={i} unexpected puck lines: home={home_line} away={away_line}")
                    row_issues += 1
                    home_vals.append(None)
                    away_vals.append(None)
                    home_probs.append(None)
                    away_probs.append(None)
                    continue

                p_home = min(max(p_home, 0.01), 0.99)
                p_away = min(max(p_away, 0.01), 0.99)

                home_probs.append(p_home)
                away_probs.append(p_away)
                home_vals.append(1 / p_home)
                away_vals.append(1 / p_away)

            pl["home_fair_puck_line_decimal"] = home_vals
            pl["away_fair_puck_line_decimal"] = away_vals
            pl["home_prob_puck_line"]         = home_probs
            pl["away_prob_puck_line"]         = away_probs

            pl_path = OUTPUT_DIR / f"{slate_date}_NHL_puck_line.csv"
            pl.to_csv(pl_path, index=False)
            files_written.append((str(pl_path), len(pl)))
            log(f"WROTE {pl_path} ({len(pl)} rows)")

        log("--- SUMMARY ---")
        log(f"Input files processed: {len(files)}")
        log(f"Empty files skipped: {empty_files}")
        log(f"Schema errors: {schema_errors}")
        log(f"Row issues: {row_issues}")
        log(f"Files written: {len(files_written)}")
        for path, count in files_written:
            log(f"  FILE: {path} ({count} rows)")
        log("STATUS: SUCCESS")

    except Exception as e:
        log(f"FATAL ERROR: {e}\n{traceback.format_exc()}")
        log("STATUS: FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
