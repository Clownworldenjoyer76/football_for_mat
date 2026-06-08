#!/usr/bin/env python3
# docs/win/final_scores/scripts/05_results/soccer/02_soccer_results_analyze.py

from datetime import datetime
from pathlib import Path
import pandas as pd

OUTPUT_DIR       = Path("docs/win/final_scores/results/soccer/graded")
INTERMEDIATE_DIR = Path("docs/win/final_scores/intermediate")
ERROR_DIR        = Path("docs/win/final_scores/errors")

INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)

ERROR_LOG   = ERROR_DIR / "soccer_results_analyze_errors.txt"
SUMMARY_LOG = ERROR_DIR / "soccer_results_analyze_summary.txt"


# =========================
# LOGGING
# =========================

def reset_logs():
    ERROR_LOG.write_text("", encoding="utf-8")
    SUMMARY_LOG.write_text("", encoding="utf-8")


def log_error(msg):
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] {msg}\n")


def log_summary(msg):
    with open(SUMMARY_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] {msg}\n")


# =========================
# BUCKETS
# =========================

def edge_bucket(v):
    if pd.isna(v):   return "no_edge"
    if v < 0.01:     return "0_to_0.01"
    if v < 0.02:     return "0.01_to_0.02"
    if v < 0.03:     return "0.02_to_0.03"
    if v < 0.05:     return "0.03_to_0.05"
    return "0.05_plus"


def odds_bucket(v):
    if pd.isna(v):   return ""
    if v < -150:     return "minus_150_or_lower"
    if v < -110:     return "minus_149_to_minus_110"
    if v < 100:      return "minus_109_to_plus_100"
    if v < 150:      return "plus_101_to_plus_150"
    return "plus_151_or_higher"


# =========================
# PREPARE
# =========================

def prepare():
    path = OUTPUT_DIR / "SOCCER_final.csv"

    if not path.exists():
        log_error("MASTER FILE MISSING")
        return

    df = pd.read_csv(path)

    if df.empty:
        log_error("MASTER FILE EMPTY")
        return

    df["market"] = "SOCCER"

    # market_type comes from the select file column (btts / total25 / total35 / match_odds)
    if "market_type" not in df.columns:
        df["market_type"] = ""

    # market_scorefile comes from the grade step — league name from score file
    if "market_scorefile" not in df.columns:
        df["market_scorefile"] = ""

    # odds_american and edge_pct mapped from select file odds/ev columns in grade step
    df["selected_edge"] = pd.to_numeric(df.get("edge_pct"), errors="coerce")
    df["selected_odds"] = pd.to_numeric(df.get("odds_american"), errors="coerce")

    df["edge_bucket"] = df["selected_edge"].apply(edge_bucket)
    df["odds_bucket"] = df["selected_odds"].apply(odds_bucket)

    out = INTERMEDIATE_DIR / "work_soccer.csv"
    df.to_csv(out, index=False)

    log_summary(f"WORK FILE CREATED | {out} | ROWS={len(df)}")
    log_summary(f"  market_types: {df['market_type'].value_counts().to_dict()}")
    log_summary(f"  results:      {df['bet_result'].value_counts().to_dict()}")


# =========================
# MAIN
# =========================

def main():
    reset_logs()
    log_summary(f"=== START 02_soccer_results_analyze.py {datetime.now().isoformat()} ===")
    prepare()
    log_summary(f"=== END 02_soccer_results_analyze.py {datetime.now().isoformat()} ===")
    print("Soccer analytics preparation complete.")


if __name__ == "__main__":
    main()
