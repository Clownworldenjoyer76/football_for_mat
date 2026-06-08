#!/usr/bin/env python3
# docs/win/final_scores/scripts/05_results/hockey/nhl_results_analyze.py

from datetime import datetime, UTC
from pathlib import Path
import pandas as pd

###############################################################
######################## PATH CONFIG ##########################
###############################################################

NHL_OUTPUT = Path("docs/win/final_scores/results/nhl/graded")

INTERMEDIATE_DIR = Path("docs/win/final_scores/intermediate")
INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

ERROR_DIR = Path("docs/win/final_scores/errors")
ERROR_DIR.mkdir(parents=True, exist_ok=True)

ERROR_LOG = ERROR_DIR / "nhl_results_analyze_errors.txt"
SUMMARY_LOG = ERROR_DIR / "nhl_results_analyze_summary.txt"

###############################################################
######################## LOGGING ##############################
###############################################################

def reset_logs():
    ERROR_LOG.write_text("", encoding="utf-8")
    SUMMARY_LOG.write_text("", encoding="utf-8")


def log_error(msg):
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now(UTC).isoformat()}] {msg}\n")


def log_summary(msg):
    with open(SUMMARY_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now(UTC).isoformat()}] {msg}\n")

###############################################################
######################## HELPERS ##############################
###############################################################

def safe_read(path):
    try:
        path = Path(path)

        if not path.exists():
            log_error(f"MISSING FILE | {path}")
            return pd.DataFrame()

        df = pd.read_csv(path)

        if df.empty:
            log_error(f"EMPTY FILE | {path}")
            return pd.DataFrame()

        return df

    except Exception as e:
        log_error(f"READ ERROR | {path} | {e}")
        return pd.DataFrame()

###############################################################
######################## WORK FILE ############################
###############################################################

def build_work():

    path = NHL_OUTPUT / "NHL_final.csv"

    df = safe_read(path)

    if df.empty:
        log_error("MASTER FILE EMPTY OR MISSING")
        return

    # =========================================================
    # 🔥 CRITICAL FIX: ALIGN WITH NEW PIPELINE
    # =========================================================

    # EV is now the edge signal
    if "ev" not in df.columns:
        log_error("MISSING COLUMN | ev")
        return

    if "dk_odds_american" not in df.columns:
        log_error("MISSING COLUMN | dk_odds_american")
        return

    df["selected_edge"] = df["ev"]
    df["take_odds"] = df["dk_odds_american"]

    # =========================================================
    # CLEANUP (OPTIONAL BUT SAFE)
    # =========================================================

    # unify league column
    if "league_x" in df.columns:
        df.rename(columns={"league_x": "league"}, inplace=True)

    if "league_y" in df.columns:
        df.drop(columns=["league_y"], inplace=True, errors="ignore")

    # =========================================================

    out = INTERMEDIATE_DIR / "work_nhl.csv"
    df.to_csv(out, index=False)

    log_summary(f"NHL WORK FILE CREATED | ROWS={len(df)} | OUT={out}")

###############################################################
######################## MAIN #################################
###############################################################

def main():

    reset_logs()

    log_summary("START nhl_results_analyze.py")

    build_work()

    log_summary("END nhl_results_analyze.py")

    print("NHL analysis prep complete.")


if __name__ == "__main__":
    main()