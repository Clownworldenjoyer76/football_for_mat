#!/usr/bin/env python3
# docs/win/final_scores/scripts/00_parsing/dk_puck_merge.py

import pandas as pd
from pathlib import Path
from datetime import datetime
import traceback
import sys

# =========================
# LOGGER UTILITY
# =========================

def audit(log_path, stage, status, msg="", df=None):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. EXHAUSTIVE LOG (TXT)
    log_mode = "w" if not log_path.exists() else "a"

    with open(log_path, log_mode) as f:
        f.write(f"\n[{ts}] [{stage}] {status}\n")
        if msg:
            f.write(f"  MSG: {msg}\n")
        if df is not None and isinstance(df, pd.DataFrame):
            f.write(f"  STATS: {len(df)} rows | {len(df.columns)} cols\n")
            f.write(f"  NULLS: {df.isnull().sum().sum()} total\n")
            f.write(f"  SAMPLE:\n{df.head(3).to_string(index=False)}\n")
        f.write("-" * 40 + "\n")

    # 2. CONDENSED SUMMARY (TXT)
    if df is not None and isinstance(df, pd.DataFrame):
        summary_path = log_path.parent / "condensed_summary.txt"

        # Identify active plays based on your headers
        play_cols = [c for c in ['home_play', 'away_play', 'over_play', 'under_play'] if c in df.columns]

        if play_cols:
            signals = df[df[play_cols].any(axis=1)].copy()

            if not signals.empty:
                summary_mode = "w" if not summary_path.exists() else "a"

                with open(summary_path, summary_mode) as f:
                    f.write(f"\n--- BETTING SIGNALS: {ts} ---\n")

                    # Filter identifying columns and edge columns
                    base_cols = ['game_date', 'home_team', 'away_team']
                    edge_cols = [c for c in df.columns if 'edge_pct' in c]

                    final_cols = [c for c in base_cols + edge_cols if c in signals.columns]

                    f.write(signals[final_cols].to_string(index=False))
                    f.write("\n" + "=" * 30 + "\n")

# =========================
# ORIGINAL SCRIPT
# =========================

# Audit Log Location
AUDIT_LOG = Path("docs/win/final_scores/scripts/00_parsing/dk_puck_audit.txt")

# =========================
# PATHS
# =========================

SPORTSBOOK_DIR = Path("docs/win/hockey/00_intake/sportsbook")
# UPDATED: Pointing to the new NHL-specific score directory
FINAL_DIR = Path("docs/win/final_scores/results/nhl/final_scores")
ERROR_DIR = Path("docs/win/final_scores/errors")
LOG_FILE = ERROR_DIR / "dk_puck_log.txt"

ERROR_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# LOGGING
# =========================

def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {message}\n")


log("========== DK PUCK SCRIPT START ==========")

try:
    sportsbook_files = sorted(SPORTSBOOK_DIR.glob("hockey_*.csv"))

    if not sportsbook_files:
        log("No sportsbook files found.")
    else:
        log(f"Found {len(sportsbook_files)} sportsbook files.")

    for sb_file in sportsbook_files:
        try:
            # Extract date from filename (e.g., hockey_2026_03_01.csv -> 2026_03_01)
            date_part = sb_file.stem.replace("hockey_", "")
            final_file = FINAL_DIR / f"{date_part}_final_scores_NHL.csv"

            log(f"Processing sportsbook file: {sb_file.name}")
            log(f"Looking for final file: {final_file.name}")

            if not final_file.exists():
                log(f"Final file {final_file.name} does NOT exist in {FINAL_DIR}. Skipping.")
                continue

            sportsbook_df = pd.read_csv(sb_file)
            final_df = pd.read_csv(final_file)

            log(f"Sportsbook rows: {len(sportsbook_df)}")
            log(f"Final rows: {len(final_df)}")

            # Validate required columns
            required_cols = ["game_date", "away_team", "home_team",
                             "away_puck_line", "home_puck_line", "total"]

            for col in required_cols:
                if col not in sportsbook_df.columns:
                    log(f"Missing column in sportsbook: {col}. Skipping file.")
                    raise ValueError(f"Missing sportsbook column {col}")

            merge_cols = ["game_date", "away_team", "home_team"]

            for col in merge_cols:
                if col not in final_df.columns:
                    log(f"Missing column in final file: {col}. Skipping file.")
                    raise ValueError(f"Missing final column {col}")

            # Rename DK columns to distinguish them from final results
            sportsbook_df = sportsbook_df.rename(columns={
                "away_puck_line": "dk_away_puck_line",
                "home_puck_line": "dk_home_puck_line",
                "total": "dk_total"
            })

            sportsbook_subset = sportsbook_df[
                merge_cols + [
                    "dk_away_puck_line",
                    "dk_home_puck_line",
                    "dk_total"
                ]
            ]

            # Left merge adds DK info to your existing score file
            merged_df = final_df.merge(
                sportsbook_subset,
                on=merge_cols,
                how="left"
            )

            # Log match diagnostics
            dk_nulls = merged_df["dk_total"].isna().sum()
            log(f"Rows with missing DK match: {dk_nulls}")

            # Overwrite the original score file with the new merged data
            merged_df.to_csv(final_file, index=False)
            
            # Audit successful update
            audit(AUDIT_LOG, "DK_MERGE", "SUCCESS", msg=f"Merged {sb_file.name} into {final_file.name}", df=merged_df)

            log(f"Successfully updated {final_file.name}")

        except Exception as file_error:
            msg = f"ERROR processing {sb_file.name}: {str(file_error)}"
            log(msg)
            log(traceback.format_exc())
            audit(AUDIT_LOG, "DK_MERGE", "ERROR", msg=msg)
            continue

except Exception as e:
    msg = f"FATAL ERROR IN DK PUCK SCRIPT: {str(e)}"
    log(msg)
    log(traceback.format_exc())
    audit(AUDIT_LOG, "DK_MERGE", "FATAL", msg=msg)

log("========== DK PUCK SCRIPT END ==========\n")
