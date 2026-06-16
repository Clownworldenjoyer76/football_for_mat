#!/usr/bin/env python3
# docs/win/final_scores/scripts/00_parsing/dedupe.py

import csv
from pathlib import Path
import traceback
import pandas as pd
from datetime import datetime
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

BASE_DIR = Path("docs/win/final_scores")
ERROR_DIR = BASE_DIR / "errors"
ERROR_DIR.mkdir(parents=True, exist_ok=True)
ERROR_LOG = ERROR_DIR / "dedupe.txt"

# Audit Log Location
AUDIT_LOG = Path("docs/win/final_scores/scripts/00_parsing/parsing_audit.txt")

def main():
    with open(ERROR_LOG, "w") as log:
        try:
            # CHANGE HERE: Use rglob to search all subfolders
            files = sorted(BASE_DIR.rglob("*_final_scores_*.csv"))

            if not files:
                log.write("No final score files found in subdirectories.\n")
                audit(AUDIT_LOG, "DEDUPE", "WARNING", msg="No final score files found.")
                return

            for path in files:
                with path.open("r", newline="", encoding="utf-8") as f:
                    reader = list(csv.DictReader(f))

                if not reader:
                    log.write(f"{path.relative_to(BASE_DIR)}: empty file\n")
                    continue

                seen = set()
                deduped = []

                for row in reader:
                    key = (
                        row.get("game_date"),
                        row.get("market"),
                        row.get("away_team"),
                        row.get("home_team"),
                    )

                    if key not in seen:
                        seen.add(key)
                        deduped.append(row)

                with path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=reader[0].keys())
                    writer.writeheader()
                    writer.writerows(deduped)

                # Log the relative path so you know which sport folder it was in
                msg = f"{path.relative_to(BASE_DIR)}: {len(reader)} -> {len(deduped)} rows"
                log.write(f"{msg}\n")
                
                # Audit call
                audit(AUDIT_LOG, "DEDUPE", "SUCCESS", msg=msg, df=pd.DataFrame(deduped))

        except Exception as e:
            err_msg = str(e)
            log.write("\n=== ERROR ===\n")
            log.write(err_msg + "\n\n")
            log.write(traceback.format_exc())
            audit(AUDIT_LOG, "DEDUPE", "ERROR", msg=err_msg)

if __name__ == "__main__":
    main()
