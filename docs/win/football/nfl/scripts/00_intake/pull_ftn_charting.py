"""
pull_ftn_charting.py

Pulls FTN play-charting data (published free via nflverse) for the given
NFL season(s) and writes one CSV per season.

Source: nflreadpy.load_ftn_charting() (primary)
Data available from 2022 onward (FTN/nflverse limitation).

Output:
    docs/win/football/nfl/00_intake/qb/ftn/{season}_ftn_charting.csv

Error/run log:
    docs/win/football/nfl/errors/00_intake/pull_ftn_charting.txt

Usage:
    python pull_ftn_charting.py --seasons 2022 2023 2024 2025
"""

import argparse
import os
import sys
import traceback
from datetime import datetime, timezone

OUTPUT_DIR = "docs/win/football/nfl/00_intake/qb/ftn"
ERROR_LOG_PATH = "docs/win/football/nfl/errors/00_intake/pull_ftn_charting.txt"


def log(lines):
    os.makedirs(os.path.dirname(ERROR_LOG_PATH), exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    with open(ERROR_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"--- run {timestamp} ---\n")
        for line in lines:
            f.write(line + "\n")
        f.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        required=True,
        help="One or more NFL seasons to pull (2022 or later).",
    )
    args = parser.parse_args()

    invalid_seasons = [s for s in args.seasons if s < 2022]
    if invalid_seasons:
        msg = f"ERROR: FTN charting data is not available before 2022. Invalid seasons requested: {invalid_seasons}"
        print(msg)
        log([msg])
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        import nflreadpy as nfl
    except ImportError as e:
        msg = f"ERROR: nflreadpy is not installed. {e}"
        print(msg)
        log([msg])
        sys.exit(1)

    log_lines = [f"Requested seasons: {args.seasons}"]

    for season in args.seasons:
        try:
            df = nfl.load_ftn_charting([season])
            row_count = df.shape[0]
            col_count = df.shape[1]

            out_path = os.path.join(OUTPUT_DIR, f"{season}_ftn_charting.csv")
            df.write_csv(out_path)

            msg = f"season={season} status=success rows={row_count} columns={col_count} output={out_path}"
            print(msg)
            log_lines.append(msg)

        except Exception as e:
            msg = f"season={season} status=failed error={e}"
            print(msg)
            log_lines.append(msg)
            log_lines.append(traceback.format_exc())

    log(log_lines)


if __name__ == "__main__":
    main()
