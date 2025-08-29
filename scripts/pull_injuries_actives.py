#!/usr/bin/env python3
"""
Pull NFL injuries/actives from ESPN and write:
  data/raw/injuries/injury_reports_latest.csv

Notes
-----
- Requires: pandas (with lxml/html5lib installed, which your workflow now does)
- Keeps the --season flag for compatibility with the workflow, but it is not
  needed for ESPN (the page is always "current").
"""

import argparse
import os
from datetime import datetime

import pandas as pd

ESPN_INJURIES_URL = "https://www.espn.com/nfl/injuries"


def fetch_espn_injuries() -> pd.DataFrame:
    """
    Scrape ESPN injuries page into a single DataFrame.
    """
    print(f"[injuries] Fetching {ESPN_INJURIES_URL}", flush=True)
    tables = pd.read_html(ESPN_INJURIES_URL)  # needs lxml
    if not tables:
        raise RuntimeError("ESPN returned no injury tables")

    df = pd.concat(tables, ignore_index=True, sort=False)

    # Basic cleanup: normalize column names a bit (best-effort)
    df.columns = [str(c).strip().upper().replace(" ", "_") for c in df.columns]

    # Add metadata
    df["SOURCE"] = "espn"
    df["FETCHED_AT_UTC"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    return df


def write_latest_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[injuries] wrote {path} ({len(df)} rows)", flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Pull NFL injuries/actives (ESPN).")
    # Kept for workflow compatibility; not actually used by ESPN scrape.
    ap.add_argument("--season", type=int, required=False, help="Unused for ESPN; kept for compatibility.")
    return ap.parse_args()


def main() -> None:
    _ = parse_args()  # season ignored but accepted
    df = fetch_espn_injuries()

    if len(df) == 0:
        raise RuntimeError("ESPN injuries scrape returned 0 rows")

    out_path = "data/raw/injuries/injury_reports_latest.csv"
    write_latest_csv(df, out_path)


if __name__ == "__main__":
    main()
