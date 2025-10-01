#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_pull_nflverse.py

Downloads core nflverse datasets into data/raw/nflverse.

What it pulls:
  - weekly_<YEAR>.csv.gz (explicit season file, if --season is provided or SEASON env is set)
  - weekly_latest.csv.gz
  - rosters_latest.csv.gz
  - depth_latest.csv.gz
  - pbp_latest.csv.gz (and an anchored pbp_<YYYYMMDD>.csv.gz snapshot)
  - schedules_latest.csv.gz
  - manifest_latest.csv.gz
  - parquet mirrors (optional, best-effort)

Usage examples:
  python scripts/01_pull_nflverse.py --season 2025 --dl-weekly --dl-rosters --dl-depth
  SEASON=2025 python scripts/01_pull_nflverse.py --dl-weekly

Notes:
- Uses URL templates from env if provided:
    NFLVERSE_WEEKLY_URL_TPL   (default: releases/download/weekly/weekly_{year}.csv.gz)
    NFLVERSE_WEEKLY_LATEST    (default: releases/download/weekly/weekly_latest.csv.gz)
    NFLVERSE_PBP_LATEST
    NFLVERSE_ROSTERS_LATEST
    NFLVERSE_DEPTH_LATEST
    NFLVERSE_SCHEDULES_LATEST
    NFLVERSE_MANIFEST_LATEST
- Fails loud (non-zero exit) if explicit season weekly file is requested and not found.
"""

from __future__ import annotations
import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen
from urllib.error import HTTPError, URLError
import shutil

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw" / "nflverse"
RAW.mkdir(parents=True, exist_ok=True)

# Defaults (CSV.GZ endpoints)
DEF_WEEKLY_TPL = "https://github.com/nflverse/nflverse-data/releases/download/weekly/weekly_{year}.csv.gz"
DEF_WEEKLY_LATEST = "https://github.com/nflverse/nflverse-data/releases/download/weekly/weekly_latest.csv.gz"
DEF_PBP_LATEST = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_latest.csv.gz"
DEF_ROSTERS_LATEST = "https://github.com/nflverse/nflverse-data/releases/download/rosters/rosters_latest.csv.gz"
DEF_DEPTH_LATEST = "https://github.com/nflverse/nflverse-data/releases/download/depth/depth_latest.csv.gz"
DEF_SCHEDULES_LATEST = "https://github.com/nflverse/nflverse-data/releases/download/schedules/schedules_latest.csv.gz"
DEF_MANIFEST_LATEST = "https://github.com/nflverse/nflverse-data/releases/download/manifest/manifest_latest.csv.gz"

def env(k: str, default: str) -> str:
    return os.getenv(k, default)

URLS = {
    "weekly_tpl": env("NFLVERSE_WEEKLY_URL_TPL", DEF_WEEKLY_TPL),
    "weekly_latest": env("NFLVERSE_WEEKLY_LATEST", DEF_WEEKLY_LATEST),
    "pbp_latest": env("NFLVERSE_PBP_LATEST", DEF_PBP_LATEST),
    "rosters_latest": env("NFLVERSE_ROSTERS_LATEST", DEF_ROSTERS_LATEST),
    "depth_latest": env("NFLVERSE_DEPTH_LATEST", DEF_DEPTH_LATEST),
    "schedules_latest": env("NFLVERSE_SCHEDULES_LATEST", DEF_SCHEDULES_LATEST),
    "manifest_latest": env("NFLVERSE_MANIFEST_LATEST", DEF_MANIFEST_LATEST),
}

def download(url: str, dest: Path, required: bool = False) -> bool:
    try:
        print(f"→ GET {url}")
        with urlopen(url) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)
        print(f"✓ Wrote {dest} ({dest.stat().st_size} bytes)")
        return True
    except HTTPError as e:
        print(f"WARNING: HTTP {e.code} for {url}")
        if required:
            print(f"ERROR: required file not found: {url}", file=sys.stderr)
        return False
    except URLError as e:
        print(f"WARNING: URL error for {url}: {e}")
        if required:
            print(f"ERROR: required file not retrievable: {url}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"WARNING: unexpected error for {url}: {e}")
        if required:
            print(f"ERROR: required file not retrievable: {url}", file=sys.stderr)
        return False

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2019, help="(unused legacy arg)")
    ap.add_argument("--dl-weekly", action="store_true", help="download weekly files")
    ap.add_argument("--dl-rosters", action="store_true")
    ap.add_argument("--dl-depth", action="store_true")
    ap.add_argument("--dl-pbp", action="store_true")
    ap.add_argument("--season", type=int, default=None, help="explicit season for weekly (e.g., 2025)")
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    season = args.season or (int(os.getenv("SEASON")) if os.getenv("SEASON") else None)

    # Always pull these “latest” datasets you already rely on
    if args.dl_rosters:
        download(URLS["rosters_latest"], RAW / "rosters_latest.csv.gz")
    if args.dl_depth:
        download(URLS["depth_latest"], RAW / "depth_latest.csv.gz")
    if args.dl_pbp:
        # latest + dated snapshot
        ts = datetime.now(timezone.utc).strftime("%Y%m%d")
        download(URLS["pbp_latest"], RAW / "play_by_play_latest.csv.gz")
        download(URLS["pbp_latest"], RAW / f"pbp_{ts}.csv.gz")
    # Schedules + manifest are helpful context
    download(URLS["schedules_latest"], RAW / "schedules_latest.csv.gz")
    download(URLS["manifest_latest"], RAW / "manifest_latest.csv.gz")

    if args.dl_weekly:
        # weekly_latest (convenience)
        download(URLS["weekly_latest"], RAW / "weekly_latest.csv.gz")
        # explicit season weekly (required if season specified)
        if season:
            url = URLS["weekly_tpl"].format(year=season)
            ok = download(url, RAW / f"weekly_{season}.csv.gz", required=True)
            if not ok:
                sys.exit(1)

if __name__ == "__main__":
    main()
