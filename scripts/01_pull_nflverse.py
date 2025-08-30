# /scripts/01_pull_nflverse.py

import os
import sys
import datetime as dt
from typing import Optional

import pandas as pd
import requests

BASE = "https://github.com/nflverse/nflverse-data/releases/download/pbp"
RAW_DIR = "data/raw"
FILENAME_TPL = "play_by_play_{year}.csv.gz"
TIMEOUT = 30  # seconds

def _asset_url(year: int) -> str:
    return f"{BASE}/{FILENAME_TPL.format(year=year)}"

def _exists(url: str) -> bool:
    r = requests.head(url, timeout=TIMEOUT)
    return r.status_code == 200

def resolve_latest_year(start_year: Optional[int] = None, min_year: int = 1999) -> int:
    """
    Probe GitHub assets descending by year to find the most recent available file.
    Tries: current_year, current_year-1, ..., down to min_year.
    """
    year = start_year or dt.datetime.utcnow().year
    for y in range(year, min_year - 1, -1):
        url = _asset_url(y)
        try:
            if _exists(url):
                return y
        except requests.RequestException:
            continue
    raise RuntimeError("No available nflverse play_by_play asset found")

def download_pbp(year: int) -> str:
    os.makedirs(RAW_DIR, exist_ok=True)
    url = _asset_url(year)
    local_path = os.path.join(RAW_DIR, FILENAME_TPL.format(year=year))
    with requests.get(url, stream=True, timeout=TIMEOUT) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 15):
                if chunk:
                    f.write(chunk)
    return local_path

def load_pbp_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, compression="gzip", low_memory=False)

def main():
    # Optional: allow explicit year via CLI arg; otherwise auto-detect latest.
    # Example: python scripts/01_pull_nflverse.py 2023
    explicit_year = int(sys.argv[1]) if len(sys.argv) > 1 else None
    latest_year = resolve_latest_year(explicit_year)
    csv_path = download_pbp(latest_year)
    df = load_pbp_csv(csv_path)
    print(f"Year {latest_year}: {len(df)} rows")
    out_path = os.path.join(RAW_DIR, f"pbp_{latest_year}.parquet")
    df.to_parquet(out_path, index=False)
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
