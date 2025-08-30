# /scripts/01_pull_nflverse.py

import os
import sys
import datetime as dt
from typing import Optional, Iterable

import pandas as pd
import requests

BASE = "https://github.com/nflverse/nflverse-data/releases/download/pbp"
RAW_DIR = "data/raw"
FILENAME_TPL = "play_by_play_{year}.csv.gz"
TIMEOUT = 30  # seconds


def _asset_url(year: int) -> str:
    return f"{BASE}/{FILENAME_TPL.format(year=year)}"


def _exists(url: str) -> bool:
    try:
        r = requests.head(url, timeout=TIMEOUT, allow_redirects=True)
        if r.status_code == 405:  # HEAD not allowed, fall back to GET
            r = requests.get(url, stream=True, timeout=TIMEOUT, allow_redirects=True)
        return r.status_code in (200, 301, 302)
    except requests.RequestException:
        return False


def resolve_latest_year(start_year: Optional[int] = None, min_year: int = 1999) -> int:
    """
    Probe GitHub assets descending by year to find the most recent available file.
    Tries: current_year, current_year-1, ..., down to min_year.
    """
    year = start_year or dt.datetime.utcnow().year
    for y in range(year, min_year - 1, -1):
        if _exists(_asset_url(y)):
            return y
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


def _parse_year_from_argv(argv: Iterable[str]) -> Optional[int]:
    """
    Accepts any numeric 4-digit token as the explicit year.
    Ignores flags like --start, etc.
    """
    now_year = dt.datetime.utcnow().year
    for tok in list(argv)[1:]:
        if tok.isdigit() and len(tok) == 4:
            y = int(tok)
            if 1999 <= y <= now_year:
                return y
    return None


def main():
    explicit_year = _parse_year_from_argv(sys.argv)
    latest_year = resolve_latest_year(explicit_year)
    csv_path = download_pbp(latest_year)
    df = load_pbp_csv(csv_path)
    print(f"Year {latest_year}: {len(df)} rows")
    out_path = os.path.join(RAW_DIR, f"pbp_{latest_year}.parquet")
    df.to_parquet(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
