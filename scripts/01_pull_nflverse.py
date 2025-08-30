# /scripts/01_pull_nflverse.py

import os
import sys
import datetime as dt
from typing import Optional, Iterable
import hashlib
import io

import pandas as pd
import requests

BASE = "https://github.com/nflverse/nflverse-data/releases/download/pbp"
RAW_ROOT = "data/raw"
NFLVERSE_DIR = os.path.join(RAW_ROOT, "nflverse")
FILENAME_TPL = "play_by_play_{year}.csv.gz"
TIMEOUT = 30  # seconds


def _asset_url(year: int) -> str:
    return f"{BASE}/{FILENAME_TPL.format(year=year)}"


def _exists(url: str) -> bool:
    try:
        r = requests.head(url, timeout=TIMEOUT, allow_redirects=True)
        if r.status_code == 405:
            r = requests.get(url, stream=True, timeout=TIMEOUT, allow_redirects=True)
        return r.status_code in (200, 301, 302)
    except requests.RequestException:
        return False


def resolve_latest_year(start_year: Optional[int] = None, min_year: int = 1999) -> int:
    year = start_year or dt.datetime.utcnow().year
    for y in range(year, min_year - 1, -1):
        if _exists(_asset_url(y)):
            return y
    raise RuntimeError("No available nflverse play_by_play asset found")


def _atomic_write_bytes(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def download_pbp(year: int) -> str:
    os.makedirs(NFLVERSE_DIR, exist_ok=True)
    url = _asset_url(year)
    local_path = os.path.join(NFLVERSE_DIR, FILENAME_TPL.format(year=year))
    with requests.get(url, stream=True, timeout=TIMEOUT) as r:
        r.raise_for_status()
        tmp_path = local_path + ".part"
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 15):
                if chunk:
                    f.write(chunk)
        os.replace(tmp_path, local_path)
    return local_path


def load_pbp_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, compression="gzip", low_memory=False)


def _parse_year_from_argv(argv: Iterable[str]) -> Optional[int]:
    now_year = dt.datetime.utcnow().year
    for tok in list(argv)[1:]:
        if tok.isdigit() and len(tok) == 4:
            y = int(tok)
            if 1999 <= y <= now_year:
                return y
    return None


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _to_gzip_csv_bytes(df: pd.DataFrame) -> bytes:
    bio = io.BytesIO()
    df.to_csv(bio, index=False)
    raw = bio.getvalue()
    import gzip
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="wb") as gz:
        gz.write(raw)
    return out.getvalue()


def main():
    explicit_year = _parse_year_from_argv(sys.argv)
    latest_year = resolve_latest_year(explicit_year)

    # 1) Download canonical season file
    csv_path = download_pbp(latest_year)
    df = load_pbp_csv(csv_path)
    if df.empty:
        raise SystemExit(f"ERROR: downloaded CSV has 0 rows: {csv_path}")

    # 2) Write Parquet (season)
    parquet_path = os.path.join(NFLVERSE_DIR, f"pbp_{latest_year}.parquet")
    df.to_parquet(parquet_path, index=False)

    # 3) Write small “latest” artifacts for Git:
    #    a) head100 preview so diffs are detectable and tiny
    head_bytes = _to_gzip_csv_bytes(df.head(100))
    latest_head_path = os.path.join(NFLVERSE_DIR, "pbp_latest.head100.csv.gz")
    _atomic_write_bytes(latest_head_path, head_bytes)

    #    b) manifest with metadata
    sha = _sha256_file(csv_path)
    manifest = pd.DataFrame(
        [{
            "file": os.path.basename(csv_path),
            "year": latest_year,
            "rows": int(len(df)),
            "sha256": sha,
            "fetched_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }]
    )
    manifest_path = os.path.join(NFLVERSE_DIR, "manifest_latest.csv.gz")
    manifest.to_csv(manifest_path, index=False, compression="gzip")

    # 4) Logs for CI
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {parquet_path}")
    print(f"Wrote: {latest_head_path}")
    print(f"Wrote: {manifest_path}")
    print(f"Rows: {len(df)} | Year: {latest_year} | SHA256: {sha[:12]}...")


if __name__ == "__main__":
    main()
