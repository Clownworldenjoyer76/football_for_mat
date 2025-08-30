# /scripts/01_pull_ngs.py

import os
import sys
import datetime as dt
from typing import Optional, Iterable
import hashlib
import io

import pandas as pd
import requests

# Public nflverse NGS asset (as used by nflreadr::load_nextgen_stats)
BASE = "https://github.com/nflverse/nflverse-data/releases/download/nextgen_stats"
ASSET = "nextgen_stats.csv.gz"

RAW_ROOT = "data/raw"
NGS_DIR = os.path.join(RAW_ROOT, "ngs")
TIMEOUT = 30  # seconds


def _asset_url() -> str:
    return f"{BASE}/{ASSET}"


def _exists(url: str) -> bool:
    try:
        r = requests.head(url, timeout=TIMEOUT, allow_redirects=True)
        if r.status_code == 405:
            r = requests.get(url, stream=True, timeout=TIMEOUT, allow_redirects=True)
        return r.status_code in (200, 301, 302)
    except requests.RequestException:
        return False


def _atomic_write_bytes(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


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


def _parse_flags(argv: Iterable[str]) -> bool:
    """
    Optional flag: --fail-missing
    If provided, exit with non-zero when NGS is unavailable.
    Otherwise, exit 0 gracefully (keeps NGS optional).
    """
    return any(tok == "--fail-missing" for tok in list(argv)[1:])


def main():
    must_exist = _parse_flags(sys.argv)
    url = _asset_url()

    if not _exists(url):
        msg = "NGS feed not available at source; skipping optional ingestion."
        if must_exist:
            raise SystemExit(f"ERROR: {msg}")
        print(f"INFO: {msg}")
        return

    os.makedirs(NGS_DIR, exist_ok=True)

    # 1) Download canonical file
    csv_gz_path = os.path.join(NGS_DIR, ASSET)
    with requests.get(url, stream=True, timeout=TIMEOUT) as r:
        r.raise_for_status()
        tmp = csv_gz_path + ".part"
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 15):
                if chunk:
                    f.write(chunk)
        os.replace(tmp, csv_gz_path)

    # 2) Load to DataFrame
    df = pd.read_csv(csv_gz_path, compression="gzip", low_memory=False)
    if df.empty:
        raise SystemExit(f"ERROR: downloaded NGS CSV has 0 rows: {csv_gz_path}")

    # 3) Write Parquet snapshot
    parquet_path = os.path.join(NGS_DIR, "ngs.parquet")
    df.to_parquet(parquet_path, index=False)

    # 4) Write small “latest” artifacts to ensure Git diffs
    head_bytes = _to_gzip_csv_bytes(df.head(100))
    latest_head_path = os.path.join(NGS_DIR, "ngs_latest.head100.csv.gz")
    _atomic_write_bytes(latest_head_path, head_bytes)

    sha = _sha256_file(csv_gz_path)
    manifest = pd.DataFrame(
        [{
            "file": os.path.basename(csv_gz_path),
            "rows": int(len(df)),
            "sha256": sha,
            "fetched_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }]
    )
    manifest_path = os.path.join(NGS_DIR, "manifest_latest.csv.gz")
    manifest.to_csv(manifest_path, index=False, compression="gzip")

    # 5) Logs for CI
    print(f"Wrote: {csv_gz_path}")
    print(f"Wrote: {parquet_path}")
    print(f"Wrote: {latest_head_path}")
    print(f"Wrote: {manifest_path}")
    print(f"Rows: {len(df)} | SHA256: {sha[:12]}...")


if __name__ == "__main__":
    main()
