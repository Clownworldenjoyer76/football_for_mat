# /scripts/01_pull_nflverse.py
#
# Tasks:
# 1) Pull latest nflverse PBP for detected season and overwrite canonical files.
# 2) Canonicalize WEEKLY and ROSTERS and ALWAYS create head100 + manifest files
#    whenever a canonical CSV exists.
# 3) Prune versioned full files to prevent repo bloat.
#
# Canonical outputs (always overwritten if source present):
#   data/raw/nflverse/play_by_play_latest.csv.gz
#   data/raw/nflverse/pbp_latest.parquet
#   data/raw/nflverse/pbp_latest.head100.csv.gz
#   data/raw/nflverse/manifest_latest.csv.gz
#
#   data/raw/nflverse/weekly_latest.csv.gz
#   data/raw/nflverse/weekly_latest.parquet
#   data/raw/nflverse/weekly_latest.head100.csv.gz
#   data/raw/nflverse/weekly_manifest_latest.csv.gz
#
#   data/raw/nflverse/rosters_latest.csv.gz
#   data/raw/nflverse/rosters_latest.parquet
#   data/raw/nflverse/rosters_latest.head100.csv.gz
#   data/raw/nflverse/rosters_manifest_latest.csv.gz

import os
import re
import io
import gzip
import glob
import sys
import shutil
import hashlib
import datetime as dt
from typing import Optional, Iterable

import pandas as pd
import requests

# ------------ Config ------------
BASE = "https://github.com/nflverse/nflverse-data/releases/download/pbp"
RAW_ROOT = "data/raw"
NFLVERSE_DIR = os.path.join(RAW_ROOT, "nflverse")

PBP_SRC_TPL = "play_by_play_{year}.csv.gz"
PBP_CANON_CSV = "play_by_play_latest.csv.gz"
PBP_CANON_PARQ = "pbp_latest.parquet"
PBP_HEAD = "pbp_latest.head100.csv.gz"
PBP_MANIFEST = "manifest_latest.csv.gz"

WEEKLY_CANON_CSV = "weekly_latest.csv.gz"
WEEKLY_CANON_PARQ = "weekly_latest.parquet"
WEEKLY_HEAD = "weekly_latest.head100.csv.gz"
WEEKLY_MANIFEST = "weekly_manifest_latest.csv.gz"

ROSTERS_CANON_CSV = "rosters_latest.csv.gz"
ROSTERS_CANON_PARQ = "rosters_latest.parquet"
ROSTERS_HEAD = "rosters_latest.head100.csv.gz"
ROSTERS_MANIFEST = "rosters_manifest_latest.csv.gz"

TIMEOUT = 30  # seconds


# ------------ Helpers ------------
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _asset_url(year: int) -> str:
    return f"{BASE}/{PBP_SRC_TPL.format(year=year)}"

def _exists(url: str) -> bool:
    try:
        r = requests.head(url, timeout=TIMEOUT, allow_redirects=True)
        if r.status_code == 405:
            r = requests.get(url, stream=True, timeout=TIMEOUT, allow_redirects=True)
        return r.status_code in (200, 301, 302)
    except requests.RequestException:
        return False

def _atomic_write_bytes(path: str, data: bytes) -> None:
    _ensure_dir(os.path.dirname(path))
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)

def _to_gzip_csv_bytes(df: pd.DataFrame) -> bytes:
    bio = io.BytesIO()
    df.to_csv(bio, index=False)
    raw = bio.getvalue()
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="wb") as gz:
        gz.write(raw)
    return out.getvalue()

def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _read_csv_gz(path: str) -> pd.DataFrame:
    return pd.read_csv(path, compression="gzip", low_memory=False)

def _best_by_year_or_mtime(paths: list[str], regex_year: re.Pattern) -> Optional[str]:
    if not paths:
        return None
    items = []
    for p in paths:
        base = os.path.basename(p)
        m = regex_year.search(base)
        yr = int(m.group(1)) if m else -1
        mt = os.path.getmtime(p)
        items.append((yr, mt, p))
    items.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return items[0][2]

def _parse_year_from_argv(argv: Iterable[str]) -> Optional[int]:
    now_year = dt.datetime.utcnow().year
    for tok in list(argv)[1:]:
        if tok.isdigit() and len(tok) == 4:
            y = int(tok)
            if 1999 <= y <= now_year:
                return y
    return None


# ------------ PBP ingestion ------------
def resolve_latest_year(start_year: Optional[int] = None, min_year: int = 1999) -> int:
    year = start_year or dt.datetime.utcnow().year
    for y in range(year, min_year - 1, -1):
        if _exists(_asset_url(y)):
            return y
    raise RuntimeError("No available nflverse play_by_play asset found")

def download_pbp(year: int) -> str:
    _ensure_dir(NFLVERSE_DIR)
    url = _asset_url(year)
    local_path = os.path.join(NFLVERSE_DIR, PBP_SRC_TPL.format(year=year))
    with requests.get(url, stream=True, timeout=TIMEOUT) as r:
        r.raise_for_status()
        tmp_path = local_path + ".part"
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 15):
                if chunk:
                    f.write(chunk)
        os.replace(tmp_path, local_path)
    return local_path

def pbp_pipeline(explicit_year: Optional[int]) -> None:
    latest_year = resolve_latest_year(explicit_year)
    season_csv = download_pbp(latest_year)

    df = _read_csv_gz(season_csv)
    if df.empty:
        raise SystemExit(f"ERROR: downloaded PBP has 0 rows: {season_csv}")

    canon_csv = os.path.join(NFLVERSE_DIR, PBP_CANON_CSV)
    canon_parq = os.path.join(NFLVERSE_DIR, PBP_CANON_PARQ)

    # overwrite canonical CSV and parquet
    shutil.copyfile(season_csv, canon_csv)
    df.to_parquet(canon_parq, index=False)

    # tiny artifacts
    _atomic_write_bytes(os.path.join(NFLVERSE_DIR, PBP_HEAD), _to_gzip_csv_bytes(df.head(100)))
    sha = _sha256_file(canon_csv)
    pd.DataFrame([{
        "file": os.path.basename(canon_csv),
        "rows": int(len(df)),
        "sha256": sha,
        "season_source": os.path.basename(season_csv),
        "year_detected": latest_year,
        "fetched_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }]).to_csv(os.path.join(NFLVERSE_DIR, PBP_MANIFEST), index=False, compression="gzip")

    # prune versioned fulls
    for p in glob.glob(os.path.join(NFLVERSE_DIR, "play_by_play_*.csv.gz")):
        if os.path.basename(p) != PBP_CANON_CSV:
            try: os.remove(p)
            except FileNotFoundError: pass
    for p in glob.glob(os.path.join(NFLVERSE_DIR, "pbp_*.parquet")):
        if os.path.basename(p) != PBP_CANON_PARQ:
            try: os.remove(p)
            except FileNotFoundError: pass

    print(f"Wrote: {canon_csv}")
    print(f"Wrote: {canon_parq}")
    print(f"Wrote: {os.path.join(NFLVERSE_DIR, PBP_HEAD)}")
    print(f"Wrote: {os.path.join(NFLVERSE_DIR, PBP_MANIFEST)}")
    print(f"PBP rows: {len(df)} | Year: {latest_year} | SHA256: {sha[:12]}...")


# ------------ WEEKLY canonicalization (with SameFile guard) ------------
def weekly_canonicalize() -> None:
    _ensure_dir(NFLVERSE_DIR)

    canon_csv = os.path.join(NFLVERSE_DIR, WEEKLY_CANON_CSV)
    # Resolve source CSV: prefer existing canonical; else best weekly_* file
    if os.path.exists(canon_csv):
        source_csv = canon_csv
    else:
        candidates = glob.glob(os.path.join(NFLVERSE_DIR, "weekly_*.csv.gz"))
        if not candidates:
            print("INFO: weekly_* not found; skipping weekly canonicalization")
            return
        best_csv = _best_by_year_or_mtime(candidates, re.compile(r"weekly_(\d{4})"))
        if not best_csv:
            print("INFO: could not resolve best weekly; skipping")
            return
        # Copy only if different path
        if os.path.abspath(best_csv) != os.path.abspath(canon_csv):
            shutil.copyfile(best_csv, canon_csv)
        source_csv = canon_csv

    # Build parquet + small artifacts from canonical CSV
    df = _read_csv_gz(source_csv)
    if df.empty:
        raise SystemExit(f"ERROR: weekly canonical empty: {source_csv}")

    canon_parq = os.path.join(NFLVERSE_DIR, WEEKLY_CANON_PARQ)
    df.to_parquet(canon_parq, index=False)

    _atomic_write_bytes(os.path.join(NFLVERSE_DIR, WEEKLY_HEAD), _to_gzip_csv_bytes(df.head(100)))
    sha = _sha256_file(source_csv)
    pd.DataFrame([{
        "file": os.path.basename(source_csv),
        "rows": int(len(df)),
        "sha256": sha,
    }]).to_csv(os.path.join(NFLVERSE_DIR, WEEKLY_MANIFEST), index=False, compression="gzip")

    # prune other versioned weekly files
    for p in glob.glob(os.path.join(NFLVERSE_DIR, "weekly_*.csv.gz")):
        if os.path.basename(p) != WEEKLY_CANON_CSV:
            try: os.remove(p)
            except FileNotFoundError: pass
    for p in glob.glob(os.path.join(NFLVERSE_DIR, "weekly_*.parquet")):
        if os.path.basename(p) != WEEKLY_CANON_PARQ:
            try: os.remove(p)
            except FileNotFoundError: pass

    print(f"Wrote: {canon_csv}")
    print(f"Wrote: {canon_parq}")
    print(f"Wrote: {os.path.join(NFLVERSE_DIR, WEEKLY_HEAD)}")
    print(f"Wrote: {os.path.join(NFLVERSE_DIR, WEEKLY_MANIFEST)}")
    print(f"Weekly rows: {len(df)} | SHA256: {sha[:12]}...")


# ------------ ROSTERS canonicalization (with SameFile guard) ------------
def rosters_canonicalize() -> None:
    _ensure_dir(NFLVERSE_DIR)

    canon_csv = os.path.join(NFLVERSE_DIR, ROSTERS_CANON_CSV)
    # Resolve source CSV: prefer existing canonical; else best rosters_* file
    if os.path.exists(canon_csv):
        source_csv = canon_csv
    else:
        candidates = glob.glob(os.path.join(NFLVERSE_DIR, "rosters_*.csv.gz"))
        if not candidates:
            print("INFO: rosters_* not found; skipping rosters canonicalization")
            return
        best_csv = _best_by_year_or_mtime(candidates, re.compile(r"rosters_(\d{4})"))
        if not best_csv:
            print("INFO: could not resolve best rosters; skipping")
            return
        if os.path.abspath(best_csv) != os.path.abspath(canon_csv):
            shutil.copyfile(best_csv, canon_csv)
        source_csv = canon_csv

    # Build parquet + small artifacts from canonical CSV
    df = _read_csv_gz(source_csv)
    if df.empty:
        raise SystemExit(f"ERROR: rosters canonical empty: {source_csv}")

    canon_parq = os.path.join(NFLVERSE_DIR, ROSTERS_CANON_PARQ)
    df.to_parquet(canon_parq, index=False)

    _atomic_write_bytes(os.path.join(NFLVERSE_DIR, ROSTERS_HEAD), _to_gzip_csv_bytes(df.head(100)))
    sha = _sha256_file(source_csv)
    pd.DataFrame([{
        "file": os.path.basename(source_csv),
        "rows": int(len(df)),
        "sha256": sha,
    }]).to_csv(os.path.join(NFLVERSE_DIR, ROSTERS_MANIFEST), index=False, compression="gzip")

    # prune other versioned rosters files
    for p in glob.glob(os.path.join(NFLVERSE_DIR, "rosters_*.csv.gz")):
        if os.path.basename(p) != ROSTERS_CANON_CSV:
            try: os.remove(p)
            except FileNotFoundError: pass
    for p in glob.glob(os.path.join(NFLVERSE_DIR, "rosters_*.parquet")):
        if os.path.basename(p) != ROSTERS_CANON_PARQ:
            try: os.remove(p)
            except FileNotFoundError: pass

    print(f"Wrote: {canon_csv}")
    print(f"Wrote: {canon_parq}")
    print(f"Wrote: {os.path.join(NFLVERSE_DIR, ROSTERS_HEAD)}")
    print(f"Wrote: {os.path.join(NFLVERSE_DIR, ROSTERS_MANIFEST)}")
    print(f"Rosters rows: {len(df)} | SHA256: {sha[:12]}...")


# ------------ Entry ------------
def main():
    explicit_year = _parse_year_from_argv(sys.argv)
    pbp_pipeline(explicit_year)
    weekly_canonicalize()
    rosters_canonicalize()

if __name__ == "__main__":
    main()
