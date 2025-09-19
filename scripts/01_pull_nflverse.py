#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/01_pull_nflverse.py

End-to-end:
1) PBP: detect latest available season from nflverse GitHub releases (or CLI year),
   download, write canonical artifacts, prune old versions.
2) WEEKLY/ROSTERS/DEPTH: optional downloads via env templates + flags; otherwise
   just canonicalize existing local files (if present).
3) Canonical artifacts written under data/raw/nflverse:
   - <type>_latest.csv.gz
   - <type>_latest.parquet
   - <type>_latest.head100.csv.gz
   - <type>_manifest_latest.csv.gz
"""

from __future__ import annotations

import os
import sys
import re
import glob
import shutil
import hashlib
import datetime as dt
from typing import Optional, Iterable

import io
import gzip

import pandas as pd
import requests


# ------------ Config ------------
# PBP base (unchanged)
BASE = "https://github.com/nflverse/nflverse-data/releases/download/pbp"

RAW_ROOT = "data/raw"
NFLVERSE_DIR = os.path.join(RAW_ROOT, "nflverse")

# Canonical names
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

# DEPTH canon
DEPTH_CANON_CSV = "depth_latest.csv.gz"
DEPTH_CANON_PARQ = "depth_latest.parquet"
DEPTH_HEAD = "depth_latest.head100.csv.gz"
DEPTH_MANIFEST = "depth_manifest_latest.csv.gz"

TIMEOUT = 30  # seconds


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


def _best_by_year_or_mtime(paths, regex_year: re.Pattern) -> Optional[str]:
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
    # Accepts --start YYYY or a bare 4-digit token; if both absent, None
    now_year = dt.datetime.utcnow().year
    tokens = list(argv)[1:]
    for i, tok in enumerate(tokens):
        if tok == "--start" and i + 1 < len(tokens):
            y = tokens[i + 1]
            if y.isdigit() and len(y) == 4 and 1999 <= int(y) <= now_year:
                return int(y)
    for tok in tokens:
        if tok.isdigit() and len(tok) == 4 and 1999 <= int(tok) <= now_year:
            return int(tok)
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

    shutil.copyfile(season_csv, canon_csv)
    df.to_parquet(canon_parq, index=False)

    _atomic_write_bytes(os.path.join(NFLVERSE_DIR, PBP_HEAD), _to_gzip_csv_bytes(df.head(100)))
    sha = _sha256_file(season_csv)
    pd.DataFrame([{
        "file": os.path.basename(season_csv),
        "rows": int(len(df)),
        "sha256": sha,
    }]).to_csv(os.path.join(NFLVERSE_DIR, PBP_MANIFEST), index=False, compression="gzip")

    for p in glob.glob(os.path.join(NFLVERSE_DIR, "play_by_play_*.csv.gz")):
        if os.path.basename(p) != PBP_CANON_CSV:
            try:
                os.remove(p)
            except OSError:
                pass
    for p in glob.glob(os.path.join(NFLVERSE_DIR, "pbp_*.parquet")):
        if os.path.basename(p) != PBP_CANON_PARQ:
            try:
                os.remove(p)
            except OSError:
                pass

    print(f"Wrote: {canon_csv}")
    print(f"Wrote: {canon_parq}")
    print(f"Wrote: {os.path.join(NFLVERSE_DIR, PBP_HEAD)}")
    print(f"Wrote: {os.path.join(NFLVERSE_DIR, PBP_MANIFEST)}")
    print(f"PBP rows: {len(df)} | Year: {latest_year} | SHA256: {sha[:12]}...")


# ------------ WEEKLY canonicalization ------------
def weekly_canonicalize() -> None:
    _ensure_dir(NFLVERSE_DIR)

    canon_csv = os.path.join(NFLVERSE_DIR, WEEKLY_CANON_CSV)
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
        if os.path.abspath(best_csv) != os.path.abspath(canon_csv):
            shutil.copyfile(best_csv, canon_csv)
        source_csv = canon_csv

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

    for p in glob.glob(os.path.join(NFLVERSE_DIR, "weekly_*.csv.gz")):
        if os.path.basename(p) != WEEKLY_CANON_CSV:
            try:
                os.remove(p)
            except OSError:
                pass
    for p in glob.glob(os.path.join(NFLVERSE_DIR, "weekly_*.parquet")):
        if os.path.basename(p) != WEEKLY_CANON_PARQ:
            try:
                os.remove(p)
            except OSError:
                pass

    print(f"Wrote: {canon_parq}")
    print(f"Wrote: {os.path.join(NFLVERSE_DIR, WEEKLY_HEAD)}")
    print(f"Wrote: {os.path.join(NFLVERSE_DIR, WEEKLY_MANIFEST)}")
    print(f"Weekly rows: {len(df)} | SHA256: {sha[:12]}...")


# ------------ ROSTERS canonicalization ------------
def rosters_canonicalize() -> None:
    _ensure_dir(NFLVERSE_DIR)

    canon_csv = os.path.join(NFLVERSE_DIR, ROSTERS_CANON_CSV)
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

    for p in glob.glob(os.path.join(NFLVERSE_DIR, "rosters_*.csv.gz")):
        if os.path.basename(p) != ROSTERS_CANON_CSV:
            try:
                os.remove(p)
            except OSError:
                pass
    for p in glob.glob(os.path.join(NFLVERSE_DIR, "rosters_*.parquet")):
        if os.path.basename(p) != ROSTERS_CANON_PARQ:
            try:
                os.remove(p)
            except OSError:
                pass

    print(f"Wrote: {canon_parq}")
    print(f"Wrote: {os.path.join(NFLVERSE_DIR, ROSTERS_HEAD)}")
    print(f"Wrote: {os.path.join(NFLVERSE_DIR, ROSTERS_MANIFEST)}")
    print(f"Rosters rows: {len(df)} | SHA256: {sha[:12]}...")


# ------------ DEPTH canonicalization ------------
def depth_canonicalize() -> None:
    _ensure_dir(NFLVERSE_DIR)

    canon_csv = os.path.join(NFLVERSE_DIR, DEPTH_CANON_CSV)
    if os.path.exists(canon_csv):
        source_csv = canon_csv
    else:
        candidates = (
            glob.glob(os.path.join(NFLVERSE_DIR, "depth_*.csv.gz"))
            + glob.glob(os.path.join(NFLVERSE_DIR, "depth_charts_*.csv.gz"))
            + [p for p in [os.path.join(NFLVERSE_DIR, "depth_charts.csv.gz")] if os.path.exists(p)]
        )
        if not candidates:
            print("INFO: depth* not found; skipping depth canonicalization")
            return
        best_csv = _best_by_year_or_mtime(
            candidates,
            re.compile(r"(?:depth|depth_charts)_(\d{4})")
        )
        # If only no-year file exists, best_csv may be None; handle that
        if not best_csv:
            # Choose the no-year file if present
            fallback = os.path.join(NFLVERSE_DIR, "depth_charts.csv.gz")
            if os.path.exists(fallback):
                best_csv = fallback
            else:
                print("INFO: could not resolve best depth; skipping")
                return
        if os.path.abspath(best_csv) != os.path.abspath(canon_csv):
            shutil.copyfile(best_csv, canon_csv)
        source_csv = canon_csv

    df = _read_csv_gz(source_csv)
    if df.empty:
        raise SystemExit(f"ERROR: depth canonical empty: {source_csv}")

    canon_parq = os.path.join(NFLVERSE_DIR, DEPTH_CANON_PARQ)
    df.to_parquet(canon_parq, index=False)

    _atomic_write_bytes(os.path.join(NFLVERSE_DIR, DEPTH_HEAD), _to_gzip_csv_bytes(df.head(100)))
    sha = _sha256_file(source_csv)
    pd.DataFrame([{
        "file": os.path.basename(source_csv),
        "rows": int(len(df)),
        "sha256": sha,
    }]).to_csv(os.path.join(NFLVERSE_DIR, DEPTH_MANIFEST), index=False, compression="gzip")

    for p in (
        glob.glob(os.path.join(NFLVERSE_DIR, "depth_*.csv.gz"))
        + glob.glob(os.path.join(NFLVERSE_DIR, "depth_charts_*.csv.gz"))
        + [os.path.join(NFLVERSE_DIR, "depth_charts.csv.gz")]
    ):
        if os.path.basename(p) != DEPTH_CANON_CSV and os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass
    for p in (
        glob.glob(os.path.join(NFLVERSE_DIR, "depth_*.parquet"))
        + glob.glob(os.path.join(NFLVERSE_DIR, "depth_charts_*.parquet"))
        + [os.path.join(NFLVERSE_DIR, "depth_charts.parquet")]
    ):
        if os.path.basename(p) != DEPTH_CANON_PARQ and os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass

    print(f"Wrote: {canon_parq}")
    print(f"Wrote: {os.path.join(NFLVERSE_DIR, DEPTH_HEAD)}")
    print(f"Wrote: {os.path.join(NFLVERSE_DIR, DEPTH_MANIFEST)}")
    print(f"Depth rows: {len(df)} | SHA256: {sha[:12]}...")


# ------------ Optional downloads ------------
# Accept either full URLs or templates with {year}
WEEKLY_URL_TPL  = os.environ.get("NFLVERSE_WEEKLY_URL_TPL",  "")
ROSTERS_URL_TPL = os.environ.get("NFLVERSE_ROSTERS_URL_TPL", "")
DEPTH_URL_TPL   = os.environ.get("NFLVERSE_DEPTH_URL_TPL",   "")

# default filenames used when a download occurs
WEEKLY_SRC_TPL  = "weekly_{year}.csv.gz"
ROSTERS_SRC_TPL = "rosters_{year}.csv.gz"
DEPTH_SRC_TPL   = "depth_charts_{year}.csv.gz"
DEPTH_SRC_NOYEAR = "depth_charts.csv.gz"


def _probe(url: str) -> bool:
    try:
        r = requests.head(url, timeout=TIMEOUT, allow_redirects=True)
        if r.status_code == 405:
            r = requests.get(url, stream=True, timeout=TIMEOUT, allow_redirects=True)
        return r.status_code in (200, 301, 302)
    except requests.RequestException:
        return False


def _download(url: str, dest_path: str) -> None:
    _ensure_dir(os.path.dirname(dest_path))
    with requests.get(url, stream=True, timeout=TIMEOUT) as r:
        r.raise_for_status()
        tmp = dest_path + ".part"
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 15):
                if chunk:
                    f.write(chunk)
        os.replace(tmp, dest_path)


def _maybe_download_from_tpl(tpl: str, year: int, local_tpl: str, label: str, *, allow_no_year: bool = False) -> Optional[str]:
    """
    If tpl is set:
      - If it contains {year}, format it and save to local_tpl.format(year=year)
      - If it does NOT contain {year} and allow_no_year=True, use tpl as-is and save to DEPTH_SRC_NOYEAR
    Returns local path if downloaded; otherwise None.
    """
    if not tpl:
        return None

    if "{year}" in tpl:
        url = tpl.format(year=year)
        local_name = local_tpl.format(year=year)
    else:
        if not allow_no_year:
            # No-year template not allowed for this label
            return None
        url = tpl
        local_name = DEPTH_SRC_NOYEAR

    if not _probe(url):
        print(f"INFO: {label} feed not available at {url}; skipping download")
        return None

    local = os.path.join(NFLVERSE_DIR, local_name)
    _download(url, local)
    print(f"Downloaded {label}: {url} -> {local}")
    return local


def _parse_flags(argv: Iterable[str]) -> dict:
    argv = list(argv)[1:]
    return {
        "dl_weekly": "--dl-weekly" in argv,
        "dl_rosters": "--dl-rosters" in argv,
        "dl_depth": "--dl-depth" in argv,
        "fail_missing": "--fail-missing" in argv,
    }


# ------------ Entry ------------
def main():
    explicit_year = _parse_year_from_argv(sys.argv)
    flags = _parse_flags(sys.argv)
    latest_year = resolve_latest_year(explicit_year)

    # PBP
    pbp_pipeline(latest_year)

    # Optional downloads BEFORE canonicalization
    # Weekly/Rosters expect {year} templates
    if flags["dl_weekly"]:
        path = _maybe_download_from_tpl(WEEKLY_URL_TPL, latest_year, WEEKLY_SRC_TPL, "WEEKLY", allow_no_year=False)
        if flags["fail_missing"] and not path:
            raise SystemExit("ERROR: WEEKLY feed missing and --fail-missing specified")

    if flags["dl_rosters"]:
        path = _maybe_download_from_tpl(ROSTERS_URL_TPL, latest_year, ROSTERS_SRC_TPL, "ROSTERS", allow_no_year=False)
        if flags["fail_missing"] and not path:
            raise SystemExit("ERROR: ROSTERS feed missing and --fail-missing specified")

    # Depth supports both {year} and no-year rolling file (e.g., depth_charts.csv.gz)
    if flags["dl_depth"]:
        path = _maybe_download_from_tpl(DEPTH_URL_TPL, latest_year, DEPTH_SRC_TPL, "DEPTH", allow_no_year=True)
        if flags["fail_missing"] and not path:
            raise SystemExit("ERROR: DEPTH feed missing and --fail-missing specified")

    # Canonicalization (picks up downloaded files if present; otherwise skips if none)
    weekly_canonicalize()
    rosters_canonicalize()
    depth_canonicalize()


if __name__ == "__main__":
    main()
