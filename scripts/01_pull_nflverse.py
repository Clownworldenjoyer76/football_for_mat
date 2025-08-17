#!/usr/bin/env python3
"""
Pull weekly, pbp, rosters, schedules from nfl_data_py and write to data/raw/nflverse.
- Snapshot files: *_YYYYMMDD.csv.gz
- Stable pointers: *_latest.csv.gz
Env overrides:
  YEARS_START (default 2016)
  YEARS_END   (default = current year)
"""
from __future__ import annotations
import os, sys, io, gzip, shutil
from datetime import datetime
from pathlib import Path
import pandas as pd

def years_range() -> list[int]:
    start = int(os.environ.get("YEARS_START", 2016))
    end = int(os.environ.get("YEARS_END", datetime.now().year))
    if end < start:
        end = start
    return list(range(start, end + 1))

def write_csv_gz(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # atomic write
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        df.to_csv(f, index=False)
    tmp.replace(path)

def save_snapshot_and_latest(df: pd.DataFrame, base: Path, stamp: str) -> tuple[Path, Path]:
    snap = base.with_name(f"{base.stem}_{stamp}{base.suffix}")
    write_csv_gz(df, snap)
    write_csv_gz(df, base.with_name(f"{base.stem}_latest{base.suffix}"))
    return snap, base

def add_generated_columns(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    out = df.copy()
    out["generated_at_utc"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    out["dataset"] = dataset
    return out

def pull():
    import nfl_data_py as nfl  # imported here so the script imports cleanly even if not installed elsewhere

    yrs = years_range()
    today = datetime.utcnow().strftime("%Y%m%d")
    root = Path("data/raw/nflverse")
    root.mkdir(parents=True, exist_ok=True)

    tasks = [
        ("weekly",     nfl.import_weekly_data,   dict(years=yrs)),
        ("pbp",        nfl.import_pbp_data,      dict(years=yrs)),
        ("rosters",    nfl.import_rosters,       dict(years=yrs)),
        ("schedules",  nfl.import_schedules,     dict(years=yrs)),
    ]

    written = []
    for name, fn, kwargs in tasks:
        print(f"➡️  Pulling {name} for years {yrs[0]}–{yrs[-1]} ...", flush=True)
        df = fn(**kwargs)
        # Light hygiene: drop exact dup rows, keep deterministic column order
        df = df.drop_duplicates().reset_index(drop=True)
        df = add_generated_columns(df, name)
        base = root / f"{name}.csv.gz"
        snap, latest = save_snapshot_and_latest(df, base, today)
        written.append((name, snap, latest, len(df)))

    # Write a small manifest for downstream jobs
    manifest = pd.DataFrame(
        [{"dataset": n, "rows": rows, "snapshot": str(snap), "latest": str(latest),
          "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds")+"Z"}
         for (n, snap, latest, rows) in written]
    )
    write_csv_gz(manifest, root / "manifest_latest.csv.gz")
    print("✅ Done.")
    for n, snap, latest, rows in written:
        print(f"   {n}: {rows:,} rows → {snap.name} (and {latest.name})")

if __name__ == "__main__":
    try:
        pull()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
