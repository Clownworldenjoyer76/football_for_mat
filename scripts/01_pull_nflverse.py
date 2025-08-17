#!/usr/bin/env python3
"""
Pull weekly, pbp, rosters, schedules from nfl_data_py (0.3.3) and write to data/raw/nflverse.
- Snapshot files: *_YYYYMMDD.csv.gz
- Stable pointers: *_latest.csv.gz
CLI:
  --start 2019 --end 2024
Env overrides (optional):
  YEARS_START, YEARS_END
"""
from __future__ import annotations
import argparse, gzip
from datetime import datetime
from pathlib import Path
import os
import pandas as pd

# Explicit imports for nfl_data_py==0.3.3
from nfl_data_py import (
    import_weekly_data,
    import_pbp_data,
    import_rosters,
    import_schedules,
)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    return ap.parse_args()

def years_range(start: int | None, end: int | None) -> list[int]:
    s = start or int(os.environ.get("YEARS_START", 2016))
    e = end or int(os.environ.get("YEARS_END", datetime.utcnow().year))
    if e < s:
        e = s
    return list(range(s, e + 1))

def write_csv_gz(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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

def pull(start: int | None = None, end: int | None = None) -> None:
    yrs = years_range(start, end)
    stamp = datetime.utcnow().strftime("%Y%m%d")
    root = Path("data/raw/nflverse")
    root.mkdir(parents=True, exist_ok=True)

    tasks = [
        ("weekly",    import_weekly_data,  dict(years=yrs)),
        ("pbp",       import_pbp_data,     dict(years=yrs)),
        ("rosters",   import_rosters,      dict(years=yrs)),
        ("schedules", import_schedules,    dict(years=yrs)),
    ]

    manifest_rows = []
    for name, fn, kwargs in tasks:
        print(f"➡️ Pulling {name} for years {yrs[0]}–{yrs[-1]} ...", flush=True)
        df = fn(**kwargs).drop_duplicates().reset_index(drop=True)
        df = add_generated_columns(df, name)
        base = root / f"{name}.csv.gz"
        snap, latest = save_snapshot_and_latest(df, base, stamp)
        manifest_rows.append({
            "dataset": name,
            "rows": len(df),
            "snapshot": str(snap),
            "latest": str(latest.with_name(f"{latest.stem}_latest{latest.suffix}")),
            "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds")+"Z",
            "years": f"{yrs[0]}-{yrs[-1]}",
        })
        print(f"   {name}: {len(df):,} rows → {snap.name} (+ {base.stem}_latest.csv.gz)", flush=True)

    manifest = pd.DataFrame(manifest_rows)
    write_csv_gz(manifest, root / "manifest_latest.csv.gz")
    print("✅ Done.")

if __name__ == "__main__":
    a = parse_args()
    pull(a.start, a.end)
