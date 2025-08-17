#!/usr/bin/env python3
"""
Pull weekly, pbp, rosters, schedules from nfl_data_py and write to data/raw/nflverse.
- Snapshot files: *_YYYYMMDD.csv.gz
- Stable pointers: *_latest.csv.gz
CLI:
  --start 2019 --end 2024
"""
from __future__ import annotations
import argparse, gzip, os
from datetime import datetime
from pathlib import Path
import pandas as pd

import nfl_data_py as nfl  # tolerate minor API differences across versions


# ---------- helpers ----------
def _resolve(fn_names):
    """
    Return the first callable on nfl_data_py whose name matches any in fn_names.
    Raises RuntimeError if none exist.
    """
    for name in fn_names:
        fn = getattr(nfl, name, None)
        if callable(fn):
            return fn
    raise RuntimeError(f"None of these functions exist in nfl_data_py: {fn_names}")

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


# ---------- main ----------
def pull(start: int | None = None, end: int | None = None) -> None:
    yrs = years_range(start, end)
    stamp = datetime.utcnow().strftime("%Y%m%d")
    root = Path("data/raw/nflverse")
    root.mkdir(parents=True, exist_ok=True)

    # Resolve function names (handles minor version diffs)
    fn_weekly    = _resolve(["import_weekly_data"])
    fn_pbp       = _resolve(["import_pbp_data"])
    fn_rosters   = _resolve(["import_roster_data", "import_rosters", "import_roster"])
    fn_schedules = _resolve(["import_schedules", "import_schedule"])

    tasks = [
        ("weekly",    fn_weekly,    dict(years=yrs)),
        ("pbp",       fn_pbp,       dict(years=yrs)),
        ("rosters",   fn_rosters,   dict(years=yrs)),
        ("schedules", fn_schedules, dict(years=yrs)),
    ]

    manifest_rows = []
    for name, fn, kwargs in tasks:
        print(f"➡️  Pulling {name} for years {yrs[0]}–{yrs[-1]} ...", flush=True)
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
        print(f"   {name}: {len(df):,} rows → {snap.name} (+ {base.stem}_latest.csv.gz)")

    manifest = pd.DataFrame(manifest_rows)
    write_csv_gz(manifest, root / "manifest_latest.csv.gz")
    print("✅ Done.")

if __name__ == "__main__":
    a = parse_args()
    pull(a.start, a.end)
