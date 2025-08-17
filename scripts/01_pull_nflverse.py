#!/usr/bin/env python3
"""
Step 01 — Pull core NFL datasets and write snapshots under data/raw/nflverse.

- Pulls weekly, pbp, rosters, schedules using nfl_data_py
- Defensive against version differences (tries multiple loader names)
- Writes both dated snapshots and *_latest.csv.gz files
- Produces manifest_latest.csv.gz summarizing what was written
"""

from __future__ import annotations
import argparse
import gzip
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

# ----------------------------- config ---------------------------------
OUT_ROOT = Path("data/raw/nflverse")


# ----------------------------- helpers --------------------------------
@dataclass
class TaskSpec:
    name: str
    candidates: List[str]  # candidate function names on nfl_data_py
    kwargs: Dict           # kwargs to pass when calling
    required: bool = False # if True and unresolved, raise; else warn & skip


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Pull core NFL datasets.")
    ap.add_argument("--start", type=int, default=None, help="Start year (e.g., 2019)")
    ap.add_argument("--end", type=int, default=None, help="End year (e.g., 2024)")
    return ap.parse_args()


def years_range(start: Optional[int], end: Optional[int]) -> List[int]:
    s = start or int(os.environ.get("YEARS_START", 2016))
    e = end or int(os.environ.get("YEARS_END", datetime.utcnow().year))
    if e < s:
        e = s
    return list(range(s, e + 1))


def resolve_loader(candidates: List[str]) -> Optional[Callable]:
    """
    Return the first callable on nfl_data_py whose name matches any in candidates.
    If none exist, return None.
    """
    import nfl_data_py as nfl
    for name in candidates:
        fn = getattr(nfl, name, None)
        if callable(fn):
            return fn
    return None


def write_csv_gz(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        df.to_csv(f, index=False)
    tmp.replace(path)


def save_snapshot_and_latest(df: pd.DataFrame, name: str, root: Path, stamp: str) -> Tuple[Path, Path]:
    """
    Always use underscore format:
    weekly_YYYYMMDD.csv.gz and weekly_latest.csv.gz
    """
    root.mkdir(parents=True, exist_ok=True)
    snap = root / f"{name}_{stamp}.csv.gz"
    latest = root / f"{name}_latest.csv.gz"
    write_csv_gz(df, snap)
    write_csv_gz(df, latest)
    return snap, latest


def add_generated_columns(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    out = df.copy()
    out["generated_at_utc"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    out["dataset"] = dataset
    return out


# ----------------------------- main -----------------------------------
def pull(start: Optional[int] = None, end: Optional[int] = None) -> None:
    yrs = years_range(start, end)
    stamp = datetime.utcnow().strftime("%Y%m%d")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Define tasks with multiple candidate function names
    tasks: List[TaskSpec] = [
        TaskSpec("weekly",    ["import_weekly_data"],                   dict(years=yrs), required=True),
        TaskSpec("pbp",       ["import_pbp_data"],                      dict(years=yrs), required=True),
        TaskSpec("rosters",   ["import_roster_data", "import_rosters", "import_roster"], dict(years=yrs), required=False),
        TaskSpec("schedules", ["import_schedules", "import_schedule"],  dict(years=yrs), required=True),
    ]

    manifest_rows: List[Dict] = []

    for t in tasks:
        print(f"➡️  {t.name}: resolving loader {t.candidates} ...", flush=True)
        fn = resolve_loader(t.candidates)

        if fn is None:
            msg = (f"WARNING: No loader found in nfl_data_py for {t.name} "
                   f"(tried {t.candidates}). Skipping {t.name}.")
            print(msg, flush=True)
            manifest_rows.append({
                "dataset": t.name,
                "rows": 0,
                "snapshot": "",
                "latest": "",
                "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds")+"Z",
                "years": f"{yrs[0]}-{yrs[-1]}",
                "note": "skipped; loader missing"
            })
            if t.required:
                pass
            continue

        print(f"   resolved → {fn.__name__}; pulling years {yrs[0]}–{yrs[-1]} ...", flush=True)
        df = fn(**t.kwargs)
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        df = df.drop_duplicates().reset_index(drop=True)
        df = add_generated_columns(df, t.name)

        snap, latest = save_snapshot_and_latest(df, t.name, OUT_ROOT, stamp)

        manifest_rows.append({
            "dataset": t.name,
            "rows": int(len(df)),
            "snapshot": str(snap),
            "latest": str(latest),
            "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds")+"Z",
            "years": f"{yrs[0]}-{yrs[-1]}",
            "note": ""
        })

        print(f"   {t.name}: {len(df):,} rows → {snap.name} (+ {latest.name})", flush=True)

    # Write manifest (always)
    manifest = pd.DataFrame(manifest_rows)
    write_csv_gz(manifest, OUT_ROOT / "manifest_latest.csv.gz")

    # Fail if required datasets missing
    missing_required = [t.name for t in tasks if t.required and not any(
        r["dataset"] == t.name and r["rows"] > 0 for r in manifest_rows
    )]
    if missing_required:
        raise RuntimeError(f"Missing required datasets: {', '.join(missing_required)}")

    print("✅ Step 01 complete.", flush=True)


if __name__ == "__main__":
    args = parse_args()
    pull(args.start, args.end)
