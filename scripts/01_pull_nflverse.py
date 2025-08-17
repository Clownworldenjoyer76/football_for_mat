#!/usr/bin/env python3
"""
Step 01 — Pull core NFL datasets and write snapshots under data/raw/nflverse.

This script is defensive against nfl_data_py version differences:
- Tries to resolve each loader function at runtime
- If a loader is not present (e.g., roster import missing in your CI build),
  it logs a WARNING, writes no rows for that dataset, and continues
- Produces both date-stamped snapshots and *_latest.csv.gz pointers
- Emits a tiny manifest_latest.csv.gz summarizing what was written

Usage:
  python scripts/01_pull_nflverse.py --start 2019 --end 2024
Env overrides:
  YEARS_START, YEARS_END
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


def save_snapshot_and_latest(df: pd.DataFrame, base: Path, stamp: str) -> Tuple[Path, Path]:
    snap = base.with_name(f"{base.stem}_{stamp}{base.suffix}")
    write_csv_gz(df, snap)
    write_csv_gz(df, base.with_name(f"{base.stem}_latest{base.suffix}"))
    return snap, base


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

    # Define tasks with multiple candidate function names (covers different nfl_data_py versions)
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
                # Required dataset missing – fail clearly after manifest is written
                pass
            continue

        print(f"   resolved → {fn.__name__}; pulling years {yrs[0]}–{yrs[-1]} ...", flush=True)
        df = fn(**t.kwargs)
        if not isinstance(df, pd.DataFrame):
            # best-effort normalize (older APIs sometimes return list/iterables)
            df = pd.DataFrame(df)

        df = df.drop_duplicates().reset_index(drop=True)
        df = add_generated_columns(df, t.name)

        base = OUT_ROOT / f"{t.name}.csv.gz"
        snap, latest = save_snapshot_and_latest(df, base, stamp)

        manifest_rows.append({
            "dataset": t.name,
            "rows": int(len(df)),
            "snapshot": str(snap),
            "latest": str(latest.with_name(f"{latest.stem}_latest{latest.suffix}")),
            "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds")+"Z",
            "years": f"{yrs[0]}-{yrs[-1]}",
            "note": ""
        })

        print(f"   {t.name}: {len(df):,} rows → {snap.name} (+ {base.stem}_latest.csv.gz)", flush=True)

    # Write manifest (always)
    manifest = pd.DataFrame(manifest_rows)
    write_csv_gz(manifest, OUT_ROOT / "manifest_latest.csv.gz")

    # If any required dataset failed to resolve, raise AFTER manifest so CI has breadcrumbs
    missing_required = [t.name for t in tasks if t.required and not any(
        r["dataset"] == t.name and r["rows"] > 0 for r in manifest_rows
    )]
    if missing_required:
        raise RuntimeError(f"Missing required datasets: {', '.join(missing_required)}")

    print("✅ Step 01 complete.", flush=True)


if __name__ == "__main__":
    args = parse_args()
    pull(args.start, args.end)
