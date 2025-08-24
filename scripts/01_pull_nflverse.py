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


def save_snapshot_and_latest(
    df: pd.DataFrame, name: str, root: Path, stamp: str
) -> Tuple[Path, Path]:
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


def file_rows_if_exists(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        return int(len(df))
    except Exception:
        return None


# ----------------------------- main -----------------------------------
def pull(start: Optional[int] = None, end: Optional[int] = None) -> None:
    yrs = years_range(start, end)
    stamp = datetime.utcnow().strftime("%Y%m%d")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Define tasks with multiple candidate function names
    tasks: List[TaskSpec] = [
        TaskSpec("weekly",    ["import_weekly_data"],                   dict(years=yrs), required=True),
        TaskSpec("pbp",       ["import_pbp_data"],                      dict(years=yrs), required=True),
        TaskSpec(
            "rosters",
            # expanded to include names used by other nfl_data_py versions
            ["import_roster_data", "import_rosters", "import_roster", "load_rosters", "get_rosters"],
            dict(years=yrs),
            required=False
        ),
        TaskSpec("schedules", ["import_schedules", "import_schedule"],  dict(years=yrs), required=True),
    ]

    manifest_rows: List[Dict] = []

    for t in tasks:
        print(f"➡️  {t.name}: resolving loader {t.candidates} ...", flush=True)
        fn = resolve_loader(t.candidates)

        snap_path: Optional[Path] = None
        latest_path: Optional[Path] = None
        rows_written: int = 0
        note: str = ""

        if fn is not None:
            print(f"   resolved → {fn.__name__}; pulling years {yrs[0]}–{yrs[-1]} ...", flush=True)
            df = fn(**t.kwargs)
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
            df = df.drop_duplicates().reset_index(drop=True)
            df = add_generated_columns(df, t.name)
            snap_path, latest_path = save_snapshot_and_latest(df, t.name, OUT_ROOT, stamp)
            rows_written = int(len(df))
            print(f"   {t.name}: {rows_written:,} rows → {snap_path.name} (+ {latest_path.name})", flush=True)
        else:
            # No loader resolved — fallback: if a _latest already exists, record it
            fallback_latest = OUT_ROOT / f"{t.name}_latest.csv.gz"
            fallback_rows = file_rows_if_exists(fallback_latest)
            if fallback_rows is not None:
                latest_path = fallback_latest
                # try to infer the most recent snapshot that matches today's stamp
                candidate_snap = OUT_ROOT / f"{t.name}_{stamp}.csv.gz"
                snap_path = candidate_snap if candidate_snap.exists() else None
                rows_written = fallback_rows
                note = "used existing latest; loader missing"
                print(f"   WARNING: loader missing, but found existing {fallback_latest.name} ({rows_written} rows).", flush=True)
            else:
                note = "skipped; loader missing"
                print(f"   WARNING: No loader found and no existing latest file; skipping {t.name}.", flush=True)

        manifest_rows.append({
            "dataset": t.name,
            "rows": rows_written,
            "snapshot": str(snap_path) if snap_path else "",
            "latest": str(latest_path) if latest_path else "",
            "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds")+"Z",
            "years": f"{yrs[0]}-{yrs[-1]}",
            "note": note
        })

    # Write manifest (always)
    manifest = pd.DataFrame(manifest_rows)
    write_csv_gz(manifest, OUT_ROOT / "manifest_latest.csv.gz")

    # Fail if required datasets missing (rows must be > 0)
    missing_required = [
        t.name for t in tasks
        if t.required and not any(r["dataset"] == t.name and r["rows"] > 0 for r in manifest_rows)
    ]
    if missing_required:
        raise RuntimeError(f"Missing required datasets: {', '.join(missing_required)}")

    print("✅ Step 01 complete.", flush=True)


if __name__ == "__main__":
    args = parse_args()
    pull(args.start, args.end)
