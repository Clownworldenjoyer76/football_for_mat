#!/usr/bin/env python3
"""
Step 01 — Pull core NFL datasets and write snapshots under data/raw/nflverse.

- Pulls weekly, pbp, rosters, schedules using nfl_data_py
- Defensive against version differences (tries multiple loader names)
- Handles per-year 404s gracefully (skips missing years, continues)
- Writes both dated snapshots and *_latest.csv.gz files
- Produces manifest_latest.csv.gz summarizing what was written
"""

from __future__ import annotations
import argparse
import gzip
import os
import shutil
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
    ap.add_argument("--end", type=int, default=None, help="End year (e.g., 2025)")
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


def fetch_per_year_concat(fn: Callable, years: List[int], base_kwargs: Dict) -> Tuple[pd.DataFrame, List[int], List[int]]:
    """
    Call the nfl_data_py loader one season at a time, concatenating results.
    Skip seasons that 404 (not published yet).
    Returns: (concat_df, succeeded_years, skipped_years)
    """
    import urllib.error
    try:
        import requests  # present in requirements; used for exception type
        RequestsHTTPError = requests.exceptions.HTTPError
    except Exception:  # keep working even if requests isn't imported properly
        class RequestsHTTPError(Exception): ...
        pass

    frames: List[pd.DataFrame] = []
    ok_years: List[int] = []
    skipped_years: List[int] = []

    for yr in years:
        # Try variants: years=[yr], then year=yr, then season=yr
        called = False
        for variant in ("years", "year", "season"):
            try_kwargs = dict(base_kwargs)
            try_kwargs.pop("years", None)
            try_kwargs.pop("year", None)
            try_kwargs.pop("season", None)
            if variant == "years":
                try_kwargs["years"] = [yr]
            else:
                try_kwargs[variant] = yr

            try:
                df = fn(**try_kwargs)
                called = True
                if not isinstance(df, pd.DataFrame):
                    df = pd.DataFrame(df)
                frames.append(df)
                ok_years.append(yr)
                break  # variant loop
            except (urllib.error.HTTPError, RequestsHTTPError) as e:
                # Skip only if it's a clear 404
                msg = str(e)
                if "404" in msg or "Not Found" in msg:
                    skipped_years.append(yr)
                    break  # try next year, not next variant
                else:
                    # For non-404 HTTP errors, re-raise
                    raise
            except TypeError:
                # Wrong signature for this variant; try next variant
                continue

        if not called and yr not in skipped_years:
            # If we tried variants and still failed (e.g., TypeError everywhere), let it bubble as a TypeError
            # or mark as skipped silently; here we mark skipped to keep job resilient
            skipped_years.append(yr)

    if frames:
        out = pd.concat(frames, ignore_index=True, sort=False)
    else:
        out = pd.DataFrame()

    return out, ok_years, skipped_years


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
            print(f"   resolved → {fn.__name__}; pulling seasons {yrs[0]}–{yrs[-1]} (per-year)…", flush=True)
            df, ok_years, skipped_years = fetch_per_year_concat(fn, yrs, t.kwargs)
            if len(skipped_years) > 0:
                note = f"skipped years: {','.join(map(str, skipped_years))}"
                print(f"   NOTE: {t.name} skipped {skipped_years} (404 not published).", flush=True)
            if len(df) > 0:
                df = df.drop_duplicates().reset_index(drop=True)
                df = add_generated_columns(df, t.name)
                snap_path, latest_path = save_snapshot_and_latest(df, t.name, OUT_ROOT, stamp)
                rows_written = int(len(df))
                print(f"   {t.name}: {rows_written:,} rows → {snap_path.name} (+ {latest_path.name})", flush=True)
            else:
                # No data at all (e.g., all years 404) → behave like missing but without error for non-required
                if t.required:
                    raise RuntimeError(f"{t.name}: no data returned for any year {yrs[0]}–{yrs[-1]}")
                note = "no data; all years skipped (404)"
                print(f"   WARNING: {t.name} had no available years; continuing.", flush=True)
        else:
            # No loader resolved — fallback: if a _latest already exists, record and copy dated snapshot
            fallback_latest = OUT_ROOT / f"{t.name}_latest.csv.gz"
            fallback_rows = file_rows_if_exists(fallback_latest)
            if fallback_rows is not None:
                candidate_snap = OUT_ROOT / f"{t.name}_{stamp}.csv.gz"
                if not candidate_snap.exists():
                    candidate_snap.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(fallback_latest, candidate_snap)
                    print(f"   {t.name}: copied {fallback_latest.name} → {candidate_snap.name}", flush=True)
                latest_path = fallback_latest
                snap_path = candidate_snap
                rows_written = fallback_rows
                note = "used existing latest; loader missing"
                print(f"   WARNING: loader missing, using existing latest ({rows_written} rows).", flush=True)
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

    # Require required datasets to have >0 rows
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
