#!/usr/bin/env python3
"""
Pull NFL data with nfl_data_py (2019..2024 by default).
- Handles roster API differences (import_rosters vs import_weekly_rosters).
- Saves BOTH CSV and Parquet for each dataset to data/raw/.
"""
import argparse
from pathlib import Path
import pandas as pd

def bind_pkg():
    import nfl_data_py as nfl  # rely on version in environment
    return nfl

def get_rosters_fn(nfl):
    if hasattr(nfl, "import_rosters"):
        return nfl.import_rosters
    if hasattr(nfl, "import_weekly_rosters"):
        return nfl.import_weekly_rosters
    opts = [x for x in dir(nfl) if x.startswith("import_")]
    raise AttributeError(f"nfl_data_py has no roster import. Available: {opts}")

def save_both(df: pd.DataFrame, base: Path):
    base.parent.mkdir(parents=True, exist_ok=True)
    # light downcast
    for col in df.select_dtypes(include="float").columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include="integer").columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    # CSV
    df.to_csv(base.with_suffix(".csv"), index=False)
    # Parquet (fastparquet/pyarrow; index not needed)
    df.to_parquet(base.with_suffix(".parquet"), index=False)
    print(f"✓ Wrote {base.with_suffix('.csv').name} & {base.with_suffix('.parquet').name} ({len(df):,} rows)")

def run(year_start: int, year_end: int):
    nfl = bind_pkg()
    years = list(range(year_start, year_end + 1))
    print(f"Years: {years}")
    out = Path("data/raw")
    out.mkdir(parents=True, exist_ok=True)

    print("Pulling weekly ...", flush=True)
    weekly = nfl.import_weekly_data(years)
    save_both(weekly, out / "weekly")

    print("Pulling play-by-play ...", flush=True)
    pbp = nfl.import_pbp_data(years)
    save_both(pbp, out / "pbp")

    print("Pulling rosters ...", flush=True)
    rosters = get_rosters_fn(nfl)(years)
    save_both(rosters, out / "rosters")

    print("Pulling schedules ...", flush=True)
    schedules = nfl.import_schedules(years)
    save_both(schedules, out / "schedules")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2019)
    ap.add_argument("--end", type=int, default=2024)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(args.start, args.end)
