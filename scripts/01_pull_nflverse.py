#!/usr/bin/env python3
"""
Chunked NFL pulls to avoid huge files.
- Handles roster API differences (import_rosters vs import_weekly_rosters).
- Saves BOTH Parquet (snappy) and gzip CSV, partitioned by year ranges.
Usage:
  python scripts/01_pull_nflverse_chunked.py --start 2010 --end 2024 --chunk-years 3
Outputs (examples):
  data/raw/weekly_2010_2012.parquet, weekly_2010_2012.csv.gz, ...
"""
import argparse
from pathlib import Path
import pandas as pd

def bind_pkg():
    import nfl_data_py as nfl
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
    # parquet (compressed)
    df.to_parquet(base.with_suffix(".parquet"), index=False, compression="snappy")
    # csv.gz
    df.to_csv(base.with_suffix(".csv.gz"), index=False, compression="gzip")
    print(f"✓ {base.name} -> wrote parquet & csv.gz ({len(df):,} rows)")

def year_chunks(start: int, end: int, step: int):
    s = start
    while s <= end:
        e = min(end, s + step - 1)
        yield list(range(s, e + 1)), s, e
        s = e + 1

def run(start: int, end: int, chunk: int):
    nfl = bind_pkg()
    rost_fn = get_rosters_fn(nfl)
    out = Path("data/raw")

    for years, ys, ye in year_chunks(start, end, chunk):
        tag = f"{ys}_{ye}"
        print(f"== Years {tag} ==")

        print("Weekly ...")
        weekly = nfl.import_weekly_data(years)
        save_both(weekly, out / f"weekly_{tag}")

        print("Play-by-play ...")
        pbp = nfl.import_pbp_data(years)
        save_both(pbp, out / f"pbp_{tag}")

        print("Rosters ...")
        rosters = rost_fn(years)
        save_both(rosters, out / f"rosters_{tag}")

        print("Schedules ...")
        schedules = nfl.import_schedules(years)
        save_both(schedules, out / f"schedules_{tag}")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2019)
    ap.add_argument("--end", type=int, default=2024)
    ap.add_argument("--chunk-years", type=int, default=3, help="years per chunk")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(args.start, args.end, args.chunk_years)
