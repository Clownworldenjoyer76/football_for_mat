#!/usr/bin/env python3
import argparse
import pandas as pd
import nfl_data_py as nfl
from utils.paths import DATA_RAW, ensure_dirs

def main(year_start: int, year_end: int):
    ensure_dirs()
    years = list(range(year_start, year_end + 1))

    weekly = nfl.import_weekly_data(years)
    pbp    = nfl.import_pbp_data(years)
    rost   = nfl.import_rosters(years)
    sched  = nfl.import_schedules(years)

    weekly.to_parquet(DATA_RAW / "weekly.parquet", index=False)
    pbp.to_parquet(DATA_RAW / "pbp.parquet", index=False)
    rost.to_parquet(DATA_RAW / "rosters.parquet", index=False)
    sched.to_parquet(DATA_RAW / "schedules.parquet", index=False)
    print("✅ Saved raw pulls to data/raw/")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2019)
    ap.add_argument("--end", type=int, default=2024)
    args = ap.parse_args()
    main(args.start, args.end)
