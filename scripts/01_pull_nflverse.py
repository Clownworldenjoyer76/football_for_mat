#!/usr/bin/env python3
"""
Pull NFL data with nfl_data_py (2019..2024 by default).
Handles roster API name differences across versions (import_rosters vs import_weekly_rosters).
Writes CSVs to data/raw/.
"""
import argparse
import sys
import subprocess
import importlib
from pathlib import Path
import pandas as pd

def ensure_pkg():
    try:
        import nfl_data_py as nfl  # noqa
    except Exception as e:
        raise RuntimeError(f"Failed to import nfl_data_py: {e}")
    globals()['nfl'] = nfl  # bind

def get_rosters_module(nfl):
    if hasattr(nfl, 'import_rosters'):
        return nfl.import_rosters
    if hasattr(nfl, 'import_weekly_rosters'):
        return nfl.import_weekly_rosters
    avail = [x for x in dir(nfl) if x.startswith('import_')]
    raise AttributeError(f"nfl_data_py has no roster import. Available: {avail}")

def save(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    for col in df.select_dtypes(include='float').columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include='integer').columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    df.to_csv(path, index=False)
    print(f"✓ Wrote {path} ({len(df):,} rows)")

def run(year_start: int, year_end: int):
    ensure_pkg()
    nfl = globals()['nfl']

    years = list(range(year_start, year_end + 1))
    print(f"Years: {years}")

    out = Path('data/raw')
    out.mkdir(parents=True, exist_ok=True)

    print('Pulling weekly ...', flush=True)
    weekly = nfl.import_weekly_data(years)
    save(weekly, out / 'weekly.csv')

    print('Pulling play-by-play ...', flush=True)
    pbp = nfl.import_pbp_data(years)
    save(pbp, out / 'pbp.csv')

    print('Pulling rosters ...', flush=True)
    roster_fn = get_rosters_module(nfl)
    rosters = roster_fn(years)
    save(rosters, out / 'rosters.csv')

    print('Pulling schedules ...', flush=True)
    schedules = nfl.import_schedules(years)
    save(schedules, out / 'schedules.csv')

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', type=int, default=2019)
    ap.add_argument('--end', type=int, default=2024)
    return ap.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run(args.start, args.end)
