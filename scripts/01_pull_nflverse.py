
#!/usr/bin/env python3
import sys, pathlib, argparse
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nfl_data_py as nfl
from utils.paths import DATA_RAW

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, required=True)
parser.add_argument('--end', type=int, required=True)

def main(args=None):
    a = parser.parse_args(args)
    years = list(range(a.start, a.end + 1))

    weekly = nfl.import_weekly_data(years)
    pbp    = nfl.import_pbp_data(years)
    rost   = nfl.import_rosters(years)
    sched  = nfl.import_schedules(years)

    weekly.to_parquet(DATA_RAW / 'weekly.parquet', index=False)
    pbp.to_parquet(DATA_RAW / 'pbp.parquet', index=False)
    rost.to_parquet(DATA_RAW / 'rosters.parquet', index=False)
    sched.to_parquet(DATA_RAW / 'schedules.parquet', index=False)
    print('wrote raw parquet files')

if __name__ == '__main__':
    main()
