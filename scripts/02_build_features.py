#!/usr/bin/env python3
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from utils.paths import DATA_RAW, DATA_FEATURES

def rolling_mean(g, col, w=5):
    return (
        g[col]
        .rolling(w, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

def pick_col(df, candidates, default=None):
    for c in candidates:
        if c in df.columns:
            return c
    if default is not None:
        return default
    raise KeyError(f"None of the candidate columns present: {candidates}. Available: {list(df.columns)}")


def main():
    # Load weekly from parquet if present, else csv
    weekly_path_parq = DATA_RAW / 'weekly.parquet'
    weekly_path_csv = DATA_RAW / 'weekly.csv'
    if weekly_path_parq.exists():
        weekly = pd.read_parquet(weekly_path_parq)
    elif weekly_path_csv.exists():
        weekly = pd.read_csv(weekly_path_csv)
    else:
        raise FileNotFoundError('weekly.parquet or weekly.csv not found in data/raw')

    use = weekly.copy()
    use = use.sort_values(['player_id','season','week'])

    # Determine team/opponent column names for this nfl_data_py version
    team_col = pick_col(use, ['team','recent_team','recent_team_abbr','posteam'])
    opp_col  = pick_col(use, ['opponent','opponent_team','defteam','opp_team','opponent_abbr'])

    wr = use[use['position'] == 'WR'].copy()
    g = wr.groupby('player_id')

    if 'targets' not in wr.columns:
        wr['targets'] = 0
    if 'routes_run' not in wr.columns:
        wr['routes_run'] = 0
    if 'receptions' not in wr.columns:
        wr['receptions'] = 0

    wr['targets_l5'] = rolling_mean(g, 'targets', 5)
    wr['routes_run_l5'] = rolling_mean(g, 'routes_run', 5)
    wr['receptions_next'] = g['receptions'].shift(-1)

    feats = wr.dropna(subset=['receptions_next'])[
        ['player_id','season','week', team_col, opp_col, 'targets_l5','routes_run_l5','receptions_next']
    ].rename(columns={team_col:'team', opp_col:'opponent'})

    DATA_FEATURES.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(DATA_FEATURES / 'wr_receptions_features.parquet', index=False)
    print('built features → data/features/wr_receptions_features.parquet')

if __name__ == '__main__':
    main()
