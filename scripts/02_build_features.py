
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

def main():
    weekly = pd.read_parquet(DATA_RAW / 'weekly.parquet')

    use = weekly.copy()
    use = use.sort_values(['player_id','season','week'])

    wr = use[use['position'] == 'WR'].copy()
    g = wr.groupby('player_id')
    if 'targets' not in wr.columns:
        wr['targets'] = 0
    if 'routes_run' not in wr.columns:
        wr['routes_run'] = 0
    wr['targets_l5'] = rolling_mean(g, 'targets', 5)
    wr['routes_run_l5'] = rolling_mean(g, 'routes_run', 5)
    wr['receptions_next'] = g['receptions'].shift(-1)

    feats = wr.dropna(subset=['receptions_next'])[[
        'player_id','season','week','team','opponent','targets_l5','routes_run_l5','receptions_next'
    ]]
    feats.to_parquet(DATA_FEATURES / 'wr_receptions_features.parquet', index=False)
    print('built features → data/features/wr_receptions_features.parquet')

if __name__ == '__main__':
    main()
