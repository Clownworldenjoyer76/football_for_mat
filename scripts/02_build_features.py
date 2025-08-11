#!/usr/bin/env python3
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from utils.paths import DATA_RAW, DATA_FEATURES, ensure_dirs
import yaml

def rolling_mean(group, col, window):
    return (
        group[col]
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

def main():
    ensure_dirs()
    cfg = yaml.safe_load(open("config/features.yml"))
    weekly = pd.read_parquet(DATA_RAW / "weekly.parquet")

    # Example: WR features
    wr = weekly[weekly['position'] == 'WR'].copy()
    wr.sort_values(['player_id','season','week'], inplace=True)

    for w in cfg.get('rolling_windows', [3,5]):
        for col in ['targets','routes_run','receptions']:
            if col in wr.columns:
                wr[f'{col}_l{w}'] = (
                    wr.groupby('player_id', group_keys=False)
                      .apply(lambda g: rolling_mean(g, col, w))
                )

    # Label: next game receptions
    wr['y_next'] = wr.groupby('player_id')['receptions'].shift(-1)

    # Guards
    wr = wr[wr.get('snaps', 0) >= cfg.get('guards', {}).get('min_snaps', 5)]
    qlow, qhi = cfg.get('guards', {}).get('clip_quantiles', [0.005, 0.995])
    num_cols = wr.select_dtypes('number').columns
    wr[num_cols] = wr[num_cols].clip(lower=wr[num_cols].quantile(qlow), upper=wr[num_cols].quantile(qhi), axis=1)

    out = wr.dropna(subset=['y_next'])
    out.to_parquet(DATA_FEATURES / "wr_receptions_features.parquet", index=False)
    print("✅ Built features → data/features/wr_receptions_features.parquet")

if __name__ == "__main__":
    main()
