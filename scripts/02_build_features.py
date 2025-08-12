#!/usr/bin/env python3
"""
Build features from chunked raw files.
- Auto-loads & concatenates data/raw/weekly_*.parquet (or csv/csv.gz).
- Falls back to single weekly.{parquet,csv,csv.gz} if present.
- Robust team/opponent column detection.
- Writes data/features/wr_receptions_features.parquet
"""
import glob
from pathlib import Path
import pandas as pd

DATA_RAW = Path('data/raw')
OUT_DIR = Path('data/features')
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_many(prefix):
    # try parquet chunks
    paths = sorted(glob.glob(str(DATA_RAW / f'{prefix}_*.parquet')))
    if paths:
        return pd.concat((pd.read_parquet(p) for p in paths), ignore_index=True)
    # try csv.gz chunks
    paths = sorted(glob.glob(str(DATA_RAW / f'{prefix}_*.csv.gz')))
    if paths:
        return pd.concat((pd.read_csv(p) for p in paths), ignore_index=True)
    # fallbacks to single files
    for suf in ('.parquet', '.csv.gz', '.csv'):
        p = DATA_RAW / f'{prefix}{suf}'
        if p.exists():
            if suf == '.parquet':
                return pd.read_parquet(p)
            else:
                return pd.read_csv(p)
    raise FileNotFoundError(f'No files found for {prefix} in {DATA_RAW}')

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def main():
    weekly = load_many('weekly')
    # minimal WR receptions example features like before
    team_col = pick_col(weekly, ['team','recent_team','posteam'])
    opp_col  = pick_col(weekly, ['opponent','opponent_team','defteam'])
    # keep safe existence
    base_cols = ['player_id','player_name','season','week','position','receptions','targets','routes_run']
    use_cols = [c for c in base_cols if c in weekly.columns]
    if team_col: use_cols.append(team_col)
    if opp_col: use_cols.append(opp_col)
    df = weekly[use_cols].copy()

    # sort & rolling on WRs
    df = df[df.get('position','').eq('WR')] if 'position' in df.columns else df
    df.sort_values(['player_id','season','week'], inplace=True)

    if 'targets' in df.columns:
        df['targets_l5'] = (
            df.groupby('player_id')['targets']
              .rolling(5, min_periods=1).mean()
              .reset_index(level=0, drop=True)
        )
    if 'routes_run' in df.columns:
        df['routes_run_l5'] = (
            df.groupby('player_id')['routes_run']
              .rolling(5, min_periods=1).mean()
              .reset_index(level=0, drop=True)
        )
    if 'receptions' in df.columns:
        df['y_next'] = df.groupby('player_id')['receptions'].shift(-1)
        df = df.dropna(subset=['y_next'])

    OUT = OUT_DIR / 'wr_receptions_features.parquet'
    df.to_parquet(OUT, index=False)
    print(f'Wrote {OUT} ({len(df):,} rows)')

if __name__ == '__main__':
    main()
