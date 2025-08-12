#!/usr/bin/env python3
import glob
import os
from pathlib import Path
import pandas as pd

DATA_RAW = Path('data/raw')
OUT_DIR = Path('data/features')
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_many(patterns):
    files = []
    for pat in patterns:
        files.extend(sorted(glob.glob(str(DATA_RAW / pat))))
    if not files:
        raise FileNotFoundError('No weekly_* files found in data/raw')
    dfs = []
    for fp in files:
        ext = os.path.splitext(fp)[1].lower()
        if ext == '.parquet':
            dfs.append(pd.read_parquet(fp))
        elif ext in ['.gz', '.csv']:
            dfs.append(pd.read_csv(fp))
    return pd.concat(dfs, ignore_index=True, copy=False)

def downcast(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.select_dtypes(include='float').columns:
        df[c] = pd.to_numeric(df[c], downcast='float')
    for c in df.select_dtypes(include='integer').columns:
        df[c] = pd.to_numeric(df[c], downcast='integer')
    return df

def main():
    df = load_many(['weekly_*.parquet', 'weekly_*.csv', 'weekly_*.csv.gz'])
    df = downcast(df)
    out_base = OUT_DIR / 'weekly_all'
    df.to_parquet(out_base.with_suffix('.parquet'), index=False, compression='snappy')
    df.to_csv(out_base.with_suffix('.csv.gz'), index=False, compression='gzip')
    # Manifest: row counts by season/week
    manifest = df.groupby(['season', 'week']).size().reset_index(name='rows')
    manifest.to_csv(OUT_DIR / 'manifest.csv', index=False)
    print(f'✓ Wrote features and manifest ({len(df):,} rows)')

if __name__ == '__main__':
    main()
