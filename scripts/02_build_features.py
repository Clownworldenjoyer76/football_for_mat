#!/usr/bin/env python3
"""
Build features (pass-through): concatenate chunked weekly files and keep original column names.
- Reads data/raw/weekly_*.parquet (preferred), falling back to .csv or .csv.gz
- Writes data/features/weekly_all.parquet (snappy) and weekly_all.csv.gz
- No renaming of columns
"""
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
        try:
            if ext == '.parquet':
                dfs.append(pd.read_parquet(fp))
            elif ext == '.gz' or ext == '.csv':
                dfs.append(pd.read_csv(fp))
            else:
                continue
        except Exception as e:
            raise RuntimeError(f'Failed to read {fp}: {e}')
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
    print(f'✓ Wrote {out_base.with_suffix(".parquet").name} & .csv.gz ({len(df):,} rows)')

if __name__ == '__main__':
    main()
