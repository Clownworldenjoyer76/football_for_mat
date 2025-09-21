#!/usr/bin/env python3
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from utils.paths import DATA_PRED, DATA_ODDS

def main():
    # Primary and fallback prediction files
    primary_pred_file = DATA_PRED / 'wr_receptions_predictions.csv'
    fallback_pred_file = DATA_PRED / 'wrte_receptions.csv'

    if primary_pred_file.exists():
        preds_path = primary_pred_file
    elif fallback_pred_file.exists():
        preds_path = fallback_pred_file
        print(f"[INFO] Using fallback predictions file: {preds_path}")
    else:
        raise FileNotFoundError(
            f"Neither {primary_pred_file} nor {fallback_pred_file} found."
        )

    preds = pd.read_csv(preds_path)

    odds_path = DATA_ODDS / 'wr_receptions_odds.csv'
    if not odds_path.exists():
        sample = preds[['player_id']].drop_duplicates().head(10).copy()
        sample['line'] = 4.5
        sample['over_odds'] = -110
        sample['under_odds'] = -110
        sample.to_csv(odds_path, index=False)
    odds = pd.read_csv(odds_path)

    df = preds.merge(odds, on='player_id', how='inner')
    out_file = DATA_PRED / 'wr_receptions_with_odds.csv'
    df.to_csv(out_file, index=False)
    print(f'merged preds+odds → {out_file}')

if __name__ == '__main__':
    main()
