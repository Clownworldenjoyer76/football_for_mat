
#!/usr/bin/env python3
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from utils.paths import DATA_PRED, DATA_ODDS

def main():
    preds = pd.read_csv(DATA_PRED / 'wr_receptions_predictions.csv')
    odds_path = DATA_ODDS / 'wr_receptions_odds.csv'
    if not odds_path.exists():
        sample = preds[['player_id']].drop_duplicates().head(10).copy()
        sample['line'] = 4.5
        sample['over_odds'] = -110
        sample['under_odds'] = -110
        sample.to_csv(odds_path, index=False)
    odds = pd.read_csv(odds_path)

    df = preds.merge(odds, on='player_id', how='inner')
    df.to_csv(DATA_PRED / 'wr_receptions_with_odds.csv', index=False)
    print('merged preds+odds → data/predictions/wr_receptions_with_odds.csv')

if __name__ == '__main__':
    main()
