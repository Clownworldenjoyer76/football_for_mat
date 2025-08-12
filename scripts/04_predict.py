
#!/usr/bin/env python3
import sys, pathlib, numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
from utils.paths import DATA_FEATURES, DATA_WAREHOUSE, DATA_PRED

def main():
    feats = pd.read_parquet(DATA_FEATURES / 'wr_receptions_features.parquet')
    X = feats[['targets_l5','routes_run_l5']].fillna(0)
    model = joblib.load(DATA_WAREHOUSE / 'wr_receptions_lgbm.pkl')
    mu = model.predict(X)

    y = feats['receptions_next']
    sigma = float(max(0.75, np.std(y - mu))) if len(y) > 1 else 1.0

    out = feats[['player_id','season','week']].copy()
    out['mu'] = mu
    out['sigma'] = sigma
    DATA_PRED.mkdir(parents=True, exist_ok=True)
    out.to_csv(DATA_PRED / 'wr_receptions_predictions.csv', index=False)
    print('predictions → data/predictions/wr_receptions_predictions.csv')

if __name__ == '__main__':
    main()
