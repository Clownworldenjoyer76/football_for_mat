
#!/usr/bin/env python3
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from utils.paths import DATA_FEATURES, DATA_WAREHOUSE

def main():
    df = pd.read_parquet(DATA_FEATURES / 'wr_receptions_features.parquet')
    df = df.sort_values(['season','week'])

    X = df[['targets_l5','routes_run_l5']].fillna(0)
    y = df['receptions_next']

    model = LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31)
    model.fit(X, y)

    DATA_WAREHOUSE.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, DATA_WAREHOUSE / 'wr_receptions_lgbm.pkl')

    pred = model.predict(X)
    mae = mean_absolute_error(y, pred)
    (DATA_WAREHOUSE / 'metrics.txt').write_text(f'MAE: {mae:.3f}\n')
    print(f'trained model, MAE={mae:.3f}')

if __name__ == '__main__':
    main()
