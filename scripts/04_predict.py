#!/usr/bin/env python3
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
import numpy as np
import yaml
from utils.paths import DATA_FEATURES, DATA_PRED, ensure_dirs

def main():
    ensure_dirs()
    model = joblib.load(DATA_PRED / "wr_receptions_model.joblib")
    feats = pd.read_parquet(DATA_FEATURES / "wr_receptions_features.parquet")

    feature_cols = [c for c in feats.columns if "_l" in c and c != "y_next"]
    mu = model.predict(feats[feature_cols])
    # crude sigma proxy from residuals (guarded by floor)
    sigma = np.maximum(np.std(feats['y_next'] - mu), 0.75)
    out = feats[['player_id','player_name','season','week']].copy()
    out['mu'] = mu
    out['sigma'] = sigma
    out.to_parquet(DATA_PRED / "wr_receptions_predictions.parquet", index=False)
    print("✅ Wrote predictions → data/predictions/wr_receptions_predictions.parquet")

if __name__ == "__main__":
    main()
