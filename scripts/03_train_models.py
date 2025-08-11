#!/usr/bin/env python3
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import yaml
import pandas as pd
from sklearn.metrics import mean_absolute_error, log_loss
from sklearn.linear_model import ElasticNet, LogisticRegression
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from utils.paths import DATA_FEATURES, DATA_PRED, ensure_dirs

def get_model(kind: str, params: dict):
    if kind == "lgbm_regressor":
        return LGBMRegressor(**params)
    if kind == "xgb_regressor":
        return XGBRegressor(**params)
    if kind == "elastic_net":
        return ElasticNet(**params)
    if kind == "logistic_regression":
        return LogisticRegression(**params)
    raise ValueError(f"Unknown model type: {kind}")

def train_receptions(cfg):
    df = pd.read_parquet(DATA_FEATURES / "wr_receptions_features.parquet")
    features = [c for c in df.columns if c.endswith(tuple([f"_l{w}" for w in cfg['rolling_windows']]))]
    X = df[features]
    y = df['y_next']

    mcfg = yaml.safe_load(open("config/models.yml"))['receptions']
    model = get_model(mcfg['type'], mcfg['params'])
    model.fit(X, y)
    pred = model.predict(X)
    mae = mean_absolute_error(y, pred)
    print(f"MAE(receptions) on in-sample features: {mae:.3f} (demo only)")
    ensure_dirs()
    joblib.dump(model, DATA_PRED / "wr_receptions_model.joblib")
    pd.DataFrame({"mu": pred}).to_parquet(DATA_PRED / "wr_receptions_in_sample.parquet", index=False)

def main():
    cfg = yaml.safe_load(open("config/features.yml"))
    train_receptions(cfg)

if __name__ == "__main__":
    main()
