#!/usr/bin/env python3
# Train simple models for pregame props using lag-only features
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMRegressor
import joblib

INP = Path("data/pregame/history_rolling_ma.csv.gz")
OUT = Path("models/pregame"); OUT.mkdir(parents=True, exist_ok=True)

TARGETS = {
  "qb_passing_yards": ("QB", "passing_yards"),
  "rb_rushing_yards": ("RB", "rushing_yards"),
  "wr_rec_yards":     ("WR", "receiving_yards"),
  "wrte_receptions":  ("WR", "receptions"),   # TE included via position startswith("TE") handled below
}

def pick_rows(df, pos_tag):
    if pos_tag == "QB":
        return df[df["position"].str.upper().fillna("").str.startswith("QB")]
    if pos_tag == "RB":
        return df[df["position"].str.upper().fillna("").str.startswith("RB")]
    if pos_tag == "WR":
        return df[df["position"].str.upper().fillna("").str.match(r"WR|TE")]
    return df

def main():
    if not INP.exists():
        raise SystemExit(f"missing {INP}")
    df = pd.read_csv(INP, low_memory=False)

    # features (lag + opponent allowed + is_home)
    feat_cols = [c for c in df.columns if c.startswith("plyr_") or c.startswith("opp_")] + ["is_home"]
    feat_cols = [c for c in feat_cols if c in df.columns]

    for model_name, (pos_tag, ycol) in TARGETS.items():
        d = pick_rows(df, pos_tag).dropna(subset=[ycol])
        if d.empty: 
            print(f"skip {model_name}: no rows"); 
            continue
        X = d[feat_cols].fillna(0)
        y = d[ycol].values
        groups = d["player_id"].astype(str)
        cv = GroupKFold(n_splits=5)
        model = LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=-1, subsample=0.8, colsample_bytree=0.8)
        for tr, va in cv.split(X, y, groups):
            model.fit(X.iloc[tr], y[tr],
                      eval_set=[(X.iloc[va], y[va])],
                      eval_metric="l2",
                      verbose=False)
        joblib.dump((model, feat_cols), OUT/f"{model_name}.joblib")
        print(f"✓ trained {model_name} -> {OUT/f'{model_name}.joblib'}")

if __name__ == "__main__":
    main() 
