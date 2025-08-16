#!/usr/bin/env python3
# Train pregame models (lag-only features) — LightGBM, no verbose kw
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMRegressor
import joblib

INP = Path("data/pregame/history_rolling_ma.csv.gz")
OUT = Path("models/pregame"); OUT.mkdir(parents=True, exist_ok=True)

# target mapping: model_name -> (position tag, label column)
TARGETS = {
    "qb_passing_yards": ("QB", "passing_yards"),
    "rb_rushing_yards": ("RB", "rushing_yards"),
    "wr_rec_yards":     ("WR", "receiving_yards"),
    "wrte_receptions":  ("WR", "receptions"),  # WR + TE together
}

def pick_rows(df, pos_tag):
    pos = df["position"].astype(str).str.upper().fillna("")
    if pos_tag == "QB":
        return df[pos.str.startswith("QB")]
    if pos_tag == "RB":
        return df[pos.str.startswith("RB")]
    if pos_tag == "WR":
        # WR/TE combined
        return df[pos.str.match(r"WR|TE")]
    return df

def main():
    if not INP.exists():
        raise SystemExit(f"missing {INP}")
    df = pd.read_csv(INP, low_memory=False)

    # feature set: rolling player stats + opponent allowed + is_home (if present)
    feat_cols = [c for c in df.columns if c.startswith("plyr_") or c.startswith("opp_")]
    if "is_home" in df.columns:
        feat_cols.append("is_home")
    if not feat_cols:
        raise SystemExit("no feature columns starting with 'plyr_' or 'opp_' found")

    if "player_id" not in df.columns:
        raise SystemExit("history file missing player_id")

    for model_name, (pos_tag, ycol) in TARGETS.items():
        if ycol not in df.columns:
            print(f"skip {model_name}: missing label '{ycol}'")
            continue

        d = pick_rows(df, pos_tag).dropna(subset=[ycol])
        if d.empty:
            print(f"skip {model_name}: no rows after position filter")
            continue

        X = d[feat_cols].fillna(0)
        y = d[ycol].values
        groups = d["player_id"].astype(str)

        # LightGBM regressor; quiet training (no verbose kw)
        model = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )

        # GroupKFold by player to avoid leakage
        cv = GroupKFold(n_splits=5)
        for tr_idx, va_idx in cv.split(X, y, groups):
            model.fit(
                X.iloc[tr_idx], y[tr_idx],
                eval_set=[(X.iloc[va_idx], y[va_idx])],
                eval_metric="l2",
            )

        joblib.dump((model, feat_cols), OUT / f"{model_name}.joblib")
        print(f"✓ trained {model_name} → {OUT / f'{model_name}.joblib'}")

if __name__ == "__main__":
    main()
