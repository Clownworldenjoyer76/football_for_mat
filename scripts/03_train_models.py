#!/usr/bin/env python3
"""
Multi-market model training on weekly_all.* without renaming columns.
Markets:
  - qb_passing_yards
  - rb_rushing_yards
  - wr_rec_yards
  - wrte_receptions
Outputs:
  data/predictions/<market>.csv
  output/<market>_metrics.json
  output/metrics_summary.json
"""
from pathlib import Path
import json
import math
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingRegressor

IN_PARQUET = Path("data/features/weekly_all.parquet")
IN_CSVGZ   = Path("data/features/weekly_all.csv.gz")
PRED_DIR   = Path("data/predictions")
OUT_DIR    = Path("output")
PRED_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- utilities ----
def load_features() -> pd.DataFrame:
    if IN_PARQUET.exists():
        return pd.read_parquet(IN_PARQUET)
    if IN_CSVGZ.exists():
        return pd.read_csv(IN_CSVGZ)
    raise FileNotFoundError("weekly_all.{parquet|csv.gz} not found")

def has_cols(df, cols):
    return [c for c in cols if c in df.columns]

def add_rolling(df: pd.DataFrame, by: str, sort_cols, base_cols, windows=(3,5)):
    df = df.sort_values(sort_cols).copy()
    for col in base_cols:
        if col not in df.columns:
            continue
        g = df.groupby(by)[col]
        for w in windows:
            newc = f"{col}_l{w}"
            df[newc] = g.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
    return df

def train_val_split_by_season(df: pd.DataFrame, season_col="season"):
    latest = int(df[season_col].max())
    trn = df[df[season_col] < latest].copy()
    val = df[df[season_col] == latest].copy()
    return trn, val, latest

def fit_reg(X, y):
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_depth=None,
        learning_rate=0.06,
        max_iter=400,
        early_stopping=True,
        random_state=42
    )
    model.fit(X, y)
    return model

def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4)}

# ---- market configs ----
MARKETS = [
    {
        "name": "qb_passing_yards",
        "position_in": ["QB"],
        "target_candidates": ["passing_yards", "pass_yards"],
        "feature_candidates": [
            "attempts", "pass_attempts", "completions",
            "passing_yards", "pass_yards", "sacks", "sack_yards",
            "air_yards", "cpoe", "epa", "success", "time_to_throw",
            "team_pass_attempts", "team_pass_plays"
        ]
    },
    {
        "name": "rb_rushing_yards",
        "position_in": ["RB"],
        "target_candidates": ["rushing_yards", "rush_yards"],
        "feature_candidates": [
            "carries", "rush_attempts", "rushing_yards", "rush_yards",
            "targets", "receptions", "snap_pct", "snap_share",
            "team_rush_attempts", "team_rush_plays", "redzone_carries"
        ]
    },
    {
        "name": "wr_rec_yards",
        "position_in": ["WR"],
        "target_candidates": ["receiving_yards", "rec_yards"],
        "feature_candidates": [
            "targets", "receptions", "routes_run", "routes", "air_yards",
            "yac", "aDoT", "target_share", "route_rate", "snap_pct", "snap_share",
            "team_pass_attempts", "team_pass_plays"
        ]
    },
    {
        "name": "wrte_receptions",
        "position_in": ["WR","TE"],
        "target_candidates": ["receptions"],
        "feature_candidates": [
            "targets", "receptions", "routes_run", "routes", "air_yards",
            "yac", "aDoT", "target_share", "route_rate", "snap_pct", "snap_share",
            "team_pass_attempts", "team_pass_plays"
        ]
    },
]

# base columns to create L3/L5 for when present
ROLL_BASES = [
    "attempts","pass_attempts","completions","passing_yards","pass_yards",
    "carries","rush_attempts","rushing_yards","rush_yards",
    "targets","receptions","routes_run","routes",
    "receiving_yards","rec_yards"
]

ID_COLS = [c for c in ["player_id","player_name","gsis_id","first_name","last_name"]]
AUX_KEEP = [c for c in ["team","recent_team","posteam","opponent","opponent_team","defteam","season","week","game_id"]]

def select_first_existing(cols, df):
    for c in cols:
        if c in df.columns:
            return c
    return None

def build_market(df0: pd.DataFrame, cfg: dict):
    df = df0.copy()
    # pick name columns
    name_col = select_first_existing(["player_name","player_display_name","name"], df)
    if name_col and "player_name" not in df.columns:
        df["player_name"] = df[name_col]

    # filter by position
    pos_col = select_first_existing(["position","pos","position_group"], df)
    if pos_col and cfg["position_in"]:
        df = df[df[pos_col].isin(cfg["position_in"])].copy()

    # ensure season/week numeric
    for c in ["season","week"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # rolling features per player
    pid_col = select_first_existing(["player_id","gsis_id"], df) or "player_id"
    if pid_col not in df.columns:
        return None, "missing_player_id"
    df = add_rolling(df, by=pid_col, sort_cols=[pid_col,"season","week"], base_cols=ROLL_BASES, windows=(3,5))

    # target y_next
    tgt = select_first_existing(cfg["target_candidates"], df)
    if not tgt:
        return None, "missing_target"
    df = df.sort_values([pid_col,"season","week"]).copy()
    df["y_next"] = df.groupby(pid_col)[tgt].shift(-1)

    # feature set: present raw + their L3/L5 if exist
    base_feats = has_cols(df, cfg["feature_candidates"])
    l3 = [f"{c}_l3" for c in base_feats if f"{c}_l3" in df.columns]
    l5 = [f"{c}_l5" for c in base_feats if f"{c}_l5" in df.columns]
    feats = list(dict.fromkeys(base_feats + l3 + l5))  # preserve order

    # guard: need at least some features
    if len(feats) == 0:
        return None, "no_features"

    # drop rows without target
    data = df.dropna(subset=["y_next"]).copy()

    # split
    trn, val, latest_season = train_val_split_by_season(data, "season")
    if len(trn) == 0 or len(val) == 0:
        return None, "insufficient_split"

    # train
    X_trn = trn[feats]
    y_trn = trn["y_next"]
    X_val = val[feats]
    y_val = val["y_next"]

    model = fit_reg(X_trn, y_trn)
    pred_val = model.predict(X_val)

    m = metrics(y_val, pred_val)

    # predictions dataframe (latest season)
    keep_cols = has_cols(val, ["season","week","game_id","team","recent_team","posteam","opponent","opponent_team","defteam","player_id","player_name"])
    pred_df = val[keep_cols].copy()
    pred_df["y_true"] = y_val.values
    pred_df["y_pred"] = pred_val

    return {"pred": pred_df, "metrics": m, "features_used": feats, "latest_season": latest_season}, None

def main():
    df = load_features()
    summary = {}
    for cfg in MARKETS:
        name = cfg["name"]
        res, err = build_market(df, cfg)
        if err:
            summary[name] = {"status": "skipped", "reason": err}
            continue

        # save predictions
        pred_path = PRED_DIR / f"{name}.csv"
        res["pred"].to_csv(pred_path, index=False)

        # save metrics
        met_path = OUT_DIR / f"{name}_metrics.json"
        with open(met_path, "w") as f:
            json.dump({
                "market": name,
                "latest_season": res["latest_season"],
                "metrics": res["metrics"],
                "n_pred_rows": int(len(res["pred"])),
                "n_features": int(len(res["features_used"]))
            }, f, indent=2)

        summary[name] = {"status": "ok", **res["metrics"], "rows": int(len(res["pred"]))}

    with open(OUT_DIR / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("✓ Trained markets:", ", ".join(summary.keys()))

if __name__ == "__main__":
    main()
