#!/usr/bin/env python3
"""
04_generate_props.py
Reads predictions & metrics from step 03.
Generates prop lines (nearest 0.5) and probability over/under using RMSE as sigma.
Saves props to data/props/props_<market>.csv
"""
from pathlib import Path
import pandas as pd
import numpy as np
import json
from scipy.stats import norm

PRED_DIR = Path("data/predictions")
METRICS_DIR = Path("output")
OUT_DIR = Path("data/props")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_rmse(market):
    met_path = METRICS_DIR / f"{market}_metrics.json"
    if not met_path.exists():
        return None
    with open(met_path) as f:
        m = json.load(f)
    return m.get("metrics", {}).get("RMSE")

def process_market(pred_file):
    market = pred_file.stem
    df = pd.read_csv(pred_file)
    rmse = load_rmse(market)
    if rmse is None or rmse <= 0:
        rmse = max(1.0, np.std(df["y_true"] - df["y_pred"]))  # fallback

    # prop line rounded to nearest 0.5
    df["line"] = (df["y_pred"] * 2).round() / 2

    # probability over/under
    df["prob_over"] = 1 - norm.cdf(df["line"], loc=df["y_pred"], scale=rmse)
    df["prob_under"] = 1 - df["prob_over"]

    # make a prop label like "receptions", "passing_yards" based on market name
    df["prop"] = market

    keep_cols = [c for c in ["player_name","team","recent_team","opponent","opponent_team","season","week","prop","line","y_pred","prob_over","prob_under"] if c in df.columns]
    df_out = df[keep_cols]

    out_path = OUT_DIR / f"props_{market}.csv"
    df_out.to_csv(out_path, index=False)
    print(f"✓ Wrote {out_path} ({len(df_out)} rows)")

def main():
    for pred_file in PRED_DIR.glob("*.csv"):
        process_market(pred_file)

if __name__ == "__main__":
    main()
