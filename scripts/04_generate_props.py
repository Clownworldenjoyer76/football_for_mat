#!/usr/bin/env python3
"""
04_generate_props.py
Reads predictions & metrics from step 03.
Generates prop lines (nearest 0.5) and probability over/under using RMSE as sigma.
Saves per-market props to data/props/props_<market>.csv
Also writes a combined data/props/props_current.csv for downstream steps and
appends the current props to data/props/history_pending.csv for future calibration.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import json
from scipy.stats import norm
from datetime import datetime

PRED_DIR = Path("data/predictions")
METRICS_DIR = Path("output")
OUT_DIR = Path("data/props")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_rmse(market: str):
    met_path = METRICS_DIR / f"{market}_metrics.json"
    if not met_path.exists():
        return None
    try:
        with open(met_path) as f:
            m = json.load(f)
        return m.get("metrics", {}).get("RMSE")
    except Exception:
        return None

def process_market(pred_file: Path) -> pd.DataFrame:
    market = pred_file.stem
    df = pd.read_csv(pred_file)

    # Fallback RMSE if metrics missing
    rmse = load_rmse(market)
    if rmse is None or rmse <= 0:
        if {"y_true", "y_pred"}.issubset(df.columns):
            rmse = max(1.0, float(np.std(df["y_true"] - df["y_pred"])))
        else:
            rmse = 10.0  # conservative default if residuals unavailable

    # prop line rounded to nearest 0.5
    if "y_pred" not in df.columns:
        raise ValueError(f"{pred_file} missing y_pred column")
    df["line"] = (df["y_pred"] * 2).round() / 2

    # probability over/under
    df["prob_over"] = 1 - norm.cdf(df["line"], loc=df["y_pred"], scale=rmse)
    df["prob_under"] = 1 - df["prob_over"]

    # market label
    df["market"] = market

    # output columns (only keep what exists)
    keep_cols_order = [
        "player_id", "player_name", "team", "recent_team", "opponent", "opponent_team",
        "season", "week", "game_id",
        "market", "line", "y_pred", "prob_over", "prob_under"
    ]
    keep_cols = [c for c in keep_cols_order if c in df.columns]
    df_out = df[keep_cols].copy()

    # per-market output
    out_path = OUT_DIR / f"props_{market}.csv"
    df_out.to_csv(out_path, index=False)
    print(f"✓ Wrote {out_path} ({len(df_out)} rows)")

    return df_out

def append_history_pending(df_all: pd.DataFrame):
    """
    Append current props to data/props/history_pending.csv with a run timestamp.
    De-dupe by a reasonable key if columns are available.
    """
    pending_path = OUT_DIR / "history_pending.csv"
    df = df_all.copy()

    # add run timestamp
    if "run_ts" not in df.columns:
        df["run_ts"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    # ensure required fields exist
    if "market" not in df.columns:
        df["market"] = "unknown"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if pending_path.exists():
        prev = pd.read_csv(pending_path)
        combined = pd.concat([prev, df], ignore_index=True)
    else:
        combined = df

    # drop duplicates using available identifier columns
    subset_cols_pref = ["season", "week", "market", "player_id", "game_id", "line", "run_ts"]
    subset = [c for c in subset_cols_pref if c in combined.columns]
    if subset:
        combined = combined.drop_duplicates(subset=subset, keep="last")
    else:
        combined = combined.drop_duplicates()

    combined.to_csv(pending_path, index=False)
    print(f"✓ Appended to {pending_path} ({len(df)} new rows)")

def main():
    all_rows = []

    # Process each market prediction file
    for pred_file in sorted(PRED_DIR.glob("*.csv")):
        try:
            df_out = process_market(pred_file)
            all_rows.append(df_out)
        except Exception as e:
            print(f"! Skipping {pred_file}: {e}")

    if not all_rows:
        print("! No prediction files found; nothing to write.")
        return

    # Combined output for downstream steps
    combined = pd.concat(all_rows, ignore_index=True)
    combined_out = OUT_DIR / "props_current.csv"
    combined.to_csv(combined_out, index=False)
    print(f"✓ Wrote {combined_out} ({len(combined)} total rows)")

    # Append to history_pending for future calibration
    try:
        append_history_pending(combined)
    except Exception as e:
        print(f"! Failed to append to history_pending.csv: {e}")

if __name__ == "__main__":
    main()
