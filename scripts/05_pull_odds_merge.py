#!/usr/bin/env python3
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from utils.paths import DATA_ODDS, DATA_PRED, ensure_dirs

def main():
    ensure_dirs()
    # Placeholder odds (expect columns: player_id,line,over_odds,under_odds,season,week)
    # If you have a feed, write it to data/odds/receptions_weekX.csv in that schema.
    preds = pd.read_parquet(DATA_PRED / "wr_receptions_predictions.parquet")

    sample_odds = pd.DataFrame({
        "player_id": preds['player_id'].head(50),
        "line": 4.5,
        "over_odds": -110,
        "under_odds": -110,
        "season": preds['season'].head(50),
        "week": preds['week'].head(50),
    })
    sample_odds.to_csv(DATA_ODDS / "receptions_weekX.csv", index=False)
    print("✅ Wrote sample odds → data/odds/receptions_weekX.csv")

if __name__ == "__main__":
    main()
