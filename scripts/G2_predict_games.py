#!/usr/bin/env python3
# Convert team yards -> predicted points, totals, spreads
from pathlib import Path
import pandas as pd

INP = Path("data/games/game_features.csv")
OUT_DIR = Path("data/games"); OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_SUM = Path("output"); OUT_SUM.mkdir(parents=True, exist_ok=True)

# Simple yards->points factor (~15 yards per point)
YPP = 1.0 / 15.0  # ≈0.0667

def clamp(x, lo, hi): return max(lo, min(hi, x))

def main():
    if not INP.exists():
        raise SystemExit(f"missing {INP}")
    df = pd.read_csv(INP)

    # points = YPP * (pass + rush)
    df["home_pts"] = (df.get("home_total_yards",0) * YPP).clip(lower=0)
    df["away_pts"] = (df.get("away_total_yards",0) * YPP).clip(lower=0)

    # Optional light smoothing
    df["home_pts"] = df["home_pts"].apply(lambda x: clamp(x, 7, 45))
    df["away_pts"] = df["away_pts"].apply(lambda x: clamp(x, 7, 45))

    df["total"] = df["home_pts"] + df["away_pts"]
    df["spread_home"] = df["home_pts"] - df["away_pts"]

    keep = [
        "season","week","home_team","away_team",
        "home_pass_yds","home_rush_yds","home_total_yards",
        "away_pass_yds","away_rush_yds","away_total_yards",
        "home_pts","away_pts","total","spread_home"
    ]
    df_out = df[[c for c in keep if c in df.columns]].copy()
    df_out.to_csv(OUT_DIR / "game_projections.csv", index=False)

    # quick summary for mobile
    summary = df_out[["season","week","home_team","away_team","home_pts","away_pts","total","spread_home"]] \
              .sort_values(["season","week","total"], ascending=[False,False,False])
    summary.to_csv(OUT_SUM / "game_projections_summary.csv", index=False)

    print(f"✓ wrote {OUT_DIR/'game_projections.csv'} ({len(df_out)} rows)")
    print(f"✓ wrote {OUT_SUM/'game_projections_summary.csv'}")

if __name__ == "__main__":
    main()
