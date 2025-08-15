#!/usr/bin/env python3
# Convert team yards -> predicted points, totals, spreads
from pathlib import Path
import pandas as pd

INP = Path("data/games/game_features.csv")
OUT_DIR = Path("data/games"); OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_SUM = Path("output"); OUT_SUM.mkdir(parents=True, exist_ok=True)

YPP = 1.0 / 15.0  # ~15 yards per point

def clamp(x, lo, hi): 
    return max(lo, min(hi, x))

def main():
    if not INP.exists():
        raise SystemExit(f"missing {INP}")
    df = pd.read_csv(INP)

    # points from total yards
    df["home_pts"] = (df.get("home_total_yards",0) * YPP).clip(lower=0)
    df["away_pts"] = (df.get("away_total_yards",0) * YPP).clip(lower=0)

    # light bounds
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
    cols = [c for c in keep if c in df.columns]
    out = df[cols].copy()

    proj_fp = OUT_DIR / "game_projections.csv"
    out.to_csv(proj_fp, index=False)

    summary = out[["season","week","home_team","away_team","home_pts","away_pts","total","spread_home"]] \
              .sort_values(["season","week","total"], ascending=[False,False,False])
    sum_fp = OUT_SUM / "game_projections_summary.csv"
    summary.to_csv(sum_fp, index=False)

    print(f"wrote {proj_fp} ({len(out)} rows)")
    print(f"wrote {sum_fp}")

if __name__ == "__main__":
    main()
