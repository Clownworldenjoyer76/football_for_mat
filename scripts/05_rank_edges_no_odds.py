#!/usr/bin/env python3
"""
Reads data/props/props_*.csv (from step 04).
Computes edge vs 50% baseline:
  edge_over  = prob_over  - 0.5
  edge_under = prob_under - 0.5
Picks the better side per row and writes:
  data/edges/edges_<market>.csv
  output/best_bets_<market>.csv  (edge >= MIN_EDGE only)
"""
from pathlib import Path
import pandas as pd

PROPS_DIR = Path("data/props")
EDGES_DIR = Path("data/edges")
OUT_DIR   = Path("output")
EDGES_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_EDGE = 0.02   # 2% edge threshold
TOP_N    = 200    # cap best_bets size

def process_market(fp: Path):
    market = fp.stem.replace("props_", "")
    df = pd.read_csv(fp)

    if "prob_over" not in df.columns or "prob_under" not in df.columns:
        print(f"skip {market}: missing prob columns")
        return

    # guard rails
    df["prob_over"]  = df["prob_over"].clip(0.0, 1.0)
    df["prob_under"] = df["prob_under"].clip(0.0, 1.0)

    df["edge_over"]  = df["prob_over"]  - 0.5
    df["edge_under"] = df["prob_under"] - 0.5

    # choose side by larger edge
    pick_over = df["edge_over"] >= df["edge_under"]
    df["pick"] = pick_over.map({True: "over", False: "under"})
    df["edge"] = df[["edge_over","edge_under"]].max(axis=1)

    # save full edges
    edges_path = EDGES_DIR / f"edges_{market}.csv"
    df.to_csv(edges_path, index=False)

    # best bets: threshold + top N by edge
    best = df[df["edge"] >= MIN_EDGE].sort_values("edge", ascending=False).head(TOP_N)
    cols = [c for c in [
        "player_name","team","recent_team","opponent","opponent_team",
        "season","week","prop","line","y_pred","pick","edge","prob_over","prob_under"
    ] if c in df.columns]
    best = best[cols]
    best_path = OUT_DIR / f"best_bets_{market}.csv"
    best.to_csv(best_path, index=False)

    print(f"✓ {market}: {len(df)} rows | best_bets={len(best)} -> {best_path.name}")

def main():
    files = sorted(PROPS_DIR.glob("props_*.csv"))
    if not files:
        print("No props files found in data/props")
        return
    for fp in files:
        process_market(fp)

if __name__ == "__main__":
    main()
