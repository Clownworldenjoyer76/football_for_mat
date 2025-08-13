#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

PROPS_DIR = Path("data/props")
EDGES_DIR = Path("data/edges")
OUT_DIR   = Path("output")
EDGES_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_EDGE = 0.02     # edge threshold (2%)
MIN_ROWS = 25       # fallback: ensure at least this many picks per market (if available)
TOP_N    = 200      # cap per-market best list

def process_market(fp: Path):
    market = fp.stem.replace("props_", "")
    df = pd.read_csv(fp)
    if "prob_over" not in df.columns or "prob_under" not in df.columns:
        return None

    df["prob_over"]  = df["prob_over"].clip(0, 1)
    df["prob_under"] = df["prob_under"].clip(0, 1)
    df["edge_over"]  = df["prob_over"]  - 0.5
    df["edge_under"] = df["prob_under"] - 0.5

    pick_over = df["edge_over"] >= df["edge_under"]
    df["pick"] = pick_over.map({True: "over", False: "under"})
    df["edge"] = df[["edge_over","edge_under"]].max(axis=1)

    # write full edges
    edges_path = EDGES_DIR / f"edges_{market}.csv"
    df.to_csv(edges_path, index=False)

    # best bets: threshold or fallback top K
    best = df[df["edge"] >= MIN_EDGE].sort_values("edge", ascending=False)
    if len(best) < MIN_ROWS:
        best = df.sort_values("edge", ascending=False).head(MIN_ROWS)
    best = best.head(TOP_N)

    cols = [c for c in [
        "player_name","team","recent_team","opponent","opponent_team",
        "season","week","prop","line","y_pred","pick","edge","prob_over","prob_under"
    ] if c in df.columns]
    best = best[cols]
    best["market"] = market

    best_path = OUT_DIR / f"best_bets_{market}.csv"
    best.to_csv(best_path, index=False)
    return best

def main():
    files = sorted(PROPS_DIR.glob("props_*.csv"))
    all_best = []
    for fp in files:
        res = process_market(fp)
        if res is not None and not res.empty:
            all_best.append(res)

    # summary across markets
    if all_best:
        summary = pd.concat(all_best, ignore_index=True).sort_values("edge", ascending=False)
        summary.to_csv(OUT_DIR / "best_bets_summary.csv", index=False)

if __name__ == "__main__":
    main()
