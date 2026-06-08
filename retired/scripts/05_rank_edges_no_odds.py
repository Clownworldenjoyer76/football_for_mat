#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

PROPS_DIR = Path("data/props")
EDGES_DIR = Path("data/edges")
OUT_DIR   = Path("output")
EDGES_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_EDGE = 0.02   # 2%
MIN_ROWS = 25     # fallback min rows per market
TOP_N    = 200    # cap per-market best list
PREVIEW_N = 200   # preview rows for large edges

def process_market(fp: Path):
    market = fp.stem.replace("props_", "")
    df = pd.read_csv(fp)
    if {"prob_over","prob_under"} - set(df.columns):
        return None

    df["prob_over"]  = df["prob_over"].clip(0,1)
    df["prob_under"] = df["prob_under"].clip(0,1)
    df["edge_over"]  = df["prob_over"]  - 0.5
    df["edge_under"] = df["prob_under"] - 0.5
    df["pick"] = (df["edge_over"] >= df["edge_under"]).map({True:"over", False:"under"})
    df["edge"] = df[["edge_over","edge_under"]].max(axis=1)

    # FULL edges (compressed) + small preview CSV
    full_gz = EDGES_DIR / f"edges_{market}.csv.gz"
    df.to_csv(full_gz, index=False, compression="gzip")
    preview_csv = EDGES_DIR / f"edges_{market}.preview.csv"
    df.sort_values("edge", ascending=False).head(PREVIEW_N).to_csv(preview_csv, index=False)

    # best bets (threshold with fallback)
    best = df[df["edge"] >= MIN_EDGE].sort_values("edge", ascending=False)
    if len(best) < MIN_ROWS:
        best = df.sort_values("edge", ascending=False).head(MIN_ROWS)
    best = best.head(TOP_N)

    cols = [c for c in [
        "player_name","team","recent_team","opponent","opponent_team",
        "season","week","prop","line","y_pred","pick","edge","prob_over","prob_under"
    ] if c in df.columns]
    best = best[cols]
    (OUT_DIR / f"best_bets_{market}.csv").write_text(best.to_csv(index=False))

    # return for summary
    best["market"] = market
    return best

def main():
    files = sorted(PROPS_DIR.glob("props_*.csv"))
    all_best = []
    for fp in files:
        res = process_market(fp)
        if res is not None and not res.empty:
            all_best.append(res)
    if all_best:
        pd.concat(all_best, ignore_index=True).sort_values("edge", ascending=False)\
          .to_csv(OUT_DIR / "best_bets_summary.csv", index=False)

if __name__ == "__main__":
    main()
