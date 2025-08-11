#!/usr/bin/env python3
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from utils.paths import OUTPUT_DIR, ensure_dirs

EDGE_THRESHOLD = 0.05  # 5%

def main():
    ensure_dirs()
    df = pd.read_csv(OUTPUT_DIR / "receptions_edges.csv")
    best = df[df['edge_over'] >= EDGE_THRESHOLD].copy()
    best.sort_values('edge_over', ascending=False, inplace=True)
    best.to_csv(OUTPUT_DIR / "best_bets_receptions_weekX.csv", index=False)
    print(f"✅ Selected {len(best)} bets with edge ≥ {EDGE_THRESHOLD:.0%} → output/best_bets_receptions_weekX.csv")

if __name__ == "__main__":
    main()
