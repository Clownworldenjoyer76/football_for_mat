
#!/usr/bin/env python3
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from utils.paths import OUTPUT_DIR

EDGE_THRESHOLD = 0.05

def main():
    df = pd.read_csv(OUTPUT_DIR / 'receptions_edges.csv')
    best = df[df['edge_over'] >= EDGE_THRESHOLD].copy()
    best = best.sort_values('edge_over', ascending=False)
    best.to_csv(OUTPUT_DIR / 'best_bets_receptions.csv', index=False)
    print(f'{len(best)} bets → output/best_bets_receptions.csv')

if __name__ == '__main__':
    main()
