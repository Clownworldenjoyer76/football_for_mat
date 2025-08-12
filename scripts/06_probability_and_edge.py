
#!/usr/bin/env python3
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import norm
from utils.paths import DATA_PRED, OUTPUT_DIR
from utils.odds import implied_prob, remove_vig_2way

def p_over(line, mu, sigma):
    return 1 - norm.cdf(line, loc=mu, scale=sigma)

def main():
    df = pd.read_csv(DATA_PRED / 'wr_receptions_with_odds.csv')
    df['p_over_model'] = p_over(df['line'], df['mu'], df['sigma'])

    df['p_over_mkt_raw'] = df['over_odds'].apply(implied_prob)
    df['p_under_mkt_raw'] = df['under_odds'].apply(implied_prob)
    pv = df.apply(lambda r: remove_vig_2way(r['p_over_mkt_raw'], r['p_under_mkt_raw']), axis=1, result_type='expand')
    df['p_over_mkt'], df['p_under_mkt'] = pv[0], pv[1]

    df['edge_over'] = df['p_over_model'] - df['p_over_mkt']
    out = df.sort_values('edge_over', ascending=False)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_DIR / 'receptions_edges.csv', index=False)
    print('edges → output/receptions_edges.csv')

if __name__ == '__main__':
    main()
