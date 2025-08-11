#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy.stats import norm
from utils.paths import DATA_ODDS, DATA_PRED, OUTPUT_DIR, ensure_dirs
from utils.odds import implied_prob, remove_vig_2way

def p_over(line: float, mu: float, sigma: float) -> float:
    return 1 - norm.cdf(line + 1e-9, loc=mu, scale=sigma)

def main():
    ensure_dirs()
    preds = pd.read_parquet(DATA_PRED / "wr_receptions_predictions.parquet")
    odds  = pd.read_csv(DATA_ODDS / "receptions_weekX.csv")
    df = preds.merge(odds, on=['player_id','season','week'], how='inner')

    df['p_over_model'] = df.apply(lambda r: p_over(r['line'], r['mu'], r['sigma']), axis=1)

    df['p_over_mkt_raw']  = df['over_odds'].apply(implied_prob)
    df['p_under_mkt_raw'] = df['under_odds'].apply(implied_prob)
    df[['p_over_mkt','p_under_mkt']] = df.apply(
        lambda r: pd.Series(remove_vig_2way(r['p_over_mkt_raw'], r['p_under_mkt_raw'])),
        axis=1
    )

    df['edge_over'] = df['p_over_model'] - df['p_over_mkt']
    df.sort_values('edge_over', ascending=False, inplace=True)
    df.to_csv(OUTPUT_DIR / "receptions_edges.csv", index=False)
    print("✅ Edges written → output/receptions_edges.csv")

if __name__ == "__main__":
    main()
