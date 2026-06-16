#!/usr/bin/env python3

import pandas as pd
from pathlib import Path

NBA_INPUT = Path("docs/win/final_scores/results/nba/graded/NBA_final.csv")
NCAAB_INPUT = Path("docs/win/final_scores/results/ncaab/graded/NCAAB_final.csv")

OUTPUT_DIR = Path("docs/win/final_scores/intermediate")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Only wipe basketball intermediate outputs before regenerating
for stale_name in ["work_nba.csv", "work_ncaab.csv"]:
    (OUTPUT_DIR / stale_name).unlink(missing_ok=True)


def get_col(row, *names):
    for name in names:
        if name in row.index:
            return row[name]
    return pd.NA


def to_float(value):
    try:
        return float(value)
    except Exception:
        return pd.NA


def build_side_group(row):
    market_type = str(row.get("market_type", "")).strip().lower()
    bet_side = str(row.get("bet_side", "")).strip().lower()

    if market_type in {"moneyline", "spread"}:
        if bet_side == "home":
            return "HOME"
        if bet_side == "away":
            return "AWAY"

    if market_type == "total":
        if bet_side == "over":
            return "OVER"
        if bet_side == "under":
            return "UNDER"

    return ""


def build_selected_edge(row):
    market_type = str(row.get("market_type", "")).strip().lower()
    side_group = str(row.get("side_group", "")).strip().upper()

    if market_type == "moneyline":
        if side_group == "HOME":
            return to_float(get_col(row, "home_ml_edge", "home_ml_edge_decimal"))
        if side_group == "AWAY":
            return to_float(get_col(row, "away_ml_edge", "away_ml_edge_decimal"))

    if market_type == "spread":
        if side_group == "HOME":
            return to_float(get_col(row, "home_spread_edge", "home_spread_edge_decimal"))
        if side_group == "AWAY":
            return to_float(get_col(row, "away_spread_edge", "away_spread_edge_decimal"))

    if market_type == "total":
        if side_group == "OVER":
            return to_float(get_col(row, "over_edge", "over_edge_decimal"))
        if side_group == "UNDER":
            return to_float(get_col(row, "under_edge", "under_edge_decimal"))

    return pd.NA


def build_moneyline_odds_value(row):
    side_group = str(row.get("side_group", "")).strip().upper()

    if side_group == "HOME":
        return to_float(get_col(row, "home_dk_moneyline_american"))
    if side_group == "AWAY":
        return to_float(get_col(row, "away_dk_moneyline_american"))

    return pd.NA


def build_spread_value(row):
    side_group = str(row.get("side_group", "")).strip().upper()

    if side_group == "HOME":
        return to_float(get_col(row, "home_spread"))
    if side_group == "AWAY":
        return to_float(get_col(row, "away_spread"))

    return pd.NA


def build_total_value(row):
    return to_float(get_col(row, "total"))


def edge_bucket(value):
    value = to_float(value)

    if pd.isna(value):
        return "UNBUCKETED"
    if value < 0:
        return "<0"
    if value < 0.01:
        return "0.00_to_0.0099"
    if value < 0.02:
        return "0.01_to_0.0199"
    if value < 0.03:
        return "0.02_to_0.0299"
    if value < 0.04:
        return "0.03_to_0.0399"
    if value < 0.05:
        return "0.04_to_0.0499"
    if value < 0.075:
        return "0.05_to_0.0749"
    if value < 0.10:
        return "0.075_to_0.0999"
    return "0.10_plus"


def odds_bucket(value):
    value = to_float(value)

    if pd.isna(value):
        return "UNBUCKETED"
    if value <= -200:
        return "minus_200_or_lower"
    if value <= -150:
        return "minus_199_to_minus_150"
    if value <= -125:
        return "minus_149_to_minus_125"
    if value <= -110:
        return "minus_124_to_minus_110"
    if value <= -101:
        return "minus_109_to_minus_101"
    if value <= 100:
        return "minus_100_to_plus_100"
    if value <= 125:
        return "plus_101_to_plus_125"
    if value <= 150:
        return "plus_126_to_plus_150"
    return "plus_201_or_higher"


def spread_bucket(value):
    value = to_float(value)

    if pd.isna(value):
        return "UNBUCKETED"

    value = float(value)
    abs_value = abs(value)

    if abs_value < 1:
        return "0.0_to_0.9"
    if abs_value < 2:
        return "1.0_to_1.9"
    if abs_value < 3:
        return "2.0_to_2.9"
    if abs_value < 4:
        return "3.0_to_3.9"
    if abs_value < 5:
        return "4.0_to_4.9"
    if abs_value < 6:
        return "5.0_to_5.9"
    if abs_value < 7:
        return "6.0_to_6.9"
    if abs_value < 8:
        return "7.0_to_7.9"
    if abs_value < 9:
        return "8.0_to_8.9"
    if abs_value < 10:
        return "9.0_to_9.9"
    if abs_value < 12:
        return "10.0_to_11.9"
    if abs_value < 15:
        return "12.0_to_14.9"
    return "15.0_plus"


def total_bucket(value):
    value = to_float(value)

    if pd.isna(value):
        return "UNBUCKETED"

    start = int(float(value) // 5) * 5
    end = start + 4.9
    return f"{start}_to_{end:.1f}"


def prepare(df, league):
    work = df.copy()

    work["market"] = league
    work["market_type"] = work["market_type"].astype(str).str.strip().str.lower()
    work["bet_side"] = work["bet_side"].astype(str).str.strip().str.lower()
    work["bet_result"] = work["bet_result"].astype(str).str.strip().str.title()

    work["side_group"] = work.apply(build_side_group, axis=1)
    work["selected_edge"] = work.apply(build_selected_edge, axis=1)
    work["moneyline_odds_value"] = work.apply(build_moneyline_odds_value, axis=1)
    work["spread_value"] = work.apply(build_spread_value, axis=1)
    work["total_value"] = work.apply(build_total_value, axis=1)

    work["edge_bucket"] = work["selected_edge"].apply(edge_bucket)
    work["odds_bucket"] = work["moneyline_odds_value"].apply(odds_bucket)
    work["spread_bucket"] = work["spread_value"].apply(spread_bucket)
    work["total_bucket"] = work["total_value"].apply(total_bucket)

    return work


def run():
    nba = pd.read_csv(NBA_INPUT)
    ncaab = pd.read_csv(NCAAB_INPUT)

    nba = prepare(nba, "NBA")
    ncaab = prepare(ncaab, "NCAAB")

    nba.to_csv(OUTPUT_DIR / "work_nba.csv", index=False)
    ncaab.to_csv(OUTPUT_DIR / "work_ncaab.csv", index=False)

    print("Analyze complete.")


if __name__ == "__main__":
    run()
