#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np

INPUT_FILE = Path("docs/win/final_scores/intermediate/work_nhl.csv")
OUTPUT_DIR = Path("docs/win/final_scores/deeper_summaries/nhl")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = OUTPUT_DIR / "nhl_summary_overall.csv"
BY_MARKET_FILE = OUTPUT_DIR / "nhl_summary_by_market.csv"
BY_SIDE_GROUP_FILE = OUTPUT_DIR / "nhl_summary_by_side_group.csv"
BY_EDGE_BUCKET_FILE = OUTPUT_DIR / "nhl_summary_by_edge_bucket.csv"
BY_ODDS_BUCKET_FILE = OUTPUT_DIR / "nhl_summary_by_odds_bucket.csv"
BY_TOTAL_BUCKET_FILE = OUTPUT_DIR / "nhl_summary_by_total_bucket.csv"
BY_DATE_FILE = OUTPUT_DIR / "nhl_summary_by_date.csv"
BY_MARKET_AND_SIDE_FILE = OUTPUT_DIR / "nhl_summary_by_market_and_side.csv"
BET_LOG_FILE = OUTPUT_DIR / "nhl_bet_log.csv"
MARKET_DETAIL_DIR = OUTPUT_DIR / "by_market"
MARKET_DETAIL_DIR.mkdir(parents=True, exist_ok=True)


def american_to_profit_per_unit(odds):
    if pd.isna(odds):
        return np.nan
    odds = float(odds)
    if odds > 0:
        return odds / 100.0
    return 100.0 / abs(odds)


def grade_to_units(row):
    result = str(row.get("bet_result", "")).strip().lower()
    odds = row.get("take_odds", np.nan)
    profit_per_unit = american_to_profit_per_unit(odds)
    if result == "win":
        return profit_per_unit if not pd.isna(profit_per_unit) else np.nan
    if result == "loss":
        return -1.0
    if result == "push":
        return 0.0
    return np.nan


def normalize_market_type(value):
    value = str(value).strip().lower()
    if value in {"moneyline", "ml"}:
        return "moneyline"
    if value in {"puck_line", "spread", "puckline"}:
        return "puck_line"
    if value in {"total", "totals"}:
        return "total"
    return value


def normalize_side_group(row):
    market = row["market_type_norm"]
    side = str(row.get("bet_side", "")).strip().lower()
    if market in {"moneyline", "puck_line"}:
        if side == "home":
            return "HOME"
        if side == "away":
            return "AWAY"
    if market == "total":
        if side == "over":
            return "OVER"
        if side == "under":
            return "UNDER"
    return side.upper()


def edge_bucket(value):
    if pd.isna(value):
        return "missing"
    v = float(value)
    if v < 0:
        return "<0"
    if v < 0.01:
        return "0.00_to_0.0099"
    if v < 0.02:
        return "0.01_to_0.0199"
    if v < 0.03:
        return "0.02_to_0.0299"
    if v < 0.04:
        return "0.03_to_0.0399"
    if v < 0.05:
        return "0.04_to_0.0499"
    if v < 0.075:
        return "0.05_to_0.0749"
    if v < 0.10:
        return "0.075_to_0.0999"
    return "0.10_plus"


def odds_bucket(value):
    if pd.isna(value):
        return "missing"
    v = int(float(value))
    if v <= -200:
        return "minus_200_or_lower"
    if -199 <= v <= -150:
        return "minus_199_to_minus_150"
    if -149 <= v <= -125:
        return "minus_149_to_minus_125"
    if -124 <= v <= -110:
        return "minus_124_to_minus_110"
    if -109 <= v <= 100:
        return "minus_109_to_plus_100"
    if 101 <= v <= 125:
        return "plus_101_to_plus_125"
    if 126 <= v <= 150:
        return "plus_126_to_plus_150"
    if 151 <= v <= 200:
        return "plus_151_to_plus_200"
    return "plus_201_or_higher"


def total_bucket(value):
    if pd.isna(value):
        return "missing"
    v = float(value)
    if v < 5.5:
        return "under_5.5"
    if v < 6.5:
        return "5.5"
    if v < 7.5:
        return "6.5"
    if v < 8.5:
        return "7.5"
    return "8.5_plus"


def kelly_bucket(value):
    if pd.isna(value):
        return "missing"
    v = float(value)
    if v <= 0:
        return "0_or_less"
    if v < 0.01:
        return "0.00_to_0.0099"
    if v < 0.02:
        return "0.01_to_0.0199"
    if v < 0.05:
        return "0.02_to_0.0499"
    if v < 0.10:
        return "0.05_to_0.0999"
    if v < 0.20:
        return "0.10_to_0.1999"
    if v < 0.50:
        return "0.20_to_0.4999"
    return "0.50_plus"


def puck_line_bucket(row):
    if row["market_type_norm"] != "puck_line":
        return ""
    line = row.get("line", np.nan)
    if pd.isna(line):
        return "missing"
    try:
        v = float(line)
    except Exception:
        return "missing"
    if v <= -2.5:
        return "minus_2.5_or_lower"
    if v == -1.5:
        return "minus_1.5"
    if v == 1.5:
        return "plus_1.5"
    return "other"


def summarize(df, group_cols):
    grouped = (
        df.groupby(group_cols, dropna=False)
        .agg(
            bets=("bet_result", "size"),
            wins=("is_win", "sum"),
            losses=("is_loss", "sum"),
            pushes=("is_push", "sum"),
            units=("units", "sum"),
            avg_edge=("selected_edge_num", "mean"),
            median_edge=("selected_edge_num", "median"),
            avg_odds=("take_odds_num", "mean"),
        )
        .reset_index()
    )
    grouped["win_rate"] = np.where(
        (grouped["wins"] + grouped["losses"]) > 0,
        grouped["wins"] / (grouped["wins"] + grouped["losses"]),
        np.nan,
    )
    grouped["roi"] = np.where(
        grouped["bets"] > 0,
        grouped["units"] / grouped["bets"],
        np.nan,
    )
    grouped = grouped.sort_values(group_cols).reset_index(drop=True)
    return grouped


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    if df.empty:
        raise ValueError(f"Input file is empty: {INPUT_FILE}")

    if "selected_edge" not in df.columns:
        df["selected_edge"] = df["ev"]
    if "take_odds" not in df.columns:
        df["take_odds"] = df["dk_odds_american"]

    df["game_date"] = df["game_date"].astype(str)
    df["market_type_norm"] = df["market_type"].apply(normalize_market_type)
    df["selected_edge_num"] = pd.to_numeric(df["selected_edge"], errors="coerce")
    df["take_odds_num"] = pd.to_numeric(df["take_odds"], errors="coerce")
    df["line_num"] = pd.to_numeric(df["line"], errors="coerce")
    df["side_group"] = df.apply(normalize_side_group, axis=1)
    df["edge_bucket"] = df["selected_edge_num"].apply(edge_bucket)
    df["odds_bucket"] = df["take_odds_num"].apply(odds_bucket)
    df["total_bucket"] = df.apply(
        lambda r: total_bucket(r["line"]) if r["market_type_norm"] == "total" else "missing",
        axis=1,
    )
    if "kelly" in df.columns:
        df["kelly_bucket"] = df["kelly"].apply(kelly_bucket)
    else:
        df["kelly_bucket"] = "missing"
    df["puck_line_bucket"] = df.apply(puck_line_bucket, axis=1)

    result_clean = df["bet_result"].astype(str).str.strip().str.lower()
    df["is_win"] = (result_clean == "win").astype(int)
    df["is_loss"] = (result_clean == "loss").astype(int)
    df["is_push"] = (result_clean == "push").astype(int)
    df["units"] = df.apply(grade_to_units, axis=1)

    bet_log_cols = [
        "game_date", "league", "away_team", "home_team",
        "market_type_norm", "side_group", "bet_side", "line",
        "take_odds", "selected_edge", "edge_bucket", "odds_bucket",
        "total_bucket", "puck_line_bucket", "bet_result", "units", "game_id",
    ]
    df[bet_log_cols].rename(columns={"market_type_norm": "market_type"}).to_csv(BET_LOG_FILE, index=False)

    overall = pd.DataFrame({
        "bets": [len(df)],
        "wins": [df["is_win"].sum()],
        "losses": [df["is_loss"].sum()],
        "pushes": [df["is_push"].sum()],
        "win_rate": [
            df["is_win"].sum() / (df["is_win"].sum() + df["is_loss"].sum())
            if (df["is_win"].sum() + df["is_loss"].sum()) > 0 else np.nan
        ],
        "units": [df["units"].sum()],
        "roi": [df["units"].sum() / len(df) if len(df) > 0 else np.nan],
        "avg_edge": [df["selected_edge_num"].mean()],
        "median_edge": [df["selected_edge_num"].median()],
        "avg_odds": [df["take_odds_num"].mean()],
    })
    overall.to_csv(SUMMARY_FILE, index=False)

    summarize(df, ["market_type_norm"]).rename(columns={"market_type_norm": "market_type"}).to_csv(BY_MARKET_FILE, index=False)
    summarize(df, ["side_group"]).to_csv(BY_SIDE_GROUP_FILE, index=False)
    summarize(df, ["edge_bucket"]).to_csv(BY_EDGE_BUCKET_FILE, index=False)
    summarize(df, ["odds_bucket"]).to_csv(BY_ODDS_BUCKET_FILE, index=False)

    total_df = df[df["market_type_norm"] == "total"].copy()
    if total_df.empty:
        pd.DataFrame(columns=[
            "total_bucket", "bets", "wins", "losses", "pushes",
            "units", "avg_edge", "median_edge", "avg_odds", "win_rate", "roi"
        ]).to_csv(BY_TOTAL_BUCKET_FILE, index=False)
    else:
        summarize(total_df, ["total_bucket"]).to_csv(BY_TOTAL_BUCKET_FILE, index=False)

    summarize(df, ["game_date"]).to_csv(BY_DATE_FILE, index=False)
    summarize(df, ["market_type_norm", "side_group"]).rename(
        columns={"market_type_norm": "market_type"}
    ).to_csv(BY_MARKET_AND_SIDE_FILE, index=False)

    for market_name in ["moneyline", "puck_line", "total"]:
        mdf = df[df["market_type_norm"] == market_name].copy()
        if mdf.empty:
            continue
        summarize(mdf, ["edge_bucket"]).to_csv(MARKET_DETAIL_DIR / f"nhl_{market_name}_by_edge.csv", index=False)
        summarize(mdf, ["odds_bucket"]).to_csv(MARKET_DETAIL_DIR / f"nhl_{market_name}_by_odds.csv", index=False)
        summarize(mdf, ["kelly_bucket"]).to_csv(MARKET_DETAIL_DIR / f"nhl_{market_name}_by_kelly.csv", index=False)
        summarize(mdf, ["side_group"]).to_csv(MARKET_DETAIL_DIR / f"nhl_{market_name}_by_side.csv", index=False)

    tally_rows = []
    for market in ["moneyline", "puck_line", "total"]:
        sub = df[df["market_type_norm"] == market]
        wins = int(sub["is_win"].sum())
        losses = int(sub["is_loss"].sum())
        pushes = int(sub["is_push"].sum())
        total = wins + losses + pushes
        win_pct = round(wins / (wins + losses), 4) if (wins + losses) > 0 else 0.0
        tally_rows.append({
            "market": "NHL", "market_type": market,
            "Win": wins, "Loss": losses, "Push": pushes,
            "Total": total, "Win_Pct": win_pct,
        })
    pd.DataFrame(tally_rows).to_csv(
        Path("docs/win/final_scores/nhl_market_tally.csv"), index=False
    )


if __name__ == "__main__":
    main()
