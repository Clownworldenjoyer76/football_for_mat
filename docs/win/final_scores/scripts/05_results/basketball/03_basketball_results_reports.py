#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path

INPUT_DIR = Path("docs/win/final_scores/intermediate")

BASE_OUTPUT_DIR = Path("docs/win/final_scores/deeper_summaries")
NBA_OUTPUT_DIR   = BASE_OUTPUT_DIR / "nba"
NCAAB_OUTPUT_DIR = BASE_OUTPUT_DIR / "ncaab"
NBA_DETAIL_DIR   = NBA_OUTPUT_DIR   / "by_market"
NCAAB_DETAIL_DIR = NCAAB_OUTPUT_DIR / "by_market"

NBA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
NCAAB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
NBA_DETAIL_DIR.mkdir(parents=True, exist_ok=True)
NCAAB_DETAIL_DIR.mkdir(parents=True, exist_ok=True)


# ─── Bucket helpers ───────────────────────────────────────────────────────────

def _edge_bucket(x):
    try:
        x = float(x)
    except:
        return "UNBUCKETED"
    if x < 0.05:  return "0.00–0.05"
    if x < 0.075: return "0.05–0.075"
    if x < 0.10:  return "0.075–0.10"
    if x < 0.15:  return "0.10–0.15"
    if x < 0.20:  return "0.15–0.20"
    return "0.20+"


def _odds_bucket(value):
    try:
        value = float(value)
    except:
        return "UNBUCKETED"
    if value <= -200: return "≤-200"
    if value <= -150: return "-199 to -150"
    if value <= -110: return "-149 to -110"
    if value <= 100:  return "-109 to +100"
    if value <= 150:  return "+101 to +150"
    return "+150+"


def _kelly_bucket(k):
    try:
        k = float(k)
    except:
        return "UNBUCKETED"
    if k < 0.01: return "0-1%"
    if k < 0.02: return "1-2%"
    if k < 0.05: return "2-5%"
    return "5%+"


def _units_won(odds, result):
    if result == "Push": return 0.0
    if result == "Loss": return -1.0
    try:
        odds = float(odds)
    except:
        return None
    return odds / 100.0 if odds > 0 else 100.0 / abs(odds)


# ─── Enrichment ───────────────────────────────────────────────────────────────

def enrich_work(df):
    df = df.copy()

    def _take_odds(row):
        mt   = str(row.get("market_type", "")).lower()
        side = str(row.get("bet_side",    "")).lower()
        if mt == "moneyline":
            col = "home_dk_moneyline_american" if side == "home" else "away_dk_moneyline_american"
        elif mt == "spread":
            col = "home_dk_spread_american" if side == "home" else "away_dk_spread_american"
        elif mt == "total":
            col = "dk_total_over_american" if side == "over" else "dk_total_under_american"
        else:
            return None
        return row.get(col)

    def _kelly_value(row):
        mt   = str(row.get("market_type", "")).lower()
        side = str(row.get("bet_side",    "")).lower()
        if mt == "moneyline":
            col = "home_ml_kelly" if side == "home" else "away_ml_kelly"
        elif mt == "spread":
            col = "home_spread_kelly" if side == "home" else "away_spread_kelly"
        elif mt == "total":
            col = "over_kelly" if side == "over" else "under_kelly"
        else:
            return None
        return row.get(col)

    df["take_odds"]   = df.apply(_take_odds,   axis=1)
    df["kelly_value"] = df.apply(_kelly_value, axis=1)
    df["kelly_bucket"] = df["kelly_value"].apply(_kelly_bucket)
    df["odds_bucket"]  = df["take_odds"].apply(_odds_bucket)
    df["bet_units"]    = df.apply(lambda r: _units_won(r["take_odds"], r["bet_result"]), axis=1)
    df["edge_bucket"]  = df["selected_edge"].apply(_edge_bucket)

    return df


# ─── Aggregation ─────────────────────────────────────────────────────────────

def aggregate_full(df, group_cols):
    rows = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        wins   = (sub["bet_result"] == "Win").sum()
        losses = (sub["bet_result"] == "Loss").sum()
        bets   = len(sub)
        units  = sub["bet_units"].sum()

        row = {col: keys[i] for i, col in enumerate(group_cols)}
        row.update({
            "bets":             bets,
            "wins":             wins,
            "losses":           losses,
            "win_rate":         round(wins / (wins + losses), 4) if (wins + losses) else 0,
            "units":            round(units, 4),
            "roi":              round(units / bets, 4) if bets else 0,
            "avg_edge":         round(sub["selected_edge"].mean(), 4),
            "avg_odds":         round(sub["take_odds"].mean(), 1),
            "avg_units_per_bet":round(units / bets, 4) if bets else 0,
            "std_units":        round(sub["bet_units"].std(), 4) if sub["bet_units"].std() is not None else None,
        })
        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


# ─── Summary files (mirrors 03_nhl_results_reports.py) ───────────────────────

def build_summary_files(df, output_dir, detail_dir, league_name):
    """Write all summary CSVs the dashboard expects."""

    # Normalise
    result_clean = df["bet_result"].astype(str).str.strip().str.lower()
    df["is_win"]  = (result_clean == "win").astype(int)
    df["is_loss"] = (result_clean == "loss").astype(int)
    df["is_push"] = (result_clean == "push").astype(int)

    take_odds_num     = pd.to_numeric(df["take_odds"],     errors="coerce")
    selected_edge_num = pd.to_numeric(df["selected_edge"], errors="coerce")

    total_wins   = df["is_win"].sum()
    total_losses = df["is_loss"].sum()
    total_bets   = len(df)
    total_units  = df["bet_units"].sum()

    # ── Overall ──────────────────────────────────────────────────────────────
    overall = pd.DataFrame({
        "bets":        [total_bets],
        "wins":        [total_wins],
        "losses":      [total_losses],
        "pushes":      [df["is_push"].sum()],
        "win_rate":    [total_wins / (total_wins + total_losses) if (total_wins + total_losses) > 0 else np.nan],
        "units":       [total_units],
        "roi":         [total_units / total_bets if total_bets > 0 else np.nan],
        "avg_edge":    [selected_edge_num.mean()],
        "median_edge": [selected_edge_num.median()],
        "avg_odds":    [take_odds_num.mean()],
    })
    overall.to_csv(output_dir / f"{league_name}_summary_overall.csv", index=False)

    # ── By market ────────────────────────────────────────────────────────────
    aggregate_full(df, ["market_type"]).to_csv(
        output_dir / f"{league_name}_summary_by_market.csv", index=False
    )

    # ── By side group ────────────────────────────────────────────────────────
    aggregate_full(df, ["side_group"]).to_csv(
        output_dir / f"{league_name}_summary_by_side_group.csv", index=False
    )

    # ── By date ──────────────────────────────────────────────────────────────
    aggregate_full(df, ["game_date"]).to_csv(
        output_dir / f"{league_name}_summary_by_date.csv", index=False
    )

    # ── Bet log ──────────────────────────────────────────────────────────────
    log_cols = [
        "game_date", "market", "away_team", "home_team",
        "market_type", "side_group", "bet_side", "line",
        "take_odds", "selected_edge", "edge_bucket", "odds_bucket",
        "bet_result", "bet_units",
    ]
    existing_log_cols = [c for c in log_cols if c in df.columns]
    df[existing_log_cols].rename(columns={"bet_units": "units"}).to_csv(
        output_dir / f"{league_name}_bet_log.csv", index=False
    )

    # ── Market tally ─────────────────────────────────────────────────────────
    tally_rows = []
    for market in ["moneyline", "spread", "total"]:
        sub    = df[df["market_type"] == market]
        wins   = int(sub["is_win"].sum())
        losses = int(sub["is_loss"].sum())
        pushes = int(sub["is_push"].sum())
        total  = wins + losses + pushes
        win_pct = round(wins / (wins + losses), 4) if (wins + losses) > 0 else 0.0
        tally_rows.append({
            "market":      league_name.upper(),
            "market_type": market,
            "Win":         wins,
            "Loss":        losses,
            "Push":        pushes,
            "Total":       total,
            "Win_Pct":     win_pct,
        })
    tally_path = Path("docs/win/final_scores") / f"{league_name}_market_tally.csv"
    pd.DataFrame(tally_rows).to_csv(tally_path, index=False)

    # ── Per-market detail (edge / odds / kelly / side) ────────────────────────
    for market in ["moneyline", "spread", "total"]:
        mdf = df[df["market_type"] == market].copy()
        if mdf.empty:
            continue
        prefix = f"{league_name}_{market}"
        aggregate_full(mdf, ["edge_bucket"]).to_csv(detail_dir / f"{prefix}_by_edge.csv",  index=False)
        aggregate_full(mdf, ["odds_bucket"]).to_csv(detail_dir / f"{prefix}_by_odds.csv",  index=False)
        aggregate_full(mdf, ["kelly_bucket"]).to_csv(detail_dir / f"{prefix}_by_kelly.csv", index=False)
        aggregate_full(mdf, ["side_group"]).to_csv(detail_dir / f"{prefix}_by_side.csv",  index=False)


# ─── Detail reports (existing behaviour, unchanged) ──────────────────────────

def build_detail_reports(work, detail_dir, league):
    for market in ["moneyline", "spread", "total"]:
        df = work[work["market_type"] == market]
        if df.empty:
            continue

        prefix = f"{league}_{market}"

        aggregate_full(df, ["edge_bucket"]).to_csv(
            detail_dir / f"{prefix}_edge.csv", index=False)

        aggregate_full(df, ["market_type", "edge_bucket"]).to_csv(
            detail_dir / f"{prefix}_market_edge.csv", index=False)

        aggregate_full(df, ["market_type", "side_group", "edge_bucket"]).to_csv(
            detail_dir / f"{prefix}_market_side_edge.csv", index=False)

        aggregate_full(df, ["edge_bucket", "odds_bucket"]).to_csv(
            detail_dir / f"{prefix}_edge_odds.csv", index=False)

        aggregate_full(df, ["edge_bucket", "kelly_bucket"]).to_csv(
            detail_dir / f"{prefix}_edge_kelly.csv", index=False)


# ─── Main ─────────────────────────────────────────────────────────────────────

def run():
    nba   = enrich_work(pd.read_csv(INPUT_DIR / "work_nba.csv"))
    ncaab = enrich_work(pd.read_csv(INPUT_DIR / "work_ncaab.csv"))

    # Summary files (new)
    build_summary_files(nba,   NBA_OUTPUT_DIR,   NBA_DETAIL_DIR,   "nba")
    build_summary_files(ncaab, NCAAB_OUTPUT_DIR, NCAAB_DETAIL_DIR, "ncaab")

    # Detail edge reports (existing)
    build_detail_reports(nba,   NBA_DETAIL_DIR,   "nba")
    build_detail_reports(ncaab, NCAAB_DETAIL_DIR, "ncaab")

    print("Basketball reports complete.")


if __name__ == "__main__":
    run()
