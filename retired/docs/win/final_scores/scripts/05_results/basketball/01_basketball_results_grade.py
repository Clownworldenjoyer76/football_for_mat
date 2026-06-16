#!/usr/bin/env python3
# docs/win/final_scores/scripts/05_results/basketball/01_basketball_results_grade.py

import pandas as pd
from pathlib import Path

BASE = Path("docs/win/basketball")
SELECT_DIR = BASE / "04_select/daily_slate"

NBA_SCORE_DIR   = Path("docs/win/final_scores/results/nba/final_scores")
NCAAB_SCORE_DIR = Path("docs/win/final_scores/results/ncaab/final_scores")

NBA_OUTPUT   = Path("docs/win/final_scores/results/nba/graded")
NCAAB_OUTPUT = Path("docs/win/final_scores/results/ncaab/graded")

ERROR_DIR = Path("docs/win/final_scores/errors")
ERROR_DIR.mkdir(parents=True, exist_ok=True)


def determine_outcome(row):
    market = str(row["market_type"]).lower()
    side   = str(row["bet_side"]).lower()

    away = float(row["away_score"])
    home = float(row["home_score"])

    if market == "moneyline":
        if away == home:
            return "Push"
        return "Win" if (home > away and side == "home") or (away > home and side == "away") else "Loss"

    if market == "spread":
        line = float(row["line"])
        diff = (home + line - away) if side == "home" else (away + line - home)
        if abs(diff) < 1e-9:
            return "Push"
        return "Win" if diff > 0 else "Loss"

    if market == "total":
        total = away + home
        line  = float(row["line"])
        if abs(total - line) < 1e-9:
            return "Push"
        return "Win" if (total > line and side == "over") or (total < line and side == "under") else "Loss"

    return "Unknown"


def clean_merge_columns(df):
    for col in list(df.columns):
        if col.endswith("_x"):
            df[col[:-2]] = df[col]
        elif col.endswith("_y"):
            base = col[:-2]
            if base not in df.columns:
                df[base] = df[col]
    df = df.drop(columns=[c for c in df.columns if c.endswith("_x") or c.endswith("_y")], errors="ignore")
    return df


def rebuild_selected_edge(df):
    has_decimal  = "home_ml_edge_decimal" in df.columns
    has_standard = "home_ml_edge" in df.columns

    def pick_edge(row):
        market = row["market_type"]
        side   = row["bet_side"]

        if market == "moneyline":
            if has_decimal:
                return row.get("home_ml_edge_decimal") if side == "home" else row.get("away_ml_edge_decimal")
            if has_standard:
                return row.get("home_ml_edge") if side == "home" else row.get("away_ml_edge")

        if market == "spread":
            if has_decimal:
                return row.get("home_spread_edge_decimal") if side == "home" else row.get("away_spread_edge_decimal")
            if has_standard:
                return row.get("home_spread_edge") if side == "home" else row.get("away_spread_edge")

        if market == "total":
            if has_decimal:
                return row.get("over_edge_decimal") if side == "over" else row.get("under_edge_decimal")
            if has_standard:
                return row.get("over_edge") if side == "over" else row.get("under_edge")

        return None

    df["selected_edge"] = df.apply(pick_edge, axis=1)
    return df


def grade_league(league):
    score_dir  = NBA_SCORE_DIR   if league == "NBA" else NCAAB_SCORE_DIR
    output_dir = NBA_OUTPUT      if league == "NBA" else NCAAB_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)

    # Wipe existing graded output before regenerating
    for stale in output_dir.glob("*.csv"):
        stale.unlink(missing_ok=True)

    # Load the select file for this league
    pattern = "*nba*.csv" if league == "NBA" else "*ncaab*.csv"
    select_files = list(SELECT_DIR.glob(pattern))

    if not select_files:
        print(f"[{league}] No select file found matching {pattern}")
        return

    select_df = pd.concat([pd.read_csv(f) for f in select_files], ignore_index=True)

    # Normalize game_date in select file (ensure underscore format)
    select_df["game_date"] = select_df["game_date"].astype(str).str.strip().str.replace("-", "_")

    # Iterate over existing score files — this is the source of truth for what dates have been played
    score_files = sorted(score_dir.glob(f"*_final_scores_{league}.csv"))

    if not score_files:
        print(f"[{league}] No score files found in {score_dir}")
        return

    all_results = []

    for score_file in score_files:
        scores = pd.read_csv(score_file)

        # Normalize game_date in score file
        scores["game_date"] = scores["game_date"].astype(str).str.strip().str.replace("-", "_")

        # Get the dates present in this score file
        score_dates = scores["game_date"].unique()

        # Filter select rows to only those dates
        sub = select_df[select_df["game_date"].isin(score_dates)].copy()

        if sub.empty:
            print(f"[{league}] No select rows found for dates {score_dates} — skipping {score_file.name}")
            continue

        merged = pd.merge(
            sub,
            scores,
            on=["away_team", "home_team", "game_date"],
            how="inner"
        )

        if merged.empty:
            print(f"[{league}] Merge returned no rows for {score_file.name} — check team name consistency")
            continue

        merged = clean_merge_columns(merged)
        merged = rebuild_selected_edge(merged)
        merged["bet_result"] = merged.apply(determine_outcome, axis=1)

        all_results.append(merged)
        print(f"[{league}] Graded {len(merged)} bets from {score_file.name}")

    if all_results:
        final = pd.concat(all_results, ignore_index=True)

        # Deduplicate — keep last in case a date appears in multiple score files
        key_cols = ["game_date", "away_team", "home_team", "market_type", "bet_side"]
        key_cols = [c for c in key_cols if c in final.columns]
        final = final.drop_duplicates(subset=key_cols, keep="last")

        out_path = output_dir / f"{league}_final.csv"
        final.to_csv(out_path, index=False)
        print(f"[{league}] Saved {len(final)} rows to {out_path}")
    else:
        print(f"[{league}] No results to save.")


def main():
    grade_league("NBA")
    grade_league("NCAAB")
    print("Grading complete.")


if __name__ == "__main__":
    main()
