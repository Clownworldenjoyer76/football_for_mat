#!/usr/bin/env python3
"""Build cumulative NFL graded-bet CSV reports from intermediate/work_nfl.csv."""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
NFL_ROOT = SCRIPT_DIR.parents[1]
FINAL_ROOT = NFL_ROOT / "04_final_results"
INPUT_FILE = FINAL_ROOT / "intermediate" / "work_nfl.csv"
SUMMARY_FILE = FINAL_ROOT / "nfl_summary_overall.csv"
REPORTS_DIR = FINAL_ROOT / "reports"
OVERVIEW_DIR = REPORTS_DIR / "overview"
ML_DIR = REPORTS_DIR / "moneyline"
SPREAD_DIR = REPORTS_DIR / "spread"
TOTAL_DIR = REPORTS_DIR / "totals"
LEAGUE = "NFL"

METRIC_COLUMNS = [
    "Win", "Loss", "Push", "Total", "bets_excluding_pushes", "bets_including_pushes",
    "Win_Pct", "Win_Pct_All_Bets", "units", "ROI_Excluding_Pushes",
    "ROI_Including_Pushes", "avg_ev", "avg_odds", "avg_model_prob",
]


def to_float(value: Any) -> float | None:
    try:
        result = float(str(value).strip())
        return result if math.isfinite(result) else None
    except Exception:
        return None


def units_won(odds: Any, result: Any) -> float | None:
    result = str(result).strip().title()
    if result == "Push": return 0.0
    if result == "Loss": return -1.0
    if result != "Win": return None
    odds_num = to_float(odds)
    if odds_num is None or odds_num == 0: return None
    return odds_num / 100.0 if odds_num > 0 else 100.0 / abs(odds_num)


def clear_outputs() -> None:
    if REPORTS_DIR.exists():
        shutil.rmtree(REPORTS_DIR)
    for directory in [OVERVIEW_DIR, ML_DIR, SPREAD_DIR, TOTAL_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def enrich(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df["market_type"] = df["market_type"].astype(str).str.strip().str.lower()
    df["bet_side"] = df["bet_side"].astype(str).str.strip().str.lower()
    df["bet_result"] = df["bet_result"].astype(str).str.strip().str.title()
    df["league"] = LEAGUE

    if "side_group" not in df.columns:
        def make_side(row: pd.Series) -> str:
            market = row["market_type"]
            side = row["bet_side"]
            if market in {"moneyline", "spread"}:
                return {"home": "HOME", "away": "AWAY"}.get(side, "")
            if market == "total":
                return {"over": "OVER", "under": "UNDER"}.get(side, "")
            return ""
        df["side_group"] = df.apply(make_side, axis=1)

    if "bet_units" in df.columns:
        supplied = pd.to_numeric(df["bet_units"], errors="coerce")
    else:
        supplied = pd.Series(index=df.index, dtype=float)
    computed = df.apply(lambda row: units_won(row.get("odds_american"), row.get("bet_result")), axis=1)
    computed = pd.to_numeric(computed, errors="coerce")
    df["bet_units"] = supplied.where(supplied.notna(), computed)
    return df


def build_metric_row(sub: pd.DataFrame) -> dict[str, Any]:
    wins = int((sub["bet_result"] == "Win").sum())
    losses = int((sub["bet_result"] == "Loss").sum())
    pushes = int((sub["bet_result"] == "Push").sum())
    excl = wins + losses
    incl = wins + losses + pushes

    unit_vals = pd.to_numeric(sub["bet_units"], errors="coerce").dropna()
    units = round(float(unit_vals.sum()), 4) if not unit_vals.empty else 0.0

    ev_vals = pd.to_numeric(sub.get("ev", pd.Series(dtype=float)), errors="coerce").dropna()
    odds_vals = pd.to_numeric(sub.get("odds_american", pd.Series(dtype=float)), errors="coerce").dropna()
    prob_vals = pd.to_numeric(sub.get("model_prob", pd.Series(dtype=float)), errors="coerce").dropna()

    return {
        "Win": wins,
        "Loss": losses,
        "Push": pushes,
        "Total": incl,
        "bets_excluding_pushes": excl,
        "bets_including_pushes": incl,
        "Win_Pct": round(wins / excl, 4) if excl else 0.0,
        "Win_Pct_All_Bets": round(wins / incl, 4) if incl else 0.0,
        "units": units,
        "ROI_Excluding_Pushes": round(units / excl, 4) if excl else 0.0,
        "ROI_Including_Pushes": round(units / incl, 4) if incl else 0.0,
        "avg_ev": round(float(ev_vals.mean()), 4) if not ev_vals.empty else None,
        "avg_odds": round(float(odds_vals.mean()), 1) if not odds_vals.empty else None,
        "avg_model_prob": round(float(prob_vals.mean()), 4) if not prob_vals.empty else None,
    }


def aggregate(df: pd.DataFrame, group_cols: list[str], variable_label: bool = False) -> pd.DataFrame:
    prefix = ["league", "market_type"] if "market_type" in group_cols else ["league"]
    if variable_label:
        prefix = [col for col in group_cols[:-1]] + ["variable"]
    else:
        prefix = list(group_cols)

    if df.empty:
        return pd.DataFrame(columns=prefix + METRIC_COLUMNS)

    rows: list[dict[str, Any]] = []
    for keys, sub in df.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: dict[str, Any] = {}
        for index, column in enumerate(group_cols):
            label = "variable" if variable_label and index == len(group_cols) - 1 else column
            row[label] = keys[index]
        row.update(build_metric_row(sub))
        rows.append(row)
    return pd.DataFrame(rows, columns=prefix + METRIC_COLUMNS)


def overall_row(df: pd.DataFrame) -> pd.DataFrame:
    row = {"league": LEAGUE}
    row.update(build_metric_row(df))
    return pd.DataFrame([row], columns=["league"] + METRIC_COLUMNS)


def write_metric_definitions() -> None:
    definitions = pd.DataFrame([
        {"metric": "Win_Pct", "definition": "wins / (wins + losses)", "push_handling": "pushes excluded"},
        {"metric": "Win_Pct_All_Bets", "definition": "wins / (wins + losses + pushes)", "push_handling": "pushes included"},
        {"metric": "units", "definition": "1 unit risked per bet; loss=-1; push=0; American-odds win payout", "push_handling": "push=0 units"},
        {"metric": "ROI_Excluding_Pushes", "definition": "units / (wins + losses)", "push_handling": "pushes excluded"},
        {"metric": "ROI_Including_Pushes", "definition": "units / (wins + losses + pushes)", "push_handling": "pushes included"},
        {"metric": "ev", "definition": "selected EV when present; otherwise derived from model probability and American odds by grader", "push_handling": "not applicable"},
        {"metric": "kelly reporting bucket", "definition": "selected kelly when present; otherwise full_kelly fallback in analyzer", "push_handling": "not applicable"},
    ])
    write_csv(definitions, OVERVIEW_DIR / "nfl_report_metric_definitions.csv")


def build_top_summary(df: pd.DataFrame) -> None:
    out = aggregate(df, ["league", "market_type"])
    write_csv(out, SUMMARY_FILE)


def build_overview(df: pd.DataFrame) -> None:
    write_metric_definitions()
    write_csv(overall_row(df), OVERVIEW_DIR / "nfl_summary_overall.csv")
    write_csv(aggregate(df, ["league", "market_type"], variable_label=True), OVERVIEW_DIR / "nfl_summary_by_market.csv")
    write_csv(aggregate(df, ["league", "side_group"], variable_label=True), OVERVIEW_DIR / "nfl_summary_by_side_group.csv")
    write_csv(aggregate(df, ["league", "season_type"], variable_label=True), OVERVIEW_DIR / "nfl_summary_by_season_type.csv")

    by_week = aggregate(df, ["league", "week"], variable_label=True)
    if not by_week.empty:
        by_week["_week_sort"] = pd.to_numeric(by_week["variable"], errors="coerce")
        by_week = by_week.sort_values(["_week_sort", "variable"], kind="mergesort").drop(columns=["_week_sort"])
        by_week["cumulative_units"] = pd.to_numeric(by_week["units"], errors="coerce").fillna(0).cumsum().round(4)
    write_csv(by_week, OVERVIEW_DIR / "nfl_summary_by_week.csv")

    by_date = aggregate(df, ["league", "game_date"], variable_label=True)
    if not by_date.empty:
        by_date = by_date.sort_values("variable", kind="mergesort")
        by_date["cumulative_units"] = pd.to_numeric(by_date["units"], errors="coerce").fillna(0).cumsum().round(4)
    write_csv(by_date, OVERVIEW_DIR / "nfl_summary_by_date.csv")

    if "day_night" in df.columns:
        write_csv(aggregate(df, ["league", "day_night"], variable_label=True), OVERVIEW_DIR / "nfl_summary_by_day_night.csv")
    else:
        write_csv(pd.DataFrame(columns=["league", "variable"] + METRIC_COLUMNS), OVERVIEW_DIR / "nfl_summary_by_day_night.csv")

    log_columns = [
        "season", "season_type", "week", "game_date", "game_id", "away_team", "home_team",
        "market_type", "bet_side", "line", "odds_american", "model_prob", "implied_prob",
        "edge", "ev", "full_kelly", "kelly", "bet_result", "bet_units", "away_score",
        "home_score", "final_total", "status", "selected_source_file",
    ]
    available = [column for column in log_columns if column in df.columns]
    write_csv(df[available].copy(), OVERVIEW_DIR / "nfl_bet_log.csv")


def write_dimension(
    source: pd.DataFrame,
    bucket_column: str,
    output_dir: Path,
    base_name: str,
    *,
    valid_sides: set[str],
) -> None:
    overall_path = output_dir / f"{base_name}.csv"
    side_path = output_dir / f"{base_name}_side_summary.csv"

    if bucket_column not in source.columns:
        write_csv(pd.DataFrame(columns=["league", "market_type", "variable"] + METRIC_COLUMNS), overall_path)
        write_csv(pd.DataFrame(columns=["league", "market_type", "side_group", "variable"] + METRIC_COLUMNS), side_path)
        return

    filtered = source[source[bucket_column].astype(str).str.strip().ne("UNBUCKETED")].copy()
    filtered = filtered[filtered[bucket_column].astype(str).str.strip().ne("")]

    write_csv(
        aggregate(filtered, ["league", "market_type", bucket_column], variable_label=True),
        overall_path,
    )

    side_filtered = filtered[filtered["side_group"].isin(valid_sides)].copy()
    write_csv(
        aggregate(side_filtered, ["league", "market_type", "side_group", bucket_column], variable_label=True),
        side_path,
    )


def build_market_reports(df: pd.DataFrame) -> None:
    specs = {
        "moneyline": {
            "directory": ML_DIR,
            "file_key": "moneyline",
            "sides": {"HOME", "AWAY"},
            "dimensions": {
                "ev": "ev_bucket", "odds": "odds_bucket", "kelly": "kelly_bucket",
                "win_prob": "win_prob_bucket",
            },
        },
        "spread": {
            "directory": SPREAD_DIR,
            "file_key": "spread",
            "sides": {"HOME", "AWAY"},
            "dimensions": {
                "ev": "ev_bucket", "odds": "odds_bucket", "kelly": "kelly_bucket",
                "win_prob": "win_prob_bucket", "spread_range": "spread_range_bucket",
                "line": "spread_line_bucket", "side": "side_group",
            },
        },
        "total": {
            "directory": TOTAL_DIR,
            "file_key": "total",
            "sides": {"OVER", "UNDER"},
            "dimensions": {
                "ev": "ev_bucket", "odds": "odds_bucket", "kelly": "kelly_bucket",
                "win_prob": "win_prob_bucket", "total_range": "total_range_bucket",
                "line": "total_line_bucket", "side": "side_group",
            },
        },
    }

    for market, spec in specs.items():
        subset = df[df["market_type"] == market].copy()
        for dimension, bucket_column in spec["dimensions"].items():
            write_dimension(
                subset,
                bucket_column,
                spec["directory"],
                f"nfl_{spec['file_key']}_by_{dimension}",
                valid_sides=spec["sides"],
            )


def main() -> None:
    clear_outputs()
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")
    frame = pd.read_csv(INPUT_FILE, dtype=str, keep_default_na=False)
    df = enrich(frame)
    build_top_summary(df)
    build_overview(df)
    build_market_reports(df)
    print(f"NFL reports complete. rows={len(df)} reports={REPORTS_DIR}")


if __name__ == "__main__":
    main()
