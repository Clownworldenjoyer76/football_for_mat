#!/usr/bin/env python3
"""Build cumulative NFL graded-bet CSV reports from intermediate/work_nfl.csv."""

from __future__ import annotations

import csv
import math
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
SCRIPTS_DIR = SCRIPT_DIR.parent
NFL_ROOT = SCRIPT_DIR.parents[1]
REPORT_ROOT = NFL_ROOT / "errors"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_reporter import PipelineReporter


pd = None

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

INPUT_COLUMNS = [
    "season", "season_type", "week", "game_id", "game_date",
    "away_team", "home_team", "away_score", "home_score", "status",
    "market", "selection", "line", "odds_american", "result",
    "game_time", "commence_time", "edt_time",
    "market_type", "bet_side", "model_prob", "implied_prob", "edge", "ev",
    "full_kelly", "kelly", "selection_reason",
    "implied_prob_source", "edge_source", "ev_source", "full_kelly_source", "kelly_source",
    "final_total", "bet_result", "bet_units",
    "selected_source_file", "grading_generated_at_utc",
    "side_group", "odds_value", "ev_value", "model_prob_value",
    "kelly_value", "kelly_value_source", "spread_value", "total_value",
    "day_night", "ev_bucket", "odds_bucket", "kelly_bucket",
    "win_prob_bucket", "spread_range_bucket", "spread_line_bucket",
    "total_range_bucket", "total_line_bucket", "week_label",
]

BET_KEY = [
    "season",
    "season_type",
    "week",
    "game_id",
    "market_type",
    "bet_side",
    "line",
]

VALID_SIDES = {
    "moneyline": {"home", "away"},
    "spread": {"home", "away"},
    "total": {"over", "under"},
}

EXPECTED_SIDE_GROUP = {
    "moneyline": {"home": "HOME", "away": "AWAY"},
    "spread": {"home": "HOME", "away": "AWAY"},
    "total": {"over": "OVER", "under": "UNDER"},
}

VALID_RESULTS = {"Win", "Loss", "Push"}
VALID_MARKETS = {"moneyline", "spread", "total"}

EV_BUCKETS = {
    "<0", "0.00_to_0.0099", "0.01_to_0.0199", "0.02_to_0.0299",
    "0.03_to_0.0399", "0.04_to_0.0499", "0.05_to_0.0749",
    "0.075_to_0.0999", "0.10_plus", "UNBUCKETED",
}
ODDS_BUCKETS = {
    "minus_200_or_lower", "minus_199_to_minus_150", "minus_149_to_minus_125",
    "minus_124_to_minus_110", "minus_109_to_minus_101",
    "minus_100_to_plus_100", "plus_101_to_plus_125",
    "plus_126_to_plus_150", "plus_151_to_plus_200",
    "plus_201_or_higher",
}
KELLY_BUCKETS = {
    "zero_or_below", "0.001_to_0.0099", "0.01_to_0.0199",
    "0.02_to_0.0299", "0.03_to_0.0499", "0.05_to_0.0999",
    "0.10_to_0.1499", "0.15_to_0.1999", "0.20_plus", "UNBUCKETED",
}
WIN_PROB_BUCKETS = {
    "<50", "50_to_54.9", "55_to_59.9", "60_to_64.9",
    "65_to_69.9", "70_to_74.9", "75_to_79.9", "80_plus",
    "UNBUCKETED",
}
SPREAD_RANGE_BUCKETS = {
    "0_to_2.5", "3_to_3.5", "4_to_6.5", "7_to_9.5",
    "10_to_13.5", "14_plus", "UNBUCKETED",
}
TOTAL_RANGE_BUCKETS = {
    "37.5_or_lower", "38_to_40.5", "41_to_43.5", "44_to_46.5",
    "47_to_49.5", "50_to_52.5", "53_plus", "UNBUCKETED",
}

METRIC_DEFINITIONS = [
    {"metric": "Win_Pct", "definition": "wins / (wins + losses)", "push_handling": "pushes excluded"},
    {"metric": "Win_Pct_All_Bets", "definition": "wins / (wins + losses + pushes)", "push_handling": "pushes included"},
    {"metric": "units", "definition": "1 unit risked per bet; loss=-1; push=0; American-odds win payout", "push_handling": "push=0 units"},
    {"metric": "ROI_Excluding_Pushes", "definition": "units / (wins + losses)", "push_handling": "pushes excluded"},
    {"metric": "ROI_Including_Pushes", "definition": "units / (wins + losses + pushes)", "push_handling": "pushes included"},
    {"metric": "ev", "definition": "selected EV when present; otherwise derived from model probability and American odds by grader", "push_handling": "not applicable"},
    {"metric": "kelly reporting bucket", "definition": "selected kelly when present; otherwise full_kelly fallback in analyzer", "push_handling": "not applicable"},
]

BET_LOG_COLUMNS = [
    "season", "season_type", "week", "game_date", "game_id", "away_team", "home_team",
    "market_type", "bet_side", "line", "odds_american", "model_prob", "implied_prob",
    "edge", "ev", "full_kelly", "kelly", "bet_result", "bet_units", "away_score",
    "home_score", "final_total", "status", "selected_source_file",
]

MARKET_REPORT_SPECS = {
    "moneyline": {
        "directory": "moneyline",
        "file_key": "moneyline",
        "sides": {"HOME", "AWAY"},
        "dimensions": {
            "ev": "ev_bucket", "odds": "odds_bucket", "kelly": "kelly_bucket",
            "win_prob": "win_prob_bucket", "week": "week_label",
        },
    },
    "spread": {
        "directory": "spread",
        "file_key": "spread",
        "sides": {"HOME", "AWAY"},
        "dimensions": {
            "ev": "ev_bucket", "odds": "odds_bucket", "kelly": "kelly_bucket",
            "win_prob": "win_prob_bucket", "spread_range": "spread_range_bucket",
            "line": "spread_line_bucket", "week": "week_label", "side": "side_group",
        },
    },
    "total": {
        "directory": "totals",
        "file_key": "total",
        "sides": {"OVER", "UNDER"},
        "dimensions": {
            "ev": "ev_bucket", "odds": "odds_bucket", "kelly": "kelly_bucket",
            "win_prob": "win_prob_bucket", "total_range": "total_range_bucket",
            "line": "total_line_bucket", "week": "week_label", "side": "side_group",
        },
    },
}

EPSILON = 1e-9


def load_runtime_dependencies(reporter: PipelineReporter) -> None:
    global pd
    try:
        import pandas as pd_module
    except Exception:
        reporter.set_detail("dependency_imports_ok", False)
        raise
    pd = pd_module
    reporter.set_detail("dependency_imports_ok", True)


def fail(message: str) -> None:
    raise RuntimeError(message)


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.casefold() in {"", "nan", "none", "null", "nat", "<na>"}:
        return ""
    return text


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


def parse_positive_int(value: Any, *, label: str) -> int:
    number = to_float(value)
    if number is None or not number.is_integer() or number <= 0:
        fail(f"{label} must be a positive integer; found {value!r}")
    return int(number)


def parse_nonnegative_int(value: Any, *, label: str) -> int:
    number = to_float(value)
    if number is None or not number.is_integer() or number < 0:
        fail(f"{label} must be a nonnegative integer; found {value!r}")
    return int(number)


def require_finite_if_present(value: Any, *, label: str) -> float | None:
    text = clean(value)
    if not text:
        return None
    number = to_float(text)
    if number is None:
        fail(f"{label} must be finite when present; found {value!r}")
    return number


def numeric_equal(left: Any, right: Any) -> bool:
    left_num = to_float(left)
    right_num = to_float(right)
    if left_num is None or right_num is None:
        return left_num is None and right_num is None
    return abs(left_num - right_num) <= EPSILON


def validate_csv_header(path: Path, *, expected: list[str], label: str) -> None:
    if not path.exists():
        fail(f"{label}: file not found: {path}")
    if not path.is_file():
        fail(f"{label}: not a regular file: {path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
    except Exception as exc:
        fail(f"{label}: unable to read CSV header: {type(exc).__name__}: {exc}")
    if header is None:
        fail(f"{label}: CSV is empty")
    if any(not str(column).strip() for column in header):
        fail(f"{label}: blank CSV header name")
    seen: set[str] = set()
    duplicates: list[str] = []
    for column in header:
        if column in seen and column not in duplicates:
            duplicates.append(column)
        seen.add(column)
    if duplicates:
        fail(f"{label}: duplicate header columns: {duplicates}")
    if header != expected:
        fail(f"{label}: column contract failed")


def read_input() -> pd.DataFrame:
    validate_csv_header(INPUT_FILE, expected=INPUT_COLUMNS, label="reports input")
    try:
        return pd.read_csv(
            INPUT_FILE,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            encoding="utf-8-sig",
        )
    except Exception as exc:
        fail(f"reports input: CSV read failed: {type(exc).__name__}: {exc}")


def validate_date(value: Any, *, label: str) -> None:
    text = clean(value)
    try:
        datetime.strptime(text, "%Y-%m-%d")
    except ValueError:
        fail(f"{label} must be YYYY-MM-DD; found {value!r}")


def validate_iso_datetime(value: Any, *, label: str) -> None:
    text = clean(value)
    candidate = f"{text[:-1]}+00:00" if text.endswith("Z") else text
    try:
        datetime.fromisoformat(candidate)
    except ValueError:
        fail(f"{label} has invalid ISO datetime {text!r}")


def is_final_status(value: Any) -> bool:
    status = clean(value).casefold()
    return status.startswith("final") or status in {"completed", "complete", "game over"}


def validate_input_frame(frame: pd.DataFrame) -> None:
    if list(frame.columns) != INPUT_COLUMNS:
        fail("reports input: DataFrame column contract failed")

    seen_keys: set[tuple[str, ...]] = set()

    for index, row in frame.iterrows():
        row_number = index + 2
        prefix = f"reports input row {row_number}"

        for column in (
            "season", "season_type", "week", "game_id", "game_date",
            "away_team", "home_team", "away_score", "home_score", "status",
            "market", "selection", "odds_american", "result", "market_type",
            "bet_side", "bet_result", "bet_units", "selected_source_file",
            "grading_generated_at_utc", "side_group", "odds_bucket", "week_label",
        ):
            if not clean(row[column]):
                fail(f"{prefix}: {column} is blank")

        parse_positive_int(row["season"], label=f"{prefix} season")
        week = parse_positive_int(row["week"], label=f"{prefix} week")
        validate_date(row["game_date"], label=f"{prefix} game_date")

        away_team = clean(row["away_team"])
        home_team = clean(row["home_team"])
        if away_team == home_team:
            fail(f"{prefix}: away_team and home_team must differ")

        away_score = parse_nonnegative_int(row["away_score"], label=f"{prefix} away_score")
        home_score = parse_nonnegative_int(row["home_score"], label=f"{prefix} home_score")

        final_total = to_float(row["final_total"])
        if final_total is None or abs(final_total - (away_score + home_score)) > EPSILON:
            fail(f"{prefix}: final_total does not match scores")

        if not is_final_status(row["status"]):
            fail(f"{prefix}: status is not final: {clean(row['status'])!r}")

        validate_iso_datetime(
            row["grading_generated_at_utc"],
            label=f"{prefix} grading_generated_at_utc",
        )

        market_type = clean(row["market_type"])
        if market_type not in VALID_MARKETS or market_type != market_type.lower():
            fail(f"{prefix}: invalid market_type={market_type!r}")
        if clean(row["market"]) != market_type.upper():
            fail(f"{prefix}: market does not match market_type")

        bet_side = clean(row["bet_side"])
        if bet_side not in VALID_SIDES[market_type]:
            fail(f"{prefix}: invalid bet_side={bet_side!r} for market_type={market_type!r}")

        expected_group = EXPECTED_SIDE_GROUP[market_type][bet_side]
        if clean(row["side_group"]) != expected_group:
            fail(f"{prefix}: side_group does not match market_type/bet_side")
        if clean(row["selection"]) != bet_side.upper():
            fail(f"{prefix}: selection does not match bet_side")

        bet_result = clean(row["bet_result"])
        if bet_result not in VALID_RESULTS:
            fail(f"{prefix}: invalid bet_result={bet_result!r}")
        if clean(row["result"]) != bet_result.upper():
            fail(f"{prefix}: result does not match bet_result")

        odds = to_float(row["odds_american"])
        if odds is None or odds == 0:
            fail(f"{prefix}: invalid odds_american")
        if not numeric_equal(row["odds_value"], odds):
            fail(f"{prefix}: odds_value does not match odds_american")

        line_text = clean(row["line"])
        line_value = to_float(line_text) if line_text else None
        if market_type in {"spread", "total"} and line_value is None:
            fail(f"{prefix}: spread/total line is invalid")
        if market_type == "moneyline" and line_text:
            fail(f"{prefix}: moneyline line must be blank")

        units = to_float(row["bet_units"])
        expected_units = units_won(odds, bet_result)
        if units is None or expected_units is None or abs(units - expected_units) > EPSILON:
            fail(f"{prefix}: bet_units does not match odds/result")

        model_prob = require_finite_if_present(row["model_prob"], label=f"{prefix} model_prob")
        ev = require_finite_if_present(row["ev"], label=f"{prefix} ev")
        require_finite_if_present(row["implied_prob"], label=f"{prefix} implied_prob")
        require_finite_if_present(row["edge"], label=f"{prefix} edge")
        require_finite_if_present(row["full_kelly"], label=f"{prefix} full_kelly")
        require_finite_if_present(row["kelly"], label=f"{prefix} kelly")

        if not numeric_equal(row["ev_value"], ev):
            fail(f"{prefix}: ev_value does not match ev")
        if not numeric_equal(row["model_prob_value"], model_prob):
            fail(f"{prefix}: model_prob_value does not match model_prob")

        kelly_value = require_finite_if_present(row["kelly_value"], label=f"{prefix} kelly_value")
        kelly_source = clean(row["kelly_value_source"])
        selected_kelly = to_float(row["kelly"])
        full_kelly = to_float(row["full_kelly"])
        if selected_kelly is not None:
            if kelly_source != "selected_kelly" or not numeric_equal(kelly_value, selected_kelly):
                fail(f"{prefix}: selected kelly contract failed")
        elif full_kelly is not None:
            if kelly_source != "full_kelly_fallback" or not numeric_equal(kelly_value, full_kelly):
                fail(f"{prefix}: full kelly fallback contract failed")
        elif kelly_source or kelly_value is not None:
            fail(f"{prefix}: empty kelly contract failed")

        spread_value = require_finite_if_present(row["spread_value"], label=f"{prefix} spread_value")
        total_value = require_finite_if_present(row["total_value"], label=f"{prefix} total_value")
        if market_type == "spread":
            if not numeric_equal(spread_value, line_value) or total_value is not None:
                fail(f"{prefix}: spread_value/total_value contract failed")
        elif market_type == "total":
            if not numeric_equal(total_value, line_value) or spread_value is not None:
                fail(f"{prefix}: spread_value/total_value contract failed")
        elif spread_value is not None or total_value is not None:
            fail(f"{prefix}: moneyline spread_value/total_value must be blank")

        if clean(row["ev_bucket"]) not in EV_BUCKETS:
            fail(f"{prefix}: invalid ev_bucket")
        if clean(row["odds_bucket"]) not in ODDS_BUCKETS:
            fail(f"{prefix}: invalid odds_bucket")
        if clean(row["kelly_bucket"]) not in KELLY_BUCKETS:
            fail(f"{prefix}: invalid kelly_bucket")
        if clean(row["win_prob_bucket"]) not in WIN_PROB_BUCKETS:
            fail(f"{prefix}: invalid win_prob_bucket")
        if clean(row["spread_range_bucket"]) not in SPREAD_RANGE_BUCKETS:
            fail(f"{prefix}: invalid spread_range_bucket")
        if clean(row["total_range_bucket"]) not in TOTAL_RANGE_BUCKETS:
            fail(f"{prefix}: invalid total_range_bucket")

        spread_line_bucket = clean(row["spread_line_bucket"])
        total_line_bucket = clean(row["total_line_bucket"])
        if market_type == "spread":
            if spread_line_bucket in {"", "UNBUCKETED"} or to_float(spread_line_bucket) is None:
                fail(f"{prefix}: invalid spread_line_bucket")
            if total_line_bucket != "UNBUCKETED":
                fail(f"{prefix}: total_line_bucket must be UNBUCKETED for spread")
        elif market_type == "total":
            if total_line_bucket in {"", "UNBUCKETED"} or to_float(total_line_bucket) is None:
                fail(f"{prefix}: invalid total_line_bucket")
            if spread_line_bucket != "UNBUCKETED":
                fail(f"{prefix}: spread_line_bucket must be UNBUCKETED for total")
        else:
            if spread_line_bucket != "UNBUCKETED" or total_line_bucket != "UNBUCKETED":
                fail(f"{prefix}: line buckets must be UNBUCKETED for moneyline")

        if clean(row["week_label"]) != f"Week {week}":
            fail(f"{prefix}: week_label contract failed")

        day_night = clean(row["day_night"])
        if day_night not in {"", "Day", "Night"}:
            fail(f"{prefix}: invalid day_night={day_night!r}")

        key = tuple(clean(row[column]) for column in BET_KEY)
        if key in seen_keys:
            fail(f"{prefix}: duplicate bet key={key}")
        seen_keys.add(key)


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


def metric_definitions_frame() -> pd.DataFrame:
    return pd.DataFrame(METRIC_DEFINITIONS, columns=["metric", "definition", "push_handling"])


def build_dimension_frames(
    source: pd.DataFrame,
    bucket_column: str,
    *,
    valid_sides: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    filtered = source[source[bucket_column].astype(str).str.strip().ne("UNBUCKETED")].copy()
    filtered = filtered[filtered[bucket_column].astype(str).str.strip().ne("")]

    overall = aggregate(
        filtered,
        ["league", "market_type", bucket_column],
        variable_label=True,
    )

    side_filtered = filtered[filtered["side_group"].isin(valid_sides)].copy()
    side = aggregate(
        side_filtered,
        ["league", "market_type", "side_group", bucket_column],
        variable_label=True,
    )
    return overall, side


def build_report_frames(df: pd.DataFrame) -> dict[Path, pd.DataFrame]:
    frames: dict[Path, pd.DataFrame] = {}

    frames[Path("nfl_summary_overall.csv")] = aggregate(df, ["league", "market_type"])
    frames[Path("reports/overview/nfl_report_metric_definitions.csv")] = metric_definitions_frame()
    frames[Path("reports/overview/nfl_summary_overall.csv")] = overall_row(df)
    frames[Path("reports/overview/nfl_summary_by_market.csv")] = aggregate(
        df, ["league", "market_type"], variable_label=True
    )
    frames[Path("reports/overview/nfl_summary_by_side_group.csv")] = aggregate(
        df, ["league", "side_group"], variable_label=True
    )
    frames[Path("reports/overview/nfl_summary_by_season_type.csv")] = aggregate(
        df, ["league", "season_type"], variable_label=True
    )

    by_week = aggregate(df, ["league", "week"], variable_label=True)
    if not by_week.empty:
        by_week["_week_sort"] = pd.to_numeric(by_week["variable"], errors="coerce")
        by_week = by_week.sort_values(["_week_sort", "variable"], kind="mergesort").drop(columns=["_week_sort"])
        by_week["cumulative_units"] = pd.to_numeric(by_week["units"], errors="coerce").fillna(0).cumsum().round(4)
    else:
        by_week["cumulative_units"] = pd.Series(dtype=float)
    frames[Path("reports/overview/nfl_summary_by_week.csv")] = by_week

    by_date = aggregate(df, ["league", "game_date"], variable_label=True)
    if not by_date.empty:
        by_date = by_date.sort_values("variable", kind="mergesort")
        by_date["cumulative_units"] = pd.to_numeric(by_date["units"], errors="coerce").fillna(0).cumsum().round(4)
    else:
        by_date["cumulative_units"] = pd.Series(dtype=float)
    frames[Path("reports/overview/nfl_summary_by_date.csv")] = by_date

    frames[Path("reports/overview/nfl_summary_by_day_night.csv")] = aggregate(
        df, ["league", "day_night"], variable_label=True
    )
    frames[Path("reports/overview/nfl_bet_log.csv")] = df[BET_LOG_COLUMNS].copy()

    for market, spec in MARKET_REPORT_SPECS.items():
        subset = df[df["market_type"] == market].copy()
        for dimension, bucket_column in spec["dimensions"].items():
            overall, side = build_dimension_frames(
                subset,
                bucket_column,
                valid_sides=spec["sides"],
            )
            base = Path("reports") / spec["directory"] / f"nfl_{spec['file_key']}_by_{dimension}"
            frames[Path(f"{base}.csv")] = overall
            frames[Path(f"{base}_side_summary.csv")] = side

    return frames


def expected_output_paths() -> list[Path]:
    paths = [SUMMARY_FILE]
    paths.extend([
        OVERVIEW_DIR / "nfl_report_metric_definitions.csv",
        OVERVIEW_DIR / "nfl_summary_overall.csv",
        OVERVIEW_DIR / "nfl_summary_by_market.csv",
        OVERVIEW_DIR / "nfl_summary_by_side_group.csv",
        OVERVIEW_DIR / "nfl_summary_by_season_type.csv",
        OVERVIEW_DIR / "nfl_summary_by_week.csv",
        OVERVIEW_DIR / "nfl_summary_by_date.csv",
        OVERVIEW_DIR / "nfl_summary_by_day_night.csv",
        OVERVIEW_DIR / "nfl_bet_log.csv",
    ])
    for spec in MARKET_REPORT_SPECS.values():
        output_dir = REPORTS_DIR / spec["directory"]
        for dimension in spec["dimensions"]:
            base_name = f"nfl_{spec['file_key']}_by_{dimension}"
            paths.append(output_dir / f"{base_name}.csv")
            paths.append(output_dir / f"{base_name}_side_summary.csv")
    return paths


def relative_output_path(path: Path) -> Path:
    try:
        return path.relative_to(FINAL_ROOT)
    except ValueError as exc:
        raise RuntimeError(f"Output is outside FINAL_ROOT: {path}") from exc


def expected_schema(relative_path: Path) -> list[str]:
    name = relative_path.name
    if relative_path == Path("nfl_summary_overall.csv"):
        return ["league", "market_type"] + METRIC_COLUMNS
    if name == "nfl_report_metric_definitions.csv":
        return ["metric", "definition", "push_handling"]
    if name == "nfl_summary_overall.csv":
        return ["league"] + METRIC_COLUMNS
    if name == "nfl_bet_log.csv":
        return BET_LOG_COLUMNS
    if name in {"nfl_summary_by_week.csv", "nfl_summary_by_date.csv"}:
        return ["league", "variable"] + METRIC_COLUMNS + ["cumulative_units"]
    if relative_path.parts[:2] == ("reports", "overview"):
        return ["league", "variable"] + METRIC_COLUMNS
    if name.endswith("_side_summary.csv"):
        return ["league", "market_type", "side_group", "variable"] + METRIC_COLUMNS
    return ["league", "market_type", "variable"] + METRIC_COLUMNS


def validate_metric_frame(frame: pd.DataFrame, *, label: str) -> None:
    for index, row in frame.iterrows():
        row_number = index + 2
        wins = parse_nonnegative_int(row["Win"], label=f"{label} row {row_number} Win")
        losses = parse_nonnegative_int(row["Loss"], label=f"{label} row {row_number} Loss")
        pushes = parse_nonnegative_int(row["Push"], label=f"{label} row {row_number} Push")
        total = parse_nonnegative_int(row["Total"], label=f"{label} row {row_number} Total")
        excl = parse_nonnegative_int(
            row["bets_excluding_pushes"], label=f"{label} row {row_number} bets_excluding_pushes"
        )
        incl = parse_nonnegative_int(
            row["bets_including_pushes"], label=f"{label} row {row_number} bets_including_pushes"
        )
        if total != wins + losses + pushes or excl != wins + losses or incl != total:
            fail(f"{label} row {row_number}: count contract failed")

        win_pct = to_float(row["Win_Pct"])
        win_pct_all = to_float(row["Win_Pct_All_Bets"])
        units = to_float(row["units"])
        roi_excl = to_float(row["ROI_Excluding_Pushes"])
        roi_incl = to_float(row["ROI_Including_Pushes"])
        if None in {win_pct, win_pct_all, units, roi_excl, roi_incl}:
            fail(f"{label} row {row_number}: required metric is non-finite")
        if not 0.0 <= win_pct <= 1.0 or not 0.0 <= win_pct_all <= 1.0:
            fail(f"{label} row {row_number}: win percentage is outside [0, 1]")

        expected_win_pct = round(wins / excl, 4) if excl else 0.0
        expected_win_pct_all = round(wins / incl, 4) if incl else 0.0
        expected_roi_excl = round(units / excl, 4) if excl else 0.0
        expected_roi_incl = round(units / incl, 4) if incl else 0.0
        if abs(win_pct - expected_win_pct) > EPSILON:
            fail(f"{label} row {row_number}: Win_Pct contract failed")
        if abs(win_pct_all - expected_win_pct_all) > EPSILON:
            fail(f"{label} row {row_number}: Win_Pct_All_Bets contract failed")
        if abs(roi_excl - expected_roi_excl) > EPSILON:
            fail(f"{label} row {row_number}: ROI_Excluding_Pushes contract failed")
        if abs(roi_incl - expected_roi_incl) > EPSILON:
            fail(f"{label} row {row_number}: ROI_Including_Pushes contract failed")

        for column in ("avg_ev", "avg_odds", "avg_model_prob"):
            require_finite_if_present(row[column], label=f"{label} row {row_number} {column}")


def validate_report_frame(relative_path: Path, frame: pd.DataFrame, *, input_rows: int) -> None:
    schema = expected_schema(relative_path)
    label = str(relative_path)
    if list(frame.columns) != schema:
        fail(f"{label}: report schema contract failed")

    if relative_path.name == "nfl_report_metric_definitions.csv":
        if len(frame) != len(METRIC_DEFINITIONS):
            fail(f"{label}: metric-definition row-count contract failed")
        return

    if relative_path.name == "nfl_bet_log.csv":
        if len(frame) != input_rows:
            fail(f"{label}: bet-log row-count contract failed")
        return

    if relative_path == Path("reports/overview/nfl_summary_overall.csv") and len(frame) != 1:
        fail(f"{label}: overall summary must contain exactly one row")

    validate_metric_frame(frame, label=label)

    if "league" in frame.columns:
        invalid_league = frame[frame["league"].astype(str).str.strip().ne(LEAGUE)]
        if not invalid_league.empty:
            fail(f"{label}: invalid league value")

    if "market_type" in frame.columns:
        invalid_market = frame[
            ~frame["market_type"].astype(str).str.strip().isin(VALID_MARKETS)
        ]
        if not invalid_market.empty:
            fail(f"{label}: invalid market_type value")

    if "side_group" in frame.columns:
        invalid_side = frame[
            ~frame["side_group"].astype(str).str.strip().isin({"HOME", "AWAY", "OVER", "UNDER"})
        ]
        if not invalid_side.empty:
            fail(f"{label}: invalid side_group value")

    if "variable" in frame.columns and relative_path.name != "nfl_summary_by_day_night.csv":
        invalid_variable = frame[frame["variable"].astype(str).str.strip().eq("")]
        if not invalid_variable.empty:
            fail(f"{label}: blank report variable")

    if "cumulative_units" in frame.columns:
        for index, value in frame["cumulative_units"].items():
            if to_float(value) is None:
                fail(f"{label} row {index + 2}: invalid cumulative_units")


def normalized_rows(frame: pd.DataFrame) -> list[tuple[str, ...]]:
    return [
        tuple(clean(value) for value in row)
        for row in frame.itertuples(index=False, name=None)
    ]


def write_staged_report(stage_root: Path, relative_path: Path, frame: pd.DataFrame) -> Path:
    stage_path = stage_root / relative_path
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(stage_path, index=False, lineterminator="\n")
    with stage_path.open("r+b") as handle:
        os.fsync(handle.fileno())
    return stage_path


def read_report(path: Path, relative_path: Path) -> pd.DataFrame:
    schema = expected_schema(relative_path)
    validate_csv_header(path, expected=schema, label=f"report {relative_path}")
    try:
        return pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            encoding="utf-8-sig",
        )
    except Exception as exc:
        fail(f"report {relative_path}: CSV read failed: {type(exc).__name__}: {exc}")


def stage_report_set(
    frames: dict[Path, pd.DataFrame],
    *,
    input_rows: int,
) -> Path:
    stage_root = Path(tempfile.mkdtemp(prefix=".nfl_reports_stage_", dir=str(FINAL_ROOT)))
    try:
        expected_relatives = {relative_output_path(path) for path in expected_output_paths()}
        if set(frames) != expected_relatives:
            missing = sorted(str(path) for path in expected_relatives - set(frames))
            extra = sorted(str(path) for path in set(frames) - expected_relatives)
            fail(f"report frame set mismatch missing={missing} extra={extra}")

        for relative_path in sorted(frames, key=str):
            frame = frames[relative_path]
            validate_report_frame(relative_path, frame, input_rows=input_rows)
            stage_path = write_staged_report(stage_root, relative_path, frame)
            staged = read_report(stage_path, relative_path)
            validate_report_frame(relative_path, staged, input_rows=input_rows)
            if normalized_rows(staged) != normalized_rows(frame):
                fail(f"{relative_path}: staged roundtrip content mismatch")
        return stage_root
    except Exception:
        shutil.rmtree(stage_root, ignore_errors=True)
        raise


def managed_live_files() -> set[Path]:
    files: set[Path] = set()
    if SUMMARY_FILE.exists():
        files.add(SUMMARY_FILE)
    if REPORTS_DIR.exists():
        files.update(path for path in REPORTS_DIR.rglob("*.csv") if path.is_file())
    return files


def validate_live_report_set(*, input_rows: int) -> None:
    expected = expected_output_paths()
    for path in expected:
        relative_path = relative_output_path(path)
        frame = read_report(path, relative_path)
        validate_report_frame(relative_path, frame, input_rows=input_rows)

    expected_set = set(expected)
    unexpected = sorted(
        str(path)
        for path in managed_live_files()
        if path not in expected_set
    )
    if unexpected:
        fail(f"unexpected managed report files remain after publication: {unexpected}")


def publish_report_set(
    stage_root: Path,
    *,
    input_rows: int,
    reporter: PipelineReporter,
) -> None:
    expected = expected_output_paths()
    expected_set = set(expected)
    existing = managed_live_files()
    managed_union = expected_set | existing
    stale = existing - expected_set

    backup_root = Path(tempfile.mkdtemp(prefix=".nfl_reports_backup_", dir=str(FINAL_ROOT)))
    backed_up: set[Path] = set()
    publication_started = False

    reporter.update_details({
        "publication_mode": "transactional_multi_file_with_rollback",
        "publication_completed": False,
        "post_publish_validation": False,
        "rollback_performed": False,
        "stale_managed_files_removed": len(stale),
    })

    try:
        for live_path in sorted(existing, key=str):
            relative_path = relative_output_path(live_path)
            backup_path = backup_root / relative_path
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(live_path, backup_path)
            backed_up.add(live_path)

        publication_started = True

        for live_path in sorted(expected, key=str):
            relative_path = relative_output_path(live_path)
            stage_path = stage_root / relative_path
            live_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage_path, live_path)

        for stale_path in sorted(stale, key=str):
            stale_path.unlink(missing_ok=True)

        validate_live_report_set(input_rows=input_rows)

        reporter.update_details({
            "publication_completed": True,
            "post_publish_validation": True,
        })
    except Exception as publish_exc:
        if publication_started:
            try:
                for live_path in sorted(managed_union, key=str, reverse=True):
                    if live_path in backed_up:
                        relative_path = relative_output_path(live_path)
                        backup_path = backup_root / relative_path
                        live_path.parent.mkdir(parents=True, exist_ok=True)
                        restore_fd, restore_raw = tempfile.mkstemp(
                            prefix=f".{live_path.name}.restore.",
                            suffix=".tmp",
                            dir=str(live_path.parent),
                        )
                        os.close(restore_fd)
                        restore_path = Path(restore_raw)
                        try:
                            shutil.copy2(backup_path, restore_path)
                            os.replace(restore_path, live_path)
                        finally:
                            restore_path.unlink(missing_ok=True)
                    else:
                        live_path.unlink(missing_ok=True)

                reporter.update_details({
                    "publication_completed": False,
                    "post_publish_validation": False,
                    "rollback_performed": True,
                })
            except Exception as rollback_exc:
                reporter.update_details({
                    "publication_completed": False,
                    "post_publish_validation": False,
                    "rollback_performed": False,
                    "rollback_error_type": type(rollback_exc).__name__,
                    "rollback_error": str(rollback_exc),
                })
                raise RuntimeError(
                    "NFL report publication failed and rollback also failed: "
                    f"publication_error={publish_exc}; rollback_error={rollback_exc}"
                ) from rollback_exc
        raise
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)
        try:
            shutil.rmtree(backup_root, ignore_errors=False)
        except Exception as cleanup_exc:
            reporter.warning(
                "Temporary NFL report backup cleanup failed",
                backup_root=str(backup_root),
                error_type=type(cleanup_exc).__name__,
                error=str(cleanup_exc),
            )


def run(reporter: PipelineReporter) -> None:
    reporter.add_input(INPUT_FILE)
    for output_path in expected_output_paths():
        reporter.add_output(output_path)

    load_runtime_dependencies(reporter)

    frame = read_input()
    validate_input_frame(frame)
    df = enrich(frame)
    frames = build_report_frames(df)

    reporter.set_rows(rows_in=len(frame), rows_out=sum(len(item) for item in frames.values()))
    reporter.update_details({
        "input_rows_validated": len(frame),
        "expected_report_files": len(expected_output_paths()),
        "generated_report_files": len(frames),
        "staged_roundtrip_verified": False,
        "publication_completed": False,
        "post_publish_validation": False,
        "rollback_performed": False,
    })

    stage_root = stage_report_set(frames, input_rows=len(frame))
    reporter.set_detail("staged_roundtrip_verified", True)

    publish_report_set(
        stage_root,
        input_rows=len(frame),
        reporter=reporter,
    )

    print(f"NFL reports complete. rows={len(df)} reports={REPORTS_DIR}")


def main() -> int:
    try:
        with PipelineReporter(
            script=SCRIPT_PATH,
            stage="04_final_results",
            report_root=REPORT_ROOT,
            pipeline="NFL",
            league="NFL",
            extra_context={"component": "graded-bet reports"},
        ) as reporter:
            run(reporter)
        return 0
    except Exception as exc:
        print(
            f"ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
