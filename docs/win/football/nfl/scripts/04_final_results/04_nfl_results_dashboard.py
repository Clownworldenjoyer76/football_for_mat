#!/usr/bin/env python3
"""Build the NFL graded-bets dashboard data from generated report CSVs.

Every report CSV is validated (schema, metrics and cross-report reconciliation),
then the datasets read live by the React NFL page are written to
frontend/data/dashboards/nfl.json as
{"generated_at": <UTC ISO>, "data": {<dataset>: [<row>, ...]}}, where each row is
one CSV row as {header: string value}.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sys
import traceback
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
SCRIPTS_DIR = SCRIPT_DIR.parent
NFL_ROOT = SCRIPT_DIR.parents[1]
REPO_ROOT = NFL_ROOT.parents[3]
REPORT_ROOT = NFL_ROOT / "errors"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_reporter import PipelineReporter


pd = None

BASE = NFL_ROOT / "04_final_results"
REPORTS = BASE / "reports"
OVERVIEW = REPORTS / "overview"
OUTPUT_FILE = REPO_ROOT / "frontend" / "data" / "dashboards" / "nfl.json"
ERROR_DIR = NFL_ROOT / "errors" / "04_final_results"
LOG_FILE = ERROR_DIR / "04_nfl_results_dashboard.txt"
LEAGUE = "NFL"
EPSILON = 1e-9

MARKETS = {
    "moneyline": {
        "label": "Moneyline", "directory": "moneyline", "file_key": "moneyline",
        "dimensions": ["ev", "odds", "kelly", "win_prob", "week"],
    },
    "spread": {
        "label": "Spread", "directory": "spread", "file_key": "spread",
        "dimensions": ["ev", "odds", "kelly", "win_prob", "spread_range", "line", "week", "side"],
    },
    "total": {
        "label": "Total", "directory": "totals", "file_key": "total",
        "dimensions": ["ev", "odds", "kelly", "win_prob", "total_range", "line", "week", "side"],
    },
}

METRIC_COLUMNS = [
    "Win", "Loss", "Push", "Total", "bets_excluding_pushes", "bets_including_pushes",
    "Win_Pct", "Win_Pct_All_Bets", "units", "ROI_Excluding_Pushes",
    "ROI_Including_Pushes", "avg_ev", "avg_odds", "avg_model_prob",
]

BET_LOG_COLUMNS = [
    "season", "season_type", "week", "game_date", "game_id", "away_team", "home_team",
    "market_type", "bet_side", "line", "odds_american", "model_prob", "implied_prob",
    "edge", "ev", "full_kelly", "kelly", "bet_result", "bet_units", "away_score",
    "home_score", "final_total", "status", "selected_source_file",
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

VALID_SIDE_GROUPS = {
    "moneyline": {"HOME", "AWAY"},
    "spread": {"HOME", "AWAY"},
    "total": {"OVER", "UNDER"},
}


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


def is_final_status(value: Any) -> bool:
    status = clean(value).casefold()
    return (
        status.startswith("final")
        or status in {"completed", "complete", "game over"}
    )


def units_won(odds: Any, result: Any) -> float | None:
    result = clean(result).title()
    if result == "Push":
        return 0.0
    if result == "Loss":
        return -1.0
    if result != "Win":
        return None
    odds_num = to_float(odds)
    if odds_num is None or odds_num == 0:
        return None
    return odds_num / 100.0 if odds_num > 0 else 100.0 / abs(odds_num)


def derive_side_group(row: pd.Series) -> str:
    market = clean(row.get("market_type")).lower()
    side = clean(row.get("bet_side")).lower()
    if market in {"moneyline", "spread"}:
        return {"home": "HOME", "away": "AWAY"}.get(side, "")
    if market == "total":
        return {"over": "OVER", "under": "UNDER"}.get(side, "")
    return ""


def validate_csv_header(path: Path, *, expected: list[str], label: str) -> None:
    if not path.exists():
        fail(f"{label}: file not found: {path}")
    if not path.is_file():
        fail(f"{label}: not a regular file: {path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle), None)
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


def expected_schema(path: Path) -> list[str]:
    if path == BASE / "nfl_summary_overall.csv":
        return ["league", "market_type"] + METRIC_COLUMNS
    if path == OVERVIEW / "nfl_summary_overall.csv":
        return ["league"] + METRIC_COLUMNS
    if path == OVERVIEW / "nfl_bet_log.csv":
        return BET_LOG_COLUMNS
    if path in {
        OVERVIEW / "nfl_summary_by_week.csv",
        OVERVIEW / "nfl_summary_by_date.csv",
    }:
        return ["league", "variable"] + METRIC_COLUMNS + ["cumulative_units"]
    if path.parent == OVERVIEW:
        return ["league", "variable"] + METRIC_COLUMNS
    if path.name.endswith("_side_summary.csv"):
        return ["league", "market_type", "side_group", "variable"] + METRIC_COLUMNS
    return ["league", "market_type", "variable"] + METRIC_COLUMNS


def dashboard_input_paths() -> list[Path]:
    paths = [
        BASE / "nfl_summary_overall.csv",
        OVERVIEW / "nfl_summary_overall.csv",
        OVERVIEW / "nfl_summary_by_market.csv",
        OVERVIEW / "nfl_summary_by_side_group.csv",
        OVERVIEW / "nfl_summary_by_week.csv",
        OVERVIEW / "nfl_summary_by_date.csv",
        OVERVIEW / "nfl_summary_by_season_type.csv",
        OVERVIEW / "nfl_summary_by_day_night.csv",
        OVERVIEW / "nfl_bet_log.csv",
    ]
    for cfg in MARKETS.values():
        market_dir = REPORTS / cfg["directory"]
        for dimension in cfg["dimensions"]:
            base_name = f"nfl_{cfg['file_key']}_by_{dimension}"
            paths.append(market_dir / f"{base_name}.csv")
            paths.append(market_dir / f"{base_name}_side_summary.csv")
    return paths


def read_report(path: Path) -> pd.DataFrame:
    schema = expected_schema(path)
    label = f"dashboard input {path}"
    validate_csv_header(path, expected=schema, label=label)
    try:
        return pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            encoding="utf-8-sig",
        )
    except Exception as exc:
        fail(f"{label}: CSV read failed: {type(exc).__name__}: {exc}")


def validate_metric_frame(frame: pd.DataFrame, *, label: str) -> None:
    for index, row in frame.iterrows():
        row_number = index + 2
        wins = parse_nonnegative_int(row["Win"], label=f"{label} row {row_number} Win")
        losses = parse_nonnegative_int(row["Loss"], label=f"{label} row {row_number} Loss")
        pushes = parse_nonnegative_int(row["Push"], label=f"{label} row {row_number} Push")
        total = parse_nonnegative_int(row["Total"], label=f"{label} row {row_number} Total")
        excl = parse_nonnegative_int(
            row["bets_excluding_pushes"],
            label=f"{label} row {row_number} bets_excluding_pushes",
        )
        incl = parse_nonnegative_int(
            row["bets_including_pushes"],
            label=f"{label} row {row_number} bets_including_pushes",
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
            require_finite_if_present(
                row[column],
                label=f"{label} row {row_number} {column}",
            )


def validate_bet_log(frame: pd.DataFrame) -> None:
    seen_keys: set[tuple[str, ...]] = set()
    for index, row in frame.iterrows():
        row_number = index + 2
        for column in (
            "season", "season_type", "week", "game_date", "game_id",
            "away_team", "home_team", "market_type", "bet_side",
            "odds_american", "bet_result", "bet_units", "away_score",
            "home_score", "final_total", "status", "selected_source_file",
        ):
            if not clean(row[column]):
                fail(f"nfl_bet_log.csv row {row_number}: {column} is blank")

        parse_positive_int(row["season"], label=f"nfl_bet_log.csv row {row_number} season")
        parse_positive_int(row["week"], label=f"nfl_bet_log.csv row {row_number} week")

        try:
            datetime.strptime(clean(row["game_date"]), "%Y-%m-%d")
        except ValueError:
            fail(f"nfl_bet_log.csv row {row_number}: invalid game_date")

        if clean(row["away_team"]) == clean(row["home_team"]):
            fail(f"nfl_bet_log.csv row {row_number}: away_team and home_team must differ")

        market = clean(row["market_type"]).lower()
        side = clean(row["bet_side"]).lower()
        if market not in VALID_SIDES:
            fail(f"nfl_bet_log.csv row {row_number}: invalid market_type={market!r}")
        if side not in VALID_SIDES[market]:
            fail(f"nfl_bet_log.csv row {row_number}: invalid bet_side={side!r} for {market}")

        line_text = clean(row["line"])
        if market in {"spread", "total"}:
            if to_float(line_text) is None:
                fail(f"nfl_bet_log.csv row {row_number}: invalid line")
        elif line_text:
            fail(f"nfl_bet_log.csv row {row_number}: moneyline line must be blank")

        odds = to_float(row["odds_american"])
        if odds is None or odds == 0:
            fail(f"nfl_bet_log.csv row {row_number}: invalid odds_american")

        for column in (
            "model_prob", "implied_prob", "edge", "ev", "full_kelly", "kelly"
        ):
            require_finite_if_present(
                row[column],
                label=f"nfl_bet_log.csv row {row_number} {column}",
            )

        bet_result = clean(row["bet_result"]).title()
        if bet_result not in {"Win", "Loss", "Push"}:
            fail(f"nfl_bet_log.csv row {row_number}: invalid bet_result")

        units = to_float(row["bet_units"])
        expected_units = units_won(odds, bet_result)
        if units is None or expected_units is None or abs(units - expected_units) > EPSILON:
            fail(f"nfl_bet_log.csv row {row_number}: bet_units contract failed")

        away_score = parse_nonnegative_int(
            row["away_score"],
            label=f"nfl_bet_log.csv row {row_number} away_score",
        )
        home_score = parse_nonnegative_int(
            row["home_score"],
            label=f"nfl_bet_log.csv row {row_number} home_score",
        )
        final_total = to_float(row["final_total"])
        if final_total is None or abs(final_total - (away_score + home_score)) > EPSILON:
            fail(f"nfl_bet_log.csv row {row_number}: final_total contract failed")

        if not is_final_status(row["status"]):
            fail(f"nfl_bet_log.csv row {row_number}: status is not final")

        key = tuple(clean(row[column]) for column in BET_KEY)
        if key in seen_keys:
            fail(f"nfl_bet_log.csv row {row_number}: duplicate bet key={key}")
        seen_keys.add(key)


def market_from_path(path: Path) -> str | None:
    parent = path.parent.name
    if parent == "moneyline":
        return "moneyline"
    if parent == "spread":
        return "spread"
    if parent == "totals":
        return "total"
    return None


def validate_report_frame(path: Path, frame: pd.DataFrame) -> None:
    schema = expected_schema(path)
    if list(frame.columns) != schema:
        fail(f"{path}: dashboard schema contract failed")

    if path == OVERVIEW / "nfl_bet_log.csv":
        validate_bet_log(frame)
        return

    if path == OVERVIEW / "nfl_summary_overall.csv" and len(frame) != 1:
        fail(f"{path}: headline summary must contain exactly one row")

    validate_metric_frame(frame, label=str(path))

    if "league" in frame.columns:
        bad = frame[frame["league"].astype(str).str.strip().ne(LEAGUE)]
        if not bad.empty:
            fail(f"{path}: invalid league value")

    if path == BASE / "nfl_summary_overall.csv":
        values = [clean(value) for value in frame["market_type"]]
        if any(value not in VALID_SIDES for value in values):
            fail(f"{path}: invalid market_type value")
        if len(values) != len(set(values)):
            fail(f"{path}: duplicate market_type row")
        return

    if path.parent == OVERVIEW:
        if path == OVERVIEW / "nfl_summary_overall.csv":
            return
        values = [clean(value) for value in frame["variable"]]
        if path != OVERVIEW / "nfl_summary_by_day_night.csv" and any(not value for value in values):
            fail(f"{path}: blank report variable")
        if len(values) != len(set(values)):
            fail(f"{path}: duplicate report variable")

        if path == OVERVIEW / "nfl_summary_by_market.csv":
            if any(value not in VALID_SIDES for value in values):
                fail(f"{path}: invalid market variable")
        elif path == OVERVIEW / "nfl_summary_by_side_group.csv":
            if any(value not in {"HOME", "AWAY", "OVER", "UNDER"} for value in values):
                fail(f"{path}: invalid side-group variable")
        elif path == OVERVIEW / "nfl_summary_by_week.csv":
            for value in values:
                parse_positive_int(value, label=f"{path} week variable")
        elif path == OVERVIEW / "nfl_summary_by_date.csv":
            for value in values:
                try:
                    datetime.strptime(value, "%Y-%m-%d")
                except ValueError:
                    fail(f"{path}: invalid date variable {value!r}")
        elif path == OVERVIEW / "nfl_summary_by_day_night.csv":
            if any(value not in {"", "Day", "Night"} for value in values):
                fail(f"{path}: invalid day/night variable")

        if "cumulative_units" in frame.columns:
            for index, value in frame["cumulative_units"].items():
                if to_float(value) is None:
                    fail(f"{path} row {index + 2}: invalid cumulative_units")
        return

    expected_market = market_from_path(path)
    if expected_market is None:
        fail(f"{path}: unrecognized market report path")

    market_values = [clean(value) for value in frame["market_type"]]
    if any(value != expected_market for value in market_values):
        fail(f"{path}: market_type does not match report directory")

    variables = [clean(value) for value in frame["variable"]]
    if any(not value for value in variables):
        fail(f"{path}: blank market-report variable")

    if "_by_week" in path.name:
        for value in variables:
            match = re.fullmatch(r"Week ([1-9]\d*)", value)
            if match is None:
                fail(f"{path}: invalid week variable {value!r}")

    if "side_group" in frame.columns:
        sides = [clean(value) for value in frame["side_group"]]
        if any(value not in VALID_SIDE_GROUPS[expected_market] for value in sides):
            fail(f"{path}: invalid side_group for market")
        keys = list(zip(market_values, sides, variables))
    else:
        keys = list(zip(market_values, variables))

    if len(keys) != len(set(keys)):
        fail(f"{path}: duplicate report row identity")


def computed_metric_row(frame: pd.DataFrame) -> dict[str, float | int | None]:
    wins = int((frame["bet_result"].astype(str).str.strip().str.title() == "Win").sum())
    losses = int((frame["bet_result"].astype(str).str.strip().str.title() == "Loss").sum())
    pushes = int((frame["bet_result"].astype(str).str.strip().str.title() == "Push").sum())
    excl = wins + losses
    incl = wins + losses + pushes

    unit_vals = pd.to_numeric(frame["bet_units"], errors="coerce").dropna()
    ev_vals = pd.to_numeric(frame["ev"], errors="coerce").dropna()
    odds_vals = pd.to_numeric(frame["odds_american"], errors="coerce").dropna()
    prob_vals = pd.to_numeric(frame["model_prob"], errors="coerce").dropna()

    units = round(float(unit_vals.sum()), 4) if not unit_vals.empty else 0.0
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


def assert_metric_row(row: pd.Series, expected: dict[str, Any], *, label: str) -> None:
    for column in METRIC_COLUMNS:
        expected_value = expected[column]
        if expected_value is None:
            if clean(row[column]):
                fail(f"{label}: {column} should be blank")
            continue
        actual = to_float(row[column])
        if actual is None or abs(actual - float(expected_value)) > EPSILON:
            fail(f"{label}: {column} does not reconcile")


def report_rows_by(frame: pd.DataFrame, columns: list[str]) -> dict[tuple[str, ...], pd.Series]:
    result: dict[tuple[str, ...], pd.Series] = {}
    for _, row in frame.iterrows():
        key = tuple(clean(row[column]) for column in columns)
        if key in result:
            fail(f"duplicate reconciliation key={key}")
        result[key] = row
    return result


def validate_group_reconciliation(
    report: pd.DataFrame,
    bet_log: pd.DataFrame,
    *,
    source_columns: list[str],
    report_columns: list[str],
    source_key_transform=None,
    label: str,
) -> None:
    expected: dict[tuple[str, ...], dict[str, Any]] = {}
    grouped = bet_log.groupby(source_columns, dropna=False, sort=True)
    for keys, sub in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        text_keys = tuple(clean(value) for value in keys)
        if source_key_transform is not None:
            text_keys = source_key_transform(text_keys)
        expected[text_keys] = computed_metric_row(sub)

    actual = report_rows_by(report, report_columns)
    if set(actual) != set(expected):
        fail(f"{label}: reconciliation key set mismatch")

    for key, metrics in expected.items():
        assert_metric_row(actual[key], metrics, label=f"{label} key={key}")


def validate_cross_report_consistency(frames: dict[Path, pd.DataFrame]) -> None:
    bet_log = frames[OVERVIEW / "nfl_bet_log.csv"]
    headline = frames[OVERVIEW / "nfl_summary_overall.csv"]
    by_market = frames[BASE / "nfl_summary_overall.csv"]
    overview_market = frames[OVERVIEW / "nfl_summary_by_market.csv"]

    assert_metric_row(
        headline.iloc[0],
        computed_metric_row(bet_log),
        label="headline vs bet log",
    )

    validate_group_reconciliation(
        by_market,
        bet_log,
        source_columns=["market_type"],
        report_columns=["market_type"],
        label="by-market summary vs bet log",
    )
    validate_group_reconciliation(
        overview_market,
        bet_log,
        source_columns=["market_type"],
        report_columns=["variable"],
        label="overview market vs bet log",
    )
    validate_group_reconciliation(
        frames[OVERVIEW / "nfl_summary_by_week.csv"],
        bet_log,
        source_columns=["week"],
        report_columns=["variable"],
        label="overview week vs bet log",
    )
    validate_group_reconciliation(
        frames[OVERVIEW / "nfl_summary_by_date.csv"],
        bet_log,
        source_columns=["game_date"],
        report_columns=["variable"],
        label="overview date vs bet log",
    )
    validate_group_reconciliation(
        frames[OVERVIEW / "nfl_summary_by_season_type.csv"],
        bet_log,
        source_columns=["season_type"],
        report_columns=["variable"],
        label="overview season type vs bet log",
    )

    side_work = bet_log.copy()
    side_work["_side_group"] = side_work.apply(derive_side_group, axis=1)
    validate_group_reconciliation(
        frames[OVERVIEW / "nfl_summary_by_side_group.csv"],
        side_work,
        source_columns=["_side_group"],
        report_columns=["variable"],
        label="overview side group vs bet log",
    )

    day_night = frames[OVERVIEW / "nfl_summary_by_day_night.csv"]
    if sum(parse_nonnegative_int(value, label="day/night Total") for value in day_night["Total"]) != len(bet_log):
        fail("overview day/night Total does not reconcile to bet log")

    for market, cfg in MARKETS.items():
        market_log = side_work[side_work["market_type"].astype(str).str.strip().eq(market)].copy()
        base_name = f"nfl_{cfg['file_key']}_by_week"
        market_dir = REPORTS / cfg["directory"]

        week_report = frames[market_dir / f"{base_name}.csv"]
        validate_group_reconciliation(
            week_report,
            market_log,
            source_columns=["week"],
            report_columns=["variable"],
            source_key_transform=lambda key: (f"Week {key[0]}",),
            label=f"{market} week report vs bet log",
        )

        week_side_report = frames[market_dir / f"{base_name}_side_summary.csv"]
        validate_group_reconciliation(
            week_side_report,
            market_log,
            source_columns=["_side_group", "week"],
            report_columns=["side_group", "variable"],
            source_key_transform=lambda key: (key[0], f"Week {key[1]}"),
            label=f"{market} week side report vs bet log",
        )


# Datasets read by the React NFL page, in the order the page's data file uses.
REACT_OVERVIEW = {
    "bet_log": "nfl_bet_log.csv",
    "summary_overall": "nfl_summary_overall.csv",
    "summary_by_market": "nfl_summary_by_market.csv",
    "summary_by_side_group": "nfl_summary_by_side_group.csv",
    "summary_by_week": "nfl_summary_by_week.csv",
    "summary_by_date": "nfl_summary_by_date.csv",
    "summary_by_day_night": "nfl_summary_by_day_night.csv",
    "summary_by_season_type": "nfl_summary_by_season_type.csv",
}

REACT_MARKETS = [
    ("moneyline", "moneyline", ["ev", "kelly", "odds", "week", "win_prob"]),
    ("spread", "spread", ["ev", "kelly", "line", "odds", "side", "spread_range", "week", "win_prob"]),
    ("total", "totals", ["ev", "kelly", "line", "odds", "side", "total_range", "week", "win_prob"]),
]


def react_dataset_paths() -> dict[str, Path]:
    paths = {key: OVERVIEW / name for key, name in REACT_OVERVIEW.items()}
    for market, directory, dimensions in REACT_MARKETS:
        for dimension in dimensions:
            base_name = f"nfl_{market}_by_{dimension}"
            paths[f"{market}_by_{dimension}"] = REPORTS / directory / f"{base_name}.csv"
            paths[f"{market}_by_{dimension}_side_summary"] = (
                REPORTS / directory / f"{base_name}_side_summary.csv"
            )
    return paths


# Same rules as the React page's CSV reader: every value stays a string, rows
# whose cells are all empty are skipped, header cells are trimmed, missing
# cells become "" and extra cells are ignored.
JS_WHITESPACE = (
    "\t\n\v\f\r \u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005"
    "\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
)


def read_csv_text(path: Path) -> str:
    text = path.read_bytes().decode("utf-8", errors="replace")
    return text[1:] if text.startswith("\ufeff") else text


def parse_csv_rows(text: str) -> list[dict[str, str]]:
    rows: list[list[str]] = []
    row: list[str] = []
    value: list[str] = []
    quoted = False
    index = 0
    length = len(text)

    while index < length:
        char = text[index]
        following = text[index + 1] if index + 1 < length else ""

        if char == '"' and quoted and following == '"':
            value.append('"')
            index += 2
            continue

        if char == '"':
            quoted = not quoted
            index += 1
            continue

        if char == "," and not quoted:
            row.append("".join(value))
            value = []
            index += 1
            continue

        if char in "\n\r" and not quoted:
            if char == "\r" and following == "\n":
                index += 1
            row.append("".join(value))
            value = []
            if any(cell != "" for cell in row):
                rows.append(row)
            row = []
            index += 1
            continue

        value.append(char)
        index += 1

    if value or row:
        row.append("".join(value))
        if any(cell != "" for cell in row):
            rows.append(row)

    if not rows:
        return []

    headers = [cell.strip(JS_WHITESPACE) for cell in rows[0]]
    records: list[dict[str, str]] = []
    for values in rows[1:]:
        record: dict[str, str] = {}
        for position, header in enumerate(headers):
            record[header] = values[position] if position < len(values) else ""
        records.append(record)
    return records


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, data) -> int:
    document = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": json_safe(data),
    }
    content = json.dumps(
        document,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
        allow_nan=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".tmp")
    staging.write_text(content, encoding="utf-8")
    os.replace(staging, path)
    return len(content.encode("utf-8"))


def reset_legacy_log(started: datetime) -> bool:
    try:
        ERROR_DIR.mkdir(parents=True, exist_ok=True)
        LOG_FILE.write_text(
            "=== 04_nfl_results_dashboard ===\n"
            f"START_TIMESTAMP_UTC: {started.isoformat()}\n",
            encoding="utf-8",
        )
        return True
    except Exception:
        return False


def legacy_log(level: str, message: str) -> None:
    try:
        with LOG_FILE.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{datetime.now(timezone.utc).isoformat()} | "
                f"{level} | {message}\n"
            )
    except Exception:
        pass


def finalize_legacy_log(
    *,
    input_file_count: int,
    input_row_count: int,
    warning_count: int,
    status: str,
) -> None:
    try:
        with LOG_FILE.open("a", encoding="utf-8") as handle:
            handle.write(
                f"INPUT_SUMMARY | files={input_file_count} | "
                f"rows={input_row_count}\n"
            )
            handle.write(f"WARNING_COUNT: {warning_count}\n")
            handle.write(f"STATUS: {status}\n")
    except Exception:
        pass


def run(reporter: PipelineReporter) -> None:
    started = datetime.now(timezone.utc)
    reset_legacy_log(started)

    input_file_count = 0
    input_row_count = 0
    warning_count = 0
    status = "FAILED"

    reporter.add_output(OUTPUT_FILE)

    try:
        load_runtime_dependencies(reporter)

        input_paths = dashboard_input_paths()
        if len(input_paths) != 51 or len(set(input_paths)) != 51:
            fail(
                "dashboard input manifest must contain exactly "
                "51 unique report CSVs"
            )

        frames: dict[Path, pd.DataFrame] = {}
        for path in input_paths:
            reporter.add_input(path)
            frame = read_report(path)
            validate_report_frame(path, frame)
            frames[path] = frame
            input_file_count += 1
            input_row_count += len(frame)
            legacy_log("INFO", f"INPUT | file={path} | rows={len(frame)}")

        validate_cross_report_consistency(frames)

        dataset_paths = react_dataset_paths()
        validated_paths = set(input_paths)
        unvalidated = [str(path) for path in dataset_paths.values() if path not in validated_paths]
        if unvalidated:
            fail(f"dashboard datasets outside the validated inputs: {unvalidated}")
        data = {
            name: parse_csv_rows(read_csv_text(path))
            for name, path in dataset_paths.items()
        }

        reporter.set_rows(
            rows_in=input_row_count,
            rows_out=1,
        )
        reporter.update_details({
            "expected_input_files": 51,
            "validated_input_files": input_file_count,
            "validated_input_rows": input_row_count,
            "cross_report_reconciliation": True,
            "payload_validation": True,
            "dashboard_datasets": len(data),
            "publication_completed": False,
            "post_publish_validation": False,
        })

        write_json(OUTPUT_FILE, data)
        published = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
        if published.get("data") != data:
            fail("published NFL dashboard data reread mismatch")
        reporter.update_details({
            "publication_completed": True,
            "post_publish_validation": True,
        })

        output_bytes = OUTPUT_FILE.stat().st_size
        reporter.set_detail("output_bytes", output_bytes)
        legacy_log("INFO", f"OUTPUT | file={OUTPUT_FILE} | bytes={output_bytes}")

        status = "SUCCESS"
        print(f"NFL dashboard data complete. output={OUTPUT_FILE}")
    except Exception as exc:
        legacy_log(
            "ERROR",
            f"Unhandled exception: {type(exc).__name__}: {exc}",
        )
        try:
            with LOG_FILE.open("a", encoding="utf-8") as handle:
                handle.write(traceback.format_exc())
        except Exception:
            pass
        raise
    finally:
        finalize_legacy_log(
            input_file_count=input_file_count,
            input_row_count=input_row_count,
            warning_count=warning_count,
            status=status,
        )


def main() -> int:
    try:
        with PipelineReporter(
            script=SCRIPT_PATH,
            stage="04_final_results",
            report_root=REPORT_ROOT,
            pipeline="NFL",
            league="NFL",
            extra_context={"component": "graded-bets dashboard"},
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
