#!/usr/bin/env python3
"""Build a self-contained NFL graded-bets dashboard from generated report CSVs."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
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
OUTPUT_FILE = REPO_ROOT / "frontend" / "nfl_dashboard.html"
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


def clean_value(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def records(frame: pd.DataFrame) -> list[dict]:
    if frame.empty:
        return []
    return [
        {key: clean_value(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def first_row(frame: pd.DataFrame) -> dict:
    rows = records(frame.head(1))
    return rows[0] if rows else {}


def collect_data(frames: dict[Path, pd.DataFrame]) -> dict:
    data = {
        "headline": first_row(frames[OVERVIEW / "nfl_summary_overall.csv"]),
        "by_market_summary": records(frames[BASE / "nfl_summary_overall.csv"]),
        "overview": {
            "by_market": records(frames[OVERVIEW / "nfl_summary_by_market.csv"]),
            "by_side_group": records(frames[OVERVIEW / "nfl_summary_by_side_group.csv"]),
            "by_week": records(frames[OVERVIEW / "nfl_summary_by_week.csv"]),
            "by_date": records(frames[OVERVIEW / "nfl_summary_by_date.csv"]),
            "by_season_type": records(frames[OVERVIEW / "nfl_summary_by_season_type.csv"]),
            "by_day_night": records(frames[OVERVIEW / "nfl_summary_by_day_night.csv"]),
            "bet_log": records(frames[OVERVIEW / "nfl_bet_log.csv"]),
        },
        "markets": {},
    }

    for market, cfg in MARKETS.items():
        market_dir = REPORTS / cfg["directory"]
        market_data = {"by": {}, "by_side": {}}
        for dimension in cfg["dimensions"]:
            base_name = f"nfl_{cfg['file_key']}_by_{dimension}"
            market_data["by"][dimension] = records(
                frames[market_dir / f"{base_name}.csv"]
            )
            market_data["by_side"][dimension] = records(
                frames[market_dir / f"{base_name}_side_summary.csv"]
            )
        data["markets"][market] = market_data
    return data


def validate_payload(data: dict) -> None:
    if set(data) != {"headline", "by_market_summary", "overview", "markets"}:
        fail("dashboard payload top-level contract failed")
    expected_overview = {
        "by_market", "by_side_group", "by_week", "by_date",
        "by_season_type", "by_day_night", "bet_log",
    }
    if set(data["overview"]) != expected_overview:
        fail("dashboard payload overview contract failed")
    if set(data["markets"]) != set(MARKETS):
        fail("dashboard payload market contract failed")

    for market, cfg in MARKETS.items():
        market_data = data["markets"][market]
        if set(market_data) != {"by", "by_side"}:
            fail(f"dashboard payload {market}: view contract failed")
        expected_dimensions = set(cfg["dimensions"])
        if set(market_data["by"]) != expected_dimensions:
            fail(f"dashboard payload {market}: overall dimension contract failed")
        if set(market_data["by_side"]) != expected_dimensions:
            fail(f"dashboard payload {market}: side dimension contract failed")

    try:
        json.dumps(
            data,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        fail(f"dashboard payload is not strict JSON: {type(exc).__name__}: {exc}")


CSS = r"""
:root{--bg:#0d1117;--panel:#161b22;--panel2:#1f2630;--border:#30363d;--text:#e6edf3;--muted:#8b949e;--good:#3fb950;--bad:#f85149;--accent:#58a6ff;--head:#21262d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:14px}
header{padding:20px 24px;border-bottom:1px solid var(--border);background:#010409;display:flex;align-items:baseline;justify-content:space-between;gap:20px;flex-wrap:wrap}h1{margin:0;font-size:24px}h2{margin:26px 0 12px;font-size:18px}h3{margin:0 0 12px;font-size:15px}.ts{color:var(--muted);font-size:12px}main{max-width:1600px;margin:0 auto;padding:20px 24px 48px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:10px}.kpi{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:13px}.kpi .label{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.05em}.kpi .value{font-size:21px;font-weight:650;margin-top:4px}.good{color:var(--good)!important}.bad{color:var(--bad)!important}
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px}.tab{padding:8px 12px;border:1px solid var(--border);border-radius:6px;background:var(--panel);color:var(--muted);cursor:pointer;user-select:none}.tab.active{background:#1f6feb;color:white;border-color:#1f6feb}.tab-body,.panel{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:12px}
.controls{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-bottom:10px}select{background:var(--panel2);color:var(--text);border:1px solid var(--border);border-radius:5px;padding:6px 8px}label{color:var(--muted)}
.table-wrap{overflow:auto;max-height:620px;border:1px solid var(--border);border-radius:6px}table{border-collapse:collapse;width:100%;min-width:840px;background:var(--panel)}th,td{padding:8px 10px;border-bottom:1px solid var(--border);white-space:nowrap;text-align:left}th{position:sticky;top:0;background:var(--head);z-index:1;color:#c9d1d9;font-size:12px;cursor:pointer}td.num{text-align:right;font-variant-numeric:tabular-nums}tbody tr:hover{background:#1b222c}.empty{color:var(--muted);padding:18px;text-align:center}.section-note{color:var(--muted);font-size:12px;margin-top:-6px;margin-bottom:12px}
@media(max-width:700px){header,main{padding-left:12px;padding-right:12px}.kpis{grid-template-columns:repeat(2,minmax(0,1fr))}}
"""


JS = r"""
function n(v){const x=Number(v);return Number.isFinite(x)?x:null}
function fmtInt(v){const x=n(v);return x===null?'ΓÇö':Math.round(x).toLocaleString()}
function fmtNum(v,d=2){const x=n(v);return x===null?'ΓÇö':x.toFixed(d)}
function fmtPct(v){const x=n(v);return x===null?'ΓÇö':(x*100).toFixed(1)+'%'}
function signedClass(v){const x=n(v);if(x===null||x===0)return'';return x>0?'good':'bad'}
function showTab(root,key){root.querySelectorAll(':scope > .tabs .tab').forEach(t=>t.classList.toggle('active',t.dataset.key===key));root.querySelectorAll(':scope > .tab-body > .tab-panel').forEach(p=>p.style.display=p.dataset.key===key?'block':'none')}
function valueText(v,fmt,decimals){if(v===null||v===undefined||v==='')return'ΓÇö';if(fmt==='int')return fmtInt(v);if(fmt==='pct')return fmtPct(v);if(fmt==='num')return fmtNum(v,decimals==null?2:decimals);return String(v)}
function renderTable(rows,columns,container){
  if(!rows||rows.length===0){container.innerHTML='<div class="empty">No data</div>';return}
  const wrap=document.createElement('div');wrap.className='table-wrap';const table=document.createElement('table');const thead=document.createElement('thead');const hr=document.createElement('tr');
  let sortKey=null,sortAsc=true;
  columns.forEach(c=>{const th=document.createElement('th');th.textContent=c.label;th.onclick=()=>{if(sortKey===c.key)sortAsc=!sortAsc;else{sortKey=c.key;sortAsc=true}draw()};hr.appendChild(th)});thead.appendChild(hr);const tbody=document.createElement('tbody');
  function draw(){tbody.innerHTML='';let data=[...rows];if(sortKey){data.sort((a,b)=>{const av=a[sortKey],bv=b[sortKey],an=n(av),bn=n(bv);let cmp;if(an!==null&&bn!==null)cmp=an-bn;else cmp=String(av??'').localeCompare(String(bv??''),undefined,{numeric:true});return sortAsc?cmp:-cmp})}data.forEach(row=>{const tr=document.createElement('tr');columns.forEach(c=>{const td=document.createElement('td');if(c.fmt)td.classList.add('num');td.textContent=valueText(row[c.key],c.fmt,c.decimals);if(c.color){const cls=signedClass(row[c.key]);if(cls)td.classList.add(cls)}tr.appendChild(td)});tbody.appendChild(tr)})}
  draw();table.appendChild(thead);table.appendChild(tbody);wrap.appendChild(table);container.innerHTML='';container.appendChild(wrap)
}
const METRICS=[
 {key:'variable',label:'Bucket'}, {key:'Win',label:'W',fmt:'int'}, {key:'Loss',label:'L',fmt:'int'}, {key:'Push',label:'P',fmt:'int'}, {key:'Total',label:'Total',fmt:'int'},
 {key:'Win_Pct',label:'Win %',fmt:'pct'}, {key:'units',label:'Units',fmt:'num',decimals:2,color:true}, {key:'ROI_Excluding_Pushes',label:'ROI excl',fmt:'pct',color:true},
 {key:'ROI_Including_Pushes',label:'ROI incl',fmt:'pct',color:true}, {key:'avg_ev',label:'Avg EV',fmt:'pct'}, {key:'avg_odds',label:'Avg odds',fmt:'num',decimals:0}, {key:'avg_model_prob',label:'Avg model',fmt:'pct'}
];
const SIDE_METRICS=[{key:'side_group',label:'Side'},...METRICS];
const SUMMARY_METRICS=[
 {key:'Win',label:'W',fmt:'int'}, {key:'Loss',label:'L',fmt:'int'}, {key:'Push',label:'P',fmt:'int'}, {key:'Total',label:'Total',fmt:'int'}, {key:'Win_Pct',label:'Win %',fmt:'pct'},
 {key:'units',label:'Units',fmt:'num',decimals:2,color:true}, {key:'ROI_Excluding_Pushes',label:'ROI excl',fmt:'pct',color:true}, {key:'ROI_Including_Pushes',label:'ROI incl',fmt:'pct',color:true},
 {key:'avg_ev',label:'Avg EV',fmt:'pct'}, {key:'avg_odds',label:'Avg odds',fmt:'num',decimals:0}, {key:'avg_model_prob',label:'Avg model',fmt:'pct'}
];
function kpi(label,value,fmt,color=false){let cls=color?signedClass(value):'';return '<div class="kpi"><div class="label">'+label+'</div><div class="value '+cls+'">'+valueText(value,fmt,2)+'</div></div>'}
function build(data){
 const h=data.headline||{};document.querySelector('.kpis').innerHTML=[kpi('Bets',h.Total,'int'),kpi('Wins',h.Win,'int'),kpi('Losses',h.Loss,'int'),kpi('Pushes',h.Push,'int'),kpi('Win %',h.Win_Pct,'pct'),kpi('Units',h.units,'num',true),kpi('ROI excl pushes',h.ROI_Excluding_Pushes,'pct',true),kpi('ROI incl pushes',h.ROI_Including_Pushes,'pct',true),kpi('Avg EV',h.avg_ev,'pct'),kpi('Avg odds',h.avg_odds,'num')].join('');
 renderTable(data.by_market_summary||[],[{key:'market_type',label:'Market'},...SUMMARY_METRICS],document.querySelector('.by-market-summary'));
 ['moneyline','spread','total'].forEach(m=>{const panel=document.querySelector('.panel-'+m);const md=(data.markets||{})[m]||{by:{},by_side:{}};const dims=Object.keys(md.by||{});panel.innerHTML='<div class="controls"><label>Dimension <select class="dim">'+dims.map(d=>'<option value="'+d+'">'+d.replaceAll('_',' ')+'</option>').join('')+'</select></label><label>View <select class="view"><option value="overall">Overall</option><option value="side">Split by side</option></select></label></div><div class="target"></div>';const dim=panel.querySelector('.dim'),view=panel.querySelector('.view'),target=panel.querySelector('.target');function refresh(){const side=view.value==='side',key=dim.value;renderTable(side?(md.by_side[key]||[]):(md.by[key]||[]),side?SIDE_METRICS:METRICS,target)}dim.onchange=refresh;view.onchange=refresh;refresh()});
 const marketArea=document.querySelector('.market-area');marketArea.querySelectorAll(':scope > .tabs .tab').forEach(t=>t.onclick=()=>showTab(marketArea,t.dataset.key));
 const ov=data.overview||{};renderTable(ov.by_market||[],[{key:'variable',label:'Market'},...SUMMARY_METRICS],document.querySelector('.ov-market'));renderTable(ov.by_side_group||[],[{key:'variable',label:'Side'},...SUMMARY_METRICS],document.querySelector('.ov-side'));renderTable(ov.by_week||[],[{key:'variable',label:'Week'},...SUMMARY_METRICS,{key:'cumulative_units',label:'Cum units',fmt:'num',decimals:2,color:true}],document.querySelector('.ov-week'));renderTable(ov.by_date||[],[{key:'variable',label:'Date'},...SUMMARY_METRICS,{key:'cumulative_units',label:'Cum units',fmt:'num',decimals:2,color:true}],document.querySelector('.ov-date'));renderTable(ov.by_season_type||[],[{key:'variable',label:'Season type'},...SUMMARY_METRICS],document.querySelector('.ov-season'));renderTable(ov.by_day_night||[],[{key:'variable',label:'Day/Night'},...SUMMARY_METRICS],document.querySelector('.ov-daynight'));
 renderTable(ov.bet_log||[],[{key:'season',label:'Season'},{key:'week',label:'Week'},{key:'game_date',label:'Date'},{key:'away_team',label:'Away'},{key:'home_team',label:'Home'},{key:'market_type',label:'Market'},{key:'bet_side',label:'Side'},{key:'line',label:'Line',fmt:'num',decimals:1},{key:'odds_american',label:'Odds',fmt:'num',decimals:0},{key:'model_prob',label:'Model',fmt:'pct'},{key:'ev',label:'EV',fmt:'pct'},{key:'bet_result',label:'Result'},{key:'bet_units',label:'Units',fmt:'num',decimals:2,color:true}],document.querySelector('.ov-log'));
 const overview=document.querySelector('.overview-area');overview.querySelectorAll(':scope > .tabs .tab').forEach(t=>t.onclick=()=>showTab(overview,t.dataset.key));
}
"""


def build_html(data: dict) -> str:
    built_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NFL Graded Bets Dashboard</title><style>{CSS}</style></head>
<body><header><h1>NFL Graded Bets Dashboard</h1><span class="ts">Built {html.escape(built_at)} UTC</span></header><main>
<div class="kpis"></div>
<h2>By Market</h2><div class="by-market-summary panel"></div>
<h2>Per-Market Drilldown</h2><div class="market-area"><div class="tabs"><div class="tab active" data-key="moneyline">Moneyline</div><div class="tab" data-key="spread">Spread</div><div class="tab" data-key="total">Total</div></div><div class="tab-body"><div class="tab-panel panel-moneyline" data-key="moneyline"></div><div class="tab-panel panel-spread" data-key="spread" style="display:none"></div><div class="tab-panel panel-total" data-key="total" style="display:none"></div></div></div>
<h2>Overview</h2><div class="section-note">Win percentage and ROI excluding pushes use wins + losses as the denominator.</div><div class="overview-area"><div class="tabs"><div class="tab active" data-key="market">Market</div><div class="tab" data-key="side">Side</div><div class="tab" data-key="week">Week</div><div class="tab" data-key="date">Date</div><div class="tab" data-key="season">Season type</div><div class="tab" data-key="daynight">Day/Night</div><div class="tab" data-key="log">Bet log</div></div><div class="tab-body"><div class="tab-panel ov-market" data-key="market"></div><div class="tab-panel ov-side" data-key="side" style="display:none"></div><div class="tab-panel ov-week" data-key="week" style="display:none"></div><div class="tab-panel ov-date" data-key="date" style="display:none"></div><div class="tab-panel ov-season" data-key="season" style="display:none"></div><div class="tab-panel ov-daynight" data-key="daynight" style="display:none"></div><div class="tab-panel ov-log" data-key="log" style="display:none"></div></div></div>
</main><script>{JS}\nconst DATA={payload};document.addEventListener('DOMContentLoaded',()=>build(DATA));</script></body></html>"""


def extract_payload_from_html(page: str) -> dict:
    start_marker = "const DATA="
    end_marker = ";document.addEventListener"
    start = page.find(start_marker)
    if start < 0:
        fail("dashboard HTML missing embedded DATA marker")
    start += len(start_marker)
    end = page.find(end_marker, start)
    if end < 0:
        fail("dashboard HTML missing DATA terminator")
    payload_text = page[start:end]
    try:
        payload = json.loads(payload_text)
    except Exception as exc:
        fail(f"dashboard HTML embedded DATA is invalid JSON: {type(exc).__name__}: {exc}")
    if not isinstance(payload, dict):
        fail("dashboard HTML embedded DATA is not an object")
    return payload


def validate_html_text(page: str, expected_data: dict) -> None:
    if len(page.encode("utf-8")) < 1000:
        fail("dashboard HTML is unexpectedly small")
    required_markers = (
        "<!doctype html>",
        '<html lang="en">',
        "<title>NFL Graded Bets Dashboard</title>",
        "<style>",
        "function build(data)",
        'class="kpis"',
        'class="by-market-summary panel"',
        'class="tab-panel panel-moneyline"',
        'class="tab-panel panel-spread"',
        'class="tab-panel panel-total"',
        'class="tab-panel ov-week"',
        'class="tab-panel ov-log"',
        "const DATA=",
    )
    for marker in required_markers:
        if marker not in page:
            fail(f"dashboard HTML missing required marker: {marker}")

    embedded = extract_payload_from_html(page)
    validate_payload(embedded)
    if embedded != expected_data:
        fail("dashboard HTML embedded payload does not match validated dashboard data")


def validate_html_file(path: Path, expected_data: dict) -> None:
    if not path.exists() or not path.is_file():
        fail(f"dashboard output not found: {path}")
    try:
        page = path.read_text(encoding="utf-8")
    except Exception as exc:
        fail(f"dashboard output read failed: {type(exc).__name__}: {exc}")
    validate_html_text(page, expected_data)


def stage_dashboard(page: str, data: dict) -> Path:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{OUTPUT_FILE.name}.stage.",
        suffix=".html",
        dir=str(OUTPUT_FILE.parent),
    )
    os.close(descriptor)
    stage_path = Path(raw_path)
    try:
        stage_path.write_text(page, encoding="utf-8")
        with stage_path.open("r+b") as handle:
            os.fsync(handle.fileno())
        validate_html_file(stage_path, data)
        return stage_path
    except Exception:
        stage_path.unlink(missing_ok=True)
        raise


def publish_dashboard(
    stage_path: Path,
    data: dict,
    *,
    reporter: PipelineReporter,
) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    backup_root = Path(
        tempfile.mkdtemp(
            prefix=".nfl_dashboard_backup_",
            dir=str(OUTPUT_FILE.parent),
        )
    )
    backup_path = backup_root / OUTPUT_FILE.name
    had_existing = OUTPUT_FILE.exists()
    published = False

    reporter.update_details({
        "publication_mode": "atomic_replace_with_rollback",
        "publication_completed": False,
        "post_publish_validation": False,
        "rollback_performed": False,
    })

    try:
        if had_existing:
            shutil.copy2(OUTPUT_FILE, backup_path)

        os.replace(stage_path, OUTPUT_FILE)
        published = True

        validate_html_file(OUTPUT_FILE, data)

        reporter.update_details({
            "publication_completed": True,
            "post_publish_validation": True,
        })
    except Exception as publish_exc:
        if published:
            try:
                if had_existing and backup_path.exists():
                    descriptor, raw_restore = tempfile.mkstemp(
                        prefix=f".{OUTPUT_FILE.name}.restore.",
                        suffix=".tmp",
                        dir=str(OUTPUT_FILE.parent),
                    )
                    os.close(descriptor)
                    restore_path = Path(raw_restore)
                    try:
                        shutil.copy2(backup_path, restore_path)
                        os.replace(restore_path, OUTPUT_FILE)
                    finally:
                        restore_path.unlink(missing_ok=True)
                else:
                    OUTPUT_FILE.unlink(missing_ok=True)

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
                    "NFL dashboard publication failed and rollback also failed: "
                    f"publication_error={publish_exc}; rollback_error={rollback_exc}"
                ) from rollback_exc
        raise
    finally:
        stage_path.unlink(missing_ok=True)
        try:
            shutil.rmtree(backup_root, ignore_errors=False)
        except Exception as cleanup_exc:
            reporter.warning(
                "Temporary NFL dashboard backup cleanup failed",
                backup_root=str(backup_root),
                error_type=type(cleanup_exc).__name__,
                error=str(cleanup_exc),
            )


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

        data = collect_data(frames)
        validate_payload(data)

        page = build_html(data)
        validate_html_text(page, data)

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
            "staged_roundtrip_verified": False,
            "publication_completed": False,
            "post_publish_validation": False,
            "rollback_performed": False,
        })

        stage_path = stage_dashboard(page, data)
        reporter.set_detail("staged_roundtrip_verified", True)

        publish_dashboard(
            stage_path,
            data,
            reporter=reporter,
        )

        output_bytes = OUTPUT_FILE.stat().st_size
        reporter.set_detail("output_bytes", output_bytes)
        legacy_log("INFO", f"OUTPUT | file={OUTPUT_FILE} | bytes={output_bytes}")

        status = "SUCCESS"
        print(f"NFL dashboard complete. output={OUTPUT_FILE}")
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
