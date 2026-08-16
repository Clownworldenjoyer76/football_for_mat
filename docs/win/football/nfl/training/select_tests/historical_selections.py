#!/usr/bin/env python3
"""
Historical NFL selection-filter tester.

Reads the saved V4 historical probability backtest and an isolated markets.yaml,
recreates moneyline/spread/total candidate metrics from historical closing odds,
applies the configured filters, grades the selected bets, and writes drilldown
reports under training/select_tests/results/.

This script does not modify live NFL configuration or production selection files.

IMPORTANT:
Historical market fields in the V4 backtest are closing market prices/lines.
Results therefore describe performance against those closing markets, not
necessarily the exact odds that would have been available earlier in the week.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
NFL_ROOT = SCRIPT_DIR.parents[1]
REPO_ROOT = NFL_ROOT.parents[3]

DEFAULT_CONFIG_PATH = SCRIPT_DIR / "markets.yaml"

REQUIRED_COLUMNS = [
    "game_id",
    "season",
    "week",
    "gameday",
    "away_team",
    "home_team",
    "actual_margin",
    "actual_total",
    "closing_spread_line",
    "closing_total_line",
    "closing_away_moneyline",
    "closing_home_moneyline",
    "closing_away_spread_odds",
    "closing_home_spread_odds",
    "closing_under_odds",
    "closing_over_odds",
    "final_calibrated_home_win_probability",
    "final_home_cover_probability_at_closing_line",
    "final_over_probability_at_closing_line",
]

DEFAULT_THRESHOLD_VALUES = {
    "min_ev": 0.00,
    "min_edge": 0.00,
    "min_kelly": 0.00,
    "max_kelly": 0.05,
    "min_odds_american": -10000.0,
    "max_odds_american": 10000.0,
    "min_model_prob": 0.00,
    "max_model_prob": 1.00,
}

BAND_FIELDS = {
    "odds_bands": "odds_american",
    "edge_bands": "edge",
    "ev_bands": "ev",
    "kelly_bands": "full_kelly",
    "prob_bands": "model_probability",
    "line_bands": "line",
}

REPORT_SPECS = {
    "ev": ("ev", "by_ev_band.csv"),
    "kelly": ("full_kelly", "by_kelly_band.csv"),
    "probability": ("model_probability", "by_probability_band.csv"),
    "edge": ("edge", "by_edge_band.csv"),
    "odds": ("odds_american", "by_odds_band.csv"),
    "line": ("line", "by_line_band.csv"),
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.casefold() in {"", "nan", "none", "null", "<na>", "nat"}:
        return ""
    return text


def parse_float(value: Any) -> float | None:
    text = clean(value)
    if not text:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_int(value: Any) -> int | None:
    number = parse_float(value)
    if number is None or not float(number).is_integer():
        return None
    return int(number)


def parse_bool(value: Any, *, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)) and value in {0, 1}:
        return bool(value)
    text = clean(value).casefold()
    if text in {"true", "yes", "y", "1", "on"}:
        return True
    if text in {"false", "no", "n", "0", "off"}:
        return False
    fail(f"{key} must be true/false; found {value!r}")


def resolve_repo_path(value: Any, *, key: str) -> Path:
    text = clean(value)
    if not text:
        fail(f"{key} is required")
    path = Path(text)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        fail(f"Missing config: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        fail(f"Config must contain a YAML mapping: {path}")
    return data


def read_history(path: Path) -> pd.DataFrame:
    if not path.is_file():
        fail(f"Missing historical probability file: {path}")
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        fail(f"Historical probability file has no rows: {path}")
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        fail(f"Historical probability file missing columns: {missing}")
    return df


def american_to_decimal(odds: float) -> float:
    if odds == 0:
        fail("American odds cannot be 0")
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)


def american_implied_probability(odds: float) -> float:
    return 1.0 / american_to_decimal(odds)


def no_vig_probabilities(first_odds: float, second_odds: float) -> tuple[float, float]:
    first_raw = american_implied_probability(first_odds)
    second_raw = american_implied_probability(second_odds)
    total = first_raw + second_raw
    if not math.isfinite(total) or total <= 0:
        fail(f"Cannot calculate no-vig probabilities from {first_odds}, {second_odds}")
    return first_raw / total, second_raw / total


def calculate_metrics(
    model_probability: float,
    odds_american: float,
    fair_market_probability: float,
) -> dict[str, float]:
    if not 0 <= model_probability <= 1:
        fail(f"Model probability outside [0,1]: {model_probability}")

    decimal_odds = american_to_decimal(odds_american)
    net_win = decimal_odds - 1.0
    loss_probability = 1.0 - model_probability
    ev = model_probability * net_win - loss_probability
    raw_kelly = (net_win * model_probability - loss_probability) / net_win

    return {
        "fair_market_probability": fair_market_probability,
        "edge": model_probability - fair_market_probability,
        "ev": ev,
        "full_kelly": max(0.0, raw_kelly),
    }


def make_candidate(
    row: pd.Series,
    *,
    market: str,
    side: str,
    line: float | None,
    odds: float,
    probability: float,
    fair_probability: float,
    is_favorite: bool = False,
    is_underdog: bool = False,
) -> dict[str, Any]:
    return {
        "game_id": str(row["game_id"]),
        "season": int(row["season"]),
        "week": int(row["week"]),
        "gameday": clean(row.get("gameday", "")),
        "away_team": clean(row.get("away_team", "")),
        "home_team": clean(row.get("home_team", "")),
        "market": market,
        "side": side,
        "line": line,
        "odds_american": odds,
        "model_probability": probability,
        "is_favorite": is_favorite,
        "is_underdog": is_underdog,
        **calculate_metrics(probability, odds, fair_probability),
    }


def numeric(row: pd.Series, column: str) -> float | None:
    return parse_float(row.get(column))


def candidate_rows(row: pd.Series) -> dict[str, list[dict[str, Any]]]:
    results: dict[str, list[dict[str, Any]]] = {
        "moneyline": [],
        "spread": [],
        "total": [],
    }

    home_ml = numeric(row, "closing_home_moneyline")
    away_ml = numeric(row, "closing_away_moneyline")
    home_win_p = numeric(row, "final_calibrated_home_win_probability")

    if (
        home_ml is not None
        and away_ml is not None
        and home_ml != 0
        and away_ml != 0
        and home_win_p is not None
        and 0 <= home_win_p <= 1
    ):
        home_fair, away_fair = no_vig_probabilities(home_ml, away_ml)
        tie = math.isclose(home_fair, away_fair, abs_tol=1e-12, rel_tol=0)
        results["moneyline"] = [
            make_candidate(
                row,
                market="moneyline",
                side="HOME",
                line=None,
                odds=home_ml,
                probability=home_win_p,
                fair_probability=home_fair,
                is_favorite=(not tie and home_fair > away_fair),
                is_underdog=(not tie and home_fair < away_fair),
            ),
            make_candidate(
                row,
                market="moneyline",
                side="AWAY",
                line=None,
                odds=away_ml,
                probability=1.0 - home_win_p,
                fair_probability=away_fair,
                is_favorite=(not tie and away_fair > home_fair),
                is_underdog=(not tie and away_fair < home_fair),
            ),
        ]

    # Historical closing_spread_line is the AWAY spread.
    away_spread = numeric(row, "closing_spread_line")
    home_spread_odds = numeric(row, "closing_home_spread_odds")
    away_spread_odds = numeric(row, "closing_away_spread_odds")
    home_cover_p = numeric(row, "final_home_cover_probability_at_closing_line")

    if (
        away_spread is not None
        and home_spread_odds is not None
        and away_spread_odds is not None
        and home_spread_odds != 0
        and away_spread_odds != 0
        and home_cover_p is not None
        and 0 <= home_cover_p <= 1
    ):
        home_fair, away_fair = no_vig_probabilities(home_spread_odds, away_spread_odds)
        results["spread"] = [
            make_candidate(
                row,
                market="spread",
                side="HOME",
                line=-away_spread,
                odds=home_spread_odds,
                probability=home_cover_p,
                fair_probability=home_fair,
                is_favorite=away_spread > 0,
                is_underdog=away_spread < 0,
            ),
            make_candidate(
                row,
                market="spread",
                side="AWAY",
                line=away_spread,
                odds=away_spread_odds,
                probability=1.0 - home_cover_p,
                fair_probability=away_fair,
                is_favorite=away_spread < 0,
                is_underdog=away_spread > 0,
            ),
        ]

    total_line = numeric(row, "closing_total_line")
    over_odds = numeric(row, "closing_over_odds")
    under_odds = numeric(row, "closing_under_odds")
    over_p = numeric(row, "final_over_probability_at_closing_line")

    if (
        total_line is not None
        and over_odds is not None
        and under_odds is not None
        and over_odds != 0
        and under_odds != 0
        and over_p is not None
        and 0 <= over_p <= 1
    ):
        over_fair, under_fair = no_vig_probabilities(over_odds, under_odds)
        results["total"] = [
            make_candidate(
                row,
                market="total",
                side="OVER",
                line=total_line,
                odds=over_odds,
                probability=over_p,
                fair_probability=over_fair,
            ),
            make_candidate(
                row,
                market="total",
                side="UNDER",
                line=total_line,
                odds=under_odds,
                probability=1.0 - over_p,
                fair_probability=under_fair,
            ),
        ]

    return results


def parse_bands(value: Any, *, key: str) -> list[tuple[float, float]] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        fail(f"{key} must be a list of [minimum, maximum] ranges")

    bands: list[tuple[float, float]] = []
    for index, band in enumerate(value, start=1):
        if not isinstance(band, (list, tuple)) or len(band) != 2:
            fail(f"{key}[{index}] must contain exactly two values")
        low = parse_float(band[0])
        high = parse_float(band[1])
        if low is None or high is None:
            fail(f"{key}[{index}] contains a non-numeric value")
        if low > high:
            fail(f"{key}[{index}] minimum cannot exceed maximum")
        bands.append((low, high))

    if not bands:
        fail(f"{key} cannot be empty")
    return bands


def in_any_band(value: float, bands: list[tuple[float, float]]) -> bool:
    return any(low <= value <= high for low, high in bands)


def resolve_thresholds(config: dict[str, Any], market_config: dict[str, Any]) -> dict[str, float]:
    defaults = config.get("selection_defaults", {})
    if not isinstance(defaults, dict):
        fail("selection_defaults must be a YAML mapping")

    resolved: dict[str, float] = {}
    for key, fallback in DEFAULT_THRESHOLD_VALUES.items():
        value = market_config.get(key, defaults.get(key, fallback))
        parsed = parse_float(value)
        if parsed is None:
            fail(f"Non-numeric threshold {key}: {value!r}")
        resolved[key] = parsed

    if resolved["min_kelly"] > resolved["max_kelly"]:
        fail("min_kelly cannot exceed max_kelly")
    if resolved["min_model_prob"] > resolved["max_model_prob"]:
        fail("min_model_prob cannot exceed max_model_prob")
    if resolved["min_odds_american"] > resolved["max_odds_american"]:
        fail("min_odds_american cannot exceed max_odds_american")
    return resolved


def thresholds_pass(candidate: dict[str, Any], thresholds: dict[str, float]) -> bool:
    return (
        thresholds["min_model_prob"]
        <= float(candidate["model_probability"])
        <= thresholds["max_model_prob"]
        and thresholds["min_odds_american"]
        <= float(candidate["odds_american"])
        <= thresholds["max_odds_american"]
        and float(candidate["edge"]) >= thresholds["min_edge"]
        and float(candidate["ev"]) >= thresholds["min_ev"]
        and float(candidate["full_kelly"]) >= thresholds["min_kelly"]
    )


def side_config_for(
    market: str,
    market_config: dict[str, Any],
    side: str,
) -> dict[str, Any]:
    key = side.casefold()
    value = market_config.get(key, {})
    if value is None:
        value = {}
    if not isinstance(value, dict):
        fail(f"{market}.{key} must be a YAML mapping")
    return value


def side_bands_pass(
    market: str,
    side: str,
    candidate: dict[str, Any],
    side_config: dict[str, Any],
) -> bool:
    for band_key, candidate_key in BAND_FIELDS.items():
        if band_key not in side_config:
            continue
        bands = parse_bands(
            side_config.get(band_key),
            key=f"{market}.{side.casefold()}.{band_key}",
        )
        if bands is None:
            continue
        value = parse_float(candidate.get(candidate_key))
        if value is None or not in_any_band(value, bands):
            return False
    return True


def parent_restrictions_pass(
    market: str,
    candidate: dict[str, Any],
    market_config: dict[str, Any],
) -> bool:
    if market in {"moneyline", "spread"}:
        home_only = parse_bool(
            market_config.get("home_only", False),
            key=f"{market}.home_only",
        )
        away_only = parse_bool(
            market_config.get("away_only", False),
            key=f"{market}.away_only",
        )
        favorite_only = parse_bool(
            market_config.get("favorite_only", False),
            key=f"{market}.favorite_only",
        )
        underdog_only = parse_bool(
            market_config.get("underdog_only", False),
            key=f"{market}.underdog_only",
        )
        if home_only and away_only:
            fail(f"{market}: home_only and away_only cannot both be true")
        if favorite_only and underdog_only:
            fail(f"{market}: favorite_only and underdog_only cannot both be true")
        if home_only and candidate["side"] != "HOME":
            return False
        if away_only and candidate["side"] != "AWAY":
            return False
        if favorite_only and not candidate["is_favorite"]:
            return False
        if underdog_only and not candidate["is_underdog"]:
            return False

    if market == "spread":
        max_abs = parse_float(market_config.get("max_spread_abs", 100.0))
        if max_abs is None or max_abs < 0:
            fail("spread.max_spread_abs must be non-negative")
        if abs(float(candidate["line"])) > max_abs:
            return False

    if market == "total":
        minimum = parse_float(market_config.get("min_total", 0.0))
        maximum = parse_float(market_config.get("max_total", 100.0))
        if minimum is None or maximum is None or minimum > maximum:
            fail("total min_total/max_total configuration is invalid")
        if not minimum <= float(candidate["line"]) <= maximum:
            return False

    return True


def select_candidates(
    market: str,
    candidates: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    markets = config.get("markets")
    if not isinstance(markets, dict):
        fail("markets.yaml must contain a markets mapping")

    market_config = markets.get(market)
    if not isinstance(market_config, dict):
        fail(f"markets.yaml missing {market} mapping")

    if not parse_bool(
        market_config.get("enabled", True),
        key=f"{market}.enabled",
    ):
        return []

    thresholds = resolve_thresholds(config, market_config)
    qualifying: list[dict[str, Any]] = []

    for candidate in candidates:
        side = str(candidate["side"])
        side_config = side_config_for(market, market_config, side)
        if not parse_bool(
            side_config.get("enabled", True),
            key=f"{market}.{side.casefold()}.enabled",
        ):
            continue
        if not thresholds_pass(candidate, thresholds):
            continue
        if not parent_restrictions_pass(market, candidate, market_config):
            continue
        if not side_bands_pass(market, side, candidate, side_config):
            continue

        chosen = candidate.copy()
        chosen["kelly"] = min(
            float(chosen["full_kelly"]),
            thresholds["max_kelly"],
        )
        qualifying.append(chosen)

    if not qualifying:
        return []

    preference = clean(market_config.get("pick_preference", "best_ev")).casefold()
    if preference not in {"best_ev", "best_prob", "all"}:
        fail(f"{market}.pick_preference must be best_ev, best_prob, or all")

    if preference == "all":
        return qualifying

    if preference == "best_prob":
        qualifying.sort(
            key=lambda x: (
                float(x["model_probability"]),
                float(x["ev"]),
                float(x["edge"]),
                str(x["side"]),
            ),
            reverse=True,
        )
    else:
        qualifying.sort(
            key=lambda x: (
                float(x["ev"]),
                float(x["edge"]),
                float(x["model_probability"]),
                str(x["side"]),
            ),
            reverse=True,
        )

    return [qualifying[0]]


def grade_candidate(candidate: dict[str, Any], source_row: pd.Series) -> tuple[str, float]:
    market = candidate["market"]
    side = candidate["side"]
    odds = float(candidate["odds_american"])

    if market == "moneyline":
        margin = numeric(source_row, "actual_margin")
        if margin is None:
            return "UNGRADED", 0.0
        if math.isclose(margin, 0.0, abs_tol=1e-12):
            result = "PUSH"
        else:
            home_won = margin > 0
            won = (side == "HOME" and home_won) or (side == "AWAY" and not home_won)
            result = "WIN" if won else "LOSS"

    elif market == "spread":
        margin = numeric(source_row, "actual_margin")
        away_line = numeric(source_row, "closing_spread_line")
        if margin is None or away_line is None:
            return "UNGRADED", 0.0
        home_line = -away_line
        home_cover_margin = margin + home_line
        if math.isclose(home_cover_margin, 0.0, abs_tol=1e-12):
            result = "PUSH"
        else:
            home_covered = home_cover_margin > 0
            won = (side == "HOME" and home_covered) or (side == "AWAY" and not home_covered)
            result = "WIN" if won else "LOSS"

    elif market == "total":
        actual_total = numeric(source_row, "actual_total")
        line = numeric(source_row, "closing_total_line")
        if actual_total is None or line is None:
            return "UNGRADED", 0.0
        difference = actual_total - line
        if math.isclose(difference, 0.0, abs_tol=1e-12):
            result = "PUSH"
        else:
            went_over = difference > 0
            won = (side == "OVER" and went_over) or (side == "UNDER" and not went_over)
            result = "WIN" if won else "LOSS"
    else:
        fail(f"Unsupported market for grading: {market}")

    if result == "WIN":
        decimal_odds = american_to_decimal(odds)
        return result, decimal_odds - 1.0
    if result == "LOSS":
        return result, -1.0
    return result, 0.0



def configured_seasons(config: dict[str, Any]) -> list[int]:
    test = config.get("test", {})
    if not isinstance(test, dict):
        fail("test must be a YAML mapping")
    seasons_raw = test.get("seasons", [2022, 2023, 2024, 2025])
    if not isinstance(seasons_raw, list) or not seasons_raw:
        fail("test.seasons must be a non-empty list")
    seasons: list[int] = []
    for value in seasons_raw:
        season = parse_int(value)
        if season is None:
            fail(f"Invalid season in test.seasons: {value!r}")
        seasons.append(season)
    return sorted(set(seasons))

def selected_bets(history: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    test = config.get("test", {})
    if not isinstance(test, dict):
        fail("test must be a YAML mapping")

    seasons = configured_seasons(config)

    working = history.loc[
        pd.to_numeric(history["season"], errors="coerce").isin(seasons)
    ].copy()

    if working.empty:
        fail(f"No historical rows found for seasons {seasons}")

    picks: list[dict[str, Any]] = []

    for _, row in working.iterrows():
        all_candidates = candidate_rows(row)

        for market in ("moneyline", "spread", "total"):
            chosen = select_candidates(market, all_candidates[market], config)
            for candidate in chosen:
                grade, profit = grade_candidate(candidate, row)
                selected = candidate.copy()
                selected["grade"] = grade
                selected["profit_units"] = profit
                picks.append(selected)

    columns = [
        "game_id",
        "season",
        "week",
        "gameday",
        "away_team",
        "home_team",
        "market",
        "side",
        "line",
        "odds_american",
        "model_probability",
        "fair_market_probability",
        "edge",
        "ev",
        "full_kelly",
        "kelly",
        "is_favorite",
        "is_underdog",
        "grade",
        "profit_units",
    ]

    return pd.DataFrame(picks, columns=columns)


def metric_summary(frame: pd.DataFrame) -> dict[str, Any]:
    graded = frame.loc[frame["grade"].isin(["WIN", "LOSS", "PUSH"])].copy()
    wins = int((graded["grade"] == "WIN").sum())
    losses = int((graded["grade"] == "LOSS").sum())
    pushes = int((graded["grade"] == "PUSH").sum())
    decisions = wins + losses
    picks = len(graded)
    profit = float(graded["profit_units"].sum()) if picks else 0.0

    return {
        "picks": picks,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_pct": (wins / decisions * 100.0) if decisions else np.nan,
        "profit_units": profit,
        "roi_pct": (profit / picks * 100.0) if picks else np.nan,
        "avg_odds": float(graded["odds_american"].mean()) if picks else np.nan,
        "avg_model_probability": float(graded["model_probability"].mean()) if picks else np.nan,
        "avg_edge": float(graded["edge"].mean()) if picks else np.nan,
        "avg_ev": float(graded["ev"].mean()) if picks else np.nan,
        "avg_full_kelly": float(graded["full_kelly"].mean()) if picks else np.nan,
    }


def side_labels_for_market(market: str) -> list[str]:
    if market in {"moneyline", "spread"}:
        return ["ALL", "HOME", "AWAY"]
    return ["ALL", "OVER", "UNDER"]


def summary_by_season(picks: pd.DataFrame, seasons: list[int] | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    if seasons is None:
        seasons = sorted(int(x) for x in picks["season"].dropna().unique())
    for season in seasons:
        season_frame = picks.loc[picks["season"] == season]
        for market in ("moneyline", "spread", "total"):
            market_frame = season_frame.loc[season_frame["market"] == market]
            for side in side_labels_for_market(market):
                subset = market_frame if side == "ALL" else market_frame.loc[market_frame["side"] == side]
                rows.append({"season": season, "market": market, "side": side, **metric_summary(subset)})

    return pd.DataFrame(rows)


def summary_by_market(picks: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for market in ("moneyline", "spread", "total"):
        market_frame = picks.loc[picks["market"] == market]
        for side in side_labels_for_market(market):
            subset = market_frame if side == "ALL" else market_frame.loc[market_frame["side"] == side]
            rows.append({"season": "ALL", "market": market, "side": side, **metric_summary(subset)})

    return pd.DataFrame(rows)


def report_band_config(config: dict[str, Any], report_key: str) -> list[tuple[float, float]]:
    report_bands = config.get("report_bands")
    if not isinstance(report_bands, dict):
        fail("markets.yaml must contain report_bands")

    bands = parse_bands(report_bands.get(report_key), key=f"report_bands.{report_key}")
    if bands is None:
        fail(f"Missing report_bands.{report_key}")
    return bands


def band_label(low: float, high: float) -> str:
    return f"{low:g} to {high:g}"


def drilldown_report(
    picks: pd.DataFrame,
    *,
    config: dict[str, Any],
    report_key: str,
    value_column: str,
) -> pd.DataFrame:
    bands = report_band_config(config, report_key)
    rows: list[dict[str, Any]] = []

    season_values: list[Any] = sorted(int(x) for x in picks["season"].dropna().unique())
    season_values.append("ALL")

    for season in season_values:
        season_frame = picks if season == "ALL" else picks.loc[picks["season"] == season]

        for market in ("moneyline", "spread", "total"):
            market_frame = season_frame.loc[season_frame["market"] == market]

            if value_column == "line" and market == "moneyline":
                continue

            for side in side_labels_for_market(market):
                side_frame = market_frame if side == "ALL" else market_frame.loc[market_frame["side"] == side]
                values = pd.to_numeric(side_frame[value_column], errors="coerce")
                assigned = pd.Series(False, index=side_frame.index)

                for low, high in bands:
                    mask = values.notna() & values.ge(low) & values.le(high) & ~assigned
                    subset = side_frame.loc[mask]
                    assigned = assigned | mask
                    if subset.empty:
                        continue

                    rows.append(
                        {
                            "season": season,
                            "market": market,
                            "side": side,
                            "band": band_label(low, high),
                            "band_min": low,
                            "band_max": high,
                            **metric_summary(subset),
                        }
                    )

    columns = [
        "season",
        "market",
        "side",
        "band",
        "band_min",
        "band_max",
        "picks",
        "wins",
        "losses",
        "pushes",
        "win_pct",
        "profit_units",
        "roi_pct",
        "avg_odds",
        "avg_model_probability",
        "avg_edge",
        "avg_ev",
        "avg_full_kelly",
    ]
    return pd.DataFrame(rows, columns=columns)


def write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temp, index=False)
    os.replace(temp, path)


def write_results(
    picks: pd.DataFrame,
    config: dict[str, Any],
    config_path: Path,
    results_dir: Path,
    input_path: Path,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    owned_names = {
        "graded_picks.csv",
        "summary_by_season.csv",
        "summary_by_market.csv",
        "by_ev_band.csv",
        "by_kelly_band.csv",
        "by_probability_band.csv",
        "by_edge_band.csv",
        "by_odds_band.csv",
        "by_line_band.csv",
        "grand_total_by_ev_band.csv",
        "grand_total_by_kelly_band.csv",
        "grand_total_by_probability_band.csv",
        "grand_total_by_edge_band.csv",
        "grand_total_by_odds_band.csv",
        "grand_total_by_line_band.csv",
        "markets_used.yaml",
        "run_manifest.json",
    }
    for name in owned_names:
        target = results_dir / name
        if target.exists():
            target.unlink()

    write_csv_atomic(picks, results_dir / "graded_picks.csv")
    seasons = configured_seasons(config)
    write_csv_atomic(summary_by_season(picks, seasons), results_dir / "summary_by_season.csv")
    write_csv_atomic(summary_by_market(picks), results_dir / "summary_by_market.csv")

    for report_key, (column, filename) in REPORT_SPECS.items():
        report = drilldown_report(
            picks,
            config=config,
            report_key=report_key,
            value_column=column,
        )
        write_csv_atomic(report, results_dir / filename)

        grand_total = report.loc[report["season"].astype(str) == "ALL"].copy()
        grand_total_filename = f"grand_total_{filename}"
        write_csv_atomic(grand_total, results_dir / grand_total_filename)

    shutil.copyfile(config_path, results_dir / "markets_used.yaml")

    manifest = {
        "input_file": str(input_path.relative_to(REPO_ROOT)),
        "config_file": str(config_path.relative_to(REPO_ROOT)),
        "results_directory": str(results_dir.relative_to(REPO_ROOT)),
        "selected_bets": int(len(picks)),
        "note": (
            "Historical odds/lines in the V4 probability file are closing markets. "
            "These results are not a claim about earlier-in-week executable prices."
        ),
    }
    with (results_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


def print_summary(picks: pd.DataFrame, config: dict[str, Any]) -> None:
    print(f"Selected historical bets: {len(picks)}")
    summary = summary_by_season(picks, configured_seasons(config))
    display = summary.loc[
        summary["side"] == "ALL",
        [
            "season",
            "market",
            "picks",
            "wins",
            "losses",
            "pushes",
            "win_pct",
            "profit_units",
            "roi_pct",
        ],
    ]
    if display.empty:
        print("No selections passed the configured filters.")
        return
    print(display.to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Isolated historical filter-test markets.yaml",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional override for historical probability CSV",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Optional override for results directory",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = read_yaml(config_path)

    test = config.get("test")
    if not isinstance(test, dict):
        fail("markets.yaml must contain a test mapping")

    input_path = (
        args.input.resolve()
        if args.input is not None
        else resolve_repo_path(test.get("input_csv"), key="test.input_csv")
    )

    results_dir = (
        args.results_dir.resolve()
        if args.results_dir is not None
        else resolve_repo_path(test.get("results_dir"), key="test.results_dir")
    )

    history = read_history(input_path)
    picks = selected_bets(history, config)
    write_results(picks, config, config_path, results_dir, input_path)

    print_summary(picks, config)
    print(f"Results written to: {results_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
