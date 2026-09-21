#!/usr/bin/env python3
"""
Step 15 NFL candidate enrichment engine.

READS:
  docs/win/football/nfl/config/settings.yaml
  docs/win/football/nfl/01_merge/week_{week}_NFL_enriched.csv
  docs/win/football/nfl/00_intake/schedule/weekly/
      week_{week}_NFL_weekly_schedule.csv

WRITES:
  docs/win/football/nfl/02_select/week_{week}_NFL_selected.csv

This step does NOT apply betting filters or choose a bet.

It preserves the existing enriched input columns and appends raw candidate
metrics for every available side:

  moneyline: HOME / AWAY
  spread:    HOME / AWAY
  total:     OVER / UNDER

The existing final selection columns are retained for downstream compatibility,
but this step leaves them unselected and marks them as DEFERRED_TO_FILTER.
A later filtering step can use the raw candidate columns to apply odds, edge,
EV, Kelly, probability, side, line, or other betting rules.

The *_implied_probability candidate columns contain the no-vig fair market
probability.

EV and full Kelly use the actual offered sportsbook odds.
Kelly is full Kelly capped at settings.yaml selection_defaults.max_kelly.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import sys
import tempfile
import uuid
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

DEFAULT_SETTINGS_PATH = NFL_ROOT / "config/settings.yaml"

def load_runtime_dependencies(
    reporter: PipelineReporter,
) -> None:
    global np, pd, yaml

    try:
        import numpy as np_module
        import pandas as pd_module
        import yaml as yaml_module
    except Exception:
        reporter.set_detail(
            "dependency_imports_ok",
            False,
        )
        raise

    np = np_module
    pd = pd_module
    yaml = yaml_module

    reporter.set_detail(
        "dependency_imports_ok",
        True,
    )

PREDICTION_COLUMNS = [
    "predicted_margin",
    "predicted_total",
    "predicted_home_score",
    "predicted_away_score",
    "home_win_probability",
    "away_win_probability",
    "home_cover_probability",
    "away_cover_probability",
    "over_probability",
    "under_probability",
]

SELECTION_COLUMNS = [
    "ml_selected",
    "ml_selection",
    "ml_selection_reason",
    "ml_odds_american",
    "ml_model_probability",
    "ml_implied_probability",
    "ml_edge",
    "ml_ev",
    "ml_full_kelly",
    "ml_kelly",
    "spread_selected",
    "spread_selection",
    "spread_selection_reason",
    "spread_line",
    "spread_odds_american",
    "spread_model_probability",
    "spread_implied_probability",
    "spread_edge",
    "spread_ev",
    "spread_full_kelly",
    "spread_kelly",
    "total_selected",
    "total_selection",
    "total_selection_reason",
    "total_line",
    "total_odds_american",
    "total_model_probability",
    "total_implied_probability",
    "total_edge",
    "total_ev",
    "total_full_kelly",
    "total_kelly",
]

CANDIDATE_COLUMNS = [
    "ml_home_available",
    "ml_home_odds_american",
    "ml_home_model_probability",
    "ml_home_implied_probability",
    "ml_home_edge",
    "ml_home_ev",
    "ml_home_full_kelly",
    "ml_home_kelly",
    "ml_away_available",
    "ml_away_odds_american",
    "ml_away_model_probability",
    "ml_away_implied_probability",
    "ml_away_edge",
    "ml_away_ev",
    "ml_away_full_kelly",
    "ml_away_kelly",
    "spread_home_available",
    "spread_home_line",
    "spread_home_odds_american",
    "spread_home_model_probability",
    "spread_home_implied_probability",
    "spread_home_edge",
    "spread_home_ev",
    "spread_home_full_kelly",
    "spread_home_kelly",
    "spread_away_available",
    "spread_away_line",
    "spread_away_odds_american",
    "spread_away_model_probability",
    "spread_away_implied_probability",
    "spread_away_edge",
    "spread_away_ev",
    "spread_away_full_kelly",
    "spread_away_kelly",
    "total_over_available",
    "total_over_line",
    "total_over_odds_american",
    "total_over_model_probability",
    "total_over_implied_probability",
    "total_over_edge",
    "total_over_ev",
    "total_over_full_kelly",
    "total_over_kelly",
    "total_under_available",
    "total_under_line",
    "total_under_odds_american",
    "total_under_model_probability",
    "total_under_implied_probability",
    "total_under_edge",
    "total_under_ev",
    "total_under_full_kelly",
    "total_under_kelly",
]

SEASON_TYPE_ALIASES = {
    "reg": "reg",
    "regular": "reg",
    "regularseason": "reg",
    "pre": "pre",
    "preseason": "pre",
    "post": "post",
    "postseason": "post",
    "playoff": "post",
    "playoffs": "post",
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def clean(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip()

    if text.casefold() in {
        "",
        "nan",
        "none",
        "null",
        "<na>",
        "nat",
    }:
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

    if (
        isinstance(value, (int, np.integer))
        and value in {0, 1}
    ):
        return bool(value)

    text = clean(value).casefold()

    if text in {
        "true",
        "yes",
        "y",
        "1",
        "on",
    }:
        return True

    if text in {
        "false",
        "no",
        "n",
        "0",
        "off",
    }:
        return False

    fail(
        f"{key} must be true/false; "
        f"found {value!r}"
    )


def normalize_game_id(value: Any) -> str:
    return re.sub(
        r"\.0$",
        "",
        clean(value),
    )


def normalize_season_type(value: Any) -> str:
    text = re.sub(
        r"[\s_-]+",
        "",
        clean(value).casefold(),
    )

    return SEASON_TYPE_ALIASES.get(
        text,
        text,
    )


def normalize_bookmaker(value: Any) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "",
        clean(value).casefold(),
    )


def read_yaml(
    path: Path,
    label: str,
) -> dict[str, Any]:
    if not path.is_file():
        fail(
            f"Missing {label}: "
            f"{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        fail(
            f"{label} must contain "
            f"a YAML mapping: {path}"
        )

    return data



def validate_csv_header(
    path: Path,
    label: str,
) -> None:
    try:
        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
    except UnicodeDecodeError as exc:
        fail(
            f"{label} is not valid UTF-8 CSV: "
            f"{path}: {exc}"
        )

    if not header:
        fail(
            f"{label} has no CSV header: "
            f"{path}"
        )

    normalized = [
        clean(column)
        for column in header
    ]

    if any(
        not column
        for column in normalized
    ):
        fail(
            f"{label} contains blank "
            f"column names: {path}"
        )

    duplicates = sorted(
        {
            column
            for column in normalized
            if normalized.count(column) > 1
        }
    )

    if duplicates:
        fail(
            f"{label} contains duplicate "
            f"column names: {duplicates}"
        )


def read_csv(
    path: Path,
    label: str,
    *,
    optional: bool = False,
) -> pd.DataFrame | None:
    if not path.is_file():
        if optional:
            return None

        fail(
            f"Missing {label}: "
            f"{path}"
        )

    validate_csv_header(
        path,
        label,
    )

    df = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        encoding="utf-8-sig",
        low_memory=False,
    )

    if df.empty and not optional:
        fail(
            f"{label} contains no "
            f"data rows: {path}"
        )

    return df


def require_columns(
    df: pd.DataFrame,
    columns: list[str],
    label: str,
) -> None:
    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        fail(
            f"{label} missing required "
            f"columns: {missing}"
        )


def validate_unique_game_ids(
    df: pd.DataFrame,
    label: str,
) -> None:
    ids = df[
        "game_id"
    ].map(normalize_game_id)

    if ids.eq("").any():
        fail(
            f"{label} contains "
            "blank game_id values"
        )

    if ids.duplicated().any():
        examples = ids[
            ids.duplicated(False)
        ].head(10).tolist()

        fail(
            f"{label} contains duplicate "
            f"game_id values: {examples}"
        )

    df["game_id"] = ids


def american_to_decimal(
    odds: float,
) -> float:
    if odds == 0:
        fail(
            "American odds cannot be 0"
        )

    if odds > 0:
        return 1.0 + odds / 100.0

    return 1.0 + 100.0 / abs(odds)


def american_implied_probability(
    odds: float,
) -> float:
    return (
        1.0
        / american_to_decimal(odds)
    )


def no_vig_probabilities(
    first_odds: float,
    second_odds: float,
) -> tuple[float, float]:
    first_raw = (
        american_implied_probability(
            first_odds
        )
    )

    second_raw = (
        american_implied_probability(
            second_odds
        )
    )

    total_raw = (
        first_raw
        + second_raw
    )

    if (
        not math.isfinite(total_raw)
        or total_raw <= 0
    ):
        fail(
            "Unable to calculate no-vig "
            "probabilities from odds "
            f"{first_odds!r}, {second_odds!r}"
        )

    first_fair = (
        first_raw
        / total_raw
    )

    second_fair = (
        second_raw
        / total_raw
    )

    if not 0.0 <= first_fair <= 1.0:
        fail(
            "Invalid first no-vig "
            f"probability: {first_fair}"
        )

    if not 0.0 <= second_fair <= 1.0:
        fail(
            "Invalid second no-vig "
            f"probability: {second_fair}"
        )

    return (
        first_fair,
        second_fair,
    )


def calculate_metrics(
    model_probability: float,
    odds_american: float,
    fair_market_probability: float,
) -> dict[str, float]:
    if not (
        0.0
        <= model_probability
        <= 1.0
    ):
        fail(
            "Model probability outside "
            f"[0,1]: {model_probability}"
        )

    if not (
        0.0
        <= fair_market_probability
        <= 1.0
    ):
        fail(
            "Fair market probability "
            "outside [0,1]: "
            f"{fair_market_probability}"
        )

    decimal_odds = (
        american_to_decimal(
            odds_american
        )
    )

    implied_probability = (
        fair_market_probability
    )

    edge = (
        model_probability
        - fair_market_probability
    )

    net_win = (
        decimal_odds
        - 1.0
    )

    loss_probability = (
        1.0
        - model_probability
    )

    ev = (
        model_probability
        * net_win
        - loss_probability
    )

    raw_kelly = (
        (
            net_win
            * model_probability
            - loss_probability
        )
        / net_win
    )

    full_kelly = max(
        0.0,
        raw_kelly,
    )

    return {
        "implied_probability": implied_probability,
        "edge": edge,
        "ev": ev,
        "full_kelly": full_kelly,
    }


def numeric_probability(
    row: pd.Series,
    column: str,
) -> float:
    value = parse_float(
        row[column]
    )

    if (
        value is None
        or not 0.0 <= value <= 1.0
    ):
        fail(
            f"game_id={row['game_id']}: "
            f"{column} must be a finite "
            "probability in [0,1]; "
            f"found {row[column]!r}"
        )

    return value


def odds_value(
    row: pd.Series,
    column: str,
) -> float | None:
    value = parse_float(
        row.get(
            column,
            "",
        )
    )

    if (
        value is None
        or value == 0
    ):
        return None

    return value


def make_candidate(
    selection: str,
    model_probability: float,
    odds_american: float,
    fair_market_probability: float,
    *,
    line: float | None = None,
    is_favorite: bool = False,
    is_underdog: bool = False,
) -> dict[str, Any]:
    return {
        "selection": selection,
        "line": line,
        "odds_american": (
            odds_american
        ),
        "model_probability": (
            model_probability
        ),
        "is_favorite": (
            is_favorite
        ),
        "is_underdog": (
            is_underdog
        ),
        **calculate_metrics(
            model_probability,
            odds_american,
            fair_market_probability,
        ),
    }


def deferred_market(
    prefix: str,
    reason: str,
    *,
    line: float | None = None,
) -> dict[str, Any]:
    output = {
        f"{prefix}_selected": 0,
        f"{prefix}_selection": "",
        f"{prefix}_selection_reason": reason,
        f"{prefix}_odds_american": np.nan,
        f"{prefix}_model_probability": np.nan,
        f"{prefix}_implied_probability": np.nan,
        f"{prefix}_edge": np.nan,
        f"{prefix}_ev": np.nan,
        f"{prefix}_full_kelly": np.nan,
        f"{prefix}_kelly": np.nan,
    }

    if prefix in {"spread", "total"}:
        output[f"{prefix}_line"] = (
            np.nan if line is None else line
        )

    return output


def blank_candidate(
    prefix: str,
    *,
    line: float | None = None,
) -> dict[str, Any]:
    output = {
        f"{prefix}_available": 0,
        f"{prefix}_odds_american": np.nan,
        f"{prefix}_model_probability": np.nan,
        f"{prefix}_implied_probability": np.nan,
        f"{prefix}_edge": np.nan,
        f"{prefix}_ev": np.nan,
        f"{prefix}_full_kelly": np.nan,
        f"{prefix}_kelly": np.nan,
    }

    if prefix.startswith("spread_") or prefix.startswith("total_"):
        output[f"{prefix}_line"] = (
            np.nan if line is None else line
        )

    return output


def candidate_columns(
    prefix: str,
    candidate: dict[str, Any],
    *,
    include_line: bool,
) -> dict[str, Any]:
    output = {
        f"{prefix}_available": 1,
        f"{prefix}_odds_american": candidate["odds_american"],
        f"{prefix}_model_probability": candidate["model_probability"],
        f"{prefix}_implied_probability": candidate["implied_probability"],
        f"{prefix}_edge": candidate["edge"],
        f"{prefix}_ev": candidate["ev"],
        f"{prefix}_full_kelly": candidate["full_kelly"],
        f"{prefix}_kelly": candidate["full_kelly"],
    }

    if include_line:
        output[f"{prefix}_line"] = candidate["line"]

    return output


def empty_candidate_set(
    reason: str,
) -> dict[str, Any]:
    return {
        **deferred_market("ml", reason),
        **deferred_market("spread", reason),
        **deferred_market("total", reason),
        **blank_candidate("ml_home"),
        **blank_candidate("ml_away"),
        **blank_candidate("spread_home"),
        **blank_candidate("spread_away"),
        **blank_candidate("total_over"),
        **blank_candidate("total_under"),
    }


def evaluate_moneyline(
    row: pd.Series,
) -> dict[str, Any]:
    home_odds = odds_value(
        row,
        "sched_home_moneyline_american",
    )
    away_odds = odds_value(
        row,
        "sched_away_moneyline_american",
    )

    if home_odds is None or away_odds is None:
        return {
            **deferred_market(
                "ml",
                "CURRENT_LINE_MISSING",
            ),
            **blank_candidate("ml_home"),
            **blank_candidate("ml_away"),
        }

    home_probability = numeric_probability(
        row,
        "home_win_probability",
    )
    away_probability = numeric_probability(
        row,
        "away_win_probability",
    )

    home_fair, away_fair = no_vig_probabilities(
        home_odds,
        away_odds,
    )

    home_candidate = make_candidate(
        "HOME",
        home_probability,
        home_odds,
        home_fair,
    )
    away_candidate = make_candidate(
        "AWAY",
        away_probability,
        away_odds,
        away_fair,
    )

    return {
        **deferred_market(
            "ml",
            "DEFERRED_TO_FILTER",
        ),
        **candidate_columns(
            "ml_home",
            home_candidate,
            include_line=False,
        ),
        **candidate_columns(
            "ml_away",
            away_candidate,
            include_line=False,
        ),
    }


def evaluate_spread(
    row: pd.Series,
) -> dict[str, Any]:
    home_line = parse_float(
        row.get("sched_home_spread", "")
    )
    away_line = parse_float(
        row.get("sched_away_spread", "")
    )
    home_odds = odds_value(
        row,
        "sched_home_spread_american",
    )
    away_odds = odds_value(
        row,
        "sched_away_spread_american",
    )

    if any(
        value is None
        for value in [
            home_line,
            away_line,
            home_odds,
            away_odds,
        ]
    ):
        return {
            **deferred_market(
                "spread",
                "CURRENT_LINE_MISSING",
            ),
            **blank_candidate(
                "spread_home",
                line=home_line,
            ),
            **blank_candidate(
                "spread_away",
                line=away_line,
            ),
        }

    home_fair, away_fair = no_vig_probabilities(
        home_odds,
        away_odds,
    )

    home_candidate = make_candidate(
        "HOME",
        numeric_probability(
            row,
            "home_cover_probability",
        ),
        home_odds,
        home_fair,
        line=home_line,
        is_favorite=home_line < 0,
        is_underdog=home_line > 0,
    )
    away_candidate = make_candidate(
        "AWAY",
        numeric_probability(
            row,
            "away_cover_probability",
        ),
        away_odds,
        away_fair,
        line=away_line,
        is_favorite=away_line < 0,
        is_underdog=away_line > 0,
    )

    return {
        **deferred_market(
            "spread",
            "DEFERRED_TO_FILTER",
        ),
        **candidate_columns(
            "spread_home",
            home_candidate,
            include_line=True,
        ),
        **candidate_columns(
            "spread_away",
            away_candidate,
            include_line=True,
        ),
    }











def evaluate_total(
    row: pd.Series,
) -> dict[str, Any]:
    total_line = parse_float(
        row.get("sched_total", "")
    )
    over_odds = odds_value(
        row,
        "sched_over_american",
    )
    under_odds = odds_value(
        row,
        "sched_under_american",
    )

    if (
        total_line is None
        or over_odds is None
        or under_odds is None
    ):
        return {
            **deferred_market(
                "total",
                "CURRENT_LINE_MISSING",
                line=total_line,
            ),
            **blank_candidate(
                "total_over",
                line=total_line,
            ),
            **blank_candidate(
                "total_under",
                line=total_line,
            ),
        }

    over_fair, under_fair = no_vig_probabilities(
        over_odds,
        under_odds,
    )

    over_candidate = make_candidate(
        "OVER",
        numeric_probability(
            row,
            "over_probability",
        ),
        over_odds,
        over_fair,
        line=total_line,
    )
    under_candidate = make_candidate(
        "UNDER",
        numeric_probability(
            row,
            "under_probability",
        ),
        under_odds,
        under_fair,
        line=total_line,
    )

    return {
        **deferred_market(
            "total",
            "DEFERRED_TO_FILTER",
            line=total_line,
        ),
        **candidate_columns(
            "total_over",
            over_candidate,
            include_line=True,
        ),
        **candidate_columns(
            "total_under",
            under_candidate,
            include_line=True,
        ),
    }


def validate_probability_pairs(
    df: pd.DataFrame,
) -> None:
    pairs = [
        (
            "home_win_probability",
            "away_win_probability",
            "moneyline",
        ),
        (
            "home_cover_probability",
            "away_cover_probability",
            "spread",
        ),
        (
            "over_probability",
            "under_probability",
            "total",
        ),
    ]

    for (
        first,
        second,
        label,
    ) in pairs:
        a = pd.to_numeric(
            df[first],
            errors="coerce",
        )

        b = pd.to_numeric(
            df[second],
            errors="coerce",
        )

        if (
            a.isna().any()
            or b.isna().any()
        ):
            fail(
                f"{label} probability "
                "columns contain "
                "blank/non-numeric values"
            )

        if (
            (
                (a < 0)
                | (a > 1)
                | (b < 0)
                | (b > 1)
            ).any()
        ):
            fail(
                f"{label} probability "
                "outside [0,1]"
            )

        if not np.allclose(
            a.to_numpy(
                dtype=float
            )
            + b.to_numpy(
                dtype=float
            ),
            1.0,
            rtol=0,
            atol=1e-9,
        ):
            fail(
                f"{label} complementary "
                "probabilities do not "
                "sum to 1"
            )


def validate_settings(
    settings: dict[str, Any],
    season_override: int | None,
    week_override: int | None,
) -> tuple[
    int,
    int,
    str,
    str,
]:
    season = (
        season_override
        if season_override
        is not None
        else parse_int(
            settings.get(
                "season"
            )
        )
    )

    week = (
        week_override
        if week_override
        is not None
        else parse_int(
            settings.get(
                "week"
            )
        )
    )

    if (
        season is None
        or season < 1900
    ):
        fail(
            f"Invalid season: "
            f"{settings.get('season')!r}"
        )

    if (
        week is None
        or week <= 0
    ):
        fail(
            f"Invalid week: "
            f"{settings.get('week')!r}"
        )

    season_type = (
        normalize_season_type(
            settings.get(
                "season_type",
                "reg",
            )
        )
    )

    if season_type not in {
        "reg",
        "pre",
        "post",
    }:
        fail(
            "Unsupported season_type: "
            f"{settings.get('season_type')!r}"
        )

    sportsbook = clean(
        settings.get(
            "sportsbook"
        )
    )

    if not sportsbook:
        fail(
            "settings.yaml sportsbook "
            "is required"
        )

    odds_format = clean(
        settings.get(
            "odds_format",
            "american",
        )
    ).casefold()

    if odds_format != "american":
        fail(
            "selections.py requires "
            "odds_format: american"
        )

    return (
        season,
        week,
        season_type,
        sportsbook,
    )



def validate_projection_outputs(
    df: pd.DataFrame,
    label: str,
) -> None:
    score_columns = [
        "predicted_margin",
        "predicted_total",
        "predicted_home_score",
        "predicted_away_score",
    ]

    numeric: dict[str, Any] = {}

    for column in score_columns:
        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        array = values.to_numpy(
            dtype=float
        )

        if not np.isfinite(array).all():
            fail(
                f"{label}: {column} "
                "contains blank/non-finite values"
            )

        numeric[column] = array

    if not np.allclose(
        numeric["predicted_home_score"]
        + numeric["predicted_away_score"],
        numeric["predicted_total"],
        rtol=0.0,
        atol=1e-9,
    ):
        fail(
            f"{label}: predicted home/away "
            "scores do not reconcile to "
            "predicted_total"
        )

    if not np.allclose(
        numeric["predicted_home_score"]
        - numeric["predicted_away_score"],
        numeric["predicted_margin"],
        rtol=0.0,
        atol=1e-9,
    ):
        fail(
            f"{label}: predicted home/away "
            "scores do not reconcile to "
            "predicted_margin"
        )


def validate_combined(
    df: pd.DataFrame,
    season: int,
    week: int,
    season_type: str,
    label: str,
) -> None:
    require_columns(
        df,
        [
            "season",
            "season_type",
            "week",
            "game_id",
            "away_team",
            "home_team",
            *PREDICTION_COLUMNS,
        ],
        label,
    )

    validate_unique_game_ids(
        df,
        label,
    )

    seasons = {
        parse_int(value)
        for value in df[
            "season"
        ]
    }

    weeks = {
        parse_int(value)
        for value in df[
            "week"
        ]
    }

    types = {
        normalize_season_type(
            value
        )
        for value in df[
            "season_type"
        ]
    }

    if seasons != {season}:
        fail(
            f"{label}: expected only "
            f"season={season}; "
            f"found {seasons}"
        )

    if weeks != {week}:
        fail(
            f"{label}: expected only "
            f"week={week}; "
            f"found {weeks}"
        )

    if types != {season_type}:
        fail(
            f"{label}: expected "
            "season_type="
            f"{season_type!r}; "
            f"found {types}"
        )

    validate_projection_outputs(
        df,
        label,
    )

    validate_probability_pairs(
        df
    )



def merge_schedule(
    combined: pd.DataFrame,
    schedule: pd.DataFrame,
    season: int,
    week: int,
    season_type: str,
    sportsbook: str,
) -> pd.DataFrame:
    require_columns(
        schedule,
        [
            "season",
            "season_type",
            "week",
            "game_id",
            "away_team",
            "home_team",
            "neutral_site",
            "roof",
            "bookmaker",
            "home_moneyline_american",
            "away_moneyline_american",
            "home_spread",
            "away_spread",
            "home_spread_american",
            "away_spread_american",
            "total",
            "over_american",
            "under_american",
            "odds_available",
        ],
        "weekly schedule",
    )

    validate_unique_game_ids(
        schedule,
        "weekly schedule",
    )

    season_values = pd.to_numeric(
        schedule["season"],
        errors="coerce",
    )

    week_values = pd.to_numeric(
        schedule["week"],
        errors="coerce",
    )

    type_values = schedule[
        "season_type"
    ].map(
        normalize_season_type
    )

    schedule = schedule.loc[
        (
            season_values
            == season
        )
        & (
            week_values
            == week
        )
        & (
            type_values
            == season_type
        )
    ].copy()

    if schedule.empty:
        fail(
            "Weekly schedule has "
            "no rows for "
            f"season={season}, "
            f"week={week}, "
            "season_type="
            f"{season_type}"
        )

    odds_available = schedule[
        "odds_available"
    ].map(clean)

    invalid_odds_available = (
        ~odds_available.isin(
            {"0", "1"}
        )
    )

    if invalid_odds_available.any():
        examples = (
            schedule.loc[
                invalid_odds_available,
                [
                    "game_id",
                    "odds_available",
                ],
            ]
            .head(10)
            .to_dict(
                "records"
            )
        )

        fail(
            "Weekly schedule contains "
            "invalid odds_available values; "
            f"expected only 0/1: {examples}"
        )

    available_rows = (
        odds_available.eq("1")
    )

    configured_book = (
        normalize_bookmaker(
            sportsbook
        )
    )

    bad_book = (
        schedule[
            "bookmaker"
        ]
        .map(
            normalize_bookmaker
        )
        .ne(
            configured_book
        )
        & available_rows
    )

    if bad_book.any():
        examples = (
            schedule.loc[
                bad_book,
                [
                    "game_id",
                    "bookmaker",
                ],
            ]
            .head(10)
            .to_dict(
                "records"
            )
        )

        fail(
            "Weekly schedule bookmaker "
            "does not match settings "
            f"sportsbook {sportsbook!r}: "
            f"{examples}"
        )

    base_ids = set(
        combined[
            "game_id"
        ]
    )

    schedule_ids = set(
        schedule[
            "game_id"
        ]
    )

    missing = sorted(
        base_ids
        - schedule_ids
    )

    if missing:
        fail(
            "Weekly schedule missing "
            f"{len(missing)} projected "
            "games; examples="
            f"{missing[:10]}"
        )

    schedule_by_id = (
        schedule.set_index(
            "game_id",
            drop=False,
        )
    )

    identity_mismatches: list[
        dict[str, str]
    ] = []

    for _, row in combined.iterrows():
        game_id = row["game_id"]

        if game_id not in schedule_by_id.index:
            continue

        schedule_row = schedule_by_id.loc[
            game_id
        ]

        projected_home = clean(
            row["home_team"]
        )
        projected_away = clean(
            row["away_team"]
        )
        schedule_home = clean(
            schedule_row["home_team"]
        )
        schedule_away = clean(
            schedule_row["away_team"]
        )

        if (
            projected_home != schedule_home
            or projected_away != schedule_away
        ):
            identity_mismatches.append(
                {
                    "game_id": game_id,
                    "projected_away": projected_away,
                    "projected_home": projected_home,
                    "schedule_away": schedule_away,
                    "schedule_home": schedule_home,
                }
            )

    if identity_mismatches:
        fail(
            "Weekly schedule team identity "
            "does not match projected input: "
            f"{identity_mismatches[:10]}"
        )

    columns = [
        "game_id",
        "neutral_site",
        "roof",
        "bookmaker",
        "home_moneyline_american",
        "away_moneyline_american",
        "home_spread",
        "away_spread",
        "home_spread_american",
        "away_spread_american",
        "total",
        "over_american",
        "under_american",
        "odds_available",
    ]

    source = schedule[
        columns
    ].copy()

    source = source.rename(
        columns={
            column: (
                f"sched_{column}"
            )
            for column
            in columns
            if column
            != "game_id"
        }
    )

    return combined.merge(
        source,
        on="game_id",
        how="left",
        validate="one_to_one",
    )





def build_output(
    original: pd.DataFrame,
    working: pd.DataFrame,
    max_kelly: float,
) -> pd.DataFrame:
    candidate_rows: list[dict[str, Any]] = []

    for _, row in working.iterrows():
        odds_available = (
            parse_int(
                row.get(
                    "sched_odds_available",
                    "",
                )
            )
            or 0
        )

        if odds_available != 1:
            result = empty_candidate_set(
                "CURRENT_ODDS_UNAVAILABLE"
            )
        else:
            result = {
                **evaluate_moneyline(row),
                **evaluate_spread(row),
                **evaluate_total(row),
            }

        candidate_rows.append(
            {
                "game_id": row["game_id"],
                **result,
            }
        )

    appended_columns = (
        SELECTION_COLUMNS
        + CANDIDATE_COLUMNS
    )

    candidate_frame = pd.DataFrame(
        candidate_rows,
        columns=[
            "game_id",
            *appended_columns,
        ],
    )

    for prefix in [
        "ml_home",
        "ml_away",
        "spread_home",
        "spread_away",
        "total_over",
        "total_under",
    ]:
        candidate_frame[
            f"{prefix}_kelly"
        ] = (
            pd.to_numeric(
                candidate_frame[
                    f"{prefix}_full_kelly"
                ],
                errors="coerce",
            )
            .clip(
                lower=0.0,
                upper=max_kelly,
            )
        )

    if len(candidate_frame) != len(original):
        fail(
            "Internal candidate row-count mismatch"
        )

    validate_unique_game_ids(
        candidate_frame,
        "candidate results",
    )

    original_ids = set(original["game_id"])
    candidate_ids = set(candidate_frame["game_id"])

    if candidate_ids != original_ids:
        missing_ids = sorted(
            original_ids - candidate_ids
        )
        extra_ids = sorted(
            candidate_ids - original_ids
        )
        fail(
            "Candidate game_id mismatch: "
            f"missing={missing_ids[:10]} "
            f"extra={extra_ids[:10]}"
        )

    candidate_frame = (
        original[["game_id"]]
        .merge(
            candidate_frame,
            on="game_id",
            how="left",
            validate="one_to_one",
            sort=False,
        )
    )

    output = original.copy()

    for column in appended_columns:
        output[column] = (
            candidate_frame[column].to_numpy()
        )

    return output



def _require_close(
    actual: float,
    expected: float,
    *,
    label: str,
    atol: float = 1e-12,
) -> None:
    if not math.isclose(
        actual,
        expected,
        rel_tol=1e-12,
        abs_tol=atol,
    ):
        fail(
            f"{label}: expected {expected!r}; "
            f"found {actual!r}"
        )


def validate_candidate_output(
    df: pd.DataFrame,
    original: pd.DataFrame,
    max_kelly: float,
    label: str,
) -> None:
    expected_columns = (
        list(original.columns)
        + SELECTION_COLUMNS
        + CANDIDATE_COLUMNS
    )

    if list(df.columns) != expected_columns:
        fail(
            f"{label}: final candidate "
            "column order/integrity check failed"
        )

    if len(df) != len(original):
        fail(
            f"{label}: row count changed; "
            f"expected={len(original)} "
            f"actual={len(df)}"
        )

    validate_unique_game_ids(
        df,
        label,
    )

    if (
        df["game_id"].tolist()
        != original["game_id"].tolist()
    ):
        fail(
            f"{label}: game_id order changed"
        )

    if (
        df["away_team"].tolist()
        != original["away_team"].tolist()
    ):
        fail(
            f"{label}: away_team changed"
        )

    if (
        df["home_team"].tolist()
        != original["home_team"].tolist()
    ):
        fail(
            f"{label}: home_team changed"
        )

    for column in original.columns:
        expected_values = [
            clean(value)
            for value in original[
                column
            ].tolist()
        ]
        actual_values = [
            clean(value)
            for value in df[
                column
            ].tolist()
        ]

        if actual_values != expected_values:
            fail(
                f"{label}: source column "
                f"{column!r} changed"
            )

    allowed_reasons = {
        "DEFERRED_TO_FILTER",
        "CURRENT_LINE_MISSING",
        "CURRENT_ODDS_UNAVAILABLE",
    }

    for _, row in df.iterrows():
        game_id = row["game_id"]

        for market in (
            "ml",
            "spread",
            "total",
        ):
            selected = parse_int(
                row.get(
                    f"{market}_selected",
                    "",
                )
            )

            if selected != 0:
                fail(
                    f"{label}: game_id={game_id} "
                    f"{market}_selected must be 0"
                )

            if clean(
                row.get(
                    f"{market}_selection",
                    "",
                )
            ):
                fail(
                    f"{label}: game_id={game_id} "
                    f"{market}_selection must be blank"
                )

            reason = clean(
                row.get(
                    f"{market}_selection_reason",
                    "",
                )
            )

            if reason not in allowed_reasons:
                fail(
                    f"{label}: game_id={game_id} "
                    f"invalid {market}_selection_reason="
                    f"{reason!r}"
                )

            for metric in (
                "odds_american",
                "model_probability",
                "implied_probability",
                "edge",
                "ev",
                "full_kelly",
                "kelly",
            ):
                if clean(
                    row.get(
                        f"{market}_{metric}",
                        "",
                    )
                ):
                    fail(
                        f"{label}: game_id={game_id} "
                        f"deferred {market}_{metric} "
                        "must be blank"
                    )

        for (
            market,
            first_prefix,
            second_prefix,
            include_line,
        ) in (
            (
                "ml",
                "ml_home",
                "ml_away",
                False,
            ),
            (
                "spread",
                "spread_home",
                "spread_away",
                True,
            ),
            (
                "total",
                "total_over",
                "total_under",
                True,
            ),
        ):
            first_available = parse_int(
                row.get(
                    f"{first_prefix}_available",
                    "",
                )
            )
            second_available = parse_int(
                row.get(
                    f"{second_prefix}_available",
                    "",
                )
            )

            if first_available not in {0, 1}:
                fail(
                    f"{label}: game_id={game_id} "
                    f"invalid {first_prefix}_available"
                )

            if second_available not in {0, 1}:
                fail(
                    f"{label}: game_id={game_id} "
                    f"invalid {second_prefix}_available"
                )

            if first_available != second_available:
                fail(
                    f"{label}: game_id={game_id} "
                    f"{market} candidate availability "
                    "must be paired"
                )

            reason = clean(
                row[
                    f"{market}_selection_reason"
                ]
            )

            if first_available == 0:
                for prefix in (
                    first_prefix,
                    second_prefix,
                ):
                    for metric in (
                        "odds_american",
                        "model_probability",
                        "implied_probability",
                        "edge",
                        "ev",
                        "full_kelly",
                        "kelly",
                    ):
                        if clean(
                            row.get(
                                f"{prefix}_{metric}",
                                "",
                            )
                        ):
                            fail(
                                f"{label}: game_id={game_id} "
                                f"unavailable {prefix}_{metric} "
                                "must be blank"
                            )

                    if include_line:
                        line_text = clean(
                            row.get(
                                f"{prefix}_line",
                                "",
                            )
                        )
                        if (
                            line_text
                            and parse_float(
                                line_text
                            )
                            is None
                        ):
                            fail(
                                f"{label}: game_id={game_id} "
                                f"{prefix}_line is non-finite"
                            )

                if reason not in {
                    "CURRENT_LINE_MISSING",
                    "CURRENT_ODDS_UNAVAILABLE",
                }:
                    fail(
                        f"{label}: game_id={game_id} "
                        f"unavailable {market} has "
                        f"unexpected reason={reason!r}"
                    )

                continue

            if reason != "DEFERRED_TO_FILTER":
                fail(
                    f"{label}: game_id={game_id} "
                    f"available {market} must be "
                    "DEFERRED_TO_FILTER"
                )

            first_odds = parse_float(
                row[
                    f"{first_prefix}_odds_american"
                ]
            )
            second_odds = parse_float(
                row[
                    f"{second_prefix}_odds_american"
                ]
            )

            if (
                first_odds is None
                or second_odds is None
                or first_odds == 0
                or second_odds == 0
            ):
                fail(
                    f"{label}: game_id={game_id} "
                    f"{market} candidate odds "
                    "must be finite and nonzero"
                )

            first_fair, second_fair = (
                no_vig_probabilities(
                    first_odds,
                    second_odds,
                )
            )

            for prefix, fair in (
                (
                    first_prefix,
                    first_fair,
                ),
                (
                    second_prefix,
                    second_fair,
                ),
            ):
                model_probability = parse_float(
                    row[
                        f"{prefix}_model_probability"
                    ]
                )
                implied_probability = parse_float(
                    row[
                        f"{prefix}_implied_probability"
                    ]
                )
                edge = parse_float(
                    row[
                        f"{prefix}_edge"
                    ]
                )
                ev = parse_float(
                    row[
                        f"{prefix}_ev"
                    ]
                )
                full_kelly = parse_float(
                    row[
                        f"{prefix}_full_kelly"
                    ]
                )
                kelly = parse_float(
                    row[
                        f"{prefix}_kelly"
                    ]
                )
                odds = parse_float(
                    row[
                        f"{prefix}_odds_american"
                    ]
                )

                if any(
                    value is None
                    for value in (
                        model_probability,
                        implied_probability,
                        edge,
                        ev,
                        full_kelly,
                        kelly,
                        odds,
                    )
                ):
                    fail(
                        f"{label}: game_id={game_id} "
                        f"{prefix} contains blank/"
                        "non-finite candidate metrics"
                    )

                assert model_probability is not None
                assert implied_probability is not None
                assert edge is not None
                assert ev is not None
                assert full_kelly is not None
                assert kelly is not None
                assert odds is not None

                if not (
                    0.0
                    <= model_probability
                    <= 1.0
                ):
                    fail(
                        f"{label}: game_id={game_id} "
                        f"{prefix}_model_probability "
                        "outside [0,1]"
                    )

                if not (
                    0.0
                    <= implied_probability
                    <= 1.0
                ):
                    fail(
                        f"{label}: game_id={game_id} "
                        f"{prefix}_implied_probability "
                        "outside [0,1]"
                    )

                expected = calculate_metrics(
                    model_probability,
                    odds,
                    fair,
                )

                _require_close(
                    implied_probability,
                    expected[
                        "implied_probability"
                    ],
                    label=(
                        f"{label}: game_id={game_id} "
                        f"{prefix}_implied_probability"
                    ),
                )
                _require_close(
                    edge,
                    expected["edge"],
                    label=(
                        f"{label}: game_id={game_id} "
                        f"{prefix}_edge"
                    ),
                )
                _require_close(
                    ev,
                    expected["ev"],
                    label=(
                        f"{label}: game_id={game_id} "
                        f"{prefix}_ev"
                    ),
                )
                _require_close(
                    full_kelly,
                    expected["full_kelly"],
                    label=(
                        f"{label}: game_id={game_id} "
                        f"{prefix}_full_kelly"
                    ),
                )
                _require_close(
                    kelly,
                    min(
                        expected[
                            "full_kelly"
                        ],
                        max_kelly,
                    ),
                    label=(
                        f"{label}: game_id={game_id} "
                        f"{prefix}_kelly"
                    ),
                )

            if include_line:
                first_line = parse_float(
                    row[
                        f"{first_prefix}_line"
                    ]
                )
                second_line = parse_float(
                    row[
                        f"{second_prefix}_line"
                    ]
                )

                if (
                    first_line is None
                    or second_line is None
                ):
                    fail(
                        f"{label}: game_id={game_id} "
                        f"{market} candidate lines "
                        "must be finite"
                    )

                if market == "spread":
                    _require_close(
                        first_line
                        + second_line,
                        0.0,
                        label=(
                            f"{label}: game_id={game_id} "
                            "spread lines must be opposites"
                        ),
                        atol=1e-9,
                    )
                else:
                    _require_close(
                        first_line,
                        second_line,
                        label=(
                            f"{label}: game_id={game_id} "
                            "total candidate lines must match"
                        ),
                        atol=1e-9,
                    )


def stage_candidate_csv(
    df: pd.DataFrame,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    descriptor, raw_path = (
        tempfile.mkstemp(
            prefix=(
                f".{output_path.name}."
                "stage."
            ),
            suffix=".csv",
            dir=str(
                output_path.parent
            ),
        )
    )
    os.close(descriptor)

    staged_path = Path(
        raw_path
    )

    try:
        df.to_csv(
            staged_path,
            index=False,
            encoding="utf-8",
        )

        if (
            not staged_path.is_file()
            or staged_path.stat().st_size
            == 0
        ):
            fail(
                "Staged candidate output "
                f"was not written: {staged_path}"
            )

        return staged_path
    except Exception:
        staged_path.unlink(
            missing_ok=True
        )
        raise


def validate_serialized_candidate_csv(
    path: Path,
    *,
    original: pd.DataFrame,
    max_kelly: float,
    label: str,
) -> pd.DataFrame:
    staged = read_csv(
        path,
        label,
    )
    assert staged is not None

    validate_candidate_output(
        staged,
        original,
        max_kelly,
        label,
    )

    return staged


def publish_candidate_csv(
    staged_path: Path,
    output_path: Path,
    *,
    original: pd.DataFrame,
    max_kelly: float,
    reporter: PipelineReporter,
) -> None:
    backup_path = (
        output_path.parent
        / (
            f".{output_path.name}."
            f"backup.{uuid.uuid4().hex}"
        )
    )
    had_existing_output = (
        output_path.exists()
    )
    published = False
    rollback_failed = False

    reporter.update_details(
        {
            "publication_mode": (
                "atomic_replace_with_"
                "backup_rollback"
            ),
            "publication_completed": False,
            "post_publish_validation": False,
            "rollback_performed": False,
        }
    )

    try:
        if had_existing_output:
            shutil.copy2(
                output_path,
                backup_path,
            )

        os.replace(
            staged_path,
            output_path,
        )
        published = True

        validate_serialized_candidate_csv(
            output_path,
            original=original,
            max_kelly=max_kelly,
            label=(
                "published candidate output"
            ),
        )

        reporter.update_details(
            {
                "publication_completed": True,
                "post_publish_validation": True,
            }
        )
    except Exception as publish_exc:
        if published:
            try:
                if (
                    had_existing_output
                    and backup_path.exists()
                ):
                    os.replace(
                        backup_path,
                        output_path,
                    )
                elif (
                    not had_existing_output
                    and output_path.exists()
                ):
                    output_path.unlink()

                reporter.update_details(
                    {
                        "publication_completed": False,
                        "post_publish_validation": False,
                        "rollback_performed": True,
                    }
                )
            except Exception as rollback_exc:
                rollback_failed = True

                reporter.update_details(
                    {
                        "publication_completed": False,
                        "post_publish_validation": False,
                        "rollback_performed": False,
                        "rollback_error_type": (
                            type(
                                rollback_exc
                            ).__name__
                        ),
                        "rollback_error": str(
                            rollback_exc
                        ),
                    }
                )

                raise RuntimeError(
                    "Candidate output publication "
                    "failed and rollback also "
                    "failed: "
                    f"publication_error="
                    f"{publish_exc}; "
                    f"rollback_error="
                    f"{rollback_exc}"
                ) from rollback_exc

        raise
    finally:
        staged_path.unlink(
            missing_ok=True
        )

        if (
            backup_path.exists()
            and not rollback_failed
            and (
                not published
                or output_path.exists()
            )
        ):
            try:
                backup_path.unlink()
            except Exception as exc:
                reporter.warning(
                    "Candidate output published "
                    "but temporary backup cleanup "
                    "failed",
                    backup_path=str(
                        backup_path
                    ),
                    error_type=(
                        type(exc).__name__
                    ),
                    error=str(exc),
                )



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--season",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--week",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--settings",
        type=Path,
        default=DEFAULT_SETTINGS_PATH,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    return parser.parse_args()


def candidate_summary(
    output: pd.DataFrame,
) -> dict[str, Any]:
    candidate_counts: dict[
        str,
        int,
    ] = {}

    for prefix in (
        "ml_home",
        "ml_away",
        "spread_home",
        "spread_away",
        "total_over",
        "total_under",
    ):
        candidate_counts[prefix] = int(
            pd.to_numeric(
                output[
                    f"{prefix}_available"
                ],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )

    reason_counts: dict[
        str,
        dict[str, int],
    ] = {}

    for market in (
        "ml",
        "spread",
        "total",
    ):
        counts = (
            output[
                f"{market}_selection_reason"
            ]
            .astype(str)
            .value_counts(
                dropna=False
            )
            .to_dict()
        )

        reason_counts[market] = {
            str(key): int(value)
            for key, value
            in counts.items()
        }

    return {
        "candidate_counts": (
            candidate_counts
        ),
        "market_reason_counts": (
            reason_counts
        ),
    }


def run(
    args: argparse.Namespace,
    reporter: PipelineReporter,
) -> None:
    load_runtime_dependencies(
        reporter,
    )

    settings_path = (
        args.settings.resolve()
    )
    reporter.add_input(
        settings_path
    )

    settings = read_yaml(
        settings_path,
        "settings config",
    )

    selection_defaults = settings.get(
        "selection_defaults"
    )

    if not isinstance(
        selection_defaults,
        dict,
    ):
        fail(
            "settings.yaml must contain "
            "selection_defaults"
        )

    max_kelly = parse_float(
        selection_defaults.get(
            "max_kelly"
        )
    )

    if (
        max_kelly is None
        or max_kelly < 0
    ):
        fail(
            "settings.yaml "
            "selection_defaults.max_kelly "
            "must be a non-negative number"
        )

    (
        season,
        week,
        season_type,
        sportsbook,
    ) = validate_settings(
        settings,
        args.season,
        args.week,
    )

    reporter.season = season
    reporter.week = week

    reporter.update_details(
        {
            "resolved_season": season,
            "resolved_week": week,
            "season_type": season_type,
            "sportsbook": sportsbook,
            "settings_path": str(
                settings_path
            ),
            "season_override": (
                args.season
            ),
            "week_override": (
                args.week
            ),
            "input_override": (
                str(
                    args.input.resolve()
                )
                if args.input
                is not None
                else None
            ),
            "output_override": (
                str(
                    args.output.resolve()
                )
                if args.output
                is not None
                else None
            ),
            "max_kelly": max_kelly,
            "candidate_formulas": (
                "existing_no_vig_edge_ev_"
                "full_kelly_preserved"
            ),
            "weather_dependency": (
                "removed_no_effect_on_"
                "candidate_artifact"
            ),
            "staged_roundtrip_verified": False,
            "publication_completed": False,
            "post_publish_validation": False,
            "rollback_performed": False,
        }
    )

    input_path = (
        args.input.resolve()
        if args.input is not None
        else (
            NFL_ROOT
            / "01_merge"
            / f"week_{week}_NFL_enriched.csv"
        )
    )

    output_path = (
        args.output.resolve()
        if args.output is not None
        else (
            NFL_ROOT
            / "02_select"
            / f"week_{week}_NFL_selected.csv"
        )
    )

    if output_path == input_path:
        fail(
            "Candidate output path must differ "
            "from the input path; selections.py "
            "will not overwrite a file it reads."
        )

    reporter.add_input(
        input_path
    )
    reporter.add_output(
        output_path
    )

    combined = read_csv(
        input_path,
        "projected combined enriched file",
    )
    assert combined is not None

    source_rows = len(
        combined
    )
    reporter.set_rows(
        rows_in=source_rows
    )

    prior_output_columns = [
        column
        for column in (
            SELECTION_COLUMNS
            + CANDIDATE_COLUMNS
        )
        if column in combined.columns
    ]

    if prior_output_columns:
        combined = combined.drop(
            columns=prior_output_columns
        )

    validate_combined(
        combined,
        season,
        week,
        season_type,
        str(input_path),
    )

    schedule_path = (
        NFL_ROOT
        / "00_intake/schedule/weekly"
        / f"week_{week}_NFL_weekly_schedule.csv"
    )
    reporter.add_input(
        schedule_path
    )

    schedule = read_csv(
        schedule_path,
        "weekly schedule",
    )
    assert schedule is not None

    working = merge_schedule(
        combined.copy(),
        schedule,
        season,
        week,
        season_type,
        sportsbook,
    )

    output = build_output(
        combined,
        working,
        max_kelly,
    )

    validate_candidate_output(
        output,
        combined,
        max_kelly,
        "in-memory candidate output",
    )

    summary = candidate_summary(
        output
    )
    reporter.update_details(
        {
            "source_rows": source_rows,
            "source_columns": len(
                combined.columns
            ),
            "output_rows": len(
                output
            ),
            "output_columns": len(
                output.columns
            ),
            **summary,
        }
    )

    staged_path = stage_candidate_csv(
        output,
        output_path,
    )

    try:
        validate_serialized_candidate_csv(
            staged_path,
            original=combined,
            max_kelly=max_kelly,
            label=(
                "staged candidate output"
            ),
        )
        reporter.set_detail(
            "staged_roundtrip_verified",
            True,
        )

        publish_candidate_csv(
            staged_path,
            output_path,
            original=combined,
            max_kelly=max_kelly,
            reporter=reporter,
        )
    finally:
        staged_path.unlink(
            missing_ok=True
        )

    reporter.set_rows(
        rows_out=len(
            output
        )
    )

    print(
        "Step 15 candidate enrichment complete: "
        f"season={season} "
        f"week={week} "
        f"games={len(output)}"
    )

    for prefix, count in (
        summary[
            "candidate_counts"
        ].items()
    ):
        print(
            f"{prefix}_candidates={count}"
        )

    print(
        f"Updated: {output_path}"
    )


def main() -> int:
    args = parse_args()

    try:
        with PipelineReporter(
            script=SCRIPT_PATH,
            stage="02_select",
            report_root=REPORT_ROOT,
            pipeline="NFL",
            league="NFL",
            extra_context={
                "component": (
                    "candidate enrichment"
                ),
                "selection_stage": (
                    "deferred_to_filter"
                ),
            },
        ) as reporter:
            run(
                args,
                reporter,
            )

        return 0
    except Exception as exc:
        print(
            f"ERROR: {type(exc).__name__}: "
            f"{exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
