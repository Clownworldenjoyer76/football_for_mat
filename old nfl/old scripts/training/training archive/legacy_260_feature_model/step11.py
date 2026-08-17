#!/usr/bin/env python3
"""
Step 11: train final NFL margin and total-points models.

Repository location:
    docs/win/football/nfl/scripts/training/step11.py

Inputs:
    docs/win/football/nfl/training/historical_core_<season>.csv

Only exact four-digit season files are used. By default the script starts at
2021 and automatically includes every contiguous season through the latest
available file. Optional --start-season / --end-season arguments can constrain
the range.

Targets:
    margin
    total_points

Uses only the approved pregame feature families from Steps 1-10:
    - DRAT
    - EPRED
    - market
    - ml_* / ats_* / totals_* enrichment
    - lagged team statistics
    - lagged QB statistics
    - schedule/rest/venue/weather/travel
    - depth/injury

No holdout/backtest is performed here; chronological backtesting belongs to
Step 13.

Outputs:
    docs/win/football/nfl/models/step11_margin_model.cbm
    docs/win/football/nfl/models/step11_total_points_model.cbm
    docs/win/football/nfl/models/step11_feature_schema.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool


SCRIPT_DIR = Path(__file__).resolve().parent
NFL_ROOT = SCRIPT_DIR.parents[1]
TRAINING_DIR = NFL_ROOT / "training"
MODELS_DIR = NFL_ROOT / "models"

HISTORICAL_FILE_PATTERN = re.compile(r"^historical_core_(\d{4})\.csv$")
DEFAULT_START_SEASON = 2021
EXPECTED_FEATURE_COUNT = 260

MARGIN_MODEL_PATH = MODELS_DIR / "step11_margin_model.cbm"
TOTAL_MODEL_PATH = MODELS_DIR / "step11_total_points_model.cbm"
SCHEMA_PATH = MODELS_DIR / "step11_feature_schema.json"

TARGETS = ["margin", "total_points"]

# Fixed model specification. Step 13 must use this same training process when
# producing chronological held-out predictions.
MODEL_PARAMS = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "iterations": 400,
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 10.0,
    "random_strength": 1.0,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 1.0,
    "one_hot_max_size": 10,
    "max_ctr_complexity": 1,
    "random_seed": 42,
    "thread_count": -1,
    "allow_writing_files": False,
    "verbose": False,
}

TEAM_METRICS = [
    "off_epa_per_play",
    "def_epa_per_play",
    "off_success_rate",
    "def_success_rate",
    "yards_per_play",
    "yards_per_play_allowed",
    "points_per_drive",
    "points_per_drive_allowed",
    "red_zone_td_rate",
    "red_zone_td_rate_allowed",
    "early_down_epa",
    "third_down_conversion_rate",
]

QB_METRICS = [
    "epa_per_play",
    "cpoe",
    "air_yards",
    "sack_rate",
    "interception_rate",
    "fumble_rate",
]

MARKET_FEATURES = [
    "away_moneyline",
    "home_moneyline",
    "spread_line",
    "away_spread_odds",
    "home_spread_odds",
    "total_line",
    "under_odds",
    "over_odds",
    "hist_odds_total",
    "hist_home_spread",
    "hist_away_spread",
]

SCHEDULE_REST_VENUE_WEATHER_FEATURES = [
    "game_type",
    "week",
    "weekday",
    "gametime",
    "away_team",
    "home_team",
    "location",
    "away_rest",
    "home_rest",
    "div_game",
    "roof",
    "surface",
    "temp",
    "wind",
    "stadium_id",
    "stadium",
    "hist_surface",
    "hist_weather_icon",
    "hist_temperature",
    "hist_precip_probability",
    "hist_precip_type",
    "hist_wind_speed",
    "hist_wind_bearing",
    "rest_diff",
    "miles_traveled",
    "time_zones_crossed",
    "east_to_west",
    "west_to_east",
    "international_flag",
    "neutral_site_flag",
]

# These are explicitly not model inputs.
FORBIDDEN_FEATURES = {
    "game_id",
    "season",
    "gameday",
    "away_score",
    "home_score",
    "margin",
    "total_points",
    "home_win",
    "home_ats_margin",
    "home_ats_result",
    "total_result",
    "away_qb_id",
    "home_qb_id",
    "away_qb_name",
    "home_qb_name",
    "away_coach",
    "home_coach",
}

FORCED_CATEGORICAL = {
    "game_type",
    "weekday",
    "gametime",
    "away_team",
    "home_team",
    "location",
    "roof",
    "surface",
    "stadium_id",
    "stadium",
    "hist_surface",
    "hist_weather_icon",
    "hist_precip_type",
}

BLANK_TOKENS = {"", "nan", "none", "null", "<na>", "nat"}
MISSING_CATEGORY = "__MISSING__"


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_training_seasons(
    start_season: int = DEFAULT_START_SEASON,
    end_season: int | None = None,
) -> list[int]:
    if not TRAINING_DIR.is_dir():
        fail(f"Training directory not found: {TRAINING_DIR}")

    available: list[int] = []
    for path in TRAINING_DIR.iterdir():
        match = HISTORICAL_FILE_PATTERN.fullmatch(path.name)
        if match and path.is_file():
            available.append(int(match.group(1)))

    available = sorted(set(available))
    if not available:
        fail(
            f"No exact historical_core_<YYYY>.csv files found in {TRAINING_DIR}"
        )

    if start_season not in available:
        fail(
            f"Start season {start_season} is missing. "
            f"Available exact season files: {available}"
        )

    resolved_end = max(available) if end_season is None else int(end_season)
    if resolved_end < start_season:
        fail(
            f"end-season {resolved_end} cannot be earlier than "
            f"start-season {start_season}"
        )

    seasons = list(range(start_season, resolved_end + 1))
    missing = [season for season in seasons if season not in available]
    if missing:
        fail(
            "Historical season files must be contiguous. "
            f"Missing: {missing}; available: {available}"
        )

    return seasons


def input_paths_for(seasons: list[int]) -> dict[int, Path]:
    return {
        season: TRAINING_DIR / f"historical_core_{season}.csv"
        for season in seasons
    }


def unique_preserve(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def team_feature_names() -> list[str]:
    output: list[str] = []
    for metric in TEAM_METRICS:
        output.extend(
            [
                f"home_{metric}",
                f"away_{metric}",
                f"{metric}_diff",
            ]
        )
    return output


def qb_feature_names() -> list[str]:
    output: list[str] = []
    for metric in QB_METRICS:
        output.extend(
            [
                f"home_qb_{metric}",
                f"away_qb_{metric}",
                f"qb_{metric}_diff",
            ]
        )
    return output


def build_feature_list(columns: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    drat = [column for column in columns if column.startswith("drat_")]
    epred = [column for column in columns if column.startswith("epred_")]
    enrichment = [
        column
        for column in columns
        if column.startswith(("ml_", "ats_", "totals_"))
    ]
    team = team_feature_names()
    qb = qb_feature_names()
    depth_injury = [
        column
        for column in columns
        if "inj_" in column or "depth_starter_changes" in column
    ]

    families = {
        "drat": drat,
        "epred": epred,
        "market": list(MARKET_FEATURES),
        "enrichment": enrichment,
        "team": team,
        "qb": qb,
        "schedule_rest_venue_weather_travel": list(
            SCHEDULE_REST_VENUE_WEATHER_FEATURES
        ),
        "depth_injury": depth_injury,
    }

    features = unique_preserve(
        drat
        + epred
        + list(MARKET_FEATURES)
        + enrichment
        + team
        + qb
        + list(SCHEDULE_REST_VENUE_WEATHER_FEATURES)
        + depth_injury
    )

    missing = [column for column in features if column not in columns]
    if missing:
        fail(f"Approved feature columns missing from historical schema: {missing}")

    forbidden = sorted(set(features) & FORBIDDEN_FEATURES)
    if forbidden:
        fail(f"Forbidden/leakage columns entered feature matrix: {forbidden}")

    return features, families


def read_inputs(
    seasons: list[int],
    input_paths: dict[int, Path],
) -> tuple[pd.DataFrame, dict[str, str]]:
    frames: list[pd.DataFrame] = []
    reference_columns: list[str] | None = None
    hashes: dict[str, str] = {}

    for season in seasons:
        path = input_paths[season]
        if not path.exists():
            fail(f"Missing input file: {path}")

        hashes[path.name] = sha256_file(path)

        df = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            encoding="utf-8-sig",
            low_memory=False,
        )

        if df.empty:
            fail(f"{path}: no data rows")
        if len(df.columns) != len(set(df.columns)):
            fail(f"{path}: duplicate column names")

        if reference_columns is None:
            reference_columns = list(df.columns)
        elif list(df.columns) != reference_columns:
            missing = [c for c in reference_columns if c not in df.columns]
            extra = [c for c in df.columns if c not in reference_columns]
            fail(
                f"{path}: schema/order differs from {seasons[0]}; "
                f"missing={missing}; extra={extra}"
            )

        required = {"game_id", "season", "margin", "total_points"}
        missing_required = sorted(required - set(df.columns))
        if missing_required:
            fail(f"{path}: missing required columns: {missing_required}")

        parsed_season = pd.to_numeric(df["season"], errors="coerce")
        if parsed_season.isna().any() or not (
            parsed_season.astype(int) == season
        ).all():
            fail(f"{path}: contains invalid/wrong-season rows")

        for target in TARGETS:
            numeric = pd.to_numeric(df[target], errors="coerce")
            if numeric.isna().any():
                fail(
                    f"{path}: target {target!r} contains "
                    f"{int(numeric.isna().sum())} blank/non-numeric rows"
                )

        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    if combined["game_id"].astype(str).str.strip().eq("").any():
        fail("Blank game_id values detected")
    if combined["game_id"].duplicated().any():
        examples = combined.loc[
            combined["game_id"].duplicated(), "game_id"
        ].head(10).tolist()
        fail(f"Duplicate game_id values detected: {examples}")

    return combined, hashes


def looks_numeric(series: pd.Series) -> bool:
    text = series.astype(str).str.strip()
    nonblank = text[~text.str.casefold().isin(BLANK_TOKENS)]
    if nonblank.empty:
        return True
    numeric = pd.to_numeric(
        nonblank.str.replace("%", "", regex=False),
        errors="coerce",
    )
    return bool(numeric.notna().all())


def infer_feature_types(
    raw: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[list[str], list[str]]:
    categorical: list[str] = []
    numeric: list[str] = []

    for column in feature_columns:
        if column in FORCED_CATEGORICAL:
            categorical.append(column)
            continue

        if (
            column.endswith("_rule_id")
            or column.endswith("_rule_ids")
            or column.endswith("_rule_conditions")
        ):
            categorical.append(column)
            continue

        if looks_numeric(raw[column]):
            numeric.append(column)
        else:
            categorical.append(column)

    return categorical, numeric


def prepare_feature_matrix(
    raw: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    numeric_columns: list[str],
) -> pd.DataFrame:
    X = raw[feature_columns].copy()

    for column in numeric_columns:
        original = X[column].astype(str).str.strip()
        cleaned = original.str.replace("%", "", regex=False)
        blank_mask = cleaned.str.casefold().isin(BLANK_TOKENS)
        cleaned = cleaned.mask(blank_mask, np.nan)
        converted = pd.to_numeric(cleaned, errors="coerce")

        bad = (~blank_mask) & converted.isna()
        if bad.any():
            examples = original[bad].head(5).tolist()
            fail(
                f"Numeric conversion failed for {column}: "
                f"examples={examples}"
            )
        X[column] = converted

    for column in categorical_columns:
        cleaned = X[column].astype(str).str.strip()
        blank_mask = cleaned.str.casefold().isin(BLANK_TOKENS)
        X[column] = cleaned.mask(blank_mask, MISSING_CATEGORY)

    return X


def categorical_indices(
    feature_columns: list[str],
    categorical_columns: list[str],
) -> list[int]:
    categorical_set = set(categorical_columns)
    return [
        index
        for index, column in enumerate(feature_columns)
        if column in categorical_set
    ]


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    cat_indices: list[int],
) -> CatBoostRegressor:
    pool = Pool(
        X,
        label=y,
        cat_features=cat_indices,
        feature_names=list(X.columns),
    )
    model = CatBoostRegressor(**MODEL_PARAMS)
    model.fit(pool, verbose=False)
    return model


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start-season",
        type=int,
        default=DEFAULT_START_SEASON,
        help="First historical season to train on (default: 2021).",
    )
    parser.add_argument(
        "--end-season",
        type=int,
        default=None,
        help="Last historical season. Default: latest exact season file present.",
    )
    args = parser.parse_args()

    seasons = discover_training_seasons(
        start_season=int(args.start_season),
        end_season=args.end_season,
    )
    input_paths = input_paths_for(seasons)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    raw, input_hashes = read_inputs(seasons, input_paths)
    feature_columns, feature_families = build_feature_list(list(raw.columns))
    categorical_columns, numeric_columns = infer_feature_types(
        raw,
        feature_columns,
    )
    X = prepare_feature_matrix(
        raw,
        feature_columns,
        categorical_columns,
        numeric_columns,
    )
    cat_indices = categorical_indices(
        feature_columns,
        categorical_columns,
    )

    if set(feature_columns) & FORBIDDEN_FEATURES:
        fail("Forbidden features detected immediately before training")
    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        fail(
            f"Expected {EXPECTED_FEATURE_COUNT} Step 11 features, "
            f"found {len(feature_columns)}"
        )

    print(
        f"Training rows={len(raw)}, features={len(feature_columns)}, "
        f"numeric={len(numeric_columns)}, categorical={len(categorical_columns)}"
    )

    y_margin = pd.to_numeric(raw["margin"], errors="raise").astype(float)
    y_total = pd.to_numeric(raw["total_points"], errors="raise").astype(float)

    print(f"Training margin model on all completed {seasons[0]}-{seasons[-1]} rows...")
    margin_model = train_model(X, y_margin, cat_indices)

    print(f"Training total_points model on all completed {seasons[0]}-{seasons[-1]} rows...")
    total_model = train_model(X, y_total, cat_indices)

    margin_model.save_model(MARGIN_MODEL_PATH)
    total_model.save_model(TOTAL_MODEL_PATH)

    schema = {
        "step": 11,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_seasons": seasons,
        "training_rows": int(len(raw)),
        "targets": TARGETS,
        "model_type": "CatBoostRegressor",
        "model_files": {
            "margin": MARGIN_MODEL_PATH.name,
            "total_points": TOTAL_MODEL_PATH.name,
        },
        "model_params": MODEL_PARAMS,
        "feature_count": len(feature_columns),
        "feature_order": feature_columns,
        "numeric_features": numeric_columns,
        "categorical_features": categorical_columns,
        "categorical_feature_indices": cat_indices,
        "feature_families": feature_families,
        "forbidden_features": sorted(FORBIDDEN_FEATURES),
        "preprocessing": {
            "numeric": "strip whitespace; remove %; blank tokens -> NaN; parse numeric",
            "categorical": f"strip whitespace; blank tokens -> {MISSING_CATEGORY}",
            "blank_tokens_case_insensitive": sorted(BLANK_TOKENS),
            "missing_category_token": MISSING_CATEGORY,
        },
        "input_files": [input_paths[season].name for season in seasons],
        "input_sha256": input_hashes,
    }

    SCHEMA_PATH.write_text(
        json.dumps(json_safe(schema), indent=2, sort_keys=False),
        encoding="utf-8",
    )

    print("Step 11 complete.")
    print(f"Training seasons: {seasons[0]}-{seasons[-1]}")
    print(f"Margin model: {MARGIN_MODEL_PATH}")
    print(f"Total model:  {TOTAL_MODEL_PATH}")
    print(f"Schema:       {SCHEMA_PATH}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
