#!/usr/bin/env python3
"""
Step 13: chronological NFL backtest using the exact Step 11 training process.

Repository location:
    docs/win/football/nfl/scripts/training/step13.py

Inputs:
    docs/win/football/nfl/training/historical_core_<season>.csv
    docs/win/football/nfl/models/step11_feature_schema.json

The season range comes from the Step 11 schema. The first season is initial
training history; every later season is held out chronologically.

Output:
    docs/win/football/nfl/training/backtests/step13_chronological_backtest.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool


SCRIPT_DIR = Path(__file__).resolve().parent
NFL_ROOT = SCRIPT_DIR.parents[1]
TRAINING_DIR = NFL_ROOT / "training"
MODELS_DIR = NFL_ROOT / "models"
BACKTEST_DIR = TRAINING_DIR / "backtests"

SCHEMA_PATH = MODELS_DIR / "step11_feature_schema.json"
OUTPUT_PATH = BACKTEST_DIR / "step13_chronological_backtest.csv"
CHECKPOINT_PATH = BACKTEST_DIR / "step13_backtest_checkpoint.csv"

BLANK_TOKENS = {"", "nan", "none", "null", "<na>", "nat"}
MISSING_CATEGORY = "__MISSING__"

REQUIRED_RESULT_COLUMNS = [
    "game_id", "season", "week", "gameday", "gametime",
    "away_team", "home_team", "spread_line", "total_line",
    "margin", "total_points", "home_win", "home_ats_result", "total_result",
]

OUTPUT_COLUMNS = [
    "season", "week", "gameday", "gametime", "game_id",
    "away_team", "home_team", "training_rows", "spread_line", "total_line",
    "predicted_margin", "actual_margin", "predicted_total", "actual_total_points",
    "predicted_home_score", "predicted_away_score", "predicted_ml_winner",
    "actual_home_win", "predicted_ats_side", "actual_home_ats_result",
    "predicted_ou", "actual_total_result",
]


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_schema() -> dict[str, Any]:
    if not SCHEMA_PATH.is_file():
        fail(f"Missing Step 11 schema: {SCHEMA_PATH}")

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    required = {
        "training_seasons", "targets", "model_params", "feature_count",
        "feature_order", "numeric_features", "categorical_features",
        "categorical_feature_indices",
    }
    missing = sorted(required - set(schema))
    if missing:
        fail(f"Step 11 schema missing required keys: {missing}")

    try:
        seasons = [int(value) for value in schema["training_seasons"]]
    except Exception as exc:
        fail(f"Invalid Step 11 training_seasons: {exc}")

    if len(seasons) < 2:
        fail("Step 13 requires at least two Step 11 training seasons")
    if seasons != sorted(set(seasons)):
        fail(f"Step 11 training_seasons must be unique and sorted: {seasons}")
    if seasons != list(range(seasons[0], seasons[-1] + 1)):
        fail(f"Step 11 training_seasons must be contiguous: {seasons}")
    if list(schema["targets"]) != ["margin", "total_points"]:
        fail(f"Unexpected Step 11 targets: {schema['targets']}")

    feature_order = list(schema["feature_order"])
    numeric = list(schema["numeric_features"])
    categorical = list(schema["categorical_features"])
    cat_indices = list(schema["categorical_feature_indices"])

    if len(feature_order) != int(schema["feature_count"]):
        fail("Step 11 feature_count does not match feature_order")
    if len(feature_order) != len(set(feature_order)):
        fail("Duplicate features in Step 11 feature_order")
    if set(numeric) & set(categorical):
        fail("Numeric/categorical overlap in Step 11 schema")
    if set(numeric) | set(categorical) != set(feature_order):
        fail("Numeric/categorical lists do not exactly cover feature_order")
    expected_indices = [
        i for i, column in enumerate(feature_order) if column in set(categorical)
    ]
    if cat_indices != expected_indices:
        fail("categorical_feature_indices do not match Step 11 feature_order")

    return schema


def read_inputs(schema: dict[str, Any], seasons: list[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    reference_columns: list[str] | None = None
    expected_hashes = schema.get("input_sha256", {})

    for season in seasons:
        path = TRAINING_DIR / f"historical_core_{season}.csv"
        if not path.is_file():
            fail(f"Missing input file: {path}")

        expected_hash = expected_hashes.get(path.name)
        if expected_hash and sha256_file(path) != expected_hash:
            fail(
                f"{path.name}: SHA256 does not match the file used by Step 11"
            )

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
            fail(
                f"{path}: schema/order differs from "
                f"historical_core_{seasons[0]}.csv"
            )

        missing = sorted(set(REQUIRED_RESULT_COLUMNS) - set(df.columns))
        if missing:
            fail(f"{path}: missing backtest columns: {missing}")
        feature_missing = [
            column for column in schema["feature_order"] if column not in df.columns
        ]
        if feature_missing:
            fail(f"{path}: missing Step 11 features: {feature_missing}")

        parsed_season = pd.to_numeric(df["season"], errors="coerce")
        if parsed_season.isna().any() or not (
            parsed_season.astype(int) == season
        ).all():
            fail(f"{path}: contains invalid/wrong-season rows")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    game_id = combined["game_id"].astype(str).str.strip()
    if game_id.eq("").any():
        fail("Blank game_id values detected")
    if game_id.duplicated().any():
        fail(
            "Duplicate game_id values detected: "
            f"{game_id[game_id.duplicated()].head(10).tolist()}"
        )
    return combined


def prepare_feature_matrix(raw: pd.DataFrame, schema: dict[str, Any]) -> pd.DataFrame:
    feature_order = list(schema["feature_order"])
    numeric_features = list(schema["numeric_features"])
    categorical_features = list(schema["categorical_features"])
    X = raw[feature_order].copy()

    for column in numeric_features:
        original = X[column].astype(str).str.strip()
        cleaned = original.str.replace("%", "", regex=False)
        blank = cleaned.str.casefold().isin(BLANK_TOKENS)
        converted = pd.to_numeric(cleaned.mask(blank, np.nan), errors="coerce")
        bad = (~blank) & converted.isna()
        if bad.any():
            fail(
                f"Numeric conversion failed for {column}: "
                f"{original[bad].head(5).tolist()}"
            )
        X[column] = converted

    for column in categorical_features:
        cleaned = X[column].astype(str).str.strip()
        blank = cleaned.str.casefold().isin(BLANK_TOKENS)
        X[column] = cleaned.mask(blank, MISSING_CATEGORY)

    return X


def build_chronology(raw: pd.DataFrame) -> pd.DataFrame:
    ordered = raw.copy()
    ordered["_season_sort"] = pd.to_numeric(ordered["season"], errors="coerce")
    ordered["_week_sort"] = pd.to_numeric(ordered["week"], errors="coerce")
    ordered["_gameday_sort"] = pd.to_datetime(
        ordered["gameday"].astype(str).str.strip(),
        format="%Y-%m-%d",
        errors="coerce",
    )
    ordered["_gametime_sort"] = pd.to_timedelta(
        ordered["gametime"].astype(str).str.strip() + ":00",
        errors="coerce",
    )

    for column in [
        "_season_sort", "_week_sort", "_gameday_sort", "_gametime_sort"
    ]:
        if ordered[column].isna().any():
            examples = ordered.loc[
                ordered[column].isna(),
                ["game_id", "season", "week", "gameday", "gametime"],
            ].head(10)
            fail(
                f"Could not parse chronological field {column}. "
                f"Examples:\n{examples.to_string(index=False)}"
            )

    return ordered.sort_values(
        [
            "_season_sort", "_week_sort", "_gameday_sort",
            "_gametime_sort", "game_id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)


def validate_results(raw: pd.DataFrame) -> None:
    for column in ["margin", "total_points", "spread_line", "total_line", "home_win"]:
        numeric = pd.to_numeric(raw[column], errors="coerce")
        if numeric.isna().any():
            fail(
                f"{column}: contains {int(numeric.isna().sum())} "
                "blank/non-numeric rows"
            )

    ats = set(raw["home_ats_result"].astype(str).str.strip())
    invalid_ats = sorted(ats - {"WIN", "LOSS", "PUSH"})
    if invalid_ats:
        fail(f"Unexpected home_ats_result values: {invalid_ats}")

    totals = set(raw["total_result"].astype(str).str.strip())
    invalid_totals = sorted(totals - {"OVER", "UNDER", "PUSH"})
    if invalid_totals:
        fail(f"Unexpected total_result values: {invalid_totals}")


def train_regressor(
    X: pd.DataFrame,
    y: pd.Series,
    cat_indices: list[int],
    model_params: dict[str, Any],
) -> CatBoostRegressor:
    pool = Pool(
        X,
        label=y,
        cat_features=cat_indices,
        feature_names=list(X.columns),
    )
    model = CatBoostRegressor(**model_params)
    model.fit(pool, verbose=False)
    return model


def predict_regressor(
    model: CatBoostRegressor,
    X: pd.DataFrame,
    cat_indices: list[int],
) -> np.ndarray:
    pool = Pool(
        X,
        cat_features=cat_indices,
        feature_names=list(X.columns),
    )
    return np.asarray(model.predict(pool), dtype=float)


def predicted_ml_winner(margin: float) -> str:
    return "HOME" if margin > 0 else "AWAY" if margin < 0 else "TIE"


def predicted_ats_side(margin: float, spread_line: float) -> str:
    edge = margin - spread_line
    return "HOME" if edge > 1e-12 else "AWAY" if edge < -1e-12 else "PUSH"


def predicted_ou(total: float, total_line: float) -> str:
    edge = total - total_line
    return "OVER" if edge > 1e-12 else "UNDER" if edge < -1e-12 else "PUSH"


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temp, index=False)
    temp.replace(path)


def sort_output(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df[OUTPUT_COLUMNS].copy()

    output = df[OUTPUT_COLUMNS].copy()
    output["_season"] = pd.to_numeric(output["season"], errors="raise")
    output["_week"] = pd.to_numeric(output["week"], errors="raise")
    output["_date"] = pd.to_datetime(output["gameday"], errors="raise")
    output["_time"] = pd.to_timedelta(
        output["gametime"].astype(str) + ":00",
        errors="raise",
    )
    return (
        output.sort_values(
            ["_season", "_week", "_date", "_time", "game_id"],
            kind="mergesort",
        )
        .drop(columns=["_season", "_week", "_date", "_time"])
        .reset_index(drop=True)
    )


def load_existing_checkpoint(
    expected_game_ids: set[str],
    backtest_seasons: list[int],
) -> pd.DataFrame:
    source = (
        CHECKPOINT_PATH
        if CHECKPOINT_PATH.exists()
        else OUTPUT_PATH
        if OUTPUT_PATH.exists()
        else None
    )
    if source is None:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    existing = pd.read_csv(source, dtype={"game_id": str})
    missing = [column for column in OUTPUT_COLUMNS if column not in existing.columns]
    if missing:
        fail(f"{source.name}: missing expected columns: {missing}")

    existing = existing[OUTPUT_COLUMNS].copy()
    existing["game_id"] = existing["game_id"].astype(str).str.strip()
    if existing["game_id"].duplicated().any():
        fail(f"{source.name}: duplicate game_id values")

    unknown = sorted(set(existing["game_id"]) - expected_game_ids)
    if unknown:
        fail(
            f"{source.name}: contains games outside the expected "
            f"{backtest_seasons[0]}-{backtest_seasons[-1]} backtest: {unknown[:10]}"
        )
    return existing


def make_result_rows(
    holdout: pd.DataFrame,
    predicted_margin: np.ndarray,
    predicted_total: np.ndarray,
    training_rows: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position, (_, row) in enumerate(holdout.iterrows()):
        margin = float(predicted_margin[position])
        total = float(predicted_total[position])
        spread_line = float(row["spread_line"])
        total_line = float(row["total_line"])

        rows.append(
            {
                "season": int(float(row["season"])),
                "week": int(float(row["week"])),
                "gameday": str(row["gameday"]).strip(),
                "gametime": str(row["gametime"]).strip(),
                "game_id": str(row["game_id"]).strip(),
                "away_team": str(row["away_team"]).strip(),
                "home_team": str(row["home_team"]).strip(),
                "training_rows": int(training_rows),
                "spread_line": spread_line,
                "total_line": total_line,
                "predicted_margin": margin,
                "actual_margin": float(row["margin"]),
                "predicted_total": total,
                "actual_total_points": float(row["total_points"]),
                "predicted_home_score": (total + margin) / 2.0,
                "predicted_away_score": (total - margin) / 2.0,
                "predicted_ml_winner": predicted_ml_winner(margin),
                "actual_home_win": int(float(row["home_win"])),
                "predicted_ats_side": predicted_ats_side(margin, spread_line),
                "actual_home_ats_result": str(row["home_ats_result"]).strip(),
                "predicted_ou": predicted_ou(total, total_line),
                "actual_total_result": str(row["total_result"]).strip(),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing backtest/checkpoint and rebuild from scratch.",
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=None,
        help="Optional smoke-test limit on new held-out kickoff groups.",
    )
    args = parser.parse_args()

    if args.max_groups is not None and args.max_groups <= 0:
        fail("--max-groups must be greater than zero")

    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    if args.reset:
        for path in [CHECKPOINT_PATH, OUTPUT_PATH]:
            if path.exists():
                path.unlink()

    schema = load_schema()
    seasons = [int(value) for value in schema["training_seasons"]]
    initial_train_season = seasons[0]
    backtest_seasons = seasons[1:]

    raw = read_inputs(schema, seasons)
    validate_results(raw)
    ordered = build_chronology(raw)

    X = prepare_feature_matrix(ordered, schema)
    y_margin = pd.to_numeric(ordered["margin"], errors="raise").astype(float)
    y_total = pd.to_numeric(ordered["total_points"], errors="raise").astype(float)
    season_numeric = pd.to_numeric(ordered["season"], errors="raise").astype(int)

    backtest_mask = season_numeric.isin(backtest_seasons)
    expected_game_ids = set(
        ordered.loc[backtest_mask, "game_id"].astype(str).str.strip()
    )
    initial_training_rows = int((season_numeric == initial_train_season).sum())
    if initial_training_rows <= 0:
        fail(f"No {initial_train_season} rows available for initial training")

    existing = load_existing_checkpoint(expected_game_ids, backtest_seasons)
    completed_ids = set(existing["game_id"].astype(str).str.strip())
    if completed_ids == expected_game_ids and args.max_groups is None:
        final = sort_output(existing)
        atomic_write_csv(final, OUTPUT_PATH)
        if CHECKPOINT_PATH.exists():
            CHECKPOINT_PATH.unlink()
        print(f"Step 13 already complete: rows={len(final)} -> {OUTPUT_PATH}")
        return 0

    heldout = ordered.loc[backtest_mask].copy()
    groups = list(
        heldout.groupby(
            ["season", "week", "gameday", "gametime"],
            sort=False,
            dropna=False,
        )
    )

    print(
        f"Historical rows={len(ordered)}, "
        f"initial_{initial_train_season}_training_rows={initial_training_rows}, "
        f"heldout_games={len(heldout)}, kickoff_groups={len(groups)}, "
        f"features={len(schema['feature_order'])}"
    )

    accumulated = existing.copy()
    processed_new_groups = 0
    cat_indices = list(schema["categorical_feature_indices"])
    model_params = dict(schema["model_params"])

    for group_number, (group_key, group_df) in enumerate(groups, start=1):
        group_ids = set(group_df["game_id"].astype(str).str.strip())
        if group_ids.issubset(completed_ids):
            continue
        if args.max_groups is not None and processed_new_groups >= args.max_groups:
            break

        group_positions = group_df.index.to_numpy(dtype=int)
        first_position = int(group_positions.min())
        train_positions = np.arange(first_position, dtype=int)

        if len(train_positions) < initial_training_rows:
            fail(
                f"Held-out group {group_key} has only {len(train_positions)} "
                f"prior rows; expected at least all {initial_train_season} rows"
            )

        margin_model = train_regressor(
            X.iloc[train_positions],
            y_margin.iloc[train_positions],
            cat_indices,
            model_params,
        )
        total_model = train_regressor(
            X.iloc[train_positions],
            y_total.iloc[train_positions],
            cat_indices,
            model_params,
        )

        predicted_margin = predict_regressor(
            margin_model, X.loc[group_positions], cat_indices
        )
        predicted_total = predict_regressor(
            total_model, X.loc[group_positions], cat_indices
        )

        new_frame = pd.DataFrame(
            make_result_rows(
                ordered.loc[group_positions],
                predicted_margin,
                predicted_total,
                len(train_positions),
            ),
            columns=OUTPUT_COLUMNS,
        )

        if not accumulated.empty:
            accumulated = accumulated.loc[
                ~accumulated["game_id"].astype(str).isin(group_ids)
            ].copy()
        accumulated = pd.concat([accumulated, new_frame], ignore_index=True)
        accumulated = sort_output(accumulated)
        atomic_write_csv(accumulated, CHECKPOINT_PATH)

        completed_ids.update(group_ids)
        processed_new_groups += 1
        if processed_new_groups == 1 or processed_new_groups % 10 == 0:
            season, week, gameday, gametime = group_key
            print(
                f"Completed new_group={processed_new_groups} "
                f"overall_group={group_number}/{len(groups)} "
                f"{season} W{week} {gameday} {gametime} "
                f"games={len(group_df)} train_rows={len(train_positions)} "
                f"completed_games={len(completed_ids)}/{len(expected_game_ids)}"
            )

    if args.max_groups is not None:
        print(
            f"Partial run complete: new_groups={processed_new_groups}, "
            f"completed_games={len(completed_ids)}/{len(expected_game_ids)}. "
            f"Checkpoint: {CHECKPOINT_PATH}"
        )
        return 0

    missing_ids = expected_game_ids - completed_ids
    if missing_ids:
        fail(
            f"Backtest ended with {len(missing_ids)} games missing. "
            f"Examples: {sorted(missing_ids)[:10]}"
        )

    final = sort_output(accumulated)
    if len(final) != len(expected_game_ids):
        fail(
            f"Final row count mismatch: expected {len(expected_game_ids)}, "
            f"got {len(final)}"
        )
    if final["game_id"].duplicated().any():
        fail("Duplicate game_id detected in final Step 13 output")

    atomic_write_csv(final, OUTPUT_PATH)
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    print(
        f"Step 13 complete: seasons={backtest_seasons[0]}-{backtest_seasons[-1]} "
        f"heldout_rows={len(final)} -> {OUTPUT_PATH}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
