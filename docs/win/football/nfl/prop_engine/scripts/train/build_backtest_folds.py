#!/usr/bin/env python3
"""
Build chronological expanding-window backtest folds for the NFL Prop Engine.

READS:
    docs/win/football/nfl/prop_engine/config/prop_engine.yaml
    docs/win/football/nfl/prop_engine/data/historical/features/player_game_features.parquet

WRITES:
    docs/win/football/nfl/prop_engine/evaluation/backtest_folds.parquet

POLICY:
    - No random split.
    - Expanding annual walk-forward validation folds.
    - Final development fold trains through the season before the development
      validation season and validates on that development validation season.
    - Final untouched test fold trains through the season before the configured
      historical_end and evaluates historical_end with test_flag=1.
    - All training seasons precede the validation/test season, so no training
      game can occur after a validation/test game's kickoff.
    - The configured historical_end is the untouched test season.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import common


OUTPUT_COLUMNS = [
    "fold_id",
    "train_start_season",
    "train_start_week",
    "train_end_season",
    "train_end_week",
    "validation_start_season",
    "validation_start_week",
    "validation_end_season",
    "validation_end_week",
    "test_flag",
]


def season_bounds(features: pd.DataFrame) -> pd.DataFrame:
    common.require_columns(
        features,
        ["season", "week", "game_id", "kickoff_timestamp"],
        "historical feature table",
    )

    frame = features[
        ["season", "week", "game_id", "kickoff_timestamp"]
    ].drop_duplicates(
        ["season", "week", "game_id"]
    ).copy()

    frame["season"] = pd.to_numeric(
        frame["season"],
        errors="raise",
    ).astype(int)

    frame["week"] = pd.to_numeric(
        frame["week"],
        errors="raise",
    ).astype(int)

    frame["kickoff_timestamp"] = pd.to_datetime(
        frame["kickoff_timestamp"],
        errors="raise",
        utc=True,
    )

    bounds = (
        frame.groupby("season", as_index=False)
        .agg(
            min_week=("week", "min"),
            max_week=("week", "max"),
            min_kickoff=("kickoff_timestamp", "min"),
            max_kickoff=("kickoff_timestamp", "max"),
            games=("game_id", "nunique"),
        )
        .sort_values("season", kind="mergesort")
        .reset_index(drop=True)
    )

    if bounds.empty:
        raise ValueError("Historical feature table contains no season data.")

    if bounds["games"].le(0).any():
        raise ValueError("At least one historical season contains no games.")

    return bounds


def validate_season_sequence(
    bounds: pd.DataFrame,
    historical_start: int,
    historical_end: int,
) -> None:
    expected = list(range(historical_start, historical_end + 1))
    actual = bounds.loc[
        bounds["season"].between(historical_start, historical_end),
        "season",
    ].astype(int).tolist()

    if actual != expected:
        raise ValueError(
            "Historical feature table does not contain a complete configured "
            f"season sequence. expected={expected}, actual={actual}"
        )

    relevant = bounds[
        bounds["season"].between(historical_start, historical_end)
    ].sort_values("season", kind="mergesort")

    previous_max = None
    previous_season = None

    for row in relevant.itertuples(index=False):
        if previous_max is not None and not previous_max < row.min_kickoff:
            raise ValueError(
                "Season kickoff ranges overlap or are nonchronological: "
                f"{previous_season} max={previous_max}, "
                f"{row.season} min={row.min_kickoff}"
            )

        previous_max = row.max_kickoff
        previous_season = int(row.season)


def build_folds(
    bounds: pd.DataFrame,
    historical_start: int,
    historical_end: int,
) -> pd.DataFrame:
    if historical_end <= historical_start:
        raise ValueError(
            "historical_end must be later than historical_start."
        )

    # The configured final historical season is reserved as the untouched test.
    test_season = historical_end
    development_validation_season = test_season - 1

    if development_validation_season <= historical_start:
        raise ValueError(
            "At least two post-start seasons are required for development "
            "validation and untouched testing."
        )

    indexed = bounds.set_index("season")

    records: list[dict] = []

    # Expanding annual walk-forward folds inside development.
    # First possible validation season is historical_start + 1.
    for validation_season in range(
        historical_start + 1,
        development_validation_season + 1,
    ):
        train_end_season = validation_season - 1

        train_start = indexed.loc[historical_start]
        train_end = indexed.loc[train_end_season]
        validation = indexed.loc[validation_season]

        if not train_end["max_kickoff"] < validation["min_kickoff"]:
            raise ValueError(
                "Temporal leakage detected while constructing development "
                f"fold for validation season {validation_season}."
            )

        records.append(
            {
                "fold_id": f"dev_{validation_season}",
                "train_start_season": historical_start,
                "train_start_week": int(train_start["min_week"]),
                "train_end_season": train_end_season,
                "train_end_week": int(train_end["max_week"]),
                "validation_start_season": validation_season,
                "validation_start_week": int(validation["min_week"]),
                "validation_end_season": validation_season,
                "validation_end_week": int(validation["max_week"]),
                "test_flag": 0,
            }
        )

    # Final untouched historical test. This row is reporting-only downstream.
    train_start = indexed.loc[historical_start]
    test_train_end = indexed.loc[test_season - 1]
    test = indexed.loc[test_season]

    if not test_train_end["max_kickoff"] < test["min_kickoff"]:
        raise ValueError(
            "Temporal leakage detected while constructing untouched test fold."
        )

    records.append(
        {
            "fold_id": f"test_{test_season}",
            "train_start_season": historical_start,
            "train_start_week": int(train_start["min_week"]),
            "train_end_season": test_season - 1,
            "train_end_week": int(test_train_end["max_week"]),
            "validation_start_season": test_season,
            "validation_start_week": int(test["min_week"]),
            "validation_end_season": test_season,
            "validation_end_week": int(test["max_week"]),
            "test_flag": 1,
        }
    )

    result = pd.DataFrame.from_records(
        records,
        columns=OUTPUT_COLUMNS,
    )

    return result


def validate_output(
    folds: pd.DataFrame,
    *,
    historical_start: int,
    historical_end: int,
) -> None:
    if list(folds.columns) != OUTPUT_COLUMNS:
        raise RuntimeError("Backtest-fold output column order mismatch.")

    if folds.empty:
        raise ValueError("No backtest folds were produced.")

    common.ensure_unique(
        folds,
        ["fold_id"],
        "backtest fold IDs",
    )

    test_rows = folds[
        pd.to_numeric(folds["test_flag"], errors="raise").eq(1)
    ]

    if len(test_rows) != 1:
        raise ValueError(
            f"Expected exactly one untouched test fold; found {len(test_rows)}."
        )

    test = test_rows.iloc[0]

    if int(test["validation_start_season"]) != historical_end:
        raise ValueError(
            "Untouched test season must equal configured historical_end."
        )

    if int(test["validation_end_season"]) != historical_end:
        raise ValueError(
            "Untouched test fold must remain within historical_end."
        )

    if int(test["train_end_season"]) != historical_end - 1:
        raise ValueError(
            "Untouched test fold must train only through the prior season."
        )

    dev = folds[
        pd.to_numeric(folds["test_flag"], errors="raise").eq(0)
    ].copy()

    if dev.empty:
        raise ValueError("No development walk-forward folds were produced.")

    if dev["validation_start_season"].max() != historical_end - 1:
        raise ValueError(
            "Final development validation season must be the season before "
            "the untouched test season."
        )

    final_dev = dev.loc[
        dev["validation_start_season"].idxmax()
    ]

    if int(final_dev["train_end_season"]) != historical_end - 2:
        raise ValueError(
            "Final development fold must train through two seasons before "
            "historical_end."
        )

    if int(final_dev["validation_start_season"]) != historical_end - 1:
        raise ValueError(
            "Final development fold must validate on the season before "
            "historical_end."
        )

    if not (folds["train_start_season"] == historical_start).all():
        raise ValueError(
            "Expanding folds must retain the configured historical_start."
        )

    if not (
        folds["train_end_season"]
        < folds["validation_start_season"]
    ).all():
        raise ValueError(
            "Every fold must end training before validation begins."
        )

    if not (
        folds["validation_start_season"]
        == folds["validation_end_season"]
    ).all():
        raise ValueError(
            "Issue 20 annual folds must validate within one season."
        )

    if not set(folds["test_flag"].tolist()).issubset({0, 1}):
        raise ValueError("test_flag must contain only 0/1.")


def main() -> int:
    # ISSUE28_MARKET_EXCLUSION_PREFLIGHT
    _issue28_audit = common.prop_root() / "scripts" / "validate" / "audit_market_exclusion.py"
    _issue28_result = __import__("subprocess").run(
        [__import__("sys").executable, str(_issue28_audit), "--preflight"],
        check=False,
    )
    if _issue28_result.returncode != 0:
        raise RuntimeError("Issue 28 market-exclusion preflight failed.")

    config = common.load_config()

    historical_start = int(
        config["seasons"]["historical_start"]
    )
    historical_end = int(
        config["seasons"]["historical_end"]
    )

    # _CONFIG_ENFORCED_BACKTEST_POLICY
    training = config["training"]
    configured_policy = {
        "model_selection_train_end_season": int(
            training["model_selection_train_end_season"]
        ),
        "development_validation_season": int(
            training["development_validation_season"]
        ),
        "final_train_end_season": int(
            training["final_train_end_season"]
        ),
        "untouched_test_season": int(
            training["untouched_test_season"]
        ),
    }
    derived_policy = {
        "model_selection_train_end_season": historical_end - 2,
        "development_validation_season": historical_end - 1,
        "final_train_end_season": historical_end - 1,
        "untouched_test_season": historical_end,
    }
    if configured_policy != derived_policy:
        raise ValueError(
            "Configured training split boundaries disagree with the "
            "chronological annual-fold contract. "
            f"configured={configured_policy}, derived={derived_policy}"
        )

    feature_path = config["paths"]["historical_features"]
    output_path = (
        "docs/win/football/nfl/prop_engine/evaluation/"
        "backtest_folds.parquet"
    )

    features = common.read_parquet_required(
        feature_path,
        ["season", "week", "game_id", "kickoff_timestamp"],
    )

    bounds = season_bounds(features)

    validate_season_sequence(
        bounds,
        historical_start,
        historical_end,
    )

    folds = build_folds(
        bounds,
        historical_start,
        historical_end,
    )

    validate_output(
        folds,
        historical_start=historical_start,
        historical_end=historical_end,
    )

    common.write_parquet_atomic(
        folds,
        output_path,
    )

    development = folds[folds["test_flag"].eq(0)]
    final_dev = development.loc[
        development["validation_start_season"].idxmax()
    ]
    test = folds.loc[folds["test_flag"].eq(1)].iloc[0]

    payload = {
        "status": "passed",
        "output": output_path,
        "rows": int(len(folds)),
        "development_folds": int(len(development)),
        "test_folds": 1,
        "historical_start": historical_start,
        "historical_end": historical_end,
        "final_development_train_end_season": int(
            final_dev["train_end_season"]
        ),
        "final_development_validation_season": int(
            final_dev["validation_start_season"]
        ),
        "untouched_test_train_end_season": int(
            test["train_end_season"]
        ),
        "untouched_test_season": int(
            test["validation_start_season"]
        ),
        "random_split_used": False,
        "expanding_window": True,
        "test_reporting_only": True,
    }

    common.log_run(
        "build_backtest_folds.py",
        payload,
    )

    print(
        json.dumps(
            {
                "script": Path(__file__).name,
                "payload": payload,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
