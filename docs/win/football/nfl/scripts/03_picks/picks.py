#!/usr/bin/env python3
"""
NFL selection layer.

READS:
  docs/win/football/nfl/02_select/*NFL_selected.csv
  docs/win/football/nfl/config/markets.yaml

WRITES:
  docs/win/football/nfl/03_picks/*NFL_picks.csv

Behavior:
- Preserves every input column.
- Does not modify raw candidate columns.
- Evaluates every enabled side against selection_defaults plus all configured bands.
- Applies spread.max_spread_abs and total.min_total/max_total.
- If multiple sides qualify in one market, uses pick_preference:
    best_ev, best_prob, or best_kelly.
- Overwrites only the existing final selection columns.
- Selection-time Kelly is min(full_kelly, resolved max_kelly), so markets.yaml
  controls Kelly independently of any upstream candidate Kelly cap.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
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

DEFAULT_INPUT_DIR = NFL_ROOT / "02_select"
DEFAULT_MARKETS_PATH = NFL_ROOT / "config/markets.yaml"
DEFAULT_OUTPUT_DIR = NFL_ROOT / "03_picks"
DEFAULT_PATTERN = "*NFL_selected.csv"

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

THRESHOLD_KEYS = {
    "min_ev",
    "min_edge",
    "min_kelly",
    "max_kelly",
    "min_odds_american",
    "max_odds_american",
    "min_model_prob",
    "max_model_prob",
}

BAND_TO_METRIC = {
    "odds_bands": "odds_american",
    "edge_bands": "edge",
    "ev_bands": "ev",
    "kelly_bands": "kelly",
    "prob_bands": "model_probability",
    "line_bands": "line",
}

PICK_METRIC = {
    "best_ev": "ev",
    "best_prob": "model_probability",
    "best_kelly": "kelly",
}

MARKETS = {
    "moneyline": {
        "output_prefix": "ml",
        "sides": {
            "home": ("ml_home", "HOME"),
            "away": ("ml_away", "AWAY"),
        },
        "market_extras": set(),
        "side_bands": {
            "odds_bands",
            "edge_bands",
            "ev_bands",
            "kelly_bands",
            "prob_bands",
        },
    },
    "spread": {
        "output_prefix": "spread",
        "sides": {
            "home": ("spread_home", "HOME"),
            "away": ("spread_away", "AWAY"),
        },
        "market_extras": {
            "max_spread_abs",
        },
        "side_bands": set(BAND_TO_METRIC),
    },
    "total": {
        "output_prefix": "total",
        "sides": {
            "over": ("total_over", "OVER"),
            "under": ("total_under", "UNDER"),
        },
        "market_extras": {
            "min_total",
            "max_total",
        },
        "side_bands": set(BAND_TO_METRIC),
    },
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


def number(
    value: Any,
    label: str,
) -> float:
    text = clean(value)

    if not text:
        fail(
            f"{label} is required"
        )

    try:
        result = float(text)
    except (TypeError, ValueError):
        fail(
            f"{label} must be numeric; "
            f"found {value!r}"
        )

    if not math.isfinite(result):
        fail(
            f"{label} must be finite; "
            f"found {value!r}"
        )

    return result


def optional_number(
    value: Any,
) -> float | None:
    text = clean(value)

    if not text:
        return None

    try:
        result = float(text)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(result):
        return None

    return result


def boolean(
    value: Any,
    label: str,
) -> bool:
    if isinstance(value, bool):
        return value

    if (
        isinstance(
            value,
            (int, np.integer),
        )
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
        f"{label} must be true/false; "
        f"found {value!r}"
    )


def require_mapping(
    value: Any,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(
            f"{label} must be a YAML mapping"
        )

    return value


def reject_unknown(
    mapping: dict[str, Any],
    allowed: set[str],
    label: str,
) -> None:
    unknown = sorted(
        set(mapping)
        - allowed
    )

    if unknown:
        fail(
            f"{label} contains unsupported "
            f"keys: {unknown}"
        )



def load_yaml(
    path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        fail(
            f"Missing markets config: {path}"
        )

    class UniqueKeyLoader(
        yaml.SafeLoader
    ):
        pass

    def construct_mapping(
        loader: Any,
        node: Any,
        deep: bool = False,
    ) -> dict[Any, Any]:
        if not isinstance(
            node,
            yaml.MappingNode,
        ):
            fail(
                f"markets.yaml expected a "
                f"mapping node: {path}"
            )

        mapping: dict[Any, Any] = {}

        for key_node, value_node in node.value:
            key = loader.construct_object(
                key_node,
                deep=deep,
            )

            try:
                duplicate = key in mapping
            except TypeError:
                fail(
                    "markets.yaml contains "
                    "an unhashable mapping key"
                )

            if duplicate:
                fail(
                    "markets.yaml contains "
                    f"duplicate key {key!r} "
                    f"at line {key_node.start_mark.line + 1}"
                )

            mapping[key] = (
                loader.construct_object(
                    value_node,
                    deep=deep,
                )
            )

        return mapping

    UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping,
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = yaml.load(
            handle,
            Loader=UniqueKeyLoader,
        )

    return require_mapping(
        data,
        "markets.yaml",
    )



def validate_csv_header(
    path: Path,
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
            f"Input is not valid UTF-8 CSV: "
            f"{path}: {exc}"
        )

    if not header:
        fail(
            f"Input has no CSV header: {path}"
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
            f"Input contains blank column "
            f"names: {path}"
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
            f"Input contains duplicate "
            f"column names: {duplicates}"
        )


def load_csv(
    path: Path,
) -> pd.DataFrame:
    if not path.is_file():
        fail(
            f"Missing input file: {path}"
        )

    validate_csv_header(
        path
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
        fail(
            f"Input contains no rows: {path}"
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


def validate_thresholds(
    values: dict[str, float],
    label: str,
) -> None:
    if (
        values["min_kelly"]
        > values["max_kelly"]
    ):
        fail(
            f"{label}: min_kelly cannot "
            "exceed max_kelly"
        )

    if (
        values["min_odds_american"]
        > values["max_odds_american"]
    ):
        fail(
            f"{label}: min_odds_american "
            "cannot exceed max_odds_american"
        )

    if (
        values["min_model_prob"]
        > values["max_model_prob"]
    ):
        fail(
            f"{label}: min_model_prob cannot "
            "exceed max_model_prob"
        )

    if (
        values["min_kelly"] < 0
        or values["max_kelly"] < 0
    ):
        fail(
            f"{label}: Kelly limits "
            "cannot be negative"
        )

    if not (
        0
        <= values["min_model_prob"]
        <= 1
    ):
        fail(
            f"{label}: min_model_prob "
            "must be in [0,1]"
        )

    if not (
        0
        <= values["max_model_prob"]
        <= 1
    ):
        fail(
            f"{label}: max_model_prob "
            "must be in [0,1]"
        )


def thresholds(
    mapping: dict[str, Any],
    label: str,
    base: dict[str, float] | None = None,
    require_all: bool = False,
) -> dict[str, float]:
    result = dict(
        base or {}
    )

    if require_all:
        missing = sorted(
            THRESHOLD_KEYS
            - set(mapping)
        )

        if missing:
            fail(
                f"{label} missing required "
                f"keys: {missing}"
            )

    for key in THRESHOLD_KEYS:
        if key in mapping:
            result[key] = number(
                mapping[key],
                f"{label}.{key}",
            )

    missing = sorted(
        THRESHOLD_KEYS
        - set(result)
    )

    if missing:
        fail(
            f"{label} missing threshold "
            f"values: {missing}"
        )

    validate_thresholds(
        result,
        label,
    )

    return result


def bands(
    value: Any,
    label: str,
) -> list[tuple[float, float]]:
    if (
        not isinstance(value, list)
        or not value
    ):
        fail(
            f"{label} must be a non-empty "
            "list of [min, max] bands"
        )

    result: list[
        tuple[float, float]
    ] = []

    for index, item in enumerate(value):
        if (
            not isinstance(
                item,
                (list, tuple),
            )
            or len(item) != 2
        ):
            fail(
                f"{label}[{index}] "
                "must be [min, max]"
            )

        low = number(
            item[0],
            f"{label}[{index}][0]",
        )

        high = number(
            item[1],
            f"{label}[{index}][1]",
        )

        if low > high:
            fail(
                f"{label}[{index}] has "
                "min greater than max"
            )

        result.append(
            (
                low,
                high,
            )
        )

    return result




def validate_band_domain(
    band_name: str,
    configured: list[
        tuple[float, float]
    ],
    label: str,
) -> None:
    if band_name == "prob_bands":
        for low, high in configured:
            if (
                low < 0.0
                or high > 1.0
            ):
                fail(
                    f"{label}: probability "
                    "bands must stay within "
                    "[0,1]"
                )

    if band_name == "kelly_bands":
        for low, high in configured:
            if (
                low < 0.0
                or high < 0.0
            ):
                fail(
                    f"{label}: Kelly bands "
                    "cannot be negative"
                )


def matches_band(
    value: float,
    configured: list[
        tuple[float, float]
    ],
) -> bool:
    return any(
        low <= value <= high
        for low, high
        in configured
    )


def normalize_config(
    raw: dict[str, Any],
) -> dict[str, Any]:
    reject_unknown(
        raw,
        {
            "selection_defaults",
            "markets",
        },
        "markets.yaml",
    )

    defaults_raw = require_mapping(
        raw.get(
            "selection_defaults"
        ),
        "markets.yaml.selection_defaults",
    )

    reject_unknown(
        defaults_raw,
        THRESHOLD_KEYS,
        "markets.yaml.selection_defaults",
    )

    defaults = thresholds(
        defaults_raw,
        "markets.yaml.selection_defaults",
        require_all=True,
    )

    markets_raw = require_mapping(
        raw.get("markets"),
        "markets.yaml.markets",
    )

    reject_unknown(
        markets_raw,
        set(MARKETS),
        "markets.yaml.markets",
    )

    output: dict[str, Any] = {
        "selection_defaults": defaults,
        "markets": {},
    }

    for (
        market_name,
        spec,
    ) in MARKETS.items():
        market_label = (
            "markets.yaml.markets."
            f"{market_name}"
        )

        market_raw = require_mapping(
            markets_raw.get(
                market_name
            ),
            market_label,
        )

        allowed_market = (
            {
                "enabled",
                "pick_preference",
            }
            | THRESHOLD_KEYS
            | set(spec["sides"])
            | set(
                spec["market_extras"]
            )
        )

        reject_unknown(
            market_raw,
            allowed_market,
            market_label,
        )

        preference = clean(
            market_raw.get(
                "pick_preference",
                "best_prob",
            )
        ).casefold()

        if preference not in PICK_METRIC:
            fail(
                f"{market_label}."
                "pick_preference must be "
                f"one of {sorted(PICK_METRIC)}"
            )

        market_thresholds = thresholds(
            market_raw,
            market_label,
            base=defaults,
        )

        normalized: dict[str, Any] = {
            "enabled": boolean(
                market_raw.get(
                    "enabled",
                    True,
                ),
                f"{market_label}.enabled",
            ),
            "pick_preference": (
                preference
            ),
            "thresholds": (
                market_thresholds
            ),
            "sides": {},
        }

        if market_name == "spread":
            value = number(
                market_raw.get(
                    "max_spread_abs",
                    100.0,
                ),
                (
                    f"{market_label}."
                    "max_spread_abs"
                ),
            )

            if value < 0:
                fail(
                    f"{market_label}."
                    "max_spread_abs "
                    "cannot be negative"
                )

            normalized[
                "max_spread_abs"
            ] = value

        if market_name == "total":
            min_total = number(
                market_raw.get(
                    "min_total",
                    0.0,
                ),
                f"{market_label}.min_total",
            )

            max_total = number(
                market_raw.get(
                    "max_total",
                    100.0,
                ),
                f"{market_label}.max_total",
            )

            if min_total > max_total:
                fail(
                    f"{market_label}."
                    "min_total cannot exceed "
                    "max_total"
                )

            normalized[
                "min_total"
            ] = min_total

            normalized[
                "max_total"
            ] = max_total

        for side_name in spec["sides"]:
            side_label = (
                f"{market_label}."
                f"{side_name}"
            )

            side_raw = require_mapping(
                market_raw.get(
                    side_name
                ),
                side_label,
            )

            allowed_side = (
                {"enabled"}
                | THRESHOLD_KEYS
                | set(
                    spec["side_bands"]
                )
            )

            reject_unknown(
                side_raw,
                allowed_side,
                side_label,
            )

            side_thresholds = (
                thresholds(
                    side_raw,
                    side_label,
                    base=market_thresholds,
                )
            )

            side_bands: dict[
                str,
                list[
                    tuple[
                        float,
                        float,
                    ]
                ],
            ] = {}

            for key in spec[
                "side_bands"
            ]:
                if key in side_raw:
                    configured_bands = bands(
                        side_raw[key],
                        f"{side_label}.{key}",
                    )
                    validate_band_domain(
                        key,
                        configured_bands,
                        f"{side_label}.{key}",
                    )
                    side_bands[key] = configured_bands

            normalized[
                "sides"
            ][side_name] = {
                "enabled": boolean(
                    side_raw.get(
                        "enabled",
                        True,
                    ),
                    (
                        f"{side_label}."
                        "enabled"
                    ),
                ),
                "thresholds": (
                    side_thresholds
                ),
                "bands": (
                    side_bands
                ),
            }

        output[
            "markets"
        ][market_name] = normalized

    return output


def selection_columns() -> list[str]:
    return [
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


def candidate_columns() -> list[str]:
    result: list[str] = []

    for (
        market_name,
        spec,
    ) in MARKETS.items():
        for (
            prefix,
            _,
        ) in spec[
            "sides"
        ].values():
            result.extend(
                [
                    f"{prefix}_available",
                    f"{prefix}_odds_american",
                    f"{prefix}_model_probability",
                    f"{prefix}_implied_probability",
                    f"{prefix}_edge",
                    f"{prefix}_ev",
                    f"{prefix}_full_kelly",
                    f"{prefix}_kelly",
                ]
            )

            if market_name in {
                "spread",
                "total",
            }:
                result.append(
                    f"{prefix}_line"
                )

    return result


def validate_input(
    df: pd.DataFrame,
    path: Path,
) -> None:
    require_columns(
        df,
        [
            "game_id",
            *selection_columns(),
            *candidate_columns(),
        ],
        str(path),
    )

    game_ids = df[
        "game_id"
    ].map(clean)

    if game_ids.eq("").any():
        fail(
            f"{path} contains "
            "blank game_id values"
        )

    if game_ids.duplicated().any():
        examples = game_ids[
            game_ids.duplicated(False)
        ].head(10).tolist()

        fail(
            f"{path} contains duplicate "
            f"game_id values: {examples}"
        )




def american_to_decimal(
    odds: float,
) -> float:
    if odds == 0:
        fail(
            "American odds cannot be 0"
        )

    if odds > 0:
        return (
            1.0
            + odds / 100.0
        )

    return (
        1.0
        + 100.0 / abs(odds)
    )


def american_implied_probability(
    odds: float,
) -> float:
    return (
        1.0
        / american_to_decimal(
            odds
        )
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
        not math.isfinite(
            total_raw
        )
        or total_raw <= 0
    ):
        fail(
            "Unable to calculate no-vig "
            "probabilities from odds "
            f"{first_odds!r}, "
            f"{second_odds!r}"
        )

    return (
        first_raw / total_raw,
        second_raw / total_raw,
    )


def candidate_metrics(
    model_probability: float,
    odds_american: float,
    fair_market_probability: float,
) -> dict[str, float]:
    decimal_odds = (
        american_to_decimal(
            odds_american
        )
    )

    net_win = (
        decimal_odds
        - 1.0
    )

    loss_probability = (
        1.0
        - model_probability
    )

    edge = (
        model_probability
        - fair_market_probability
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

    return {
        "implied_probability": (
            fair_market_probability
        ),
        "edge": edge,
        "ev": ev,
        "full_kelly": max(
            0.0,
            raw_kelly,
        ),
    }


def require_close(
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
            f"{label}: expected "
            f"{expected!r}; "
            f"found {actual!r}"
        )


def validate_candidate_contract(
    df: pd.DataFrame,
    path: Path,
) -> None:
    for _, row in df.iterrows():
        game_id = clean(
            row["game_id"]
        )

        for (
            market_name,
            first_prefix,
            second_prefix,
            include_line,
        ) in (
            (
                "moneyline",
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
            first_available = (
                optional_number(
                    row.get(
                        f"{first_prefix}_available",
                        "",
                    )
                )
            )
            second_available = (
                optional_number(
                    row.get(
                        f"{second_prefix}_available",
                        "",
                    )
                )
            )

            if first_available not in {
                0.0,
                1.0,
            }:
                fail(
                    f"{path}: game_id={game_id}: "
                    f"{first_prefix}_available "
                    "must be 0 or 1"
                )

            if second_available not in {
                0.0,
                1.0,
            }:
                fail(
                    f"{path}: game_id={game_id}: "
                    f"{second_prefix}_available "
                    "must be 0 or 1"
                )

            if (
                first_available
                != second_available
            ):
                fail(
                    f"{path}: game_id={game_id}: "
                    f"{market_name} candidate "
                    "availability must be paired"
                )

            if first_available == 0.0:
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
                                f"{path}: "
                                f"game_id={game_id}: "
                                f"unavailable "
                                f"{prefix}_{metric} "
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
                            and optional_number(
                                line_text
                            )
                            is None
                        ):
                            fail(
                                f"{path}: "
                                f"game_id={game_id}: "
                                f"{prefix}_line "
                                "must be finite "
                                "when present"
                            )

                continue

            first_odds = (
                optional_number(
                    row[
                        f"{first_prefix}_odds_american"
                    ]
                )
            )
            second_odds = (
                optional_number(
                    row[
                        f"{second_prefix}_odds_american"
                    ]
                )
            )

            if (
                first_odds is None
                or second_odds is None
                or first_odds == 0
                or second_odds == 0
            ):
                fail(
                    f"{path}: game_id={game_id}: "
                    f"{market_name} odds must "
                    "be finite and nonzero"
                )

            first_fair, second_fair = (
                no_vig_probabilities(
                    first_odds,
                    second_odds,
                )
            )

            model_values: list[
                float
            ] = []

            for (
                prefix,
                fair_probability,
            ) in (
                (
                    first_prefix,
                    first_fair,
                ),
                (
                    second_prefix,
                    second_fair,
                ),
            ):
                values: dict[
                    str,
                    float,
                ] = {}

                for metric in (
                    "odds_american",
                    "model_probability",
                    "implied_probability",
                    "edge",
                    "ev",
                    "full_kelly",
                    "kelly",
                ):
                    parsed = (
                        optional_number(
                            row.get(
                                f"{prefix}_{metric}",
                                "",
                            )
                        )
                    )

                    if parsed is None:
                        fail(
                            f"{path}: "
                            f"game_id={game_id}: "
                            f"{prefix}_{metric} "
                            "must be finite"
                        )

                    values[
                        metric
                    ] = parsed

                if (
                    values[
                        "odds_american"
                    ]
                    == 0
                ):
                    fail(
                        f"{path}: "
                        f"game_id={game_id}: "
                        f"{prefix}_odds_american "
                        "cannot be 0"
                    )

                if not (
                    0.0
                    <= values[
                        "model_probability"
                    ]
                    <= 1.0
                ):
                    fail(
                        f"{path}: "
                        f"game_id={game_id}: "
                        f"{prefix}_model_probability "
                        "outside [0,1]"
                    )

                if not (
                    0.0
                    <= values[
                        "implied_probability"
                    ]
                    <= 1.0
                ):
                    fail(
                        f"{path}: "
                        f"game_id={game_id}: "
                        f"{prefix}_implied_probability "
                        "outside [0,1]"
                    )

                if (
                    values[
                        "full_kelly"
                    ]
                    < 0
                    or values["kelly"] < 0
                ):
                    fail(
                        f"{path}: "
                        f"game_id={game_id}: "
                        f"{prefix} Kelly values "
                        "cannot be negative"
                    )

                if (
                    values["kelly"]
                    > values[
                        "full_kelly"
                    ]
                    + 1e-12
                ):
                    fail(
                        f"{path}: "
                        f"game_id={game_id}: "
                        f"{prefix}_kelly cannot "
                        "exceed full_kelly"
                    )

                expected = (
                    candidate_metrics(
                        values[
                            "model_probability"
                        ],
                        values[
                            "odds_american"
                        ],
                        fair_probability,
                    )
                )

                for metric in (
                    "implied_probability",
                    "edge",
                    "ev",
                    "full_kelly",
                ):
                    require_close(
                        values[metric],
                        expected[metric],
                        label=(
                            f"{path}: "
                            f"game_id={game_id}: "
                            f"{prefix}_{metric}"
                        ),
                    )

                model_values.append(
                    values[
                        "model_probability"
                    ]
                )

            require_close(
                sum(
                    model_values
                ),
                1.0,
                label=(
                    f"{path}: "
                    f"game_id={game_id}: "
                    f"{market_name} model "
                    "probabilities"
                ),
                atol=1e-9,
            )

            if include_line:
                first_line = (
                    optional_number(
                        row[
                            f"{first_prefix}_line"
                        ]
                    )
                )
                second_line = (
                    optional_number(
                        row[
                            f"{second_prefix}_line"
                        ]
                    )
                )

                if (
                    first_line is None
                    or second_line is None
                ):
                    fail(
                        f"{path}: "
                        f"game_id={game_id}: "
                        f"{market_name} lines "
                        "must be finite"
                    )

                if market_name == "spread":
                    require_close(
                        first_line
                        + second_line,
                        0.0,
                        label=(
                            f"{path}: "
                            f"game_id={game_id}: "
                            "spread lines"
                        ),
                        atol=1e-9,
                    )
                else:
                    require_close(
                        first_line,
                        second_line,
                        label=(
                            f"{path}: "
                            f"game_id={game_id}: "
                            "total lines"
                        ),
                        atol=1e-9,
                    )


def is_available(
    row: pd.Series,
    prefix: str,
) -> bool:
    value = optional_number(
        row.get(
            f"{prefix}_available",
            "",
        )
    )

    if value not in {
        0.0,
        1.0,
    }:
        fail(
            f"game_id={row['game_id']}: "
            f"{prefix}_available must be "
            "0 or 1; found "
            f"{row.get(f'{prefix}_available')!r}"
        )

    return value == 1.0


def candidate(
    row: pd.Series,
    market_name: str,
    side_name: str,
    prefix: str,
    selection: str,
    side_cfg: dict[str, Any],
) -> dict[str, Any] | None:
    if not side_cfg["enabled"]:
        return None

    if not is_available(
        row,
        prefix,
    ):
        return None

    result: dict[str, Any] = {
        "side_name": side_name,
        "selection": selection,
        "prefix": prefix,
    }

    for metric in [
        "odds_american",
        "model_probability",
        "implied_probability",
        "edge",
        "ev",
        "full_kelly",
    ]:
        column = (
            f"{prefix}_{metric}"
        )

        value = optional_number(
            row.get(
                column,
                "",
            )
        )

        if value is None:
            fail(
                f"game_id={row['game_id']}: "
                "available candidate "
                f"{prefix} has blank/"
                f"non-numeric {column}"
            )

        result[metric] = value

    if not (
        0
        <= result[
            "model_probability"
        ]
        <= 1
    ):
        fail(
            f"game_id={row['game_id']}: "
            f"{prefix}_model_probability "
            "outside [0,1]"
        )

    if not (
        0
        <= result[
            "implied_probability"
        ]
        <= 1
    ):
        fail(
            f"game_id={row['game_id']}: "
            f"{prefix}_implied_probability "
            "outside [0,1]"
        )

    if result["full_kelly"] < 0:
        fail(
            f"game_id={row['game_id']}: "
            f"{prefix}_full_kelly "
            "cannot be negative"
        )

    resolved = side_cfg[
        "thresholds"
    ]

    result["kelly"] = min(
        result["full_kelly"],
        resolved["max_kelly"],
    )

    if market_name in {
        "spread",
        "total",
    }:
        line = optional_number(
            row.get(
                f"{prefix}_line",
                "",
            )
        )

        if line is None:
            fail(
                f"game_id={row['game_id']}: "
                "available candidate "
                f"{prefix} has blank/"
                "non-numeric "
                f"{prefix}_line"
            )

        result["line"] = line

    else:
        result["line"] = None

    return result


def qualifies(
    item: dict[str, Any],
    market_name: str,
    market_cfg: dict[str, Any],
    side_cfg: dict[str, Any],
) -> bool:
    limits = side_cfg[
        "thresholds"
    ]

    if (
        item["ev"]
        < limits["min_ev"]
    ):
        return False

    if (
        item["edge"]
        < limits["min_edge"]
    ):
        return False

    if (
        item["kelly"]
        < limits["min_kelly"]
    ):
        return False

    if not (
        limits[
            "min_odds_american"
        ]
        <= item[
            "odds_american"
        ]
        <= limits[
            "max_odds_american"
        ]
    ):
        return False

    if not (
        limits[
            "min_model_prob"
        ]
        <= item[
            "model_probability"
        ]
        <= limits[
            "max_model_prob"
        ]
    ):
        return False

    if market_name == "spread":
        if (
            abs(item["line"])
            > market_cfg[
                "max_spread_abs"
            ]
        ):
            return False

    if market_name == "total":
        if not (
            market_cfg[
                "min_total"
            ]
            <= item["line"]
            <= market_cfg[
                "max_total"
            ]
        ):
            return False

    for (
        band_name,
        configured,
    ) in side_cfg[
        "bands"
    ].items():
        metric = BAND_TO_METRIC[
            band_name
        ]

        value = item[
            metric
        ]

        if (
            value is None
            or not matches_band(
                value,
                configured,
            )
        ):
            return False

    return True


def choose(
    items: list[
        dict[str, Any]
    ],
    preference: str,
) -> dict[str, Any]:
    primary = PICK_METRIC[
        preference
    ]

    def ranking(
        item: dict[str, Any],
    ) -> tuple[
        float,
        float,
        float,
        float,
    ]:
        return (
            float(
                item[primary]
            ),
            float(
                item[
                    "model_probability"
                ]
            ),
            float(
                item["ev"]
            ),
            float(
                item["kelly"]
            ),
        )

    return max(
        items,
        key=ranking,
    )


def empty_selection(
    prefix: str,
    reason: str,
) -> dict[str, Any]:
    result = {
        f"{prefix}_selected": 0,
        f"{prefix}_selection": "",
        f"{prefix}_selection_reason": (
            reason
        ),
        f"{prefix}_odds_american": (
            np.nan
        ),
        f"{prefix}_model_probability": (
            np.nan
        ),
        f"{prefix}_implied_probability": (
            np.nan
        ),
        f"{prefix}_edge": np.nan,
        f"{prefix}_ev": np.nan,
        f"{prefix}_full_kelly": (
            np.nan
        ),
        f"{prefix}_kelly": np.nan,
    }

    if prefix in {
        "spread",
        "total",
    }:
        result[
            f"{prefix}_line"
        ] = np.nan

    return result


def selected_values(
    prefix: str,
    item: dict[str, Any],
) -> dict[str, Any]:
    result = {
        f"{prefix}_selected": 1,
        f"{prefix}_selection": (
            item["selection"]
        ),
        f"{prefix}_selection_reason": (
            "SELECTED_BY_MARKETS_YAML"
        ),
        f"{prefix}_odds_american": (
            item["odds_american"]
        ),
        f"{prefix}_model_probability": (
            item[
                "model_probability"
            ]
        ),
        f"{prefix}_implied_probability": (
            item[
                "implied_probability"
            ]
        ),
        f"{prefix}_edge": (
            item["edge"]
        ),
        f"{prefix}_ev": (
            item["ev"]
        ),
        f"{prefix}_full_kelly": (
            item["full_kelly"]
        ),
        f"{prefix}_kelly": (
            item["kelly"]
        ),
    }

    if prefix in {
        "spread",
        "total",
    }:
        result[
            f"{prefix}_line"
        ] = item["line"]

    return result


def evaluate_market(
    row: pd.Series,
    market_name: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    spec = MARKETS[
        market_name
    ]

    market_cfg = config[
        "markets"
    ][market_name]

    output_prefix = spec[
        "output_prefix"
    ]

    if not market_cfg["enabled"]:
        return empty_selection(
            output_prefix,
            "MARKET_DISABLED",
        )

    qualifying: list[
        dict[str, Any]
    ] = []

    for (
        side_name,
        (
            candidate_prefix,
            selection,
        ),
    ) in spec[
        "sides"
    ].items():
        side_cfg = market_cfg[
            "sides"
        ][side_name]

        item = candidate(
            row,
            market_name,
            side_name,
            candidate_prefix,
            selection,
            side_cfg,
        )

        if (
            item is not None
            and qualifies(
                item,
                market_name,
                market_cfg,
                side_cfg,
            )
        ):
            qualifying.append(
                item
            )

    if not qualifying:
        return empty_selection(
            output_prefix,
            "NO_QUALIFYING_CANDIDATE",
        )

    winner = choose(
        qualifying,
        market_cfg[
            "pick_preference"
        ],
    )

    return selected_values(
        output_prefix,
        winner,
    )



def compare_selection_value(
    actual: Any,
    expected: Any,
    *,
    label: str,
) -> None:
    expected_text = clean(
        expected
    )
    actual_text = clean(
        actual
    )

    if not expected_text:
        if actual_text:
            fail(
                f"{label}: expected blank; "
                f"found {actual!r}"
            )
        return

    if isinstance(
        expected,
        (
            int,
            float,
            np.integer,
            np.floating,
        ),
    ):
        expected_number = (
            optional_number(
                expected
            )
        )
        actual_number = (
            optional_number(
                actual
            )
        )

        if (
            expected_number is None
            or actual_number is None
        ):
            fail(
                f"{label}: expected numeric "
                f"value {expected!r}; "
                f"found {actual!r}"
            )

        require_close(
            actual_number,
            expected_number,
            label=label,
            atol=1e-12,
        )
        return

    if actual_text != expected_text:
        fail(
            f"{label}: expected "
            f"{expected_text!r}; "
            f"found {actual_text!r}"
        )


def validate_processed_output(
    source: pd.DataFrame,
    output: pd.DataFrame,
    config: dict[str, Any],
    label: str,
) -> None:
    if (
        list(output.columns)
        != list(source.columns)
    ):
        fail(
            f"{label}: column order changed"
        )

    if len(output) != len(source):
        fail(
            f"{label}: row count changed; "
            f"expected={len(source)} "
            f"actual={len(output)}"
        )

    source_ids = (
        source["game_id"]
        .map(clean)
        .tolist()
    )
    output_ids = (
        output["game_id"]
        .map(clean)
        .tolist()
    )

    if output_ids != source_ids:
        fail(
            f"{label}: game_id order changed"
        )

    if (
        pd.Series(
            output_ids
        )
        .duplicated()
        .any()
    ):
        fail(
            f"{label}: duplicate game_id "
            "values"
        )

    non_selection = [
        column
        for column
        in source.columns
        if column
        not in selection_columns()
    ]

    for column in non_selection:
        source_values = [
            clean(value)
            for value in source[
                column
            ].tolist()
        ]
        output_values = [
            clean(value)
            for value in output[
                column
            ].tolist()
        ]

        if (
            output_values
            != source_values
        ):
            fail(
                f"{label}: non-selection "
                f"column {column!r} changed"
            )

    validate_candidate_contract(
        output,
        Path(label),
    )

    selection_set = set(
        selection_columns()
    )

    for position in range(
        len(source)
    ):
        source_row = (
            source.iloc[
                position
            ]
        )
        output_row = (
            output.iloc[
                position
            ]
        )

        expected_updates: dict[
            str,
            Any,
        ] = {}

        for market_name in MARKETS:
            expected_updates.update(
                evaluate_market(
                    source_row,
                    market_name,
                    config,
                )
            )

        if set(
            expected_updates
        ) != selection_set:
            fail(
                f"{label}: internal "
                "selection-column contract "
                "mismatch"
            )

        game_id = clean(
            source_row["game_id"]
        )

        for (
            column,
            expected,
        ) in expected_updates.items():
            compare_selection_value(
                output_row[
                    column
                ],
                expected,
                label=(
                    f"{label}: "
                    f"game_id={game_id}: "
                    f"{column}"
                ),
            )


def process_file(
    input_path: Path,
    config: dict[str, Any],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    df = load_csv(
        input_path
    )

    validate_input(
        df,
        input_path,
    )
    validate_candidate_contract(
        df,
        input_path,
    )

    output = df.copy()
    for column in selection_columns():
        output[column] = (
            output[column]
            .astype(object)
        )

    original_columns = list(
        df.columns
    )

    for index, row in df.iterrows():
        updates: dict[
            str,
            Any,
        ] = {}

        for market_name in MARKETS:
            updates.update(
                evaluate_market(
                    row,
                    market_name,
                    config,
                )
            )

        for (
            column,
            value,
        ) in updates.items():
            output.at[
                index,
                column,
            ] = value

    if (
        list(output.columns)
        != original_columns
    ):
        fail(
            "Column order changed while "
            f"processing {input_path}"
        )

    validate_processed_output(
        df,
        output,
        config,
        f"in-memory output {input_path}",
    )

    return (
        df,
        output,
    )


def output_name(
    input_path: Path,
) -> str:
    suffix = (
        "NFL_selected.csv"
    )

    if not input_path.name.endswith(
        suffix
    ):
        fail(
            "Unexpected input filename: "
            f"{input_path.name}"
        )

    return (
        input_path.name[
            :-len(suffix)
        ]
        + "NFL_picks.csv"
    )




def stage_output(
    output: pd.DataFrame,
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
        output.to_csv(
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
                f"Staged picks output "
                f"was not written: "
                f"{staged_path}"
            )

        return staged_path
    except Exception:
        staged_path.unlink(
            missing_ok=True
        )
        raise


def validate_serialized_output(
    path: Path,
    *,
    source: pd.DataFrame,
    config: dict[str, Any],
    label: str,
) -> pd.DataFrame:
    serialized = load_csv(
        path
    )

    validate_processed_output(
        source,
        serialized,
        config,
        label,
    )

    return serialized


def selection_counts(
    output: pd.DataFrame,
) -> dict[str, int]:
    return {
        "ml": int(
            pd.to_numeric(
                output[
                    "ml_selected"
                ],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        ),
        "spread": int(
            pd.to_numeric(
                output[
                    "spread_selected"
                ],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        ),
        "total": int(
            pd.to_numeric(
                output[
                    "total_selected"
                ],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        ),
    }


def publish_output_set(
    entries: list[
        dict[str, Any]
    ],
    stale_paths: list[Path],
    *,
    config: dict[str, Any],
    reporter: PipelineReporter,
) -> None:
    backups: dict[
        Path,
        Path,
    ] = {}
    live_modified = False
    rollback_failed = False

    managed_paths = [
        entry["output_path"]
        for entry in entries
    ] + list(
        stale_paths
    )

    reporter.update_details(
        {
            "publication_mode": (
                "transactional_multi_file_"
                "atomic_replace_with_rollback"
            ),
            "publication_completed": False,
            "post_publish_validation": False,
            "rollback_performed": False,
        }
    )

    try:
        for path in managed_paths:
            if path.exists():
                backup = (
                    path.parent
                    / (
                        f".{path.name}."
                        f"backup."
                        f"{uuid.uuid4().hex}"
                    )
                )
                shutil.copy2(
                    path,
                    backup,
                )
                backups[path] = (
                    backup
                )

        for entry in entries:
            os.replace(
                entry["staged_path"],
                entry["output_path"],
            )
            live_modified = True

        for stale_path in stale_paths:
            if stale_path.exists():
                stale_path.unlink()
                live_modified = True

        for entry in entries:
            validate_serialized_output(
                entry["output_path"],
                source=entry["source"],
                config=config,
                label=(
                    "published picks output "
                    f"{entry['output_path']}"
                ),
            )

        remaining_stale = [
            str(path)
            for path in stale_paths
            if path.exists()
        ]

        if remaining_stale:
            fail(
                "Stale root picks files "
                "remain after publication: "
                f"{remaining_stale}"
            )

        reporter.update_details(
            {
                "publication_completed": True,
                "post_publish_validation": True,
            }
        )
    except Exception as publish_exc:
        if live_modified:
            try:
                expected_paths = {
                    entry[
                        "output_path"
                    ]
                    for entry in entries
                }

                for path in expected_paths:
                    backup = backups.get(
                        path
                    )

                    if (
                        backup is not None
                        and backup.exists()
                    ):
                        os.replace(
                            backup,
                            path,
                        )
                    elif path.exists():
                        path.unlink()

                for path in stale_paths:
                    backup = backups.get(
                        path
                    )

                    if (
                        backup is not None
                        and backup.exists()
                    ):
                        os.replace(
                            backup,
                            path,
                        )
                    elif path.exists():
                        path.unlink()

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
                    "Picks publication failed "
                    "and rollback also failed: "
                    f"publication_error="
                    f"{publish_exc}; "
                    f"rollback_error="
                    f"{rollback_exc}"
                ) from rollback_exc

        raise
    finally:
        for entry in entries:
            Path(
                entry[
                    "staged_path"
                ]
            ).unlink(
                missing_ok=True
            )

        if not rollback_failed:
            for backup in backups.values():
                if backup.exists():
                    try:
                        backup.unlink()
                    except Exception as exc:
                        reporter.warning(
                            "Temporary picks backup "
                            "cleanup failed",
                            backup_path=str(
                                backup
                            ),
                            error_type=(
                                type(exc).__name__
                            ),
                            error=str(
                                exc
                            ),
                        )



def parse_args() -> argparse.Namespace:
    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
    )

    parser.add_argument(
        "--markets",
        type=Path,
        default=DEFAULT_MARKETS_PATH,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
    )

    return parser.parse_args()


def run(
    args: argparse.Namespace,
    reporter: PipelineReporter,
) -> None:
    load_runtime_dependencies(
        reporter
    )

    input_dir = (
        args.input_dir.resolve()
    )
    markets_path = (
        args.markets.resolve()
    )
    output_dir = (
        args.output_dir.resolve()
    )

    reporter.add_input(
        markets_path
    )

    reporter.update_details(
        {
            "input_dir": str(
                input_dir
            ),
            "output_dir": str(
                output_dir
            ),
            "pattern": args.pattern,
            "markets_path": str(
                markets_path
            ),
            "managed_set_sync": (
                args.pattern
                == DEFAULT_PATTERN
            ),
            "dependency_imports_ok": True,
            "all_outputs_staged": False,
            "staged_roundtrip_verified": False,
            "publication_completed": False,
            "post_publish_validation": False,
            "rollback_performed": False,
        }
    )

    if not input_dir.is_dir():
        fail(
            "Missing input directory: "
            f"{input_dir}"
        )

    config = normalize_config(
        load_yaml(
            markets_path
        )
    )

    input_files = sorted(
        path
        for path
        in input_dir.glob(
            args.pattern
        )
        if path.is_file()
    )

    if not input_files:
        fail(
            "No input files matched "
            f"{args.pattern!r} "
            f"in {input_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    entries: list[
        dict[str, Any]
    ] = []

    totals = {
        "games": 0,
        "ml": 0,
        "spread": 0,
        "total": 0,
    }

    per_file: list[
        dict[str, Any]
    ] = []

    try:
        for input_path in input_files:
            output_path = (
                output_dir
                / output_name(
                    input_path
                )
            )

            reporter.add_input(
                input_path
            )
            reporter.add_output(
                output_path
            )

            (
                source,
                output,
            ) = process_file(
                input_path,
                config,
            )

            counts = (
                selection_counts(
                    output
                )
            )

            totals["games"] += len(
                output
            )

            for market in (
                "ml",
                "spread",
                "total",
            ):
                totals[market] += (
                    counts[
                        market
                    ]
                )

            staged_path = (
                stage_output(
                    output,
                    output_path,
                )
            )

            validate_serialized_output(
                staged_path,
                source=source,
                config=config,
                label=(
                    "staged picks output "
                    f"{output_path}"
                ),
            )

            entries.append(
                {
                    "input_path": (
                        input_path
                    ),
                    "output_path": (
                        output_path
                    ),
                    "source": source,
                    "output": output,
                    "staged_path": (
                        staged_path
                    ),
                    "counts": counts,
                }
            )

            per_file.append(
                {
                    "input": str(
                        input_path
                    ),
                    "output": str(
                        output_path
                    ),
                    "games": len(
                        output
                    ),
                    "ml_picks": (
                        counts["ml"]
                    ),
                    "spread_picks": (
                        counts[
                            "spread"
                        ]
                    ),
                    "total_picks": (
                        counts[
                            "total"
                        ]
                    ),
                }
            )

        reporter.update_details(
            {
                "files_processed": len(
                    entries
                ),
                "per_file": per_file,
                "total_games": (
                    totals[
                        "games"
                    ]
                ),
                "total_ml_picks": (
                    totals["ml"]
                ),
                "total_spread_picks": (
                    totals[
                        "spread"
                    ]
                ),
                "total_total_picks": (
                    totals[
                        "total"
                    ]
                ),
                "all_outputs_staged": True,
                "staged_roundtrip_verified": True,
            }
        )

        expected_paths = {
            entry[
                "output_path"
            ].resolve()
            for entry in entries
        }

        stale_paths: list[
            Path
        ] = []

        if (
            args.pattern
            == DEFAULT_PATTERN
        ):
            existing_root_picks = {
                path.resolve()
                for path
                in output_dir.glob(
                    "*NFL_picks.csv"
                )
                if path.is_file()
            }

            stale_paths = sorted(
                existing_root_picks
                - expected_paths
            )

        reporter.update_details(
            {
                "expected_root_pick_files": [
                    str(path)
                    for path in sorted(
                        expected_paths
                    )
                ],
                "stale_root_pick_files": [
                    str(path)
                    for path in stale_paths
                ],
            }
        )

        publish_output_set(
            entries,
            stale_paths,
            config=config,
            reporter=reporter,
        )
    except Exception:
        for entry in entries:
            staged = entry.get(
                "staged_path"
            )
            if staged is not None:
                Path(
                    staged
                ).unlink(
                    missing_ok=True
                )
        raise

    reporter.set_rows(
        rows_in=totals[
            "games"
        ],
        rows_out=totals[
            "games"
        ],
    )

    for entry in entries:
        counts = entry[
            "counts"
        ]

        print(
            f"Processed: "
            f"{entry['input_path'].name} "
            f"-> "
            f"{entry['output_path'].name} "
            f"games="
            f"{len(entry['output'])} "
            f"ml_picks="
            f"{counts['ml']} "
            f"spread_picks="
            f"{counts['spread']} "
            f"total_picks="
            f"{counts['total']}"
        )

    print(
        "NFL selection layer complete: "
        f"files={len(entries)} "
        f"games={totals['games']} "
        f"ml_picks={totals['ml']} "
        f"spread_picks="
        f"{totals['spread']} "
        f"total_picks="
        f"{totals['total']}"
    )


def main() -> int:
    args = parse_args()

    try:
        with PipelineReporter(
            script=SCRIPT_PATH,
            stage="03_picks",
            report_root=REPORT_ROOT,
            pipeline="NFL",
            league="NFL",
            extra_context={
                "component": (
                    "markets.yaml "
                    "selection layer"
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
