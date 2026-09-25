#!/usr/bin/env python3
"""
NFL Prop Engine shared utilities.

READS:
    docs/win/football/prop_engine/config/prop_engine.yaml

WRITES:
    Nothing directly.

RESPONSIBILITIES:
    - repository path resolution
    - configuration loading
    - CSV/parquet loading
    - team normalization
    - player ID normalization
    - date/time normalization
    - season/week/game ID validation
    - feature-column allow/deny validation
    - deterministic sorting
    - atomic output writes
    - structured logging helpers

MARKET POLICY:
    Reject configured sportsbook, odds, market, DRAT, and external
    prediction tokens from model feature sets.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml


_CONFIG_RELATIVE_PATH = Path(
    "docs/win/football/prop_engine/config/prop_engine.yaml"
)

_REQUIRED_CONFIG_SECTIONS = (
    "system",
    "paths",
    "seasons",
    "targets",
    "positions",
    "rolling_windows",
    "ewm",
    "eligibility",
    "injuries",
    "weather",
    "training",
    "validation",
    "uncertainty",
    "models",
    "forbidden_features",
    "forbidden_input_paths",
    "output",
)

_TEAM_ALIASES = {
    "SD": "LAC",
    "OAK": "LV",
    "STL": "LAR",
    "WAS": "WSH",
    "LA": "LAR",
    "JAC": "JAX",
}

USAGE_CANDIDATES = {
    "passing_yards": [
        "player_pass_attempts_roll3_mean",
        "player_pass_attempts_roll5_mean",
        "player_pass_attempts_ewm5",
        "player_pass_attempts_career_prior",
    ],
    "passing_tds": [
        "player_pass_attempts_roll3_mean",
        "player_pass_attempts_roll5_mean",
        "player_pass_attempts_ewm5",
        "player_pass_attempts_career_prior",
    ],
    "rushing_yards": [
        "player_carries_roll3_mean",
        "player_carries_roll5_mean",
        "player_carries_ewm5",
        "player_carries_career_prior",
    ],
    "rushing_tds": [
        "player_goal_line_carries_roll3_mean",
        "player_goal_line_carries_roll5_mean",
        "player_carries_roll3_mean",
        "player_carries_career_prior",
    ],
    "receiving_yards": [
        "player_targets_roll3_mean",
        "player_targets_roll5_mean",
        "player_targets_ewm5",
        "player_targets_career_prior",
    ],
    "receiving_tds": [
        "player_red_zone_targets_roll3_mean",
        "player_red_zone_targets_roll5_mean",
        "player_targets_roll3_mean",
        "player_targets_career_prior",
    ],
    "kicking_points": [
        "player_field_goal_attempts_roll3_mean",
        "player_field_goal_attempts_roll5_mean",
        "player_field_goal_attempts_career_prior",
    ],
    "tackles": [
        "player_defense_participation_roll3_mean",
        "role_participation_roll3",
        "player_defense_participation_career_prior",
    ],
    "sacks": [
        "player_defense_participation_roll3_mean",
        "role_participation_roll3",
        "player_defense_participation_career_prior",
    ],
}

KICKING_USAGE_COMPONENTS = [
    "player_field_goal_attempts_roll3_mean",
    "player_extra_point_attempts_roll3_mean",
]


HISTORICAL_RELOCATION_ALIASES = {
    "SD": "LAC",
    "OAK": "LV",
    "STL": "LAR",
}

_NFLVERSE_OPPONENT_GAME_ID_RE = re.compile(
    r"^(?P<season>\d{4})_(?P<week>\d{1,2})_"
    r"(?P<away>[A-Za-z0-9]+)_(?P<home>[A-Za-z0-9]+)$"
)


_NFLVERSE_GAME_ID_RE = re.compile(
    r"^(?P<season>\d{4})_(?P<week>\d{1,2})_"
    r"(?P<away>[A-Za-z]{2,3})_(?P<home>[A-Za-z]{2,3})$"
)

_NUMERIC_GAME_ID_RE = re.compile(r"^\d+$")

_LOGGER = logging.getLogger("nfl_prop_engine")

if not _LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _LOGGER.addHandler(_handler)
    _LOGGER.setLevel(logging.INFO)
    _LOGGER.propagate = False


def repo_root() -> Path:
    """Return the repository root containing docs/win/football/nfl."""
    start = Path(__file__).resolve()

    for candidate in start.parents:
        if (
            (candidate / ".git").exists()
            and (candidate / "docs/win/football/nfl").is_dir()
        ):
            return candidate

    for candidate in start.parents:
        if (candidate / "docs/win/football/nfl").is_dir():
            return candidate

    raise RuntimeError(
        "Unable to resolve repository root from "
        f"{start}. Expected ancestor containing docs/win/football/nfl."
    )


def nfl_root() -> Path:
    """Return docs/win/football/nfl."""
    path = repo_root() / "docs/win/football/nfl"

    if not path.is_dir():
        raise FileNotFoundError(f"NFL root does not exist: {path}")

    return path


def prop_root() -> Path:
    """Return docs/win/football/prop_engine."""
    path = repo_root() / "docs/win/football/prop_engine"

    if not path.is_dir():
        raise FileNotFoundError(f"Prop Engine root does not exist: {path}")

    return path


def run_market_exclusion_preflight() -> None:
    """Require the canonical market-exclusion audit to pass before training."""
    audit = prop_root() / "scripts" / "validate" / "audit_market_exclusion.py"
    result = subprocess.run(
        [sys.executable, str(audit), "--preflight"],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("Issue 28 market-exclusion preflight failed.")


def _resolve_repo_path(path: str | os.PathLike[str]) -> Path:
    value = Path(path).expanduser()

    if not value.is_absolute():
        value = repo_root() / value

    return value.resolve()


def _resolve_prop_output_path(path: str | os.PathLike[str]) -> Path:
    resolved = _resolve_repo_path(path)
    root = prop_root().resolve()

    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "Prop Engine writes are restricted to "
            f"{root}; received output path {resolved}"
        ) from exc

    return resolved


def load_json_mapping(
    path: str | os.PathLike[str],
    *,
    missing_message: str | None = None,
) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(
            missing_message or f"Required JSON missing: {resolved}"
        )
    with resolved.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {resolved}")
    return value


def load_yaml_mapping(
    path: str | os.PathLike[str],
    *,
    missing_message: str | None = None,
) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(
            missing_message or f"Required YAML missing: {resolved}"
        )
    with resolved.open("r", encoding="utf-8-sig") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {resolved}")
    return value


def load_config() -> dict:
    """Load and validate the shared Prop Engine YAML contract."""
    path = repo_root() / _CONFIG_RELATIVE_PATH

    if not path.is_file():
        raise FileNotFoundError(
            f"Prop Engine config does not exist: {path}"
        )

    with path.open("r", encoding="utf-8-sig") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(
            f"Prop Engine config must be a YAML mapping: {path}"
        )

    missing = [
        key
        for key in _REQUIRED_CONFIG_SECTIONS
        if key not in config
    ]

    if missing:
        raise ValueError(
            "Prop Engine config is missing required top-level section(s): "
            + ", ".join(missing)
        )

    system = config.get("system")

    if not isinstance(system, dict):
        raise ValueError(
            "Config section 'system' must be a mapping."
        )

    if system.get("market_data_allowed") is not False:
        raise ValueError(
            "Prop Engine config must set "
            "system.market_data_allowed: false"
        )

    forbidden = config.get("forbidden_features")

    if not isinstance(forbidden, list) or not forbidden:
        raise ValueError(
            "Config section 'forbidden_features' "
            "must be a non-empty list."
        )

    forbidden_inputs = config.get("forbidden_input_paths")

    if not isinstance(forbidden_inputs, list) or not forbidden_inputs:
        raise ValueError(
            "Config section 'forbidden_input_paths' "
            "must be a non-empty list."
        )

    normalized_forbidden_inputs = [
        str(value).strip().replace("\\", "/")
        for value in forbidden_inputs
        if str(value).strip()
    ]

    if len(normalized_forbidden_inputs) != len(forbidden_inputs):
        raise ValueError(
            "Config forbidden_input_paths cannot contain blank values."
        )

    if len(set(normalized_forbidden_inputs)) != len(normalized_forbidden_inputs):
        raise ValueError(
            "Config forbidden_input_paths cannot contain duplicates."
        )

    return config



def forbidden_input_paths(
    config: Mapping[str, Any] | None = None,
) -> tuple[tuple[Path, bool, str], ...]:
    active = load_config() if config is None else config
    values = active.get("forbidden_input_paths")

    if not isinstance(values, list) or not values:
        raise ValueError(
            "Config forbidden_input_paths must be a non-empty list."
        )

    rules: list[tuple[Path, bool, str]] = []

    for raw in values:
        reference = str(raw).strip().replace("\\", "/")

        if not reference:
            raise ValueError(
                "Config forbidden_input_paths cannot contain blank values."
            )

        rules.append(
            (
                _resolve_repo_path(reference),
                reference.endswith("/"),
                reference,
            )
        )

    return tuple(rules)


def reject_forbidden_input_path(
    path: str | os.PathLike[str],
    config: Mapping[str, Any] | None = None,
) -> None:
    resolved = _resolve_repo_path(path)

    for forbidden, is_directory, reference in forbidden_input_paths(config):
        blocked = (
            resolved == forbidden
            or (
                is_directory
                and forbidden in resolved.parents
            )
        )

        if blocked:
            raise ValueError(
                "Prop Engine direct input is forbidden by Issue 54: "
                f"{reference} (requested {resolved})"
            )

def require_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
    label: str,
) -> None:
    """Raise when required columns are absent."""
    required = list(columns or [])

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{label}: missing required column(s): "
            + ", ".join(missing)
        )


def read_csv_required(
    path: str | os.PathLike[str],
    required_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read a required CSV and optionally validate its columns."""
    resolved = _resolve_repo_path(path)
    reject_forbidden_input_path(resolved)

    if not resolved.is_file():
        raise FileNotFoundError(
            f"Required CSV does not exist: {resolved}"
        )

    df = pd.read_csv(resolved)

    if required_columns:
        require_columns(
            df,
            required_columns,
            str(resolved),
        )

    return df


def read_parquet_required(
    path: str | os.PathLike[str],
    required_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read a required parquet file and optionally validate its columns."""
    resolved = _resolve_repo_path(path)
    reject_forbidden_input_path(resolved)

    if not resolved.is_file():
        raise FileNotFoundError(
            f"Required parquet does not exist: {resolved}"
        )

    df = pd.read_parquet(resolved)

    if required_columns:
        require_columns(
            df,
            required_columns,
            str(resolved),
        )

    return df


def _validate_season_week(df: pd.DataFrame) -> None:
    if "season" in df.columns:
        season = pd.to_numeric(
            df["season"],
            errors="coerce",
        )

        invalid = (
            season.isna()
            | (season % 1 != 0)
            | ~season.between(1900, 2200)
        )

        if invalid.any():
            sample = (
                df.loc[invalid, "season"]
                .head(10)
                .tolist()
            )

            raise ValueError(
                f"Invalid season value(s): {sample}"
            )

    if "week" in df.columns:
        week = pd.to_numeric(
            df["week"],
            errors="coerce",
        )

        invalid = (
            week.isna()
            | (week % 1 != 0)
            | ~week.between(1, 25)
        )

        if invalid.any():
            sample = (
                df.loc[invalid, "week"]
                .head(10)
                .tolist()
            )

            raise ValueError(
                f"Invalid week value(s): {sample}"
            )


def season_week_sort(df: pd.DataFrame) -> pd.DataFrame:
    """Validate season/week and return a stable canonical sort."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "season_week_sort expects a pandas DataFrame."
        )

    result = df.copy()

    _validate_season_week(result)

    sort_columns = [
        column
        for column in (
            "season",
            "week",
            "game_id",
            "player_id",
            "gsis_id",
            "team",
            "position",
            "player_name",
        )
        if column in result.columns
    ]

    if sort_columns:
        result = result.sort_values(
            sort_columns,
            kind="mergesort",
            na_position="last",
        )

    return result.reset_index(drop=True)


def _atomic_replace(
    temp_path: Path,
    destination: Path,
) -> None:
    try:
        os.replace(
            temp_path,
            destination,
        )
    finally:
        if temp_path.exists():
            temp_path.unlink()


def write_parquet_atomic(
    df: pd.DataFrame,
    path: str | os.PathLike[str],
) -> None:
    """Deterministically sort and atomically replace parquet output."""
    destination = _resolve_prop_output_path(path)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ordered = season_week_sort(df)

    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    )

    temp_path = Path(handle.name)
    handle.close()

    try:
        ordered.to_parquet(
            temp_path,
            index=False,
        )

        _atomic_replace(
            temp_path,
            destination,
        )

    except Exception:
        if temp_path.exists():
            temp_path.unlink()

        raise


def read_csv_dict_rows(
    path: str | os.PathLike[str],
) -> tuple[list[dict[str, str]], list[str]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Missing input file: {source}")
    with source.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        return rows, list(reader.fieldnames or [])


def write_csv_dict_rows_atomic(
    path: str | os.PathLike[str],
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        encoding="utf-8",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_path, destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def write_csv_atomic(
    df: pd.DataFrame,
    path: str | os.PathLike[str],
) -> None:
    """Deterministically sort and atomically replace CSV output."""
    destination = _resolve_prop_output_path(path)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ordered = season_week_sort(df)

    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    )

    temp_path = Path(handle.name)
    handle.close()

    try:
        ordered.to_csv(
            temp_path,
            index=False,
            encoding="utf-8",
            lineterminator="\n",
        )

        _atomic_replace(
            temp_path,
            destination,
        )

    except Exception:
        if temp_path.exists():
            temp_path.unlink()

        raise



def stable_json_bytes(value: dict[str, Any]) -> bytes:
    """Serialize a JSON object deterministically as UTF-8 with a final newline."""
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def write_json_default_str_atomic(
    path: str | os.PathLike[str],
    value: dict[str, Any],
) -> None:
    # Preserve legacy sorted/indented JSON with default=str.
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    )
    temp_path = Path(handle.name)

    try:
        with handle:
            json.dump(
                value,
                handle,
                indent=2,
                sort_keys=True,
                default=str,
            )
            handle.write("\n")
        os.replace(temp_path, destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()

def write_json_atomic(
    path: str | os.PathLike[str],
    value: dict[str, Any],
) -> None:
    """Write deterministic JSON atomically inside the Prop Engine root."""
    destination = _resolve_prop_output_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            handle.write(stable_json_bytes(value))
        _atomic_replace(temp_path, destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def lightgbm_regression_params(seed: int) -> dict[str, Any]:
    """Return the deterministic regression parameters shared by component trainers."""
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_data_in_leaf": 40,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
        "max_bin": 255,
        "verbosity": -1,
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "data_random_seed": seed,
        "deterministic": True,
        "force_col_wise": True,
        "num_threads": 1,
    }

def apply_calibration_mapping(
    values: np.ndarray,
    mapping: Mapping[str, Any],
) -> np.ndarray:
    """Apply a persisted monotone calibration mapping."""
    xp = np.asarray(mapping["knots_x"], dtype="float64")
    fp = np.asarray(mapping["knots_y"], dtype="float64")
    output = np.interp(
        np.asarray(values, dtype="float64"),
        xp,
        fp,
        left=float(mapping["left_value"]),
        right=float(mapping["right_value"]),
    )
    bounds = mapping.get("output_bounds", [None, None])
    if bounds[0] is not None:
        output = np.maximum(output, float(bounds[0]))
    if bounds[1] is not None:
        output = np.minimum(output, float(bounds[1]))
    return output


def calibrated_count_outputs(
    raw_selected: np.ndarray,
    payload: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Apply persisted count calibration to point predictions."""
    count_calibration = payload["count_calibration"]
    raw = np.maximum(np.asarray(raw_selected, dtype="float64"), 0.0)
    expected = apply_calibration_mapping(
        raw,
        count_calibration["expected_count"]["mapping"],
    )
    poisson_p1 = 1.0 - np.exp(-expected)
    poisson_p2 = 1.0 - np.exp(-expected) * (1.0 + expected)
    p1 = apply_calibration_mapping(
        poisson_p1,
        count_calibration["probability_1_plus"]["mapping"],
    )
    p2 = apply_calibration_mapping(
        poisson_p2,
        count_calibration["probability_2_plus"]["mapping"],
    )
    return {
        "expected_count": np.maximum(expected, 0.0),
        "probability_1_plus": np.clip(p1, 0.0, 1.0),
        "probability_2_plus": np.clip(p2, 0.0, 1.0),
    }


def apply_usage_buckets(
    values: np.ndarray,
    thresholds: Mapping[str, Any],
) -> np.ndarray:
    """Assign low/medium/high usage buckets using persisted thresholds."""
    low = float(thresholds["low_max"])
    medium = float(thresholds["medium_max"])
    labels = np.full(len(values), "low", dtype=object)
    finite = np.isfinite(values)
    labels[finite & (values > low)] = "medium"
    labels[finite & (values > medium)] = "high"
    labels[~finite] = "low"
    return labels


def enforce_monotone_quantiles(
    output: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Ensure q10 <= q25 <= q50 <= q75 <= q90 row-wise."""
    names = ("q10", "q25", "q50", "q75", "q90")
    matrix = np.column_stack([output[name] for name in names])
    matrix = np.maximum.accumulate(matrix, axis=1)
    for index, name in enumerate(names):
        output[name] = matrix[:, index]
    return output


def regression_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float | None]:
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    error = y - p
    rmse = float(np.sqrt(np.mean(np.square(error))))
    mae = float(np.mean(np.abs(error)))
    denominator = float(np.sum(np.square(y - y.mean())))
    if denominator <= 0.0:
        r2 = None
    else:
        value = 1.0 - float(np.sum(np.square(error)) / denominator)
        r2 = value if math.isfinite(value) else None
    return {"rmse": rmse, "mae": mae, "r2": r2}


def poisson_deviance(
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    negative_actual_message: str = (
        "Poisson deviance cannot be computed with negative actual values."
    ),
) -> float:
    """Return mean Poisson deviance using a strictly positive prediction floor."""
    y = np.asarray(actual, dtype="float64")
    lam = np.maximum(np.asarray(predicted, dtype="float64"), 1e-12)
    if np.any(y < 0.0):
        raise ValueError(negative_actual_message)
    terms = np.empty_like(y)
    zero = y <= 0.0
    terms[zero] = lam[zero]
    nonzero = ~zero
    terms[nonzero] = (
        y[nonzero] * np.log(y[nonzero] / lam[nonzero])
        - (y[nonzero] - lam[nonzero])
    )
    return float(2.0 * np.mean(terms))


def brier_1plus(
    actual: np.ndarray,
    probability: np.ndarray,
) -> float:
    """Return Brier score for the event actual >= 1."""
    event = (np.asarray(actual, dtype="float64") >= 1.0).astype("float64")
    clipped = np.clip(np.asarray(probability, dtype="float64"), 0.0, 1.0)
    return float(np.mean(np.square(clipped - event)))


def _is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True

    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False

    if isinstance(missing, bool):
        return missing

    return False


def clean_text(value: Any) -> str:
    # Preserve the exact nullable-text semantics used by Prop Engine scripts.
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.casefold() in {"", "nan", "none", "null", "<na>", "nat"}:
        return ""
    return text


def numeric_series_required(
    series: pd.Series,
    *,
    label: str,
    fill_zero: bool = False,
    invalid_description: str = "non-numeric values found",
    examples_label: str = "Examples",
) -> pd.Series:
    converted = pd.to_numeric(series, errors="coerce")
    invalid = (
        series.notna()
        & series.astype(str).str.strip().ne("")
        & converted.isna()
    )
    if invalid.any():
        examples = series.loc[invalid].astype(str).head(10).tolist()
        raise ValueError(
            f"{label}: {invalid_description}. "
            f"{examples_label}={examples}"
        )
    converted = converted.astype(float)
    return converted.fillna(0.0) if fill_zero else converted


def safe_divide_nonzero(
    numerator: pd.Series,
    denominator: pd.Series,
    *,
    replace_infinite: bool = False,
) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce").astype("float64")
    den = pd.to_numeric(denominator, errors="coerce").astype("float64")
    if replace_infinite:
        num = num.replace([np.inf, -np.inf], np.nan)
        den = den.replace([np.inf, -np.inf], np.nan)
    result = pd.Series(np.nan, index=num.index, dtype="float64")
    valid = num.notna() & den.notna() & den.ne(0.0)
    result.loc[valid] = num.loc[valid] / den.loc[valid]
    return result


def safe_divide_positive(
    numerator: pd.Series,
    denominator: pd.Series,
    *,
    replace_result_infinite: bool = True,
) -> pd.Series:
    num = (
        pd.to_numeric(numerator, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .astype("float64")
    )
    den = (
        pd.to_numeric(denominator, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .astype("float64")
    )
    result = pd.Series(np.nan, index=num.index, dtype="float64")
    valid = num.notna() & den.notna() & den.gt(0.0)
    result.loc[valid] = num.loc[valid] / den.loc[valid]
    if replace_result_infinite:
        result = result.replace([np.inf, -np.inf], np.nan)
    return result


def normalize_team(value: Any) -> str:
    """Normalize a team code using Prop Engine aliases."""
    if _is_missing_scalar(value):
        return ""

    key = str(value).strip().upper()

    if not key:
        return ""

    return _TEAM_ALIASES.get(
        key,
        key,
    )


def normalize_player_id(value: Any) -> str:
    """Normalize an ID without changing GSIS punctuation."""
    if _is_missing_scalar(value):
        return ""

    if (
        isinstance(value, float)
        and math.isfinite(value)
        and value.is_integer()
    ):
        return str(int(value))

    text = str(value).strip()

    if not text:
        return ""

    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]

    return text


def normalize_name(value: Any) -> str:
    """Return a normalized name suitable for deterministic matching."""
    if _is_missing_scalar(value):
        return ""

    text = unicodedata.normalize(
        "NFKD",
        str(value),
    )

    text = "".join(
        char
        for char in text
        if not unicodedata.combining(char)
    )

    text = text.casefold()

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    return " ".join(
        text.split()
    )


def opponent_from_nflverse_game_id(
    game_id: Any,
    team: Any,
    *,
    invalid_message: str,
    mismatch_message: str,
) -> str:
    text = clean_text(game_id)
    match = _NFLVERSE_OPPONENT_GAME_ID_RE.fullmatch(text)

    if not match:
        raise ValueError(invalid_message)

    def canonical(value: Any) -> str:
        normalized = normalize_team(value)
        return HISTORICAL_RELOCATION_ALIASES.get(normalized, normalized)

    away = canonical(match.group("away"))
    home = canonical(match.group("home"))
    club = canonical(team)

    if club == away:
        return home

    if club == home:
        return away

    raise ValueError(mismatch_message)


def parse_game_id(value: Any) -> str:
    """
    Validate numeric ESPN/GSIS-style or nflverse-style game IDs.
    """
    if _is_missing_scalar(value):
        raise ValueError(
            "game_id cannot be null."
        )

    if (
        isinstance(value, float)
        and math.isfinite(value)
        and value.is_integer()
    ):
        text = str(int(value))
    else:
        text = str(value).strip()

    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]

    if _NUMERIC_GAME_ID_RE.fullmatch(text):
        return text

    match = _NFLVERSE_GAME_ID_RE.fullmatch(text)

    if not match:
        raise ValueError(
            "Unsupported game_id format. Expected a numeric ID "
            "or YYYY_WW_AWAY_HOME; received "
            f"{value!r}"
        )

    season = int(
        match.group("season")
    )

    week = int(
        match.group("week")
    )

    if not 1900 <= season <= 2200:
        raise ValueError(
            f"Invalid game_id season: {season}"
        )

    if not 1 <= week <= 25:
        raise ValueError(
            f"Invalid game_id week: {week}"
        )

    away = normalize_team(
        match.group("away")
    )

    home = normalize_team(
        match.group("home")
    )

    return (
        f"{season:04d}_"
        f"{week:02d}_"
        f"{away}_"
        f"{home}"
    )


def ensure_unique(
    df: pd.DataFrame,
    columns: Sequence[str],
    label: str,
) -> None:
    """Raise when data is not unique at the requested grain."""
    keys = list(columns)

    if not keys:
        raise ValueError(
            f"{label}: uniqueness columns cannot be empty."
        )

    require_columns(
        df,
        keys,
        label,
    )

    duplicates = df.loc[
        df.duplicated(
            keys,
            keep=False,
        ),
        keys,
    ].copy()

    if duplicates.empty:
        return

    duplicates = (
        duplicates
        .sort_values(
            keys,
            kind="mergesort",
            na_position="last",
        )
        .head(20)
    )

    raise ValueError(
        f"{label}: duplicate rows found for key {keys}. "
        f"Sample: "
        f"{duplicates.to_dict(orient='records')}"
    )


def reject_forbidden_feature_columns(
    columns: Iterable[str],
    config: Mapping[str, Any],
) -> None:
    """
    Reject feature columns containing configured forbidden tokens.
    """
    forbidden = config.get(
        "forbidden_features"
    )

    if not isinstance(forbidden, list) or not forbidden:
        raise ValueError(
            "Config section 'forbidden_features' "
            "must be a non-empty list."
        )

    normalized_forbidden = [
        str(value).strip().casefold()
        for value in forbidden
        if str(value).strip()
    ]

    rejected: dict[str, list[str]] = {}

    for column in columns:
        column_text = str(column)
        normalized_column = column_text.casefold()

        matches = [
            token
            for token in normalized_forbidden
            if token in normalized_column
        ]

        if matches:
            rejected[column_text] = sorted(
                set(matches)
            )

    if rejected:
        details = "; ".join(
            f"{column} -> {tokens}"
            for column, tokens
            in sorted(rejected.items())
        )

        raise ValueError(
            "Forbidden market/prediction feature "
            "column(s) detected: "
            + details
        )


def safe_numeric(
    series: pd.Series,
) -> pd.Series:
    """Coerce values to numeric and replace infinities with NaN."""
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    return numeric.replace(
        [
            float("inf"),
            float("-inf"),
        ],
        float("nan"),
    )


def kickoff_timestamp(
    row: Mapping[str, Any] | pd.Series,
) -> pd.Timestamp:
    """
    Normalize kickoff date/time using verified repository schemas.

    Current schedules:
        game_date + game_time + game_timezone

    Historical games:
        gameday + gametime

    Historical rows without an explicit timezone remain timezone-naive.
    """
    date_value = row.get(
        "game_date"
    )

    if (
        _is_missing_scalar(date_value)
        or not str(date_value).strip()
    ):
        date_value = row.get(
            "gameday"
        )

    time_value = row.get(
        "game_time"
    )

    if (
        _is_missing_scalar(time_value)
        or not str(time_value).strip()
    ):
        time_value = row.get(
            "gametime"
        )

    if (
        _is_missing_scalar(date_value)
        or not str(date_value).strip()
    ):
        raise ValueError(
            "Kickoff date is missing; expected "
            "game_date or gameday."
        )

    if (
        _is_missing_scalar(time_value)
        or not str(time_value).strip()
    ):
        raise ValueError(
            "Kickoff time is missing; expected "
            "game_time or gametime."
        )

    timestamp = pd.to_datetime(
        f"{str(date_value).strip()} "
        f"{str(time_value).strip()}",
        errors="raise",
    )

    timezone_value = row.get(
        "game_timezone"
    )

    if (
        not _is_missing_scalar(timezone_value)
        and str(timezone_value).strip()
    ):
        try:
            timestamp = timestamp.tz_localize(
                str(timezone_value).strip()
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Invalid game_timezone "
                f"{timezone_value!r}"
            ) from exc

    return timestamp


def log_run(
    script_name: str,
    payload: Mapping[str, Any],
) -> None:
    """Emit one deterministic structured JSON log record."""
    if not script_name or not str(script_name).strip():
        raise ValueError(
            "script_name cannot be empty."
        )

    if not isinstance(payload, Mapping):
        raise TypeError(
            "payload must be a mapping."
        )

    record = {
        "script": str(script_name).strip(),
        "payload": dict(payload),
    }

    _LOGGER.info(
        json.dumps(
            record,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
    )
