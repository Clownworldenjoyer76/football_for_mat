#!/usr/bin/env python3
"""
NFL Prop Engine shared utilities.

READS:
    docs/win/football/nfl/prop_engine/config/prop_engine.yaml

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

import json
import logging
import math
import os
import re
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import yaml


_CONFIG_RELATIVE_PATH = Path(
    "docs/win/football/nfl/prop_engine/config/prop_engine.yaml"
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
    "output",
)

_TEAM_ALIASES = {
    "WAS": "WSH",
    "LA": "LAR",
    "JAC": "JAX",
}

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
    """Return docs/win/football/nfl/prop_engine."""
    path = nfl_root() / "prop_engine"

    if not path.is_dir():
        raise FileNotFoundError(f"Prop Engine root does not exist: {path}")

    return path


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

    return config


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
