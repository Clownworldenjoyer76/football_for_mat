#!/usr/bin/env python3
"""
Refresh current-season nflverse player data for the Prop Engine.

USES:
    Already-integrated nflreadpy/nfl_data_py football sources.

WRITES ONLY:
    docs/win/football/nfl/prop_engine/data/current/source/

OUTPUTS:
    stats_player_week_{season}.parquet
    roster_weekly_{season}.parquet
    snap_counts_{season}.parquet
    pbp_participation_{season}.parquet
    players.parquet

DO NOT:
    overwrite docs/win/football/nfl/data/historic_data/
    query sportsbook endpoints
    create market-derived data
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent

if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import common


FAMILIES = (
    "player_stats",
    "weekly_rosters",
    "snap_counts",
    "participation",
    "players",
)


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh isolated current-season nflverse "
            "player data for the Prop Engine."
        )
    )

    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help=(
            "NFL season to refresh. Defaults to "
            "seasons.current in prop_engine.yaml."
        ),
    )

    return parser.parse_args()


def resolve_season(
    args: argparse.Namespace,
    config: dict,
) -> int:
    if args.season is not None:
        season = int(args.season)
    else:
        try:
            season = int(
                config["seasons"]["current"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Unable to resolve default season from "
                "config seasons.current."
            ) from exc

    if not 1900 <= season <= 2200:
        raise ValueError(
            f"Invalid season: {season}"
        )

    return season


def to_pandas(data: Any) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()

    if hasattr(data, "to_pandas"):
        converted = data.to_pandas()

        if isinstance(converted, pd.DataFrame):
            return converted

    return pd.DataFrame(data)


def import_source(
    module_name: str,
) -> Any:
    return importlib.import_module(
        module_name
    )


def module_version(module: Any) -> str:
    return str(
        getattr(
            module,
            "__version__",
            "",
        )
        or ""
    )


def require_function(
    module: Any,
    name: str,
) -> Callable[..., Any]:
    function = getattr(
        module,
        name,
        None,
    )

    if not callable(function):
        raise AttributeError(
            f"{module.__name__} does not provide {name}()."
        )

    return function


def call_typeerror_variants(
    function: Callable[..., Any],
    variants: list[
        tuple[
            tuple[Any, ...],
            dict[str, Any],
        ]
    ],
) -> Any:
    """
    Try only API-signature variants.

    A non-TypeError is treated as a real source failure and is
    immediately propagated.
    """
    last_error: TypeError | None = None

    for args, kwargs in variants:
        try:
            return function(
                *args,
                **kwargs,
            )
        except TypeError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error

    raise RuntimeError(
        "No function-call variants were configured."
    )


def load_nflreadpy(
    family: str,
    season: int,
) -> tuple[pd.DataFrame, str]:
    nfl = import_source(
        "nflreadpy"
    )

    if family == "player_stats":
        function = require_function(
            nfl,
            "load_player_stats",
        )

        data = call_typeerror_variants(
            function,
            [
                (
                    (),
                    {
                        "seasons": [season],
                        "summary_level": "week",
                    },
                ),
                (
                    (),
                    {
                        "seasons": season,
                        "summary_level": "week",
                    },
                ),
                (
                    ([season],),
                    {
                        "summary_level": "week",
                    },
                ),
            ],
        )

    elif family == "weekly_rosters":
        function = require_function(
            nfl,
            "load_rosters_weekly",
        )

        data = call_typeerror_variants(
            function,
            [
                (
                    (),
                    {"seasons": [season]},
                ),
                (
                    (),
                    {"seasons": season},
                ),
                (([season],), {}),
            ],
        )

    elif family == "snap_counts":
        function = require_function(
            nfl,
            "load_snap_counts",
        )

        data = call_typeerror_variants(
            function,
            [
                (
                    (),
                    {"seasons": [season]},
                ),
                (
                    (),
                    {"seasons": season},
                ),
                (([season],), {}),
            ],
        )

    elif family == "participation":
        function = require_function(
            nfl,
            "load_participation",
        )

        data = call_typeerror_variants(
            function,
            [
                (
                    (),
                    {"seasons": [season]},
                ),
                (
                    (),
                    {"seasons": season},
                ),
                (([season],), {}),
            ],
        )

    elif family == "players":
        function = require_function(
            nfl,
            "load_players",
        )

        data = function()

    else:
        raise ValueError(
            f"Unsupported family: {family}"
        )

    return (
        to_pandas(data),
        module_version(nfl),
    )


def load_nfl_data_py(
    family: str,
    season: int,
) -> tuple[pd.DataFrame, str]:
    nfl = import_source(
        "nfl_data_py"
    )

    if family == "player_stats":
        function = require_function(
            nfl,
            "import_weekly_data",
        )

        data = call_typeerror_variants(
            function,
            [
                (
                    ([season],),
                    {
                        "columns": None,
                        "downcast": True,
                    },
                ),
                (([season],), {}),
            ],
        )

    elif family == "weekly_rosters":
        function = require_function(
            nfl,
            "import_weekly_rosters",
        )

        data = call_typeerror_variants(
            function,
            [
                (([season],), {}),
                (
                    (),
                    {"years": [season]},
                ),
            ],
        )

    elif family == "snap_counts":
        function = require_function(
            nfl,
            "import_snap_counts",
        )

        data = call_typeerror_variants(
            function,
            [
                (([season],), {}),
                (
                    (),
                    {"years": [season]},
                ),
            ],
        )

    elif family == "participation":
        function = getattr(
            nfl,
            "import_participation",
            None,
        )

        if not callable(function):
            raise NotImplementedError(
                "nfl_data_py has no direct "
                "import_participation() loader."
            )

        data = call_typeerror_variants(
            function,
            [
                (([season],), {}),
                (
                    (),
                    {"years": [season]},
                ),
            ],
        )

    elif family == "players":
        function = require_function(
            nfl,
            "import_players",
        )

        data = function()

    else:
        raise ValueError(
            f"Unsupported family: {family}"
        )

    return (
        to_pandas(data),
        module_version(nfl),
    )


def output_filename(
    family: str,
    season: int,
) -> str:
    names = {
        "player_stats": (
            f"stats_player_week_{season}.parquet"
        ),
        "weekly_rosters": (
            f"roster_weekly_{season}.parquet"
        ),
        "snap_counts": (
            f"snap_counts_{season}.parquet"
        ),
        "participation": (
            f"pbp_participation_{season}.parquet"
        ),
        "players": "players.parquet",
    }

    return names[family]


def assert_descendant(
    path: Path,
    root: Path,
    *,
    label: str,
) -> None:
    resolved_path = path.resolve()
    resolved_root = root.resolve()

    try:
        resolved_path.relative_to(
            resolved_root
        )
    except ValueError as exc:
        raise ValueError(
            f"{label} must remain under "
            f"{resolved_root}; received "
            f"{resolved_path}"
        ) from exc


def week_bounds(
    df: pd.DataFrame,
) -> tuple[int | float | None, int | float | None]:
    if "week" not in df.columns:
        return None, None

    values = pd.to_numeric(
        df["week"],
        errors="coerce",
    ).dropna()

    if values.empty:
        return None, None

    minimum = float(
        values.min()
    )
    maximum = float(
        values.max()
    )

    min_value: int | float = (
        int(minimum)
        if minimum.is_integer()
        else minimum
    )

    max_value: int | float = (
        int(maximum)
        if maximum.is_integer()
        else maximum
    )

    return (
        min_value,
        max_value,
    )


def dataframe_metadata(
    df: pd.DataFrame,
) -> dict[str, Any]:
    min_week, max_week = week_bounds(
        df
    )

    native_id_columns = [
        column
        for column in (
            "player_id",
            "gsis_id",
            "nflverse_player_id",
            "pfr_player_id",
            "pfr_id",
            "espn_id",
        )
        if column in df.columns
    ]

    return {
        "row_count": int(len(df)),
        "column_count": int(
            len(df.columns)
        ),
        "min_week": min_week,
        "max_week": max_week,
        "native_id_columns": (
            native_id_columns
        ),
    }


def safe_error(
    exc: Exception,
) -> dict[str, str]:
    return {
        "error_type": (
            type(exc).__name__
        ),
        "error": str(exc)[:2000],
    }


def write_json_atomic(
    payload: dict[str, Any],
    path: Path,
    prop_root: Path,
) -> None:
    assert_descendant(
        path,
        prop_root,
        label="JSON log path",
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )

    temp_path = Path(
        handle.name
    )

    try:
        with handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                default=str,
            )
            handle.write("\n")

        os.replace(
            temp_path,
            path,
        )

    finally:
        if temp_path.exists():
            temp_path.unlink()


def refresh_family(
    *,
    family: str,
    season: int,
    config: dict,
    source_root: Path,
) -> dict[str, Any]:
    output_path = (
        source_root
        / output_filename(
            family,
            season,
        )
    )

    assert_descendant(
        output_path,
        source_root,
        label=f"{family} output",
    )

    result: dict[str, Any] = {
        "family": family,
        "season": season,
        "output": str(
            output_path.relative_to(
                common.repo_root()
            )
        ),
        "status": "unavailable",
        "source": None,
        "source_version": None,
        "row_count": 0,
        "column_count": 0,
        "min_week": None,
        "max_week": None,
        "refresh_timestamp": (
            utc_now()
        ),
        "output_written": False,
        "attempts": [],
    }

    loaders = (
        (
            "nflreadpy",
            load_nflreadpy,
        ),
        (
            "nfl_data_py",
            load_nfl_data_py,
        ),
    )

    for (
        source_name,
        loader,
    ) in loaders:
        attempt: dict[
            str,
            Any,
        ] = {
            "source": source_name,
        }

        try:
            df, version = loader(
                family,
                season,
            )

            metadata = (
                dataframe_metadata(
                    df
                )
            )

            attempt.update(
                {
                    "status": (
                        "success"
                        if not df.empty
                        else "empty"
                    ),
                    "source_version": (
                        version
                    ),
                    **metadata,
                }
            )

            result[
                "attempts"
            ].append(attempt)

            if df.empty:
                continue

            # Raw current-source files must not contain configured
            # sportsbook/market/external prediction fields.
            common.reject_forbidden_feature_columns(
                df.columns,
                config,
            )

            # Preserve all source columns and native IDs exactly as
            # returned. Only deterministic row ordering is applied by
            # the shared atomic parquet writer.
            common.write_parquet_atomic(
                df,
                output_path,
            )

            result.update(
                {
                    "status": "success",
                    "source": source_name,
                    "source_version": (
                        version
                    ),
                    **metadata,
                    "refresh_timestamp": (
                        utc_now()
                    ),
                    "output_written": (
                        True
                    ),
                }
            )

            return result

        except Exception as exc:
            attempt.update(
                {
                    "status": "failed",
                    **safe_error(exc),
                }
            )

            result[
                "attempts"
            ].append(
                attempt
            )

    result[
        "existing_output_preserved"
    ] = output_path.exists()

    result[
        "refresh_timestamp"
    ] = utc_now()

    return result


def main() -> int:
    args = parse_args()
    config = common.load_config()
    repo = common.repo_root()
    prop = common.prop_root()

    season = resolve_season(
        args,
        config,
    )

    source_root = (
        repo
        / config["paths"][
            "current_source_root"
        ]
    ).resolve()

    assert_descendant(
        source_root,
        prop,
        label="current source root",
    )

    source_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path = (
        repo
        / config["paths"]["log_root"]
        / (
            "refresh_nflverse_"
            f"player_data_{season}.json"
        )
    ).resolve()

    assert_descendant(
        log_path,
        prop,
        label="refresh log",
    )

    started_at = utc_now()

    results: dict[
        str,
        dict[str, Any],
    ] = {}

    for family in FAMILIES:
        results[family] = (
            refresh_family(
                family=family,
                season=season,
                config=config,
                source_root=source_root,
            )
        )

    success_count = sum(
        1
        for result in results.values()
        if result["status"] == "success"
    )

    unavailable_count = (
        len(results)
        - success_count
    )

    if success_count == len(
        results
    ):
        overall_status = "passed"
    elif success_count > 0:
        overall_status = "partial"
    else:
        overall_status = "failed"

    payload = {
        "script": (
            "refresh_nflverse_player_data.py"
        ),
        "season": season,
        "started_at": started_at,
        "completed_at": utc_now(),
        "status": overall_status,
        "policy": {
            "nflreadpy_first": True,
            "fallback_source": (
                "nfl_data_py"
            ),
            "market_data_allowed": (
                config["system"][
                    "market_data_allowed"
                ]
            ),
            "native_gsis_ids_preserved": (
                True
            ),
            "fabricate_missing_rows": (
                False
            ),
            "historical_data_writes": (
                False
            ),
        },
        "counts": {
            "families": len(
                results
            ),
            "successful_families": (
                success_count
            ),
            "unavailable_families": (
                unavailable_count
            ),
        },
        "families": results,
    }

    write_json_atomic(
        payload,
        log_path,
        prop,
    )

    common.log_run(
        "refresh_nflverse_player_data.py",
        {
            "season": season,
            "status": overall_status,
            "successful_families": (
                success_count
            ),
            "unavailable_families": (
                unavailable_count
            ),
        },
    )

    print(
        json.dumps(
            {
                "season": season,
                "status": (
                    overall_status
                ),
                "successful_families": (
                    success_count
                ),
                "unavailable_families": (
                    unavailable_count
                ),
                "log": str(
                    log_path.relative_to(
                        repo
                    )
                ),
            },
            sort_keys=True,
        )
    )

    return (
        0
        if success_count > 0
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
