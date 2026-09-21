#!/usr/bin/env python3
"""
Build final NFL selected-bet and weekly projection outputs.

READS:
  docs/win/football/nfl/03_picks/week_{week}_NFL_picks.csv

WRITES:
  docs/win/football/nfl/03_picks/selected/
      week_{week}_NFL_select_picks.csv

  docs/win/football/nfl/03_picks/locked/
      week_{week}_NFL_select_picks_{timestamp}.csv

  docs/win/football/nfl/03_picks/projection/
      week_{week}_NFL_projection.csv

Selected output:
- Includes a game when any of these equal 1:
    ml_selected
    spread_selected
    total_selected
- Preserves the selected-market probability, implied probability, edge, EV,
  full Kelly, Kelly, and selection-reason fields required by graded reporting.
- If an existing selected file contains a game whose commence_time has passed,
  that existing row is preserved exactly and cannot be replaced or removed.

Locked output:
- Timestamped immutable copy of the selected output.
- Timestamp uses America/New_York time.

Projection output:
- Includes every game currently present in the weekly picks input.

edt_time:
- Converts commence_time from UTC to America/New_York.
- Output format is HH:MM.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
SCRIPTS_DIR = SCRIPT_DIR.parent
NFL_ROOT = SCRIPT_DIR.parents[1]
REPORT_ROOT = NFL_ROOT / "errors"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_reporter import PipelineReporter


DEFAULT_INPUT_DIR = NFL_ROOT / "03_picks"
SELECTED_OUTPUT_DIR = NFL_ROOT / "03_picks" / "selected"
LOCKED_OUTPUT_DIR = NFL_ROOT / "03_picks" / "locked"
PROJECTION_OUTPUT_DIR = NFL_ROOT / "03_picks" / "projection"

EASTERN_TZ = None

ROOT_PICKS_PATTERN = re.compile(
    r"^week_(\d+)_NFL_picks\.csv$"
)
SELECTED_LIVE_PATTERN = re.compile(
    r"^week_(\d+)_NFL_select_picks\.csv$"
)
PROJECTION_LIVE_PATTERN = re.compile(
    r"^week_(\d+)_NFL_projection\.csv$"
)


SELECTED_OUTPUT_COLUMNS = [
    "week",
    "game_id",
    "game_date",
    "edt_time",
    "away_team",
    "home_team",
    "ml_selection",
    "ml_selection_reason",
    "ml_odds_american",
    "ml_model_probability",
    "ml_implied_probability",
    "ml_edge",
    "ml_ev",
    "ml_full_kelly",
    "ml_kelly",
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
    "season",
    "season_type",
    "ml_selected",
    "spread_selected",
    "total_selected",
    "game_time",
    "commence_time",
]


PROJECTION_OUTPUT_COLUMNS = [
    "week",
    "game_id",
    "game_date",
    "edt_time",
    "away_team",
    "home_team",
    "predicted_away_score",
    "predicted_home_score",
    "predicted_total",
    "season",
    "season_type",
    "game_time",
    "commence_time",
]


REQUIRED_INPUT_COLUMNS = sorted(
    set(
        SELECTED_OUTPUT_COLUMNS
        + PROJECTION_OUTPUT_COLUMNS
        + [
            "ml_selected",
            "spread_selected",
            "total_selected",
        ]
    )
    - {"edt_time"}
)


MARKET_SELECTION_CONTRACTS = {
    "ml": {
        "allowed_sides": {
            "HOME",
            "AWAY",
        },
        "line_column": None,
    },
    "spread": {
        "allowed_sides": {
            "HOME",
            "AWAY",
        },
        "line_column": "spread_line",
    },
    "total": {
        "allowed_sides": {
            "OVER",
            "UNDER",
        },
        "line_column": "total_line",
    },
}


def load_runtime_dependencies(
    reporter: PipelineReporter,
) -> None:
    global pd, EASTERN_TZ

    try:
        import pandas as pd_module
        eastern_tz = ZoneInfo(
            "America/New_York"
        )
    except Exception:
        reporter.set_detail(
            "dependency_imports_ok",
            False,
        )
        raise

    pd = pd_module
    EASTERN_TZ = eastern_tz

    reporter.set_detail(
        "dependency_imports_ok",
        True,
    )


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


def require_columns(
    df: pd.DataFrame,
    required: list[str],
    label: str,
) -> None:
    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        fail(
            f"{label}: missing required columns: "
            f"{missing}"
        )


def selection_flag(value: Any) -> bool:
    text = clean(value)

    if not text:
        return False

    try:
        return float(text) == 1.0
    except (TypeError, ValueError):
        return False


def convert_to_eastern_time(
    value: Any,
    *,
    row_number: int,
) -> str:
    text = clean(value)

    if not text:
        fail(
            f"Row {row_number}: commence_time is blank"
        )

    try:
        timestamp = pd.to_datetime(
            text,
            utc=True,
            errors="raise",
        )
    except Exception as exc:
        fail(
            f"Row {row_number}: invalid commence_time "
            f"{value!r}: {exc}"
        )

    return (
        timestamp
        .tz_convert(EASTERN_TZ)
        .strftime("%H:%M")
    )


def add_edt_time(
    df: pd.DataFrame,
) -> pd.DataFrame:
    output = df.copy()

    output["edt_time"] = [
        convert_to_eastern_time(
            value,
            row_number=index + 2,
        )
        for index, value in enumerate(
            output["commence_time"]
        )
    ]

    return output


def build_selected_output(
    source: pd.DataFrame,
) -> pd.DataFrame:
    selected_mask = (
        source["ml_selected"].map(selection_flag)
        | source["spread_selected"].map(selection_flag)
        | source["total_selected"].map(selection_flag)
    )

    selected = source.loc[
        selected_mask
    ].copy()

    selected = add_edt_time(selected)

    return selected[
        SELECTED_OUTPUT_COLUMNS
    ].copy()


def build_projection_output(
    source: pd.DataFrame,
) -> pd.DataFrame:
    projection = add_edt_time(
        source.copy()
    )

    return projection[
        PROJECTION_OUTPUT_COLUMNS
    ].copy()


def started_mask(df: pd.DataFrame) -> pd.Series:
    kickoff = pd.to_datetime(
        df["commence_time"],
        utc=True,
        errors="coerce",
    )

    now_utc = pd.Timestamp(datetime.now(timezone.utc))
    return kickoff.notna() & kickoff.le(now_utc)


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
            f"{label}: invalid UTF-8 CSV: "
            f"{path}: {exc}"
        )

    if not header:
        fail(
            f"{label}: missing CSV header: "
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
            f"{label}: blank CSV column "
            f"name found: {path}"
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
            f"{label}: duplicate CSV "
            f"column names: {duplicates}"
        )


def read_csv(
    path: Path,
    label: str,
    *,
    allow_empty: bool = False,
) -> pd.DataFrame:
    if not path.is_file():
        fail(
            f"{label} not found: {path}"
        )

    validate_csv_header(
        path,
        label,
    )

    frame = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        encoding="utf-8-sig",
        low_memory=False,
    )

    if (
        frame.empty
        and not allow_empty
    ):
        fail(
            f"{label} contains no rows: "
            f"{path}"
        )

    return frame


def parse_positive_int(
    value: Any,
    *,
    label: str,
) -> int:
    text = clean(
        value
    )

    if not text:
        fail(
            f"{label} is blank"
        )

    try:
        number = float(
            text
        )
    except (
        TypeError,
        ValueError,
    ):
        fail(
            f"{label} must be a "
            f"positive integer; "
            f"found {value!r}"
        )

    if (
        not math.isfinite(
            number
        )
        or not number.is_integer()
        or number <= 0
    ):
        fail(
            f"{label} must be a "
            f"positive integer; "
            f"found {value!r}"
        )

    return int(
        number
    )


def parse_finite(
    value: Any,
    *,
    label: str,
) -> float:
    text = clean(
        value
    )

    if not text:
        fail(
            f"{label} is blank"
        )

    try:
        number = float(
            text
        )
    except (
        TypeError,
        ValueError,
    ):
        fail(
            f"{label} must be numeric; "
            f"found {value!r}"
        )

    if not math.isfinite(
        number
    ):
        fail(
            f"{label} must be finite; "
            f"found {value!r}"
        )

    return number


def strict_selection_flag(
    value: Any,
    *,
    label: str,
) -> int:
    number = parse_finite(
        value,
        label=label,
    )

    if number not in {
        0.0,
        1.0,
    }:
        fail(
            f"{label} must be 0 or 1; "
            f"found {value!r}"
        )

    return int(
        number
    )


def validate_identity_columns(
    frame: pd.DataFrame,
    *,
    label: str,
) -> None:
    game_ids = frame[
        "game_id"
    ].map(
        clean
    )

    if game_ids.eq("").any():
        fail(
            f"{label}: blank game_id found"
        )

    duplicates = (
        game_ids[
            game_ids.duplicated(
                keep=False
            )
        ]
        .drop_duplicates()
        .tolist()
    )

    if duplicates:
        fail(
            f"{label}: duplicate game_id "
            f"values: {duplicates[:10]}"
        )

    for column in (
        "away_team",
        "home_team",
    ):
        blank = (
            frame[column]
            .map(clean)
            .eq("")
        )

        if blank.any():
            examples = (
                game_ids[
                    blank
                ]
                .head(10)
                .tolist()
            )

            fail(
                f"{label}: blank {column} "
                f"for game_id values "
                f"{examples}"
            )


def validate_commence_times(
    frame: pd.DataFrame,
    *,
    label: str,
    validate_edt: bool,
) -> None:
    for index, row in frame.iterrows():
        row_number = index + 2

        expected_edt = (
            convert_to_eastern_time(
                row[
                    "commence_time"
                ],
                row_number=row_number,
            )
        )

        if validate_edt:
            actual_edt = clean(
                row["edt_time"]
            )

            if (
                actual_edt
                != expected_edt
            ):
                fail(
                    f"{label}: "
                    f"game_id="
                    f"{clean(row['game_id'])}: "
                    "edt_time does not match "
                    "commence_time; "
                    f"expected {expected_edt!r} "
                    f"found {actual_edt!r}"
                )


def validate_selected_market(
    row: pd.Series,
    *,
    market: str,
    game_id: str,
    label: str,
) -> int:
    spec = (
        MARKET_SELECTION_CONTRACTS[
            market
        ]
    )

    selected = (
        strict_selection_flag(
            row[
                f"{market}_selected"
            ],
            label=(
                f"{label}: "
                f"game_id={game_id}: "
                f"{market}_selected"
            ),
        )
    )

    if selected == 0:
        return 0

    selection = clean(
        row[
            f"{market}_selection"
        ]
    )

    if (
        selection
        not in spec[
            "allowed_sides"
        ]
    ):
        fail(
            f"{label}: "
            f"game_id={game_id}: "
            f"invalid {market}_selection="
            f"{selection!r}"
        )

    reason = clean(
        row[
            f"{market}_selection_reason"
        ]
    )

    if not reason:
        fail(
            f"{label}: "
            f"game_id={game_id}: "
            f"{market}_selection_reason "
            "is blank"
        )

    odds = parse_finite(
        row[
            f"{market}_odds_american"
        ],
        label=(
            f"{label}: "
            f"game_id={game_id}: "
            f"{market}_odds_american"
        ),
    )

    if odds == 0:
        fail(
            f"{label}: "
            f"game_id={game_id}: "
            f"{market}_odds_american "
            "cannot be 0"
        )

    model_probability = (
        parse_finite(
            row[
                f"{market}_model_probability"
            ],
            label=(
                f"{label}: "
                f"game_id={game_id}: "
                f"{market}_model_probability"
            ),
        )
    )

    implied_probability = (
        parse_finite(
            row[
                f"{market}_implied_probability"
            ],
            label=(
                f"{label}: "
                f"game_id={game_id}: "
                f"{market}_implied_probability"
            ),
        )
    )

    for (
        metric,
        value,
    ) in (
        (
            "model_probability",
            model_probability,
        ),
        (
            "implied_probability",
            implied_probability,
        ),
    ):
        if not (
            0.0
            <= value
            <= 1.0
        ):
            fail(
                f"{label}: "
                f"game_id={game_id}: "
                f"{market}_{metric} "
                "outside [0,1]"
            )

    for metric in (
        "edge",
        "ev",
    ):
        parse_finite(
            row[
                f"{market}_{metric}"
            ],
            label=(
                f"{label}: "
                f"game_id={game_id}: "
                f"{market}_{metric}"
            ),
        )

    full_kelly = (
        parse_finite(
            row[
                f"{market}_full_kelly"
            ],
            label=(
                f"{label}: "
                f"game_id={game_id}: "
                f"{market}_full_kelly"
            ),
        )
    )

    kelly = parse_finite(
        row[
            f"{market}_kelly"
        ],
        label=(
            f"{label}: "
            f"game_id={game_id}: "
            f"{market}_kelly"
        ),
    )

    if (
        full_kelly < 0
        or kelly < 0
    ):
        fail(
            f"{label}: "
            f"game_id={game_id}: "
            f"{market} Kelly values "
            "cannot be negative"
        )

    if (
        kelly
        > full_kelly
        + 1e-12
    ):
        fail(
            f"{label}: "
            f"game_id={game_id}: "
            f"{market}_kelly cannot "
            "exceed full_kelly"
        )

    line_column = spec[
        "line_column"
    ]

    if line_column is not None:
        parse_finite(
            row[
                line_column
            ],
            label=(
                f"{label}: "
                f"game_id={game_id}: "
                f"{line_column}"
            ),
        )

    return 1


def validate_source(
    source: pd.DataFrame,
    *,
    input_path: Path,
    requested_week: int,
) -> tuple[
    int,
    str,
]:
    label = str(
        input_path
    )

    require_columns(
        source,
        REQUIRED_INPUT_COLUMNS,
        label,
    )

    validate_identity_columns(
        source,
        label=label,
    )

    seasons: set[int] = set()
    weeks: set[int] = set()
    season_types: set[str] = set()

    for index, row in source.iterrows():
        row_number = index + 2
        game_id = clean(
            row["game_id"]
        )

        season = parse_positive_int(
            row["season"],
            label=(
                f"{label}: "
                f"row={row_number}: season"
            ),
        )

        week = parse_positive_int(
            row["week"],
            label=(
                f"{label}: "
                f"row={row_number}: week"
            ),
        )

        season_type = clean(
            row["season_type"]
        )

        if not season_type:
            fail(
                f"{label}: "
                f"row={row_number}: "
                "season_type is blank"
            )

        if week != requested_week:
            fail(
                f"{label}: "
                f"game_id={game_id}: "
                f"source week={week} "
                "does not match requested "
                f"--week={requested_week}"
            )

        seasons.add(
            season
        )
        weeks.add(
            week
        )
        season_types.add(
            season_type
        )

        selected_count = 0

        for market in (
            "ml",
            "spread",
            "total",
        ):
            selected_count += (
                validate_selected_market(
                    row,
                    market=market,
                    game_id=game_id,
                    label=label,
                )
            )

        if selected_count > 3:
            fail(
                f"{label}: "
                f"game_id={game_id}: "
                "invalid selected-market "
                "count"
            )

    if len(seasons) != 1:
        fail(
            f"{label}: source spans "
            f"multiple seasons: "
            f"{sorted(seasons)}"
        )

    if weeks != {
        requested_week
    }:
        fail(
            f"{label}: source week "
            f"integrity failed: "
            f"{sorted(weeks)}"
        )

    if len(
        season_types
    ) != 1:
        fail(
            f"{label}: source spans "
            "multiple season_type "
            f"values: "
            f"{sorted(season_types)}"
        )

    validate_commence_times(
        source,
        label=label,
        validate_edt=False,
    )

    return (
        next(iter(seasons)),
        next(
            iter(
                season_types
            )
        ),
    )


def validate_selected_frame(
    frame: pd.DataFrame,
    *,
    label: str,
    requested_week: int,
    allow_empty: bool,
) -> None:
    if list(
        frame.columns
    ) != SELECTED_OUTPUT_COLUMNS:
        fail(
            f"{label}: selected output "
            "column contract failed"
        )

    if (
        frame.empty
        and allow_empty
    ):
        return

    if frame.empty:
        fail(
            f"{label}: selected output "
            "contains no rows"
        )

    validate_identity_columns(
        frame,
        label=label,
    )

    for index, row in frame.iterrows():
        row_number = index + 2
        game_id = clean(
            row["game_id"]
        )

        week = parse_positive_int(
            row["week"],
            label=(
                f"{label}: "
                f"row={row_number}: week"
            ),
        )

        if week != requested_week:
            fail(
                f"{label}: "
                f"game_id={game_id}: "
                f"week={week} does not "
                f"match expected "
                f"{requested_week}"
            )

        parse_positive_int(
            row["season"],
            label=(
                f"{label}: "
                f"row={row_number}: season"
            ),
        )

        if not clean(
            row["season_type"]
        ):
            fail(
                f"{label}: "
                f"game_id={game_id}: "
                "season_type is blank"
            )

        selected_count = 0

        for market in (
            "ml",
            "spread",
            "total",
        ):
            selected_count += (
                validate_selected_market(
                    row,
                    market=market,
                    game_id=game_id,
                    label=label,
                )
            )

        if selected_count < 1:
            fail(
                f"{label}: "
                f"game_id={game_id}: "
                "selected output row has "
                "no selected market"
            )

    validate_commence_times(
        frame,
        label=label,
        validate_edt=True,
    )


def compare_frame(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    columns: list[str],
    label: str,
) -> None:
    if list(
        actual.columns
    ) != columns:
        fail(
            f"{label}: column contract "
            "failed"
        )

    if len(
        actual
    ) != len(
        expected
    ):
        fail(
            f"{label}: row-count mismatch; "
            f"expected={len(expected)} "
            f"actual={len(actual)}"
        )

    for column in columns:
        actual_values = [
            clean(value)
            for value in actual[
                column
            ].tolist()
        ]
        expected_values = [
            clean(value)
            for value in expected[
                column
            ].tolist()
        ]

        if (
            actual_values
            != expected_values
        ):
            fail(
                f"{label}: column "
                f"{column!r} differs "
                "from expected output"
            )


def validate_selected_output(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    label: str,
    requested_week: int,
) -> None:
    compare_frame(
        actual,
        expected,
        columns=(
            SELECTED_OUTPUT_COLUMNS
        ),
        label=label,
    )

    validate_selected_frame(
        actual,
        label=label,
        requested_week=requested_week,
        allow_empty=True,
    )


def validate_projection_output(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    label: str,
) -> None:
    compare_frame(
        actual,
        expected,
        columns=(
            PROJECTION_OUTPUT_COLUMNS
        ),
        label=label,
    )

    if actual.empty:
        fail(
            f"{label}: projection "
            "output contains no rows"
        )

    validate_identity_columns(
        actual,
        label=label,
    )

    validate_commence_times(
        actual,
        label=label,
        validate_edt=True,
    )


def preserve_started_selected_rows(
    selected_output: pd.DataFrame,
    selected_output_path: Path,
    *,
    requested_week: int,
    reporter: PipelineReporter,
) -> pd.DataFrame:
    if not selected_output_path.exists():
        reporter.set_detail(
            "preserved_started_rows",
            0,
        )
        return selected_output

    reporter.add_input(
        selected_output_path
    )

    existing = read_csv(
        selected_output_path,
        "existing selected output",
        allow_empty=True,
    )

    validate_selected_frame(
        existing,
        label=(
            "existing selected output"
        ),
        requested_week=requested_week,
        allow_empty=True,
    )

    if existing.empty:
        reporter.set_detail(
            "preserved_started_rows",
            0,
        )
        return selected_output

    existing_started = existing.loc[
        started_mask(existing)
    ].copy()

    if existing_started.empty:
        reporter.set_detail(
            "preserved_started_rows",
            0,
        )
        return selected_output

    started_game_ids = {
        clean(value)
        for value in existing_started[
            "game_id"
        ]
        if clean(value)
    }

    current_unstarted = (
        selected_output.loc[
            ~selected_output[
                "game_id"
            ]
            .map(clean)
            .isin(
                started_game_ids
            )
        ]
        .copy()
    )

    combined = pd.concat(
        [
            existing_started,
            current_unstarted,
        ],
        ignore_index=True,
    )

    combined = combined[
        SELECTED_OUTPUT_COLUMNS
    ].copy()

    validate_selected_frame(
        combined,
        label=(
            "preserved selected output"
        ),
        requested_week=requested_week,
        allow_empty=True,
    )

    reporter.set_detail(
        "preserved_started_rows",
        len(
            existing_started
        ),
    )

    return combined


def stage_frame(
    frame: pd.DataFrame,
    target_path: Path,
) -> Path:
    target_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    descriptor, raw_path = (
        tempfile.mkstemp(
            prefix=(
                f".{target_path.name}."
                "stage."
            ),
            suffix=".csv",
            dir=str(
                target_path.parent
            ),
        )
    )
    os.close(
        descriptor
    )

    staged_path = Path(
        raw_path
    )

    try:
        frame.to_csv(
            staged_path,
            index=False,
            lineterminator="\n",
            encoding="utf-8",
        )

        if (
            not staged_path.is_file()
            or staged_path.stat().st_size
            == 0
        ):
            fail(
                "Staged output was not "
                f"written: {staged_path}"
            )

        return staged_path
    except Exception:
        staged_path.unlink(
            missing_ok=True
        )
        raise


def stage_lock_copy(
    selected_stage: Path,
    locked_output_path: Path,
) -> Path:
    locked_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    descriptor, raw_path = (
        tempfile.mkstemp(
            prefix=(
                f".{locked_output_path.name}."
                "stage."
            ),
            suffix=".csv",
            dir=str(
                locked_output_path.parent
            ),
        )
    )
    os.close(
        descriptor
    )

    staged_path = Path(
        raw_path
    )

    try:
        shutil.copyfile(
            selected_stage,
            staged_path,
        )

        if (
            not staged_path.is_file()
            or staged_path.stat().st_size
            == 0
        ):
            fail(
                "Staged locked output "
                f"was not written: "
                f"{staged_path}"
            )

        return staged_path
    except Exception:
        staged_path.unlink(
            missing_ok=True
        )
        raise


def read_selected_output(
    path: Path,
    label: str,
) -> pd.DataFrame:
    return read_csv(
        path,
        label,
        allow_empty=True,
    )


def validate_serialized_selected(
    path: Path,
    expected: pd.DataFrame,
    *,
    label: str,
    requested_week: int,
) -> pd.DataFrame:
    frame = read_selected_output(
        path,
        label,
    )

    validate_selected_output(
        frame,
        expected,
        label=label,
        requested_week=requested_week,
    )

    return frame


def validate_serialized_projection(
    path: Path,
    expected: pd.DataFrame,
    *,
    label: str,
) -> pd.DataFrame:
    frame = read_csv(
        path,
        label,
    )

    validate_projection_output(
        frame,
        expected,
        label=label,
    )

    return frame


def root_source_weeks() -> set[int]:
    if not DEFAULT_INPUT_DIR.is_dir():
        fail(
            "Missing root picks "
            f"directory: {DEFAULT_INPUT_DIR}"
        )

    weeks: set[int] = set()

    for path in DEFAULT_INPUT_DIR.glob(
        "week_*_NFL_picks.csv"
    ):
        if not path.is_file():
            continue

        match = (
            ROOT_PICKS_PATTERN
            .fullmatch(
                path.name
            )
        )

        if match is None:
            continue

        week = int(
            match.group(1)
        )

        if week <= 0:
            fail(
                "Invalid root picks "
                f"filename: {path.name}"
            )

        if week in weeks:
            fail(
                "Duplicate root picks "
                f"week discovered: {week}"
            )

        weeks.add(
            week
        )

    if not weeks:
        fail(
            "No managed root picks "
            f"files found in "
            f"{DEFAULT_INPUT_DIR}"
        )

    return weeks


def stale_live_paths(
    directory: Path,
    pattern: re.Pattern[str],
    managed_weeks: set[int],
) -> list[Path]:
    if not directory.exists():
        return []

    stale: list[Path] = []

    for path in directory.iterdir():
        if not path.is_file():
            continue

        match = pattern.fullmatch(
            path.name
        )

        if match is None:
            continue

        week = int(
            match.group(1)
        )

        if week not in managed_weeks:
            stale.append(
                path.resolve()
            )

    return sorted(
        stale
    )


def unique_locked_path(
    week: int,
) -> Path:
    if EASTERN_TZ is None:
        fail(
            "Eastern timezone is not "
            "initialized"
        )

    timestamp = datetime.now(
        EASTERN_TZ
    ).strftime(
        "%Y%m%d_%H%M%S"
    )

    base = (
        LOCKED_OUTPUT_DIR
        / (
            f"week_{week}_NFL_"
            "select_picks_"
            f"{timestamp}.csv"
        )
    )

    if not base.exists():
        return base.resolve()

    counter = 1

    while True:
        candidate = (
            LOCKED_OUTPUT_DIR
            / (
                f"week_{week}_NFL_"
                "select_picks_"
                f"{timestamp}_{counter}.csv"
            )
        )

        if not candidate.exists():
            return candidate.resolve()

        counter += 1


def copy_exclusive(
    source: Path,
    destination: Path,
) -> None:
    if destination.exists():
        fail(
            "Locked output already "
            f"exists: {destination}"
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        with source.open(
            "rb"
        ) as src, destination.open(
            "xb"
        ) as dst:
            shutil.copyfileobj(
                src,
                dst,
            )
            dst.flush()
            os.fsync(
                dst.fileno()
            )
    except Exception:
        destination.unlink(
            missing_ok=True
        )
        raise


def publish_transaction(
    *,
    selected_stage: Path,
    selected_output_path: Path,
    selected_expected: pd.DataFrame,
    locked_stage: Path,
    locked_output_path: Path,
    projection_stage: Path,
    projection_output_path: Path,
    projection_expected: pd.DataFrame,
    stale_selected_paths: list[Path],
    stale_projection_paths: list[Path],
    requested_week: int,
    reporter: PipelineReporter,
) -> None:
    live_paths = [
        selected_output_path,
        projection_output_path,
        *stale_selected_paths,
        *stale_projection_paths,
    ]

    backups: dict[
        Path,
        Path,
    ] = {}

    lock_created = False
    live_modified = False
    rollback_failed = False

    reporter.update_details(
        {
            "publication_mode": (
                "transactional_selected_"
                "locked_projection_with_"
                "backup_rollback"
            ),
            "publication_completed": False,
            "post_publish_validation": False,
            "rollback_performed": False,
        }
    )

    try:
        if locked_output_path.exists():
            fail(
                "Refusing to overwrite "
                "existing locked output: "
                f"{locked_output_path}"
            )

        for path in live_paths:
            if not path.exists():
                continue

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

            backups[
                path
            ] = backup

        os.replace(
            selected_stage,
            selected_output_path,
        )
        live_modified = True

        copy_exclusive(
            locked_stage,
            locked_output_path,
        )
        lock_created = True

        os.replace(
            projection_stage,
            projection_output_path,
        )
        live_modified = True

        for path in (
            stale_selected_paths
            + stale_projection_paths
        ):
            if path.exists():
                path.unlink()
                live_modified = True

        validate_serialized_selected(
            selected_output_path,
            selected_expected,
            label=(
                "published selected output"
            ),
            requested_week=requested_week,
        )

        validate_serialized_selected(
            locked_output_path,
            selected_expected,
            label=(
                "published locked output"
            ),
            requested_week=requested_week,
        )

        validate_serialized_projection(
            projection_output_path,
            projection_expected,
            label=(
                "published projection output"
            ),
        )

        remaining_stale = [
            str(path)
            for path in (
                stale_selected_paths
                + stale_projection_paths
            )
            if path.exists()
        ]

        if remaining_stale:
            fail(
                "Stale live final-picks "
                "outputs remain after "
                f"publication: "
                f"{remaining_stale}"
            )

        reporter.update_details(
            {
                "publication_completed": True,
                "post_publish_validation": True,
                "locked_output_created": True,
            }
        )
    except Exception as publish_exc:
        if (
            live_modified
            or lock_created
        ):
            try:
                if (
                    lock_created
                    and locked_output_path.exists()
                ):
                    locked_output_path.unlink()

                for path in live_paths:
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
                        "locked_output_created": False,
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
                    "Final-picks publication "
                    "failed and rollback also "
                    "failed: "
                    f"publication_error="
                    f"{publish_exc}; "
                    f"rollback_error="
                    f"{rollback_exc}"
                ) from rollback_exc

        raise
    finally:
        for staged_path in (
            selected_stage,
            locked_stage,
            projection_stage,
        ):
            staged_path.unlink(
                missing_ok=True
            )

        if not rollback_failed:
            for backup in backups.values():
                if not backup.exists():
                    continue

                try:
                    backup.unlink()
                except Exception as exc:
                    reporter.warning(
                        "Temporary final-picks "
                        "backup cleanup failed",
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
    parser = argparse.ArgumentParser(
        description=(
            "Build final NFL selected picks and "
            "weekly projection CSVs."
        )
    )

    parser.add_argument(
        "--week",
        required=True,
        type=int,
        help="NFL week number",
    )

    return parser.parse_args()


def run(
    args: argparse.Namespace,
    reporter: PipelineReporter,
) -> None:
    load_runtime_dependencies(
        reporter
    )

    if args.week <= 0:
        fail(
            "--week must be greater than 0"
        )

    week = args.week

    input_path = (
        DEFAULT_INPUT_DIR
        / f"week_{week}_NFL_picks.csv"
    ).resolve()

    selected_output_path = (
        SELECTED_OUTPUT_DIR
        / f"week_{week}_NFL_select_picks.csv"
    ).resolve()

    projection_output_path = (
        PROJECTION_OUTPUT_DIR
        / f"week_{week}_NFL_projection.csv"
    ).resolve()

    reporter.add_input(
        input_path
    )
    reporter.add_output(
        selected_output_path
    )
    reporter.add_output(
        projection_output_path
    )

    reporter.update_details(
        {
            "requested_week": week,
            "staged_roundtrip_verified": False,
            "publication_completed": False,
            "post_publish_validation": False,
            "rollback_performed": False,
            "locked_output_created": False,
        }
    )

    source = read_csv(
        input_path,
        "weekly picks input",
    )

    (
        season,
        season_type,
    ) = validate_source(
        source,
        input_path=input_path,
        requested_week=week,
    )

    reporter.season = season
    reporter.week = week

    reporter.set_rows(
        rows_in=len(
            source
        )
    )

    selected_output = (
        build_selected_output(
            source
        )
    )

    projection_output = (
        build_projection_output(
            source
        )
    )

    selected_output = (
        preserve_started_selected_rows(
            selected_output,
            selected_output_path,
            requested_week=week,
            reporter=reporter,
        )
    )

    validate_selected_output(
        selected_output,
        selected_output,
        label=(
            "in-memory selected output"
        ),
        requested_week=week,
    )

    validate_projection_output(
        projection_output,
        projection_output,
        label=(
            "in-memory projection output"
        ),
    )

    SELECTED_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    LOCKED_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    PROJECTION_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    locked_output_path = (
        unique_locked_path(
            week
        )
    )

    reporter.add_output(
        locked_output_path
    )

    managed_weeks = (
        root_source_weeks()
    )

    if week not in managed_weeks:
        fail(
            "Requested week is not "
            "present in the managed "
            f"root picks set: {week}"
        )

    stale_selected_paths = (
        stale_live_paths(
            SELECTED_OUTPUT_DIR,
            SELECTED_LIVE_PATTERN,
            managed_weeks,
        )
    )

    stale_projection_paths = (
        stale_live_paths(
            PROJECTION_OUTPUT_DIR,
            PROJECTION_LIVE_PATTERN,
            managed_weeks,
        )
    )

    reporter.update_details(
        {
            "resolved_season": season,
            "resolved_week": week,
            "season_type": season_type,
            "input_path": str(
                input_path
            ),
            "selected_output_path": str(
                selected_output_path
            ),
            "locked_output_path": str(
                locked_output_path
            ),
            "projection_output_path": str(
                projection_output_path
            ),
            "source_rows": len(
                source
            ),
            "selected_rows": len(
                selected_output
            ),
            "projection_rows": len(
                projection_output
            ),
            "managed_source_weeks": sorted(
                managed_weeks
            ),
            "stale_selected_paths": [
                str(path)
                for path
                in stale_selected_paths
            ],
            "stale_projection_paths": [
                str(path)
                for path
                in stale_projection_paths
            ],
        }
    )

    selected_stage = (
        stage_frame(
            selected_output,
            selected_output_path,
        )
    )

    projection_stage = (
        stage_frame(
            projection_output,
            projection_output_path,
        )
    )

    locked_stage = (
        stage_lock_copy(
            selected_stage,
            locked_output_path,
        )
    )

    try:
        validate_serialized_selected(
            selected_stage,
            selected_output,
            label=(
                "staged selected output"
            ),
            requested_week=week,
        )

        validate_serialized_selected(
            locked_stage,
            selected_output,
            label=(
                "staged locked output"
            ),
            requested_week=week,
        )

        validate_serialized_projection(
            projection_stage,
            projection_output,
            label=(
                "staged projection output"
            ),
        )

        reporter.set_detail(
            "staged_roundtrip_verified",
            True,
        )

        publish_transaction(
            selected_stage=selected_stage,
            selected_output_path=(
                selected_output_path
            ),
            selected_expected=(
                selected_output
            ),
            locked_stage=locked_stage,
            locked_output_path=(
                locked_output_path
            ),
            projection_stage=(
                projection_stage
            ),
            projection_output_path=(
                projection_output_path
            ),
            projection_expected=(
                projection_output
            ),
            stale_selected_paths=(
                stale_selected_paths
            ),
            stale_projection_paths=(
                stale_projection_paths
            ),
            requested_week=week,
            reporter=reporter,
        )
    finally:
        for path in (
            selected_stage,
            locked_stage,
            projection_stage,
        ):
            path.unlink(
                missing_ok=True
            )

    reporter.set_rows(
        rows_out=len(
            projection_output
        )
    )

    print(
        f"WROTE {selected_output_path} "
        f"| rows={len(selected_output)}"
    )

    print(
        f"WROTE {locked_output_path} "
        f"| rows={len(selected_output)}"
    )

    print(
        f"WROTE {projection_output_path} "
        f"| rows={len(projection_output)}"
    )


def main() -> int:
    try:
        with PipelineReporter(
            script=SCRIPT_PATH,
            stage="03_picks",
            report_root=REPORT_ROOT,
            pipeline="NFL",
            league="NFL",
            extra_context={
                "component": (
                    "final selected and "
                    "projection outputs"
                ),
            },
        ) as reporter:
            args = parse_args()

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