#!/usr/bin/env python3
"""Prepare the cumulative NFL graded-bet master for reporting."""

from __future__ import annotations

import csv
import math
import os
import shutil
import sys
import tempfile
import uuid
from datetime import datetime
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


pd = None

INPUT_FILE = NFL_ROOT / "04_final_results" / "results" / "graded" / "NFL_final.csv"
OUTPUT_DIR = NFL_ROOT / "04_final_results" / "intermediate"
OUTPUT_FILE = OUTPUT_DIR / "work_nfl.csv"

INPUT_COLUMNS = [
    "season", "season_type", "week", "game_id", "game_date",
    "away_team", "home_team", "away_score", "home_score", "status",
    "market", "selection", "line", "odds_american", "result",
    "game_time", "commence_time", "edt_time",
    "market_type", "bet_side", "model_prob", "implied_prob", "edge", "ev",
    "full_kelly", "kelly", "selection_reason",
    "implied_prob_source", "edge_source", "ev_source", "full_kelly_source", "kelly_source",
    "final_total", "bet_result", "bet_units",
    "selected_source_file", "grading_generated_at_utc",
]

ENRICHED_COLUMNS = [
    "side_group",
    "odds_value",
    "ev_value",
    "model_prob_value",
    "kelly_value",
    "kelly_value_source",
    "spread_value",
    "total_value",
    "day_night",
    "ev_bucket",
    "odds_bucket",
    "kelly_bucket",
    "win_prob_bucket",
    "spread_range_bucket",
    "spread_line_bucket",
    "total_range_bucket",
    "total_line_bucket",
    "week_label",
]

OUTPUT_COLUMNS = INPUT_COLUMNS + ENRICHED_COLUMNS

BET_KEY = [
    "season",
    "season_type",
    "week",
    "game_id",
    "market_type",
    "bet_side",
    "line",
]

VALID_SIDES = {
    "moneyline": {"home", "away"},
    "spread": {"home", "away"},
    "total": {"over", "under"},
}

VALID_MARKETS = {
    "moneyline": "MONEYLINE",
    "spread": "SPREAD",
    "total": "TOTAL",
}

VALID_RESULTS = {
    "Win": "WIN",
    "Loss": "LOSS",
    "Push": "PUSH",
}

TIME_FORMATS = (
    "%H:%M",
    "%H:%M:%S",
    "%I:%M %p",
    "%I:%M%p",
)

EPSILON = 1e-9


def load_runtime_dependencies(
    reporter: PipelineReporter,
) -> None:
    global pd

    try:
        import pandas as pd_module
    except Exception:
        reporter.set_detail(
            "dependency_imports_ok",
            False,
        )
        raise

    pd = pd_module

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
        "nat",
        "<na>",
    }:
        return ""

    return text


def to_float(value: Any) -> float | None:
    try:
        result = float(str(value).strip())
        return result if math.isfinite(result) else None
    except Exception:
        return None


def side_group(row: pd.Series) -> str:
    market = str(row.get("market_type", "")).strip().lower()
    side = str(row.get("bet_side", "")).strip().lower()
    if market in {"moneyline", "spread"}:
        return {"home": "HOME", "away": "AWAY"}.get(side, "")
    if market == "total":
        return {"over": "OVER", "under": "UNDER"}.get(side, "")
    return ""


def day_night(row: pd.Series) -> str:
    raw = str(row.get("edt_time", "") or row.get("game_time", "")).strip()
    if not raw:
        return ""
    for fmt in ("%H:%M", "%H:%M:%S", "%I:%M %p", "%I:%M%p"):
        try:
            parsed = datetime.strptime(raw, fmt)
            return "Day" if parsed.hour < 17 else "Night"
        except ValueError:
            continue
    return ""


def ev_bucket(value: Any) -> str:
    value = to_float(value)
    if value is None: return "UNBUCKETED"
    if value < 0: return "<0"
    if value < 0.01: return "0.00_to_0.0099"
    if value < 0.02: return "0.01_to_0.0199"
    if value < 0.03: return "0.02_to_0.0299"
    if value < 0.04: return "0.03_to_0.0399"
    if value < 0.05: return "0.04_to_0.0499"
    if value < 0.075: return "0.05_to_0.0749"
    if value < 0.10: return "0.075_to_0.0999"
    return "0.10_plus"


def odds_bucket(value: Any) -> str:
    value = to_float(value)
    if value is None: return "UNBUCKETED"
    if value <= -200: return "minus_200_or_lower"
    if value <= -150: return "minus_199_to_minus_150"
    if value <= -125: return "minus_149_to_minus_125"
    if value <= -110: return "minus_124_to_minus_110"
    if value <= -101: return "minus_109_to_minus_101"
    if value <= 100: return "minus_100_to_plus_100"
    if value <= 125: return "plus_101_to_plus_125"
    if value <= 150: return "plus_126_to_plus_150"
    if value <= 200: return "plus_151_to_plus_200"
    return "plus_201_or_higher"


def kelly_bucket(value: Any) -> str:
    value = to_float(value)
    if value is None: return "UNBUCKETED"
    if value <= 0: return "zero_or_below"
    if value < 0.01: return "0.001_to_0.0099"
    if value < 0.02: return "0.01_to_0.0199"
    if value < 0.03: return "0.02_to_0.0299"
    if value < 0.05: return "0.03_to_0.0499"
    if value < 0.10: return "0.05_to_0.0999"
    if value < 0.15: return "0.10_to_0.1499"
    if value < 0.20: return "0.15_to_0.1999"
    return "0.20_plus"


def model_prob_bucket(value: Any) -> str:
    value = to_float(value)
    if value is None: return "UNBUCKETED"
    pct = value * 100.0 if value <= 1.0 else value
    if pct < 50: return "<50"
    if pct < 55: return "50_to_54.9"
    if pct < 60: return "55_to_59.9"
    if pct < 65: return "60_to_64.9"
    if pct < 70: return "65_to_69.9"
    if pct < 75: return "70_to_74.9"
    if pct < 80: return "75_to_79.9"
    return "80_plus"


def spread_range_bucket(value: Any) -> str:
    value = to_float(value)
    if value is None: return "UNBUCKETED"
    absolute = abs(value)
    if absolute <= 2.5: return "0_to_2.5"
    if absolute <= 3.5: return "3_to_3.5"
    if absolute <= 6.5: return "4_to_6.5"
    if absolute <= 9.5: return "7_to_9.5"
    if absolute <= 13.5: return "10_to_13.5"
    return "14_plus"


def spread_line_bucket(value: Any) -> str:
    value = to_float(value)
    if value is None: return "UNBUCKETED"
    return f"{value:+g}"


def total_range_bucket(value: Any) -> str:
    value = to_float(value)
    if value is None: return "UNBUCKETED"
    if value <= 37.5: return "37.5_or_lower"
    if value <= 40.5: return "38_to_40.5"
    if value <= 43.5: return "41_to_43.5"
    if value <= 46.5: return "44_to_46.5"
    if value <= 49.5: return "47_to_49.5"
    if value <= 52.5: return "50_to_52.5"
    return "53_plus"


def total_line_bucket(value: Any) -> str:
    value = to_float(value)
    if value is None: return "UNBUCKETED"
    return f"{value:g}"


def effective_kelly(row: pd.Series) -> tuple[float | None, str]:
    selected = to_float(row.get("kelly"))
    if selected is not None:
        return selected, "selected_kelly"
    full = to_float(row.get("full_kelly"))
    if full is not None:
        return full, "full_kelly_fallback"
    return None, ""


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    for column in ["market_type", "bet_side"]:
        if column not in work.columns:
            work[column] = ""
        work[column] = work[column].astype(str).str.strip().str.lower()

    if "bet_result" not in work.columns:
        work["bet_result"] = ""
    work["bet_result"] = work["bet_result"].astype(str).str.strip().str.title()

    work["side_group"] = work.apply(side_group, axis=1)
    work["odds_value"] = work.get("odds_american", pd.Series(index=work.index, dtype=object)).map(to_float)
    work["ev_value"] = work.get("ev", pd.Series(index=work.index, dtype=object)).map(to_float)
    work["model_prob_value"] = work.get("model_prob", pd.Series(index=work.index, dtype=object)).map(to_float)

    kelly_pairs = work.apply(effective_kelly, axis=1)
    work["kelly_value"] = [pair[0] for pair in kelly_pairs]
    work["kelly_value_source"] = [pair[1] for pair in kelly_pairs]

    work["spread_value"] = work.apply(
        lambda row: to_float(row.get("line")) if row.get("market_type") == "spread" else None,
        axis=1,
    )
    work["total_value"] = work.apply(
        lambda row: to_float(row.get("line")) if row.get("market_type") == "total" else None,
        axis=1,
    )
    work["day_night"] = work.apply(day_night, axis=1)

    work["ev_bucket"] = work["ev_value"].map(ev_bucket)
    work["odds_bucket"] = work["odds_value"].map(odds_bucket)
    work["kelly_bucket"] = work["kelly_value"].map(kelly_bucket)
    work["win_prob_bucket"] = work["model_prob_value"].map(model_prob_bucket)
    work["spread_range_bucket"] = work["spread_value"].map(spread_range_bucket)
    work["spread_line_bucket"] = work["spread_value"].map(spread_line_bucket)
    work["total_range_bucket"] = work["total_value"].map(total_range_bucket)
    work["total_line_bucket"] = work["total_value"].map(total_line_bucket)

    work["week_label"] = work.get("week", pd.Series(index=work.index, dtype=object)).map(
        lambda value: f"Week {str(value).strip()}" if str(value).strip() else "UNBUCKETED"
    )
    return work


def validate_csv_header(
    path: Path,
    *,
    expected: list[str],
    label: str,
) -> None:
    if not path.is_file():
        fail(f"{label}: file not found: {path}")

    try:
        with path.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            header = next(csv.reader(handle), [])
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

    if any(not column for column in normalized):
        fail(
            f"{label}: blank CSV header "
            f"column: {path}"
        )

    seen: set[str] = set()
    duplicates: list[str] = []

    for column in normalized:
        if column in seen and column not in duplicates:
            duplicates.append(column)
        seen.add(column)

    if duplicates:
        fail(
            f"{label}: duplicate header "
            f"columns: {duplicates}"
        )

    if header != expected:
        missing = [
            column
            for column in expected
            if column not in header
        ]
        unexpected = [
            column
            for column in header
            if column not in expected
        ]
        fail(
            f"{label}: column contract failed; "
            f"missing={missing} "
            f"unexpected={unexpected} "
            f"expected_order={expected} "
            f"actual_order={header}"
        )


def read_input() -> pd.DataFrame:
    validate_csv_header(
        INPUT_FILE,
        expected=INPUT_COLUMNS,
        label="graded input",
    )

    try:
        frame = pd.read_csv(
            INPUT_FILE,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            encoding="utf-8-sig",
        )
    except Exception as exc:
        fail(
            "graded input: CSV read failed: "
            f"{type(exc).__name__}: {exc}"
        )

    return frame


def parse_positive_int(
    value: Any,
    *,
    label: str,
) -> int:
    number = to_float(value)

    if (
        number is None
        or not number.is_integer()
        or number <= 0
    ):
        fail(
            f"{label} must be a positive "
            f"integer; found {value!r}"
        )

    return int(number)


def parse_nonnegative_integer(
    value: Any,
    *,
    label: str,
) -> float:
    number = to_float(value)

    if (
        number is None
        or number < 0
        or not number.is_integer()
    ):
        fail(
            f"{label} must be a nonnegative "
            f"integer; found {value!r}"
        )

    return number


def require_finite_if_present(
    value: Any,
    *,
    label: str,
) -> float | None:
    text = clean(value)

    if not text:
        return None

    number = to_float(text)

    if number is None:
        fail(
            f"{label} must be finite when "
            f"present; found {value!r}"
        )

    return number


def validate_clock_if_present(
    value: Any,
    *,
    label: str,
) -> None:
    text = clean(value)

    if not text:
        return

    for fmt in TIME_FORMATS:
        try:
            datetime.strptime(
                text,
                fmt,
            )
            return
        except ValueError:
            continue

    fail(
        f"{label} has invalid clock time "
        f"{text!r}"
    )


def validate_iso_datetime_if_present(
    value: Any,
    *,
    label: str,
) -> None:
    text = clean(value)

    if not text:
        return

    candidate = (
        text[:-1] + "+00:00"
        if text.endswith("Z")
        else text
    )

    try:
        datetime.fromisoformat(
            candidate
        )
    except ValueError:
        fail(
            f"{label} has invalid ISO "
            f"datetime {text!r}"
        )


def is_final_status(
    value: Any,
) -> bool:
    status = clean(value).casefold()

    return (
        status.startswith("final")
        or status in {
            "completed",
            "complete",
            "game over",
        }
    )


def expected_units(
    odds: float,
    bet_result: str,
) -> float:
    if bet_result == "Push":
        return 0.0

    if bet_result == "Loss":
        return -1.0

    return (
        odds / 100.0
        if odds > 0
        else 100.0 / abs(odds)
    )


def validate_input_frame(
    frame: pd.DataFrame,
) -> None:
    if list(frame.columns) != INPUT_COLUMNS:
        fail(
            "graded input: DataFrame column "
            "contract failed"
        )

    seen_keys: set[
        tuple[str, ...]
    ] = set()

    for index, row in frame.iterrows():
        row_number = index + 2

        for column in (
            "season",
            "season_type",
            "week",
            "game_id",
            "game_date",
            "away_team",
            "home_team",
            "away_score",
            "home_score",
            "status",
            "market",
            "selection",
            "odds_american",
            "result",
            "market_type",
            "bet_side",
            "bet_result",
            "bet_units",
            "selected_source_file",
            "grading_generated_at_utc",
        ):
            if not clean(row.get(column)):
                fail(
                    f"graded input row "
                    f"{row_number}: "
                    f"{column} is blank"
                )

        parse_positive_int(
            row["season"],
            label=(
                f"graded input row "
                f"{row_number} season"
            ),
        )
        parse_positive_int(
            row["week"],
            label=(
                f"graded input row "
                f"{row_number} week"
            ),
        )

        try:
            datetime.strptime(
                clean(row["game_date"]),
                "%Y-%m-%d",
            )
        except ValueError:
            fail(
                f"graded input row "
                f"{row_number}: invalid "
                f"game_date="
                f"{clean(row['game_date'])!r}"
            )

        away_team = clean(
            row["away_team"]
        )
        home_team = clean(
            row["home_team"]
        )

        if away_team == home_team:
            fail(
                f"graded input row "
                f"{row_number}: away_team "
                "and home_team must differ"
            )

        away_score = parse_nonnegative_integer(
            row["away_score"],
            label=(
                f"graded input row "
                f"{row_number} away_score"
            ),
        )
        home_score = parse_nonnegative_integer(
            row["home_score"],
            label=(
                f"graded input row "
                f"{row_number} home_score"
            ),
        )

        if not is_final_status(
            row["status"]
        ):
            fail(
                f"graded input row "
                f"{row_number}: status is "
                f"not final: "
                f"{clean(row['status'])!r}"
            )

        market_type = clean(
            row["market_type"]
        )

        if market_type not in VALID_SIDES:
            fail(
                f"graded input row "
                f"{row_number}: invalid "
                f"market_type="
                f"{market_type!r}"
            )

        if (
            market_type
            != market_type.lower()
        ):
            fail(
                f"graded input row "
                f"{row_number}: market_type "
                "must be canonical lowercase"
            )

        bet_side = clean(
            row["bet_side"]
        )

        if (
            bet_side
            not in VALID_SIDES[
                market_type
            ]
        ):
            fail(
                f"graded input row "
                f"{row_number}: invalid "
                f"bet_side={bet_side!r} "
                f"for market_type="
                f"{market_type!r}"
            )

        selection = clean(
            row["selection"]
        )

        if (
            selection
            != bet_side.upper()
        ):
            fail(
                f"graded input row "
                f"{row_number}: selection "
                "does not match bet_side"
            )

        if clean(
            row["market"]
        ) != VALID_MARKETS[
            market_type
        ]:
            fail(
                f"graded input row "
                f"{row_number}: market does "
                "not match market_type"
            )

        bet_result = clean(
            row["bet_result"]
        )

        if bet_result not in VALID_RESULTS:
            fail(
                f"graded input row "
                f"{row_number}: invalid "
                f"bet_result="
                f"{bet_result!r}"
            )

        if clean(
            row["result"]
        ) != VALID_RESULTS[
            bet_result
        ]:
            fail(
                f"graded input row "
                f"{row_number}: result does "
                "not match bet_result"
            )

        odds = to_float(
            row["odds_american"]
        )

        if (
            odds is None
            or odds == 0
        ):
            fail(
                f"graded input row "
                f"{row_number}: invalid "
                "odds_american"
            )

        line = clean(
            row["line"]
        )

        if market_type in {
            "spread",
            "total",
        }:
            if to_float(line) is None:
                fail(
                    f"graded input row "
                    f"{row_number}: invalid "
                    "spread/total line"
                )
        elif line:
            fail(
                f"graded input row "
                f"{row_number}: moneyline "
                "line must be blank"
            )

        for column in (
            "model_prob",
            "implied_prob",
            "edge",
            "ev",
            "full_kelly",
            "kelly",
        ):
            require_finite_if_present(
                row[column],
                label=(
                    f"graded input row "
                    f"{row_number} {column}"
                ),
            )

        final_total = to_float(
            row["final_total"]
        )

        if (
            final_total is None
            or abs(
                final_total
                - (
                    away_score
                    + home_score
                )
            )
            > EPSILON
        ):
            fail(
                f"graded input row "
                f"{row_number}: final_total "
                "does not match scores"
            )

        units = to_float(
            row["bet_units"]
        )

        if units is None:
            fail(
                f"graded input row "
                f"{row_number}: invalid "
                "bet_units"
            )

        expected = expected_units(
            odds,
            bet_result,
        )

        if abs(
            units - expected
        ) > EPSILON:
            fail(
                f"graded input row "
                f"{row_number}: bet_units "
                "does not match odds/result"
            )

        validate_clock_if_present(
            row["edt_time"],
            label=(
                f"graded input row "
                f"{row_number} edt_time"
            ),
        )
        validate_clock_if_present(
            row["game_time"],
            label=(
                f"graded input row "
                f"{row_number} game_time"
            ),
        )
        validate_iso_datetime_if_present(
            row["commence_time"],
            label=(
                f"graded input row "
                f"{row_number} commence_time"
            ),
        )
        validate_iso_datetime_if_present(
            row["grading_generated_at_utc"],
            label=(
                f"graded input row "
                f"{row_number} "
                "grading_generated_at_utc"
            ),
        )

        key = tuple(
            clean(
                row.get(
                    column,
                    "",
                )
            )
            for column in BET_KEY
        )

        if key in seen_keys:
            fail(
                f"graded input row "
                f"{row_number}: duplicate "
                f"bet key={key}"
            )

        seen_keys.add(
            key
        )


def numeric_equal(
    actual: Any,
    expected: float | None,
) -> bool:
    actual_value = to_float(
        actual
    )

    if expected is None:
        return actual_value is None

    return (
        actual_value is not None
        and abs(
            actual_value - expected
        ) <= EPSILON
    )


def validate_output_frame(
    work: pd.DataFrame,
    source: pd.DataFrame,
) -> None:
    if list(work.columns) != OUTPUT_COLUMNS:
        fail(
            "analyzer output: column "
            "contract failed"
        )

    if len(work) != len(source):
        fail(
            "analyzer output: row-count "
            f"mismatch source={len(source)} "
            f"output={len(work)}"
        )

    seen_keys: set[
        tuple[str, ...]
    ] = set()

    for index in work.index:
        row_number = int(index) + 2
        row = work.loc[index]
        source_row = source.loc[index]

        for column in INPUT_COLUMNS:
            if clean(
                row.get(
                    column,
                    "",
                )
            ) != clean(
                source_row.get(
                    column,
                    "",
                )
            ):
                fail(
                    f"analyzer output row "
                    f"{row_number}: source "
                    f"column changed: "
                    f"{column}"
                )

        expected_side = side_group(
            source_row
        )

        if clean(
            row["side_group"]
        ) != expected_side:
            fail(
                f"analyzer output row "
                f"{row_number}: side_group "
                "contract failed"
            )

        odds = to_float(
            source_row[
                "odds_american"
            ]
        )
        ev = to_float(
            source_row[
                "ev"
            ]
        )
        model_prob = to_float(
            source_row[
                "model_prob"
            ]
        )
        kelly_value, kelly_source = (
            effective_kelly(
                source_row
            )
        )

        if not numeric_equal(
            row["odds_value"],
            odds,
        ):
            fail(
                f"analyzer output row "
                f"{row_number}: odds_value "
                "contract failed"
            )

        if not numeric_equal(
            row["ev_value"],
            ev,
        ):
            fail(
                f"analyzer output row "
                f"{row_number}: ev_value "
                "contract failed"
            )

        if not numeric_equal(
            row["model_prob_value"],
            model_prob,
        ):
            fail(
                f"analyzer output row "
                f"{row_number}: "
                "model_prob_value contract "
                "failed"
            )

        if not numeric_equal(
            row["kelly_value"],
            kelly_value,
        ):
            fail(
                f"analyzer output row "
                f"{row_number}: kelly_value "
                "contract failed"
            )

        if clean(
            row["kelly_value_source"]
        ) != kelly_source:
            fail(
                f"analyzer output row "
                f"{row_number}: "
                "kelly_value_source "
                "contract failed"
            )

        market_type = clean(
            source_row[
                "market_type"
            ]
        )
        line = to_float(
            source_row[
                "line"
            ]
        )

        expected_spread = (
            line
            if market_type == "spread"
            else None
        )
        expected_total = (
            line
            if market_type == "total"
            else None
        )

        if not numeric_equal(
            row["spread_value"],
            expected_spread,
        ):
            fail(
                f"analyzer output row "
                f"{row_number}: spread_value "
                "contract failed"
            )

        if not numeric_equal(
            row["total_value"],
            expected_total,
        ):
            fail(
                f"analyzer output row "
                f"{row_number}: total_value "
                "contract failed"
            )

        expected_day_night = (
            day_night(
                source_row
            )
        )

        if clean(
            row["day_night"]
        ) != expected_day_night:
            fail(
                f"analyzer output row "
                f"{row_number}: day_night "
                "contract failed"
            )

        expected_text = {
            "ev_bucket": ev_bucket(
                ev
            ),
            "odds_bucket": odds_bucket(
                odds
            ),
            "kelly_bucket": kelly_bucket(
                kelly_value
            ),
            "win_prob_bucket": (
                model_prob_bucket(
                    model_prob
                )
            ),
            "spread_range_bucket": (
                spread_range_bucket(
                    expected_spread
                )
            ),
            "spread_line_bucket": (
                spread_line_bucket(
                    expected_spread
                )
            ),
            "total_range_bucket": (
                total_range_bucket(
                    expected_total
                )
            ),
            "total_line_bucket": (
                total_line_bucket(
                    expected_total
                )
            ),
            "week_label": (
                f"Week "
                f"{clean(source_row['week'])}"
                if clean(
                    source_row[
                        "week"
                    ]
                )
                else "UNBUCKETED"
            ),
        }

        for column, expected_value in (
            expected_text.items()
        ):
            if clean(
                row[column]
            ) != expected_value:
                fail(
                    f"analyzer output row "
                    f"{row_number}: "
                    f"{column} contract "
                    "failed"
                )

        key = tuple(
            clean(
                row.get(
                    column,
                    "",
                )
            )
            for column in BET_KEY
        )

        if key in seen_keys:
            fail(
                f"analyzer output row "
                f"{row_number}: duplicate "
                f"bet key={key}"
            )

        seen_keys.add(
            key
        )


def read_output(
    path: Path,
) -> pd.DataFrame:
    validate_csv_header(
        path,
        expected=OUTPUT_COLUMNS,
        label="analyzer output",
    )

    try:
        return pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            encoding="utf-8-sig",
        )
    except Exception as exc:
        fail(
            "analyzer output: CSV read "
            "failed: "
            f"{type(exc).__name__}: {exc}"
        )


def stage_output(
    work: pd.DataFrame,
    source: pd.DataFrame,
) -> Path:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    descriptor, raw_path = (
        tempfile.mkstemp(
            prefix=(
                f".{OUTPUT_FILE.name}."
                "stage."
            ),
            suffix=".csv",
            dir=str(
                OUTPUT_DIR
            ),
        )
    )
    os.close(
        descriptor
    )

    stage_path = Path(
        raw_path
    )

    try:
        work.to_csv(
            stage_path,
            index=False,
            lineterminator="\n",
        )

        with stage_path.open(
            "rb",
        ) as handle:
            os.fsync(
                handle.fileno()
            )

        staged = read_output(
            stage_path
        )
        validate_output_frame(
            staged,
            source,
        )

        return stage_path
    except Exception:
        stage_path.unlink(
            missing_ok=True
        )
        raise


def publish_output(
    stage_path: Path,
    source: pd.DataFrame,
    *,
    reporter: PipelineReporter,
) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    backup_root = Path(
        tempfile.mkdtemp(
            prefix="nfl_analyze_backup_"
        )
    )
    backup_path = (
        backup_root
        / OUTPUT_FILE.name
    )
    had_existing = (
        OUTPUT_FILE.exists()
    )
    published = False

    reporter.update_details({
        "publication_mode": (
            "atomic_replace_with_rollback"
        ),
        "publication_completed": False,
        "post_publish_validation": False,
        "rollback_performed": False,
    })

    try:
        if had_existing:
            shutil.copy2(
                OUTPUT_FILE,
                backup_path,
            )

        os.replace(
            stage_path,
            OUTPUT_FILE,
        )
        published = True

        live = read_output(
            OUTPUT_FILE
        )
        validate_output_frame(
            live,
            source,
        )

        reporter.update_details({
            "publication_completed": True,
            "post_publish_validation": True,
        })
    except Exception as publish_exc:
        if published:
            try:
                if (
                    had_existing
                    and backup_path.exists()
                ):
                    descriptor, raw_restore = (
                        tempfile.mkstemp(
                            prefix=(
                                f".{OUTPUT_FILE.name}."
                                "restore."
                            ),
                            suffix=".tmp",
                            dir=str(
                                OUTPUT_DIR
                            ),
                        )
                    )
                    os.close(
                        descriptor
                    )
                    restore_path = Path(
                        raw_restore
                    )

                    try:
                        shutil.copy2(
                            backup_path,
                            restore_path,
                        )
                        os.replace(
                            restore_path,
                            OUTPUT_FILE,
                        )
                    finally:
                        restore_path.unlink(
                            missing_ok=True
                        )
                elif OUTPUT_FILE.exists():
                    OUTPUT_FILE.unlink()

                reporter.update_details({
                    "publication_completed": False,
                    "post_publish_validation": False,
                    "rollback_performed": True,
                })
            except Exception as rollback_exc:
                reporter.update_details({
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
                })

                raise RuntimeError(
                    "NFL analyzer publication "
                    "failed and rollback also "
                    "failed: "
                    f"publication_error="
                    f"{publish_exc}; "
                    f"rollback_error="
                    f"{rollback_exc}"
                ) from rollback_exc

        raise
    finally:
        stage_path.unlink(
            missing_ok=True
        )

        try:
            shutil.rmtree(
                backup_root,
                ignore_errors=False,
            )
        except Exception as cleanup_exc:
            reporter.warning(
                "Temporary NFL analyzer "
                "backup cleanup failed",
                backup_root=str(
                    backup_root
                ),
                error_type=(
                    type(
                        cleanup_exc
                    ).__name__
                ),
                error=str(
                    cleanup_exc
                ),
            )


def run(
    reporter: PipelineReporter,
) -> None:
    reporter.add_input(
        INPUT_FILE
    )
    reporter.add_output(
        OUTPUT_FILE
    )

    load_runtime_dependencies(
        reporter
    )

    frame = read_input()
    validate_input_frame(
        frame
    )

    work = prepare(
        frame
    )
    validate_output_frame(
        work,
        frame,
    )

    reporter.set_rows(
        rows_in=len(frame),
        rows_out=len(work),
    )

    reporter.update_details({
        "input_columns": len(
            INPUT_COLUMNS
        ),
        "output_columns": len(
            OUTPUT_COLUMNS
        ),
        "input_rows_validated": len(
            frame
        ),
        "output_rows_validated": len(
            work
        ),
        "staged_roundtrip_verified": False,
        "publication_completed": False,
        "post_publish_validation": False,
        "rollback_performed": False,
    })

    stage_path = stage_output(
        work,
        frame,
    )

    reporter.set_detail(
        "staged_roundtrip_verified",
        True,
    )

    publish_output(
        stage_path,
        frame,
        reporter=reporter,
    )

    print(
        "NFL analyze complete. "
        f"rows={len(work)} "
        f"output={OUTPUT_FILE}"
    )


def main() -> int:
    try:
        with PipelineReporter(
            script=SCRIPT_PATH,
            stage="04_final_results",
            report_root=REPORT_ROOT,
            pipeline="NFL",
            league="NFL",
            extra_context={
                "component": (
                    "graded-bet analyzer"
                ),
            },
        ) as reporter:
            run(
                reporter
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
    raise SystemExit(
        main()
    )
