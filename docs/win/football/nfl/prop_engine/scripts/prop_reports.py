#!/usr/bin/env python3
"""Build NFL Prop Engine dashboard reports from the 2026 graded prop file."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any


PROP_ENGINE_ROOT = Path("docs/win/football/nfl/prop_engine")
GRADED_INPUT = (
    PROP_ENGINE_ROOT
    / "05_final"
    / "graded"
    / "2026"
    / "2026_all_props_graded.csv"
)
DASHBOARD_ROOT = (
    PROP_ENGINE_ROOT
    / "05_final"
    / "reports"
    / "dashboard"
)

MARKETS = {
    "kicking_points": "actual_prop_total_kicking_points",
    "passing_rushing_yards": "actual_prop_total_passing_plus_rushing_yards",
    "passing_yards": "actual_prop_total_passing_yards",
    "receiving_yards": "actual_prop_total_receiving_yards",
    "rushing_receiving_yards": "actual_prop_total_rushing_plus_receiving_yards",
    "rushing_yards": "actual_prop_total_rushing_yards",
    "tackles": "actual_prop_total_tackles",
}

VARIABLES = {
    "actual_prop_total_*": {
        "filename": "prop_total.csv",
        "column": None,
        "bucket_width": 5.0,
        "display_increment": 0.1,
        "fixed_max": None,
    },
    "pick_prob": {
        "filename": "pick_prob.csv",
        "column": "pick_prob",
        "bucket_width": 0.05,
        "display_increment": 0.0001,
        "fixed_max": 1.0,
    },
    "over_prob": {
        "filename": "over_prob.csv",
        "column": "over_prob",
        "bucket_width": 0.05,
        "display_increment": 0.0001,
        "fixed_max": 1.0,
    },
    "under_prob": {
        "filename": "under_prob.csv",
        "column": "under_prob",
        "bucket_width": 0.05,
        "display_increment": 0.0001,
        "fixed_max": 1.0,
    },
}

SIDES = ("over", "under")

OUTPUT_FIELDS = [
    "market_type",
    "side",
    "variable",
    "wins",
    "losses",
    "win_rate",
    "bets",
    "pushes",
    "ungraded",
]

VALID_GRADES = {
    "win",
    "loss",
    "push",
    "ungraded",
}


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

    if not math.isfinite(number):
        return None

    return number


def format_number(value: float, decimals: int = 4) -> str:
    rounded = round(value, decimals)

    if math.isclose(
        rounded,
        round(rounded),
        abs_tol=10 ** (-(decimals + 1)),
    ):
        return str(int(round(rounded)))

    return f"{rounded:.{decimals}f}".rstrip("0").rstrip(".")


def read_graded_rows() -> list[dict[str, str]]:
    if not GRADED_INPUT.is_file():
        raise FileNotFoundError(
            f"Missing graded prop file: {GRADED_INPUT}"
        )

    with GRADED_INPUT.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(handle)

        fieldnames = set(reader.fieldnames or [])
        required = {
            "prop_type",
            "pick",
            "pick_prob",
            "over_prob",
            "under_prob",
            "grade",
            *MARKETS.values(),
        }

        missing = sorted(required - fieldnames)

        if missing:
            raise RuntimeError(
                "Graded prop file missing required columns: "
                f"{missing}"
            )

        rows: list[dict[str, str]] = []

        for raw_row in reader:
            row = {
                key: clean(value)
                for key, value in raw_row.items()
                if key is not None
            }

            if not any(row.values()):
                continue

            rows.append(row)

    return rows


def build_buckets(
    values: list[float],
    bucket_width: float,
    display_increment: float,
    fixed_max: float | None,
) -> list[tuple[float, float, str]]:
    if fixed_max is None:
        if not values:
            return []

        maximum = max(values)

        if maximum < 0:
            raise RuntimeError(
                "Dashboard bucket values cannot all be negative."
            )

        upper_limit = max(
            bucket_width,
            math.ceil(maximum / bucket_width) * bucket_width,
        )
    else:
        upper_limit = fixed_max

    bucket_count = int(
        round(upper_limit / bucket_width)
    )

    buckets: list[tuple[float, float, str]] = []

    for index in range(bucket_count):
        lower_boundary = index * bucket_width
        upper_boundary = (index + 1) * bucket_width

        if index == 0:
            label_lower = lower_boundary
        else:
            label_lower = lower_boundary + display_increment

        label = (
            f"{format_number(label_lower)}-"
            f"{format_number(upper_boundary)}"
        )

        buckets.append(
            (
                lower_boundary,
                upper_boundary,
                label,
            )
        )

    return buckets


def value_in_bucket(
    value: float,
    lower_boundary: float,
    upper_boundary: float,
    first_bucket: bool,
) -> bool:
    if first_bucket:
        return lower_boundary <= value <= upper_boundary

    return lower_boundary < value <= upper_boundary


def summarize_rows(
    rows: list[dict[str, str]],
) -> dict[str, str]:
    wins = 0
    losses = 0
    pushes = 0
    ungraded = 0

    for row in rows:
        grade = clean(
            row.get("grade")
        ).casefold()

        if grade not in VALID_GRADES:
            raise RuntimeError(
                "Unsupported grade value "
                f"{row.get('grade')!r} "
                f"for prop_type={row.get('prop_type')!r}"
            )

        if grade == "win":
            wins += 1
        elif grade == "loss":
            losses += 1
        elif grade == "push":
            pushes += 1
        else:
            ungraded += 1

    decisive = wins + losses

    win_rate = (
        ""
        if decisive == 0
        else f"{wins / decisive:.4f}"
    )

    return {
        "wins": str(wins),
        "losses": str(losses),
        "win_rate": win_rate,
        "bets": str(len(rows)),
        "pushes": str(pushes),
        "ungraded": str(ungraded),
    }


def write_report(
    market_name: str,
    market_rows: list[dict[str, str]],
    market_line_column: str,
    variable_name: str,
    variable_config: dict[str, Any],
) -> None:
    source_column = (
        market_line_column
        if variable_name == "actual_prop_total_*"
        else str(variable_config["column"])
    )

    numeric_rows: list[tuple[dict[str, str], float]] = []

    for row in market_rows:
        value = parse_float(
            row.get(source_column)
        )

        if value is None:
            continue

        numeric_rows.append(
            (
                row,
                value,
            )
        )

    values = [
        value
        for _, value in numeric_rows
    ]

    buckets = build_buckets(
        values=values,
        bucket_width=float(
            variable_config["bucket_width"]
        ),
        display_increment=float(
            variable_config["display_increment"]
        ),
        fixed_max=variable_config["fixed_max"],
    )

    output_rows: list[dict[str, str]] = []

    for side in SIDES:
        side_rows = [
            (row, value)
            for row, value in numeric_rows
            if clean(row.get("pick")).casefold() == side
        ]

        for index, (
            lower_boundary,
            upper_boundary,
            label,
        ) in enumerate(buckets):
            bucket_rows = [
                row
                for row, value in side_rows
                if value_in_bucket(
                    value=value,
                    lower_boundary=lower_boundary,
                    upper_boundary=upper_boundary,
                    first_bucket=(index == 0),
                )
            ]

            metrics = summarize_rows(
                bucket_rows
            )

            output_rows.append(
                {
                    "market_type": variable_name,
                    "side": side,
                    "variable": label,
                    **metrics,
                }
            )

    output_path = (
        DASHBOARD_ROOT
        / market_name
        / str(variable_config["filename"])
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_FIELDS,
        )

        writer.writeheader()
        writer.writerows(output_rows)


def main() -> None:
    rows = read_graded_rows()

    for market_name, market_line_column in MARKETS.items():
        market_rows = [
            row
            for row in rows
            if clean(row.get("prop_type")) == market_name
        ]

        for variable_name, variable_config in VARIABLES.items():
            write_report(
                market_name=market_name,
                market_rows=market_rows,
                market_line_column=market_line_column,
                variable_name=variable_name,
                variable_config=variable_config,
            )


if __name__ == "__main__":
    main()
