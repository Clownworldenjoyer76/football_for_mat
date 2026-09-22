#!/usr/bin/env python3
"""Build NFL Prop Engine dashboard reports using the active markets configuration."""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import common
from pipeline_reporter import PipelineReporter
from props_combined import (
    PROP_LINE_COLUMNS,
    filter_rows as apply_market_filters,
    load_markets,
)


REPO_ROOT = common.repo_root().resolve()
PROP_ENGINE_ROOT = common.prop_root().resolve()
FINAL_ROOT = PROP_ENGINE_ROOT / "05_final"
GRADED_ROOT = FINAL_ROOT / "graded"
DASHBOARD_ROOT = FINAL_ROOT / "reports" / "dashboard"
MARKETS_PATH = PROP_ENGINE_ROOT / "config" / "markets.yaml"

MARKETS = dict(PROP_LINE_COLUMNS)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build NFL Prop Engine dashboard reports using "
            "the active markets configuration."
        )
    )
    parser.add_argument(
        "--season",
        type=int,
    )
    args = parser.parse_args()

    if args.season is not None and not 1900 <= args.season <= 2200:
        parser.error("--season must be between 1900 and 2200")

    return args


def resolve_season(cli_season: int | None) -> str:
    if cli_season is not None:
        return str(cli_season)

    env_season = str(os.environ.get("NFL_SEASON", "") or "").strip()
    if env_season:
        if not env_season.isdigit():
            raise RuntimeError(
                f"NFL_SEASON must be numeric; found {env_season!r}"
            )

        season = int(env_season)
        if not 1900 <= season <= 2200:
            raise RuntimeError(
                f"NFL_SEASON must be between 1900 and 2200; found {season}"
            )
        return str(season)

    available: list[int] = []

    if GRADED_ROOT.is_dir():
        for season_dir in GRADED_ROOT.iterdir():
            if not season_dir.is_dir() or not season_dir.name.isdigit():
                continue

            season = int(season_dir.name)
            graded_path = season_dir / f"{season}_all_props_graded.csv"
            if graded_path.is_file():
                available.append(season)

    if not available:
        raise FileNotFoundError(
            f"No graded season files found under {GRADED_ROOT}"
        )

    return str(max(available))


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


def read_graded_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing graded prop file: {path}"
        )

    with path.open(
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


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    handle = tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temp_path = Path(handle.name)

    try:
        with handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
            )
            writer.writeheader()
            writer.writerows(rows)

        os.replace(
            temp_path,
            path,
        )
    finally:
        if temp_path.exists():
            temp_path.unlink()


def write_report(
    market_name: str,
    market_rows: list[dict[str, str]],
    market_line_column: str,
    variable_name: str,
    variable_config: dict[str, Any],
) -> dict[str, Any]:
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

    skipped_non_numeric = len(market_rows) - len(numeric_rows)

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

    write_csv(
        output_path,
        OUTPUT_FIELDS,
        output_rows,
    )

    return {
        "output_path": output_path,
        "output_rows": int(len(output_rows)),
        "market_rows": int(len(market_rows)),
        "numeric_values": int(len(numeric_rows)),
        "skipped_non_numeric_values": int(skipped_non_numeric),
    }


def _run(reporter: PipelineReporter) -> None:
    args = parse_args()
    season = resolve_season(args.season)

    graded_input = (
        GRADED_ROOT
        / season
        / f"{season}_all_props_graded.csv"
    )

    reporter.add_input(graded_input)
    reporter.add_input(MARKETS_PATH)

    markets = load_markets()
    input_rows = read_graded_rows(graded_input)

    filtered_rows = apply_market_filters(
        input_rows,
        markets,
    )

    market_row_counts: dict[str, int] = {}
    output_stats: dict[str, dict[str, Any]] = {}
    skipped_non_numeric_values = 0
    total_output_rows = 0

    for market_name, market_line_column in MARKETS.items():
        market_rows = [
            row
            for row in filtered_rows
            if clean(row.get("prop_type")) == market_name
        ]
        market_row_counts[market_name] = int(len(market_rows))

        for variable_name, variable_config in VARIABLES.items():
            stats = write_report(
                market_name=market_name,
                market_rows=market_rows,
                market_line_column=market_line_column,
                variable_name=variable_name,
                variable_config=variable_config,
            )

            output_path = stats.pop("output_path")
            if not isinstance(output_path, Path):
                raise RuntimeError(
                    "Dashboard output path statistics are invalid."
                )

            reporter.add_output(output_path)

            key = (
                f"{market_name}/"
                f"{variable_config['filename']}"
            )
            output_stats[key] = {
                **stats,
                "output_file": (
                    output_path
                    .relative_to(REPO_ROOT)
                    .as_posix()
                ),
            }

            skipped_non_numeric_values += int(
                stats["skipped_non_numeric_values"]
            )
            total_output_rows += int(
                stats["output_rows"]
            )

    reporter.set_rows(
        rows_in=int(len(input_rows)),
        rows_out=int(len(filtered_rows)),
    )
    reporter.update_details(
        {
            "season": int(season),
            "graded_input": (
                graded_input
                .relative_to(REPO_ROOT)
                .as_posix()
            ),
            "input_rows": int(len(input_rows)),
            "market_filtered_rows": int(len(filtered_rows)),
            "market_row_counts": market_row_counts,
            "dashboard_files": int(len(output_stats)),
            "dashboard_output_rows": int(total_output_rows),
            "skipped_non_numeric_values": int(
                skipped_non_numeric_values
            ),
            "output_stats": output_stats,
        }
    )

    if skipped_non_numeric_values:
        reporter.warning(
            "Dashboard report generation skipped non-numeric values.",
            season=int(season),
            skipped_non_numeric_values=int(
                skipped_non_numeric_values
            ),
        )

    print(
        "PROP REPORTS: PASS "
        f"season={season} "
        f"rows_in={len(input_rows)} "
        f"filtered={len(filtered_rows)} "
        f"outputs={len(output_stats)} "
        f"skipped_values={skipped_non_numeric_values}"
    )


def main() -> None:
    with PipelineReporter(
        script=Path(__file__).name,
        stage="props",
        report_root=common.prop_root() / "logs" / "pipeline_reports",
    ) as reporter:
        _run(reporter)


if __name__ == "__main__":
    main()
