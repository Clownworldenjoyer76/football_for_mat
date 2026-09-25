#!/usr/bin/env python3
"""Organize weekly prop-selection CSVs into final prop-pick files."""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
from pathlib import Path

import common
from pipeline_reporter import PipelineReporter


REPO_ROOT = common.repo_root().resolve()
PROP_ENGINE_ROOT = common.prop_root().resolve()
OUTPUT_ROOT = PROP_ENGINE_ROOT / "output"
FINAL_ROOT = PROP_ENGINE_ROOT / "prop_picks_final"


SIMPLE_OUTPUTS = {
    "defense": {
        "actual": "actual_prop_total_tackles",
        "engine": "prop_engine_tackles",
        "low": "prop_engine_tackles_low",
        "high": "prop_engine_tackles_high",
    },
    "kicking": {
        "actual": "actual_prop_total_kicking_points",
        "engine": "prop_engine_kicking_points",
        "low": "prop_engine_kicking_points_low",
        "high": "prop_engine_kicking_points_high",
    },
    "passing": {
        "actual": "actual_prop_total_passing_yards",
        "engine": "prop_engine_passing_yards",
        "low": "prop_engine_passing_yards_low",
        "high": "prop_engine_passing_yards_high",
    },
    "receiving": {
        "actual": "actual_prop_total_receiving_yards",
        "engine": "prop_engine_receiving_yards",
        "low": "prop_engine_receiving_yards_low",
        "high": "prop_engine_receiving_yards_high",
    },
    "rushing": {
        "actual": "actual_prop_total_rushing_yards",
        "engine": "prop_engine_rushing_yards",
        "low": "prop_engine_rushing_yards_low",
        "high": "prop_engine_rushing_yards_high",
    },
}


IDENTITY_COLUMNS = [
    "game_date",
    "game_id",
    "player_name",
    "espn_player_id",
    "prop_engine_player_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Organize weekly NFL prop-selection CSVs."
    )
    parser.add_argument(
        "--season",
        type=int,
        default=2026,
    )
    parser.add_argument(
        "--week",
        type=int,
        required=True,
    )
    return parser.parse_args()


def require_columns(path: Path, fieldnames: list[str], required: list[str]) -> None:
    missing = [column for column in required if column not in fieldnames]
    if missing:
        raise RuntimeError(
            f"{path} is missing required columns: {', '.join(missing)}"
        )


def add_values(left: str, right: str) -> str:
    left_text = str(left or "").strip()
    right_text = str(right or "").strip()

    if not left_text or not right_text:
        return ""

    try:
        total = Decimal(left_text) + Decimal(right_text)
    except InvalidOperation as exc:
        raise ValueError(
            f"Cannot add numeric values {left_text!r} and {right_text!r}"
        ) from exc

    text = format(total, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def simple_input_path(season: int, week: int, category: str) -> Path:
    return (
        OUTPUT_ROOT
        / str(season)
        / f"week_{week}_props"
        / "selections"
        / category
        / f"week_{week}_props_select.csv"
    )


def simple_output_path(season: int, week: int, category: str) -> Path:
    return (
        FINAL_ROOT
        / str(season)
        / "stage_1"
        / f"week_{week}"
        / category
        / f"week_{week}_{category}.csv"
    )


def build_simple_output(
    season: int,
    week: int,
    category: str,
    reporter: PipelineReporter,
) -> dict[str, object]:
    config = SIMPLE_OUTPUTS[category]
    input_path = simple_input_path(season, week, category)
    reporter.add_input(input_path)
    rows, fieldnames = common.read_csv_dict_rows(input_path)

    output_columns = [
        "game_date",
        "game_id",
        "player_name",
        config["actual"],
        config["engine"],
        config["low"],
        config["high"],
        "espn_player_id",
        "prop_engine_player_id",
    ]

    require_columns(input_path, fieldnames, output_columns)

    output_rows = [
        {column: row.get(column, "") for column in output_columns}
        for row in rows
        if str(row.get(config["actual"], "") or "").strip()
    ]

    output_path = simple_output_path(season, week, category)
    common.write_csv_dict_rows_atomic(
        output_path,
        output_columns,
        output_rows,
    )
    reporter.add_output(output_path)

    missing_engine_projection = sum(
        not str(row.get(config["engine"], "") or "").strip()
        for row in output_rows
    )
    missing_player_id = sum(
        not str(row.get("prop_engine_player_id", "") or "").strip()
        for row in output_rows
    )

    return {
        "input_file": input_path.relative_to(REPO_ROOT).as_posix(),
        "output_file": output_path.relative_to(REPO_ROOT).as_posix(),
        "input_rows": int(len(rows)),
        "output_rows": int(len(output_rows)),
        "missing_engine_projection": int(missing_engine_projection),
        "missing_engine_player_id": int(missing_player_id),
    }


def build_combo_outputs(
    season: int,
    week: int,
    reporter: PipelineReporter,
) -> dict[str, object]:
    input_path = simple_input_path(season, week, "combo")
    reporter.add_input(input_path)
    rows, fieldnames = common.read_csv_dict_rows(input_path)

    required = [
        *IDENTITY_COLUMNS,
        "actual_prop_total_passing_plus_rushing_yards",
        "actual_prop_total_rushing_plus_receiving_yards",
        "prop_engine_passing_yards",
        "prop_engine_passing_yards_low",
        "prop_engine_passing_yards_high",
        "prop_engine_rushing_yards",
        "prop_engine_rushing_yards_low",
        "prop_engine_rushing_yards_high",
        "prop_engine_receiving_yards",
        "prop_engine_receiving_yards_low",
        "prop_engine_receiving_yards_high",
    ]
    require_columns(input_path, fieldnames, required)

    pass_rush_columns = [
        "game_date",
        "game_id",
        "player_name",
        "actual_prop_total_passing_plus_rushing_yards",
        "prop_engine_pr",
        "prop_engine_pr_low",
        "prop_engine_pr_high",
        "espn_player_id",
        "prop_engine_player_id",
    ]

    pass_rush_rows: list[dict[str, str]] = []
    for row in rows:
        if not str(
            row.get("actual_prop_total_passing_plus_rushing_yards", "") or ""
        ).strip():
            continue

        pass_rush_rows.append(
            {
                "game_date": row.get("game_date", ""),
                "game_id": row.get("game_id", ""),
                "player_name": row.get("player_name", ""),
                "actual_prop_total_passing_plus_rushing_yards": row.get(
                    "actual_prop_total_passing_plus_rushing_yards", ""
                ),
                "prop_engine_pr": add_values(
                    row.get("prop_engine_passing_yards", ""),
                    row.get("prop_engine_rushing_yards", ""),
                ),
                "prop_engine_pr_low": add_values(
                    row.get("prop_engine_passing_yards_low", ""),
                    row.get("prop_engine_rushing_yards_low", ""),
                ),
                "prop_engine_pr_high": add_values(
                    row.get("prop_engine_passing_yards_high", ""),
                    row.get("prop_engine_rushing_yards_high", ""),
                ),
                "espn_player_id": row.get("espn_player_id", ""),
                "prop_engine_player_id": row.get("prop_engine_player_id", ""),
            }
        )

    pass_rush_path = (
        FINAL_ROOT
        / str(season)
        / "stage_1"
        / f"week_{week}"
        / "combo"
        / "pass_rush_yds"
        / f"week_{week}_pass_rush_yds.csv"
    )
    common.write_csv_dict_rows_atomic(pass_rush_path, pass_rush_columns, pass_rush_rows)
    reporter.add_output(pass_rush_path)

    rec_rush_columns = [
        "game_date",
        "game_id",
        "player_name",
        "actual_prop_total_rushing_plus_receiving_yards",
        "prop_engine_rr",
        "prop_engine_rr_low",
        "prop_engine_rr_high",
        "espn_player_id",
        "prop_engine_player_id",
    ]

    rec_rush_rows: list[dict[str, str]] = []
    for row in rows:
        if not str(
            row.get("actual_prop_total_rushing_plus_receiving_yards", "") or ""
        ).strip():
            continue

        rec_rush_rows.append(
            {
                "game_date": row.get("game_date", ""),
                "game_id": row.get("game_id", ""),
                "player_name": row.get("player_name", ""),
                "actual_prop_total_rushing_plus_receiving_yards": row.get(
                    "actual_prop_total_rushing_plus_receiving_yards", ""
                ),
                "prop_engine_rr": add_values(
                    row.get("prop_engine_receiving_yards", ""),
                    row.get("prop_engine_rushing_yards", ""),
                ),
                "prop_engine_rr_low": add_values(
                    row.get("prop_engine_receiving_yards_low", ""),
                    row.get("prop_engine_rushing_yards_low", ""),
                ),
                "prop_engine_rr_high": add_values(
                    row.get("prop_engine_receiving_yards_high", ""),
                    row.get("prop_engine_rushing_yards_high", ""),
                ),
                "espn_player_id": row.get("espn_player_id", ""),
                "prop_engine_player_id": row.get("prop_engine_player_id", ""),
            }
        )

    rec_rush_path = (
        FINAL_ROOT
        / str(season)
        / "stage_1"
        / f"week_{week}"
        / "combo"
        / "rec_rush_yds"
        / f"week_{week}_rec_rush_yds.csv"
    )
    common.write_csv_dict_rows_atomic(rec_rush_path, rec_rush_columns, rec_rush_rows)
    reporter.add_output(rec_rush_path)

    pass_rush_missing_engine = sum(
        not str(row.get("prop_engine_pr", "") or "").strip()
        for row in pass_rush_rows
    )
    pass_rush_missing_player = sum(
        not str(row.get("prop_engine_player_id", "") or "").strip()
        for row in pass_rush_rows
    )
    rec_rush_missing_engine = sum(
        not str(row.get("prop_engine_rr", "") or "").strip()
        for row in rec_rush_rows
    )
    rec_rush_missing_player = sum(
        not str(row.get("prop_engine_player_id", "") or "").strip()
        for row in rec_rush_rows
    )

    return {
        "input_file": input_path.relative_to(REPO_ROOT).as_posix(),
        "input_rows": int(len(rows)),
        "outputs": {
            "pass_rush_yds": {
                "output_file": pass_rush_path.relative_to(REPO_ROOT).as_posix(),
                "output_rows": int(len(pass_rush_rows)),
                "missing_engine_projection": int(pass_rush_missing_engine),
                "missing_engine_player_id": int(pass_rush_missing_player),
            },
            "rec_rush_yds": {
                "output_file": rec_rush_path.relative_to(REPO_ROOT).as_posix(),
                "output_rows": int(len(rec_rush_rows)),
                "missing_engine_projection": int(rec_rush_missing_engine),
                "missing_engine_player_id": int(rec_rush_missing_player),
            },
        },
    }


def _run(reporter: PipelineReporter) -> None:
    args = parse_args()

    if not 1900 <= int(args.season) <= 2200:
        raise ValueError(f"Invalid season: {args.season}")
    if not 1 <= int(args.week) <= 22:
        raise ValueError(f"Invalid NFL week: {args.week}")

    reporter.update_details(
        {
            "season": int(args.season),
            "week": int(args.week),
        }
    )

    simple_stats: dict[str, dict[str, object]] = {}

    for category in (
        "defense",
        "kicking",
        "passing",
        "receiving",
        "rushing",
    ):
        simple_stats[category] = build_simple_output(
            args.season,
            args.week,
            category,
            reporter,
        )

    combo_stats = build_combo_outputs(
        args.season,
        args.week,
        reporter,
    )

    total_input_rows = sum(
        int(stats["input_rows"])
        for stats in simple_stats.values()
    ) + int(combo_stats["input_rows"])

    combo_outputs = combo_stats["outputs"]
    if not isinstance(combo_outputs, dict):
        raise RuntimeError("Invalid combo output statistics.")

    total_output_rows = sum(
        int(stats["output_rows"])
        for stats in simple_stats.values()
    ) + sum(
        int(stats["output_rows"])
        for stats in combo_outputs.values()
        if isinstance(stats, dict)
    )

    missing_engine_projection = sum(
        int(stats["missing_engine_projection"])
        for stats in simple_stats.values()
    ) + sum(
        int(stats["missing_engine_projection"])
        for stats in combo_outputs.values()
        if isinstance(stats, dict)
    )

    missing_engine_player_id = sum(
        int(stats["missing_engine_player_id"])
        for stats in simple_stats.values()
    ) + sum(
        int(stats["missing_engine_player_id"])
        for stats in combo_outputs.values()
        if isinstance(stats, dict)
    )

    reporter.set_rows(
        rows_in=int(total_input_rows),
        rows_out=int(total_output_rows),
    )
    reporter.update_details(
        {
            "simple_outputs": simple_stats,
            "combo": combo_stats,
            "missing_engine_projection_rows": int(
                missing_engine_projection
            ),
            "missing_engine_player_id_rows": int(
                missing_engine_player_id
            ),
        }
    )

    if missing_engine_projection or missing_engine_player_id:
        reporter.warning(
            "Some stage-1 prop rows are missing Prop Engine projection data.",
            missing_engine_projection_rows=int(
                missing_engine_projection
            ),
            missing_engine_player_id_rows=int(
                missing_engine_player_id
            ),
        )

    print(
        "PROP ORGANIZER: PASS "
        f"season={args.season} week={args.week} "
        f"rows={total_output_rows} "
        f"missing_projection={missing_engine_projection}"
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
