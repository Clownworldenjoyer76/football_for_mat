#!/usr/bin/env python3
"""Build Stage 2 NFL prop picks with over/under probabilities."""

from __future__ import annotations

import csv
import math
import os
import tempfile
from pathlib import Path

import common
from pipeline_reporter import PipelineReporter


REPO_ROOT = common.repo_root().resolve()
PROP_ENGINE_ROOT = common.prop_root().resolve()
FINAL_ROOT = PROP_ENGINE_ROOT / "prop_picks_final"

Z_90 = 1.2815515655446004
PROBABILITY_DECIMALS = 4


FILE_CONFIGS = (
    {
        "relative_path": Path("combo/pass_rush_yds"),
        "filename_suffix": "pass_rush_yds",
        "actual": "actual_prop_total_passing_plus_rushing_yards",
        "engine": "prop_engine_pr",
        "low": "prop_engine_pr_low",
        "high": "prop_engine_pr_high",
        "model": "continuous",
    },
    {
        "relative_path": Path("combo/rec_rush_yds"),
        "filename_suffix": "rec_rush_yds",
        "actual": "actual_prop_total_rushing_plus_receiving_yards",
        "engine": "prop_engine_rr",
        "low": "prop_engine_rr_low",
        "high": "prop_engine_rr_high",
        "model": "continuous",
    },
    {
        "relative_path": Path("defense"),
        "filename_suffix": "defense",
        "actual": "actual_prop_total_tackles",
        "engine": "prop_engine_tackles",
        "low": "prop_engine_tackles_low",
        "high": "prop_engine_tackles_high",
        "model": "tackles",
    },
    {
        "relative_path": Path("kicking"),
        "filename_suffix": "kicking",
        "actual": "actual_prop_total_kicking_points",
        "engine": "prop_engine_kicking_points",
        "low": "prop_engine_kicking_points_low",
        "high": "prop_engine_kicking_points_high",
        "model": "kicking",
    },
    {
        "relative_path": Path("passing"),
        "filename_suffix": "passing",
        "actual": "actual_prop_total_passing_yards",
        "engine": "prop_engine_passing_yards",
        "low": "prop_engine_passing_yards_low",
        "high": "prop_engine_passing_yards_high",
        "model": "continuous",
    },
    {
        "relative_path": Path("receiving"),
        "filename_suffix": "receiving",
        "actual": "actual_prop_total_receiving_yards",
        "engine": "prop_engine_receiving_yards",
        "low": "prop_engine_receiving_yards_low",
        "high": "prop_engine_receiving_yards_high",
        "model": "continuous",
    },
    {
        "relative_path": Path("rushing"),
        "filename_suffix": "rushing",
        "actual": "actual_prop_total_rushing_yards",
        "engine": "prop_engine_rushing_yards",
        "low": "prop_engine_rushing_yards_low",
        "high": "prop_engine_rushing_yards_high",
        "model": "continuous",
    },
)


IDENTITY_COLUMNS = [
    "game_date",
    "game_id",
    "player_name",
    "espn_player_id",
    "prop_engine_player_id",
]


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing input file: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        return rows, list(reader.fieldnames or [])


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

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
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def require_columns(
    path: Path,
    fieldnames: list[str],
    required: list[str],
) -> None:
    missing = [column for column in required if column not in fieldnames]
    if missing:
        raise RuntimeError(
            f"{path} is missing required columns: {', '.join(missing)}"
        )


def parse_number(value: str) -> float | None:
    text = str(value or "").strip()

    if not text:
        return None

    try:
        number = float(text)
    except ValueError:
        return None

    if not math.isfinite(number):
        return None

    return number


def clamp_probability(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def format_probability(value: float | None) -> str:
    if value is None:
        return ""

    return f"{clamp_probability(value):.{PROBABILITY_DECIMALS}f}"


def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def asymmetric_continuous_cdf(
    value: float,
    projection: float,
    low: float,
    high: float,
) -> float | None:
    if low > projection or high < projection:
        return None

    if value < projection:
        spread = projection - low
    else:
        spread = high - projection

    if spread <= 0.0:
        return None

    sigma = spread / Z_90

    if sigma <= 0.0 or not math.isfinite(sigma):
        return None

    z = (value - projection) / sigma
    return clamp_probability(normal_cdf(z))


def continuous_probabilities(
    line: float,
    projection: float,
    low: float,
    high: float,
) -> tuple[float, float] | None:
    under_prob = asymmetric_continuous_cdf(
        line,
        projection,
        low,
        high,
    )

    if under_prob is None:
        return None

    over_prob = 1.0 - under_prob

    return (
        clamp_probability(over_prob),
        clamp_probability(under_prob),
    )


def poisson_cdf(k: int, mean: float) -> float:
    if k < 0:
        return 0.0

    if mean <= 0.0:
        return 1.0

    probability = math.exp(-mean)
    total = probability

    for value in range(1, k + 1):
        probability *= mean / value
        total += probability

    return clamp_probability(total)


def negative_binomial_cdf(
    k: int,
    mean: float,
    variance: float,
) -> float:
    if k < 0:
        return 0.0

    if mean <= 0.0:
        return 1.0

    if variance <= mean:
        return poisson_cdf(k, mean)

    dispersion = (mean * mean) / (variance - mean)

    if (
        not math.isfinite(dispersion)
        or dispersion <= 0.0
        or dispersion > 1_000_000.0
    ):
        return poisson_cdf(k, mean)

    success_probability = dispersion / (dispersion + mean)
    failure_probability = 1.0 - success_probability

    probability = math.exp(
        dispersion * math.log(success_probability)
    )
    total = probability

    for value in range(0, k):
        probability *= (
            (value + dispersion)
            / (value + 1.0)
            * failure_probability
        )
        total += probability

    return clamp_probability(total)


def tackles_probabilities(
    line: float,
    projection: float,
    low: float,
    high: float,
) -> tuple[float, float] | None:
    if projection < 0.0:
        return None

    if high <= low:
        return None

    interval_sd = (high - low) / (2.0 * Z_90)

    if interval_sd <= 0.0 or not math.isfinite(interval_sd):
        return None

    variance = interval_sd * interval_sd

    under_max = math.ceil(line) - 1
    over_min = math.floor(line) + 1

    if variance > projection:
        under_prob = negative_binomial_cdf(
            under_max,
            projection,
            variance,
        )
        over_prob = 1.0 - negative_binomial_cdf(
            over_min - 1,
            projection,
            variance,
        )
    else:
        under_prob = poisson_cdf(
            under_max,
            projection,
        )
        over_prob = 1.0 - poisson_cdf(
            over_min - 1,
            projection,
        )

    return (
        clamp_probability(over_prob),
        clamp_probability(under_prob),
    )


def kicking_integer_cdf(
    max_points: int,
    projection: float,
    low: float,
    high: float,
) -> float | None:
    if max_points < 0:
        return 0.0

    boundary = max_points + 0.5

    return asymmetric_continuous_cdf(
        boundary,
        projection,
        low,
        high,
    )


def kicking_probabilities(
    line: float,
    projection: float,
    low: float,
    high: float,
) -> tuple[float, float] | None:
    if projection < 0.0:
        return None

    if low > projection or high < projection:
        return None

    under_max = math.ceil(line) - 1
    over_min = math.floor(line) + 1

    under_prob = kicking_integer_cdf(
        under_max,
        projection,
        low,
        high,
    )

    over_boundary_prob = kicking_integer_cdf(
        over_min - 1,
        projection,
        low,
        high,
    )

    if under_prob is None or over_boundary_prob is None:
        return None

    over_prob = 1.0 - over_boundary_prob

    return (
        clamp_probability(over_prob),
        clamp_probability(under_prob),
    )


def calculate_probabilities(
    model: str,
    line: float,
    projection: float,
    low: float,
    high: float,
) -> tuple[float, float] | None:
    if model == "continuous":
        return continuous_probabilities(
            line,
            projection,
            low,
            high,
        )

    if model == "tackles":
        return tackles_probabilities(
            line,
            projection,
            low,
            high,
        )

    if model == "kicking":
        return kicking_probabilities(
            line,
            projection,
            low,
            high,
        )

    raise ValueError(f"Unsupported probability model: {model}")


def build_pick_fields(
    model: str,
    line_text: str,
    projection_text: str,
    low_text: str,
    high_text: str,
) -> dict[str, str]:
    line = parse_number(line_text)
    projection = parse_number(projection_text)
    low = parse_number(low_text)
    high = parse_number(high_text)

    if (
        line is None
        or projection is None
        or low is None
        or high is None
    ):
        return {
            "pick": "no_bet",
            "pick_prob": "",
            "over_prob": "",
            "under_prob": "",
        }

    probabilities = calculate_probabilities(
        model,
        line,
        projection,
        low,
        high,
    )

    if probabilities is None:
        return {
            "pick": "no_bet",
            "pick_prob": "",
            "over_prob": "",
            "under_prob": "",
        }

    over_prob, under_prob = probabilities

    over_text = format_probability(over_prob)
    under_text = format_probability(under_prob)

    over_rounded = float(over_text)
    under_rounded = float(under_text)

    if over_rounded > under_rounded:
        pick = "over"
        pick_prob = over_text
    elif under_rounded > over_rounded:
        pick = "under"
        pick_prob = under_text
    else:
        pick = "no_bet"
        pick_prob = over_text

    return {
        "pick": pick,
        "pick_prob": pick_prob,
        "over_prob": over_text,
        "under_prob": under_text,
    }


def process_file(
    season: str,
    week_name: str,
    config: dict[str, object],
    reporter: PipelineReporter,
) -> dict[str, object]:
    stage_1_root = FINAL_ROOT / season / "stage_1" / week_name
    stage_2_root = FINAL_ROOT / season / "stage_2" / week_name

    week_number = week_name.removeprefix("week_")

    relative_path = config["relative_path"]
    filename_suffix = str(config["filename_suffix"])
    actual_column = str(config["actual"])
    engine_column = str(config["engine"])
    low_column = str(config["low"])
    high_column = str(config["high"])
    model = str(config["model"])

    if not isinstance(relative_path, Path):
        raise TypeError("relative_path must be a Path")

    input_path = (
        stage_1_root
        / relative_path
        / f"week_{week_number}_{filename_suffix}.csv"
    )

    if not input_path.is_file():
        reporter.warning(
            "Missing expected Stage 1 prop file.",
            season=season,
            week=week_number,
            input_path=input_path.relative_to(REPO_ROOT).as_posix(),
        )
        return {
            "status": "missing",
            "season": season,
            "week": week_number,
            "filename_suffix": filename_suffix,
            "input_file": input_path.relative_to(REPO_ROOT).as_posix(),
            "input_rows": 0,
            "output_rows": 0,
            "pick_counts": {
                "over": 0,
                "under": 0,
                "no_bet": 0,
            },
            "missing_or_invalid_numeric_inputs": 0,
            "invalid_probability_inputs": 0,
        }

    reporter.add_input(input_path)

    output_path = (
        stage_2_root
        / relative_path
        / f"week_{week_number}_{filename_suffix}.csv"
    )

    rows, fieldnames = read_csv(input_path)

    required_columns = [
        *IDENTITY_COLUMNS,
        actual_column,
        engine_column,
        low_column,
        high_column,
    ]

    require_columns(
        input_path,
        fieldnames,
        required_columns,
    )

    output_columns = [
        "game_date",
        "game_id",
        "player_name",
        actual_column,
        "pick",
        "pick_prob",
        "over_prob",
        "under_prob",
        engine_column,
        low_column,
        high_column,
        "espn_player_id",
        "prop_engine_player_id",
    ]

    output_rows: list[dict[str, str]] = []
    pick_counts = {
        "over": 0,
        "under": 0,
        "no_bet": 0,
    }
    missing_or_invalid_numeric_inputs = 0
    invalid_probability_inputs = 0

    for row in rows:
        line_text = row.get(actual_column, "")
        projection_text = row.get(engine_column, "")
        low_text = row.get(low_column, "")
        high_text = row.get(high_column, "")

        numeric_inputs = (
            parse_number(line_text),
            parse_number(projection_text),
            parse_number(low_text),
            parse_number(high_text),
        )

        has_invalid_numeric_input = any(
            value is None
            for value in numeric_inputs
        )

        if has_invalid_numeric_input:
            missing_or_invalid_numeric_inputs += 1

        pick_fields = build_pick_fields(
            model=model,
            line_text=line_text,
            projection_text=projection_text,
            low_text=low_text,
            high_text=high_text,
        )

        pick = pick_fields["pick"]
        if pick not in pick_counts:
            raise RuntimeError(f"Unexpected pick value: {pick!r}")

        pick_counts[pick] += 1

        if (
            pick == "no_bet"
            and not pick_fields["pick_prob"]
            and not has_invalid_numeric_input
        ):
            invalid_probability_inputs += 1

        output_rows.append(
            {
                "game_date": row.get("game_date", ""),
                "game_id": row.get("game_id", ""),
                "player_name": row.get("player_name", ""),
                actual_column: row.get(actual_column, ""),
                "pick": pick,
                "pick_prob": pick_fields["pick_prob"],
                "over_prob": pick_fields["over_prob"],
                "under_prob": pick_fields["under_prob"],
                engine_column: row.get(engine_column, ""),
                low_column: row.get(low_column, ""),
                high_column: row.get(high_column, ""),
                "espn_player_id": row.get("espn_player_id", ""),
                "prop_engine_player_id": row.get(
                    "prop_engine_player_id",
                    "",
                ),
            }
        )

    write_csv(
        output_path,
        output_columns,
        output_rows,
    )
    reporter.add_output(output_path)

    return {
        "status": "processed",
        "season": season,
        "week": week_number,
        "filename_suffix": filename_suffix,
        "model": model,
        "input_file": input_path.relative_to(REPO_ROOT).as_posix(),
        "output_file": output_path.relative_to(REPO_ROOT).as_posix(),
        "input_rows": int(len(rows)),
        "output_rows": int(len(output_rows)),
        "pick_counts": pick_counts,
        "missing_or_invalid_numeric_inputs": int(
            missing_or_invalid_numeric_inputs
        ),
        "invalid_probability_inputs": int(
            invalid_probability_inputs
        ),
    }


def discover_stage_1_weeks() -> list[tuple[str, str]]:
    discovered: list[tuple[str, str]] = []

    if not FINAL_ROOT.is_dir():
        return discovered

    for season_path in sorted(FINAL_ROOT.iterdir()):
        if not season_path.is_dir():
            continue

        if not season_path.name.isdigit():
            continue

        stage_1_path = season_path / "stage_1"

        if not stage_1_path.is_dir():
            continue

        for week_path in sorted(stage_1_path.iterdir()):
            if not week_path.is_dir():
                continue

            if not week_path.name.startswith("week_"):
                continue

            week_number = week_path.name.removeprefix("week_")

            if not week_number.isdigit():
                continue

            discovered.append(
                (
                    season_path.name,
                    week_path.name,
                )
            )

    return discovered


def _run(reporter: PipelineReporter) -> None:
    stage_1_weeks = discover_stage_1_weeks()

    if not stage_1_weeks:
        raise FileNotFoundError(
            f"No Stage 1 week folders found under {FINAL_ROOT}"
        )

    file_stats: dict[str, dict[str, object]] = {}

    for season, week_name in stage_1_weeks:
        for config in FILE_CONFIGS:
            stats = process_file(
                season,
                week_name,
                config,
                reporter,
            )
            key = (
                f"{season}/{week_name}/"
                f"{stats['filename_suffix']}"
            )
            file_stats[key] = stats

    processed_stats = [
        stats
        for stats in file_stats.values()
        if stats["status"] == "processed"
    ]
    missing_stats = [
        stats
        for stats in file_stats.values()
        if stats["status"] == "missing"
    ]

    total_input_rows = sum(
        int(stats["input_rows"])
        for stats in processed_stats
    )
    total_output_rows = sum(
        int(stats["output_rows"])
        for stats in processed_stats
    )

    pick_counts = {
        "over": 0,
        "under": 0,
        "no_bet": 0,
    }

    for stats in processed_stats:
        counts = stats["pick_counts"]
        if not isinstance(counts, dict):
            raise RuntimeError("Invalid Stage 2 pick-count statistics.")

        for pick in pick_counts:
            pick_counts[pick] += int(counts.get(pick, 0))

    missing_or_invalid_numeric_inputs = sum(
        int(stats["missing_or_invalid_numeric_inputs"])
        for stats in processed_stats
    )
    invalid_probability_inputs = sum(
        int(stats["invalid_probability_inputs"])
        for stats in processed_stats
    )

    reporter.set_rows(
        rows_in=int(total_input_rows),
        rows_out=int(total_output_rows),
    )
    reporter.update_details(
        {
            "stage_1_weeks": [
                {
                    "season": season,
                    "week": week_name.removeprefix("week_"),
                }
                for season, week_name in stage_1_weeks
            ],
            "weeks_processed": int(len(stage_1_weeks)),
            "files_expected": int(
                len(stage_1_weeks) * len(FILE_CONFIGS)
            ),
            "files_processed": int(len(processed_stats)),
            "files_missing": int(len(missing_stats)),
            "pick_counts": pick_counts,
            "missing_or_invalid_numeric_inputs": int(
                missing_or_invalid_numeric_inputs
            ),
            "invalid_probability_inputs": int(
                invalid_probability_inputs
            ),
            "file_stats": file_stats,
        }
    )

    print(
        "PROP ORGANIZER STAGE 2: PASS "
        f"weeks={len(stage_1_weeks)} "
        f"files={len(processed_stats)} "
        f"missing={len(missing_stats)} "
        f"rows={total_output_rows} "
        f"no_bet={pick_counts['no_bet']}"
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
