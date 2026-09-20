#!/usr/bin/env python3
"""
Validate and clean one complete season of hardened ESPN predictor files.

Input:
    docs/win/football/nfl/00_intake/predictions/e_predictions/
        {season}_{season_type}_{week}_e_predictions.csv

Output:
    docs/win/football/nfl/00_intake/predictions/clean/
        {season}_{season_type}_{week}_predictions.csv

The transform remains season-wide. Source files are validated against
pull_e_predictions.py before any clean prediction file is replaced.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import shutil
import sys
import tempfile
import uuid
from collections import defaultdict
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from pathlib import Path
from types import ModuleType
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
NFL_ROOT = SCRIPT_PATH.parents[2]

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_reporter import PipelineReporter

PRODUCER_PATH = SCRIPT_PATH.with_name("pull_e_predictions.py")
IN_DIR = NFL_ROOT / "00_intake" / "predictions" / "e_predictions"
OUT_DIR = NFL_ROOT / "00_intake" / "predictions" / "clean"
LOG_PATH = NFL_ROOT / "errors" / "00_intake" / "clean_e_pred.txt"
REPORT_ROOT = NFL_ROOT / "errors"

OUT_HEADERS = [
    "game_id",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
    "matchupQuality",
    "home_prob",
    "away_prob",
    "tie_prob",
    "away_projected_pts",
    "home_projected_pts",
    "total_projected_pts",
    "home_PtDiff",
    "away_PtDiff",
    "home_rating",
    "away_rating",
    "game_name",
    "season",
    "season_type",
    "week",
    "sport",
    "league",
]

INTENTIONALLY_BLANK_FIELDS = {
    "game_date",
    "game_time",
    "away_projected_pts",
    "home_projected_pts",
    "total_projected_pts",
}

REQUIRED_CLEAN_FIELDS = [
    field
    for field in OUT_HEADERS
    if field not in INTENTIONALLY_BLANK_FIELDS
]


class CleanPredictionError(RuntimeError):
    pass


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def fail(message: str) -> None:
    raise CleanPredictionError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and clean one complete season of ESPN "
            "predictor files."
        )
    )
    parser.add_argument("--season", type=int, required=True)
    args = parser.parse_args()

    if args.season < 2000 or args.season > 2100:
        parser.error("--season must be between 2000 and 2100")

    return args


def load_producer_module() -> ModuleType:
    if not PRODUCER_PATH.is_file():
        fail(
            f"Missing hardened ESPN prediction producer: "
            f"{PRODUCER_PATH}"
        )

    spec = importlib.util.spec_from_file_location(
        "_clean_e_pred_producer_contract",
        PRODUCER_PATH,
    )
    if spec is None or spec.loader is None:
        fail(
            f"Could not load producer contract: "
            f"{PRODUCER_PATH}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv(
    path: Path,
    *,
    label: str,
    exact_columns: list[str],
) -> list[dict[str, str]]:
    if not path.is_file():
        fail(f"Missing {label}: {path}")

    if path.stat().st_size == 0:
        fail(f"Zero-byte {label}: {path}")

    try:
        with path.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            rows = list(reader)
    except Exception as exc:
        fail(
            f"Could not read {label} {path}: "
            f"{type(exc).__name__}: {exc}"
        )

    if fieldnames != exact_columns:
        fail(
            f"{label} has unexpected column order/schema. "
            f"Expected={exact_columns} actual={fieldnames}"
        )

    if not rows:
        fail(f"{label} contains no data rows: {path}")

    return rows


def finite_decimal(raw: Any, *, label: str) -> Decimal:
    text = clean(raw)
    if not text:
        fail(f"{label} is blank")

    try:
        value = Decimal(text)
    except InvalidOperation:
        fail(
            f"{label} must be numeric; received={text!r}"
        )

    if not value.is_finite():
        fail(
            f"{label} must be finite; received={text!r}"
        )

    return value


def to_decimal_prob(
    raw: Any,
    *,
    label: str = "probability percentage",
) -> str:
    """
    Percentage -> decimal string truncated to four decimal places.

    This preserves the legacy Decimal/ROUND_DOWN transform.
    """
    value = finite_decimal(raw, label=label)

    if value < 0 or value > 100:
        fail(
            f"{label} must be between 0 and 100; "
            f"received={value}"
        )

    return str(
        (value / Decimal(100)).quantize(
            Decimal("0.0001"),
            rounding=ROUND_DOWN,
        )
    )


def decimal_equal(
    left: Any,
    right: Any,
    *,
    label: str,
) -> bool:
    return finite_decimal(
        left,
        label=f"{label} left",
    ) == finite_decimal(
        right,
        label=f"{label} right",
    )


def source_group_key(
    rows: list[dict[str, str]],
    *,
    path: Path,
) -> tuple[str, str, str]:
    targets = {
        (
            clean(row.get("season")),
            clean(row.get("season_type")),
            clean(row.get("week")),
        )
        for row in rows
    }

    if len(targets) != 1:
        fail(
            f"{path} contains multiple season/type/week "
            f"targets: {sorted(targets)}"
        )

    key = next(iter(targets))
    if not all(key):
        fail(
            f"{path} contains blank season/type/week target"
        )

    return key


def load_and_validate_sources(
    *,
    season: int,
    producer: ModuleType,
    reporter: PipelineReporter,
) -> tuple[
    list[dict[str, str]],
    dict[tuple[str, str, str], list[dict[str, str]]],
    dict[tuple[str, str, str], Path],
]:
    team_map = producer.load_team_map()
    schedule_path, schedule_rows = producer.load_schedule(
        season=season,
        team_map=team_map,
    )
    producer_paths = producer.expected_output_paths(
        schedule_rows
    )

    reporter.add_input(schedule_path)
    reporter.add_input(producer.TEAM_MAP_PATH)

    if not IN_DIR.is_dir():
        fail(
            f"Missing ESPN prediction input directory: "
            f"{IN_DIR}"
        )

    expected_names = {
        path.name
        for path in producer_paths.values()
    }
    actual_paths = sorted(
        IN_DIR.glob(
            f"{season}_*_e_predictions.csv"
        )
    )
    actual_names = {
        path.name
        for path in actual_paths
    }

    if actual_names != expected_names:
        fail(
            "ESPN prediction source file set mismatch "
            f"missing={sorted(expected_names - actual_names)} "
            f"extra={sorted(actual_names - expected_names)}"
        )

    rows_by_group: dict[
        tuple[str, str, str],
        list[dict[str, str]],
    ] = {}
    all_rows: list[dict[str, str]] = []

    for key, path in sorted(
        producer_paths.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            int(item[0][2]),
        ),
    ):
        reporter.add_input(path)

        rows = read_csv(
            path,
            label="ESPN prediction source",
            exact_columns=list(producer.OUTPUT_HEADER),
        )

        actual_key = source_group_key(
            rows,
            path=path,
        )
        if actual_key != key:
            fail(
                f"{path} target={actual_key} does not "
                f"match schedule/filename target={key}"
            )

        rows_by_group[key] = rows
        all_rows.extend(rows)

    producer.validate_generation(
        all_rows,
        schedule_rows=schedule_rows,
        team_map=team_map,
    )

    return (
        schedule_rows,
        rows_by_group,
        producer_paths,
    )


def transform_game(
    game_id: str,
    sides: dict[str, dict[str, str]],
    *,
    path: Path,
) -> dict[str, str]:
    if set(sides) != {"homeTeam", "awayTeam"}:
        fail(
            f"{path} game_id={game_id} does not contain "
            "exactly homeTeam and awayTeam"
        )

    home = sides["homeTeam"]
    away = sides["awayTeam"]

    home_target = (
        clean(home.get("season")),
        clean(home.get("season_type")),
        clean(home.get("week")),
    )
    away_target = (
        clean(away.get("season")),
        clean(away.get("season_type")),
        clean(away.get("week")),
    )
    if home_target != away_target:
        fail(
            f"{path} game_id={game_id} side target mismatch "
            f"home={home_target} away={away_target}"
        )

    home_name = clean(home.get("game_name"))
    away_name = clean(away.get("game_name"))

    if home_name != away_name:
        fail(
            f"{path} game_id={game_id} game_name differs "
            "between sides"
        )

    if " at " not in home_name:
        fail(
            f"{path} game_id={game_id} cannot split "
            f"game_name={home_name!r} on ' at '"
        )

    away_team, home_team = [
        part.strip()
        for part in home_name.split(" at ", 1)
    ]
    if not away_team or not home_team:
        fail(
            f"{path} game_id={game_id} has invalid "
            f"game_name={home_name!r}"
        )

    if not decimal_equal(
        home.get("matchupQuality"),
        away.get("matchupQuality"),
        label=(
            f"{path} game_id={game_id} "
            "matchupQuality"
        ),
    ):
        fail(
            f"{path} game_id={game_id} matchupQuality "
            "differs between sides"
        )

    home_tie = to_decimal_prob(
        home.get("teamChanceTie"),
        label=(
            f"{path} game_id={game_id} "
            "homeTeam teamChanceTie"
        ),
    )
    away_tie = to_decimal_prob(
        away.get("teamChanceTie"),
        label=(
            f"{path} game_id={game_id} "
            "awayTeam teamChanceTie"
        ),
    )
    if home_tie != away_tie:
        fail(
            f"{path} game_id={game_id} transformed "
            "teamChanceTie differs between sides"
        )

    record = {
        header: ""
        for header in OUT_HEADERS
    }
    record.update(
        {
            "game_id": game_id,
            "home_team": home_team,
            "away_team": away_team,
            "matchupQuality": clean(
                home.get("matchupQuality")
            ),
            "home_prob": to_decimal_prob(
                home.get("gameProjection"),
                label=(
                    f"{path} game_id={game_id} "
                    "homeTeam gameProjection"
                ),
            ),
            "away_prob": to_decimal_prob(
                away.get("gameProjection"),
                label=(
                    f"{path} game_id={game_id} "
                    "awayTeam gameProjection"
                ),
            ),
            "tie_prob": home_tie,
            "home_PtDiff": clean(
                home.get("teamPredPtDiff")
            ),
            "away_PtDiff": clean(
                away.get("teamPredPtDiff")
            ),
            # Preserve legacy opponent-rating assignment.
            "home_rating": clean(
                away.get("oppSeasonStrengthRating")
            ),
            "away_rating": clean(
                home.get("oppSeasonStrengthRating")
            ),
            "game_name": home_name,
            "season": home_target[0],
            "season_type": home_target[1],
            "week": home_target[2],
            "sport": "football",
            "league": "nfl",
        }
    )
    return record


def transform_source_file(
    rows: list[dict[str, str]],
    *,
    path: Path,
) -> list[dict[str, str]]:
    order: list[str] = []
    by_game: dict[
        str,
        dict[str, dict[str, str]],
    ] = defaultdict(dict)

    for line_number, row in enumerate(rows, start=2):
        game_id = clean(row.get("game_id"))
        side = clean(row.get("home_away"))

        if not game_id:
            fail(
                f"{path} line {line_number} has blank game_id"
            )

        if side not in {"homeTeam", "awayTeam"}:
            fail(
                f"{path} line {line_number} has invalid "
                f"home_away={side!r}"
            )

        if side in by_game[game_id]:
            fail(
                f"{path} contains duplicate "
                f"(game_id, home_away)=({game_id}, {side})"
            )

        if game_id not in order:
            order.append(game_id)

        by_game[game_id][side] = row

    output = [
        transform_game(
            game_id,
            by_game[game_id],
            path=path,
        )
        for game_id in order
    ]

    if len(output) * 2 != len(rows):
        fail(
            f"{path} source/output cardinality mismatch "
            f"source_rows={len(rows)} clean_rows={len(output)}"
        )

    return output


def validate_clean_rows(
    rows: list[dict[str, str]],
    *,
    source_rows: list[dict[str, str]],
    key: tuple[str, str, str],
    path: Path,
) -> None:
    source_ids = {
        clean(row.get("game_id"))
        for row in source_rows
    }

    if len(rows) * 2 != len(source_rows):
        fail(
            f"{path} clean row count does not equal one "
            "row per two source side rows"
        )

    seen_ids: set[str] = set()

    for line_number, row in enumerate(rows, start=2):
        for field in REQUIRED_CLEAN_FIELDS:
            if not clean(row.get(field)):
                fail(
                    f"{path} line {line_number} has blank "
                    f"required field={field}"
                )

        for field in INTENTIONALLY_BLANK_FIELDS:
            if clean(row.get(field)):
                fail(
                    f"{path} line {line_number} field={field} "
                    "must remain blank until finalize_pred.py"
                )

        game_id = clean(row.get("game_id"))
        if game_id in seen_ids:
            fail(
                f"{path} contains duplicate game_id={game_id}"
            )
        seen_ids.add(game_id)

        actual_target = (
            clean(row.get("season")),
            clean(row.get("season_type")),
            clean(row.get("week")),
        )
        if actual_target != key:
            fail(
                f"{path} line {line_number} has target="
                f"{actual_target}; expected={key}"
            )

        if clean(row.get("sport")) != "football":
            fail(
                f"{path} line {line_number} has invalid sport"
            )
        if clean(row.get("league")) != "nfl":
            fail(
                f"{path} line {line_number} has invalid league"
            )

        for field in (
            "home_prob",
            "away_prob",
            "tie_prob",
        ):
            probability = finite_decimal(
                row.get(field),
                label=(
                    f"{path} line {line_number} {field}"
                ),
            )
            if probability < 0 or probability > 1:
                fail(
                    f"{path} line {line_number} has "
                    f"{field} outside 0..1"
                )

        for field in (
            "matchupQuality",
            "home_PtDiff",
            "away_PtDiff",
            "home_rating",
            "away_rating",
        ):
            finite_decimal(
                row.get(field),
                label=(
                    f"{path} line {line_number} {field}"
                ),
            )

    if seen_ids != source_ids:
        fail(
            f"{path} clean/source game universe mismatch "
            f"missing={sorted(source_ids - seen_ids)[:10]} "
            f"extra={sorted(seen_ids - source_ids)[:10]}"
        )


def expected_clean_paths(
    producer_paths: dict[
        tuple[str, str, str],
        Path,
    ],
) -> dict[
    tuple[str, str, str],
    Path,
]:
    return {
        key: (
            OUT_DIR
            / (
                f"{key[0]}_{key[1]}_{key[2]}"
                "_predictions.csv"
            )
        )
        for key in producer_paths
    }


def build_clean_generation(
    rows_by_group: dict[
        tuple[str, str, str],
        list[dict[str, str]],
    ],
) -> dict[
    tuple[str, str, str],
    list[dict[str, str]],
]:
    clean_by_group: dict[
        tuple[str, str, str],
        list[dict[str, str]],
    ] = {}

    for key, source_rows in sorted(
        rows_by_group.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            int(item[0][2]),
        ),
    ):
        source_path = (
            IN_DIR
            / (
                f"{key[0]}_{key[1]}_{key[2]}"
                "_e_predictions.csv"
            )
        )
        output_path = (
            OUT_DIR
            / (
                f"{key[0]}_{key[1]}_{key[2]}"
                "_predictions.csv"
            )
        )

        clean_rows = transform_source_file(
            source_rows,
            path=source_path,
        )
        validate_clean_rows(
            clean_rows,
            source_rows=source_rows,
            key=key,
            path=output_path,
        )
        clean_by_group[key] = clean_rows

    return clean_by_group


def normalize_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    return [
        {
            header: clean(row.get(header))
            for header in OUT_HEADERS
        }
        for row in rows
    ]


def write_csv(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUT_HEADERS,
        )
        writer.writeheader()
        writer.writerows(
            normalize_rows(rows)
        )
        handle.flush()
        os.fsync(handle.fileno())


def build_staged_root(
    *,
    season: int,
    clean_by_group: dict[
        tuple[str, str, str],
        list[dict[str, str]],
    ],
    rows_by_group: dict[
        tuple[str, str, str],
        list[dict[str, str]],
    ],
    clean_paths: dict[
        tuple[str, str, str],
        Path,
    ],
) -> Path:
    OUT_DIR.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    stage_root = Path(
        tempfile.mkdtemp(
            prefix=".clean_predictions_stage_",
            dir=OUT_DIR.parent,
        )
    )

    try:
        if OUT_DIR.exists():
            shutil.copytree(
                OUT_DIR,
                stage_root,
                dirs_exist_ok=True,
            )

        for stale in stage_root.glob(
            f"{season}_*_predictions.csv"
        ):
            stale.unlink()

        for key, production_path in sorted(
            clean_paths.items(),
            key=lambda item: (
                item[0][0],
                item[0][1],
                int(item[0][2]),
            ),
        ):
            staged_path = (
                stage_root
                / production_path.name
            )
            clean_rows = clean_by_group[key]

            write_csv(
                staged_path,
                clean_rows,
            )

            staged_rows = read_csv(
                staged_path,
                label="staged clean ESPN prediction output",
                exact_columns=OUT_HEADERS,
            )
            validate_clean_rows(
                staged_rows,
                source_rows=rows_by_group[key],
                key=key,
                path=staged_path,
            )

            if staged_rows != normalize_rows(clean_rows):
                fail(
                    "Staged clean prediction file differs "
                    "from validated in-memory transform: "
                    f"{staged_path}"
                )

        expected_names = {
            path.name
            for path in clean_paths.values()
        }
        actual_names = {
            path.name
            for path in stage_root.glob(
                f"{season}_*_predictions.csv"
            )
        }

        if actual_names != expected_names:
            fail(
                "Staged clean prediction file set mismatch "
                f"expected={sorted(expected_names)} "
                f"actual={sorted(actual_names)}"
            )

        return stage_root

    except Exception:
        shutil.rmtree(
            stage_root,
            ignore_errors=True,
        )
        raise


def publish_staged_root(
    stage_root: Path,
    *,
    reporter: PipelineReporter,
) -> None:
    backup_root = (
        OUT_DIR.parent
        / (
            f".{OUT_DIR.name}_backup_"
            f"{uuid.uuid4().hex}"
        )
    )

    try:
        if OUT_DIR.exists():
            os.replace(
                OUT_DIR,
                backup_root,
            )

        os.replace(
            stage_root,
            OUT_DIR,
        )

    except Exception:
        if OUT_DIR.exists():
            shutil.rmtree(
                OUT_DIR,
                ignore_errors=True,
            )

        if backup_root.exists():
            os.replace(
                backup_root,
                OUT_DIR,
            )
        raise

    if backup_root.exists():
        try:
            shutil.rmtree(backup_root)
        except Exception as exc:
            reporter.warning(
                "Clean predictions were published but the "
                "temporary backup directory could not be removed",
                backup_path=str(backup_root),
                error_type=type(exc).__name__,
                error=str(exc),
            )


def write_legacy_log(
    log_lines: list[str],
) -> None:
    LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with LOG_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "\n".join(log_lines) + "\n"
        )


def safe_write_legacy_log(
    log_lines: list[str],
    *,
    reporter: PipelineReporter,
) -> None:
    try:
        write_legacy_log(log_lines)
        reporter.add_output(LOG_PATH)
        reporter.set_detail(
            "legacy_log_written",
            True,
        )
    except Exception as exc:
        reporter.warning(
            "Legacy clean_e_pred text log could not be written",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        reporter.set_detail(
            "legacy_log_written",
            False,
        )


def run(
    reporter: PipelineReporter,
    *,
    season: int,
    log_lines: list[str],
) -> None:
    producer = load_producer_module()

    (
        schedule_rows,
        rows_by_group,
        producer_paths,
    ) = load_and_validate_sources(
        season=season,
        producer=producer,
        reporter=reporter,
    )

    clean_paths = expected_clean_paths(
        producer_paths
    )
    clean_by_group = build_clean_generation(
        rows_by_group
    )

    source_rows_total = sum(
        len(rows)
        for rows in rows_by_group.values()
    )
    clean_rows_total = sum(
        len(rows)
        for rows in clean_by_group.values()
    )

    reporter.set_rows(
        rows_in=source_rows_total,
        rows_out=0,
    )
    reporter.update_details(
        {
            "season": season,
            "refresh_scope": "full season",
            "scheduled_games": len(schedule_rows),
            "source_files": len(rows_by_group),
            "source_rows": source_rows_total,
            "clean_files": len(clean_by_group),
            "expected_clean_rows": clean_rows_total,
            "output_columns": len(OUT_HEADERS),
            "probability_conversion": (
                "percentage_to_decimal_truncate_4dp"
            ),
            "publication_mode": (
                "validated_directory_swap_with_rollback"
            ),
            "publication_completed": False,
            "staged_roundtrip_verified": False,
        }
    )

    stage_root: Path | None = None

    try:
        stage_root = build_staged_root(
            season=season,
            clean_by_group=clean_by_group,
            rows_by_group=rows_by_group,
            clean_paths=clean_paths,
        )

        reporter.set_detail(
            "staged_roundtrip_verified",
            True,
        )

        publish_staged_root(
            stage_root,
            reporter=reporter,
        )
        stage_root = None

    finally:
        if (
            stage_root is not None
            and stage_root.exists()
        ):
            shutil.rmtree(
                stage_root,
                ignore_errors=True,
            )

    for key, path in sorted(
        clean_paths.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            int(item[0][2]),
        ),
    ):
        reporter.add_output(path)
        log_lines.append(
            f"wrote {len(clean_by_group[key])} rows to {path}"
        )

    reporter.set_rows(
        rows_in=source_rows_total,
        rows_out=clean_rows_total,
    )
    reporter.update_details(
        {
            "rows_published": clean_rows_total,
            "files_published": len(clean_paths),
            "publication_completed": True,
        }
    )

    summary = (
        f"SUMMARY: files_read={len(rows_by_group)} "
        f"rows_read={source_rows_total} "
        f"games={clean_rows_total} "
        f"files_written={len(clean_paths)} "
        f"games_written={clean_rows_total}"
    )
    print(summary)
    log_lines.append(summary)


def main() -> int:
    args = parse_args()
    log_lines: list[str] = []

    try:
        with PipelineReporter(
            script=SCRIPT_PATH,
            stage="00_intake",
            report_root=REPORT_ROOT,
            pipeline="NFL",
            league="NFL",
            season=args.season,
            extra_context={
                "component": "ESPN prediction cleaner",
                "refresh_scope": "full season",
            },
        ) as reporter:
            try:
                run(
                    reporter,
                    season=args.season,
                    log_lines=log_lines,
                )
            except Exception as exc:
                log_lines.append(
                    f"FATAL: {type(exc).__name__}: {exc}"
                )
                safe_write_legacy_log(
                    log_lines,
                    reporter=reporter,
                )
                raise

            safe_write_legacy_log(
                log_lines,
                reporter=reporter,
            )

        return 0

    except Exception as exc:
        print(
            f"ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
