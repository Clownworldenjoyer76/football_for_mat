#!/usr/bin/env python3
"""Grade Stage 3 NFL props from completed ESPN game box scores and build reports."""

from __future__ import annotations

import csv
import json
import math
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import common
from pipeline_reporter import PipelineReporter


REPO_ROOT = common.repo_root().resolve()
PROP_ENGINE_ROOT = common.prop_root().resolve()
PROP_PICKS_ROOT = PROP_ENGINE_ROOT / "prop_picks_final"
FINAL_ROOT = PROP_ENGINE_ROOT / "05_final"
GRADED_ROOT = FINAL_ROOT / "graded"
REPORTS_ROOT = FINAL_ROOT / "reports"
LOCKED_ROOT = PROP_PICKS_ROOT / "locked"

ESPN_SUMMARY_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={game_id}"
)
USER_AGENT = "football_for_mat-prop-grader/1.0"
REQUEST_TIMEOUT_SECONDS = 30
HTTP_RETRIES = 4
TRANSIENT_HTTP_CODES = {
    408,
    425,
    429,
    500,
    502,
    503,
    504,
}


PROP_DEFS = [
    {
        "prop_type": "passing_rushing_yards",
        "line_column": "actual_prop_total_passing_plus_rushing_yards",
        "stat_kind": "passing_rushing_yards",
    },
    {
        "prop_type": "rushing_receiving_yards",
        "line_column": "actual_prop_total_rushing_plus_receiving_yards",
        "stat_kind": "rushing_receiving_yards",
    },
    {
        "prop_type": "tackles",
        "line_column": "actual_prop_total_tackles",
        "stat_kind": "tackles",
    },
    {
        "prop_type": "kicking_points",
        "line_column": "actual_prop_total_kicking_points",
        "stat_kind": "kicking_points",
    },
    {
        "prop_type": "passing_yards",
        "line_column": "actual_prop_total_passing_yards",
        "stat_kind": "passing_yards",
    },
    {
        "prop_type": "receiving_yards",
        "line_column": "actual_prop_total_receiving_yards",
        "stat_kind": "receiving_yards",
    },
    {
        "prop_type": "rushing_yards",
        "line_column": "actual_prop_total_rushing_yards",
        "stat_kind": "rushing_yards",
    },
]

REPORT_METRIC_COLUMNS = [
    "bets",
    "graded_bets",
    "wins",
    "losses",
    "pushes",
    "ungraded",
    "win_rate",
]


PROBABILITY_BUCKETS = [
    (0.50, 0.55),
    (0.55, 0.60),
    (0.60, 0.65),
    (0.65, 0.70),
    (0.70, 0.75),
    (0.75, 0.80),
    (0.80, 0.85),
    (0.85, 0.90),
    (0.90, 0.95),
    (0.95, 1.0000000001),
]


STAGE3_RE = re.compile(r"^(?P<season>\d{4})_(?P<week>\d+)_all_props\.csv$")
LOCKED_RE = re.compile(
    r"^(?P<season>\d{4})_(?P<week>\d+)_all_props_"
    r"(?P<timestamp>\d{8}_\d{6})\.csv$"
)


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.casefold() in {"", "nan", "none", "null", "<na>"}:
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


def format_number(value: float | None) -> str:
    if value is None:
        return ""
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.10f}".rstrip("0").rstrip(".")


def normalize_stat_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", clean(value).casefold())


def first_number(value: Any) -> float | None:
    text = clean(value)
    if not text:
        return None
    if "/" in text:
        text = text.split("/", 1)[0].strip()
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def discover_stage3() -> dict[str, list[tuple[int, Path]]]:
    seasons: dict[str, list[tuple[int, Path]]] = defaultdict(list)

    if not PROP_PICKS_ROOT.is_dir():
        return {}

    for season_dir in sorted(PROP_PICKS_ROOT.iterdir()):
        if not season_dir.is_dir() or not season_dir.name.isdigit():
            continue

        stage3_dir = season_dir / "stage_3"
        if not stage3_dir.is_dir():
            continue

        for path in sorted(stage3_dir.glob("*_all_props.csv")):
            match = STAGE3_RE.match(path.name)
            if not match:
                continue
            if match.group("season") != season_dir.name:
                continue
            seasons[season_dir.name].append((int(match.group("week")), path))

    return dict(seasons)



def discover_latest_locked() -> dict[tuple[str, int], Path]:
    latest: dict[tuple[str, int], Path] = {}

    if not LOCKED_ROOT.is_dir():
        return latest

    for path in sorted(LOCKED_ROOT.glob("*_all_props_*.csv")):
        match = LOCKED_RE.match(path.name)
        if not match:
            continue

        key = (
            match.group("season"),
            int(match.group("week")),
        )
        current = latest.get(key)

        if current is None or path.name > current.name:
            latest[key] = path

    return latest


def discover_grading_inputs(
    reporter: PipelineReporter,
) -> dict[str, list[tuple[int, Path]]]:
    stage3 = discover_stage3()
    stage3_map: dict[tuple[str, int], Path] = {
        (season, week): path
        for season, files in stage3.items()
        for week, path in files
    }
    locked = discover_latest_locked()

    keys = sorted(
        set(stage3_map) | set(locked),
        key=lambda item: (item[0], item[1]),
    )

    seasons: dict[str, list[tuple[int, Path]]] = defaultdict(list)

    for season, week in keys:
        locked_path = locked.get((season, week))

        if locked_path is not None:
            seasons[season].append(
                (
                    week,
                    locked_path,
                )
            )
            continue

        stage3_path = stage3_map.get((season, week))
        if stage3_path is None:
            continue

        reporter.warning(
            "No locked prop snapshot found; using Stage 3 fallback.",
            season=season,
            week=int(week),
            input_path=stage3_path.relative_to(REPO_ROOT).as_posix(),
        )
        seasons[season].append(
            (
                week,
                stage3_path,
            )
        )

    return dict(seasons)


def read_stage3_files(files: list[tuple[int, Path]]) -> tuple[list[str], list[dict[str, str]]]:
    fieldnames: list[str] = []
    rows: list[dict[str, str]] = []

    for week, path in sorted(files):
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for field in reader.fieldnames or []:
                if field not in fieldnames:
                    fieldnames.append(field)

            for row in reader:
                if not any(clean(value) for value in row.values()):
                    continue
                copied = {key: clean(value) for key, value in row.items() if key is not None}
                copied["week"] = str(week)
                rows.append(copied)

    return fieldnames, rows


def detect_prop(row: dict[str, str]) -> tuple[dict[str, str] | None, float | None, str]:
    found: list[tuple[dict[str, str], float]] = []

    for definition in PROP_DEFS:
        line = parse_float(row.get(definition["line_column"]))
        if line is not None:
            found.append((definition, line))

    if not found:
        return None, None, "no_prop_line"
    if len(found) > 1:
        return None, None, "ambiguous_prop_line"

    definition, line = found[0]
    return definition, line, ""


def fetch_espn_summary(game_id: str) -> dict[str, Any]:
    url = ESPN_SUMMARY_URL.format(game_id=game_id)
    last_error: Exception | None = None

    for attempt in range(1, HTTP_RETRIES + 1):
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=REQUEST_TIMEOUT_SECONDS,
            ) as response:
                payload = json.load(response)

            if not isinstance(payload, dict):
                raise ValueError(
                    "ESPN summary response is not a JSON object"
                )

            return payload

        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in TRANSIENT_HTTP_CODES:
                raise

        except (
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            last_error = exc

        if attempt < HTTP_RETRIES:
            time.sleep(
                min(
                    2 ** (attempt - 1),
                    8,
                )
            )

    raise RuntimeError(
        f"ESPN summary request failed after {HTTP_RETRIES} attempts: "
        f"game_id={game_id}: {last_error}"
    )


def game_is_final(payload: dict[str, Any]) -> bool:
    competitions = payload.get("header", {}).get("competitions", [])
    if not isinstance(competitions, list):
        return False

    for competition in competitions:
        if not isinstance(competition, dict):
            continue
        status_type = competition.get("status", {}).get("type", {})
        if not isinstance(status_type, dict):
            continue
        # noinspection PySimplifyBooleanCheck
        if status_type.get("completed") is True:
            return True
        state = clean(status_type.get("state")).casefold()
        name = clean(status_type.get("name")).casefold()
        description = clean(status_type.get("description")).casefold()
        if state == "post" or name in {"status_final", "final"} or description == "final":
            return True

    return False


def player_stat_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    team_blocks = payload.get("boxscore", {}).get("players", [])
    if not isinstance(team_blocks, list):
        return output

    for team_block in team_blocks:
        if not isinstance(team_block, dict):
            continue
        categories = team_block.get("statistics", [])
        if not isinstance(categories, list):
            continue

        for category in categories:
            if not isinstance(category, dict):
                continue

            category_name = normalize_stat_key(category.get("name"))
            keys = category.get("keys", [])
            if not isinstance(keys, list):
                keys = []
            labels = category.get("labels", [])
            if not isinstance(labels, list):
                labels = []
            athletes = category.get("athletes", [])
            if not isinstance(athletes, list):
                continue

            for athlete_row in athletes:
                if not isinstance(athlete_row, dict):
                    continue
                athlete = athlete_row.get("athlete", {})
                if not isinstance(athlete, dict):
                    continue
                player_id = clean(athlete.get("id"))
                if not player_id:
                    continue

                player = output.setdefault(
                    player_id,
                    {
                        "seen": True,
                        "categories": set(),
                        "stats": {},
                    },
                )
                player["seen"] = True
                if category_name:
                    player["categories"].add(category_name)

                stats = athlete_row.get("stats", [])
                if not isinstance(stats, list):
                    continue

                for index, raw_value in enumerate(stats):
                    candidates: list[str] = []
                    if index < len(keys):
                        candidates.append(normalize_stat_key(keys[index]))
                    if index < len(labels):
                        candidates.append(normalize_stat_key(labels[index]))

                    for candidate in candidates:
                        if not candidate:
                            continue
                        player["stats"][candidate] = raw_value
                        if category_name:
                            player["stats"][category_name + candidate] = raw_value

    return output


def lookup_numeric(player: dict[str, Any], aliases: Iterable[str]) -> float | None:
    stats = player.get("stats", {})
    if not isinstance(stats, dict):
        return None
    for alias in aliases:
        raw = stats.get(normalize_stat_key(alias))
        value = first_number(raw)
        if value is not None:
            return value
    return None


def final_stat_for_prop(player: dict[str, Any], stat_kind: str) -> float | None:
    passing = lookup_numeric(
        player,
        [
            "passingYards",
            "passing passingYards",
        ],
    )
    rushing = lookup_numeric(
        player,
        [
            "rushingYards",
            "rushing rushingYards",
        ],
    )
    receiving = lookup_numeric(
        player,
        [
            "receivingYards",
            "receiving receivingYards",
        ],
    )

    if stat_kind == "passing_yards":
        return 0.0 if passing is None and player.get("seen") else passing

    if stat_kind == "rushing_yards":
        return 0.0 if rushing is None and player.get("seen") else rushing

    if stat_kind == "receiving_yards":
        return 0.0 if receiving is None and player.get("seen") else receiving

    if stat_kind == "passing_rushing_yards":
        if not player.get("seen"):
            return None
        return (passing or 0.0) + (rushing or 0.0)

    if stat_kind == "rushing_receiving_yards":
        if not player.get("seen"):
            return None
        return (rushing or 0.0) + (receiving or 0.0)

    if stat_kind == "tackles":
        total = lookup_numeric(
            player,
            [
                "totalTackles",
                "defensive totalTackles",
                "tackles",
                "defensive tackles",
                "TOT",
                "defensive TOT",
            ],
        )
        if total is not None:
            return total

        solo = lookup_numeric(
            player,
            [
                "soloTackles",
                "defensive soloTackles",
                "solo",
                "defensive solo",
            ],
        )
        assists = lookup_numeric(
            player,
            [
                "assistedTackles",
                "tackleAssists",
                "defensive assistedTackles",
                "defensive tackleAssists",
                "AST",
                "defensive AST",
            ],
        )
        if solo is not None or assists is not None:
            return (solo or 0.0) + (assists or 0.0)
        return 0.0 if player.get("seen") else None

    if stat_kind == "kicking_points":
        total_points = lookup_numeric(
            player,
            [
                "totalKickingPoints",
                "kicking totalKickingPoints",
                "kickingPoints",
                "kicking kickingPoints",
                "PTS",
                "kicking PTS",
            ],
        )
        if total_points is not None:
            return total_points

        field_goals_made = lookup_numeric(
            player,
            [
                "fieldGoalsMade",
                "fieldGoalsMade/fieldGoalAttempts",
                "kicking fieldGoalsMade",
                "kicking fieldGoalsMade/fieldGoalAttempts",
                "FG",
                "kicking FG",
            ],
        )
        extra_points_made = lookup_numeric(
            player,
            [
                "extraPointsMade",
                "extraPointsMade/extraPointAttempts",
                "kicking extraPointsMade",
                "kicking extraPointsMade/extraPointAttempts",
                "XP",
                "kicking XP",
            ],
        )
        if field_goals_made is not None or extra_points_made is not None:
            return 3.0 * (field_goals_made or 0.0) + (extra_points_made or 0.0)
        return None

    return None


def current_row_key(row: dict[str, str], prop_type: str, line: float | None) -> tuple[str, ...]:
    return (
        clean(row.get("game_id")),
        clean(row.get("espn_player_id")),
        clean(row.get("prop_engine_player_id")),
        prop_type,
        format_number(line),
        clean(row.get("pick")).casefold(),
    )


def load_existing_grades(path: Path) -> dict[tuple[str, ...], dict[str, str]]:
    if not path.is_file():
        return {}

    existing: dict[tuple[str, ...], dict[str, str]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            prop_type = clean(row.get("prop_type"))
            definition = next(
                (item for item in PROP_DEFS if item["prop_type"] == prop_type),
                None,
            )
            if definition is None:
                continue
            line = parse_float(row.get(definition["line_column"]))
            key = current_row_key(row, prop_type, line)
            existing[key] = {key_name: clean(value) for key_name, value in row.items()}

    return existing


def build_game_requests(
    rows: list[dict[str, str]],
    existing: dict[tuple[str, ...], dict[str, str]],
) -> set[str]:
    needed: set[str] = set()

    for row in rows:
        definition, line, _ = detect_prop(row)
        if definition is None:
            continue

        pick = clean(row.get("pick")).casefold()
        if pick == "no_bet":
            continue

        key = current_row_key(row, definition["prop_type"], line)
        previous = existing.get(key)
        if previous and clean(previous.get("grade")) in {"win", "loss", "push"}:
            continue

        game_id = clean(row.get("game_id"))
        if game_id:
            needed.add(game_id)

    return needed


def grade_row(
    row: dict[str, str],
    summaries: dict[str, dict[str, Any]],
    player_maps: dict[str, dict[str, dict[str, Any]]],
    summary_errors: dict[str, str],
    existing: dict[tuple[str, ...], dict[str, str]],
) -> dict[str, str]:
    output = dict(row)
    definition, line, detect_reason = detect_prop(row)

    if definition is None:
        output["prop_type"] = ""
        output["final_stat"] = ""
        output["grade"] = "ungraded"
        output["grade_reason"] = detect_reason
        return output

    prop_type = definition["prop_type"]
    output["prop_type"] = prop_type

    pick = clean(row.get("pick")).casefold()
    if pick == "no_bet":
        output["final_stat"] = ""
        output["grade"] = "no_bet"
        output["grade_reason"] = ""
        return output

    if pick not in {"over", "under"}:
        output["final_stat"] = ""
        output["grade"] = "ungraded"
        output["grade_reason"] = "invalid_pick"
        return output

    key = current_row_key(row, prop_type, line)
    previous = existing.get(key)
    if previous and clean(previous.get("grade")) in {"win", "loss", "push"}:
        output["final_stat"] = clean(previous.get("final_stat"))
        output["grade"] = clean(previous.get("grade"))
        output["grade_reason"] = ""
        return output

    game_id = clean(row.get("game_id"))
    if not game_id:
        output["final_stat"] = ""
        output["grade"] = "ungraded"
        output["grade_reason"] = "missing_game_id"
        return output

    if game_id in summary_errors:
        output["final_stat"] = ""
        output["grade"] = "ungraded"
        output["grade_reason"] = "espn_fetch_error"
        return output

    payload = summaries.get(game_id)
    if payload is None:
        output["final_stat"] = ""
        output["grade"] = "ungraded"
        output["grade_reason"] = "missing_espn_summary"
        return output

    if not game_is_final(payload):
        output["final_stat"] = ""
        output["grade"] = "ungraded"
        output["grade_reason"] = "game_not_final"
        return output

    espn_player_id = clean(row.get("espn_player_id"))
    if not espn_player_id:
        output["final_stat"] = ""
        output["grade"] = "ungraded"
        output["grade_reason"] = "missing_espn_player_id"
        return output

    players = player_maps.get(game_id, {})
    player = players.get(espn_player_id)
    if player is None:
        output["final_stat"] = ""
        output["grade"] = "ungraded"
        output["grade_reason"] = "player_not_in_boxscore"
        return output

    final_stat = final_stat_for_prop(player, definition["stat_kind"])
    if final_stat is None:
        output["final_stat"] = ""
        output["grade"] = "ungraded"
        output["grade_reason"] = "stat_unavailable"
        return output

    output["final_stat"] = format_number(final_stat)
    output["grade_reason"] = ""

    if final_stat == line:
        output["grade"] = "push"
    elif pick == "over":
        output["grade"] = "win" if final_stat > line else "loss"
    else:
        output["grade"] = "win" if final_stat < line else "loss"

    return output


def betting_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if clean(row.get("pick")).casefold() in {"over", "under"}]


def metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    bets = betting_rows(rows)
    wins = sum(clean(row.get("grade")) == "win" for row in bets)
    losses = sum(clean(row.get("grade")) == "loss" for row in bets)
    pushes = sum(clean(row.get("grade")) == "push" for row in bets)
    ungraded = sum(clean(row.get("grade")) == "ungraded" for row in bets)
    graded_bets = wins + losses + pushes
    decisive = wins + losses
    win_rate = wins / decisive if decisive else None

    return {
        "bets": len(bets),
        "graded_bets": graded_bets,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "ungraded": ungraded,
        "win_rate": "" if win_rate is None else f"{win_rate:.4f}",
    }


def grouped_report(
    rows: list[dict[str, str]],
    key_name: str,
    key_func,
) -> tuple[list[str], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = clean(key_func(row))
        if key:
            groups[key].append(row)

    report_rows: list[dict[str, Any]] = []
    for key in sorted(groups, key=natural_sort_key):
        report_rows.append({key_name: key, **metrics(groups[key])})

    return [key_name, *REPORT_METRIC_COLUMNS], report_rows


def natural_sort_key(value: str) -> tuple[Any, ...]:
    parts = re.split(r"(\d+(?:\.\d+)?)", str(value))
    key: list[Any] = []
    for part in parts:
        if not part:
            continue
        try:
            key.append((0, float(part)))
        except ValueError:
            key.append((1, part.casefold()))
    return tuple(key)


def probability_bucket(value: Any) -> str:
    probability = parse_float(value)
    if probability is None:
        return ""

    if probability < 0.50:
        return "<0.50"
    if probability > 1.0:
        return ">1.00"

    for lower, upper in PROBABILITY_BUCKETS:
        if lower <= probability < upper:
            display_upper = min(upper, 1.0)
            return f"{lower:.2f}-{display_upper:.2f}"

    return ""


def build_reports(
    season: str,
    rows: list[dict[str, str]],
    reporter: PipelineReporter,
) -> list[Path]:
    report_dir = REPORTS_ROOT / season
    bets = betting_rows(rows)

    outputs: list[Path] = []

    overall = {"season": season, **metrics(rows)}
    overall_path = report_dir / "overall.csv"
    common.write_filtered_csv_dict_rows_atomic(
        overall_path,
        ["season", *REPORT_METRIC_COLUMNS],
        [overall],
    )
    outputs.append(overall_path)

    fields, report = grouped_report(bets, "prop_type", lambda report_row: report_row.get("prop_type"))
    by_prop_type_path = report_dir / "by_prop_type.csv"
    common.write_filtered_csv_dict_rows_atomic(by_prop_type_path, fields, report)
    outputs.append(by_prop_type_path)

    fields, report = grouped_report(bets, "probability_bucket", lambda report_row: probability_bucket(report_row.get("pick_prob")))
    by_probability_path = report_dir / "by_probability.csv"
    common.write_filtered_csv_dict_rows_atomic(by_probability_path, fields, report)
    outputs.append(by_probability_path)

    fields, report = grouped_report(bets, "pick", lambda report_row: clean(report_row.get("pick")).casefold())
    by_pick_path = report_dir / "by_pick_direction.csv"
    common.write_filtered_csv_dict_rows_atomic(by_pick_path, fields, report)
    outputs.append(by_pick_path)

    fields, report = grouped_report(bets, "week", lambda report_row: report_row.get("week"))
    by_week_path = report_dir / "by_week.csv"
    common.write_filtered_csv_dict_rows_atomic(by_week_path, fields, report)
    outputs.append(by_week_path)

    calibration_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in bets:
        if clean(row.get("grade")) not in {"win", "loss"}:
            continue
        bucket = probability_bucket(row.get("pick_prob"))
        if bucket:
            calibration_groups[bucket].append(row)

    calibration_rows: list[dict[str, Any]] = []
    for bucket in sorted(calibration_groups, key=natural_sort_key):
        group = calibration_groups[bucket]
        probabilities = [parse_float(row.get("pick_prob")) for row in group]
        probabilities = [value for value in probabilities if value is not None]
        wins = sum(clean(row.get("grade")) == "win" for row in group)
        graded = len(group)
        avg_prob = sum(probabilities) / len(probabilities) if probabilities else None
        observed = wins / graded if graded else None
        error = observed - avg_prob if observed is not None and avg_prob is not None else None

        calibration_rows.append(
            {
                "probability_bucket": bucket,
                "graded_bets": graded,
                "avg_pick_prob": "" if avg_prob is None else f"{avg_prob:.4f}",
                "observed_win_rate": "" if observed is None else f"{observed:.4f}",
                "calibration_error": "" if error is None else f"{error:.4f}",
                "absolute_calibration_error": "" if error is None else f"{abs(error):.4f}",
            }
        )

    calibration_path = report_dir / "calibration.csv"
    common.write_filtered_csv_dict_rows_atomic(
        calibration_path,
        [
            "probability_bucket",
            "graded_bets",
            "avg_pick_prob",
            "observed_win_rate",
            "calibration_error",
            "absolute_calibration_error",
        ],
        calibration_rows,
    )

    outputs.append(calibration_path)

    for path in outputs:
        reporter.add_output(path)

    return outputs


def process_season(
    season: str,
    files: list[tuple[int, Path]],
    reporter: PipelineReporter,
) -> dict[str, Any]:
    for _, path in files:
        reporter.add_input(path)

    original_fields, rows = read_stage3_files(files)
    if not rows:
        return {
            "season": season,
            "rows": 0,
            "games_requested": 0,
            "fetch_failures": 0,
            "grade_counts": {},
            "ungraded_reasons": {},
            "source_files": [
                path.relative_to(REPO_ROOT).as_posix()
                for _, path in files
            ],
        }

    graded_path = GRADED_ROOT / season / f"{season}_all_props_graded.csv"

    if graded_path.is_file():
        reporter.add_input(graded_path)

    existing = load_existing_grades(graded_path)
    game_ids = build_game_requests(rows, existing)

    summaries: dict[str, dict[str, Any]] = {}
    summary_errors: dict[str, str] = {}

    for game_id in sorted(game_ids):
        try:
            summaries[game_id] = fetch_espn_summary(game_id)
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            ValueError,
            json.JSONDecodeError,
            RuntimeError,
        ) as exc:
            summary_errors[game_id] = f"{type(exc).__name__}: {exc}"

    player_maps = {
        game_id: player_stat_map(payload)
        for game_id, payload in summaries.items()
    }

    graded_rows = [
        grade_row(
            row,
            summaries,
            player_maps,
            summary_errors,
            existing,
        )
        for row in rows
    ]

    fieldnames = list(original_fields)
    for field in [
        "week",
        "prop_type",
        "final_stat",
        "grade",
        "grade_reason",
    ]:
        if field not in fieldnames:
            fieldnames.append(field)

    common.write_filtered_csv_dict_rows_atomic(graded_path, fieldnames, graded_rows)
    reporter.add_output(graded_path)
    report_outputs = build_reports(
        season,
        graded_rows,
        reporter,
    )

    grade_counts: dict[str, int] = defaultdict(int)
    ungraded_reasons: dict[str, int] = defaultdict(int)

    for row in graded_rows:
        grade = clean(row.get("grade")) or "blank"
        grade_counts[grade] += 1

        if grade == "ungraded":
            reason = clean(row.get("grade_reason")) or "unspecified"
            ungraded_reasons[reason] += 1

    if summary_errors:
        reporter.warning(
            "ESPN summary fetch failures left some props ungraded.",
            season=season,
            failures=summary_errors,
        )

    return {
        "season": season,
        "rows": int(len(graded_rows)),
        "games_requested": int(len(game_ids)),
        "summaries_fetched": int(len(summaries)),
        "fetch_failures": int(len(summary_errors)),
        "grade_counts": dict(sorted(grade_counts.items())),
        "ungraded_reasons": dict(sorted(ungraded_reasons.items())),
        "source_files": [
            path.relative_to(REPO_ROOT).as_posix()
            for _, path in files
        ],
        "graded_output": graded_path.relative_to(REPO_ROOT).as_posix(),
        "report_outputs": [
            path.relative_to(REPO_ROOT).as_posix()
            for path in report_outputs
        ],
    }


def _run(reporter: PipelineReporter) -> None:
    discovered = discover_grading_inputs(reporter)

    if not discovered:
        raise FileNotFoundError(
            f"No locked or Stage 3 prop files found under "
            f"{PROP_PICKS_ROOT}"
        )

    season_stats: dict[str, dict[str, Any]] = {}

    for season in sorted(discovered):
        season_stats[season] = process_season(
            season,
            discovered[season],
            reporter,
        )

    total_rows = sum(
        int(stats["rows"])
        for stats in season_stats.values()
    )
    total_games_requested = sum(
        int(stats["games_requested"])
        for stats in season_stats.values()
    )
    total_fetch_failures = sum(
        int(stats["fetch_failures"])
        for stats in season_stats.values()
    )

    grade_counts: dict[str, int] = defaultdict(int)
    ungraded_reasons: dict[str, int] = defaultdict(int)

    for stats in season_stats.values():
        for grade, count in stats["grade_counts"].items():
            grade_counts[str(grade)] += int(count)

        for reason, count in stats["ungraded_reasons"].items():
            ungraded_reasons[str(reason)] += int(count)

    reporter.set_rows(
        rows_in=int(total_rows),
        rows_out=int(total_rows),
    )
    reporter.update_details(
        {
            "seasons": int(len(season_stats)),
            "games_requested": int(total_games_requested),
            "fetch_failures": int(total_fetch_failures),
            "grade_counts": dict(sorted(grade_counts.items())),
            "ungraded_reasons": dict(
                sorted(ungraded_reasons.items())
            ),
            "season_stats": season_stats,
        }
    )

    print(
        "GRADE PROPS: PASS "
        f"seasons={len(season_stats)} "
        f"rows={total_rows} "
        f"wins={grade_counts['win']} "
        f"losses={grade_counts['loss']} "
        f"pushes={grade_counts['push']} "
        f"ungraded={grade_counts['ungraded']} "
        f"fetch_failures={total_fetch_failures}"
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
