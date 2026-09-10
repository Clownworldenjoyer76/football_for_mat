#!/usr/bin/env python3
"""Grade Stage 3 NFL props from completed ESPN game box scores and build reports."""

from __future__ import annotations

import csv
import json
import math
import re
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PROP_ENGINE_ROOT = Path("docs/win/football/nfl/prop_engine")
PROP_PICKS_ROOT = PROP_ENGINE_ROOT / "prop_picks_final"
FINAL_ROOT = PROP_ENGINE_ROOT / "05_final"
GRADED_ROOT = FINAL_ROOT / "graded"
REPORTS_ROOT = FINAL_ROOT / "reports"

ESPN_SUMMARY_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={game_id}"
)
USER_AGENT = "football_for_mat-prop-grader/1.0"
REQUEST_TIMEOUT_SECONDS = 30


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
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        },
    )

    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        payload = json.load(response)

    if not isinstance(payload, dict):
        raise ValueError("ESPN summary response is not a JSON object")

    return payload


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

    players = player_stat_map(payload)
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


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


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


def build_reports(season: str, rows: list[dict[str, str]]) -> None:
    report_dir = REPORTS_ROOT / season
    bets = betting_rows(rows)

    overall = {"season": season, **metrics(rows)}
    write_csv(
        report_dir / "overall.csv",
        ["season", *REPORT_METRIC_COLUMNS],
        [overall],
    )

    fields, report = grouped_report(bets, "prop_type", lambda row: row.get("prop_type"))
    write_csv(report_dir / "by_prop_type.csv", fields, report)

    fields, report = grouped_report(bets, "probability_bucket", lambda row: probability_bucket(row.get("pick_prob")))
    write_csv(report_dir / "by_probability.csv", fields, report)

    fields, report = grouped_report(bets, "pick", lambda row: clean(row.get("pick")).casefold())
    write_csv(report_dir / "by_pick_direction.csv", fields, report)

    fields, report = grouped_report(bets, "week", lambda row: row.get("week"))
    write_csv(report_dir / "by_week.csv", fields, report)

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

    write_csv(
        report_dir / "calibration.csv",
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


def process_season(season: str, files: list[tuple[int, Path]]) -> None:
    original_fields, rows = read_stage3_files(files)
    if not rows:
        return

    graded_path = GRADED_ROOT / season / f"{season}_all_props_graded.csv"
    existing = load_existing_grades(graded_path)
    game_ids = build_game_requests(rows, existing)

    summaries: dict[str, dict[str, Any]] = {}
    summary_errors: dict[str, str] = {}

    for game_id in sorted(game_ids):
        try:
            summaries[game_id] = fetch_espn_summary(game_id)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            summary_errors[game_id] = f"{type(exc).__name__}: {exc}"

    graded_rows = [
        grade_row(row, summaries, summary_errors, existing)
        for row in rows
    ]

    fieldnames = list(original_fields)
    for field in ["week", "prop_type", "final_stat", "grade", "grade_reason"]:
        if field not in fieldnames:
            fieldnames.append(field)

    write_csv(graded_path, fieldnames, graded_rows)
    build_reports(season, graded_rows)

    grade_counts: dict[str, int] = defaultdict(int)
    for row in graded_rows:
        grade_counts[clean(row.get("grade")) or "blank"] += 1

    print(
        f"{season}: rows={len(graded_rows)} "
        f"wins={grade_counts['win']} losses={grade_counts['loss']} "
        f"pushes={grade_counts['push']} ungraded={grade_counts['ungraded']} "
        f"no_bet={grade_counts['no_bet']}"
    )

    if summary_errors:
        print(f"{season}: ESPN fetch failures={len(summary_errors)}", file=sys.stderr)
        for game_id, error in sorted(summary_errors.items()):
            print(f"  {game_id}: {error}", file=sys.stderr)


def main() -> None:
    discovered = discover_stage3()
    if not discovered:
        raise FileNotFoundError(f"No Stage 3 prop files found under {PROP_PICKS_ROOT}")

    for season in sorted(discovered):
        process_season(season, discovered[season])


if __name__ == "__main__":
    main()
