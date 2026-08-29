#!/usr/bin/env python3
"""
Grade final NFL selected bets against populated final-score files.

READS:
  docs/win/football/nfl/03_picks/selected/
      week_{week}_NFL_select_picks.csv

  docs/win/football/nfl/04_final_results/results/
      {season}_{season_type}_{week}.csv

WRITES:
  docs/win/football/nfl/04_final_results/results/graded/
      {season}_{season_type}_{week}_graded.csv

Behavior:
- Uses game_id to match selected bets to final results.
- Grades only games whose result status begins with "Final".
- Emits one row per selected market.
- Grade values are WIN, LOSS, or PUSH.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
NFL_ROOT = SCRIPT_DIR.parents[1]

SELECTED_DIR = NFL_ROOT / "03_picks" / "selected"
RESULTS_DIR = NFL_ROOT / "04_final_results" / "results"
GRADED_DIR = RESULTS_DIR / "graded"

SELECTED_PATTERN = "week_*_NFL_select_picks.csv"

SELECTED_REQUIRED_COLUMNS = [
    "season",
    "season_type",
    "week",
    "game_id",
    "game_date",
    "away_team",
    "home_team",
    "ml_selected",
    "ml_selection",
    "ml_odds_american",
    "spread_selected",
    "spread_selection",
    "spread_line",
    "spread_odds_american",
    "total_selected",
    "total_selection",
    "total_line",
    "total_odds_american",
]

RESULT_REQUIRED_COLUMNS = [
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
]

OUTPUT_COLUMNS = [
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
    "line",
    "odds_american",
    "result",
]

EPSILON = 1e-9


def clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def fail(message: str) -> None:
    raise RuntimeError(message)


def require_columns(fieldnames: list[str] | None, required: list[str], label: str) -> None:
    available = set(fieldnames or [])
    missing = [column for column in required if column not in available]
    if missing:
        fail(f"{label}: missing required columns: {missing}")


def read_csv(path: Path, required: list[str]) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require_columns(reader.fieldnames, required, str(path))
        return list(reader)


def selection_flag(value: Any) -> bool:
    text = clean(value)
    if not text:
        return False
    try:
        return float(text) == 1.0
    except ValueError:
        return False


def parse_number(value: Any, *, label: str) -> float:
    text = clean(value)
    if not text:
        fail(f"Missing numeric value for {label}")
    try:
        return float(text)
    except ValueError as exc:
        fail(f"Invalid numeric value for {label}: {value!r}")
        raise exc


def compare(left: float, right: float) -> int:
    difference = left - right
    if abs(difference) <= EPSILON:
        return 0
    return 1 if difference > 0 else -1


def is_final(status: Any) -> bool:
    return clean(status).casefold().startswith("final")


def grade_moneyline(selection: str, away_score: float, home_score: float) -> str:
    side = selection.upper()
    if side == "HOME":
        outcome = compare(home_score, away_score)
    elif side == "AWAY":
        outcome = compare(away_score, home_score)
    else:
        fail(f"Invalid moneyline selection: {selection!r}")

    if outcome > 0:
        return "WIN"
    if outcome < 0:
        return "LOSS"
    return "PUSH"


def grade_spread(
    selection: str,
    line: float,
    away_score: float,
    home_score: float,
) -> str:
    side = selection.upper()
    if side == "HOME":
        outcome = compare(home_score + line, away_score)
    elif side == "AWAY":
        outcome = compare(away_score + line, home_score)
    else:
        fail(f"Invalid spread selection: {selection!r}")

    if outcome > 0:
        return "WIN"
    if outcome < 0:
        return "LOSS"
    return "PUSH"


def grade_total(
    selection: str,
    line: float,
    away_score: float,
    home_score: float,
) -> str:
    side = selection.upper()
    actual_total = away_score + home_score
    outcome = compare(actual_total, line)

    if side == "OVER":
        if outcome > 0:
            return "WIN"
        if outcome < 0:
            return "LOSS"
        return "PUSH"

    if side == "UNDER":
        if outcome < 0:
            return "WIN"
        if outcome > 0:
            return "LOSS"
        return "PUSH"

    fail(f"Invalid total selection: {selection!r}")
    return ""


def build_output_row(
    selected: dict[str, str],
    result: dict[str, str],
    *,
    market: str,
    selection: str,
    line: str,
    odds_american: str,
    grade: str,
) -> dict[str, str]:
    return {
        "season": clean(selected["season"]),
        "season_type": clean(selected["season_type"]),
        "week": clean(selected["week"]),
        "game_id": clean(selected["game_id"]),
        "game_date": clean(result["game_date"]) or clean(selected["game_date"]),
        "away_team": clean(result["away_team"]) or clean(selected["away_team"]),
        "home_team": clean(result["home_team"]) or clean(selected["home_team"]),
        "away_score": clean(result["away_score"]),
        "home_score": clean(result["home_score"]),
        "status": clean(result["status"]),
        "market": market,
        "selection": selection,
        "line": line,
        "odds_american": odds_american,
        "result": grade,
    }


def grade_selected_row(
    selected: dict[str, str],
    result: dict[str, str],
) -> list[dict[str, str]]:
    game_id = clean(selected["game_id"])
    away_score = parse_number(result["away_score"], label=f"game_id={game_id} away_score")
    home_score = parse_number(result["home_score"], label=f"game_id={game_id} home_score")

    graded: list[dict[str, str]] = []

    if selection_flag(selected["ml_selected"]):
        selection = clean(selected["ml_selection"]).upper()
        grade = grade_moneyline(selection, away_score, home_score)
        graded.append(
            build_output_row(
                selected,
                result,
                market="MONEYLINE",
                selection=selection,
                line="",
                odds_american=clean(selected["ml_odds_american"]),
                grade=grade,
            )
        )

    if selection_flag(selected["spread_selected"]):
        selection = clean(selected["spread_selection"]).upper()
        line_text = clean(selected["spread_line"])
        line = parse_number(line_text, label=f"game_id={game_id} spread_line")
        grade = grade_spread(selection, line, away_score, home_score)
        graded.append(
            build_output_row(
                selected,
                result,
                market="SPREAD",
                selection=selection,
                line=line_text,
                odds_american=clean(selected["spread_odds_american"]),
                grade=grade,
            )
        )

    if selection_flag(selected["total_selected"]):
        selection = clean(selected["total_selection"]).upper()
        line_text = clean(selected["total_line"])
        line = parse_number(line_text, label=f"game_id={game_id} total_line")
        grade = grade_total(selection, line, away_score, home_score)
        graded.append(
            build_output_row(
                selected,
                result,
                market="TOTAL",
                selection=selection,
                line=line_text,
                odds_american=clean(selected["total_odds_american"]),
                grade=grade,
            )
        )

    return graded


def load_selected_groups() -> dict[tuple[str, str, str], list[dict[str, str]]]:
    selected_files = sorted(SELECTED_DIR.glob(SELECTED_PATTERN))
    if not selected_files:
        fail(f"No selected pick files found in {SELECTED_DIR}")

    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)

    for path in selected_files:
        for row in read_csv(path, SELECTED_REQUIRED_COLUMNS):
            season = clean(row["season"])
            season_type = clean(row["season_type"])
            week = clean(row["week"])
            game_id = clean(row["game_id"])

            if not season or not season_type or not week or not game_id:
                fail(f"{path}: selected row missing season/season_type/week/game_id")

            groups[(season, season_type, week)].append(row)

    return groups


def index_results(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(path, RESULT_REQUIRED_COLUMNS)
    indexed: dict[str, dict[str, str]] = {}

    for row in rows:
        game_id = clean(row["game_id"])
        if not game_id:
            fail(f"{path}: result row missing game_id")
        if game_id in indexed:
            fail(f"{path}: duplicate game_id={game_id}")
        indexed[game_id] = row

    return indexed


def write_graded(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    groups = load_selected_groups()
    GRADED_DIR.mkdir(parents=True, exist_ok=True)

    files_written = 0
    bets_graded = 0
    pending_games = 0

    for (season, season_type, week), selected_rows in sorted(groups.items()):
        results_path = RESULTS_DIR / f"{season}_{season_type}_{week}.csv"
        if not results_path.exists():
            fail(f"Missing final-results file: {results_path}")

        results_by_game = index_results(results_path)
        graded_rows: list[dict[str, str]] = []

        for selected in selected_rows:
            game_id = clean(selected["game_id"])
            result = results_by_game.get(game_id)
            if result is None:
                fail(f"{results_path}: missing selected game_id={game_id}")

            if not is_final(result["status"]):
                pending_games += 1
                continue

            graded_rows.extend(grade_selected_row(selected, result))

        output_path = GRADED_DIR / f"{season}_{season_type}_{week}_graded.csv"
        write_graded(output_path, graded_rows)

        files_written += 1
        bets_graded += len(graded_rows)
        print(f"wrote {len(graded_rows)} graded bets to {output_path}")

    print(
        f"files_written={files_written} bets_graded={bets_graded} "
        f"pending_selected_games={pending_games}"
    )


if __name__ == "__main__":
    main()
