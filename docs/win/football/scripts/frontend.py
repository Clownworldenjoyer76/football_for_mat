#!/usr/bin/env python3
"""Export the three football frontend CSV files (CFB, NFL, NFL props).

Run from any working directory:
    python frontend.py

Writes to frontend/data/football/:
    football_slate.csv  CFB + NFL schedule/odds + selected picks, one row per game, all weeks
    football_props.csv  all NFL player-prop files stacked (season, week, prop_type added)
    football_final.csv  CFB graded results, all weeks stacked

Slate rows: CFB starts from the season schedule, NFL from the weekly schedule; weekly
odds and selected picks are joined on game_id. Pick columns are blank for games
without a selection. Source files are never modified.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

LEAGUES = ("cfb", "nfl")
PROP_TYPES = (
    ("passing", "actual_prop_total_passing_yards"),
    ("rushing", "actual_prop_total_rushing_yards"),
    ("receiving", "actual_prop_total_receiving_yards"),
    ("kicking", "actual_prop_total_kicking_points"),
    ("defense", "actual_prop_total_tackles"),
    ("pass_rush", "actual_prop_total_passing_plus_rushing_yards"),
    ("rec_rush", "actual_prop_total_rushing_plus_receiving_yards"),
)
SLATE_FIRST_COLUMNS = (
    "league", "season", "season_type", "week", "game_id", "game_date",
    "game_time", "away_team", "home_team",
)
PROPS_FIRST_COLUMNS = (
    "season", "week", "prop_type", "game_date", "game_id", "player_name",
    "pick", "pick_prob", "over_prob", "under_prob",
)
FINAL_FIRST_COLUMNS = (
    "league", "season", "season_type", "week", "game_id", "game_date",
    "game_time", "away_team", "home_team",
)


def find_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "docs/win/football").is_dir():
            return parent
    raise FileNotFoundError("Could not find docs/win/football above " + str(here))


ROOT = find_root()
FB = ROOT / "docs/win/football"
OUT = ROOT / "frontend/data/football"
SLATE_DEST = OUT / "football_slate.csv"
PROPS_DEST = OUT / "football_props.csv"
FINAL_DEST = OUT / "football_final.csv"


def clean(value: str | None) -> str:
    if value is None:
        return ""
    result = value.strip()
    return "" if result.lower() in {"nan", "none", "null"} else result


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        return [], []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        if not headers:
            return [], []
        if any(not h for h in headers) or len(headers) != len(set(headers)):
            raise ValueError(f"Invalid or duplicate CSV headers in {path}")
        rows = []
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"Malformed CSV row in {path} at line {line_number}")
            rows.append({k: clean(v) for k, v in row.items()})
    return headers, rows


def fill_missing(target: dict[str, str], source_row: dict[str, str]) -> None:
    for field, value in source_row.items():
        if value and not target.get(field, ""):
            target[field] = value


def write_csv(path: Path, headers: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def ordered_headers(first: tuple[str, ...], seen: list[str]) -> list[str]:
    headers = [h for h in first if h in seen]
    headers += [h for h in seen if h not in headers]
    return headers


def numbered_files(folder: Path, pattern: str) -> list[tuple[int, Path]]:
    """Files in folder whose name fully matches pattern (one group: the week number)."""
    if not folder.is_dir():
        return []
    regex = re.compile(pattern, re.IGNORECASE)
    found = [
        (int(match.group(1)), path)
        for path in folder.iterdir()
        if path.is_file() and (match := regex.fullmatch(path.name))
    ]
    return sorted(found, key=lambda item: item[0])


def finish(path: Path, first: tuple[str, ...], columns: list[str],
           rows: list[dict[str, str]], label: str) -> None:
    if not rows:
        print(f"WARNING: No {label} rows found; {rel(path)} not written")
        return
    headers = ordered_headers(first, columns)
    for row in rows:
        for field in headers:
            row.setdefault(field, "")
    write_csv(path, headers, rows)
    print(f"WROTE {rel(path)}: {len(rows)} rows")


def build_slate() -> None:
    columns: list[str] = []
    rows_out: list[dict[str, str]] = []

    def include(fields: list[str]) -> None:
        for field in fields:
            if field not in columns:
                columns.append(field)

    include(["league"])

    for league in LEAGUES:
        league_up = league.upper()
        games: dict[str, dict[str, str]] = {}

        def add_rows(path: Path, rows: list[dict[str, str]], *, create: bool) -> None:
            for row in rows:
                game_id = row.get("game_id", "")
                if not game_id:
                    print(f"WARNING: Skipping row with blank game_id in {rel(path)}")
                    continue
                game = games.get(game_id)
                if game is None:
                    if not create:
                        print(f"WARNING: Unmatched game_id={game_id} in {rel(path)}; added as its own row")
                    game = {"league": league_up}
                    games[game_id] = game
                fill_missing(game, row)

        # CFB season schedules list every game; NFL games come from the weekly schedule.
        schedule_dir = FB / f"{league}/00_intake/schedule"
        if schedule_dir.is_dir():
            season_pattern = re.compile(r"(\d{4})_schedule\.csv", re.IGNORECASE)
            for path in sorted(schedule_dir.iterdir()):
                if path.is_file() and season_pattern.fullmatch(path.name):
                    headers, rows = read_csv(path)
                    include(headers)
                    add_rows(path, rows, create=True)

        for _, path in numbered_files(
            schedule_dir / "weekly", rf"week_(\d+)_{league_up}_weekly_schedule\.csv"
        ):
            headers, rows = read_csv(path)
            include(headers)
            add_rows(path, rows, create=True)

        pick_games = 0
        for _, path in numbered_files(
            FB / f"{league}/03_picks/selected", rf"week_(\d+)_{league_up}_select_picks\.csv"
        ):
            headers, rows = read_csv(path)
            include(headers)
            add_rows(path, rows, create=False)
            pick_games += len(rows)

        rows_out.extend(games.values())
        print(f"{league_up}: {len(games)} games, {pick_games} pick rows")

    rows_out.sort(key=lambda r: (r.get("league", ""), r.get("game_date", ""),
                                 r.get("game_time", ""), r.get("game_id", "")))
    finish(SLATE_DEST, SLATE_FIRST_COLUMNS, columns, rows_out, "football slate")


def build_props() -> None:
    root = FB / "prop_engine/prop_picks_final"
    columns: list[str] = []
    rows_out: list[dict[str, str]] = []
    files_read = 0

    def include(fields: list[str]) -> None:
        for field in fields:
            if field not in columns:
                columns.append(field)

    include(["season", "week", "prop_type"])

    if root.is_dir():
        for season_dir in sorted(root.iterdir()):
            if not (season_dir.is_dir() and re.fullmatch(r"\d{4}", season_dir.name)):
                continue
            stage = season_dir / "stage_3"
            for week, path in numbered_files(
                stage, rf"{season_dir.name}_(\d+)_all_props\.csv"
            ):
                headers, rows = read_csv(path)
                if not headers:
                    raise RuntimeError(f"Empty or invalid Stage 3 file: {path}")
                files_read += 1
                include(headers)
                for row in rows:
                    categories = [
                        prop_type
                        for prop_type, line_field in PROP_TYPES
                        if clean(row.get(line_field))
                    ]
                    if len(categories) != 1:
                        raise RuntimeError(
                            f"Expected one prop type in filtered Stage 3: {path} "
                            f"game_id={row.get('game_id')} "
                            f"player_name={row.get('player_name')} "
                            f"categories={categories}"
                        )
                    row["season"] = season_dir.name
                    row["week"] = str(week)
                    row["prop_type"] = categories[0]
                    rows_out.append(row)

    if files_read == 0 or not rows_out:
        raise RuntimeError("No filtered Stage 3 props found; refusing stale frontend output")

    print(f"Filtered props: {len(rows_out)} rows")
    finish(PROPS_DEST, PROPS_FIRST_COLUMNS, columns, rows_out, "props")


def build_final() -> None:
    columns: list[str] = []
    rows_out: list[dict[str, str]] = []
    for week, path in numbered_files(
        FB / "cfb/04_final_results/graded", r"week_(\d+)_CFB_graded\.csv"
    ):
        headers, rows = read_csv(path)
        for field in headers:
            if field not in columns:
                columns.append(field)
        for row in rows:
            row["league"] = "CFB"
            rows_out.append(row)
    if "league" not in columns:
        columns.append("league")
    finish(FINAL_DEST, FINAL_FIRST_COLUMNS, columns, rows_out, "final")


def main() -> None:
    build_slate()
    build_props()
    build_final()
    print("Export complete")


if __name__ == "__main__":
    main()
