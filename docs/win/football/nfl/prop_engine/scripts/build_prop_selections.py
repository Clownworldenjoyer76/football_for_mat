#!/usr/bin/env python3
"""
Combine weekly ESPN prop files with prop-engine projections.

Inputs:
  docs/win/football/nfl/prop_engine/output/{season}/week_{week}_props/
  docs/win/football/nfl/prop_engine/output/{season}/week_{week}_player_projections_wide.csv
  docs/win/football/nfl/data/master/roster_master.csv

Outputs:
  docs/win/football/nfl/prop_engine/output/{season}/week_{week}_props/
      selections/{category}/week_{week}_props_select.csv

The output is prop-row driven: every sportsbook prop row is preserved.
Actual sportsbook fields are prefixed with "actual_prop_".
Prop-engine fields are prefixed with "prop_engine_".
"""

from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from pathlib import Path
from typing import Iterable

BASE_DIR = Path("docs/win/football/nfl")
OUTPUT_ROOT = BASE_DIR / "prop_engine" / "output"
ROSTER_PATH = BASE_DIR / "data" / "master" / "roster_master.csv"

CATEGORIES = (
    "passing",
    "rushing",
    "receiving",
    "combo",
    "tds",
    "1sthalf",
    "1stquarter",
    "defense",
    "kicking",
)

EXPECTED_PROP_COLUMNS = {
    "passing": [
        "Total Passing Yards",
        "Total Pass Completions",
        "Total Passing Attempts",
        "Total Passing Touchdowns",
        "Total Passing Interceptions",
    ],
    "rushing": [
        "Total Carries",
        "Total Rushing Yards",
        "Longest Rush",
    ],
    "receiving": [
        "Total Receiving Yards",
        "Total Receptions",
        "Longest Reception",
        "Receiving Yards",
    ],
    "combo": [
        "Total Passing Plus Rushing Yards",
        "Total Rushing Plus Receiving Yards",
    ],
    "tds": [
        "Anytime Touchdown Scorer",
        "First Touchdown Scorer",
        "Last Touchdown Scorer",
        "First Team Touchdown Scorer",
        "Player to score 2+ touchdowns",
        "Player to score 3+ touchdowns",
    ],
    "1sthalf": [
        "Total Passing Yards",
        "Total Receiving Yards",
        "Total Rushing Yards",
        "Touchdown Scorer",
    ],
    "1stquarter": [
        "Total Passing Yards",
        "Total Receiving Yards",
        "Total Rushing Yards",
    ],
    "defense": [
        "Total Tackles",
        "Total Assists",
        "Total Tackles Plus Assists",
        "Total Sacks",
    ],
    "kicking": [
        "Total Kicking Points",
        "Total Field Goals Made",
        "Total Extra Points Made",
    ],
}

PROP_KEY_FIELDS = {
    "game_date",
    "game_id",
    "player_name",
    "player_id",
    "espn_player_id",
    "athlete_id",
}

ENGINE_KEY_FIELDS = {
    "game_date",
    "game_id",
    "player_name",
    "player_id",
}


def normalize_id(value: object) -> str:
    text = str(value or "").strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def normalize_name(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        return rows, list(reader.fieldnames or [])


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def resolve_week(season: int, explicit_week: int | None) -> int:
    if explicit_week is not None:
        return explicit_week

    season_root = OUTPUT_ROOT / str(season)
    candidates: list[int] = []

    if season_root.exists():
        for path in season_root.iterdir():
            if not path.is_dir():
                continue
            match = re.fullmatch(r"week_(\d+)_props", path.name)
            if not match:
                continue
            week = int(match.group(1))
            projections = season_root / f"week_{week}_player_projections_wide.csv"
            if projections.exists():
                candidates.append(week)

    if not candidates:
        raise FileNotFoundError(
            f"No week_N_props folder with a matching week_N_player_projections_wide.csv "
            f"was found under {season_root}"
        )

    return max(candidates)


def load_roster() -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    if not ROSTER_PATH.exists():
        raise FileNotFoundError(f"Missing roster master: {ROSTER_PATH}")

    rows, _ = read_csv(ROSTER_PATH)

    by_id: dict[str, dict[str, str]] = {}
    by_name: dict[str, dict[str, str]] = {}

    for row in rows:
        for field in ("id", "alternateIds.sdr"):
            value = normalize_id(row.get(field))
            if value:
                by_id[value] = row

        for field in ("fullName", "displayName", "shortName"):
            value = normalize_name(row.get(field))
            if value and value not in by_name:
                by_name[value] = row

    return by_id, by_name


def load_projections(
    path: Path,
) -> tuple[
    list[str],
    dict[tuple[str, str], dict[str, str]],
    dict[str, dict[str, str]],
]:
    if not path.exists():
        raise FileNotFoundError(f"Missing projection file: {path}")

    rows, fieldnames = read_csv(path)

    required = {"game_id", "player_name"}
    missing = required - set(fieldnames)
    if missing:
        raise RuntimeError(
            f"{path} is missing required columns: {', '.join(sorted(missing))}"
        )

    by_game_name: dict[tuple[str, str], dict[str, str]] = {}
    name_buckets: dict[str, list[dict[str, str]]] = {}

    for row in rows:
        game_id = normalize_id(row.get("game_id"))
        player_name = normalize_name(row.get("player_name"))

        if game_id and player_name:
            by_game_name[(game_id, player_name)] = row

        if player_name:
            name_buckets.setdefault(player_name, []).append(row)

    unique_by_name = {
        name: bucket[0]
        for name, bucket in name_buckets.items()
        if len(bucket) == 1
    }

    return fieldnames, by_game_name, unique_by_name


def prop_player_id(row: dict[str, str]) -> str:
    for field in ("espn_player_id", "player_id", "athlete_id"):
        value = normalize_id(row.get(field))
        if value:
            return value
    return ""


def prop_player_name(row: dict[str, str]) -> str:
    return str(row.get("player_name") or "").strip()


def find_roster_row(
    prop_row: dict[str, str],
    roster_by_id: dict[str, dict[str, str]],
    roster_by_name: dict[str, dict[str, str]],
) -> dict[str, str] | None:
    espn_id = prop_player_id(prop_row)
    if espn_id and espn_id in roster_by_id:
        return roster_by_id[espn_id]

    name = normalize_name(prop_player_name(prop_row))
    if name and name in roster_by_name:
        return roster_by_name[name]

    return None


def roster_names(row: dict[str, str] | None) -> list[str]:
    if not row:
        return []

    values: list[str] = []
    for field in ("fullName", "displayName", "shortName"):
        name = normalize_name(row.get(field))
        if name and name not in values:
            values.append(name)
    return values


def find_projection_row(
    prop_row: dict[str, str],
    roster_row: dict[str, str] | None,
    by_game_name: dict[tuple[str, str], dict[str, str]],
    unique_by_name: dict[str, dict[str, str]],
) -> dict[str, str] | None:
    game_id = normalize_id(prop_row.get("game_id"))

    candidate_names = roster_names(roster_row)
    raw_prop_name = normalize_name(prop_player_name(prop_row))
    if raw_prop_name and raw_prop_name not in candidate_names:
        candidate_names.append(raw_prop_name)

    for name in candidate_names:
        match = by_game_name.get((game_id, name))
        if match:
            return match

    for name in candidate_names:
        match = unique_by_name.get(name)
        if match:
            return match

    return None


def discover_prop_files(category_dir: Path) -> list[Path]:
    if not category_dir.exists():
        return []

    return sorted(
        path
        for path in category_dir.glob("*.csv")
        if path.is_file() and not path.name.endswith("_props_select.csv")
    )


def union_headers(headers: list[list[str]]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    for group in headers:
        for field in group:
            if field not in seen:
                seen.add(field)
                result.append(field)

    return result


def build_category(
    *,
    season: int,
    week: int,
    category: str,
    props_root: Path,
    projection_fields: list[str],
    projection_by_game_name: dict[tuple[str, str], dict[str, str]],
    projection_unique_by_name: dict[str, dict[str, str]],
    roster_by_id: dict[str, dict[str, str]],
    roster_by_name: dict[str, dict[str, str]],
) -> None:
    source_dir = props_root / category
    source_files = discover_prop_files(source_dir)

    source_rows: list[dict[str, str]] = []
    source_headers: list[list[str]] = []

    for source_file in source_files:
        rows, fields = read_csv(source_file)
        source_rows.extend(rows)
        source_headers.append(fields)

    discovered_prop_fields = [
        field
        for field in union_headers(source_headers)
        if field not in PROP_KEY_FIELDS
    ]

    actual_fields = []
    for field in EXPECTED_PROP_COLUMNS.get(category, []) + discovered_prop_fields:
        if field not in actual_fields:
            actual_fields.append(field)

    engine_fields = [
        field
        for field in projection_fields
        if field not in ENGINE_KEY_FIELDS
    ]

    output_fields = [
        "game_date",
        "game_id",
        "player_name",
        "espn_player_id",
        "prop_engine_player_id",
        "prop_engine_player_name",
    ]
    output_fields.extend(f"actual_prop_{field}" for field in actual_fields)
    output_fields.extend(f"prop_engine_{field}" for field in engine_fields)

    output_rows: list[dict[str, object]] = []
    matched = 0

    for prop_row in source_rows:
        roster_row = find_roster_row(prop_row, roster_by_id, roster_by_name)
        projection_row = find_projection_row(
            prop_row,
            roster_row,
            projection_by_game_name,
            projection_unique_by_name,
        )

        if projection_row:
            matched += 1

        espn_id = prop_player_id(prop_row)
        if not espn_id and roster_row:
            espn_id = normalize_id(roster_row.get("id"))

        canonical_name = ""
        if roster_row:
            canonical_name = str(
                roster_row.get("fullName")
                or roster_row.get("displayName")
                or ""
            ).strip()

        if not canonical_name:
            canonical_name = prop_player_name(prop_row)

        output_row: dict[str, object] = {
            "game_date": prop_row.get("game_date", ""),
            "game_id": normalize_id(prop_row.get("game_id")),
            "player_name": canonical_name,
            "espn_player_id": espn_id,
            "prop_engine_player_id": (
                projection_row.get("player_id", "") if projection_row else ""
            ),
            "prop_engine_player_name": (
                projection_row.get("player_name", "") if projection_row else ""
            ),
        }

        for field in actual_fields:
            output_row[f"actual_prop_{field}"] = prop_row.get(field, "")

        for field in engine_fields:
            output_row[f"prop_engine_{field}"] = (
                projection_row.get(field, "") if projection_row else ""
            )

        output_rows.append(output_row)

    output_path = (
        props_root
        / "selections"
        / category
        / f"week_{week}_props_select.csv"
    )

    write_csv(output_path, output_rows, output_fields)

    print(
        f"{category}: rows={len(output_rows)} "
        f"projection_matches={matched} "
        f"unmatched={len(output_rows) - matched} "
        f"output={output_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--week", type=int)
    args = parser.parse_args()

    season = args.season
    week = resolve_week(season, args.week)

    season_root = OUTPUT_ROOT / str(season)
    props_root = season_root / f"week_{week}_props"
    projection_path = season_root / f"week_{week}_player_projections_wide.csv"

    if not props_root.exists():
        raise FileNotFoundError(f"Missing weekly props folder: {props_root}")

    roster_by_id, roster_by_name = load_roster()

    (
        projection_fields,
        projection_by_game_name,
        projection_unique_by_name,
    ) = load_projections(projection_path)

    for category in CATEGORIES:
        build_category(
            season=season,
            week=week,
            category=category,
            props_root=props_root,
            projection_fields=projection_fields,
            projection_by_game_name=projection_by_game_name,
            projection_unique_by_name=projection_unique_by_name,
            roster_by_id=roster_by_id,
            roster_by_name=roster_by_name,
        )


if __name__ == "__main__":
    main()
