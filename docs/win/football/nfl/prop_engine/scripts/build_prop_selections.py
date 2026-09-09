#!/usr/bin/env python3
"""
Combine weekly ESPN prop files with prop-engine projections.

Inputs:
  docs/win/football/nfl/prop_engine/output/{season}/week_{week}_props/
  docs/win/football/nfl/prop_engine/output/{season}/week_{week}_player_projections_wide.csv
  docs/win/football/nfl/prop_engine/data/identity/player_crosswalk.parquet
  docs/win/football/nfl/data/master/roster_master.csv
  docs/win/football/nfl/data/master/depth_charts/*/*_depth.csv

Outputs:
  docs/win/football/nfl/prop_engine/output/{season}/week_{week}_props/
      selections/{category}/week_{week}_props_select.csv

The output is prop-row driven: every sportsbook prop row is preserved.
Actual sportsbook fields are prefixed with "actual_prop_".
Prop-engine fields are prefixed with "prop_engine_".
Each category only carries projection fields relevant to that prop family.

Player matching:
  1. Read espn_player_id from the prop file.
  2. Find that ESPN ID in player_crosswalk.parquet.
  3. Get the corresponding gsis_id.
  4. Match that gsis_id to player_id in week_{week}_player_projections_wide.csv.
  5. No player-name matching is used for projection matching.
"""

from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import pandas as pd


BASE_DIR = Path("docs/win/football/nfl")
OUTPUT_ROOT = BASE_DIR / "prop_engine" / "output"
CROSSWALK_PATH = (
    BASE_DIR
    / "prop_engine"
    / "data"
    / "identity"
    / "player_crosswalk.parquet"
)
ROSTER_PATH = BASE_DIR / "data" / "master" / "roster_master.csv"
DEPTH_CHART_ROOT = BASE_DIR / "data" / "master" / "depth_charts"


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


ACTUAL_FIELD_MAP = {
    "passing": {
        "Total Passing Yards": "actual_prop_total_passing_yards",
        "Total Pass Completions": "actual_prop_total_pass_completions",
        "Total Passing Attempts": "actual_prop_total_passing_attempts",
        "Total Passing Touchdowns": "actual_prop_total_passing_touchdowns",
        "Total Passing Interceptions": "actual_prop_total_passing_interceptions",
    },
    "rushing": {
        "Total Carries": "actual_prop_total_carries",
        "Total Rushing Yards": "actual_prop_total_rushing_yards",
        "Longest Rush": "actual_prop_longest_rush",
    },
    "receiving": {
        "Total Receiving Yards": "actual_prop_total_receiving_yards",
        "Total Receptions": "actual_prop_total_receptions",
        "Longest Reception": "actual_prop_longest_reception",
        "Receiving Yards": "actual_prop_receiving_yards_milestones",
    },
    "combo": {
        "Total Passing Plus Rushing Yards": "actual_prop_total_passing_plus_rushing_yards",
        "Total Rushing Plus Receiving Yards": "actual_prop_total_rushing_plus_receiving_yards",
    },
    "tds": {
        "Anytime Touchdown Scorer": "actual_prop_anytime_touchdown_scorer",
        "First Touchdown Scorer": "actual_prop_first_touchdown_scorer",
        "Last Touchdown Scorer": "actual_prop_last_touchdown_scorer",
        "First Team Touchdown Scorer": "actual_prop_first_team_touchdown_scorer",
        "Player to score 2+ touchdowns": "actual_prop_player_2plus_touchdowns",
        "Player to score 3+ touchdowns": "actual_prop_player_3plus_touchdowns",
    },
    "1sthalf": {
        "Total Passing Yards": "actual_prop_1st_half_total_passing_yards",
        "Total Receiving Yards": "actual_prop_1st_half_total_receiving_yards",
        "Total Rushing Yards": "actual_prop_1st_half_total_rushing_yards",
        "Touchdown Scorer": "actual_prop_1st_half_touchdown_scorer",
    },
    "1stquarter": {
        "Total Passing Yards": "actual_prop_1st_quarter_total_passing_yards",
        "Total Receiving Yards": "actual_prop_1st_quarter_total_receiving_yards",
        "Total Rushing Yards": "actual_prop_1st_quarter_total_rushing_yards",
    },
    "defense": {
        "Total Tackles": "actual_prop_total_tackles",
        "Total Assists": "actual_prop_total_assists",
        "Total Tackles Plus Assists": "actual_prop_total_tackles_plus_assists",
        "Total Sacks": "actual_prop_total_sacks",
    },
    "kicking": {
        "Total Kicking Points": "actual_prop_total_kicking_points",
        "Total Field Goals Made": "actual_prop_total_field_goals_made",
        "Total Extra Points Made": "actual_prop_total_extra_points_made",
    },
}


COMMON_ENGINE_FIELDS = [
    "season",
    "week",
    "team",
    "opponent",
    "position",
    "injury_game_status",
    "role_status",
    "generated_at",
]


CATEGORY_ENGINE_FIELDS = {
    "passing": [
        "passing_yards",
        "passing_yards_low",
        "passing_yards_high",
        "passing_tds",
        "passing_tds_prob_1plus",
    ],
    "rushing": [
        "rushing_yards",
        "rushing_yards_low",
        "rushing_yards_high",
    ],
    "receiving": [
        "receiving_yards",
        "receiving_yards_low",
        "receiving_yards_high",
    ],
    "combo": [
        "passing_yards",
        "passing_yards_low",
        "passing_yards_high",
        "rushing_yards",
        "rushing_yards_low",
        "rushing_yards_high",
        "receiving_yards",
        "receiving_yards_low",
        "receiving_yards_high",
    ],
    "tds": [],
    "1sthalf": [
        "passing_yards",
        "passing_yards_low",
        "passing_yards_high",
        "rushing_yards",
        "rushing_yards_low",
        "rushing_yards_high",
        "receiving_yards",
        "receiving_yards_low",
        "receiving_yards_high",
    ],
    "1stquarter": [
        "passing_yards",
        "passing_yards_low",
        "passing_yards_high",
        "rushing_yards",
        "rushing_yards_low",
        "rushing_yards_high",
        "receiving_yards",
        "receiving_yards_low",
        "receiving_yards_high",
    ],
    "defense": [
        "tackles",
        "tackles_low",
        "tackles_high",
    ],
    "kicking": [
        "kicking_points",
        "kicking_points_low",
        "kicking_points_high",
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


def normalize_id(value: object) -> str:
    text = str(value or "").strip()

    if text.lower() in {
        "",
        "nan",
        "none",
        "null",
        "<na>",
    }:
        return ""

    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]

    return text


def normalize_name(value: object) -> str:
    text = str(value or "").strip()

    if not text:
        return ""

    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        ch
        for ch in text
        if not unicodedata.combining(ch)
    )

    return re.sub(
        r"[^a-z0-9]+",
        "",
        text.lower(),
    )


def read_csv(
    path: Path,
) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]

        return rows, list(reader.fieldnames or [])


def write_csv(
    path: Path,
    rows: Iterable[dict[str, object]],
    fieldnames: list[str],
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
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    name: row.get(name, "")
                    for name in fieldnames
                }
            )


def resolve_week(
    season: int,
    explicit_week: int | None,
) -> int:
    if explicit_week is not None:
        return explicit_week

    season_root = OUTPUT_ROOT / str(season)
    candidates: list[int] = []

    if season_root.exists():
        for path in season_root.iterdir():
            if not path.is_dir():
                continue

            match = re.fullmatch(
                r"week_(\d+)_props",
                path.name,
            )

            if not match:
                continue

            week = int(match.group(1))

            projections = (
                season_root
                / f"week_{week}_player_projections_wide.csv"
            )

            if projections.exists():
                candidates.append(week)

    if not candidates:
        raise FileNotFoundError(
            f"No week_N_props folder with a matching "
            f"week_N_player_projections_wide.csv "
            f"was found under {season_root}"
        )

    return max(candidates)


def load_player_crosswalk() -> dict[str, str]:
    if not CROSSWALK_PATH.exists():
        raise FileNotFoundError(
            f"Missing player crosswalk: {CROSSWALK_PATH}"
        )

    frame = pd.read_parquet(CROSSWALK_PATH)

    required = {
        "espn_id",
        "gsis_id",
    }

    missing = required - set(frame.columns)

    if missing:
        raise RuntimeError(
            f"{CROSSWALK_PATH} is missing required columns: "
            f"{', '.join(sorted(missing))}"
        )

    espn_to_gsis: dict[str, str] = {}

    for row in frame.to_dict("records"):
        espn_id = normalize_id(row.get("espn_id"))
        gsis_id = normalize_id(row.get("gsis_id"))

        if not espn_id or not gsis_id:
            continue

        existing = espn_to_gsis.get(espn_id)

        if existing and existing != gsis_id:
            raise RuntimeError(
                f"Conflicting GSIS IDs for ESPN ID {espn_id}: "
                f"{existing} and {gsis_id}"
            )

        espn_to_gsis[espn_id] = gsis_id

    if not espn_to_gsis:
        raise RuntimeError(
            f"No ESPN-to-GSIS player mappings found in "
            f"{CROSSWALK_PATH}"
        )

    return espn_to_gsis


def load_roster() -> dict[str, dict[str, str]]:
    if not ROSTER_PATH.exists():
        raise FileNotFoundError(
            f"Missing roster master: {ROSTER_PATH}"
        )

    rows, _ = read_csv(ROSTER_PATH)

    by_id: dict[str, dict[str, str]] = {}

    for row in rows:
        for field in (
            "id",
            "alternateIds.sdr",
        ):
            value = normalize_id(row.get(field))

            if value:
                by_id[value] = row

    return by_id


def load_depth_charts() -> dict[str, dict[str, str]]:
    if not DEPTH_CHART_ROOT.exists():
        raise FileNotFoundError(
            f"Missing depth-chart folder: {DEPTH_CHART_ROOT}"
        )

    by_id: dict[str, dict[str, str]] = {}

    files = sorted(
        DEPTH_CHART_ROOT.glob("*/*_depth.csv")
    )

    if not files:
        raise FileNotFoundError(
            f"No depth-chart CSV files found under "
            f"{DEPTH_CHART_ROOT}"
        )

    for path in files:
        rows, _ = read_csv(path)

        for row in rows:
            espn_id = normalize_id(
                row.get("player_id")
            )

            if espn_id and espn_id not in by_id:
                by_id[espn_id] = row

    return by_id


def load_projections(
    path: Path,
) -> tuple[
    list[str],
    dict[tuple[str, str], dict[str, str]],
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing projection file: {path}"
        )

    rows, fieldnames = read_csv(path)

    required = {
        "game_id",
        "player_id",
    }

    missing = required - set(fieldnames)

    if missing:
        raise RuntimeError(
            f"{path} is missing required columns: "
            f"{', '.join(sorted(missing))}"
        )

    by_game_player_id: dict[
        tuple[str, str],
        dict[str, str],
    ] = {}

    for row in rows:
        game_id = normalize_id(
            row.get("game_id")
        )

        player_id = normalize_id(
            row.get("player_id")
        )

        if not game_id or not player_id:
            continue

        key = (
            game_id,
            player_id,
        )

        if key in by_game_player_id:
            raise RuntimeError(
                f"Duplicate projection row for "
                f"game_id={game_id}, "
                f"player_id={player_id}"
            )

        by_game_player_id[key] = row

    return (
        fieldnames,
        by_game_player_id,
    )


def prop_player_id(
    row: dict[str, str],
) -> str:
    for field in (
        "espn_player_id",
        "player_id",
        "athlete_id",
    ):
        value = normalize_id(
            row.get(field)
        )

        if value:
            return value

    return ""


def prop_player_name(
    row: dict[str, str],
) -> str:
    return str(
        row.get("player_name") or ""
    ).strip()


def find_roster_row(
    prop_row: dict[str, str],
    roster_by_id: dict[str, dict[str, str]],
) -> dict[str, str] | None:
    espn_id = prop_player_id(prop_row)

    if not espn_id:
        return None

    return roster_by_id.get(espn_id)


def find_depth_chart_row(
    prop_row: dict[str, str],
    depth_by_id: dict[str, dict[str, str]],
) -> dict[str, str] | None:
    espn_id = prop_player_id(prop_row)

    if not espn_id:
        return None

    return depth_by_id.get(espn_id)


def find_projection_row(
    prop_row: dict[str, str],
    espn_to_gsis: dict[str, str],
    projection_by_game_player_id: dict[
        tuple[str, str],
        dict[str, str],
    ],
) -> tuple[
    dict[str, str] | None,
    str,
    str,
]:
    game_id = normalize_id(
        prop_row.get("game_id")
    )

    espn_id = prop_player_id(
        prop_row
    )

    if not espn_id:
        return (
            None,
            "",
            "missing_espn_player_id",
        )

    gsis_id = espn_to_gsis.get(
        espn_id,
        "",
    )

    if not gsis_id:
        return (
            None,
            "",
            "espn_id_not_in_player_crosswalk",
        )

    if not game_id:
        return (
            None,
            gsis_id,
            "missing_game_id",
        )

    projection_row = (
        projection_by_game_player_id.get(
            (
                game_id,
                gsis_id,
            )
        )
    )

    if projection_row:
        return (
            projection_row,
            gsis_id,
            "game_id+gsis_player_id",
        )

    return (
        None,
        gsis_id,
        "gsis_player_id_not_in_projection_file",
    )


def discover_prop_files(
    category_dir: Path,
) -> list[Path]:
    if not category_dir.exists():
        return []

    return sorted(
        path
        for path in category_dir.glob("*.csv")
        if path.is_file()
        and not path.name.endswith(
            "_props_select.csv"
        )
    )


def union_headers(
    headers: list[list[str]],
) -> list[str]:
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
    projection_by_game_player_id: dict[
        tuple[str, str],
        dict[str, str],
    ],
    espn_to_gsis: dict[str, str],
    roster_by_id: dict[str, dict[str, str]],
    depth_by_id: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    source_dir = (
        props_root
        / category
    )

    source_files = discover_prop_files(
        source_dir
    )

    source_rows: list[
        dict[str, str]
    ] = []

    source_headers: list[
        list[str]
    ] = []

    for source_file in source_files:
        rows, fields = read_csv(
            source_file
        )

        source_rows.extend(rows)
        source_headers.append(fields)

    field_map = ACTUAL_FIELD_MAP.get(
        category,
        {},
    )

    discovered_prop_fields = [
        field
        for field in union_headers(
            source_headers
        )
        if field not in PROP_KEY_FIELDS
        and field not in field_map
    ]

    extra_actual_map = {
        field: (
            "actual_prop_"
            + re.sub(
                r"[^a-z0-9]+",
                "_",
                field.strip().lower(),
            ).strip("_")
        )
        for field in discovered_prop_fields
    }

    actual_map = {
        **field_map,
        **extra_actual_map,
    }

    desired_engine_fields = (
        COMMON_ENGINE_FIELDS
        + CATEGORY_ENGINE_FIELDS.get(
            category,
            [],
        )
    )

    engine_fields = [
        field
        for field in desired_engine_fields
        if field in projection_fields
    ]

    output_fields = [
        "game_date",
        "game_id",
        "player_name",
        "espn_player_id",
        "prop_engine_player_id",
        "projection_match_status",
        "projection_match_method",
    ]

    output_fields.extend(
        actual_map.values()
    )

    output_fields.extend(
        f"prop_engine_{field}"
        for field in engine_fields
    )

    output_rows: list[
        dict[str, object]
    ] = []

    unmatched_rows: list[
        dict[str, str]
    ] = []

    matched = 0

    for prop_row in source_rows:
        roster_row = find_roster_row(
            prop_row,
            roster_by_id,
        )

        depth_row = find_depth_chart_row(
            prop_row,
            depth_by_id,
        )

        (
            projection_row,
            gsis_id,
            match_method,
        ) = find_projection_row(
            prop_row,
            espn_to_gsis,
            projection_by_game_player_id,
        )

        if projection_row:
            matched += 1

        espn_id = prop_player_id(
            prop_row
        )

        canonical_name = ""

        if depth_row:
            canonical_name = str(
                depth_row.get("name") or ""
            ).strip()

        if (
            not canonical_name
            and roster_row
        ):
            canonical_name = str(
                roster_row.get("fullName")
                or roster_row.get("displayName")
                or ""
            ).strip()

        if not canonical_name:
            canonical_name = prop_player_name(
                prop_row
            )

        match_status = (
            "matched"
            if projection_row
            else "unmatched"
        )

        output_row: dict[
            str,
            object,
        ] = {
            "game_date": prop_row.get(
                "game_date",
                "",
            ),
            "game_id": normalize_id(
                prop_row.get("game_id")
            ),
            "player_name": canonical_name,
            "espn_player_id": espn_id,
            "prop_engine_player_id": gsis_id,
            "projection_match_status": match_status,
            "projection_match_method": match_method,
        }

        for (
            source_field,
            output_field,
        ) in actual_map.items():
            output_row[output_field] = (
                prop_row.get(
                    source_field,
                    "",
                )
            )

        for field in engine_fields:
            output_row[
                f"prop_engine_{field}"
            ] = (
                projection_row.get(
                    field,
                    "",
                )
                if projection_row
                else ""
            )

        output_rows.append(
            output_row
        )

        if not projection_row:
            unmatched_rows.append(
                {
                    "category": category,
                    "game_id": normalize_id(
                        prop_row.get(
                            "game_id"
                        )
                    ),
                    "player_name": canonical_name,
                    "espn_player_id": espn_id,
                    "gsis_player_id": gsis_id,
                    "match_failure": match_method,
                    "depth_chart_found": (
                        "yes"
                        if depth_row
                        else "no"
                    ),
                    "depth_chart_team": (
                        str(
                            depth_row.get(
                                "team"
                            )
                            or ""
                        ).strip()
                        if depth_row
                        else ""
                    ),
                    "depth_chart_name": (
                        str(
                            depth_row.get(
                                "name"
                            )
                            or ""
                        ).strip()
                        if depth_row
                        else ""
                    ),
                }
            )

    output_path = (
        props_root
        / "selections"
        / category
        / f"week_{week}_props_select.csv"
    )

    write_csv(
        output_path,
        output_rows,
        output_fields,
    )

    print(
        f"{category}: "
        f"rows={len(output_rows)} "
        f"projection_matches={matched} "
        f"unmatched={len(unmatched_rows)} "
        f"output={output_path}"
    )

    return unmatched_rows


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--season",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--week",
        type=int,
    )

    args = parser.parse_args()

    season = args.season
    week = resolve_week(
        season,
        args.week,
    )

    season_root = (
        OUTPUT_ROOT
        / str(season)
    )

    props_root = (
        season_root
        / f"week_{week}_props"
    )

    projection_path = (
        season_root
        / f"week_{week}_player_projections_wide.csv"
    )

    if not props_root.exists():
        raise FileNotFoundError(
            f"Missing weekly props folder: "
            f"{props_root}"
        )

    espn_to_gsis = (
        load_player_crosswalk()
    )

    roster_by_id = load_roster()

    depth_by_id = (
        load_depth_charts()
    )

    (
        projection_fields,
        projection_by_game_player_id,
    ) = load_projections(
        projection_path
    )

    all_unmatched: list[
        dict[str, str]
    ] = []

    for category in CATEGORIES:
        all_unmatched.extend(
            build_category(
                season=season,
                week=week,
                category=category,
                props_root=props_root,
                projection_fields=projection_fields,
                projection_by_game_player_id=projection_by_game_player_id,
                espn_to_gsis=espn_to_gsis,
                roster_by_id=roster_by_id,
                depth_by_id=depth_by_id,
            )
        )

    print()
    print("UNMATCHED PLAYERS")
    print("-----------------")

    if not all_unmatched:
        print("None")

    else:
        seen_category_records: set[
            tuple[str, str, str, str]
        ] = set()

        unique_players: dict[
            str,
            dict[str, str],
        ] = {}

        for row in sorted(
            all_unmatched,
            key=lambda r: (
                r["category"],
                r["game_id"],
                r["player_name"],
                r["espn_player_id"],
            ),
        ):
            record_key = (
                row["category"],
                row["game_id"],
                row["player_name"],
                row["espn_player_id"],
            )

            if record_key in seen_category_records:
                continue

            seen_category_records.add(
                record_key
            )

            player_key = (
                row["espn_player_id"]
                or normalize_name(
                    row["player_name"]
                )
            )

            unique_players.setdefault(
                player_key,
                row,
            )

            print(
                f'{row["category"]} | '
                f'game_id={row["game_id"]} | '
                f'player={row["player_name"]} | '
                f'espn_player_id={row["espn_player_id"]} | '
                f'gsis_player_id={row["gsis_player_id"]} | '
                f'match_failure={row["match_failure"]} | '
                f'depth_chart_found={row["depth_chart_found"]} | '
                f'depth_chart_team={row["depth_chart_team"]}'
            )

        print(
            f"Total unmatched category records: "
            f"{len(seen_category_records)}"
        )

        print(
            f"Total unique unmatched players: "
            f"{len(unique_players)}"
        )

        print()
        print("UNIQUE UNMATCHED PLAYERS")
        print("------------------------")

        for row in sorted(
            unique_players.values(),
            key=lambda r: (
                r["player_name"],
                r["espn_player_id"],
            ),
        ):
            print(
                f'player={row["player_name"]} | '
                f'espn_player_id={row["espn_player_id"]} | '
                f'gsis_player_id={row["gsis_player_id"]} | '
                f'match_failure={row["match_failure"]} | '
                f'depth_chart_found={row["depth_chart_found"]} | '
                f'depth_chart_team={row["depth_chart_team"]} | '
                f'depth_chart_name={row["depth_chart_name"]}'
            )


if __name__ == "__main__":
    main()
