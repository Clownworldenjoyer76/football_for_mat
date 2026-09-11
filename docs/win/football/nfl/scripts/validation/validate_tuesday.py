#!/usr/bin/env python3

import argparse
import csv
import re
import sys
from pathlib import Path

import pandas as pd

NFL_ROOT = Path("docs/win/football/nfl")


def fail(message):
    print(f"VALIDATION FAILED: {message}", file=sys.stderr)
    raise SystemExit(1)


def passed(message):
    print(f"PASS: {message}")


def warning(message):
    print(f"WARNING: {message}", file=sys.stderr)


TEAM_ABBR_ALIASES = {
    "WAS": "WSH",
    "LA": "LAR",
    "JAC": "JAX",
}


def normalize_team_abbr(value):
    text = str(value).strip().upper()
    return TEAM_ABBR_ALIASES.get(text, text)


def normalize_team_name(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().casefold())


def load_team_name_to_abbr():
    path = NFL_ROOT / "config/mapping/team_map.csv"

    if not path.is_file():
        fail(f"Missing file: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []

        missing = [
            column
            for column in ["canonical_team", "team_abbr"]
            if column not in columns
        ]

        if missing:
            fail(f"{path} is missing columns: {missing}")

        lookup = {}

        for row in reader:
            abbr = normalize_team_abbr(row.get("team_abbr", ""))
            canonical = str(row.get("canonical_team", "")).strip()

            if not abbr or not canonical:
                continue

            key = normalize_team_name(canonical)
            previous = lookup.get(key)

            if previous is not None and previous != abbr:
                fail(
                    f"{path} maps team {canonical!r} to multiple abbreviations: "
                    f"{previous}, {abbr}"
                )

            lookup[key] = abbr

    if not lookup:
        fail(f"{path} contains no usable canonical team mappings")

    return lookup


def schedule_game_identity(row, team_lookup):
    season_text = str(row.get("season", "")).strip()
    week_text = str(row.get("week", "")).strip()

    try:
        season = int(float(season_text))
        week = int(float(week_text))
    except ValueError:
        fail(
            "Schedule row has invalid season/week for "
            f"game_id={row.get('game_id', '')!r}: "
            f"season={season_text!r} week={week_text!r}"
        )

    away_name = str(row.get("away_team", "")).strip()
    home_name = str(row.get("home_team", "")).strip()

    away_key = normalize_team_name(away_name)
    home_key = normalize_team_name(home_name)

    if away_key not in team_lookup:
        fail(f"Could not map schedule away_team to abbreviation: {away_name!r}")

    if home_key not in team_lookup:
        fail(f"Could not map schedule home_team to abbreviation: {home_name!r}")

    return (
        season,
        week,
        team_lookup[away_key],
        team_lookup[home_key],
    )


def pbp_game_identity(game_id):
    text = str(game_id).strip()

    match = re.fullmatch(
        r"(\d{4})_(\d{1,2})_([A-Za-z0-9]+)_([A-Za-z0-9]+)",
        text,
    )

    if not match:
        fail(f"Unrecognized nflverse PBP game_id format: {text!r}")

    return (
        int(match.group(1)),
        int(match.group(2)),
        normalize_team_abbr(match.group(3)),
        normalize_team_abbr(match.group(4)),
    )


def read_csv(path, required_columns, *, allow_empty=False, unique_by=None):
    if not path.is_file():
        fail(f"Missing file: {path}")

    if path.stat().st_size == 0:
        fail(f"Zero-byte file: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        rows = list(reader)

    missing = [column for column in required_columns if column not in columns]

    if missing:
        fail(f"{path} is missing columns: {missing}")

    if not rows and not allow_empty:
        fail(f"{path} has a header but no data rows")

    if unique_by and rows:
        seen = set()
        duplicates = []

        for line_number, row in enumerate(rows, start=2):
            key = tuple(
                str(row.get(column, "")).strip()
                for column in unique_by
            )

            if not all(key):
                fail(
                    f"{path} line {line_number} has a blank unique-key value "
                    f"for columns {unique_by}: {key}"
                )

            if key in seen:
                duplicates.append(key)

            seen.add(key)

        if duplicates:
            fail(
                f"{path} contains duplicate keys for {unique_by}. "
                f"Examples: {duplicates[:5]}"
            )

    passed(f"{path} | rows={len(rows)}")
    return rows


def read_pbp(path, required_columns, *, require_rows):
    if not path.is_file():
        if require_rows:
            fail(f"Missing file: {path}")

        passed(f"{path} not required before completed games exist")
        return pd.DataFrame()

    if path.stat().st_size == 0:
        if require_rows:
            fail(f"Zero-byte file: {path}")

        passed(f"{path} is empty and allowed before completed games exist")
        return pd.DataFrame()

    try:
        frame = pd.read_csv(
            path,
            compression="gzip",
            low_memory=False,
        )

    except pd.errors.EmptyDataError:
        if require_rows:
            fail(f"{path} contains no PBP data")

        passed(
            f"{path} contains no PBP data and is allowed "
            "before completed games exist"
        )
        return pd.DataFrame()

    except Exception as exc:
        fail(f"Could not read compressed PBP file {path}: {exc}")

    if frame.empty and not require_rows:
        passed(
            f"{path} has no PBP rows and is allowed "
            "before completed games exist"
        )
        return frame

    missing = [
        column
        for column in required_columns
        if column not in frame.columns
    ]

    if missing:
        fail(f"{path} is missing columns: {missing}")

    if require_rows and frame.empty:
        fail(
            f"{path} contains no PBP rows even though "
            "completed games exist"
        )

    if not frame.empty:
        keys = frame[
            ["game_id", "play_id"]
        ].copy()

        keys = keys.dropna(
            subset=["game_id", "play_id"]
        )

        if keys.duplicated().any():
            examples = (
                keys[
                    keys.duplicated(keep=False)
                ]
                .head(5)
                .to_dict("records")
            )

            fail(
                f"{path} has duplicate game_id/play_id rows: "
                f"{examples}"
            )

    passed(
        f"{path} | rows={len(frame)} "
        f"columns={len(frame.columns)}"
    )

    return frame


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--season",
        required=True,
        type=int,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    season = args.season

    print(
        f"Validating NFL Tuesday workflow outputs "
        f"for season {season}"
    )

    schedule_path = (
        NFL_ROOT
        / f"00_intake/schedule/{season}_schedule.csv"
    )

    schedule_rows = read_csv(
        schedule_path,
        [
            "season",
            "season_type",
            "week",
            "game_id",
            "game_date",
            "game_time",
            "away_team",
            "home_team",
            "stadium",
            "game_timezone",
        ],
        unique_by=["game_id"],
    )

    schedule_game_ids = {
        str(row["game_id"]).strip()
        for row in schedule_rows
    }

    team_lookup = load_team_name_to_abbr()

    schedule_identity_by_game_id = {
        str(row["game_id"]).strip():
            schedule_game_identity(
                row,
                team_lookup,
            )
        for row in schedule_rows
    }

    schedule_identities = set(
        schedule_identity_by_game_id.values()
    )

    result_files = sorted(
        (
            NFL_ROOT
            / "04_final_results/results"
        ).glob(
            f"{season}_*.csv"
        )
    )

    if not result_files:
        warning(
            f"No final-score files found for season {season}. "
            "Continuing because final-score data may not yet be available."
        )

    result_rows = []
    result_game_ids = set()
    completed_game_ids = set()

    for path in result_files:
        rows = read_csv(
            path,
            [
                "season",
                "season_type",
                "week",
                "game_id",
                "game_date",
                "game_time",
                "away_team",
                "home_team",
                "away_score",
                "home_score",
                "status",
            ],
            unique_by=["game_id"],
        )

        for row in rows:
            game_id = str(
                row["game_id"]
            ).strip()

            if game_id in result_game_ids:
                fail(
                    "Duplicate game_id across "
                    f"final-score files: {game_id}"
                )

            result_game_ids.add(
                game_id
            )

            result_rows.append(
                row
            )

            status = str(
                row.get(
                    "status",
                    "",
                )
            ).strip().lower()

            if (
                "final" in status
                or "completed" in status
            ):
                completed_game_ids.add(
                    game_id
                )

                if (
                    str(
                        row.get(
                            "away_score",
                            "",
                        )
                    ).strip()
                    == ""
                ):
                    fail(
                        f"Completed game {game_id} "
                        "has a blank away_score"
                    )

                if (
                    str(
                        row.get(
                            "home_score",
                            "",
                        )
                    ).strip()
                    == ""
                ):
                    fail(
                        f"Completed game {game_id} "
                        "has a blank home_score"
                    )

    # Final-score output is produced by week, so it is not expected to cover
    # the entire season schedule. Every final-score row that does exist must,
    # however, belong to the canonical season schedule.
    extra_result_games = (
        result_game_ids
        - schedule_game_ids
    )

    if extra_result_games:
        fail(
            "Final-score files contain game IDs absent from "
            "the season schedule. "
            f"Extra={sorted(extra_result_games)[:5]}"
        )

    completed_games_exist = bool(
        completed_game_ids
    )

    print(
        f"Completed games detected: "
        f"{len(completed_game_ids)}"
    )

    pbp_path = (
        NFL_ROOT
        / f"00_intake/pbp/{season}_pbp.csv.gz"
    )

    pbp = read_pbp(
        pbp_path,
        [
            "season",
            "week",
            "game_id",
            "play_id",
            "posteam",
            "defteam",
            "play_type",
            "yards_gained",
            "epa",
            "success",
            "down",
            "yardline_100",
            "posteam_score",
            "posteam_score_post",
            "touchdown",
            "pass_touchdown",
            "rush_touchdown",
            "third_down_converted",
            "third_down_failed",
            "passer_player_id",
            "passer_player_name",
            "qb_dropback",
            "pass_attempt",
            "qb_epa",
            "cpoe",
            "air_yards",
            "sack",
            "interception",
            "fumbled_1_player_id",
        ],
        require_rows=completed_games_exist,
    )

    if not pbp.empty:
        pbp_game_ids = {
            str(value).strip()
            for value in (
                pbp["game_id"]
                .dropna()
                .astype(str)
                .tolist()
            )
        }

        pbp_identity_by_game_id = {
            game_id:
                pbp_game_identity(
                    game_id
                )
            for game_id in pbp_game_ids
        }

        pbp_identities = set(
            pbp_identity_by_game_id.values()
        )

        unknown_pbp_games = sorted(
            game_id
            for game_id, identity
            in pbp_identity_by_game_id.items()
            if identity not in schedule_identities
        )

        if unknown_pbp_games:
            fail(
                f"{pbp_path} contains games absent "
                f"from the schedule: "
                f"{unknown_pbp_games[:5]}"
            )

        completed_identities = {
            schedule_identity_by_game_id[
                game_id
            ]
            for game_id
            in completed_game_ids
            if game_id
            in schedule_identity_by_game_id
        }

        missing_completed_identities = (
            completed_identities
            - pbp_identities
        )

        if missing_completed_identities:
            missing_completed_schedule_ids = sorted(
                game_id
                for game_id, identity
                in schedule_identity_by_game_id.items()
                if identity
                in missing_completed_identities
            )

            fail(
                f"{pbp_path} is missing completed games: "
                f"{missing_completed_schedule_ids[:5]}"
            )

    team_stats_path = (
        NFL_ROOT
        / f"00_intake/team_stats/{season}_team_stats.csv"
    )

    read_csv(
        team_stats_path,
        [
            "season",
            "week",
            "team",
            "off_epa_per_play",
            "def_epa_per_play",
            "off_success_rate",
            "def_success_rate",
            "yards_per_play",
            "yards_per_play_allowed",
            "points_per_drive",
            "points_per_drive_allowed",
            "red_zone_td_rate",
            "red_zone_td_rate_allowed",
            "early_down_epa",
            "third_down_conversion_rate",
        ],
        allow_empty=not completed_games_exist,
        unique_by=[
            "season",
            "week",
            "team",
        ],
    )

    qb_stats_path = (
        NFL_ROOT
        / f"00_intake/qb/{season}_qb_stats.csv"
    )

    if completed_games_exist:
        read_csv(
            qb_stats_path,
            [
                "season",
                "week",
                "team",
                "player_id",
                "qb_name",
                "dropbacks",
                "epa_per_play",
                "cpoe",
                "air_yards",
                "sack_rate",
                "interception_rate",
                "fumble_rate",
            ],
            unique_by=[
                "season",
                "week",
                "team",
                "player_id",
            ],
        )

    elif qb_stats_path.exists():
        read_csv(
            qb_stats_path,
            [
                "season",
                "week",
                "team",
                "player_id",
                "qb_name",
                "dropbacks",
            ],
            allow_empty=True,
            unique_by=[
                "season",
                "week",
                "team",
                "player_id",
            ],
        )

    else:
        passed(
            f"{qb_stats_path} not required "
            "before completed games exist"
        )

    league_master_rows = read_csv(
        NFL_ROOT
        / "data/master/league_master.csv",
        [
            "team_id",
            "team_abbr",
            "conference",
            "conference_abbr",
            "division",
            "division_abbr",
            "season",
        ],
        unique_by=["team_id"],
    )

    if len(league_master_rows) != 32:
        fail(
            "league_master.csv must contain exactly "
            "32 NFL teams; "
            f"found {len(league_master_rows)}"
        )

    read_csv(
        NFL_ROOT
        / "data/master/league_standings.csv",
        [
            "team_id",
            "team_abbr",
            "conference",
            "division",
            "standings_type",
            "stat_name",
            "stat_value",
            "season",
        ],
    )

    coaches_rows = read_csv(
        NFL_ROOT
        / "data/master/coaches_master.csv",
        [
            "name",
            "team",
            "team_id",
            "experience",
            "id",
            "uid",
        ],
        unique_by=["team_id"],
    )

    if len(coaches_rows) != 32:
        fail(
            "coaches_master.csv must contain exactly "
            "32 head coaches; "
            f"found {len(coaches_rows)}"
        )

    qbr_files = sorted(
        (
            NFL_ROOT
            / f"data/qb_data/qbr_data/{season}"
        ).glob("*.csv")
    )

    if completed_games_exist and not qbr_files:
        fail(
            f"No QBR files found for season {season}"
        )

    for path in qbr_files:
        read_csv(
            path,
            [
                "season",
                "week",
                "athlete_id",
                "team_id",
            ],
            unique_by=[
                "season",
                "week",
                "athlete_id",
                "team_id",
            ],
        )

    fpi_rows = read_csv(
        NFL_ROOT
        / (
            "data/team_power_index/"
            f"team_power_index_{season}.csv"
        ),
        [
            "season",
            "team_id",
            "lastUpdated",
        ],
        unique_by=["team_id"],
    )

    if len(fpi_rows) != 32:
        fail(
            f"team_power_index_{season}.csv "
            "must contain exactly 32 teams; "
            f"found {len(fpi_rows)}"
        )

    leaders_path = (
        NFL_ROOT
        / (
            "data/league_leaders/"
            f"league_leaders_{season}.csv"
        )
    )

    if completed_games_exist:
        read_csv(
            leaders_path,
            [
                "season",
                "category",
                "rank",
                "athlete_id",
                "team_id",
                "value",
                "displayValue",
            ],
            unique_by=[
                "season",
                "category",
                "rank",
            ],
        )

    elif leaders_path.exists():
        read_csv(
            leaders_path,
            [
                "season",
                "category",
                "rank",
                "athlete_id",
                "team_id",
                "value",
                "displayValue",
            ],
            allow_empty=True,
            unique_by=[
                "season",
                "category",
                "rank",
            ],
        )

    else:
        passed(
            f"{leaders_path} not required "
            "before completed games exist"
        )

    read_csv(
        NFL_ROOT
        / (
            "data/market_futures/"
            f"market_futures_{season}.csv"
        ),
        [
            "season",
            "future_id",
            "future_name",
            "provider_id",
            "provider_name",
            "athlete_id",
            "team_id",
            "value",
        ],
    )

    weekly_files = sorted(
        (
            NFL_ROOT
            / "00_intake/schedule/weekly"
        ).glob(
            "week_*_NFL_weekly_schedule.csv"
        )
    )

    current_weekly_files = 0

    for schedule_week_path in weekly_files:
        match = re.fullmatch(
            r"week_(\d+)_NFL_weekly_schedule\.csv",
            schedule_week_path.name,
        )

        if not match:
            continue

        week_rows = read_csv(
            schedule_week_path,
            [
                "season",
                "week",
                "game_id",
                "away_team",
                "home_team",
                "neutral_site",
            ],
            unique_by=["game_id"],
        )

        file_seasons = {
            str(row["season"]).strip()
            for row in week_rows
        }

        if file_seasons != {
            str(season)
        }:
            continue

        current_weekly_files += 1

        week = int(
            match.group(1)
        )

        schedule_week_ids = {
            str(row["game_id"]).strip()
            for row in week_rows
        }

        travel_path = (
            NFL_ROOT
            / (
                f"data/travel/"
                f"{season}_week_{week}_travel.csv"
            )
        )

        travel_rows = read_csv(
            travel_path,
            [
                "game_id",
                "away_team",
                "home_team",
                "away_lat",
                "away_lon",
                "home_lat",
                "home_lon",
                "miles_traveled",
                "time_zones_crossed",
                "east_to_west",
                "west_to_east",
                "international_flag",
                "neutral_site_flag",
            ],
            unique_by=["game_id"],
        )

        travel_ids = {
            str(row["game_id"]).strip()
            for row in travel_rows
        }

        if travel_ids != schedule_week_ids:
            fail(
                f"{travel_path} game IDs do not "
                f"exactly match week {week} "
                "schedule game IDs"
            )

    if current_weekly_files == 0:
        fail(
            f"No weekly schedule files found "
            f"for season {season}"
        )

    print("TUESDAY VALIDATION PASSED")


if __name__ == "__main__":
    main()