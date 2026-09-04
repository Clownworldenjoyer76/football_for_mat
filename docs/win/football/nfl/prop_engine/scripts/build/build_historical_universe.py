#!/usr/bin/env python3
"""
Build the historical player-game universe before outcomes are joined.

READS:
    docs/win/football/nfl/data/historic_data/games/games_2010_2025.csv
    docs/win/football/nfl/data/historic_data/weekly_rosters/roster_weekly_{season}.parquet
    docs/win/football/nfl/data/historic_data/depth_charts/depth_charts_{season}.parquet
    docs/win/football/nfl/data/historic_data/snap_counts/snap_counts_{season}.parquet
    docs/win/football/nfl/data/historic_data/participation/pbp_participation_{season}.parquet
    docs/win/football/nfl/prop_engine/data/identity/player_crosswalk.parquet

WRITES:
    docs/win/football/nfl/prop_engine/data/historical/universe/player_game_universe.parquet

GRAIN:
    season + week + game_id + player_id
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable
import math
import re
import sys

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent

if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import common


OUTPUT_COLUMNS = [
    "season",
    "season_type",
    "week",
    "game_id",
    "gameday",
    "kickoff_timestamp",
    "player_id",
    "player_name",
    "team",
    "opponent",
    "position",
    "position_group",
    "home_flag",
    "away_flag",
    "roster_flag",
    "depth_present_flag",
    "depth_rank",
    "depth_slot",
    "depth_starter_flag",
    "prior_offense_snap_pct",
    "prior_defense_snap_pct",
    "prior_offense_participation",
    "prior_defense_participation",
    "played_game_flag",
    "universe_source",
]


GSIS_PATTERN = re.compile(
    r"00-\d{7}"
)

GAME_ID_PATTERN = re.compile(
    r"^(\d{4})_(\d{1,2})_"
    r"([A-Za-z0-9]+)_([A-Za-z0-9]+)$"
)


SOURCE_ORDER = {
    "roster": 0,
    "depth": 1,
    "snap": 2,
    "participation": 3,
}


OFFENSE_POSITIONS = {
    "QB",
    "RB",
    "HB",
    "FB",
    "WR",
    "TE",
    "C",
    "G",
    "OG",
    "LG",
    "RG",
    "T",
    "OT",
    "LT",
    "RT",
    "OL",
}

DEFENSE_POSITIONS = {
    "DL",
    "DE",
    "LDE",
    "RDE",
    "DT",
    "LDT",
    "RDT",
    "NT",
    "EDGE",
    "LB",
    "ILB",
    "OLB",
    "MLB",
    "WLB",
    "SLB",
    "DB",
    "CB",
    "LCB",
    "RCB",
    "NB",
    "S",
    "FS",
    "SS",
}

SPECIAL_TEAMS_POSITIONS = {
    "K",
    "PK",
    "P",
    "LS",
    "H",
    "KR",
    "PR",
}


def clean(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    text = str(value).strip()

    if text.casefold() in {
        "",
        "nan",
        "none",
        "null",
        "<na>",
        "nat",
    }:
        return ""

    return text


def choose_column(
    df: pd.DataFrame,
    aliases: Iterable[str],
    *,
    label: str,
    required: bool = False,
) -> str | None:
    choices = list(aliases)

    for column in choices:
        if column in df.columns:
            return column

    if required:
        raise ValueError(
            f"{label}: none of the required aliases "
            f"are present: {choices}"
        )

    return None


def parse_int(
    value: Any,
) -> int | None:
    text = clean(value)

    if not text:
        return None

    try:
        number = float(text)
    except ValueError:
        return None

    if (
        not math.isfinite(number)
        or abs(number - round(number)) > 1e-9
    ):
        return None

    return int(round(number))


def normalize_position(
    value: Any,
) -> str:
    raw = (
        clean(value)
        .upper()
        .replace(" ", "")
    )

    aliases = {
        "CORNERBACK": "CB",
        "SAFETY": "S",
        "DEFENSIVEBACK": "DB",
        "DEFENSIVEEND": "DE",
        "DEFENSIVETACKLE": "DT",
        "LINEBACKER": "LB",
        "OFFENSIVELINE": "OL",
        "OFFENSIVETACKLE": "OT",
        "OFFENSIVEGUARD": "G",
        "RUNNINGBACK": "RB",
        "WIDERECEIVER": "WR",
        "TIGHTEND": "TE",
        "QUARTERBACK": "QB",
        "PLACEKICKER": "K",
    }

    return aliases.get(
        raw,
        raw,
    )


def infer_position_group(
    position: Any,
) -> str:
    position = normalize_position(
        position
    )

    if position in OFFENSE_POSITIONS:
        return "OFF"

    if position in DEFENSE_POSITIONS:
        return "DEF"

    if position in SPECIAL_TEAMS_POSITIONS:
        return "ST"

    return ""


def normalize_percentage(
    value: Any,
) -> float:
    text = clean(value).replace(
        "%",
        "",
    )

    if not text:
        return 0.0

    try:
        number = float(text)
    except ValueError:
        return 0.0

    if (
        not math.isfinite(number)
        or number < 0
    ):
        return 0.0

    if number > 1.5:
        number /= 100.0

    return max(
        0.0,
        number,
    )


def positive_numeric(
    value: Any,
) -> bool:
    text = clean(value)

    if not text:
        return False

    try:
        return float(text) > 0
    except ValueError:
        return False


class IdentityResolver:
    def __init__(
        self,
        df: pd.DataFrame,
    ) -> None:
        common.require_columns(
            df,
            [
                "player_id",
                "gsis_id",
                "espn_id",
                "pfr_id",
                "display_name",
                "normalized_name",
                "position",
                "position_group",
                "resolution_status",
            ],
            "player crosswalk",
        )

        self.by_gsis: dict[
            str,
            dict[str, str],
        ] = {}

        espn_candidates: dict[
            str,
            set[str],
        ] = defaultdict(set)

        pfr_candidates: dict[
            str,
            set[str],
        ] = defaultdict(set)

        name_candidates: dict[
            str,
            set[str],
        ] = defaultdict(set)

        for row in df.to_dict(
            orient="records"
        ):
            player_id = (
                common.normalize_player_id(
                    row["player_id"]
                )
            )

            gsis_id = (
                common.normalize_player_id(
                    row["gsis_id"]
                )
            )

            if not player_id or not gsis_id:
                continue

            record = {
                "player_id": player_id,
                "gsis_id": gsis_id,
                "espn_id": (
                    common.normalize_player_id(
                        row["espn_id"]
                    )
                ),
                "pfr_id": (
                    common.normalize_player_id(
                        row["pfr_id"]
                    )
                ),
                "display_name": clean(
                    row["display_name"]
                ),
                "normalized_name": (
                    clean(
                        row[
                            "normalized_name"
                        ]
                    )
                    or common.normalize_name(
                        row["display_name"]
                    )
                ),
                "position": (
                    normalize_position(
                        row["position"]
                    )
                ),
                "position_group": (
                    clean(
                        row[
                            "position_group"
                        ]
                    ).upper()
                ),
            }

            self.by_gsis[
                gsis_id
            ] = record

            if record["espn_id"]:
                espn_candidates[
                    record["espn_id"]
                ].add(gsis_id)

            if record["pfr_id"]:
                pfr_candidates[
                    record["pfr_id"]
                ].add(gsis_id)

            if record[
                "normalized_name"
            ]:
                name_candidates[
                    record[
                        "normalized_name"
                    ]
                ].add(gsis_id)

        self.by_espn = {
            alias: next(iter(ids))
            for alias, ids
            in espn_candidates.items()
            if len(ids) == 1
        }

        self.by_pfr = {
            alias: next(iter(ids))
            for alias, ids
            in pfr_candidates.items()
            if len(ids) == 1
        }

        self.by_name = {
            alias: next(iter(ids))
            for alias, ids
            in name_candidates.items()
            if len(ids) == 1
        }

    def get(
        self,
        player_id: Any,
    ) -> dict[str, str] | None:
        return self.by_gsis.get(
            common.normalize_player_id(
                player_id
            )
        )

    def resolve(
        self,
        *,
        gsis_id: Any = "",
        espn_id: Any = "",
        pfr_id: Any = "",
        name: Any = "",
    ) -> dict[str, str] | None:
        gsis = (
            common.normalize_player_id(
                gsis_id
            )
        )

        if (
            gsis
            and gsis in self.by_gsis
        ):
            return self.by_gsis[
                gsis
            ]

        espn = (
            common.normalize_player_id(
                espn_id
            )
        )

        if espn:
            resolved = self.by_espn.get(
                espn
            )

            if resolved:
                return self.by_gsis[
                    resolved
                ]

        pfr = (
            common.normalize_player_id(
                pfr_id
            )
        )

        if pfr:
            resolved = self.by_pfr.get(
                pfr
            )

            if resolved:
                return self.by_gsis[
                    resolved
                ]

        normalized = (
            common.normalize_name(
                name
            )
        )

        if normalized:
            resolved = self.by_name.get(
                normalized
            )

            if resolved:
                return self.by_gsis[
                    resolved
                ]

        return None


def parse_nflverse_game_id(
    value: Any,
) -> tuple[
    int,
    int,
    str,
    str,
] | None:
    match = GAME_ID_PATTERN.match(
        clean(value)
    )

    if not match:
        return None

    return (
        int(match.group(1)),
        int(match.group(2)),
        common.normalize_team(
            match.group(3)
        ),
        common.normalize_team(
            match.group(4)
        ),
    )


def extract_gsis_ids(
    value: Any,
) -> set[str]:
    return set(
        GSIS_PATTERN.findall(
            clean(value)
        )
    )


def conservative_depth_cutoff(
    game: dict[str, Any],
) -> pd.Timestamp:
    """
    Timestamp depth snapshots are restricted to strictly
    before midnight UTC on the target gameday.

    This matches the existing repository's conservative
    leakage protection when a timezone-safe kickoff
    timestamp is unavailable.
    """
    return pd.Timestamp(
        clean(
            game["gameday"]
        ),
        tz="UTC",
    )


def load_games(
    path: Path,
    start_season: int,
    end_season: int,
) -> tuple[
    list[dict[str, Any]],
    dict[
        tuple[int, int, str],
        dict[str, Any],
    ],
]:
    allowed = {
        "game_id",
        "season",
        "game_type",
        "season_type",
        "week",
        "gameday",
        "gametime",
        "away_team",
        "home_team",
    }

    # Deliberately do not ingest the betting/market
    # columns that also exist in this historical file.
    games = pd.read_csv(
        path,
        usecols=lambda column: (
            column in allowed
        ),
        encoding="utf-8-sig",
        low_memory=False,
    )

    common.require_columns(
        games,
        [
            "game_id",
            "season",
            "week",
            "gameday",
            "away_team",
            "home_team",
        ],
        str(path),
    )

    type_col = choose_column(
        games,
        [
            "game_type",
            "season_type",
        ],
        label=str(path),
        required=True,
    )

    rows: list[
        dict[str, Any]
    ] = []

    by_team_week: dict[
        tuple[int, int, str],
        dict[str, Any],
    ] = {}

    for _, row in games.iterrows():
        season = parse_int(
            row["season"]
        )

        week = parse_int(
            row["week"]
        )

        if (
            season is None
            or week is None
        ):
            continue

        if not (
            start_season
            <= season
            <= end_season
        ):
            continue

        if (
            clean(
                row[type_col]
            ).upper()
            != "REG"
        ):
            continue

        game_id = clean(
            row["game_id"]
        )

        if not game_id:
            raise ValueError(
                f"{path}: blank game_id "
                f"season={season} "
                f"week={week}"
            )

        # Validate the ID but preserve the
        # source representation for joins.
        common.parse_game_id(
            game_id
        )

        home_team = (
            common.normalize_team(
                row["home_team"]
            )
        )

        away_team = (
            common.normalize_team(
                row["away_team"]
            )
        )

        if (
            not home_team
            or not away_team
            or home_team == away_team
        ):
            raise ValueError(
                f"{path}: invalid teams "
                f"for {game_id}: "
                f"{away_team} at "
                f"{home_team}"
            )

        kickoff = (
            common.kickoff_timestamp(
                row
            )
        )

        game = {
            "season": season,
            "season_type": "REG",
            "week": week,
            "game_id": game_id,
            "gameday": clean(
                row["gameday"]
            ),
            "kickoff_timestamp": (
                kickoff
            ),
            "home_team": home_team,
            "away_team": away_team,
        }

        rows.append(
            game
        )

        for team in (
            home_team,
            away_team,
        ):
            key = (
                season,
                week,
                team,
            )

            if key in by_team_week:
                raise ValueError(
                    "Regular-season team/week "
                    "maps to multiple games: "
                    f"{key}"
                )

            by_team_week[
                key
            ] = game

    if not rows:
        raise RuntimeError(
            "No regular-season games "
            f"found for {start_season}-"
            f"{end_season}."
        )

    return (
        rows,
        by_team_week,
    )


def default_record(
    game: dict[str, Any],
    player_id: str,
    team: str,
    identity: dict[str, str]
    | None,
) -> dict[str, Any]:
    opponent = (
        game["away_team"]
        if team
        == game["home_team"]
        else game["home_team"]
    )

    name = (
        identity[
            "display_name"
        ]
        if identity
        else ""
    )

    position = (
        identity["position"]
        if identity
        else ""
    )

    group = (
        identity[
            "position_group"
        ]
        if identity
        else ""
    )

    if not group:
        group = (
            infer_position_group(
                position
            )
        )

    return {
        "season": (
            game["season"]
        ),
        "season_type": (
            game["season_type"]
        ),
        "week": game["week"],
        "game_id": (
            game["game_id"]
        ),
        "gameday": (
            game["gameday"]
        ),
        "kickoff_timestamp": (
            game[
                "kickoff_timestamp"
            ]
        ),
        "player_id": player_id,
        "player_name": name,
        "team": team,
        "opponent": opponent,
        "position": position,
        "position_group": group,
        "home_flag": int(
            team
            == game["home_team"]
        ),
        "away_flag": int(
            team
            == game["away_team"]
        ),
        "roster_flag": 0,
        "depth_present_flag": 0,
        "depth_rank": pd.NA,
        "depth_slot": "",
        "depth_starter_flag": 0,
        "prior_offense_snap_pct": (
            float("nan")
        ),
        "prior_defense_snap_pct": (
            float("nan")
        ),
        (
            "prior_offense_"
            "participation"
        ): float("nan"),
        (
            "prior_defense_"
            "participation"
        ): float("nan"),
        "played_game_flag": 0,
        "_sources": set(),
    }


def upsert_candidate(
    rows: dict[
        tuple[
            int,
            int,
            str,
            str,
        ],
        dict[str, Any],
    ],
    *,
    game: dict[str, Any],
    player_id: str,
    team: str,
    identity: dict[str, str]
    | None,
    source: str,
    player_name: str = "",
    position: str = "",
    position_group: str = "",
) -> dict[str, Any]:
    if team not in {
        game["home_team"],
        game["away_team"],
    }:
        raise ValueError(
            f"Attempted to add "
            f"{player_id} from "
            f"{team} to "
            f"{game['game_id']}."
        )

    key = (
        game["season"],
        game["week"],
        game["game_id"],
        player_id,
    )

    record = rows.get(
        key
    )

    if record is None:
        record = default_record(
            game,
            player_id,
            team,
            identity,
        )

        rows[key] = record

    elif record["team"] != team:
        raise ValueError(
            "Player resolves to both "
            "teams in same game: "
            f"game={game['game_id']} "
            f"player={player_id} "
            f"teams="
            f"{record['team']},"
            f"{team}"
        )

    record[
        "_sources"
    ].add(source)

    if (
        player_name
        and (
            source == "roster"
            or not record[
                "player_name"
            ]
        )
    ):
        record[
            "player_name"
        ] = clean(
            player_name
        )

    normalized_position = (
        normalize_position(
            position
        )
    )

    if (
        normalized_position
        and (
            source == "roster"
            or not record[
                "position"
            ]
        )
    ):
        record[
            "position"
        ] = normalized_position

    group = (
        clean(
            position_group
        ).upper()
    )

    if (
        group
        and (
            source == "roster"
            or not record[
                "position_group"
            ]
        )
    ):
        record[
            "position_group"
        ] = group

    if not record[
        "position_group"
    ]:
        record[
            "position_group"
        ] = infer_position_group(
            record["position"]
        )

    return record


def add_roster_candidates(
    rows,
    *,
    roster: pd.DataFrame,
    resolver: IdentityResolver,
    games_by_team_week,
    season: int,
    label: str,
) -> None:
    team_col = choose_column(
        roster,
        ["team", "club_code"],
        label=label,
        required=True,
    )

    week_col = choose_column(
        roster,
        ["week"],
        label=label,
        required=True,
    )

    gsis_col = choose_column(
        roster,
        ["gsis_id"],
        label=label,
    )

    espn_col = choose_column(
        roster,
        ["espn_id"],
        label=label,
    )

    pfr_col = choose_column(
        roster,
        [
            "pfr_id",
            "pfr_player_id",
        ],
        label=label,
    )

    name_col = choose_column(
        roster,
        [
            "full_name",
            "display_name",
            "player_name",
        ],
        label=label,
    )

    position_col = choose_column(
        roster,
        [
            "position",
            "depth_chart_position",
            "ngs_position",
        ],
        label=label,
    )

    group_col = choose_column(
        roster,
        ["position_group"],
        label=label,
    )

    for row in roster.to_dict(
        orient="records"
    ):
        team = (
            common.normalize_team(
                row[team_col]
            )
        )

        week = parse_int(
            row[week_col]
        )

        if (
            not team
            or week is None
        ):
            continue

        game = (
            games_by_team_week.get(
                (
                    season,
                    week,
                    team,
                )
            )
        )

        if game is None:
            continue

        raw_gsis = (
            common.normalize_player_id(
                row[gsis_col]
            )
            if gsis_col
            else ""
        )

        identity = resolver.resolve(
            gsis_id=raw_gsis,
            espn_id=(
                row[espn_col]
                if espn_col
                else ""
            ),
            pfr_id=(
                row[pfr_col]
                if pfr_col
                else ""
            ),
            name=(
                row[name_col]
                if name_col
                else ""
            ),
        )

        # Native GSIS on a weekly
        # roster remains authoritative.
        player_id = raw_gsis

        if (
            not player_id
            and identity
            is not None
        ):
            player_id = (
                identity[
                    "player_id"
                ]
            )

        if not player_id:
            continue

        if identity is None:
            identity = (
                resolver.get(
                    player_id
                )
            )

        record = upsert_candidate(
            rows,
            game=game,
            player_id=player_id,
            team=team,
            identity=identity,
            source="roster",
            player_name=(
                row[name_col]
                if name_col
                else ""
            ),
            position=(
                row[position_col]
                if position_col
                else ""
            ),
            position_group=(
                row[group_col]
                if group_col
                else ""
            ),
        )

        record[
            "roster_flag"
        ] = 1


def merge_depth_entry(
    snapshot,
    *,
    player_id: str,
    player_name: str,
    position: str,
    rank: int,
    slot: str,
) -> None:
    candidate = {
        "player_id": player_id,
        "player_name": (
            clean(player_name)
        ),
        "position": (
            normalize_position(
                position
            )
        ),
        "rank": rank,
        "slot": (
            clean(slot).upper()
        ),
        "starter": int(
            rank == 1
        ),
    }

    existing = snapshot.get(
        player_id
    )

    if existing is None:
        snapshot[
            player_id
        ] = candidate
        return

    starter = max(
        existing["starter"],
        candidate["starter"],
    )

    if (
        candidate["rank"],
        candidate["slot"],
    ) < (
        existing["rank"],
        existing["slot"],
    ):
        snapshot[
            player_id
        ] = candidate

    snapshot[
        player_id
    ]["starter"] = starter


def build_depth_provider(
    depth: pd.DataFrame,
    resolver: IdentityResolver,
    *,
    label: str,
):
    has_week = (
        "week" in depth.columns
        and pd.to_numeric(
            depth["week"],
            errors="coerce",
        ).notna().any()
    )

    weekly = defaultdict(
        dict
    )

    timestamp_rows = (
        defaultdict(dict)
    )

    team_col = choose_column(
        depth,
        ["club_code", "team"],
        label=label,
        required=True,
    )

    rank_col = choose_column(
        depth,
        [
            "depth_team",
            "pos_rank",
        ],
        label=label,
        required=True,
    )

    position_col = choose_column(
        depth,
        [
            "position",
            "pos_abb",
        ],
        label=label,
    )

    slot_col = choose_column(
        depth,
        [
            "depth_position",
            "pos_slot",
            "position",
        ],
        label=label,
        required=True,
    )

    group_col = choose_column(
        depth,
        [
            "formation",
            "pos_grp",
        ],
        label=label,
    )

    gsis_col = choose_column(
        depth,
        ["gsis_id"],
        label=label,
    )

    espn_col = choose_column(
        depth,
        ["espn_id"],
        label=label,
    )

    name_col = choose_column(
        depth,
        [
            "full_name",
            "player_name",
        ],
        label=label,
    )

    if has_week:
        week_col = choose_column(
            depth,
            ["week"],
            label=label,
            required=True,
        )

        timestamp_col = None

    else:
        week_col = None

        timestamp_col = (
            choose_column(
                depth,
                ["dt", "timestamp"],
                label=label,
                required=True,
            )
        )

    for row in depth.to_dict(
        orient="records"
    ):
        team = (
            common.normalize_team(
                row[team_col]
            )
        )

        rank = parse_int(
            row[rank_col]
        )

        if (
            not team
            or rank is None
            or rank < 1
        ):
            continue

        identity = resolver.resolve(
            gsis_id=(
                row[gsis_col]
                if gsis_col
                else ""
            ),
            espn_id=(
                row[espn_col]
                if espn_col
                else ""
            ),
            name=(
                row[name_col]
                if name_col
                else ""
            ),
        )

        # Depth-only players may
        # create a universe row only
        # if canonical identity resolves.
        if identity is None:
            continue

        player_id = (
            identity[
                "player_id"
            ]
        )

        player_name = (
            clean(
                row[name_col]
            )
            if name_col
            else identity[
                "display_name"
            ]
        )

        position = (
            normalize_position(
                row[
                    position_col
                ]
            )
            if position_col
            else identity[
                "position"
            ]
        )

        slot_value = (
            clean(
                row[slot_col]
            ).upper()
        )

        group_value = (
            clean(
                row[group_col]
            ).upper()
            if group_col
            else ""
        )

        slot = (
            f"{group_value}|"
            f"{slot_value}"
            if group_value
            else slot_value
        )

        if has_week:
            week = parse_int(
                row[week_col]
            )

            if week is None:
                continue

            snapshot = weekly[
                (
                    team,
                    week,
                )
            ]

        else:
            timestamp = (
                pd.to_datetime(
                    clean(
                        row[
                            timestamp_col
                        ]
                    ),
                    utc=True,
                    errors="coerce",
                )
            )

            if pd.isna(
                timestamp
            ):
                continue

            snapshot = (
                timestamp_rows[
                    (
                        team,
                        pd.Timestamp(
                            timestamp
                        ),
                    )
                ]
            )

        merge_depth_entry(
            snapshot,
            player_id=player_id,
            player_name=player_name,
            position=position,
            rank=rank,
            slot=slot,
        )

    timestamped = defaultdict(
        list
    )

    for (
        team,
        timestamp,
    ), snapshot in (
        timestamp_rows.items()
    ):
        timestamped[
            team
        ].append(
            (
                timestamp,
                snapshot,
            )
        )

    for team in timestamped:
        timestamped[
            team
        ].sort(
            key=lambda item: item[0]
        )

    return (
        (
            "weekly"
            if has_week
            else "timestamp"
        ),
        dict(weekly),
        dict(timestamped),
    )


def depth_snapshot_for_game(
    *,
    mode: str,
    weekly,
    timestamped,
    team: str,
    week: int,
    game,
):
    if mode == "weekly":
        exact = weekly.get(
            (
                team,
                week,
            )
        )

        if exact is not None:
            return exact

        available = sorted(
            source_week
            for (
                source_team,
                source_week,
            ) in weekly
            if (
                source_team == team
                and source_week
                <= week
            )
        )

        if not available:
            return {}

        return weekly[
            (
                team,
                available[-1],
            )
        ]

    values = (
        timestamped.get(
            team,
            [],
        )
    )

    if not values:
        return {}

    cutoff = (
        conservative_depth_cutoff(
            game
        )
    )

    timestamps = [
        item[0]
        for item in values
    ]

    index = (
        bisect_left(
            timestamps,
            cutoff,
        )
        - 1
    )

    if index < 0:
        return {}

    return values[
        index
    ][1]


def apply_depth_candidates(
    rows,
    *,
    games,
    resolver,
    depth,
    label,
) -> None:
    (
        mode,
        weekly,
        timestamped,
    ) = build_depth_provider(
        depth,
        resolver,
        label=label,
    )

    for game in games:
        for team in (
            game["home_team"],
            game["away_team"],
        ):
            snapshot = (
                depth_snapshot_for_game(
                    mode=mode,
                    weekly=weekly,
                    timestamped=(
                        timestamped
                    ),
                    team=team,
                    week=game["week"],
                    game=game,
                )
            )

            for (
                player_id,
                depth_record,
            ) in snapshot.items():
                identity = (
                    resolver.get(
                        player_id
                    )
                )

                if identity is None:
                    continue

                candidate_key = (
                    game["season"],
                    game["week"],
                    game["game_id"],
                    player_id,
                )

                existing = rows.get(
                    candidate_key
                )

                if (
                    existing is not None
                    and existing["team"] != team
                ):
                    # The target-game weekly roster is authoritative for
                    # game-specific team assignment. A stale/conflicting
                    # depth snapshot cannot move a rostered player to the
                    # opponent.
                    if int(
                        existing.get(
                            "roster_flag",
                            0,
                        )
                    ) == 1:
                        continue

                    # If neither side has roster evidence, do not guess
                    # between conflicting depth-only assignments.
                    raise ValueError(
                        "Depth-only player resolves to both teams "
                        "in same game: "
                        f"game={game['game_id']} "
                        f"player={player_id} "
                        f"teams={existing['team']},{team}"
                    )

                record = (
                    upsert_candidate(
                        rows,
                        game=game,
                        player_id=(
                            player_id
                        ),
                        team=team,
                        identity=(
                            identity
                        ),
                        source="depth",
                        player_name=(
                            depth_record[
                                "player_name"
                            ]
                        ),
                        position=(
                            depth_record[
                                "position"
                            ]
                        ),
                    )
                )

                record[
                    "depth_present_flag"
                ] = 1

                record[
                    "depth_rank"
                ] = depth_record[
                    "rank"
                ]

                record[
                    "depth_slot"
                ] = depth_record[
                    "slot"
                ]

                record[
                    "depth_starter_flag"
                ] = depth_record[
                    "starter"
                ]


def load_snap_data(
    *,
    seasons,
    config,
    repo,
    resolver,
):
    current = {}
    played = set()

    for season in seasons:
        path = (
            repo
            / config["paths"][
                "historical_snaps_pattern"
            ].format(
                season=season
            )
        )

        snap = (
            common.read_parquet_required(
                path
            )
        )

        label = str(path)

        team_col = choose_column(
            snap,
            ["team"],
            label=label,
            required=True,
        )

        week_col = choose_column(
            snap,
            ["week"],
            label=label,
            required=True,
        )

        gsis_col = choose_column(
            snap,
            ["gsis_id"],
            label=label,
        )

        pfr_col = choose_column(
            snap,
            [
                "pfr_player_id",
                "pfr_id",
            ],
            label=label,
        )

        name_col = choose_column(
            snap,
            [
                "player",
                "full_name",
                "player_name",
            ],
            label=label,
        )

        position_col = choose_column(
            snap,
            ["position"],
            label=label,
        )

        offense_pct_col = (
            choose_column(
                snap,
                ["offense_pct"],
                label=label,
                required=True,
            )
        )

        defense_pct_col = (
            choose_column(
                snap,
                ["defense_pct"],
                label=label,
                required=True,
            )
        )

        offense_snaps_col = (
            choose_column(
                snap,
                ["offense_snaps"],
                label=label,
            )
        )

        defense_snaps_col = (
            choose_column(
                snap,
                ["defense_snaps"],
                label=label,
            )
        )

        st_snaps_col = (
            choose_column(
                snap,
                [
                    "st_snaps",
                    "special_teams_snaps",
                ],
                label=label,
            )
        )

        st_pct_col = (
            choose_column(
                snap,
                [
                    "st_pct",
                    "special_teams_pct",
                ],
                label=label,
            )
        )

        if (
            gsis_col is None
            and pfr_col is None
            and name_col is None
        ):
            raise ValueError(
                f"{label}: no player "
                "identity column."
            )

        # Snap-count PFR aliases are not globally authoritative.
        # Resolve them against the target-week team roster first so
        # duplicate names, stale aliases, and trades cannot assign a
        # player to the opponent.
        roster_path = (
            repo
            / config["paths"][
                "historical_rosters_pattern"
            ].format(
                season=season
            )
        )

        roster_context = (
            common.read_parquet_required(
                roster_path
            )
        )

        roster_label = str(
            roster_path
        )

        roster_team_col = choose_column(
            roster_context,
            ["team", "club_code"],
            label=roster_label,
            required=True,
        )

        roster_week_col = choose_column(
            roster_context,
            ["week"],
            label=roster_label,
            required=True,
        )

        roster_gsis_col = choose_column(
            roster_context,
            ["gsis_id"],
            label=roster_label,
        )

        roster_pfr_col = choose_column(
            roster_context,
            [
                "pfr_id",
                "pfr_player_id",
            ],
            label=roster_label,
        )

        roster_name_col = choose_column(
            roster_context,
            [
                "full_name",
                "display_name",
                "player_name",
            ],
            label=roster_label,
        )

        roster_position_col = choose_column(
            roster_context,
            [
                "position",
                "depth_chart_position",
                "ngs_position",
            ],
            label=roster_label,
        )

        roster_by_name = defaultdict(set)
        roster_by_pfr = defaultdict(set)
        roster_position = {}

        for roster_row in roster_context.to_dict(
            orient="records"
        ):
            roster_team = (
                common.normalize_team(
                    roster_row[
                        roster_team_col
                    ]
                )
            )

            roster_week = parse_int(
                roster_row[
                    roster_week_col
                ]
            )

            roster_gsis = (
                common.normalize_player_id(
                    roster_row[
                        roster_gsis_col
                    ]
                )
                if roster_gsis_col
                else ""
            )

            if (
                not roster_team
                or roster_week is None
                or not roster_gsis
            ):
                continue

            # Require canonical identity to exist in the
            # crosswalk before using roster context.
            if resolver.get(
                roster_gsis
            ) is None:
                continue

            roster_name = (
                common.normalize_name(
                    roster_row[
                        roster_name_col
                    ]
                )
                if roster_name_col
                else ""
            )

            roster_pfr = (
                common.normalize_player_id(
                    roster_row[
                        roster_pfr_col
                    ]
                )
                if roster_pfr_col
                else ""
            )

            roster_pos = (
                normalize_position(
                    roster_row[
                        roster_position_col
                    ]
                )
                if roster_position_col
                else ""
            )

            if roster_name:
                roster_by_name[
                    (
                        roster_week,
                        roster_team,
                        roster_name,
                    )
                ].add(
                    roster_gsis
                )

            if roster_pfr:
                roster_by_pfr[
                    (
                        roster_week,
                        roster_team,
                        roster_pfr,
                    )
                ].add(
                    roster_gsis
                )

            roster_position[
                (
                    roster_week,
                    roster_team,
                    roster_gsis,
                )
            ] = roster_pos

        for row in snap.to_dict(
            orient="records"
        ):
            team = (
                common.normalize_team(
                    row[team_col]
                )
            )

            week = parse_int(
                row[week_col]
            )

            if (
                not team
                or week is None
            ):
                continue

            snap_gsis = (
                common.normalize_player_id(
                    row[gsis_col]
                )
                if gsis_col
                else ""
            )

            snap_pfr = (
                common.normalize_player_id(
                    row[pfr_col]
                )
                if pfr_col
                else ""
            )

            snap_name = (
                clean(
                    row[name_col]
                )
                if name_col
                else ""
            )

            snap_name_key = (
                common.normalize_name(
                    snap_name
                )
            )

            snap_position = (
                normalize_position(
                    row[position_col]
                )
                if position_col
                else ""
            )

            # Native GSIS remains authoritative.
            identity = (
                resolver.get(
                    snap_gsis
                )
                if snap_gsis
                else None
            )

            context_candidates = set()

            if identity is None:
                candidate_sets = []

                if snap_pfr:
                    pfr_candidates = (
                        roster_by_pfr.get(
                            (
                                week,
                                team,
                                snap_pfr,
                            ),
                            set(),
                        )
                    )

                    if pfr_candidates:
                        candidate_sets.append(
                            set(
                                pfr_candidates
                            )
                        )

                if snap_name_key:
                    name_candidates = (
                        roster_by_name.get(
                            (
                                week,
                                team,
                                snap_name_key,
                            ),
                            set(),
                        )
                    )

                    if name_candidates:
                        candidate_sets.append(
                            set(
                                name_candidates
                            )
                        )

                if candidate_sets:
                    context_candidates = set(
                        candidate_sets[0]
                    )

                    for candidates in (
                        candidate_sets[1:]
                    ):
                        intersection = (
                            context_candidates
                            & candidates
                        )

                        if intersection:
                            context_candidates = (
                                intersection
                            )
                        else:
                            context_candidates |= (
                                candidates
                            )

                # Position is used only to disambiguate multiple
                # players already established on this team/week.
                if (
                    len(
                        context_candidates
                    ) > 1
                    and snap_position
                ):
                    position_matches = {
                        candidate
                        for candidate
                        in context_candidates
                        if normalize_position(
                            roster_position.get(
                                (
                                    week,
                                    team,
                                    candidate,
                                ),
                                "",
                            )
                        )
                        == snap_position
                    }

                    if position_matches:
                        context_candidates = (
                            position_matches
                        )

                if (
                    len(
                        context_candidates
                    ) == 1
                ):
                    contextual_gsis = next(
                        iter(
                            context_candidates
                        )
                    )

                    identity = (
                        resolver.get(
                            contextual_gsis
                        )
                    )

            if identity is None:
                global_identity = (
                    resolver.resolve(
                        pfr_id=snap_pfr,
                        name=snap_name,
                    )
                )

                if global_identity is not None:
                    global_gsis = (
                        global_identity[
                            "player_id"
                        ]
                    )

                    # If target-week roster context exists, a
                    # global alias may only resolve inside that
                    # team's candidate set.
                    if (
                        not context_candidates
                        or global_gsis
                        in context_candidates
                    ):
                        identity = (
                            global_identity
                        )

            if identity is None:
                continue

            player_id = (
                identity[
                    "player_id"
                ]
            )

            offense = (
                normalize_percentage(
                    row[
                        offense_pct_col
                    ]
                )
            )

            defense = (
                normalize_percentage(
                    row[
                        defense_pct_col
                    ]
                )
            )

            key = (
                season,
                week,
                team,
                player_id,
            )

            previous = (
                current.get(
                    key,
                    (
                        0.0,
                        0.0,
                    ),
                )
            )

            current[key] = (
                max(
                    previous[0],
                    offense,
                ),
                max(
                    previous[1],
                    defense,
                ),
            )

            did_play = (
                offense > 0
                or defense > 0
                or (
                    offense_snaps_col
                    is not None
                    and positive_numeric(
                        row[
                            offense_snaps_col
                        ]
                    )
                )
                or (
                    defense_snaps_col
                    is not None
                    and positive_numeric(
                        row[
                            defense_snaps_col
                        ]
                    )
                )
                or (
                    st_snaps_col
                    is not None
                    and positive_numeric(
                        row[
                            st_snaps_col
                        ]
                    )
                )
                or (
                    st_pct_col
                    is not None
                    and (
                        normalize_percentage(
                            row[
                                st_pct_col
                            ]
                        )
                        > 0
                    )
                )
            )

            if did_play:
                played.add(
                    key
                )

    by_player_week = {}

    for (
        season,
        week,
        _team,
        player_id,
    ), value in current.items():
        key = (
            player_id,
            season,
            week,
        )

        previous = (
            by_player_week.get(
                key,
                (
                    0.0,
                    0.0,
                ),
            )
        )

        by_player_week[
            key
        ] = (
            max(
                previous[0],
                value[0],
            ),
            max(
                previous[1],
                value[1],
            ),
        )

    history = defaultdict(
        list
    )

    for (
        player_id,
        season,
        week,
    ), (
        offense,
        defense,
    ) in (
        by_player_week.items()
    ):
        history[
            player_id
        ].append(
            (
                (
                    season,
                    week,
                ),
                offense,
                defense,
            )
        )

    for player_id in history:
        history[
            player_id
        ].sort(
            key=lambda item: item[0]
        )

    return (
        current,
        played,
        dict(history),
    )


def load_participation_data(
    *,
    seasons,
    config,
    repo,
):
    offense_den = defaultdict(
        int
    )

    defense_den = defaultdict(
        int
    )

    offense_num = defaultdict(
        int
    )

    defense_num = defaultdict(
        int
    )

    # Participation files can contain isolated identity corruption
    # where the same GSIS ID is assigned to both teams in one game.
    # Such player-game identities are ambiguous and must not create
    # universe candidates or prior-participation history.
    participant_game_teams = defaultdict(
        set
    )

    for season in seasons:
        if season < 2016:
            continue

        path = (
            repo
            / config["paths"][
                "historical_participation_pattern"
            ].format(
                season=season
            )
        )

        participation = (
            common.read_parquet_required(
                path
            )
        )

        label = str(path)

        game_col = choose_column(
            participation,
            [
                "nflverse_game_id",
                "game_id",
            ],
            label=label,
            required=True,
        )

        possession_col = (
            choose_column(
                participation,
                [
                    "possession_team",
                    "posteam",
                ],
                label=label,
                required=True,
            )
        )

        offense_col = choose_column(
            participation,
            ["offense_players"],
            label=label,
            required=True,
        )

        defense_col = choose_column(
            participation,
            ["defense_players"],
            label=label,
            required=True,
        )

        for row in (
            participation.to_dict(
                orient="records"
            )
        ):
            source_game_id = clean(
                row[game_col]
            )

            parsed = (
                parse_nflverse_game_id(
                    source_game_id
                )
            )

            if parsed is None:
                continue

            (
                parsed_season,
                week,
                away_team,
                home_team,
            ) = parsed

            if (
                parsed_season
                != season
            ):
                continue

            possession = (
                common.normalize_team(
                    row[
                        possession_col
                    ]
                )
            )

            if (
                possession
                == away_team
            ):
                defense_team = (
                    home_team
                )

            elif (
                possession
                == home_team
            ):
                defense_team = (
                    away_team
                )

            else:
                continue

            offense_players = (
                extract_gsis_ids(
                    row[
                        offense_col
                    ]
                )
            )

            defense_players = (
                extract_gsis_ids(
                    row[
                        defense_col
                    ]
                )
            )

            if offense_players:
                offense_den[
                    (
                        season,
                        week,
                        possession,
                    )
                ] += 1

                for player_id in (
                    offense_players
                ):
                    offense_num[
                        (
                            season,
                            week,
                            possession,
                            player_id,
                        )
                    ] += 1

                    participant_game_teams[
                        (
                            season,
                            week,
                            source_game_id,
                            player_id,
                        )
                    ].add(
                        possession
                    )

            if defense_players:
                defense_den[
                    (
                        season,
                        week,
                        defense_team,
                    )
                ] += 1

                for player_id in (
                    defense_players
                ):
                    defense_num[
                        (
                            season,
                            week,
                            defense_team,
                            player_id,
                        )
                    ] += 1

                    participant_game_teams[
                        (
                            season,
                            week,
                            source_game_id,
                            player_id,
                        )
                    ].add(
                        defense_team
                    )

    ambiguous_player_weeks = {
        (
            season,
            week,
            player_id,
        )
        for (
            season,
            week,
            _game_id,
            player_id,
        ), teams in participant_game_teams.items()
        if len(teams) > 1
    }

    current = {}

    all_keys = (
        set(offense_num)
        | set(defense_num)
    )

    for key in all_keys:
        (
            season,
            week,
            team,
            _player_id,
        ) = key

        if (
            season,
            week,
            _player_id,
        ) in ambiguous_player_weeks:
            continue

        off_den = (
            offense_den.get(
                (
                    season,
                    week,
                    team,
                ),
                0,
            )
        )

        def_den = (
            defense_den.get(
                (
                    season,
                    week,
                    team,
                ),
                0,
            )
        )

        off_share = (
            offense_num.get(
                key,
                0,
            )
            / off_den
            if off_den
            else 0.0
        )

        def_share = (
            defense_num.get(
                key,
                0,
            )
            / def_den
            if def_den
            else 0.0
        )

        current[key] = (
            off_share,
            def_share,
        )

    played = set(
        current
    )

    by_player_week = {}

    for (
        season,
        week,
        _team,
        player_id,
    ), value in current.items():
        key = (
            player_id,
            season,
            week,
        )

        previous = (
            by_player_week.get(
                key,
                (
                    0.0,
                    0.0,
                ),
            )
        )

        by_player_week[
            key
        ] = (
            max(
                previous[0],
                value[0],
            ),
            max(
                previous[1],
                value[1],
            ),
        )

    history = defaultdict(
        list
    )

    for (
        player_id,
        season,
        week,
    ), (
        offense,
        defense,
    ) in (
        by_player_week.items()
    ):
        history[
            player_id
        ].append(
            (
                (
                    season,
                    week,
                ),
                offense,
                defense,
            )
        )

    for player_id in history:
        history[
            player_id
        ].sort(
            key=lambda item: item[0]
        )

    return (
        current,
        played,
        dict(history),
    )


def latest_prior(
    history,
    player_id: str,
    season: int,
    week: int,
):
    values = history.get(
        player_id
    )

    if not values:
        return None

    keys = [
        item[0]
        for item in values
    ]

    index = (
        bisect_left(
            keys,
            (
                season,
                week,
            ),
        )
        - 1
    )

    if index < 0:
        return None

    (
        _source_key,
        offense,
        defense,
    ) = values[index]

    return (
        offense,
        defense,
    )


def add_actual_participants(
    rows,
    *,
    source: str,
    played,
    resolver,
    games_by_team_week,
) -> None:
    for (
        season,
        week,
        team,
        raw_player_id,
    ) in sorted(
        played
    ):
        game = (
            games_by_team_week.get(
                (
                    season,
                    week,
                    team,
                )
            )
        )

        if game is None:
            continue

        player_id = (
            common.normalize_player_id(
                raw_player_id
            )
        )

        if not player_id:
            continue

        identity = (
            resolver.get(
                player_id
            )
        )

        record = (
            upsert_candidate(
                rows,
                game=game,
                player_id=player_id,
                team=team,
                identity=identity,
                source=source,
            )
        )

        record[
            "played_game_flag"
        ] = 1


def finalize_usage(
    rows,
    *,
    snap_history,
    participation_history,
    snap_played,
    participation_played,
) -> None:
    for record in rows.values():
        player_id = (
            record["player_id"]
        )

        season = int(
            record["season"]
        )

        week = int(
            record["week"]
        )

        team = record["team"]

        prior_snap = (
            latest_prior(
                snap_history,
                player_id,
                season,
                week,
            )
        )

        if prior_snap is not None:
            record[
                "prior_offense_snap_pct"
            ] = prior_snap[0]

            record[
                "prior_defense_snap_pct"
            ] = prior_snap[1]

        prior_participation = (
            latest_prior(
                participation_history,
                player_id,
                season,
                week,
            )
        )

        if (
            prior_participation
            is not None
        ):
            record[
                "prior_offense_participation"
            ] = (
                prior_participation[
                    0
                ]
            )

            record[
                "prior_defense_participation"
            ] = (
                prior_participation[
                    1
                ]
            )

        played_key = (
            season,
            week,
            team,
            player_id,
        )

        record[
            "played_game_flag"
        ] = int(
            record[
                "played_game_flag"
            ]
            or (
                played_key
                in snap_played
            )
            or (
                played_key
                in participation_played
            )
        )


def validate_output(
    output,
    games_by_team_week,
    config,
) -> None:
    common.require_columns(
        output,
        OUTPUT_COLUMNS,
        (
            "historical "
            "player-game universe"
        ),
    )

    common.ensure_unique(
        output,
        [
            "season",
            "week",
            "game_id",
            "player_id",
        ],
        (
            "historical "
            "player-game universe grain"
        ),
    )

    common.reject_forbidden_feature_columns(
        output.columns,
        config,
    )

    invalid = []

    for row in output[
        [
            "season",
            "week",
            "game_id",
            "player_id",
            "team",
        ]
    ].to_dict(
        orient="records"
    ):
        game = (
            games_by_team_week.get(
                (
                    int(
                        row[
                            "season"
                        ]
                    ),
                    int(
                        row[
                            "week"
                        ]
                    ),
                    clean(
                        row[
                            "team"
                        ]
                    ),
                )
            )
        )

        if (
            game is None
            or row["game_id"]
            != game["game_id"]
        ):
            invalid.append(
                row
            )

            if len(invalid) >= 10:
                break

    if invalid:
        raise ValueError(
            "Players assigned outside "
            "their target game. "
            f"Sample={invalid}"
        )

    if not output[
        "season_type"
    ].eq("REG").all():
        raise ValueError(
            "Universe contains "
            "non-REG rows."
        )

    home_away = (
        output[
            "home_flag"
        ].astype(int)
        + output[
            "away_flag"
        ].astype(int)
    )

    if not home_away.eq(
        1
    ).all():
        raise ValueError(
            "Every row must be "
            "exactly home or away."
        )

    if not (
        output[
            "player_id"
        ]
        .fillna("")
        .astype(str)
        .str.len()
        .gt(0)
        .all()
    ):
        raise ValueError(
            "Blank player_id "
            "found in universe."
        )


def main() -> int:
    config = common.load_config()
    repo = common.repo_root()

    start_season = int(
        config[
            "seasons"
        ][
            "historical_start"
        ]
    )

    end_season = int(
        config[
            "seasons"
        ][
            "historical_end"
        ]
    )

    if (
        start_season
        > end_season
    ):
        raise ValueError(
            "historical_start "
            "cannot exceed "
            "historical_end."
        )

    seasons = range(
        start_season,
        end_season + 1,
    )

    games_path = (
        repo
        / config[
            "paths"
        ][
            "historical_games"
        ]
    )

    crosswalk_path = (
        repo
        / config[
            "paths"
        ][
            "identity_crosswalk"
        ]
    )

    output_path = (
        repo
        / config[
            "paths"
        ][
            "historical_universe"
        ]
    )

    (
        games,
        games_by_team_week,
    ) = load_games(
        games_path,
        start_season,
        end_season,
    )

    games_by_season = (
        defaultdict(list)
    )

    for game in games:
        games_by_season[
            int(
                game["season"]
            )
        ].append(
            game
        )

    crosswalk = (
        common.read_parquet_required(
            crosswalk_path
        )
    )

    resolver = (
        IdentityResolver(
            crosswalk
        )
    )

    universe_rows = {}

    for season in seasons:
        roster_path = (
            repo
            / config[
                "paths"
            ][
                "historical_rosters_pattern"
            ].format(
                season=season
            )
        )

        roster = (
            common.read_parquet_required(
                roster_path
            )
        )

        add_roster_candidates(
            universe_rows,
            roster=roster,
            resolver=resolver,
            games_by_team_week=(
                games_by_team_week
            ),
            season=season,
            label=str(
                roster_path
            ),
        )

        depth_path = (
            repo
            / config[
                "paths"
            ][
                "historical_depth_pattern"
            ].format(
                season=season
            )
        )

        depth = (
            common.read_parquet_required(
                depth_path
            )
        )

        apply_depth_candidates(
            universe_rows,
            games=(
                games_by_season.get(
                    season,
                    [],
                )
            ),
            resolver=resolver,
            depth=depth,
            label=str(
                depth_path
            ),
        )

    (
        _snap_current,
        snap_played,
        snap_history,
    ) = load_snap_data(
        seasons=seasons,
        config=config,
        repo=repo,
        resolver=resolver,
    )

    (
        _part_current,
        participation_played,
        participation_history,
    ) = load_participation_data(
        seasons=seasons,
        config=config,
        repo=repo,
    )

    # Backstop roster/depth data gaps
    # with players proven to have taken
    # a target-game snap. This preserves
    # legitimate zero-event player-games.
    add_actual_participants(
        universe_rows,
        source="snap",
        played=snap_played,
        resolver=resolver,
        games_by_team_week=(
            games_by_team_week
        ),
    )

    add_actual_participants(
        universe_rows,
        source="participation",
        played=(
            participation_played
        ),
        resolver=resolver,
        games_by_team_week=(
            games_by_team_week
        ),
    )

    finalize_usage(
        universe_rows,
        snap_history=(
            snap_history
        ),
        participation_history=(
            participation_history
        ),
        snap_played=snap_played,
        participation_played=(
            participation_played
        ),
    )

    records = []

    for record in (
        universe_rows.values()
    ):
        sources = sorted(
            record.pop(
                "_sources"
            ),
            key=lambda value: (
                SOURCE_ORDER[
                    value
                ]
            ),
        )

        record[
            "universe_source"
        ] = ";".join(
            sources
        )

        records.append(
            record
        )

    output = pd.DataFrame(
        records,
        columns=OUTPUT_COLUMNS,
    )

    if output.empty:
        raise RuntimeError(
            "Historical player-game "
            "universe is empty."
        )

    for column in (
        "season",
        "week",
        "home_flag",
        "away_flag",
        "roster_flag",
        "depth_present_flag",
        "depth_starter_flag",
        "played_game_flag",
    ):
        output[
            column
        ] = pd.to_numeric(
            output[column],
            errors="raise",
        ).astype(
            "int64"
        )

    output[
        "depth_rank"
    ] = pd.array(
        output[
            "depth_rank"
        ],
        dtype="Int64",
    )

    for column in (
        "prior_offense_snap_pct",
        "prior_defense_snap_pct",
        "prior_offense_participation",
        "prior_defense_participation",
    ):
        output[
            column
        ] = pd.to_numeric(
            output[column],
            errors="coerce",
        )

    validate_output(
        output,
        games_by_team_week,
        config,
    )

    common.write_parquet_atomic(
        output,
        output_path,
    )

    common.log_run(
        "build_historical_universe.py",
        {
            "historical_start": (
                start_season
            ),
            "historical_end": (
                end_season
            ),
            (
                "regular_season_"
                "games"
            ): len(games),
            "rows": len(output),
            "unique_players": int(
                output[
                    "player_id"
                ].nunique()
            ),
            "roster_rows": int(
                output[
                    "roster_flag"
                ].sum()
            ),
            "depth_present_rows": int(
                output[
                    "depth_present_flag"
                ].sum()
            ),
            "played_rows": int(
                output[
                    "played_game_flag"
                ].sum()
            ),
            (
                "participation_"
                "available_start"
            ): 2016,
            (
                "played_game_flag_"
                "target_only"
            ): True,
            "output": str(
                output_path.relative_to(
                    repo
                )
            ),
        },
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
