#!/usr/bin/env python3
"""
Build player-level weekly opportunity and efficiency measurements.

READS:
    docs/win/football/nfl/00_intake/pbp/{season}_pbp.csv.gz
    docs/win/football/nfl/data/historic_data/player_stats/stats_player_week_{season}.parquet
    docs/win/football/nfl/data/historic_data/snap_counts/snap_counts_{season}.parquet
    docs/win/football/nfl/data/historic_data/participation/pbp_participation_{season}.parquet
    docs/win/football/nfl/prop_engine/data/identity/player_crosswalk.parquet

WRITES:
    docs/win/football/nfl/prop_engine/data/historical/opportunity/player_week_opportunity.parquet

Same-week realized values are raw measurements only. Downstream model
feature builders must lag them before prediction use.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any
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
    "season", "week", "game_id", "player_id", "team", "position", "position_group",
    "pass_attempts", "completions", "dropbacks", "passing_air_yards", "passing_yards",
    "passing_tds", "carries", "rushing_yards", "rushing_tds", "targets", "receptions",
    "receiving_yards", "receiving_tds", "receiving_air_yards", "red_zone_pass_attempts",
    "red_zone_carries", "goal_line_carries", "red_zone_targets", "inside_10_targets",
    "inside_5_targets", "field_goal_attempts", "field_goals_made", "extra_point_attempts",
    "extra_points_made", "tackles", "sacks", "qb_hits", "offense_snap_pct",
    "defense_snap_pct", "offense_participation", "defense_participation", "target_share",
    "carry_share", "air_yards_share", "red_zone_target_share", "red_zone_carry_share",
    "goal_line_carry_share", "yards_per_attempt", "yards_per_carry", "yards_per_target",
    "catch_rate", "passing_td_rate", "rushing_td_rate", "receiving_td_rate",
    "tackle_rate_per_def_play", "sack_rate_per_def_play", "qb_hit_rate_per_def_play",
]

GRAIN = ["season", "week", "game_id", "player_id"]
TEAM_GRAIN = ["season", "week", "game_id", "team"]

RICH_COUNT_COLUMNS = [
    "dropbacks", "red_zone_pass_attempts", "red_zone_carries", "goal_line_carries",
    "red_zone_targets", "inside_10_targets", "inside_5_targets",
]

CORE_NONNEGATIVE_COLUMNS = [
    "pass_attempts", "completions", "passing_tds", "carries", "rushing_tds",
    "targets", "receptions", "receiving_tds", "field_goal_attempts",
    "field_goals_made", "extra_point_attempts", "extra_points_made",
    "tackles", "sacks", "qb_hits",
]

PERCENT_COLUMNS = [
    "offense_snap_pct", "defense_snap_pct",
    "offense_participation", "defense_participation",
]

GSIS_RE = re.compile(r"00-\d{7}")
GAME_RE = re.compile(
    r"^(?P<season>\d{4})_(?P<week>\d{1,2})_"
    r"(?P<away>[A-Za-z0-9]+)_(?P<home>[A-Za-z0-9]+)$"
)

HISTORICAL_FRANCHISE_ALIASES = {"SD": "LAC", "OAK": "LV", "STL": "LAR"}


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.casefold() in {"", "nan", "none", "null", "<na>", "nat"}:
        return ""
    return text


def canonical_team(value: Any) -> str:
    team = common.normalize_team(value)
    return HISTORICAL_FRANCHISE_ALIASES.get(team, team)


def numeric_series(
    series: pd.Series,
    *,
    label: str,
    fill_zero: bool = False,
) -> pd.Series:
    converted = pd.to_numeric(series, errors="coerce")
    invalid = (
        series.notna()
        & series.astype(str).str.strip().ne("")
        & converted.isna()
    )
    if invalid.any():
        examples = series.loc[invalid].astype(str).head(10).tolist()
        raise ValueError(f"{label}: non-numeric values found. Examples={examples}")
    converted = converted.astype(float)
    return converted.fillna(0.0) if fill_zero else converted


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce").astype(float)
    den = pd.to_numeric(denominator, errors="coerce").astype(float)
    valid = num.notna() & den.notna() & den.ne(0.0)
    result = pd.Series(float("nan"), index=num.index, dtype="float64")
    result.loc[valid] = num.loc[valid] / den.loc[valid]
    return result


def extract_gsis_ids(value: Any) -> list[str]:
    text = clean(value)
    if not text:
        return []
    return list(dict.fromkeys(GSIS_RE.findall(text)))


def parse_game_context(game_id: Any) -> tuple[int, int, str, str]:
    text = clean(game_id)
    match = GAME_RE.fullmatch(text)
    if not match:
        raise ValueError(f"Unsupported nflverse game_id: {game_id!r}")
    return (
        int(match.group("season")),
        int(match.group("week")),
        canonical_team(match.group("away")),
        canonical_team(match.group("home")),
    )


def unique_crosswalk_map(
    crosswalk: pd.DataFrame,
    key_column: str,
) -> dict[str, str]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for key_value, gsis_value in zip(crosswalk[key_column], crosswalk["gsis_id"]):
        key = clean(key_value)
        gsis = common.normalize_player_id(gsis_value)
        if key and gsis:
            grouped[key].add(gsis)
    return {
        key: next(iter(values))
        for key, values in grouped.items()
        if len(values) == 1
    }


def build_crosswalk_maps(
    crosswalk: pd.DataFrame,
) -> tuple[dict[str, str], dict[str, str]]:
    common.require_columns(
        crosswalk,
        ["gsis_id", "pfr_id", "normalized_name"],
        "player crosswalk",
    )
    working = crosswalk.copy()
    working["gsis_id"] = working["gsis_id"].map(common.normalize_player_id)
    working["pfr_id"] = working["pfr_id"].map(clean)
    names = working["normalized_name"].map(clean)
    if "display_name" in working.columns:
        fallback = working["display_name"].map(common.normalize_name)
        names = names.where(names.ne(""), fallback)
    working["_name_key"] = names
    return (
        unique_crosswalk_map(working, "pfr_id"),
        unique_crosswalk_map(working, "_name_key"),
    )


def prepare_stats(
    source: pd.DataFrame,
    *,
    season: int,
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    required = [
        "season", "week", "season_type", "game_id", "player_id", "team",
        "position", "position_group", "completions", "attempts",
        "passing_air_yards", "passing_yards", "passing_tds", "carries",
        "rushing_yards", "rushing_tds", "targets", "receptions",
        "receiving_yards", "receiving_tds", "receiving_air_yards",
        "fg_att", "fg_made", "pat_att", "pat_made", "def_tackles_solo",
        "def_tackle_assists", "def_sacks", "def_qb_hits",
    ]
    common.require_columns(source, required, str(path))

    working = source.loc[
        pd.to_numeric(source["season"], errors="coerce").eq(season)
        & source["season_type"].astype(str).str.upper().eq("REG")
    ].copy()
    if working.empty:
        raise RuntimeError(f"{path}: no regular-season rows for {season}.")

    working["season"] = season
    working["week"] = pd.to_numeric(working["week"], errors="raise").astype(int)
    working["game_id"] = working["game_id"].map(clean)
    working["player_id"] = working["player_id"].map(common.normalize_player_id)
    working["team"] = working["team"].map(canonical_team)

    if working["game_id"].eq("").any():
        raise ValueError(f"{path}: blank game_id in regular season.")
    if working["team"].eq("").any():
        raise ValueError(f"{path}: blank team in regular season.")

    source_map = {
        "pass_attempts": "attempts",
        "completions": "completions",
        "passing_air_yards": "passing_air_yards",
        "passing_yards": "passing_yards",
        "passing_tds": "passing_tds",
        "carries": "carries",
        "rushing_yards": "rushing_yards",
        "rushing_tds": "rushing_tds",
        "targets": "targets",
        "receptions": "receptions",
        "receiving_yards": "receiving_yards",
        "receiving_tds": "receiving_tds",
        "receiving_air_yards": "receiving_air_yards",
        "field_goal_attempts": "fg_att",
        "field_goals_made": "fg_made",
        "extra_point_attempts": "pat_att",
        "extra_points_made": "pat_made",
        "_solo_tackles": "def_tackles_solo",
        "_assisted_tackles": "def_tackle_assists",
        "sacks": "def_sacks",
        "qb_hits": "def_qb_hits",
    }
    for output_column, source_column in source_map.items():
        working[output_column] = numeric_series(
            working[source_column],
            label=f"{path}:{source_column}",
            fill_zero=True,
        )

    working["tackles"] = working["_solo_tackles"] + working["_assisted_tackles"]

    # Team denominators are computed before blank player IDs are removed.
    # This preserves unresolved player events in true team totals.
    team_denominators = (
        working.groupby(TEAM_GRAIN, as_index=False, dropna=False)
        .agg(
            _team_targets=("targets", "sum"),
            _team_rush_attempts_stats=("carries", "sum"),
            _team_receiving_air_yards=("receiving_air_yards", "sum"),
        )
    )

    blank_id = working["player_id"].eq("")
    diagnostics = {
        "regular_rows": int(len(working)),
        "blank_player_id_rows": int(blank_id.sum()),
    }
    working = working.loc[~blank_id].copy()

    common.ensure_unique(working, GRAIN, f"{path} regular player-game grain")

    base_columns = [
        "season", "week", "game_id", "player_id", "team", "position",
        "position_group", "pass_attempts", "completions", "passing_air_yards",
        "passing_yards", "passing_tds", "carries", "rushing_yards",
        "rushing_tds", "targets", "receptions", "receiving_yards",
        "receiving_tds", "receiving_air_yards", "field_goal_attempts",
        "field_goals_made", "extra_point_attempts", "extra_points_made",
        "tackles", "sacks", "qb_hits",
    ]
    result = working[base_columns].copy()
    result["position"] = result["position"].map(clean).str.upper()
    result["position_group"] = result["position_group"].map(clean).str.upper()
    return result, team_denominators, diagnostics


def prepare_snaps(
    source: pd.DataFrame,
    *,
    season: int,
    path: Path,
    pfr_map: dict[str, str],
    name_map: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    required = [
        "season", "week", "game_id", "player", "pfr_player_id", "team",
        "offense_pct", "defense_pct", "defense_snaps",
    ]
    common.require_columns(source, required, str(path))

    empty_columns = [
        *GRAIN, "_snap_team", "offense_snap_pct",
        "defense_snap_pct", "_defense_snaps",
    ]
    if source.empty:
        return pd.DataFrame(columns=empty_columns), {
            "rows": 0, "resolved_rows": 0, "unresolved_rows": 0
        }

    working = source.loc[
        pd.to_numeric(source["season"], errors="coerce").eq(season)
    ].copy()

    if "game_type" in working.columns:
        regular = working["game_type"].astype(str).str.upper().eq("REG")
        if regular.any():
            working = working.loc[regular].copy()

    if working.empty:
        return pd.DataFrame(columns=empty_columns), {
            "rows": 0, "resolved_rows": 0, "unresolved_rows": 0
        }

    working["season"] = season
    working["week"] = pd.to_numeric(working["week"], errors="raise").astype(int)
    working["game_id"] = working["game_id"].map(clean)
    working["_snap_team"] = working["team"].map(canonical_team)

    pfr = working["pfr_player_id"].map(clean)
    names = working["player"].map(common.normalize_name)
    resolved = pfr.map(pfr_map)
    fallback = names.map(name_map)
    working["player_id"] = (
        resolved.where(resolved.notna() & resolved.astype(str).ne(""), fallback)
        .fillna("")
        .map(common.normalize_player_id)
    )

    working["offense_snap_pct"] = numeric_series(
        working["offense_pct"], label=f"{path}:offense_pct"
    )
    working["defense_snap_pct"] = numeric_series(
        working["defense_pct"], label=f"{path}:defense_pct"
    )
    working["_defense_snaps"] = numeric_series(
        working["defense_snaps"], label=f"{path}:defense_snaps"
    )

    for column in ["offense_snap_pct", "defense_snap_pct"]:
        invalid = working[column].notna() & ~working[column].between(0.0, 1.0)
        if invalid.any():
            sample = working.loc[
                invalid, ["game_id", "player", column]
            ].head(10).to_dict(orient="records")
            raise ValueError(f"{path}: {column} outside [0, 1]. Sample={sample}")

    unresolved = working["player_id"].eq("")
    diagnostics = {
        "rows": int(len(working)),
        "resolved_rows": int((~unresolved).sum()),
        "unresolved_rows": int(unresolved.sum()),
    }
    working = working.loc[~unresolved].copy()
    common.ensure_unique(
        working,
        [
            *GRAIN,
            "_snap_team",
        ],
        f"{path} resolved snap player-game-team grain",
    )
    return working[empty_columns].copy(), diagnostics


def build_participation(
    source: pd.DataFrame,
    *,
    season: int,
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    required = [
        "nflverse_game_id", "play_id", "possession_team",
        "offense_players", "defense_players",
    ]
    common.require_columns(source, required, str(path))

    off_den: dict[tuple[int, int, str, str], int] = defaultdict(int)
    def_den: dict[tuple[int, int, str, str], int] = defaultdict(int)
    off_num: dict[tuple[int, int, str, str, str], int] = defaultdict(int)
    def_num: dict[tuple[int, int, str, str, str], int] = defaultdict(int)
    valid_rows = 0

    for row in source.itertuples(index=False):
        game_id = clean(getattr(row, "nflverse_game_id"))
        if not game_id:
            continue
        try:
            game_season, week, away, home = parse_game_context(game_id)
        except ValueError:
            continue
        if game_season != season:
            continue

        possession = canonical_team(getattr(row, "possession_team"))
        if possession == away:
            defense_team = home
        elif possession == home:
            defense_team = away
        else:
            continue

        offense_ids = extract_gsis_ids(getattr(row, "offense_players"))
        defense_ids = extract_gsis_ids(getattr(row, "defense_players"))
        if not offense_ids and not defense_ids:
            continue
        valid_rows += 1

        if offense_ids:
            team_key = (season, week, game_id, possession)
            off_den[team_key] += 1
            for player_id in offense_ids:
                off_num[(*team_key, player_id)] += 1

        if defense_ids:
            team_key = (season, week, game_id, defense_team)
            def_den[team_key] += 1
            for player_id in defense_ids:
                def_num[(*team_key, player_id)] += 1

    player_records = []
    for key in set(off_num) | set(def_num):
        key_season, week, game_id, team, player_id = key
        team_key = (key_season, week, game_id, team)
        offense_count = off_num.get(key, 0)
        defense_count = def_num.get(key, 0)
        offense_denominator = off_den.get(team_key, 0)
        defense_denominator = def_den.get(team_key, 0)
        player_records.append({
            "season": key_season,
            "week": week,
            "game_id": game_id,
            "team": team,
            "player_id": player_id,
            "offense_participation": (
                offense_count / offense_denominator if offense_denominator else float("nan")
            ),
            "defense_participation": (
                defense_count / defense_denominator if defense_denominator else float("nan")
            ),
            "_participation_defense_plays": (
                float(defense_count) if defense_denominator else float("nan")
            ),
        })

    player_columns = [
        *GRAIN, "team", "offense_participation",
        "defense_participation", "_participation_defense_plays",
    ]
    player_frame = pd.DataFrame(player_records, columns=player_columns)
    if not player_frame.empty:
        common.ensure_unique(
            player_frame,
            [*GRAIN, "team"],
            f"{path} participation player-game-team grain",
        )

    team_records = []
    for key in set(off_den) | set(def_den):
        key_season, week, game_id, team = key
        team_records.append({
            "season": key_season,
            "week": week,
            "game_id": game_id,
            "team": team,
            "_offense_participation_available": int(off_den.get(key, 0) > 0),
            "_defense_participation_available": int(def_den.get(key, 0) > 0),
        })

    team_columns = [
        *TEAM_GRAIN, "_offense_participation_available",
        "_defense_participation_available",
    ]
    team_frame = pd.DataFrame(team_records, columns=team_columns)
    if not team_frame.empty:
        common.ensure_unique(team_frame, TEAM_GRAIN, f"{path} participation team-game grain")

    return player_frame, team_frame, {
        "rows": int(len(source)),
        "valid_rows": int(valid_rows),
        "player_rows": int(len(player_frame)),
    }


def group_player_event(
    pbp: pd.DataFrame,
    *,
    mask: pd.Series,
    id_column: str,
    output_column: str,
) -> pd.DataFrame:
    subset = pbp.loc[
        mask & pbp[id_column].ne(""),
        ["season", "week", "game_id", "team", id_column],
    ].copy()
    keys = ["season", "week", "game_id", "team", id_column]
    if subset.empty:
        return pd.DataFrame(
            columns=["season", "week", "game_id", "team", "player_id", output_column]
        )
    result = (
        subset.groupby(keys, as_index=False, dropna=False)
        .size()
        .rename(columns={id_column: "player_id", "size": output_column})
    )
    result[output_column] = result[output_column].astype(float)
    return result


def build_pbp_rich(
    source: pd.DataFrame,
    *,
    season: int,
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, set[str], dict[str, int]]:
    required = [
        "season_type", "week", "game_id", "posteam", "yardline_100",
        "pass_attempt", "qb_dropback", "rush_attempt", "qb_kneel",
        "passer_player_id", "rusher_player_id", "receiver_player_id",
    ]
    common.require_columns(source, required, str(path))

    working = source.loc[
        source["season_type"].astype(str).str.upper().eq("REG")
    ].copy()
    if working.empty:
        raise RuntimeError(f"{path}: no regular-season PBP rows.")

    working["season"] = season
    working["week"] = pd.to_numeric(working["week"], errors="raise").astype(int)
    working["game_id"] = working["game_id"].map(clean)
    working["team"] = working["posteam"].map(canonical_team)

    for column in ["pass_attempt", "qb_dropback", "rush_attempt", "qb_kneel"]:
        working[column] = numeric_series(
            working[column], label=f"{path}:{column}", fill_zero=True
        )
    working["yardline_100"] = numeric_series(
        working["yardline_100"], label=f"{path}:yardline_100"
    )
    for column in ["passer_player_id", "rusher_player_id", "receiver_player_id"]:
        working[column] = working[column].map(common.normalize_player_id)

    # Scramble fallback: qb_dropback may identify the quarterback only as rusher.
    working["_dropback_player_id"] = working["passer_player_id"].where(
        working["passer_player_id"].ne(""),
        working["rusher_player_id"],
    )

    yardline = working["yardline_100"]
    rz = yardline.notna() & yardline.le(20.0)
    inside_10 = yardline.notna() & yardline.le(10.0)
    inside_5 = yardline.notna() & yardline.le(5.0)
    pass_attempt = working["pass_attempt"].eq(1.0)
    rush_attempt = working["rush_attempt"].eq(1.0)
    dropback = working["qb_dropback"].eq(1.0)
    target = pass_attempt & working["receiver_player_id"].ne("")

    event_specs = [
        (dropback, "_dropback_player_id", "dropbacks"),
        (rz & pass_attempt, "passer_player_id", "red_zone_pass_attempts"),
        (rz & rush_attempt, "rusher_player_id", "red_zone_carries"),
        (inside_5 & rush_attempt, "rusher_player_id", "goal_line_carries"),
        (rz & target, "receiver_player_id", "red_zone_targets"),
        (inside_10 & target, "receiver_player_id", "inside_10_targets"),
        (inside_5 & target, "receiver_player_id", "inside_5_targets"),
    ]

    player_keys = [*GRAIN, "team"]
    player_frame = pd.DataFrame(columns=player_keys)
    for mask, id_column, output_column in event_specs:
        event = group_player_event(
            working,
            mask=mask,
            id_column=id_column,
            output_column=output_column,
        )
        if len(player_frame.columns) == len(player_keys) and player_frame.empty:
            player_frame = event
        else:
            player_frame = player_frame.merge(
                event, on=player_keys, how="outer", validate="one_to_one"
            )

    for column in RICH_COUNT_COLUMNS:
        if column not in player_frame.columns:
            player_frame[column] = float("nan")

    team_base = working[TEAM_GRAIN].drop_duplicates().copy()

    def team_count(mask: pd.Series, name: str) -> pd.DataFrame:
        counts = (
            working.loc[mask, TEAM_GRAIN]
            .groupby(TEAM_GRAIN, as_index=False, dropna=False)
            .size()
            .rename(columns={"size": name})
        )
        if not counts.empty:
            counts[name] = counts[name].astype(float)
        result = team_base.merge(
            counts, on=TEAM_GRAIN, how="left", validate="one_to_one"
        )
        result[name] = result[name].fillna(0.0)
        return result

    team_frame = (
        team_count(rz & target, "_team_red_zone_targets")
        .merge(
            team_count(rz & rush_attempt, "_team_red_zone_carries"),
            on=TEAM_GRAIN, how="outer", validate="one_to_one"
        )
        .merge(
            team_count(inside_5 & rush_attempt, "_team_goal_line_carries"),
            on=TEAM_GRAIN, how="outer", validate="one_to_one"
        )
        .merge(
            team_count(rush_attempt, "_team_rush_attempts_pbp"),
            on=TEAM_GRAIN, how="outer", validate="one_to_one"
        )
    )

    available_games = set(working["game_id"].dropna().astype(str))
    diagnostics = {
        "rows": int(len(working)),
        "games": int(len(available_games)),
        "unattributed_dropbacks": int(
            (dropback & working["_dropback_player_id"].eq("")).sum()
        ),
        "qb_kneel_rows": int(working["qb_kneel"].eq(1.0).sum()),
    }
    return (
        player_frame[[*player_keys, *RICH_COUNT_COLUMNS]].copy(),
        team_frame,
        available_games,
        diagnostics,
    )


def attach_snaps(base: pd.DataFrame, snaps: pd.DataFrame) -> pd.DataFrame:
    # Team is part of the attachment key.
    #
    # Snap files identify players primarily through PFR ID/name and are
    # canonicalized through the Prop Engine crosswalk. A bad or ambiguous
    # historical identity mapping must never transfer another team's snap
    # measurements onto a player merely because game_id + GSIS ID happens
    # to collide.
    #
    # Example audited collision:
    #   2023_03_LA_CIN
    #   CIN OT Jonah Williams, PFR WillJo10
    #   incorrectly resolves through the crosswalk to GSIS 00-0035944,
    #   which belongs to LA DE Jonah Williams in the player-stat source.
    #
    # Requiring team makes that source row safely unmatched rather than
    # attaching CIN offensive snaps to the LA defender.
    common.ensure_unique(
        base,
        [
            *GRAIN,
            "team",
        ],
        "player-stat snap attachment base",
    )

    common.ensure_unique(
        snaps,
        [
            *GRAIN,
            "_snap_team",
        ],
        "resolved snap attachment source",
    )

    return base.merge(
        snaps,
        how="left",
        left_on=[
            *GRAIN,
            "team",
        ],
        right_on=[
            *GRAIN,
            "_snap_team",
        ],
        validate="one_to_one",
    )


def attach_participation(
    base: pd.DataFrame,
    players: pd.DataFrame,
    teams: pd.DataFrame,
) -> pd.DataFrame:
    output = base.merge(
        players, on=[*GRAIN, "team"], how="left", validate="one_to_one"
    )
    output = output.merge(
        teams, on=TEAM_GRAIN, how="left", validate="many_to_one"
    )
    off_available = pd.to_numeric(
        output["_offense_participation_available"], errors="coerce"
    ).fillna(0).eq(1)
    def_available = pd.to_numeric(
        output["_defense_participation_available"], errors="coerce"
    ).fillna(0).eq(1)

    output.loc[
        off_available & output["offense_participation"].isna(),
        "offense_participation",
    ] = 0.0
    output.loc[
        def_available & output["defense_participation"].isna(),
        "defense_participation",
    ] = 0.0
    output.loc[
        def_available & output["_participation_defense_plays"].isna(),
        "_participation_defense_plays",
    ] = 0.0
    return output


def attach_rich_pbp(
    base: pd.DataFrame,
    players: pd.DataFrame,
    teams: pd.DataFrame,
    *,
    available_games: set[str],
) -> pd.DataFrame:
    output = base.merge(
        players, on=[*GRAIN, "team"], how="left", validate="one_to_one"
    )
    output = output.merge(
        teams, on=TEAM_GRAIN, how="left", validate="many_to_one"
    )
    available = output["game_id"].isin(available_games)
    for column in RICH_COUNT_COLUMNS:
        output.loc[available & output[column].isna(), column] = 0.0
    return output


def validate_output(
    output: pd.DataFrame,
    *,
    config: dict,
    rich_feature_start: int,
) -> None:
    if list(output.columns) != OUTPUT_COLUMNS:
        raise ValueError(
            "Player opportunity headers/order do not match contract. "
            f"Got={list(output.columns)}"
        )

    common.ensure_unique(output, GRAIN, "player weekly opportunity grain")
    common.reject_forbidden_feature_columns(output.columns, config)

    for column in CORE_NONNEGATIVE_COLUMNS + RICH_COUNT_COLUMNS:
        values = pd.to_numeric(output[column], errors="coerce")
        negative = values.notna() & values.lt(0.0)
        if negative.any():
            sample = output.loc[
                negative, [*GRAIN, column]
            ].head(10).to_dict(orient="records")
            raise ValueError(
                f"Negative count/opportunity value in {column!r}. Sample={sample}"
            )

    for column in PERCENT_COLUMNS:
        values = pd.to_numeric(output[column], errors="coerce")
        invalid = values.notna() & ~values.between(0.0, 1.0)
        if invalid.any():
            sample = output.loc[
                invalid, [*GRAIN, column]
            ].head(10).to_dict(orient="records")
            raise ValueError(f"{column} outside [0, 1]. Sample={sample}")

    for column in OUTPUT_COLUMNS:
        if column in {"game_id", "player_id", "team", "position", "position_group"}:
            continue
        values = pd.to_numeric(output[column], errors="coerce")
        infinite = values.map(
            lambda value: False if pd.isna(value) else math.isinf(float(value))
        )
        if infinite.any():
            raise ValueError(f"Infinite values found in {column!r}.")

    pre_rich = output["season"].lt(rich_feature_start)
    rich_columns = [
        *RICH_COUNT_COLUMNS,
        "red_zone_target_share",
        "red_zone_carry_share",
        "goal_line_carry_share",
    ]
    for column in rich_columns:
        if output.loc[pre_rich, column].notna().any():
            raise ValueError(
                f"{column} must remain null before rich_feature_start={rich_feature_start}."
            )

    pre_participation = output["season"].lt(2016)
    for column in ["offense_participation", "defense_participation"]:
        if output.loc[pre_participation, column].notna().any():
            raise ValueError(f"{column} must remain null before 2016.")

    season_2012 = output["season"].eq(2012)
    for column in [
        "offense_snap_pct", "defense_snap_pct",
        "tackle_rate_per_def_play", "sack_rate_per_def_play",
        "qb_hit_rate_per_def_play",
    ]:
        if output.loc[season_2012, column].notna().any():
            raise ValueError(
                f"{column} must remain null in 2012 because snap_counts_2012 is empty."
            )


def run() -> dict[str, Any]:
    config = common.load_config()
    repo = common.repo_root()
    start_season = int(config["seasons"]["historical_start"])
    end_season = int(config["seasons"]["historical_end"])
    rich_feature_start = int(config["seasons"]["rich_feature_start"])

    crosswalk_path = repo / config["paths"]["identity_crosswalk"]
    output_path = repo / config["paths"]["player_opportunity"]

    crosswalk = common.read_parquet_required(crosswalk_path)
    pfr_map, name_map = build_crosswalk_maps(crosswalk)

    frames = []
    diagnostics: dict[str, dict[str, Any]] = {}

    for season in range(start_season, end_season + 1):
        stats_path = repo / config["paths"]["historical_player_stats_pattern"].format(
            season=season
        )
        snap_path = repo / config["paths"]["historical_snaps_pattern"].format(
            season=season
        )
        part_path = repo / config["paths"]["historical_participation_pattern"].format(
            season=season
        )
        pbp_path = repo / config["paths"]["pbp_pattern"].format(season=season)

        stats_source = common.read_parquet_required(stats_path)
        base, team_denominators, stats_diag = prepare_stats(
            stats_source, season=season, path=stats_path
        )
        base = base.merge(
            team_denominators, on=TEAM_GRAIN, how="left", validate="many_to_one"
        )

        base["target_share"] = safe_divide(base["targets"], base["_team_targets"])
        base["air_yards_share"] = safe_divide(
            base["receiving_air_yards"], base["_team_receiving_air_yards"]
        )

        snap_source = common.read_parquet_required(snap_path)
        snaps, snap_diag = prepare_snaps(
            snap_source,
            season=season,
            path=snap_path,
            pfr_map=pfr_map,
            name_map=name_map,
        )
        base = attach_snaps(base, snaps)

        if part_path.is_file():
            part_source = common.read_parquet_required(part_path)
            part_players, part_teams, part_diag = build_participation(
                part_source, season=season, path=part_path
            )
            base = attach_participation(base, part_players, part_teams)
        else:
            base["offense_participation"] = float("nan")
            base["defense_participation"] = float("nan")
            base["_participation_defense_plays"] = float("nan")
            part_diag = {"status": "unavailable"}

        if season >= rich_feature_start:
            if not pbp_path.is_file():
                raise FileNotFoundError(
                    f"Rich-feature season requires PBP but file is missing: {pbp_path}"
                )
            required_pbp_columns = [
                "season_type", "week", "game_id", "posteam", "yardline_100",
                "pass_attempt", "qb_dropback", "rush_attempt", "qb_kneel",
                "passer_player_id", "rusher_player_id", "receiver_player_id",
            ]
            pbp_source = pd.read_csv(
                pbp_path, usecols=required_pbp_columns, low_memory=False
            )
            rich_players, rich_teams, available_games, pbp_diag = build_pbp_rich(
                pbp_source, season=season, path=pbp_path
            )
            base_games = set(base["game_id"].dropna().astype(str))
            missing_games = sorted(base_games - available_games)
            if missing_games:
                raise ValueError(
                    f"{pbp_path}: PBP missing regular-season player-stat game(s). "
                    f"Sample={missing_games[:10]}"
                )
            base = attach_rich_pbp(
                base, rich_players, rich_teams, available_games=available_games
            )
        else:
            for column in RICH_COUNT_COLUMNS:
                base[column] = float("nan")
            base["_team_red_zone_targets"] = float("nan")
            base["_team_red_zone_carries"] = float("nan")
            base["_team_goal_line_carries"] = float("nan")
            base["_team_rush_attempts_pbp"] = float("nan")
            pbp_diag = {"status": f"unavailable_before_{rich_feature_start}"}

        # Exact carry-share denominator uses PBP team rush_attempt when PBP is
        # available. Before rich PBP coverage, use the player-stat team carry
        # total as the best verified historical fallback.
        team_rush_attempts = base["_team_rush_attempts_pbp"].where(
            base["_team_rush_attempts_pbp"].notna(),
            base["_team_rush_attempts_stats"],
        )
        base["carry_share"] = safe_divide(base["carries"], team_rush_attempts)

        base["red_zone_target_share"] = safe_divide(
            base["red_zone_targets"], base["_team_red_zone_targets"]
        )
        base["red_zone_carry_share"] = safe_divide(
            base["red_zone_carries"], base["_team_red_zone_carries"]
        )
        base["goal_line_carry_share"] = safe_divide(
            base["goal_line_carries"], base["_team_goal_line_carries"]
        )

        base["yards_per_attempt"] = safe_divide(
            base["passing_yards"], base["pass_attempts"]
        )
        base["yards_per_carry"] = safe_divide(
            base["rushing_yards"], base["carries"]
        )
        base["yards_per_target"] = safe_divide(
            base["receiving_yards"], base["targets"]
        )
        base["catch_rate"] = safe_divide(base["receptions"], base["targets"])
        base["passing_td_rate"] = safe_divide(
            base["passing_tds"], base["pass_attempts"]
        )
        base["rushing_td_rate"] = safe_divide(
            base["rushing_tds"], base["carries"]
        )
        base["receiving_td_rate"] = safe_divide(
            base["receiving_tds"], base["targets"]
        )

        defense_plays = base["_defense_snaps"].where(
            base["_defense_snaps"].notna(),
            base["_participation_defense_plays"],
        )
        base["tackle_rate_per_def_play"] = safe_divide(
            base["tackles"], defense_plays
        )
        base["sack_rate_per_def_play"] = safe_divide(
            base["sacks"], defense_plays
        )
        base["qb_hit_rate_per_def_play"] = safe_divide(
            base["qb_hits"], defense_plays
        )

        season_output = base[OUTPUT_COLUMNS].copy()
        common.ensure_unique(
            season_output, GRAIN, f"player opportunity {season} grain"
        )
        frames.append(season_output)
        diagnostics[str(season)] = {
            "stats": stats_diag,
            "snaps": snap_diag,
            "participation": part_diag,
            "pbp": pbp_diag,
            "rows": int(len(season_output)),
        }

    output = pd.concat(frames, ignore_index=True)[OUTPUT_COLUMNS].copy()
    validate_output(
        output,
        config=config,
        rich_feature_start=rich_feature_start,
    )
    common.write_parquet_atomic(output, output_path)

    payload = {
        "status": "passed",
        "historical_start": start_season,
        "historical_end": end_season,
        "rich_feature_start": rich_feature_start,
        "rows": int(len(output)),
        "unique_players": int(output["player_id"].nunique()),
        "games": int(output["game_id"].nunique()),
        "carry_denominator_policy": (
            "2021-2025: PBP team rush_attempt count; 2012-2020: sum of "
            "player-stat carries as verified fallback because PBP is unavailable. "
            "qb_kneel is not excluded; audited player-stat carry totals reconcile "
            "materially better to all rush_attempt rows than to rush_attempt - qb_kneel"
        ),
        "kneel_policy": (
            "verified PBP field qb_kneel exists for 2021-2025 but is not excluded"
        ),
        "red_zone_definition": "yardline_100 <= 20",
        "inside_10_definition": "yardline_100 <= 10",
        "goal_line_definition": "yardline_100 <= 5",
        "zero_denominator_policy": "null rate/share; never infinity",
        "defensive_rate_denominator": (
            "resolved official defense_snaps; fallback to counted defensive "
            "participation plays when snap mapping is unavailable"
        ),
        "same_week_policy": (
            "raw realized measurements only; downstream model builders must lag"
        ),
        "source_availability": {
            "player_stats": "2012-2025",
            "snap_counts": "2013-2025; 2012 file empty",
            "participation": "2016-2025",
            "pbp_rich": f"{rich_feature_start}-{end_season}",
        },
        "season_diagnostics": diagnostics,
        "output": str(output_path.relative_to(repo)),
    }

    common.log_run("build_player_opportunity.py", payload)
    return payload


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
