#!/usr/bin/env python3
"""
Build leakage-safe historical player rolling-form features.

READS:
    docs/win/football/nfl/prop_engine/config/prop_engine.yaml
    configured historical_universe
    configured player_opportunity
    2010-2011 historical player-stat parquet files for pre-2012 prior seeding

WRITES:
    docs/win/football/nfl/prop_engine/data/historical/features/player_form.parquet

CONTRACT:
    - Target grain is the historical universe: season + week + game_id + player_id.
    - Every player feature uses only source games with kickoff strictly before the
      target kickoff.
    - 2010-2011 regular-season player stats seed leakage-safe priors for metrics
      whose definitions are source-compatible with player_opportunity.
    - Metrics unavailable in the canonical source remain unavailable until their
      source begins; future values are never backfilled.
    - Career/non-share history survives team changes.
    - Team-share metrics reset to the current franchise stint after a team change.
    - Metric-specific no-history values fall back to strictly prior position priors.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import common


GRAIN = ["season", "week", "game_id", "player_id"]
BASE_ID_COLUMNS = [
    "season", "week", "game_id", "player_id", "team", "position", "position_group",
]

BASE_METRICS = [
    "pass_attempts",
    "dropbacks",
    "completions",
    "passing_yards",
    "passing_tds",
    "yards_per_attempt",
    "passing_td_rate",
    "passing_air_yards",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "yards_per_carry",
    "carry_share",
    "red_zone_carries",
    "goal_line_carries",
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "yards_per_target",
    "catch_rate",
    "target_share",
    "air_yards_share",
    "red_zone_targets",
    "red_zone_target_share",
    "field_goal_attempts",
    "field_goals_made",
    "extra_point_attempts",
    "extra_points_made",
    "tackles",
    "sacks",
    "qb_hits",
    "tackle_rate_per_def_play",
    "sack_rate_per_def_play",
    "qb_hit_rate_per_def_play",
    "offense_snap_pct",
    "defense_snap_pct",
    "offense_participation",
    "defense_participation",
]

FEATURE_SUFFIXES = [
    "lag1",
    "roll3_mean",
    "roll5_mean",
    "roll8_mean",
    "roll3_median",
    "roll5_std",
    "ewm3",
    "ewm5",
    "season_to_date",
    "career_prior",
]

TEAM_SHARE_METRICS = {
    "carry_share",
    "target_share",
    "air_yards_share",
    "red_zone_target_share",
}

# Preserve historical output/team labels, but treat relocated franchises as the
# same franchise for team-history/stint logic.
TEAM_HISTORY_ALIASES = {
    "SD": "LAC",
    "OAK": "LV",
    "STL": "LAR",
}

PREHISTORY_SEASONS = (2010, 2011)

# Only metrics whose 2010-2011 player-stat definitions match the canonical
# opportunity definitions are seeded here. PBP-only/participation/snap metrics
# intentionally remain null until their canonical source begins.
PREHISTORY_DIRECT_MAP = {
    "pass_attempts": "attempts",
    "completions": "completions",
    "passing_yards": "passing_yards",
    "passing_tds": "passing_tds",
    "passing_air_yards": "passing_air_yards",
    "carries": "carries",
    "rushing_yards": "rushing_yards",
    "rushing_tds": "rushing_tds",
    "targets": "targets",
    "receptions": "receptions",
    "receiving_yards": "receiving_yards",
    "receiving_tds": "receiving_tds",
    "field_goal_attempts": "fg_att",
    "field_goals_made": "fg_made",
    "extra_point_attempts": "pat_att",
    "extra_points_made": "pat_made",
    "sacks": "def_sacks",
    "qb_hits": "def_qb_hits",
}

NUMERIC_COUNT_LIKE = {
    "pass_attempts",
    "completions",
    "passing_tds",
    "carries",
    "rushing_tds",
    "targets",
    "receptions",
    "receiving_tds",
    "field_goal_attempts",
    "field_goals_made",
    "extra_point_attempts",
    "extra_points_made",
    "tackles",
    "sacks",
    "qb_hits",
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
    if text.casefold() in {"", "nan", "none", "null", "<na>", "nat"}:
        return ""
    return text


def normalize_team(value: Any) -> str:
    return common.normalize_team(value)


def history_team(value: Any) -> str:
    team = normalize_team(value)
    return TEAM_HISTORY_ALIASES.get(team, team)


def normalize_position(value: Any) -> str:
    return clean(value).upper()


def normalize_position_group(value: Any, position: Any = "") -> str:
    group = clean(value).upper()
    if group:
        return group
    return normalize_position(position)


def numeric(series: pd.Series, *, fill_zero: bool = False) -> pd.Series:
    result = pd.to_numeric(series, errors="coerce").astype("float64")
    result = result.replace([np.inf, -np.inf], np.nan)
    if fill_zero:
        result = result.fillna(0.0)
    return result


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = numeric(numerator)
    den = numeric(denominator)
    result = pd.Series(np.nan, index=num.index, dtype="float64")
    valid = num.notna() & den.notna() & den.ne(0.0)
    result.loc[valid] = num.loc[valid] / den.loc[valid]
    return result


def feature_columns() -> list[str]:
    return [
        f"{metric}_{suffix}"
        for metric in BASE_METRICS
        for suffix in FEATURE_SUFFIXES
    ]


def validate_config(config: dict) -> None:
    windows = [int(v) for v in config.get("rolling_windows", {}).get("games", [])]
    spans = [int(v) for v in config.get("ewm", {}).get("spans", [])]
    adjust = config.get("ewm", {}).get("adjust")

    for required in (1, 3, 5, 8):
        if required not in windows:
            raise ValueError(
                f"rolling_windows.games must contain {required}; found {windows}"
            )

    for required in (3, 5):
        if required not in spans:
            raise ValueError(
                f"ewm.spans must contain {required}; found {spans}"
            )

    if adjust is not False:
        raise ValueError("ewm.adjust must be false for Issue 12.")


def load_targets(config: dict) -> pd.DataFrame:
    path = config["paths"]["historical_universe"]
    required = BASE_ID_COLUMNS + ["kickoff_timestamp"]
    target = common.read_parquet_required(path, required)

    start = int(config["seasons"]["historical_start"])
    end = int(config["seasons"]["historical_end"])

    target = target.loc[
        pd.to_numeric(target["season"], errors="coerce").between(start, end)
    ].copy()

    target["season"] = pd.to_numeric(target["season"], errors="raise").astype(int)
    target["week"] = pd.to_numeric(target["week"], errors="raise").astype(int)
    target["game_id"] = target["game_id"].map(clean)
    target["player_id"] = target["player_id"].map(common.normalize_player_id)
    target["team"] = target["team"].map(normalize_team)
    target["position"] = target["position"].map(normalize_position)
    target["position_group"] = [
        normalize_position_group(g, p)
        for g, p in zip(target["position_group"], target["position"])
    ]
    target["_history_team"] = target["team"].map(history_team)
    target["_kickoff"] = pd.to_datetime(
        target["kickoff_timestamp"], utc=True, errors="raise"
    )

    if target["player_id"].eq("").any():
        raise ValueError("Historical universe contains blank player_id.")
    if target["game_id"].eq("").any():
        raise ValueError("Historical universe contains blank game_id.")
    if target["_kickoff"].isna().any():
        raise ValueError("Historical universe contains invalid kickoff_timestamp.")

    common.ensure_unique(target, GRAIN, "Issue 12 historical universe")

    target = target.sort_values(
        ["season", "week", "game_id", "player_id", "team", "position"],
        kind="mergesort",
    ).reset_index(drop=True)
    target["_row_id"] = np.arange(len(target), dtype=np.int64)
    return target


def load_opportunity_source(config: dict, target: pd.DataFrame) -> pd.DataFrame:
    path = config["paths"]["player_opportunity"]
    required = BASE_ID_COLUMNS + BASE_METRICS
    source = common.read_parquet_required(path, required).copy()

    source["season"] = pd.to_numeric(source["season"], errors="raise").astype(int)
    source["week"] = pd.to_numeric(source["week"], errors="raise").astype(int)
    source["game_id"] = source["game_id"].map(clean)
    source["player_id"] = source["player_id"].map(common.normalize_player_id)
    source["team"] = source["team"].map(normalize_team)
    source["position"] = source["position"].map(normalize_position)
    source["position_group"] = [
        normalize_position_group(g, p)
        for g, p in zip(source["position_group"], source["position"])
    ]

    kickoff = (
        target[["season", "week", "game_id", "_kickoff"]]
        .drop_duplicates(["season", "week", "game_id"])
    )
    source = source.merge(
        kickoff,
        on=["season", "week", "game_id"],
        how="left",
        validate="many_to_one",
    )

    if source["_kickoff"].isna().any():
        sample = source.loc[source["_kickoff"].isna(), GRAIN].head(10)
        raise ValueError(
            "player_opportunity contains game(s) without universe kickoff. Sample="
            f"{sample.to_dict(orient='records')}"
        )

    for metric in BASE_METRICS:
        source[metric] = numeric(source[metric])

    source["_history_team"] = source["team"].map(history_team)
    source["_prehistory"] = 0
    return source


def load_prehistory_source(config: dict) -> pd.DataFrame:
    pattern = config["paths"]["historical_player_stats_pattern"]
    frames: list[pd.DataFrame] = []

    needed = [
        "season", "week", "season_type", "game_id", "player_id", "team",
        "position", "position_group",
        *sorted(set(PREHISTORY_DIRECT_MAP.values())),
        "receiving_air_yards", "def_tackles_solo", "def_tackle_assists",
    ]

    for season in PREHISTORY_SEASONS:
        path = pattern.format(season=season)
        raw = common.read_parquet_required(path, needed)
        raw = raw.loc[
            pd.to_numeric(raw["season"], errors="coerce").eq(season)
            & raw["season_type"].astype(str).str.upper().eq("REG")
        ].copy()

        if raw.empty:
            raise RuntimeError(f"No regular-season prehistory rows for {season}: {path}")

        raw["season"] = season
        raw["week"] = pd.to_numeric(raw["week"], errors="raise").astype(int)
        raw["game_id"] = raw["game_id"].map(clean)
        raw["player_id"] = raw["player_id"].map(common.normalize_player_id)
        raw["team"] = raw["team"].map(normalize_team)
        raw["position"] = raw["position"].map(normalize_position)
        raw["position_group"] = [
            normalize_position_group(g, p)
            for g, p in zip(raw["position_group"], raw["position"])
        ]

        # Team denominators are built before blank player IDs are removed so that
        # unresolved player rows remain part of the team opportunity total.
        for c in set(PREHISTORY_DIRECT_MAP.values()) | {
            "receiving_air_yards", "def_tackles_solo", "def_tackle_assists"
        }:
            raw[c] = numeric(raw[c], fill_zero=True)

        team_grain = ["season", "week", "game_id", "team"]
        denominators = (
            raw.groupby(team_grain, as_index=False, dropna=False)
            .agg(
                _team_carries=("carries", "sum"),
                _team_targets=("targets", "sum"),
                _team_receiving_air_yards=("receiving_air_yards", "sum"),
            )
        )
        raw = raw.merge(
            denominators,
            on=team_grain,
            how="left",
            validate="many_to_one",
        )

        frame = raw[
            ["season", "week", "game_id", "player_id", "team", "position", "position_group"]
        ].copy()

        for metric in BASE_METRICS:
            frame[metric] = np.nan

        for metric, source_column in PREHISTORY_DIRECT_MAP.items():
            frame[metric] = raw[source_column].astype("float64")

        frame["tackles"] = (
            raw["def_tackles_solo"].astype("float64")
            + raw["def_tackle_assists"].astype("float64")
        )
        frame["yards_per_attempt"] = safe_divide(raw["passing_yards"], raw["attempts"])
        frame["passing_td_rate"] = safe_divide(raw["passing_tds"], raw["attempts"])
        frame["yards_per_carry"] = safe_divide(raw["rushing_yards"], raw["carries"])
        frame["yards_per_target"] = safe_divide(raw["receiving_yards"], raw["targets"])
        frame["catch_rate"] = safe_divide(raw["receptions"], raw["targets"])
        frame["carry_share"] = safe_divide(raw["carries"], raw["_team_carries"])
        frame["target_share"] = safe_divide(raw["targets"], raw["_team_targets"])
        frame["air_yards_share"] = safe_divide(
            raw["receiving_air_yards"], raw["_team_receiving_air_yards"]
        )

        # Synthetic ordering is sufficient because every prehistory row precedes
        # the 2012 target universe. Season/week ordering preserves player rolling
        # sequence without inventing target-time information.
        frame["_kickoff"] = (
            pd.Timestamp(f"{season}-01-01", tz="UTC")
            + pd.to_timedelta(frame["week"].astype(int) * 7, unit="D")
        )
        frame["_history_team"] = frame["team"].map(history_team)
        frame["_prehistory"] = 1
        frame = frame.loc[frame["player_id"].ne("")].copy()
        frames.append(frame)

    pre = pd.concat(frames, ignore_index=True, sort=False)
    common.ensure_unique(pre, GRAIN, "Issue 12 pre-2012 player history")
    return pre


def prepare_source(config: dict, target: pd.DataFrame) -> pd.DataFrame:
    opportunity = load_opportunity_source(config, target)
    prehistory = load_prehistory_source(config)

    columns = BASE_ID_COLUMNS + BASE_METRICS + [
        "_kickoff", "_history_team", "_prehistory"
    ]
    source = pd.concat(
        [prehistory[columns], opportunity[columns]],
        ignore_index=True,
        sort=False,
    )

    source = source.loc[source["player_id"].ne("")].copy()
    source = source.sort_values(
        ["player_id", "_kickoff", "season", "week", "game_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    if source.duplicated(GRAIN).any():
        sample = source.loc[source.duplicated(GRAIN, keep=False), GRAIN].head(20)
        raise ValueError(
            "Combined Issue 12 source has duplicate canonical grain. Sample="
            f"{sample.to_dict(orient='records')}"
        )

    previous_team = source.groupby("player_id", sort=False)["_history_team"].shift(1)
    new_stint = previous_team.isna() | source["_history_team"].ne(previous_team)
    source["_stint_id"] = (
        new_stint.astype("int64")
        .groupby(source["player_id"], sort=False)
        .cumsum()
        .astype("int64")
    )
    source["_source_history_games"] = (
        source.groupby("player_id", sort=False).cumcount() + 1
    ).astype("int64")

    return source


def _asof_sort(df: pd.DataFrame, by: Iterable[str]) -> pd.DataFrame:
    # merge_asof requires global sorting by the as-of key. Secondary by-columns
    # make equal-time ordering deterministic without allowing equal-time matches.
    return df.sort_values(["_kickoff", *list(by)], kind="mergesort")


def attach_player_history_meta(target: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    left = target[
        ["_row_id", "player_id", "_kickoff", "_history_team"]
    ].copy()
    right = source[
        [
            "player_id", "_kickoff", "_history_team", "_stint_id",
            "_source_history_games",
        ]
    ].copy()
    right = right.rename(columns={"_history_team": "_prior_history_team"})

    merged = pd.merge_asof(
        _asof_sort(left, ["player_id"]),
        _asof_sort(right, ["player_id"]),
        on="_kickoff",
        by="player_id",
        direction="backward",
        allow_exact_matches=False,
    ).sort_values("_row_id", kind="mergesort")

    history_games = merged["_source_history_games"].fillna(0).astype("int32")
    no_history = history_games.eq(0).astype("int8")
    prior_team = merged["_prior_history_team"].fillna("")
    new_team = (
        history_games.gt(0)
        & merged["_history_team"].fillna("").ne(prior_team)
    ).astype("int8")

    target = target.copy()
    target["history_games"] = history_games.to_numpy()
    target["no_nfl_history_flag"] = no_history.to_numpy()
    target["new_team_flag"] = new_team.to_numpy()
    target["_target_stint_id"] = np.where(
        (history_games.gt(0) & new_team.eq(0)).to_numpy(),
        merged["_stint_id"].fillna(-1).astype("int64").to_numpy(),
        -1,
    ).astype("int64")
    return target


def grouped_rolling(
    valid: pd.DataFrame,
    group_keys: list[str],
    metric: str,
    window: int,
    operation: str,
) -> pd.Series:
    rolled = valid.groupby(group_keys, sort=False)[metric].rolling(
        window=window,
        min_periods=1,
    )

    if operation == "mean":
        values = rolled.mean()
    elif operation == "median":
        values = rolled.median()
    elif operation == "std0":
        values = rolled.std(ddof=0)
    else:
        raise ValueError(f"Unsupported rolling operation: {operation}")

    values = values.reset_index(
        level=list(range(len(group_keys))),
        drop=True,
    )
    return values.reindex(valid.index)


def grouped_ewm(
    valid: pd.DataFrame,
    group_keys: list[str],
    metric: str,
    span: int,
) -> pd.Series:
    values = (
        valid.groupby(group_keys, sort=False)[metric]
        .ewm(span=span, adjust=False)
        .mean()
        .reset_index(level=list(range(len(group_keys))), drop=True)
    )
    return values.reindex(valid.index)


def build_metric_state(source: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, list[str]]:
    valid = source.loc[source[metric].notna(), [
        "season", "player_id", "position_group", "_kickoff", "_stint_id", metric
    ]].copy()

    names = [f"{metric}_{suffix}" for suffix in FEATURE_SUFFIXES]
    if valid.empty:
        return valid, names

    valid = valid.sort_values(
        ["player_id", "_kickoff", "season", "_stint_id"],
        kind="mergesort",
    )

    group_keys = ["player_id", "_stint_id"] if metric in TEAM_SHARE_METRICS else ["player_id"]

    valid[names[0]] = valid[metric].astype("float64")
    valid[names[1]] = grouped_rolling(valid, group_keys, metric, 3, "mean")
    valid[names[2]] = grouped_rolling(valid, group_keys, metric, 5, "mean")
    valid[names[3]] = grouped_rolling(valid, group_keys, metric, 8, "mean")
    valid[names[4]] = grouped_rolling(valid, group_keys, metric, 3, "median")
    valid[names[5]] = grouped_rolling(valid, group_keys, metric, 5, "std0")
    valid[names[6]] = grouped_ewm(valid, group_keys, metric, 3)
    valid[names[7]] = grouped_ewm(valid, group_keys, metric, 5)

    season_keys = [*group_keys, "season"]
    season_cumsum = valid.groupby(season_keys, sort=False)[metric].cumsum()
    season_count = valid.groupby(season_keys, sort=False).cumcount() + 1
    valid[names[8]] = season_cumsum / season_count

    career_cumsum = valid.groupby(group_keys, sort=False)[metric].cumsum()
    career_count = valid.groupby(group_keys, sort=False).cumcount() + 1
    valid[names[9]] = career_cumsum / career_count

    return valid, names


def build_position_prior_state(valid: pd.DataFrame, metric: str) -> pd.DataFrame:
    if valid.empty:
        return pd.DataFrame(
            columns=["position_group", "_kickoff", "_position_prior_mean", "_position_prior_std"]
        )

    x = valid.loc[valid["position_group"].ne(""), [
        "position_group", "_kickoff", metric
    ]].copy()
    if x.empty:
        return pd.DataFrame(
            columns=["position_group", "_kickoff", "_position_prior_mean", "_position_prior_std"]
        )

    x["_square"] = x[metric].astype("float64") ** 2
    agg = (
        x.groupby(["position_group", "_kickoff"], as_index=False, sort=True)
        .agg(
            _sum=(metric, "sum"),
            _count=(metric, "count"),
            _sumsq=("_square", "sum"),
        )
        .sort_values(["position_group", "_kickoff"], kind="mergesort")
    )

    agg["_cum_sum"] = agg.groupby("position_group", sort=False)["_sum"].cumsum()
    agg["_cum_count"] = agg.groupby("position_group", sort=False)["_count"].cumsum()
    agg["_cum_sumsq"] = agg.groupby("position_group", sort=False)["_sumsq"].cumsum()
    agg["_position_prior_mean"] = agg["_cum_sum"] / agg["_cum_count"]
    variance = (
        agg["_cum_sumsq"] / agg["_cum_count"]
        - agg["_position_prior_mean"] ** 2
    ).clip(lower=0.0)
    agg["_position_prior_std"] = np.sqrt(variance)
    return agg[[
        "position_group", "_kickoff", "_position_prior_mean", "_position_prior_std"
    ]]


def metric_features_for_targets(
    target: pd.DataFrame,
    source: pd.DataFrame,
    metric: str,
) -> np.ndarray:
    state, names = build_metric_state(source, metric)
    n = len(target)
    result = np.full((n, len(names)), np.nan, dtype="float64")

    # Strictly prior position prior for fallback.
    prior_state = build_position_prior_state(state, metric)
    if prior_state.empty:
        prior_mean = np.full(n, np.nan, dtype="float64")
        prior_std = np.full(n, np.nan, dtype="float64")
    else:
        left_prior = target[["_row_id", "position_group", "_kickoff"]].copy()
        prior_join = pd.merge_asof(
            _asof_sort(left_prior, ["position_group"]),
            _asof_sort(prior_state, ["position_group"]),
            on="_kickoff",
            by="position_group",
            direction="backward",
            allow_exact_matches=False,
        ).sort_values("_row_id", kind="mergesort")
        prior_mean = prior_join["_position_prior_mean"].to_numpy(dtype="float64")
        prior_std = prior_join["_position_prior_std"].to_numpy(dtype="float64")

    if state.empty:
        # No player source exists for the metric yet. Position prior is also empty
        # in this case, so the block correctly remains NaN.
        return result

    if metric in TEAM_SHARE_METRICS:
        left = target[
            ["_row_id", "player_id", "_target_stint_id", "_kickoff", "season"]
        ].rename(columns={"_target_stint_id": "_stint_id"})
        right = state[
            ["player_id", "_stint_id", "_kickoff", "season", *names]
        ].copy()
        by = ["player_id", "_stint_id"]
    else:
        left = target[["_row_id", "player_id", "_kickoff", "season"]].copy()
        right = state[["player_id", "_kickoff", "season", *names]].copy()
        by = ["player_id"]

    right = right.rename(columns={"season": "_state_season"})
    joined = pd.merge_asof(
        _asof_sort(left, by),
        _asof_sort(right, by),
        on="_kickoff",
        by=by,
        direction="backward",
        allow_exact_matches=False,
    ).sort_values("_row_id", kind="mergesort")

    player_history_exists = joined[names[0]].notna().to_numpy()

    for j, name in enumerate(names):
        values = joined[name].to_numpy(dtype="float64")

        # season_to_date is only valid if the latest metric observation belongs
        # to the target season. Do not reuse a prior-season season mean.
        if name.endswith("_season_to_date"):
            same_season = (
                joined["_state_season"].notna()
                & joined["season"].eq(joined["_state_season"])
            ).to_numpy()
            values = np.where(same_season, values, np.nan)

        # Position priors apply only when there is no player history for this
        # metric/current team-share stint. Established-player season_to_date
        # remains NaN before that player's first observation of the season.
        fallback = prior_std if name.endswith("_roll5_std") else prior_mean
        values = np.where(~player_history_exists, fallback, values)
        result[:, j] = values

    return result


def validate_output(
    target: pd.DataFrame,
    features: np.ndarray,
    columns: list[str],
) -> None:
    if features.shape != (len(target), len(columns)):
        raise ValueError(
            f"Feature matrix shape {features.shape} != {(len(target), len(columns))}"
        )

    if np.isinf(features).any():
        raise ValueError("Issue 12 feature matrix contains infinity.")

    if not target["no_nfl_history_flag"].isin([0, 1]).all():
        raise ValueError("no_nfl_history_flag must be binary.")
    if not target["new_team_flag"].isin([0, 1]).all():
        raise ValueError("new_team_flag must be binary.")
    if (target["history_games"] < 0).any():
        raise ValueError("history_games must be nonnegative.")

    common.reject_forbidden_feature_columns(columns, common.load_config())


def write_output_atomic(
    target: pd.DataFrame,
    features: np.ndarray,
    feature_names: list[str],
    output_path: Path,
) -> None:
    destination = output_path.resolve()
    prop_root = common.prop_root().resolve()
    try:
        destination.relative_to(prop_root)
    except ValueError as exc:
        raise ValueError(
            f"Issue 12 output must remain under Prop Engine root: {destination}"
        ) from exc

    destination.parent.mkdir(parents=True, exist_ok=True)

    base = target[BASE_ID_COLUMNS].copy()
    base["no_nfl_history_flag"] = target["no_nfl_history_flag"].astype("int8")
    base["new_team_flag"] = target["new_team_flag"].astype("int8")
    base["history_games"] = target["history_games"].astype("int32")

    feature_df = pd.DataFrame(
        features.astype("float32", copy=False),
        columns=feature_names,
        index=base.index,
    )
    output = pd.concat([base, feature_df], axis=1, copy=False)

    expected = BASE_ID_COLUMNS + [
        "no_nfl_history_flag", "new_team_flag", "history_games"
    ] + feature_names
    if list(output.columns) != expected:
        raise ValueError("Issue 12 output column order mismatch before write.")
    common.ensure_unique(output, GRAIN, "Issue 12 player form output")

    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    )
    temp_path = Path(handle.name)
    handle.close()

    try:
        output.to_parquet(temp_path, index=False)
        os.replace(temp_path, destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def main() -> None:
    config = common.load_config()
    validate_config(config)

    target = load_targets(config)
    source = prepare_source(config, target)
    target = attach_player_history_meta(target, source)

    names = feature_columns()
    matrix = np.full((len(target), len(names)), np.nan, dtype="float32")

    for metric_index, metric in enumerate(BASE_METRICS):
        block = metric_features_for_targets(target, source, metric)
        start = metric_index * len(FEATURE_SUFFIXES)
        end = start + len(FEATURE_SUFFIXES)
        matrix[:, start:end] = block.astype("float32")
        print(
            f"ISSUE12 metric {metric_index + 1:02d}/{len(BASE_METRICS):02d}: {metric}",
            flush=True,
        )

    validate_output(target, matrix, names)

    configured = config.get("paths", {}).get("player_form")
    if configured:
        output_path = common.repo_root() / configured
    else:
        output_path = (
            common.prop_root()
            / "data/historical/features/player_form.parquet"
        )

    write_output_atomic(target, matrix, names, output_path)

    nonnull_counts = {
        metric: int(
            np.isfinite(matrix[:, i * len(FEATURE_SUFFIXES)]).sum()
        )
        for i, metric in enumerate(BASE_METRICS)
    }

    common.log_run(
        "build_player_form.py",
        {
            "status": "passed",
            "output": str(output_path.relative_to(common.repo_root())),
            "rows": int(len(target)),
            "players": int(target["player_id"].nunique()),
            "games": int(target["game_id"].nunique()),
            "base_metrics": len(BASE_METRICS),
            "feature_columns": len(names),
            "total_columns": len(BASE_ID_COLUMNS) + 3 + len(names),
            "prehistory_seasons": list(PREHISTORY_SEASONS),
            "strict_prior_kickoff": True,
            "same_game_realized_forbidden": True,
            "position_prior_policy": (
                "metric-specific strictly prior expanding position mean; "
                "roll5_std uses strictly prior expanding position population std"
            ),
            "position_prior_seed_policy": (
                "2010-2011 source-compatible core metrics only; unavailable metrics "
                "remain null until canonical source begins"
            ),
            "share_reset_metrics": sorted(TEAM_SHARE_METRICS),
            "team_share_policy": "reset to current franchise stint after team change",
            "career_policy": "non-share career history survives team changes",
            "rolling_policy": "last N observed prior metric games",
            "roll5_std_ddof": 0,
            "ewm_adjust": False,
            "history_games_policy": (
                "count all strictly prior realized/source player-games including "
                "2010-2011 prehistory"
            ),
            "no_nfl_history_policy": "history_games == 0",
            "new_team_policy": (
                "prior NFL/source history exists and target franchise differs from "
                "latest strictly prior source-game franchise"
            ),
            "lag1_nonnull_by_metric": nonnull_counts,
        },
    )


if __name__ == "__main__":
    main()
