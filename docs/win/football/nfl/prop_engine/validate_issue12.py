from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import math

import numpy as np
import pandas as pd

NFL = Path("docs/win/football/nfl")

OUT = NFL / "prop_engine/data/historical/features/player_form.parquet"
UNI = NFL / "prop_engine/data/historical/universe/player_game_universe.parquet"
OPP = NFL / "prop_engine/data/historical/opportunity/player_week_opportunity.parquet"
STATS_PATTERN = NFL / "data/historic_data/player_stats/stats_player_week_{season}.parquet"

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

SUFFIXES = [
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

TEAM_ALIASES = {
    "WAS": "WSH",
    "LA": "LAR",
    "JAC": "JAX",
}

TEAM_HISTORY_ALIASES = {
    "SD": "LAC",
    "OAK": "LV",
    "STL": "LAR",
}

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


def fail(message: str) -> None:
    raise SystemExit("FAIL: " + message)


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


def norm_team(value: Any) -> str:
    team = clean(value).upper()
    return TEAM_ALIASES.get(team, team)


def history_team(value: Any) -> str:
    team = norm_team(value)
    return TEAM_HISTORY_ALIASES.get(team, team)


def norm_position(value: Any) -> str:
    return clean(value).upper()


def norm_group(group: Any, position: Any = "") -> str:
    g = clean(group).upper()
    return g if g else norm_position(position)


def numeric(series: pd.Series, fill_zero: bool = False) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").astype("float64")
    x = x.replace([np.inf, -np.inf], np.nan)
    return x.fillna(0.0) if fill_zero else x


def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    a = numeric(num)
    b = numeric(den)
    result = pd.Series(np.nan, index=a.index, dtype="float64")
    valid = a.notna() & b.notna() & b.ne(0.0)
    result.loc[valid] = a.loc[valid] / b.loc[valid]
    return result


def asof_sort(df: pd.DataFrame, by: Iterable[str]) -> pd.DataFrame:
    return df.sort_values(["_kickoff", *list(by)], kind="mergesort")


def expected_feature_columns() -> list[str]:
    return [
        f"{metric}_{suffix}"
        for metric in BASE_METRICS
        for suffix in SUFFIXES
    ]


def assert_numeric_same(
    actual: pd.Series,
    expected: pd.Series | np.ndarray,
    label: str,
    rtol: float = 2e-5,
    atol: float = 2e-5,
) -> None:
    a = pd.to_numeric(actual, errors="coerce").to_numpy(dtype="float64")
    e = np.asarray(expected, dtype="float64")

    ok = (
        (np.isnan(a) & np.isnan(e))
        |
        (
            np.isfinite(a)
            & np.isfinite(e)
            & np.isclose(a, e, rtol=rtol, atol=atol)
        )
    )

    if not np.all(ok):
        idx = np.flatnonzero(~ok)
        sample = idx[:10]
        details = [
            {
                "row": int(i),
                "actual": None if np.isnan(a[i]) else float(a[i]),
                "expected": None if np.isnan(e[i]) else float(e[i]),
            }
            for i in sample
        ]
        fail(f"{label}: {len(idx):,} mismatches; sample={details}")


def grouped_rolling(
    valid: pd.DataFrame,
    group_keys: list[str],
    metric: str,
    window: int,
    op: str,
) -> pd.Series:
    rolled = valid.groupby(group_keys, sort=False)[metric].rolling(
        window=window,
        min_periods=1,
    )

    if op == "mean":
        values = rolled.mean()
    elif op == "median":
        values = rolled.median()
    elif op == "std0":
        values = rolled.std(ddof=0)
    else:
        raise ValueError(op)

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
    return (
        valid.groupby(group_keys, sort=False)[metric]
        .ewm(span=span, adjust=False)
        .mean()
        .reset_index(level=list(range(len(group_keys))), drop=True)
        .reindex(valid.index)
    )


print("=" * 100)
print("ISSUE 12 ACCEPTANCE VALIDATION")
print("=" * 100)

out = pd.read_parquet(OUT)
uni = pd.read_parquet(UNI)
opp = pd.read_parquet(OPP)

feature_cols = expected_feature_columns()
expected_cols = BASE_ID_COLUMNS + [
    "no_nfl_history_flag",
    "new_team_flag",
    "history_games",
] + feature_cols

# ---------------------------------------------------------------------
# Contract / exact output grain.
# ---------------------------------------------------------------------

if list(out.columns) != expected_cols:
    missing = [c for c in expected_cols if c not in out.columns]
    extra = [c for c in out.columns if c not in expected_cols]
    fail(
        "exact header/order mismatch; "
        f"missing={missing[:20]} extra={extra[:20]}"
    )

if len(feature_cols) != 390:
    fail(f"expected 390 derived columns, got {len(feature_cols)}")

if len(out.columns) != 400:
    fail(f"expected 400 total columns, got {len(out.columns)}")

if out.duplicated(GRAIN).any():
    fail("duplicate season/week/game_id/player_id rows")

if len(out) != len(uni):
    fail(f"output rows {len(out):,} != universe rows {len(uni):,}")

print("PASS: Exact 400-column contract, 390 derived features, and canonical grain.")

# ---------------------------------------------------------------------
# Target universe normalization.
# ---------------------------------------------------------------------

target = uni[
    BASE_ID_COLUMNS + ["kickoff_timestamp"]
].copy()

target["season"] = pd.to_numeric(target["season"], errors="raise").astype(int)
target["week"] = pd.to_numeric(target["week"], errors="raise").astype(int)
target["game_id"] = target["game_id"].map(clean)
target["player_id"] = target["player_id"].map(clean)
target["team"] = target["team"].map(norm_team)
target["position"] = target["position"].map(norm_position)
target["position_group"] = [
    norm_group(g, p)
    for g, p in zip(target["position_group"], target["position"])
]
target["_history_team"] = target["team"].map(history_team)
target["_kickoff"] = pd.to_datetime(
    target["kickoff_timestamp"],
    utc=True,
    errors="raise",
)

target = target.sort_values(
    ["season", "week", "game_id", "player_id", "team", "position"],
    kind="mergesort",
).reset_index(drop=True)
target["_row_id"] = np.arange(len(target), dtype=np.int64)

out_sorted = out.sort_values(
    ["season", "week", "game_id", "player_id", "team", "position"],
    kind="mergesort",
).reset_index(drop=True)

for c in BASE_ID_COLUMNS:
    if not out_sorted[c].fillna("").astype(str).eq(
        target[c].fillna("").astype(str)
    ).all():
        fail(f"identifier/base column differs from universe: {c}")

print("PASS: Output identifiers/team/position exactly match normalized universe.")

# ---------------------------------------------------------------------
# Canonical 2012-2025 opportunity source.
# ---------------------------------------------------------------------

required = BASE_ID_COLUMNS + BASE_METRICS
missing = [c for c in required if c not in opp.columns]
if missing:
    fail(f"player opportunity missing required columns: {missing}")

source = opp[required].copy()
source["season"] = pd.to_numeric(source["season"], errors="raise").astype(int)
source["week"] = pd.to_numeric(source["week"], errors="raise").astype(int)
source["game_id"] = source["game_id"].map(clean)
source["player_id"] = source["player_id"].map(clean)
source["team"] = source["team"].map(norm_team)
source["position"] = source["position"].map(norm_position)
source["position_group"] = [
    norm_group(g, p)
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
    fail("opportunity row missing target-game kickoff")

for metric in BASE_METRICS:
    source[metric] = numeric(source[metric])

source["_history_team"] = source["team"].map(history_team)
source["_prehistory"] = 0

# ---------------------------------------------------------------------
# Independent 2010-2011 prehistory reconstruction.
# ---------------------------------------------------------------------

pre_frames = []

needed = [
    "season", "week", "season_type", "game_id", "player_id", "team",
    "position", "position_group",
    *sorted(set(PREHISTORY_DIRECT_MAP.values())),
    "receiving_air_yards",
    "def_tackles_solo",
    "def_tackle_assists",
]

for season in (2010, 2011):
    path = Path(str(STATS_PATTERN).format(season=season))
    raw = pd.read_parquet(path)

    miss = [c for c in needed if c not in raw.columns]
    if miss:
        fail(f"{season} prehistory missing columns: {miss}")

    raw = raw.loc[
        pd.to_numeric(raw["season"], errors="coerce").eq(season)
        & raw["season_type"].astype(str).str.upper().eq("REG")
    ].copy()

    raw["season"] = season
    raw["week"] = pd.to_numeric(raw["week"], errors="raise").astype(int)
    raw["game_id"] = raw["game_id"].map(clean)
    raw["player_id"] = raw["player_id"].map(clean)
    raw["team"] = raw["team"].map(norm_team)
    raw["position"] = raw["position"].map(norm_position)
    raw["position_group"] = [
        norm_group(g, p)
        for g, p in zip(raw["position_group"], raw["position"])
    ]

    for c in set(PREHISTORY_DIRECT_MAP.values()) | {
        "receiving_air_yards",
        "def_tackles_solo",
        "def_tackle_assists",
    }:
        raw[c] = numeric(raw[c], fill_zero=True)

    tg = ["season", "week", "game_id", "team"]
    den = (
        raw.groupby(tg, as_index=False, dropna=False)
        .agg(
            _team_carries=("carries", "sum"),
            _team_targets=("targets", "sum"),
            _team_air=("receiving_air_yards", "sum"),
        )
    )
    raw = raw.merge(den, on=tg, how="left", validate="many_to_one")

    frame = raw[BASE_ID_COLUMNS].copy()

    for metric in BASE_METRICS:
        frame[metric] = np.nan

    for metric, source_col in PREHISTORY_DIRECT_MAP.items():
        frame[metric] = raw[source_col].astype("float64")

    frame["tackles"] = (
        raw["def_tackles_solo"].astype("float64")
        + raw["def_tackle_assists"].astype("float64")
    )
    frame["yards_per_attempt"] = safe_div(raw["passing_yards"], raw["attempts"])
    frame["passing_td_rate"] = safe_div(raw["passing_tds"], raw["attempts"])
    frame["yards_per_carry"] = safe_div(raw["rushing_yards"], raw["carries"])
    frame["yards_per_target"] = safe_div(raw["receiving_yards"], raw["targets"])
    frame["catch_rate"] = safe_div(raw["receptions"], raw["targets"])
    frame["carry_share"] = safe_div(raw["carries"], raw["_team_carries"])
    frame["target_share"] = safe_div(raw["targets"], raw["_team_targets"])
    frame["air_yards_share"] = safe_div(
        raw["receiving_air_yards"],
        raw["_team_air"],
    )

    frame["_kickoff"] = (
        pd.Timestamp(f"{season}-01-01", tz="UTC")
        + pd.to_timedelta(frame["week"].astype(int) * 7, unit="D")
    )
    frame["_history_team"] = frame["team"].map(history_team)
    frame["_prehistory"] = 1
    frame = frame.loc[frame["player_id"].ne("")].copy()

    if frame.duplicated(GRAIN).any():
        fail(f"{season} prehistory has duplicate canonical grain")

    pre_frames.append(frame)

pre = pd.concat(pre_frames, ignore_index=True, sort=False)

combined_cols = BASE_ID_COLUMNS + BASE_METRICS + [
    "_kickoff", "_history_team", "_prehistory"
]

source = pd.concat(
    [pre[combined_cols], source[combined_cols]],
    ignore_index=True,
    sort=False,
)

source = source.loc[source["player_id"].ne("")].copy()
source = source.sort_values(
    ["player_id", "_kickoff", "season", "week", "game_id"],
    kind="mergesort",
).reset_index(drop=True)

if source.duplicated(GRAIN).any():
    fail("combined 2010-2025 source has duplicate canonical grain")

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

print("PASS: 2010-2011 prehistory reconstructed without unsupported rich metrics.")

# ---------------------------------------------------------------------
# Exact history_games / no_nfl_history / new_team.
# ---------------------------------------------------------------------

left = target[
    ["_row_id", "player_id", "_kickoff", "_history_team"]
].copy()

right = source[
    [
        "player_id", "_kickoff", "_history_team",
        "_stint_id", "_source_history_games",
    ]
].copy().rename(
    columns={"_history_team": "_prior_history_team"}
)

meta = pd.merge_asof(
    asof_sort(left, ["player_id"]),
    asof_sort(right, ["player_id"]),
    on="_kickoff",
    by="player_id",
    direction="backward",
    allow_exact_matches=False,
).sort_values("_row_id", kind="mergesort")

history_games = meta["_source_history_games"].fillna(0).astype("int32")
no_history = history_games.eq(0).astype("int8")
prior_team = meta["_prior_history_team"].fillna("")
new_team = (
    history_games.gt(0)
    & meta["_history_team"].fillna("").ne(prior_team)
).astype("int8")

assert_numeric_same(
    out_sorted["history_games"],
    history_games.to_numpy(),
    "history_games",
    rtol=0,
    atol=0,
)
assert_numeric_same(
    out_sorted["no_nfl_history_flag"],
    no_history.to_numpy(),
    "no_nfl_history_flag",
    rtol=0,
    atol=0,
)
assert_numeric_same(
    out_sorted["new_team_flag"],
    new_team.to_numpy(),
    "new_team_flag",
    rtol=0,
    atol=0,
)

if not out_sorted["no_nfl_history_flag"].isin([0, 1]).all():
    fail("no_nfl_history_flag not binary")
if not out_sorted["new_team_flag"].isin([0, 1]).all():
    fail("new_team_flag not binary")
if (out_sorted["history_games"] < 0).any():
    fail("history_games contains negative values")

target["_target_stint_id"] = np.where(
    (history_games.gt(0) & new_team.eq(0)).to_numpy(),
    meta["_stint_id"].fillna(-1).astype("int64").to_numpy(),
    -1,
).astype("int64")

print("PASS: history_games, no_nfl_history_flag, and new_team_flag are exact.")

# ---------------------------------------------------------------------
# Full reconstruction of every required metric block.
# ---------------------------------------------------------------------

for metric_index, metric in enumerate(BASE_METRICS, start=1):
    print(f"VALIDATE metric {metric_index:02d}/{len(BASE_METRICS):02d}: {metric}")

    names = [f"{metric}_{suffix}" for suffix in SUFFIXES]

    valid = source.loc[
        source[metric].notna(),
        [
            "season", "player_id", "position_group",
            "_kickoff", "_stint_id", metric,
        ],
    ].copy()

    if valid.empty:
        for name in names:
            if out_sorted[name].notna().any():
                fail(f"{name} should be entirely null: no source history exists")
        continue

    valid = valid.sort_values(
        ["player_id", "_kickoff", "season", "_stint_id"],
        kind="mergesort",
    )

    group_keys = (
        ["player_id", "_stint_id"]
        if metric in TEAM_SHARE_METRICS
        else ["player_id"]
    )

    valid[names[0]] = valid[metric].astype("float64")
    valid[names[1]] = grouped_rolling(valid, group_keys, metric, 3, "mean")
    valid[names[2]] = grouped_rolling(valid, group_keys, metric, 5, "mean")
    valid[names[3]] = grouped_rolling(valid, group_keys, metric, 8, "mean")
    valid[names[4]] = grouped_rolling(valid, group_keys, metric, 3, "median")
    valid[names[5]] = grouped_rolling(valid, group_keys, metric, 5, "std0")
    valid[names[6]] = grouped_ewm(valid, group_keys, metric, 3)
    valid[names[7]] = grouped_ewm(valid, group_keys, metric, 5)

    season_keys = [*group_keys, "season"]
    season_sum = valid.groupby(season_keys, sort=False)[metric].cumsum()
    season_count = valid.groupby(season_keys, sort=False).cumcount() + 1
    valid[names[8]] = season_sum / season_count

    career_sum = valid.groupby(group_keys, sort=False)[metric].cumsum()
    career_count = valid.groupby(group_keys, sort=False).cumcount() + 1
    valid[names[9]] = career_sum / career_count

    # Strictly prior position prior from valid historical observations.
    px = valid.loc[
        valid["position_group"].ne(""),
        ["position_group", "_kickoff", metric],
    ].copy()

    if px.empty:
        prior_mean = np.full(len(target), np.nan)
        prior_std = np.full(len(target), np.nan)
    else:
        px["_square"] = px[metric].astype("float64") ** 2

        pagg = (
            px.groupby(
                ["position_group", "_kickoff"],
                as_index=False,
                sort=True,
            )
            .agg(
                _sum=(metric, "sum"),
                _count=(metric, "count"),
                _sumsq=("_square", "sum"),
            )
            .sort_values(
                ["position_group", "_kickoff"],
                kind="mergesort",
            )
        )

        pagg["_cum_sum"] = pagg.groupby(
            "position_group", sort=False
        )["_sum"].cumsum()
        pagg["_cum_count"] = pagg.groupby(
            "position_group", sort=False
        )["_count"].cumsum()
        pagg["_cum_sumsq"] = pagg.groupby(
            "position_group", sort=False
        )["_sumsq"].cumsum()

        pagg["_prior_mean"] = (
            pagg["_cum_sum"] / pagg["_cum_count"]
        )

        variance = (
            pagg["_cum_sumsq"] / pagg["_cum_count"]
            - pagg["_prior_mean"] ** 2
        ).clip(lower=0.0)

        pagg["_prior_std"] = np.sqrt(variance)

        pleft = target[
            ["_row_id", "position_group", "_kickoff"]
        ].copy()

        pjoin = pd.merge_asof(
            asof_sort(pleft, ["position_group"]),
            asof_sort(
                pagg[
                    [
                        "position_group", "_kickoff",
                        "_prior_mean", "_prior_std",
                    ]
                ],
                ["position_group"],
            ),
            on="_kickoff",
            by="position_group",
            direction="backward",
            allow_exact_matches=False,
        ).sort_values("_row_id", kind="mergesort")

        prior_mean = pjoin["_prior_mean"].to_numpy(dtype="float64")
        prior_std = pjoin["_prior_std"].to_numpy(dtype="float64")

    if metric in TEAM_SHARE_METRICS:
        mleft = target[
            [
                "_row_id", "player_id",
                "_target_stint_id", "_kickoff", "season",
            ]
        ].rename(columns={"_target_stint_id": "_stint_id"})

        mright = valid[
            ["player_id", "_stint_id", "_kickoff", "season", *names]
        ].copy()

        by = ["player_id", "_stint_id"]
    else:
        mleft = target[
            ["_row_id", "player_id", "_kickoff", "season"]
        ].copy()

        mright = valid[
            ["player_id", "_kickoff", "season", *names]
        ].copy()

        by = ["player_id"]

    mright = mright.rename(columns={"season": "_state_season"})

    joined = pd.merge_asof(
        asof_sort(mleft, by),
        asof_sort(mright, by),
        on="_kickoff",
        by=by,
        direction="backward",
        allow_exact_matches=False,
    ).sort_values("_row_id", kind="mergesort")

    player_metric_history = joined[names[0]].notna().to_numpy()

    for name in names:
        expected = joined[name].to_numpy(dtype="float64")

        if name.endswith("_season_to_date"):
            same_season = (
                joined["_state_season"].notna()
                & joined["season"].eq(joined["_state_season"])
            ).to_numpy()

            expected = np.where(
                same_season,
                expected,
                np.nan,
            )

        fallback = (
            prior_std
            if name.endswith("_roll5_std")
            else prior_mean
        )

        expected = np.where(
            ~player_metric_history,
            fallback,
            expected,
        )

        assert_numeric_same(
            out_sorted[name],
            expected,
            name,
        )

print("PASS: All 390 rolling-form columns exactly reconstruct from strictly prior source history.")

# ---------------------------------------------------------------------
# Explicit leakage / source-boundary checks.
# ---------------------------------------------------------------------

rich_only = [
    "dropbacks",
    "red_zone_carries",
    "goal_line_carries",
    "red_zone_targets",
    "red_zone_target_share",
]

pre_2021 = out_sorted["season"].lt(2021)

for metric in rich_only:
    # Before canonical rich source begins, no direct lag can exist.
    if out_sorted.loc[pre_2021, f"{metric}_lag1"].notna().any():
        fail(f"{metric}_lag1 populated before 2021 canonical source")

pre_2016 = out_sorted["season"].lt(2016)

for metric in [
    "offense_participation",
    "defense_participation",
]:
    if out_sorted.loc[pre_2016, f"{metric}_lag1"].notna().any():
        fail(f"{metric}_lag1 populated before 2016 participation source")

# Same-game leakage guard:
# every non-prior-fallback player state was matched with allow_exact_matches=False
# above. Check that same kickoff cannot be source history for an actual first
# observed metric row by testing the canonical 2021 rich-source boundary.
for metric in rich_only:
    first_source = (
        source.loc[source[metric].notna()]
        .groupby("player_id", as_index=False)["_kickoff"]
        .min()
        .rename(columns={"_kickoff": "_first_metric_kickoff"})
    )

    probe = target.merge(first_source, on="player_id", how="inner")
    equal = probe["_kickoff"].eq(probe["_first_metric_kickoff"])

    # At exact first-source kickoff, lag1 may be a position prior, but may not
    # equal a player-current-game value through a player history match. Full
    # reconstruction above enforces that behavior.
    if not equal.any():
        fail(f"no first-source kickoff probes found for {metric}")

print("PASS: Rich/PBP and participation source boundaries preserve null history until available.")

# ---------------------------------------------------------------------
# Finite / range sanity.
# ---------------------------------------------------------------------

numeric_features = out_sorted[feature_cols].apply(
    pd.to_numeric,
    errors="coerce",
)

if np.isinf(numeric_features.to_numpy(dtype="float64")).any():
    fail("feature matrix contains infinity")

if not out_sorted["no_nfl_history_flag"].eq(
    out_sorted["history_games"].eq(0).astype(int)
).all():
    fail("no_nfl_history_flag != (history_games == 0)")

print("PASS: No infinities and required history-flag semantics hold.")

print("=" * 100)
print("ISSUE 12 ACCEPTANCE: PASS")
print("=" * 100)
