from __future__ import annotations

from pathlib import Path
from typing import Any
import re

import numpy as np
import pandas as pd


ROOT = Path("docs/win/football/nfl/prop_engine/data/historical")

UNIVERSE = ROOT / "universe/player_game_universe.parquet"
PLAYER_FORM = ROOT / "features/player_form.parquet"
ROLE_HISTORY = ROOT / "features/player_role_history.parquet"
TEAM_FORM = ROOT / "features/team_form.parquet"
OPPONENT_FORM = ROOT / "features/opponent_form.parquet"
ENVIRONMENT = ROOT / "features/environment.parquet"
OUT = ROOT / "features/kicking_features.parquet"

GRAIN = ["season", "week", "game_id", "player_id"]

EXPECTED_COLUMNS = [
    "season",
    "week",
    "game_id",
    "player_id",
    "team",
    "fg_attempts_lag1",
    "fg_attempts_roll3",
    "fg_attempts_roll5",
    "fg_make_pct_career_prior",
    "fg_make_pct_season_prior",
    "pat_attempts_roll3",
    "pat_make_pct_career_prior",
    "team_drives_roll3",
    "team_points_per_drive_roll3",
    "team_red_zone_td_rate_roll3",
    "opponent_points_per_drive_allowed_roll3",
    "opponent_red_zone_td_rate_allowed_roll3",
    "temperature",
    "wind",
    "roof",
    "surface",
    "primary_kicker_flag",
]

TEAM_ALIASES = {
    "WAS": "WSH",
    "LA": "LAR",
    "JAC": "JAX",
    "SD": "LAC",
    "OAK": "LV",
    "STL": "LAR",
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


def canonical_team(value: Any) -> str:
    key = clean(value).upper()
    return TEAM_ALIASES.get(key, key)


def numeric(series: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .astype("float64")
    )


def safe_ratio(numerator, denominator):
    num = numeric(numerator)
    den = numeric(denominator)

    result = pd.Series(np.nan, index=num.index, dtype="float64")

    valid = num.notna() & den.notna() & den.gt(0)

    result.loc[valid] = num.loc[valid] / den.loc[valid]

    return result


def assert_same_numeric(actual, expected, label, rtol=1e-10, atol=1e-10):
    a = numeric(actual).to_numpy(dtype="float64")
    e = numeric(expected).to_numpy(dtype="float64")

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
        bad = np.flatnonzero(~ok)
        sample = [
            {
                "row": int(i),
                "actual": None if np.isnan(a[i]) else float(a[i]),
                "expected": None if np.isnan(e[i]) else float(e[i]),
            }
            for i in bad[:10]
        ]
        fail(f"{label}: {len(bad):,} mismatches; sample={sample}")


def depth_role_score(value: Any) -> int:
    text = clean(value).upper()
    compact = re.sub(r"\s+", "", text)

    if not compact:
        return 1

    tail = compact.split("|")[-1]

    if tail in {"FG", "K", "PK"}:
        return 5

    if "FG" in tail:
        return 5

    if tail in {"K/KO", "KICKER"}:
        return 4

    if tail in {"KO", "KOS"}:
        return 2

    if tail in {"P", "H"}:
        return 0

    return 1


print("=" * 110)
print("ISSUE 16 ACCEPTANCE VALIDATION")
print("=" * 110)

u = pd.read_parquet(UNIVERSE)
pf = pd.read_parquet(PLAYER_FORM)
rh = pd.read_parquet(ROLE_HISTORY)
tf = pd.read_parquet(TEAM_FORM)
of = pd.read_parquet(OPPONENT_FORM)
env = pd.read_parquet(ENVIRONMENT)
out = pd.read_parquet(OUT)

# Exact contract.
if list(out.columns) != EXPECTED_COLUMNS:
    fail("exact header/order mismatch")

if out.duplicated(GRAIN).any():
    fail("duplicate kicker player-game grain")

kicker_mask = (
    u["position"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.upper()
    .isin(["K", "PK"])
)

expected_keys = (
    u.loc[kicker_mask, GRAIN]
    .sort_values(GRAIN, kind="mergesort")
    .reset_index(drop=True)
)

actual_keys = (
    out[GRAIN]
    .sort_values(GRAIN, kind="mergesort")
    .reset_index(drop=True)
)

if len(out) != len(expected_keys):
    fail(f"row count {len(out):,} != expected kicker rows {len(expected_keys):,}")

if not actual_keys.astype(str).eq(expected_keys.astype(str)).all().all():
    fail("output key set differs from K/PK universe")

print("PASS: Exact 22-column contract, kicker row count, grain, and K/PK universe key set.")

# Stable order.
u = u.loc[kicker_mask].sort_values(GRAIN, kind="mergesort").reset_index(drop=True)
pf = pf.sort_values(GRAIN, kind="mergesort").reset_index(drop=True)
rh = rh.sort_values(GRAIN, kind="mergesort").reset_index(drop=True)
out = out.sort_values(GRAIN, kind="mergesort").reset_index(drop=True)

pfk = u[GRAIN].merge(
    pf,
    on=GRAIN,
    how="left",
    validate="one_to_one",
)

# Team identity.
if not out["team"].astype(str).eq(u["team"].astype(str)).all():
    fail("team differs from universe")

# Direct player form mappings.
direct_map = {
    "fg_attempts_lag1": "field_goal_attempts_lag1",
    "fg_attempts_roll3": "field_goal_attempts_roll3_mean",
    "fg_attempts_roll5": "field_goal_attempts_roll5_mean",
    "pat_attempts_roll3": "extra_point_attempts_roll3_mean",
}

for target, source in direct_map.items():
    assert_same_numeric(
        out[target],
        pfk[source],
        f"{target} <- player_form.{source}",
    )

# Make percentage formulas.
assert_same_numeric(
    out["fg_make_pct_career_prior"],
    safe_ratio(
        pfk["field_goals_made_career_prior"],
        pfk["field_goal_attempts_career_prior"],
    ),
    "fg_make_pct_career_prior",
)

assert_same_numeric(
    out["fg_make_pct_season_prior"],
    safe_ratio(
        pfk["field_goals_made_season_to_date"],
        pfk["field_goal_attempts_season_to_date"],
    ),
    "fg_make_pct_season_prior",
)

assert_same_numeric(
    out["pat_make_pct_career_prior"],
    safe_ratio(
        pfk["extra_points_made_career_prior"],
        pfk["extra_point_attempts_career_prior"],
    ),
    "pat_make_pct_career_prior",
)

print("PASS: Kicker attempt history and prior FG/PAT percentage formulas are exact.")

# Team form.
base = u[GRAIN + ["team", "opponent"]].copy()
base["_team_key"] = base["team"].map(canonical_team)
base["_opponent_key"] = base["opponent"].map(canonical_team)

tf2 = tf.copy()
tf2["team"] = tf2["team"].map(canonical_team)

if tf2.duplicated(["season", "week", "team"]).any():
    fail("team_form duplicate canonical team-week key")

team_join = base.merge(
    tf2[
        [
            "season",
            "week",
            "team",
            "drives_roll3_mean",
            "points_per_drive_roll3_mean",
            "red_zone_td_rate_roll3_mean",
        ]
    ].rename(columns={"team": "_team_key"}),
    on=["season", "week", "_team_key"],
    how="left",
    validate="many_to_one",
).sort_values(GRAIN, kind="mergesort").reset_index(drop=True)

for target, source in {
    "team_drives_roll3": "drives_roll3_mean",
    "team_points_per_drive_roll3": "points_per_drive_roll3_mean",
    "team_red_zone_td_rate_roll3": "red_zone_td_rate_roll3_mean",
}.items():
    assert_same_numeric(out[target], team_join[source], target)

print("PASS: Team offensive roll3 context exactly matches already-lagged team_form.")

# Opponent form.
of2 = of.copy()
of2["team"] = of2["team"].map(canonical_team)

if of2.duplicated(["season", "week", "team"]).any():
    fail("opponent_form duplicate canonical team-week key")

opp_join = base.merge(
    of2[
        [
            "season",
            "week",
            "team",
            "points_per_drive_allowed_roll3_mean",
            "red_zone_td_rate_allowed_roll3_mean",
        ]
    ].rename(columns={"team": "_opponent_key"}),
    on=["season", "week", "_opponent_key"],
    how="left",
    validate="many_to_one",
).sort_values(GRAIN, kind="mergesort").reset_index(drop=True)

assert_same_numeric(
    out["opponent_points_per_drive_allowed_roll3"],
    opp_join["points_per_drive_allowed_roll3_mean"],
    "opponent_points_per_drive_allowed_roll3",
)

assert_same_numeric(
    out["opponent_red_zone_td_rate_allowed_roll3"],
    opp_join["red_zone_td_rate_allowed_roll3_mean"],
    "opponent_red_zone_td_rate_allowed_roll3",
)

print("PASS: Opponent defensive roll3 context exactly matches already-lagged opponent_form.")

# Environment.
env_join = u[GRAIN].merge(
    env[
        [
            "season",
            "week",
            "game_id",
            "temperature",
            "wind",
            "roof",
            "surface",
        ]
    ],
    on=["season", "week", "game_id"],
    how="left",
    validate="many_to_one",
).sort_values(GRAIN, kind="mergesort").reset_index(drop=True)

assert_same_numeric(out["temperature"], env_join["temperature"], "temperature")
assert_same_numeric(out["wind"], env_join["wind"], "wind")

if not out["roof"].fillna("").astype(str).eq(
    env_join["roof"].fillna("").astype(str)
).all():
    fail("roof differs from environment")

if not out["surface"].fillna("").astype(str).eq(
    env_join["surface"].fillna("").astype(str)
).all():
    fail("surface differs from environment")

print("PASS: Weather/roof/surface exactly match historical environment.")

# Primary-kicker reconstruction.
rk = u[
    GRAIN
    + [
        "team",
        "roster_flag",
        "depth_present_flag",
        "depth_rank",
        "depth_slot",
        "depth_starter_flag",
    ]
].copy()

inj = u[GRAIN].merge(
    rh[GRAIN + ["injury_out_flag"]],
    on=GRAIN,
    how="left",
    validate="one_to_one",
)

rk = rk.merge(inj, on=GRAIN, how="left", validate="one_to_one")

rk = rk.merge(
    pf[
        GRAIN
        + [
            "field_goal_attempts_roll3_mean",
            "extra_point_attempts_roll3_mean",
            "field_goal_attempts_lag1",
            "field_goal_attempts_career_prior",
            "extra_point_attempts_career_prior",
        ]
    ],
    on=GRAIN,
    how="left",
    validate="one_to_one",
)

rk["_not_out"] = numeric(rk["injury_out_flag"]).fillna(0).eq(0).astype(int)
rk["_roster"] = numeric(rk["roster_flag"]).fillna(0).astype(int)
rk["_depth_present"] = numeric(rk["depth_present_flag"]).fillna(0).astype(int)
rk["_role_score"] = rk["depth_slot"].map(depth_role_score).astype(int)
rk["_starter"] = numeric(rk["depth_starter_flag"]).fillna(0).astype(int)
rk["_rank_score"] = -numeric(rk["depth_rank"]).fillna(999.0)

usage_cols = [
    "field_goal_attempts_roll3_mean",
    "extra_point_attempts_roll3_mean",
    "field_goal_attempts_lag1",
    "field_goal_attempts_career_prior",
    "extra_point_attempts_career_prior",
]

for c in usage_cols:
    rk[f"_usage_{c}"] = numeric(rk[c]).fillna(-1.0)

group_cols = ["season", "week", "game_id", "team"]

sort_cols = (
    group_cols
    + [
        "_not_out",
        "_roster",
        "_depth_present",
        "_role_score",
        "_starter",
        "_rank_score",
    ]
    + [f"_usage_{c}" for c in usage_cols]
    + ["player_id"]
)

ascending = (
    [True] * len(group_cols)
    + [False] * (6 + len(usage_cols))
    + [True]
)

ranked = rk.sort_values(
    sort_cols,
    ascending=ascending,
    kind="mergesort",
)

chosen = (
    ranked.groupby(group_cols, sort=False, as_index=False)
    .head(1)
    .index
)

expected_primary = pd.Series(0, index=rk.index, dtype=int)
expected_primary.loc[chosen] = 1

expected_primary = (
    pd.DataFrame({
        **{c: rk[c] for c in GRAIN},
        "primary": expected_primary,
    })
    .sort_values(GRAIN, kind="mergesort")
    .reset_index(drop=True)
)

assert_same_numeric(
    out["primary_kicker_flag"],
    expected_primary["primary"],
    "primary_kicker_flag",
    rtol=0,
    atol=0,
)

counts = (
    out.groupby(["season", "week", "game_id", "team"])
    ["primary_kicker_flag"]
    .sum()
)

if not counts.eq(1).all():
    fail("not exactly one primary kicker per kicker-present team-game")

print("PASS: primary_kicker_flag exactly reconstructs from pregame-only ranking policy.")

# Sanity.
for c in [
    "fg_make_pct_career_prior",
    "fg_make_pct_season_prior",
    "pat_make_pct_career_prior",
]:
    x = numeric(out[c])
    if (x.notna() & (~x.between(0, 1))).any():
        fail(f"{c} outside [0,1]")

numeric_cols = [
    c for c in EXPECTED_COLUMNS
    if c not in {"game_id", "player_id", "team", "roof", "surface"}
]

matrix = (
    out[numeric_cols]
    .apply(pd.to_numeric, errors="coerce")
    .to_numpy(dtype="float64")
)

if np.isinf(matrix).any():
    fail("output contains infinity")

print("PASS: Percentage ranges and finite-value checks.")
print("=" * 110)
print("ISSUE 16 ACCEPTANCE: PASS")
print("=" * 110)
