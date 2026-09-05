from __future__ import annotations

from pathlib import Path
from typing import Any
import math

import numpy as np
import pandas as pd

ROOT = Path("docs/win/football/nfl/prop_engine/data/historical")

UNIVERSE = ROOT / "universe/player_game_universe.parquet"
PLAYER_FORM = ROOT / "features/player_form.parquet"
ROLE_HISTORY = ROOT / "features/player_role_history.parquet"
TEAM_FORM = ROOT / "features/team_form.parquet"
OPP_RAW = ROOT / "opportunity/opponent_week_opportunity.parquet"
OUT = ROOT / "features/defensive_features.parquet"

GRAIN = ["season", "week", "game_id", "player_id"]

EXPECTED_COLUMNS = [
    "season",
    "week",
    "game_id",
    "player_id",
    "position",
    "def_snap_pct_lag1",
    "def_snap_pct_roll3",
    "def_participation_lag1",
    "def_participation_roll3",
    "tackles_lag1",
    "tackles_roll3",
    "tackles_roll5",
    "tackle_rate_roll3",
    "tackle_rate_roll5",
    "sacks_lag1",
    "sacks_roll3",
    "sacks_roll5",
    "sack_rate_roll5",
    "qb_hits_roll3",
    "qb_hits_roll5",
    "opponent_plays_roll3",
    "opponent_dropbacks_roll3",
    "opponent_rush_rate_roll3",
    "opponent_pass_rate_roll3",
    "team_def_sack_rate_roll3",
    "starter_flag",
    "front7_flag",
    "secondary_flag",
]

PLAYER_FORM_MAP = {
    "def_snap_pct_lag1": "defense_snap_pct_lag1",
    "def_snap_pct_roll3": "defense_snap_pct_roll3_mean",
    "def_participation_lag1": "defense_participation_lag1",
    "def_participation_roll3": "defense_participation_roll3_mean",
    "tackles_lag1": "tackles_lag1",
    "tackles_roll3": "tackles_roll3_mean",
    "tackles_roll5": "tackles_roll5_mean",
    "tackle_rate_roll3": "tackle_rate_per_def_play_roll3_mean",
    "tackle_rate_roll5": "tackle_rate_per_def_play_roll5_mean",
    "sacks_lag1": "sacks_lag1",
    "sacks_roll3": "sacks_roll3_mean",
    "sacks_roll5": "sacks_roll5_mean",
    "sack_rate_roll5": "sack_rate_per_def_play_roll5_mean",
    "qb_hits_roll3": "qb_hits_roll3_mean",
    "qb_hits_roll5": "qb_hits_roll5_mean",
}

TEAM_FORM_MAP = {
    "opponent_plays_roll3": "offensive_plays_roll3_mean",
    "opponent_dropbacks_roll3": "dropbacks_roll3_mean",
    "opponent_rush_rate_roll3": "rush_rate_roll3_mean",
    "opponent_pass_rate_roll3": "pass_rate_roll3_mean",
}

FRONT7 = {
    "DL","DE","LDE","RDE","DT","LDT","RDT","NT","EDGE",
    "LB","ILB","OLB","MLB","WLB","SLB",
}

SECONDARY = {
    "DB","CB","LCB","RCB","NB","S","SAF","FS","SS",
}

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


def norm_position(value: Any) -> str:
    return clean(value).upper().replace(" ", "")


def assert_same_numeric(actual, expected, label, rtol=1e-10, atol=1e-10):
    a = pd.to_numeric(actual, errors="coerce").to_numpy(dtype="float64")
    e = pd.to_numeric(expected, errors="coerce").to_numpy(dtype="float64")

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


print("=" * 110)
print("ISSUE 15 ACCEPTANCE VALIDATION")
print("=" * 110)

u = pd.read_parquet(UNIVERSE)
pf = pd.read_parquet(PLAYER_FORM)
rh = pd.read_parquet(ROLE_HISTORY)
tf = pd.read_parquet(TEAM_FORM)
opp = pd.read_parquet(OPP_RAW)
out = pd.read_parquet(OUT)

# ---------------------------------------------------------------------
# Exact output contract and canonical grain.
# ---------------------------------------------------------------------

if list(out.columns) != EXPECTED_COLUMNS:
    missing = [c for c in EXPECTED_COLUMNS if c not in out.columns]
    extra = [c for c in out.columns if c not in EXPECTED_COLUMNS]
    fail(
        "exact header/order mismatch; "
        f"missing={missing} extra={extra}"
    )

if len(out) != len(u):
    fail(f"row count {len(out):,} != universe {len(u):,}")

if out.duplicated(GRAIN).any():
    fail("duplicate canonical player-game grain")

u_keys = u[GRAIN].sort_values(GRAIN, kind="mergesort").reset_index(drop=True)
o_keys = out[GRAIN].sort_values(GRAIN, kind="mergesort").reset_index(drop=True)

if not o_keys.astype(str).eq(u_keys.astype(str)).all().all():
    fail("output key set differs from historical universe")

print("PASS: Exact 28-column contract, row count, canonical grain, and universe key set.")

# Stable sort for rowwise independent comparisons.
u = u.sort_values(GRAIN, kind="mergesort").reset_index(drop=True)
pf = pf.sort_values(GRAIN, kind="mergesort").reset_index(drop=True)
rh = rh.sort_values(GRAIN, kind="mergesort").reset_index(drop=True)
out = out.sort_values(GRAIN, kind="mergesort").reset_index(drop=True)

# ---------------------------------------------------------------------
# Player form columns must be exact copies of already leakage-safe history.
# ---------------------------------------------------------------------

if not out["position"].fillna("").astype(str).eq(
    u["position"].fillna("").astype(str)
).all():
    fail("position differs from universe")

for target, source in PLAYER_FORM_MAP.items():
    assert_same_numeric(
        out[target],
        pf[source],
        f"{target} <- player_form.{source}",
    )

print("PASS: All 15 player defensive-history features exactly match leakage-safe player_form.")

# ---------------------------------------------------------------------
# Starter flag must match pregame role history.
# ---------------------------------------------------------------------

expected_starter = (
    pd.to_numeric(
        rh["depth_starter_flag_pregame"],
        errors="coerce",
    )
    .fillna(0)
    .astype(int)
)

assert_same_numeric(
    out["starter_flag"],
    expected_starter,
    "starter_flag",
    rtol=0,
    atol=0,
)

print("PASS: starter_flag exactly matches pregame depth starter state.")

# ---------------------------------------------------------------------
# Opponent offensive matchup context.
# ---------------------------------------------------------------------

tf = tf.copy()
tf["team"] = tf["team"].map(canonical_team)

if tf.duplicated(["season", "week", "team"]).any():
    fail("team_form duplicate canonical season/week/team after aliasing")

base = u[
    GRAIN + ["team", "opponent"]
].copy()

base["_matchup_team"] = base["opponent"].map(canonical_team)

tf_join = tf[
    ["season", "week", "team"] + list(TEAM_FORM_MAP.values())
].rename(columns={"team": "_matchup_team"})

expected_matchup = base.merge(
    tf_join,
    on=["season", "week", "_matchup_team"],
    how="left",
    validate="many_to_one",
)

expected_matchup = expected_matchup.sort_values(
    GRAIN,
    kind="mergesort",
).reset_index(drop=True)

for target, source in TEAM_FORM_MAP.items():
    assert_same_numeric(
        out[target],
        expected_matchup[source],
        f"{target} <- opponent team_form.{source}",
    )

print("PASS: Opponent offensive roll3 context exactly matches opponent's already-lagged team_form.")

# ---------------------------------------------------------------------
# Independently reconstruct team_def_sack_rate_roll3.
# ---------------------------------------------------------------------

opp = opp[
    ["season", "week", "team", "sacks", "opponent_dropbacks"]
].copy()

opp["season"] = pd.to_numeric(opp["season"], errors="raise").astype(int)
opp["week"] = pd.to_numeric(opp["week"], errors="raise").astype(int)
opp["team"] = opp["team"].map(canonical_team)

if opp.duplicated(["season", "week", "team"]).any():
    fail("opponent opportunity duplicate canonical season/week/team")

sacks = pd.to_numeric(opp["sacks"], errors="coerce").astype(float)
dropbacks = pd.to_numeric(
    opp["opponent_dropbacks"],
    errors="coerce",
).astype(float)

opp["game_sack_rate"] = np.where(
    sacks.notna() & dropbacks.notna() & dropbacks.ne(0),
    sacks / dropbacks,
    np.nan,
)

opp = opp.sort_values(
    ["team", "season", "week"],
    kind="mergesort",
).reset_index(drop=True)

expected_roll = np.full(len(opp), np.nan, dtype="float64")

for _, indices in opp.groupby("team", sort=False).indices.items():
    history = []

    for pos in indices:
        if history:
            expected_roll[pos] = float(np.mean(history[-3:]))

        current = opp.at[pos, "game_sack_rate"]

        if pd.notna(current) and math.isfinite(float(current)):
            history.append(float(current))

opp["team_def_sack_rate_roll3"] = expected_roll

def_lookup = opp[
    ["season", "week", "team", "team_def_sack_rate_roll3"]
].rename(columns={"team": "_defense_team"})

base["_defense_team"] = base["team"].map(canonical_team)

expected_def = base.merge(
    def_lookup,
    on=["season", "week", "_defense_team"],
    how="left",
    validate="many_to_one",
)

expected_def = expected_def.sort_values(
    GRAIN,
    kind="mergesort",
).reset_index(drop=True)

assert_same_numeric(
    out["team_def_sack_rate_roll3"],
    expected_def["team_def_sack_rate_roll3"],
    "team_def_sack_rate_roll3",
)

print("PASS: team_def_sack_rate_roll3 exactly reconstructs from strictly prior per-game sacks/opponent_dropbacks rates.")

# ---------------------------------------------------------------------
# Position flags.
# ---------------------------------------------------------------------

position = u["position"].map(norm_position)

expected_front7 = position.isin(FRONT7).astype(int)
expected_secondary = position.isin(SECONDARY).astype(int)

assert_same_numeric(
    out["front7_flag"],
    expected_front7,
    "front7_flag",
    rtol=0,
    atol=0,
)

assert_same_numeric(
    out["secondary_flag"],
    expected_secondary,
    "secondary_flag",
    rtol=0,
    atol=0,
)

if (
    out["front7_flag"]
    + out["secondary_flag"]
    > 1
).any():
    fail("front7_flag and secondary_flag overlap")

print("PASS: front7_flag and secondary_flag exactly match locked position families.")

# ---------------------------------------------------------------------
# Safety and sanity.
# ---------------------------------------------------------------------

for c in ["starter_flag", "front7_flag", "secondary_flag"]:
    if not out[c].isin([0, 1]).all():
        fail(f"{c}: contains non-binary values")

if any("pass_rush_share" in c.casefold() for c in out.columns):
    fail("forbidden pass_rush_share column exists")

numeric_cols = [
    c for c in out.columns
    if c not in {"game_id", "player_id", "position"}
]

matrix = (
    out[numeric_cols]
    .apply(pd.to_numeric, errors="coerce")
    .to_numpy(dtype="float64")
)

if np.isinf(matrix).any():
    fail("output contains infinity")

print("PASS: Binary, finite-value, and pass_rush_share prohibition checks.")

print("=" * 110)
print("ISSUE 15 ACCEPTANCE: PASS")
print("=" * 110)
