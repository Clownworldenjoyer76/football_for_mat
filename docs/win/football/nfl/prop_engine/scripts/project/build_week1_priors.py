#!/usr/bin/env python3
"""Build explicit Week 1 player priors for the NFL Prop Engine.

READS:
    docs/win/football/nfl/prop_engine/config/prop_engine.yaml
    docs/win/football/nfl/prop_engine/data/current/{season}_week_1_roles.parquet
    docs/win/football/nfl/prop_engine/data/current/features/{season}_week_1_features.parquet
    configured historical player-game universe
    configured historical player-game feature table (prior-season team reference)

WRITES:
    docs/win/football/nfl/prop_engine/data/current/{season}_week_1_priors.parquet
    docs/win/football/nfl/prop_engine/logs/week1_priors_{season}.json

POLICY:
    - Same-team veterans blend late prior-season signal with career signal.
    - New-team veterans preserve career efficiency but never import old-team
      target/carry/snap/participation role into the new team.
    - Rookies/zero-history players receive position/depth priors only. No NFL
      production or player usage value is fabricated for them.
    - Week 1 role projection is participation-like and bounded to [0, 1].
    - Rookie, new-team, low-history, stale-history, committee, low-confidence,
      and materially changed roles widen uncertainty deterministically.
    - Market-exclusion validation runs before the build.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import common


OUTPUT_COLUMNS = [
    "player_id",
    "team",
    "position",
    "prior_season_games",
    "career_games",
    "prior_team",
    "new_team_flag",
    "prior_snap_share",
    "prior_participation",
    "prior_target_share",
    "prior_carry_share",
    "prior_efficiency",
    "position_prior",
    "depth_rank",
    "week1_role_projection",
    "week1_uncertainty_multiplier",
]

ROLE_REQUIRED = [
    "season", "week", "game_id", "team", "player_id", "player_name",
    "position", "depth_rank", "starter_flag", "primary_qb_flag",
    "primary_kicker_flag", "primary_role_flag", "committee_role_flag",
    "role_confidence", "role_reason",
]

FEATURE_REQUIRED = [
    "season", "week", "game_id", "player_id", "team", "position",
    "position_group", "history_history_games", "history_no_nfl_history_flag",
    "history_new_team_flag",
    "role_prior_offense_snap_pct", "role_prior_defense_snap_pct",
    "role_snap_pct_ewm5", "role_prior_offense_participation",
    "role_prior_defense_participation", "role_participation_roll5",
    "player_target_share_ewm5", "player_target_share_career_prior",
    "player_carry_share_ewm5", "player_carry_share_career_prior",
    "player_yards_per_attempt_ewm5", "player_yards_per_attempt_career_prior",
    "player_yards_per_carry_ewm5", "player_yards_per_carry_career_prior",
    "player_yards_per_target_ewm5", "player_yards_per_target_career_prior",
    "player_field_goal_attempts_roll5_mean",
    "player_field_goals_made_roll5_mean",
    "player_field_goal_attempts_career_prior",
    "player_field_goals_made_career_prior",
    "player_tackle_rate_per_def_play_ewm5",
    "player_tackle_rate_per_def_play_career_prior",
]

HISTORICAL_REQUIRED = [
    "season", "week", "game_id", "player_id", "team",
    "kickoff_timestamp", "played_game_flag",
]

PRIOR_FEATURE_REQUIRED = [
    "season", "week", "game_id", "kickoff_timestamp", "player_id", "team",
]

TEAM_HISTORY_ALIASES = {"SD": "LAC", "OAK": "LV", "STL": "LAR"}

POSITION_GROUP_ALIASES = {
    "HB": "RB", "FB": "RB", "H-BACK": "RB",
    "K": "SPEC", "PK": "SPEC",
    "DE": "DL", "LDE": "DL", "RDE": "DL", "DT": "DL",
    "LDT": "DL", "RDT": "DL", "NT": "DL", "EDGE": "DL",
    "ILB": "LB", "OLB": "LB", "MLB": "LB", "WLB": "LB", "SLB": "LB",
    "CB": "DB", "LCB": "DB", "RCB": "DB", "NB": "DB",
    "S": "DB", "FS": "DB", "SS": "DB",
}

RECENT_WEIGHT = 0.65
CAREER_WEIGHT = 0.35
ROLE_HISTORY_WEIGHT = 0.65
ROLE_DEPTH_WEIGHT = 0.35

ROOKIE_UNCERTAINTY = 1.40
NEW_TEAM_UNCERTAINTY = 1.25
NEW_ROLE_UNCERTAINTY = 1.15
NO_PRIOR_SEASON_UNCERTAINTY = 1.15
LOW_HISTORY_UNCERTAINTY = 1.10
COMMITTEE_UNCERTAINTY = 1.08
KICKER_AMBIGUITY_UNCERTAINTY = 1.25
MAX_UNCERTAINTY = 2.25


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


def as_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")


def flag(series: pd.Series) -> pd.Series:
    return as_num(series).fillna(0).gt(0).astype("int8")


def norm_team(value: Any) -> str:
    team = common.normalize_team(value)
    return TEAM_HISTORY_ALIASES.get(team, team)


def norm_position(value: Any) -> str:
    return clean(value).upper().replace(" ", "")


def norm_position_group(position: Any, group: Any) -> str:
    g = norm_position(group)
    p = norm_position(position)
    if g in {"QB", "RB", "WR", "TE", "DL", "LB", "DB", "SPEC"}:
        return g
    if g in POSITION_GROUP_ALIASES:
        return POSITION_GROUP_ALIASES[g]
    if p in {"QB", "RB", "WR", "TE", "DL", "LB", "DB", "SPEC"}:
        return p
    return POSITION_GROUP_ALIASES.get(p, p)


def repo_relative(path: Path) -> str:
    return str(path.resolve().relative_to(common.repo_root().resolve())).replace("\\", "/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build explicit NFL Week 1 priors.")
    parser.add_argument("--season", type=int, default=None)
    return parser.parse_args()


def resolve_season(args: argparse.Namespace, config: dict) -> int:
    season = int(args.season) if args.season is not None else int(config["seasons"]["current"])
    if not 1900 <= season <= 2200:
        raise ValueError(f"Invalid season: {season}")
    return season


def run_market_preflight() -> dict[str, Any]:
    audit_path = SCRIPTS_ROOT / "validate" / "audit_market_exclusion.py"
    if not audit_path.is_file():
        raise FileNotFoundError(f"Issue 28 market-exclusion validator missing: {audit_path}")
    completed = subprocess.run(
        [sys.executable, str(audit_path)],
        cwd=common.repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or "MARKET EXCLUSION AUDIT: PASS" not in completed.stdout:
        raise RuntimeError(
            "Market-exclusion preflight failed before Week 1 priors. "
            f"stdout={completed.stdout[-2000:]!r} stderr={completed.stderr[-2000:]!r}"
        )
    return {
        "passed": True,
        "validator": repo_relative(audit_path),
        "pass_marker": "MARKET EXCLUSION AUDIT: PASS",
    }


def write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    destination = path.resolve()
    prop = common.prop_root().resolve()
    try:
        destination.relative_to(prop)
    except ValueError as exc:
        raise ValueError(f"Week 1 prior log must remain under Prop Engine: {destination}") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n",
        prefix=f".{destination.name}.", suffix=".tmp",
        dir=destination.parent, delete=False,
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False, default=str)
            handle.write("\n")
        os.replace(temp_path, destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def blend(recent: pd.Series, career: pd.Series, recent_weight: float = RECENT_WEIGHT) -> pd.Series:
    r = as_num(recent)
    c = as_num(career)
    out = pd.Series(np.nan, index=r.index, dtype="float64")
    both = r.notna() & c.notna()
    out.loc[both] = recent_weight * r.loc[both] + (1.0 - recent_weight) * c.loc[both]
    out.loc[r.notna() & c.isna()] = r.loc[r.notna() & c.isna()]
    out.loc[r.isna() & c.notna()] = c.loc[r.isna() & c.notna()]
    return out


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    n = as_num(numerator)
    d = as_num(denominator)
    out = pd.Series(np.nan, index=n.index, dtype="float64")
    valid = n.notna() & d.notna() & d.gt(0)
    out.loc[valid] = n.loc[valid] / d.loc[valid]
    return out


def played_history(historical: pd.DataFrame, season: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    h = historical.copy()
    h["season"] = pd.to_numeric(h["season"], errors="raise").astype(int)
    h["week"] = pd.to_numeric(h["week"], errors="raise").astype(int)
    h["player_id"] = h["player_id"].map(clean)
    h["team"] = h["team"].map(norm_team)
    h["_played"] = flag(h["played_game_flag"]).eq(1)
    h["_kickoff"] = pd.to_datetime(h["kickoff_timestamp"], errors="coerce", utc=True)
    h = h.loc[h["season"].lt(season) & h["_played"] & h["player_id"].ne("")].copy()

    prior_season = season - 1
    prior_season_games = (
        h.loc[h["season"].eq(prior_season)]
        .groupby("player_id", sort=False)
        .size()
        .astype("int64")
    )
    modeled_career_games = h.groupby("player_id", sort=False).size().astype("int64")

    h = h.sort_values(["player_id", "_kickoff", "season", "week", "game_id"], kind="mergesort")
    prior_team = h.groupby("player_id", sort=False).tail(1).set_index("player_id")["team"]
    return prior_season_games, modeled_career_games, prior_team


def prior_feature_team(
    historical_features: pd.DataFrame,
    season: int,
) -> pd.Series:
    """Reproduce Issue 31 Week 1 latest prior-season feature-row team."""
    src = historical_features.copy()
    src["season"] = pd.to_numeric(src["season"], errors="raise").astype(int)
    src["week"] = pd.to_numeric(src["week"], errors="raise").astype(int)
    src["game_id"] = src["game_id"].map(clean)
    src["player_id"] = src["player_id"].map(clean)
    src["team"] = src["team"].map(norm_team)
    src["_kickoff"] = pd.to_datetime(
        src["kickoff_timestamp"], errors="raise", utc=True
    )
    src = src.loc[
        src["season"].eq(season - 1) & src["player_id"].ne("")
    ].copy()
    if src.empty:
        raise RuntimeError(
            f"Issue 32 has no prior-season historical feature rows for {season - 1}."
        )
    src = src.sort_values(
        ["player_id", "_kickoff", "week", "game_id"],
        kind="mergesort",
    )
    return (
        src.groupby("player_id", sort=False)
        .tail(1)
        .set_index("player_id")["team"]
    )


def history_role(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    offense_snap = as_num(frame["role_prior_offense_snap_pct"])
    defense_snap = as_num(frame["role_prior_defense_snap_pct"])
    lag_snap = pd.concat([offense_snap, defense_snap], axis=1).max(axis=1, skipna=True)
    lag_snap = lag_snap.where(offense_snap.notna() | defense_snap.notna())
    prior_snap = blend(
        lag_snap,
        as_num(frame["role_snap_pct_ewm5"]),
        recent_weight=0.70,
    ).clip(0.0, 1.0)

    offense_part = as_num(frame["role_prior_offense_participation"])
    defense_part = as_num(frame["role_prior_defense_participation"])
    lag_part = pd.concat([offense_part, defense_part], axis=1).max(axis=1, skipna=True)
    lag_part = lag_part.where(offense_part.notna() | defense_part.notna())
    prior_part = blend(
        lag_part,
        as_num(frame["role_participation_roll5"]),
        recent_weight=0.70,
    ).clip(0.0, 1.0)
    return prior_snap, prior_part


def efficiency_state(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return recent efficiency, career efficiency, and normalized position group."""
    groups = pd.Series(
        [norm_position_group(p, g) for p, g in zip(frame["position"], frame["position_group"])],
        index=frame.index,
        dtype="object",
    )
    recent = pd.Series(np.nan, index=frame.index, dtype="float64")
    career = pd.Series(np.nan, index=frame.index, dtype="float64")

    mappings = {
        "QB": ("player_yards_per_attempt_ewm5", "player_yards_per_attempt_career_prior"),
        "RB": ("player_yards_per_carry_ewm5", "player_yards_per_carry_career_prior"),
        "WR": ("player_yards_per_target_ewm5", "player_yards_per_target_career_prior"),
        "TE": ("player_yards_per_target_ewm5", "player_yards_per_target_career_prior"),
        "DL": ("player_tackle_rate_per_def_play_ewm5", "player_tackle_rate_per_def_play_career_prior"),
        "LB": ("player_tackle_rate_per_def_play_ewm5", "player_tackle_rate_per_def_play_career_prior"),
        "DB": ("player_tackle_rate_per_def_play_ewm5", "player_tackle_rate_per_def_play_career_prior"),
    }
    for group, (recent_col, career_col) in mappings.items():
        mask = groups.eq(group)
        recent.loc[mask] = as_num(frame.loc[mask, recent_col])
        career.loc[mask] = as_num(frame.loc[mask, career_col])

    spec = groups.eq("SPEC")
    if spec.any():
        recent.loc[spec] = safe_ratio(
            frame.loc[spec, "player_field_goals_made_roll5_mean"],
            frame.loc[spec, "player_field_goal_attempts_roll5_mean"],
        )
        career.loc[spec] = safe_ratio(
            frame.loc[spec, "player_field_goals_made_career_prior"],
            frame.loc[spec, "player_field_goal_attempts_career_prior"],
        )
    return recent, career, groups


def depth_role_projection(roles: pd.DataFrame) -> pd.Series:
    rank = as_num(roles["depth_rank"])
    starter = flag(roles["starter_flag"]).eq(1)
    primary_qb = flag(roles["primary_qb_flag"]).eq(1)
    primary_k = flag(roles["primary_kicker_flag"]).eq(1)
    primary = flag(roles["primary_role_flag"]).eq(1)
    committee = flag(roles["committee_role_flag"]).eq(1)

    base = pd.Series(0.10, index=roles.index, dtype="float64")
    base.loc[rank.eq(3)] = 0.18
    base.loc[rank.eq(2)] = 0.30
    base.loc[committee] = 0.42
    base.loc[primary & ~starter] = 0.60
    base.loc[starter] = 0.70
    base.loc[starter & committee] = 0.68
    base.loc[starter & primary] = 0.82
    base.loc[primary_qb] = 0.98
    base.loc[primary_k] = 1.00
    return base.clip(0.0, 1.0)


def weighted_position_priors(groups: pd.Series, career_eff: pd.Series, career_games: pd.Series) -> dict[str, float]:
    work = pd.DataFrame({
        "group": groups,
        "eff": as_num(career_eff),
        "games": as_num(career_games).fillna(0),
    })
    work = work.loc[work["group"].ne("") & work["eff"].notna() & work["games"].gt(0)].copy()
    priors: dict[str, float] = {}
    for group, rows in work.groupby("group", sort=True):
        weights = np.sqrt(rows["games"].clip(lower=1.0, upper=64.0).to_numpy(dtype="float64"))
        values = rows["eff"].to_numpy(dtype="float64")
        if weights.sum() > 0:
            priors[str(group)] = float(np.average(values, weights=weights))
    return priors


def build_priors(features: pd.DataFrame, roles: pd.DataFrame, historical: pd.DataFrame, historical_features: pd.DataFrame, season: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    common.require_columns(features, FEATURE_REQUIRED, "Issue 31 Week 1 current features")
    common.require_columns(roles, ROLE_REQUIRED, "Issue 30 Week 1 roles")
    common.require_columns(historical, HISTORICAL_REQUIRED, "historical player-game universe")
    common.require_columns(historical_features, PRIOR_FEATURE_REQUIRED, "historical player-game features")

    for label, frame in [("features", features), ("roles", roles)]:
        if not pd.to_numeric(frame["season"], errors="coerce").eq(season).all():
            raise ValueError(f"Issue 32 {label} contains season other than {season}.")
        if not pd.to_numeric(frame["week"], errors="coerce").eq(1).all():
            raise ValueError(f"Issue 32 {label} must be Week 1 only.")
        common.ensure_unique(frame, ["season", "week", "game_id", "player_id"], f"Issue 32 {label} grain")

    role_keys = roles[["season", "week", "game_id", "player_id"]].copy()
    feature_keys = features[["season", "week", "game_id", "player_id"]].copy()
    if len(role_keys) != len(feature_keys):
        raise ValueError("Issue 30 roles and Issue 31 features row counts differ.")
    probe = feature_keys.merge(role_keys, on=["season", "week", "game_id", "player_id"], how="outer", indicator=True, validate="one_to_one")
    if not probe["_merge"].eq("both").all():
        raise ValueError("Issue 30 roles and Issue 31 features do not have identical Week 1 grain.")

    role_use = roles[[
        "season", "week", "game_id", "player_id", "team", "position", "depth_rank",
        "starter_flag", "primary_qb_flag", "primary_kicker_flag", "primary_role_flag",
        "committee_role_flag", "role_confidence", "role_reason",
    ]].copy()
    role_use = role_use.rename(columns={"team": "_role_team", "position": "_role_position"})
    work = features.merge(role_use, on=["season", "week", "game_id", "player_id"], how="left", validate="one_to_one", sort=False)
    work["team"] = work["team"].map(norm_team)
    work["position"] = work["position"].map(norm_position)
    role_team = work["_role_team"].map(norm_team)
    role_position = work["_role_position"].map(norm_position)
    if not role_team.eq(work["team"]).all() or not role_position.eq(work["position"]).all():
        raise ValueError("Issue 30/31 current team or position mismatch.")

    prior_season_games_map, modeled_career_map, played_prior_team_map = played_history(historical, season)
    prior_feature_team_map = prior_feature_team(historical_features, season)
    work["prior_season_games"] = work["player_id"].map(prior_season_games_map).fillna(0).astype("int16")
    work["prior_team"] = work["player_id"].map(prior_feature_team_map).fillna("").astype(str)

    # career_games is the strict-prior realized/source-history counter produced
    # by Issue 12 and carried through the accepted Issue 31 feature table.  It
    # is intentionally NOT forced to equal the historical-universe
    # played_game_flag count.  The universe contains participation/backstop
    # played rows that can legitimately have no source-compatible player-form
    # observation, while Issue 12 can also include 2010-2011 prehistory.  These
    # counters therefore have different provenance and neither is a valid
    # lower/upper bound for the other.
    feature_career = as_num(work["history_history_games"]).fillna(0)
    if (feature_career < 0).any() or ((feature_career % 1) != 0).any():
        raise ValueError("history_history_games must be a nonnegative integer count.")
    work["career_games"] = feature_career.astype("int32")

    # Keep modeled played-game history only as an audit diagnostic.  It remains
    # authoritative for prior_season_games, but prior_team follows the exact
    # latest prior-season historical feature-row team used by Issue 31.
    modeled_career = work["player_id"].map(modeled_career_map).fillna(0).astype("int64")
    played_prior_team = work["player_id"].map(played_prior_team_map).fillna("").astype(str)
    feature_vs_played_prior_team_mismatch = (
        played_prior_team.ne("")
        & work["prior_team"].ne("")
        & played_prior_team.ne(work["prior_team"])
    )
    source_career = work["career_games"].astype("int64")
    modeled_gt_source = modeled_career.gt(source_career)
    source_gt_modeled = source_career.gt(modeled_career)

    veteran = work["career_games"].gt(0)
    missing_prior_team = veteran & work["prior_team"].eq("")
    if missing_prior_team.any():
        sample = work.loc[missing_prior_team, ["player_id", "team", "career_games"]].head(20).to_dict("records")
        raise ValueError(f"Veteran Week 1 player lacks a resolvable prior team; sample={sample}")

    work["new_team_flag"] = (veteran & work["prior_team"].ne(work["team"])).astype("int8")
    feature_new_team = flag(work["history_new_team_flag"])
    if not feature_new_team.eq(work["new_team_flag"]).all():
        bad = work.loc[feature_new_team.ne(work["new_team_flag"]), ["player_id", "team", "prior_team", "history_new_team_flag"]].head(20)
        raise ValueError(f"Issue 31 current new-team flag disagrees with its reconstructed prior-season feature team; sample={bad.to_dict('records')}")

    no_history = flag(work["history_no_nfl_history_flag"])
    if not no_history.eq((~veteran).astype("int8")).all():
        raise ValueError("Issue 31 no-NFL-history flag disagrees with career_games.")

    prior_snap, prior_part = history_role(work)
    target_share = blend(work["player_target_share_ewm5"], work["player_target_share_career_prior"], recent_weight=0.70).clip(0.0, 1.0)
    carry_share = blend(work["player_carry_share_ewm5"], work["player_carry_share_career_prior"], recent_weight=0.70).clip(0.0, 1.0)
    recent_eff, career_eff, groups = efficiency_state(work)
    blended_eff = blend(recent_eff, career_eff, recent_weight=RECENT_WEIGHT)

    same_team_veteran = veteran & work["new_team_flag"].eq(0)
    new_team_veteran = veteran & work["new_team_flag"].eq(1)
    rookie = ~veteran

    work["prior_snap_share"] = prior_snap.where(same_team_veteran)
    work["prior_participation"] = prior_part.where(same_team_veteran)
    work["prior_target_share"] = target_share.where(same_team_veteran)
    work["prior_carry_share"] = carry_share.where(same_team_veteran)
    work["prior_efficiency"] = blended_eff.where(same_team_veteran)
    work.loc[new_team_veteran, "prior_efficiency"] = career_eff.loc[new_team_veteran]
    work.loc[rookie, "prior_efficiency"] = np.nan

    position_priors = weighted_position_priors(groups, career_eff, work["career_games"])
    work["position_prior"] = groups.map(position_priors).astype("float64")
    rookie_missing_position = rookie & work["position_prior"].isna()
    if rookie_missing_position.any():
        sample = work.loc[rookie_missing_position, ["player_id", "team", "position", "position_group"]].head(20).to_dict("records")
        raise ValueError(f"Rookie position prior unavailable; refusing invented production. sample={sample}")

    base_role = depth_role_projection(work)
    historical_role = work["prior_participation"].where(work["prior_participation"].notna(), work["prior_snap_share"])
    role_projection = base_role.copy()
    blend_mask = same_team_veteran & historical_role.notna()
    role_projection.loc[blend_mask] = (
        ROLE_HISTORY_WEIGHT * historical_role.loc[blend_mask]
        + ROLE_DEPTH_WEIGHT * base_role.loc[blend_mask]
    )
    role_projection = role_projection.clip(0.0, 1.0)
    work["week1_role_projection"] = role_projection

    current_starter = flag(work["starter_flag"]).eq(1)
    current_primary = flag(work["primary_role_flag"]).eq(1) | flag(work["primary_qb_flag"]).eq(1) | flag(work["primary_kicker_flag"]).eq(1)
    role_gap = (base_role - historical_role).abs()
    new_role = same_team_veteran & (
        (current_starter & historical_role.fillna(0).lt(0.45))
        | (current_primary & historical_role.fillna(0).lt(0.50))
        | (historical_role.notna() & role_gap.gt(0.30))
        | historical_role.isna()
    )

    uncertainty = pd.Series(1.0, index=work.index, dtype="float64")
    uncertainty.loc[rookie] *= ROOKIE_UNCERTAINTY
    uncertainty.loc[new_team_veteran] *= NEW_TEAM_UNCERTAINTY
    uncertainty.loc[new_role] *= NEW_ROLE_UNCERTAINTY
    stale = veteran & work["prior_season_games"].eq(0)
    uncertainty.loc[stale] *= NO_PRIOR_SEASON_UNCERTAINTY
    low_history = veteran & work["career_games"].le(4)
    uncertainty.loc[low_history] *= LOW_HISTORY_UNCERTAINTY
    committee = flag(work["committee_role_flag"]).eq(1)
    uncertainty.loc[committee] *= COMMITTEE_UNCERTAINTY

    confidence = as_num(work["role_confidence"]).fillna(0.5).clip(0.0, 1.0)
    uncertainty *= 1.0 + (1.0 - confidence) * 0.50

    kicker_ambiguous = work["role_reason"].fillna("").astype(str).str.contains("kicker_ambiguous_", case=False, regex=False)
    uncertainty.loc[kicker_ambiguous] *= KICKER_AMBIGUITY_UNCERTAINTY
    work["week1_uncertainty_multiplier"] = uncertainty.clip(lower=1.0, upper=MAX_UNCERTAINTY)

    # Rookies must not receive fabricated player NFL production/usage priors.
    rookie_player_prior_cols = [
        "prior_snap_share", "prior_participation", "prior_target_share",
        "prior_carry_share", "prior_efficiency",
    ]
    if work.loc[rookie, rookie_player_prior_cols].notna().any().any():
        raise ValueError("Rookie rows contain invented player NFL production/usage priors.")

    if work.loc[new_team_veteran, ["prior_snap_share", "prior_participation", "prior_target_share", "prior_carry_share"]].notna().any().any():
        raise ValueError("New-team rows imported old-team role/share priors.")

    if not work["week1_role_projection"].between(0.0, 1.0).all():
        raise ValueError("week1_role_projection must be in [0,1].")
    if not work["week1_uncertainty_multiplier"].ge(1.0).all():
        raise ValueError("Week 1 uncertainty multiplier must be >=1.")
    if not work.loc[rookie, "week1_uncertainty_multiplier"].gt(1.0).all():
        raise ValueError("Every rookie must have widened Week 1 uncertainty.")
    if not work.loc[new_team_veteran, "week1_uncertainty_multiplier"].gt(1.0).all():
        raise ValueError("Every new-team veteran must have widened Week 1 uncertainty.")
    if not work.loc[new_role, "week1_uncertainty_multiplier"].gt(1.0).all():
        raise ValueError("Every materially new role must have widened Week 1 uncertainty.")

    out = work[OUTPUT_COLUMNS].copy()
    out["player_id"] = out["player_id"].map(clean).astype("string")
    out["team"] = out["team"].map(norm_team).astype("string")
    out["position"] = out["position"].map(norm_position).astype("string")
    out["prior_team"] = out["prior_team"].map(norm_team).astype("string")
    out["prior_season_games"] = out["prior_season_games"].astype("int16")
    out["career_games"] = out["career_games"].astype("int32")
    out["new_team_flag"] = out["new_team_flag"].astype("int8")
    for col in [
        "prior_snap_share", "prior_participation", "prior_target_share",
        "prior_carry_share", "prior_efficiency", "position_prior", "depth_rank",
        "week1_role_projection", "week1_uncertainty_multiplier",
    ]:
        out[col] = as_num(out[col]).astype("float64")

    common.ensure_unique(out, ["player_id"], "Issue 32 Week 1 priors player grain")
    common.reject_forbidden_feature_columns(list(out.columns), common.load_config())

    audit = {
        "same_team_veterans": int(same_team_veteran.sum()),
        "new_team_veterans": int(new_team_veteran.sum()),
        "rookies": int(rookie.sum()),
        "new_roles": int(new_role.sum()),
        "stale_history_players": int(stale.sum()),
        "low_history_players": int(low_history.sum()),
        "committee_players": int(committee.sum()),
        "ambiguous_kickers": int(kicker_ambiguous.sum()),
        "career_counter_provenance": "Issue12 strict-prior realized/source history; not historical-universe played_game_flag",
        "modeled_played_games_gt_source_history_players": int(modeled_gt_source.sum()),
        "source_history_gt_modeled_played_games_players": int(source_gt_modeled.sum()),
        "prior_feature_team_differs_from_latest_played_team_players": int(feature_vs_played_prior_team_mismatch.sum()),
        "position_prior_groups": {k: float(v) for k, v in sorted(position_priors.items())},
        "prior_efficiency_mapping": {
            "QB": "yards_per_attempt",
            "RB": "yards_per_carry",
            "WR": "yards_per_target",
            "TE": "yards_per_target",
            "SPEC": "field_goal_conversion",
            "DL": "tackle_rate_per_defensive_play",
            "LB": "tackle_rate_per_defensive_play",
            "DB": "tackle_rate_per_defensive_play",
        },
    }
    return out, audit


def main() -> int:
    args = parse_args()
    config = common.load_config()
    season = resolve_season(args, config)
    market = run_market_preflight()
    repo = common.repo_root()
    prop = common.prop_root()

    roles_path = prop / "data" / "current" / f"{season}_week_1_roles.parquet"
    features_path = prop / "data" / "current" / "features" / f"{season}_week_1_features.parquet"
    historical_path = repo / config["paths"]["historical_universe"]
    historical_features_path = repo / config["paths"]["historical_features"]
    output_path = prop / "data" / "current" / f"{season}_week_1_priors.parquet"
    log_path = prop / "logs" / f"week1_priors_{season}.json"

    for path in [roles_path, features_path, historical_path, historical_features_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Issue 32 required input missing: {path}")

    roles = pd.read_parquet(roles_path)
    features = pd.read_parquet(features_path)
    historical = pd.read_parquet(historical_path, columns=HISTORICAL_REQUIRED)
    historical_features = pd.read_parquet(
        historical_features_path, columns=PRIOR_FEATURE_REQUIRED
    )

    priors, audit = build_priors(features, roles, historical, historical_features, season)
    common.write_parquet_atomic(priors[OUTPUT_COLUMNS], output_path)

    payload = {
        "script": Path(__file__).name,
        "status": "passed",
        "season": season,
        "week": 1,
        "rows": int(len(priors)),
        "teams": int(priors["team"].nunique()),
        "rookies": audit["rookies"],
        "new_team_veterans": audit["new_team_veterans"],
        "new_roles": audit["new_roles"],
        "uncertainty_widened_rows": int(priors["week1_uncertainty_multiplier"].gt(1.0).sum()),
        "market_exclusion_passed": bool(market["passed"]),
        "market_features_used": False,
        "output": repo_relative(output_path),
        "log": repo_relative(log_path),
    }
    log_payload = {
        **payload,
        "inputs": {
            "roles": repo_relative(roles_path),
            "features": repo_relative(features_path),
            "historical_universe": repo_relative(historical_path),
            "historical_features": repo_relative(historical_features_path),
        },
        "policy": {
            "returning_player_recent_plus_career": True,
            "prior_team_source": "latest_prior_season_historical_feature_row_team",
            "new_team_career_efficiency_only": True,
            "new_team_old_team_role_share_imported": False,
            "rookie_player_nfl_production_invented": False,
            "rookie_position_depth_prior": True,
            "rookie_uncertainty_widened": True,
            "new_role_uncertainty_widened": True,
            "position_prior_source": "current_week1_veteran_career_efficiency_by_position_group",
            "market_exclusion_preflight": True,
        },
        "constants": {
            "recent_weight": RECENT_WEIGHT,
            "career_weight": CAREER_WEIGHT,
            "role_history_weight": ROLE_HISTORY_WEIGHT,
            "role_depth_weight": ROLE_DEPTH_WEIGHT,
            "rookie_uncertainty": ROOKIE_UNCERTAINTY,
            "new_team_uncertainty": NEW_TEAM_UNCERTAINTY,
            "new_role_uncertainty": NEW_ROLE_UNCERTAINTY,
            "no_prior_season_uncertainty": NO_PRIOR_SEASON_UNCERTAINTY,
            "low_history_uncertainty": LOW_HISTORY_UNCERTAINTY,
            "committee_uncertainty": COMMITTEE_UNCERTAINTY,
            "kicker_ambiguity_uncertainty": KICKER_AMBIGUITY_UNCERTAINTY,
            "max_uncertainty": MAX_UNCERTAINTY,
        },
        "audit": audit,
        "market_preflight": market,
    }
    write_json_atomic(log_payload, log_path)
    common.log_run(Path(__file__).name, payload)

    print(json.dumps({"script": Path(__file__).name, "payload": payload}, sort_keys=True, separators=(",", ":")))
    print("WEEK 1 PRIORS BUILD: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
