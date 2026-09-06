#!/usr/bin/env python3
"""Independent acceptance validator for Prop Engine Issue 32."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = THIS_DIR / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import common

OUTPUT_COLUMNS = [
    "player_id", "team", "position", "prior_season_games", "career_games",
    "prior_team", "new_team_flag", "prior_snap_share", "prior_participation",
    "prior_target_share", "prior_carry_share", "prior_efficiency",
    "position_prior", "depth_rank", "week1_role_projection",
    "week1_uncertainty_multiplier",
]

HISTORICAL_REQUIRED = [
    "season", "week", "game_id", "player_id", "team",
    "kickoff_timestamp", "played_game_flag",
]

PRIOR_FEATURE_REQUIRED = [
    "season", "week", "game_id", "kickoff_timestamp", "player_id", "team",
]

FEATURE_REQUIRED = [
    "season", "week", "game_id", "player_id", "team", "position",
    "position_group", "history_history_games", "history_no_nfl_history_flag",
    "history_new_team_flag", "role_prior_offense_snap_pct",
    "role_prior_defense_snap_pct", "role_snap_pct_ewm5",
    "role_prior_offense_participation", "role_prior_defense_participation",
    "role_participation_roll5", "player_target_share_ewm5",
    "player_target_share_career_prior", "player_carry_share_ewm5",
    "player_carry_share_career_prior", "player_yards_per_attempt_ewm5",
    "player_yards_per_attempt_career_prior", "player_yards_per_carry_ewm5",
    "player_yards_per_carry_career_prior", "player_yards_per_target_ewm5",
    "player_yards_per_target_career_prior", "player_field_goal_attempts_roll5_mean",
    "player_field_goals_made_roll5_mean", "player_field_goal_attempts_career_prior",
    "player_field_goals_made_career_prior", "player_tackle_rate_per_def_play_ewm5",
    "player_tackle_rate_per_def_play_career_prior",
]

ROLE_REQUIRED = [
    "season", "week", "game_id", "team", "player_id", "position", "depth_rank",
    "starter_flag", "primary_qb_flag", "primary_kicker_flag", "primary_role_flag",
    "committee_role_flag", "role_confidence", "role_reason",
]

TEAM_ALIASES = {"SD": "LAC", "OAK": "LV", "STL": "LAR"}
POS_ALIASES = {
    "HB": "RB", "FB": "RB", "H-BACK": "RB", "K": "SPEC", "PK": "SPEC",
    "DE": "DL", "LDE": "DL", "RDE": "DL", "DT": "DL", "LDT": "DL",
    "RDT": "DL", "NT": "DL", "EDGE": "DL", "ILB": "LB", "OLB": "LB",
    "MLB": "LB", "WLB": "LB", "SLB": "LB", "CB": "DB", "LCB": "DB",
    "RCB": "DB", "NB": "DB", "S": "DB", "FS": "DB", "SS": "DB",
}

RECENT_WEIGHT = 0.65
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


def clean(v: Any) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(v).strip()
    if text.casefold() in {"", "nan", "none", "null", "<na>", "nat"}:
        return ""
    return text


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).astype(float)


def flg(s: pd.Series) -> pd.Series:
    return num(s).fillna(0).gt(0).astype("int8")


def team(v: Any) -> str:
    x = common.normalize_team(v)
    return TEAM_ALIASES.get(x, x)


def pos(v: Any) -> str:
    return clean(v).upper().replace(" ", "")


def group(position: Any, position_group: Any) -> str:
    g = pos(position_group)
    p = pos(position)
    if g in {"QB", "RB", "WR", "TE", "DL", "LB", "DB", "SPEC"}:
        return g
    if g in POS_ALIASES:
        return POS_ALIASES[g]
    if p in {"QB", "RB", "WR", "TE", "DL", "LB", "DB", "SPEC"}:
        return p
    return POS_ALIASES.get(p, p)


def blend(recent: pd.Series, career: pd.Series, w: float) -> pd.Series:
    r, c = num(recent), num(career)
    o = pd.Series(np.nan, index=r.index, dtype=float)
    both = r.notna() & c.notna()
    o.loc[both] = w * r.loc[both] + (1 - w) * c.loc[both]
    o.loc[r.notna() & c.isna()] = r.loc[r.notna() & c.isna()]
    o.loc[r.isna() & c.notna()] = c.loc[r.isna() & c.notna()]
    return o


def ratio(n: pd.Series, d: pd.Series) -> pd.Series:
    n, d = num(n), num(d)
    o = pd.Series(np.nan, index=n.index, dtype=float)
    ok = n.notna() & d.notna() & d.gt(0)
    o.loc[ok] = n.loc[ok] / d.loc[ok]
    return o


def historical_maps(hist: pd.DataFrame, season: int):
    x = hist.copy()
    x["season"] = pd.to_numeric(x["season"], errors="raise").astype(int)
    x["week"] = pd.to_numeric(x["week"], errors="raise").astype(int)
    x["player_id"] = x["player_id"].map(clean)
    x["team"] = x["team"].map(team)
    x["_played"] = flg(x["played_game_flag"]).eq(1)
    x["_kickoff"] = pd.to_datetime(x["kickoff_timestamp"], errors="coerce", utc=True)
    x = x.loc[x["season"].lt(season) & x["_played"] & x["player_id"].ne("")].copy()
    prior_games = x.loc[x["season"].eq(season - 1)].groupby("player_id").size()
    career_modeled = x.groupby("player_id").size()
    x = x.sort_values(["player_id", "_kickoff", "season", "week", "game_id"], kind="mergesort")
    prior_team = x.groupby("player_id").tail(1).set_index("player_id")["team"]
    return prior_games, career_modeled, prior_team


def prior_feature_team(historical_features: pd.DataFrame, season: int) -> pd.Series:
    src = historical_features.copy()
    src["season"] = pd.to_numeric(src["season"], errors="raise").astype(int)
    src["week"] = pd.to_numeric(src["week"], errors="raise").astype(int)
    src["game_id"] = src["game_id"].map(clean)
    src["player_id"] = src["player_id"].map(clean)
    src["team"] = src["team"].map(team)
    src["_kickoff"] = pd.to_datetime(src["kickoff_timestamp"], errors="raise", utc=True)
    src = src.loc[src["season"].eq(season - 1) & src["player_id"].ne("")].copy()
    if src.empty:
        raise AssertionError(f"No prior-season historical feature rows for {season - 1}.")
    src = src.sort_values(["player_id", "_kickoff", "week", "game_id"], kind="mergesort")
    return src.groupby("player_id", sort=False).tail(1).set_index("player_id")["team"]


def efficiency(frame: pd.DataFrame):
    groups = pd.Series([group(p, g) for p, g in zip(frame["position"], frame["position_group"])], index=frame.index)
    recent = pd.Series(np.nan, index=frame.index, dtype=float)
    career = pd.Series(np.nan, index=frame.index, dtype=float)
    mapping = {
        "QB": ("player_yards_per_attempt_ewm5", "player_yards_per_attempt_career_prior"),
        "RB": ("player_yards_per_carry_ewm5", "player_yards_per_carry_career_prior"),
        "WR": ("player_yards_per_target_ewm5", "player_yards_per_target_career_prior"),
        "TE": ("player_yards_per_target_ewm5", "player_yards_per_target_career_prior"),
        "DL": ("player_tackle_rate_per_def_play_ewm5", "player_tackle_rate_per_def_play_career_prior"),
        "LB": ("player_tackle_rate_per_def_play_ewm5", "player_tackle_rate_per_def_play_career_prior"),
        "DB": ("player_tackle_rate_per_def_play_ewm5", "player_tackle_rate_per_def_play_career_prior"),
    }
    for g, (rc, cc) in mapping.items():
        m = groups.eq(g)
        recent.loc[m] = num(frame.loc[m, rc])
        career.loc[m] = num(frame.loc[m, cc])
    m = groups.eq("SPEC")
    if m.any():
        recent.loc[m] = ratio(frame.loc[m, "player_field_goals_made_roll5_mean"], frame.loc[m, "player_field_goal_attempts_roll5_mean"])
        career.loc[m] = ratio(frame.loc[m, "player_field_goals_made_career_prior"], frame.loc[m, "player_field_goal_attempts_career_prior"])
    return recent, career, groups


def role_history(frame: pd.DataFrame):
    osnap, dsnap = num(frame["role_prior_offense_snap_pct"]), num(frame["role_prior_defense_snap_pct"])
    lag_snap = pd.concat([osnap, dsnap], axis=1).max(axis=1, skipna=True).where(osnap.notna() | dsnap.notna())
    snap = blend(lag_snap, frame["role_snap_pct_ewm5"], 0.70).clip(0, 1)
    opart, dpart = num(frame["role_prior_offense_participation"]), num(frame["role_prior_defense_participation"])
    lag_part = pd.concat([opart, dpart], axis=1).max(axis=1, skipna=True).where(opart.notna() | dpart.notna())
    part = blend(lag_part, frame["role_participation_roll5"], 0.70).clip(0, 1)
    return snap, part


def depth_base(frame: pd.DataFrame):
    rank = num(frame["depth_rank"])
    starter = flg(frame["starter_flag"]).eq(1)
    pq = flg(frame["primary_qb_flag"]).eq(1)
    pk = flg(frame["primary_kicker_flag"]).eq(1)
    primary = flg(frame["primary_role_flag"]).eq(1)
    committee = flg(frame["committee_role_flag"]).eq(1)
    b = pd.Series(0.10, index=frame.index, dtype=float)
    b.loc[rank.eq(3)] = 0.18
    b.loc[rank.eq(2)] = 0.30
    b.loc[committee] = 0.42
    b.loc[primary & ~starter] = 0.60
    b.loc[starter] = 0.70
    b.loc[starter & committee] = 0.68
    b.loc[starter & primary] = 0.82
    b.loc[pq] = 0.98
    b.loc[pk] = 1.00
    return b.clip(0, 1)


def position_priors(groups: pd.Series, career_eff: pd.Series, games: pd.Series):
    x = pd.DataFrame({"g": groups, "e": num(career_eff), "n": num(games).fillna(0)})
    x = x.loc[x["g"].ne("") & x["e"].notna() & x["n"].gt(0)]
    result = {}
    for g, rows in x.groupby("g", sort=True):
        w = np.sqrt(rows["n"].clip(1, 64).to_numpy(float))
        result[str(g)] = float(np.average(rows["e"].to_numpy(float), weights=w))
    return result


def approx_equal(a: pd.Series, b: pd.Series, tol=1e-10) -> pd.Series:
    aa, bb = num(a), num(b)
    both_null = aa.isna() & bb.isna()
    both = aa.notna() & bb.notna() & np.isclose(aa, bb, rtol=1e-9, atol=tol)
    return both_null | both


def run_market() -> None:
    path = common.prop_root() / "scripts" / "validate" / "audit_market_exclusion.py"
    completed = subprocess.run([sys.executable, str(path)], cwd=common.repo_root(), capture_output=True, text=True, check=False)
    if completed.returncode != 0 or "MARKET EXCLUSION AUDIT: PASS" not in completed.stdout:
        raise AssertionError("Issue 28 market exclusion did not pass for Issue 32.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None)
    args = ap.parse_args()
    config = common.load_config()
    season = int(args.season) if args.season is not None else int(config["seasons"]["current"])
    repo, prop = common.repo_root(), common.prop_root()

    builder = prop / "scripts" / "project" / "build_week1_priors.py"
    output_path = prop / "data" / "current" / f"{season}_week_1_priors.parquet"
    log_path = prop / "logs" / f"week1_priors_{season}.json"
    roles_path = prop / "data" / "current" / f"{season}_week_1_roles.parquet"
    features_path = prop / "data" / "current" / "features" / f"{season}_week_1_features.parquet"
    historical_path = repo / config["paths"]["historical_universe"]
    historical_features_path = repo / config["paths"]["historical_features"]

    print("CHECK 01: required builder/output/log and exact headers")
    for p in [builder, output_path, log_path, roles_path, features_path, historical_path, historical_features_path]:
        if not p.is_file():
            raise AssertionError(f"Missing Issue 32 artifact/input: {p}")
    out = pd.read_parquet(output_path)
    if list(out.columns) != OUTPUT_COLUMNS:
        raise AssertionError(f"Issue 32 exact header mismatch: {list(out.columns)}")
    common.ensure_unique(out, ["player_id"], "Issue 32 output")

    roles = pd.read_parquet(roles_path)
    features = pd.read_parquet(features_path)
    hist = pd.read_parquet(historical_path, columns=HISTORICAL_REQUIRED)
    historical_features = pd.read_parquet(historical_features_path, columns=PRIOR_FEATURE_REQUIRED)
    common.require_columns(roles, ROLE_REQUIRED, "Issue 30 roles")
    common.require_columns(features, FEATURE_REQUIRED, "Issue 31 features")
    if len(out) != len(roles) or len(out) != len(features):
        raise AssertionError("Issue 32 row count must equal Issue 30/31 eligible Week 1 rows.")

    print("CHECK 02: reconstruct prior-season games, career history, prior team, and new-team flag")
    pg, modeled, played_pt = historical_maps(hist, season)
    pt = prior_feature_team(historical_features, season)
    idx = out.set_index("player_id")
    feat = features.copy()
    feat["player_id"] = feat["player_id"].map(clean)
    feat = feat.set_index("player_id")
    expected_prior_games = idx.index.to_series().map(pg).fillna(0).astype(int)
    if not idx["prior_season_games"].astype(int).eq(expected_prior_games).all():
        raise AssertionError("prior_season_games mismatch.")
    if not idx["career_games"].astype(int).eq(num(feat.loc[idx.index, "history_history_games"]).fillna(0).astype(int).to_numpy()).all():
        raise AssertionError("career_games does not match strict-prior history_history_games.")
    expected_prior_team = idx.index.to_series().map(pt).fillna("").map(team)
    if not idx["prior_team"].fillna("").map(team).eq(expected_prior_team).all():
        raise AssertionError("prior_team mismatch.")
    current_team = idx["team"].map(team)
    veteran = idx["career_games"].astype(int).gt(0)
    expected_new = (veteran & expected_prior_team.ne(current_team)).astype(int)
    if not idx["new_team_flag"].astype(int).eq(expected_new).all():
        raise AssertionError("new_team_flag mismatch.")

    expected_feature_new = flg(feat.loc[idx.index, "history_new_team_flag"]).astype(int)
    if not expected_feature_new.reset_index(drop=True).eq(expected_new.reset_index(drop=True)).all():
        raise AssertionError("Issue 31 history_new_team_flag does not match its prior-season historical feature-row team.")

    print("CHECK 03: independently reconstruct returning/new-team/rookie priors")
    role = roles.copy()
    role["player_id"] = role["player_id"].map(clean)
    role = role.set_index("player_id").loc[idx.index].reset_index()
    f = features.copy()
    f["player_id"] = f["player_id"].map(clean)
    f = f.set_index("player_id").loc[idx.index].reset_index()
    snap, part = role_history(f)
    t_share = blend(f["player_target_share_ewm5"], f["player_target_share_career_prior"], 0.70).clip(0, 1)
    c_share = blend(f["player_carry_share_ewm5"], f["player_carry_share_career_prior"], 0.70).clip(0, 1)
    recent_eff, career_eff, groups = efficiency(f)
    blended_eff = blend(recent_eff, career_eff, RECENT_WEIGHT)
    same = veteran.reset_index(drop=True) & expected_new.reset_index(drop=True).eq(0)
    new_team = veteran.reset_index(drop=True) & expected_new.reset_index(drop=True).eq(1)
    rookie = ~veteran.reset_index(drop=True)

    exp_snap = snap.where(same)
    exp_part = part.where(same)
    exp_t = t_share.where(same)
    exp_c = c_share.where(same)
    exp_eff = blended_eff.where(same)
    exp_eff.loc[new_team] = career_eff.loc[new_team]
    exp_eff.loc[rookie] = np.nan
    for col, expected in [
        ("prior_snap_share", exp_snap), ("prior_participation", exp_part),
        ("prior_target_share", exp_t), ("prior_carry_share", exp_c),
        ("prior_efficiency", exp_eff),
    ]:
        if not approx_equal(out[col].reset_index(drop=True), expected).all():
            raise AssertionError(f"{col} policy mismatch.")
    rookie_cols = ["prior_snap_share", "prior_participation", "prior_target_share", "prior_carry_share", "prior_efficiency"]
    if out.loc[rookie.to_numpy(), rookie_cols].notna().any().any():
        raise AssertionError("Rookie contains invented NFL player production/usage prior.")
    if out.loc[new_team.to_numpy(), ["prior_snap_share", "prior_participation", "prior_target_share", "prior_carry_share"]].notna().any().any():
        raise AssertionError("New-team veteran imported old-team role/share prior.")

    print("CHECK 04: independently reconstruct position/depth role prior and uncertainty widening")
    pri = position_priors(groups, career_eff, out["career_games"].reset_index(drop=True))
    exp_pos = groups.map(pri).astype(float)
    if not approx_equal(out["position_prior"].reset_index(drop=True), exp_pos).all():
        raise AssertionError("position_prior mismatch.")
    if out.loc[rookie.to_numpy(), "position_prior"].isna().any():
        raise AssertionError("Rookie missing position prior.")

    base = depth_base(role)
    hist_role = exp_part.where(exp_part.notna(), exp_snap)
    exp_role = base.copy()
    bm = same & hist_role.notna()
    exp_role.loc[bm] = ROLE_HISTORY_WEIGHT * hist_role.loc[bm] + ROLE_DEPTH_WEIGHT * base.loc[bm]
    exp_role = exp_role.clip(0, 1)
    if not approx_equal(out["week1_role_projection"].reset_index(drop=True), exp_role).all():
        raise AssertionError("week1_role_projection mismatch.")

    starter = flg(role["starter_flag"]).eq(1)
    primary = flg(role["primary_role_flag"]).eq(1) | flg(role["primary_qb_flag"]).eq(1) | flg(role["primary_kicker_flag"]).eq(1)
    gap = (base - hist_role).abs()
    new_role = same & ((starter & hist_role.fillna(0).lt(0.45)) | (primary & hist_role.fillna(0).lt(0.50)) | (hist_role.notna() & gap.gt(0.30)) | hist_role.isna())
    u = pd.Series(1.0, index=role.index, dtype=float)
    u.loc[rookie] *= ROOKIE_UNCERTAINTY
    u.loc[new_team] *= NEW_TEAM_UNCERTAINTY
    u.loc[new_role] *= NEW_ROLE_UNCERTAINTY
    stale = veteran.reset_index(drop=True) & out["prior_season_games"].reset_index(drop=True).eq(0)
    u.loc[stale] *= NO_PRIOR_SEASON_UNCERTAINTY
    low = veteran.reset_index(drop=True) & out["career_games"].reset_index(drop=True).le(4)
    u.loc[low] *= LOW_HISTORY_UNCERTAINTY
    committee = flg(role["committee_role_flag"]).eq(1)
    u.loc[committee] *= COMMITTEE_UNCERTAINTY
    conf = num(role["role_confidence"]).fillna(0.5).clip(0, 1)
    u *= 1 + (1 - conf) * 0.50
    amb = role["role_reason"].fillna("").astype(str).str.contains("kicker_ambiguous_", case=False, regex=False)
    u.loc[amb] *= KICKER_AMBIGUITY_UNCERTAINTY
    u = u.clip(1, MAX_UNCERTAINTY)
    if not approx_equal(out["week1_uncertainty_multiplier"].reset_index(drop=True), u).all():
        raise AssertionError("week1_uncertainty_multiplier mismatch.")
    if not out.loc[rookie.to_numpy(), "week1_uncertainty_multiplier"].gt(1).all():
        raise AssertionError("Rookie uncertainty was not widened.")
    if not out.loc[new_role.to_numpy(), "week1_uncertainty_multiplier"].gt(1).all():
        raise AssertionError("New-role uncertainty was not widened.")

    print("CHECK 05: market exclusion, log policy, and final summary")
    common.reject_forbidden_feature_columns(list(out.columns), config)
    run_market()
    log = json.loads(log_path.read_text(encoding="utf-8-sig"))
    required_markers = {
        "returning_player_recent_plus_career": True,
        "new_team_career_efficiency_only": True,
        "new_team_old_team_role_share_imported": False,
        "rookie_player_nfl_production_invented": False,
        "rookie_position_depth_prior": True,
        "rookie_uncertainty_widened": True,
        "new_role_uncertainty_widened": True,
    }
    for k, v in required_markers.items():
        if log.get("policy", {}).get(k) is not v:
            raise AssertionError(f"Issue 32 log policy marker mismatch: {k}")
    if log.get("policy", {}).get("prior_team_source") != "latest_prior_season_historical_feature_row_team":
        raise AssertionError("Issue 32 prior_team provenance marker mismatch.")
    if log.get("market_features_used") is not False or log.get("status") != "passed":
        raise AssertionError("Issue 32 log status/market policy invalid.")

    print(f"season={season}")
    print("week=1")
    print(f"rows={len(out)}")
    print(f"teams={out['team'].nunique()}")
    print(f"same_team_veterans={int(same.sum())}")
    print(f"new_team_veterans={int(new_team.sum())}")
    print(f"rookies={int(rookie.sum())}")
    print(f"new_roles={int(new_role.sum())}")
    print(f"uncertainty_widened_rows={int(out['week1_uncertainty_multiplier'].gt(1).sum())}")
    print("rookie_nfl_production_invented=false")
    print("old_team_role_share_imported=false")
    print("market_features_used=false")
    print("ISSUE 32 ACCEPTANCE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
