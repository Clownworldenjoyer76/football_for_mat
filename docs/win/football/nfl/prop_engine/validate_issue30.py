#!/usr/bin/env python3
"""Independent acceptance validator for Prop Engine Issue 30."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROP_REL = Path("docs/win/football/nfl/prop_engine")
SCRIPT_REL = PROP_REL / "scripts/project/select_roles.py"

OUTPUT_COLUMNS = [
    "season", "week", "game_id", "team", "player_id", "player_name",
    "position", "depth_rank", "starter_flag", "primary_qb_flag",
    "primary_kicker_flag", "primary_role_flag", "committee_role_flag",
    "role_confidence", "role_reason",
]

KICKER_POSITIONS = {"K", "PK"}


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


def find_repo_root() -> Path:
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / "docs/win/football/nfl/prop_engine/config/prop_engine.yaml").exists():
            return parent
    raise RuntimeError("Unable to locate repository root.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    return parser.parse_args()


def eligible_mask(frame: pd.DataFrame) -> pd.Series:
    return frame["eligibility_status"].fillna("").astype(str).str.strip().str.casefold().eq("eligible")


def rank_series(frame: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(frame["depth_rank"], errors="coerce").astype(float)


def flag(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def main() -> int:
    args = parse_args()
    repo = find_repo_root()
    prop = repo / PROP_REL
    if str(prop / "scripts") not in sys.path:
        sys.path.insert(0, str(prop / "scripts"))
    import common

    season = int(args.season)
    week = int(args.week)
    builder = repo / SCRIPT_REL
    universe_path = prop / "data/current" / f"{season}_week_{week}_universe.parquet"
    roles_path = prop / "data/current" / f"{season}_week_{week}_roles.parquet"
    log_path = prop / "logs" / f"current_roles_{season}_week_{week}.json"

    print("CHECK 01: required role selector, output, log, and exact headers")
    for path in [builder, universe_path, roles_path, log_path]:
        if not path.exists():
            raise AssertionError(f"Missing Issue 30 artifact: {path}")
    roles = pd.read_parquet(roles_path)
    if list(roles.columns) != OUTPUT_COLUMNS:
        raise AssertionError(f"Role header mismatch. expected={OUTPUT_COLUMNS} actual={list(roles.columns)}")
    log = json.loads(log_path.read_text(encoding="utf-8"))
    if log.get("status") != "passed":
        raise AssertionError("Role log status is not passed.")

    print("CHECK 02: grain, eligibility subset, flags, and confidence")
    universe = pd.read_parquet(universe_path)
    common.require_columns(
        universe,
        [
            "season", "week", "game_id", "team", "player_id", "player_name",
            "position", "depth_rank", "depth_starter_flag", "depth_backup_flag",
            "depth_injury", "injury_game_status", "eligibility_status",
        ],
        "Issue 29 universe",
    )
    common.ensure_unique(roles, ["season", "week", "game_id", "team", "player_id"], "Issue 30 roles")
    if not roles["season"].eq(season).all() or not roles["week"].eq(week).all():
        raise AssertionError("Role output contains rows outside requested season/week.")
    universe_keys = set(
        universe.loc[eligible_mask(universe), ["season", "week", "game_id", "team", "player_id"]]
        .astype(str).agg("|".join, axis=1)
    )
    role_keys = set(roles[["season", "week", "game_id", "team", "player_id"]].astype(str).agg("|".join, axis=1))
    if role_keys != universe_keys:
        missing = list(universe_keys - role_keys)[:5]
        extra = list(role_keys - universe_keys)[:5]
        raise AssertionError(f"Roles must cover exactly the eligible universe rows. missing={missing} extra={extra}")
    for column in ["starter_flag", "primary_qb_flag", "primary_kicker_flag", "primary_role_flag", "committee_role_flag"]:
        values = set(pd.to_numeric(roles[column], errors="raise").astype(int).unique())
        if not values.issubset({0, 1}):
            raise AssertionError(f"{column} contains nonbinary values: {values}")
    confidence = pd.to_numeric(roles["role_confidence"], errors="raise")
    if confidence.isna().any() or not confidence.between(0.0, 1.0).all():
        raise AssertionError("role_confidence must be within [0,1].")
    if roles["role_reason"].map(clean).eq("").any():
        raise AssertionError("Every role row requires a role_reason.")

    print("CHECK 03: independently reconstruct primary QB rules")
    merged = roles.merge(
        universe[
            [
                "season", "week", "game_id", "team", "player_id", "position",
                "depth_rank", "depth_starter_flag", "depth_injury", "injury_game_status",
                "eligibility_status",
            ]
        ],
        on=["season", "week", "game_id", "team", "player_id"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_u"),
    )
    primary_qbs = roles.loc[flag(roles["primary_qb_flag"]).eq(1)]
    teams = sorted(universe["team"].map(common.normalize_team).unique())
    qb_counts = primary_qbs.groupby("team").size()
    if set(qb_counts.index) != set(teams) or not qb_counts.eq(1).all():
        raise AssertionError(f"Each scheduled team must have one primary QB. counts={qb_counts.to_dict()}")
    for team in teams:
        qbs = universe.loc[universe["team"].map(common.normalize_team).eq(team) & universe["position"].astype(str).str.upper().eq("QB")].copy()
        available = qbs.loc[eligible_mask(qbs)].copy()
        if available.empty:
            raise AssertionError(f"Team {team} has no eligible QB but role selector passed.")
        available["_rank"] = rank_series(available)
        ranked = available.loc[available["_rank"].notna()].copy()
        if ranked.empty:
            explicit = available.loc[flag(available["depth_starter_flag"]).eq(1)]
            if len(explicit) != 1:
                raise AssertionError(f"Team {team} has unresolved QB starter and should have failed.")
            expected_id = clean(explicit.iloc[0]["player_id"])
        else:
            best_rank = float(ranked["_rank"].min())
            tied = ranked.loc[ranked["_rank"].eq(best_rank)]
            if len(tied) == 1:
                expected_id = clean(tied.iloc[0]["player_id"])
            else:
                explicit = tied.loc[flag(tied["depth_starter_flag"]).eq(1)]
                if len(explicit) != 1:
                    raise AssertionError(f"Team {team} has ambiguous best-ranked QBs and selector should have failed.")
                expected_id = clean(explicit.iloc[0]["player_id"])
        actual_id = clean(primary_qbs.loc[primary_qbs["team"].eq(team), "player_id"].iloc[0])
        if actual_id != expected_id:
            raise AssertionError(f"Primary QB mismatch for {team}: expected={expected_id} actual={actual_id}")
        actual = primary_qbs.loc[primary_qbs["team"].eq(team)].iloc[0]
        if int(actual["starter_flag"]) != 1 or int(actual["primary_role_flag"]) != 1:
            raise AssertionError(f"Primary QB for {team} must be effective starter and primary role.")

    print("CHECK 04: kicker depth/usage selection and ambiguity uncertainty contract")
    primary_kickers = roles.loc[flag(roles["primary_kicker_flag"]).eq(1)]
    kicker_counts = primary_kickers.groupby("team").size()
    if set(kicker_counts.index) != set(teams) or not kicker_counts.eq(1).all():
        raise AssertionError(f"Each scheduled team must have one primary kicker. counts={kicker_counts.to_dict()}")
    if not primary_kickers["position"].astype(str).str.upper().isin(KICKER_POSITIONS).all():
        raise AssertionError("Primary kicker must be K/PK.")

    config = common.load_config()
    opp = common.read_parquet_required(
        config["paths"]["player_opportunity"],
        ["season", "week", "player_id", "field_goal_attempts", "extra_point_attempts"],
    ).copy()
    opp["season"] = pd.to_numeric(opp["season"], errors="coerce")
    opp["week"] = pd.to_numeric(opp["week"], errors="coerce")
    opp["player_id"] = opp["player_id"].map(common.normalize_player_id)
    opp = opp.loc[opp["season"].lt(season) | (opp["season"].eq(season) & opp["week"].lt(week))].copy()
    opp["attempts"] = (
        pd.to_numeric(opp["field_goal_attempts"], errors="coerce").fillna(0).clip(lower=0)
        + pd.to_numeric(opp["extra_point_attempts"], errors="coerce").fillna(0).clip(lower=0)
    )
    opp = opp.sort_values(["player_id", "season", "week"], ascending=[True, False, False], kind="mergesort")
    recent = opp.groupby("player_id", group_keys=False).head(5).groupby("player_id")["attempts"].sum().to_dict()

    log_kickers = {row["team"]: row for row in log.get("audit", {}).get("kicker_assignments", [])}
    if set(log_kickers) != set(teams):
        raise AssertionError("Role log must contain kicker assignment for every team.")
    for team in teams:
        candidates = universe.loc[
            universe["team"].map(common.normalize_team).eq(team)
            & universe["position"].astype(str).str.upper().isin(KICKER_POSITIONS)
            & eligible_mask(universe)
        ].copy()
        if candidates.empty:
            raise AssertionError(f"No eligible K/PK for {team}.")
        candidates["_rank"] = rank_series(candidates).fillna(999.0)
        candidates["_starter"] = flag(candidates["depth_starter_flag"])
        candidates["_recent"] = candidates["player_id"].map(lambda x: float(recent.get(common.normalize_player_id(x), 0.0)))
        candidates = candidates.sort_values(
            ["_rank", "_starter", "_recent", "player_id"],
            ascending=[True, False, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        expected = clean(candidates.iloc[0]["player_id"])
        actual = clean(primary_kickers.loc[primary_kickers["team"].eq(team), "player_id"].iloc[0])
        if actual != expected:
            raise AssertionError(f"Primary kicker mismatch for {team}: expected={expected} actual={actual}")
        logged = log_kickers[team]
        if clean(logged.get("player_id")) != actual:
            raise AssertionError(f"Kicker log mismatch for {team}.")
        ambiguous = bool(logged.get("ambiguous"))
        if ambiguous:
            conf = float(primary_kickers.loc[primary_kickers["team"].eq(team), "role_confidence"].iloc[0])
            if conf > 0.65:
                raise AssertionError(f"Ambiguous kicker for {team} must have reduced confidence; got {conf}")
            multiplier = float(logged.get("uncertainty_widening_multiplier", 1.0))
            if multiplier <= 1.0:
                raise AssertionError(f"Ambiguous kicker for {team} must widen uncertainty.")
            reason = clean(primary_kickers.loc[primary_kickers["team"].eq(team), "role_reason"].iloc[0]).casefold()
            if "widen_uncertainty" not in reason:
                raise AssertionError(f"Ambiguous kicker reason for {team} lacks widening marker.")

    print("CHECK 05: market exclusion and policy markers")
    if log.get("market_exclusion_passed") is not True or log.get("market_features_used") is not False:
        raise AssertionError("Issue 30 market exclusion markers failed.")
    policy = log.get("policy", {})
    required_policy = {
        "qb1_available_is_primary": True,
        "qb1_ineligible_promotes_next_eligible": True,
        "unresolved_qb_starter_fails": True,
        "kicker_uses_depth_rank": True,
        "kicker_uses_strictly_prior_attempts": True,
        "ambiguous_kicker_reduces_confidence": True,
        "ambiguous_kicker_widens_uncertainty": True,
        "market_exclusion_preflight": True,
    }
    for key, expected in required_policy.items():
        if policy.get(key) is not expected:
            raise AssertionError(f"Missing/incorrect Issue 30 policy marker: {key}")
    source = builder.read_text(encoding="utf-8")
    if "audit_market_exclusion.py" not in source or "MARKET EXCLUSION AUDIT: PASS" not in source:
        raise AssertionError("Role selector does not call Issue 28 market-exclusion validator.")

    print("CHECK 06: summarize independently validated current roles")
    ambiguous_teams = sorted(log.get("ambiguous_kicker_teams", []))
    print(f"season={season}")
    print(f"week={week}")
    print(f"teams={len(teams)}")
    print(f"rows={len(roles)}")
    print(f"primary_qbs={int(roles['primary_qb_flag'].sum())}")
    print(f"primary_kickers={int(roles['primary_kicker_flag'].sum())}")
    print(f"primary_roles={int(roles['primary_role_flag'].sum())}")
    print(f"committee_roles={int(roles['committee_role_flag'].sum())}")
    print(f"ambiguous_kicker_teams={len(ambiguous_teams)}")
    if ambiguous_teams:
        print("ambiguous_kicker_team_list=" + ",".join(ambiguous_teams))
    print("market_features_used=false")
    print("ISSUE 30 ACCEPTANCE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
