#!/usr/bin/env python3
"""
Select current starters and projection roles for the NFL Prop Engine.

READS:
    docs/win/football/nfl/prop_engine/config/prop_engine.yaml
    docs/win/football/nfl/prop_engine/data/current/{season}_week_{week}_universe.parquet
    configured historical player_opportunity (for strictly prior kicker usage)
    optional current/source/stats_player_week_{season}.parquet (weeks < target week only)

WRITES:
    docs/win/football/nfl/prop_engine/data/current/{season}_week_{week}_roles.parquet
    docs/win/football/nfl/prop_engine/logs/current_roles_{season}_week_{week}.json

POLICY:
    - Only eligible current-universe players receive roles.
    - Exactly one primary QB and primary kicker are selected per scheduled team.
    - An available depth-rank-1 QB is primary. If QB1 is ineligible, the next
      eligible ranked QB is promoted. Genuine QB depth ambiguity is a hard fail.
    - Kicker selection uses current depth rank plus strictly prior realized kick
      attempts. Ambiguous two-kicker situations remain deterministic but receive
      reduced role confidence and an uncertainty-widening marker in the log.
    - Market-exclusion validation runs before weekly role selection.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import os
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
    "season",
    "week",
    "game_id",
    "team",
    "player_id",
    "player_name",
    "position",
    "depth_rank",
    "starter_flag",
    "primary_qb_flag",
    "primary_kicker_flag",
    "primary_role_flag",
    "committee_role_flag",
    "role_confidence",
    "role_reason",
]

UNIVERSE_REQUIRED = [
    "season",
    "week",
    "game_id",
    "player_id",
    "player_name",
    "team",
    "position",
    "depth_rank",
    "depth_starter_flag",
    "depth_backup_flag",
    "depth_injury",
    "injury_game_status",
    "eligibility_status",
    "eligibility_reason",
    "role_status",
]

QB_POSITIONS = {"QB"}
KICKER_POSITIONS = {"K", "PK"}
COMMITTEE_POSITIONS = {
    "RB", "FB", "WR", "TE",
    "DL", "DE", "DT", "NT", "EDGE",
    "LB", "ILB", "OLB", "MLB",
    "DB", "CB", "S", "FS", "SS", "NB",
}

KICKER_AMBIGUITY_CONFIDENCE = 0.60
KICKER_AMBIGUITY_UNCERTAINTY_MULTIPLIER = 1.25


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


def normalize_position(value: Any) -> str:
    return clean(value).upper()


def numeric(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def integer_flag(value: Any) -> int:
    value_num = numeric(value, 0.0)
    return int(value_num > 0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select current NFL starter and projection roles.")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week", type=int, required=True)
    return parser.parse_args()


def resolve_season(args: argparse.Namespace, config: dict) -> int:
    season = int(args.season) if args.season is not None else int(config["seasons"]["current"])
    if not 1900 <= season <= 2200:
        raise ValueError(f"Invalid season: {season}")
    return season


def repo_relative(path: Path) -> str:
    return str(path.resolve().relative_to(common.repo_root().resolve())).replace("\\", "/")


def write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    prop_root = common.prop_root().resolve()
    destination = path.resolve()
    try:
        destination.relative_to(prop_root)
    except ValueError as exc:
        raise ValueError(f"Role log must remain under Prop Engine: {destination}") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
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


def run_market_preflight() -> dict[str, Any]:
    audit_path = SCRIPTS_ROOT / "validate" / "audit_market_exclusion.py"
    if not audit_path.exists():
        raise FileNotFoundError(f"Issue 28 market-exclusion validator missing: {audit_path}")
    completed = subprocess.run(
        [sys.executable, str(audit_path)],
        cwd=common.repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Market-exclusion preflight failed before role selection. "
            f"stdout={completed.stdout[-2000:]!r} stderr={completed.stderr[-2000:]!r}"
        )
    if "MARKET EXCLUSION AUDIT: PASS" not in completed.stdout:
        raise RuntimeError("Market-exclusion validator returned zero without PASS marker.")
    return {
        "passed": True,
        "validator": repo_relative(audit_path),
        "pass_marker": "MARKET EXCLUSION AUDIT: PASS",
    }


def current_universe_path(season: int, week: int) -> Path:
    return common.prop_root() / "data" / "current" / f"{season}_week_{week}_universe.parquet"


def output_path(season: int, week: int) -> Path:
    return common.prop_root() / "data" / "current" / f"{season}_week_{week}_roles.parquet"


def log_path(season: int, week: int) -> Path:
    return common.prop_root() / "logs" / f"current_roles_{season}_week_{week}.json"


def load_universe(season: int, week: int) -> pd.DataFrame:
    path = current_universe_path(season, week)
    if not path.exists():
        raise FileNotFoundError(f"Issue 29 current universe is missing: {path}")
    frame = pd.read_parquet(path)
    common.require_columns(frame, UNIVERSE_REQUIRED, "Issue 29 current universe")
    if list(frame.columns)[:0] != []:
        pass
    frame = frame.copy()
    frame["season"] = pd.to_numeric(frame["season"], errors="raise").astype(int)
    frame["week"] = pd.to_numeric(frame["week"], errors="raise").astype(int)
    frame["player_id"] = frame["player_id"].map(common.normalize_player_id)
    frame["team"] = frame["team"].map(common.normalize_team)
    frame["position"] = frame["position"].map(normalize_position)
    frame["game_id"] = frame["game_id"].map(clean)
    bad_period = ~frame["season"].eq(season) | ~frame["week"].eq(week)
    if bad_period.any():
        raise ValueError("Current universe contains rows outside requested season/week.")
    if frame["player_id"].eq("").any():
        raise ValueError("Current universe contains blank player_id.")
    common.ensure_unique(frame, ["season", "week", "game_id", "player_id"], "current universe")
    return frame


def eligible_mask(frame: pd.DataFrame) -> pd.Series:
    return frame["eligibility_status"].fillna("").astype(str).str.strip().str.casefold().eq("eligible")


def depth_rank_series(frame: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(frame["depth_rank"], errors="coerce").astype("float64")


def injury_is_questionable(row: pd.Series) -> bool:
    values = [row.get("injury_game_status", ""), row.get("depth_injury", "")]
    return any("questionable" in clean(value).casefold() for value in values)


def select_primary_qb(team_all: pd.DataFrame) -> tuple[str, float, str, dict[str, Any]]:
    qbs = team_all.loc[team_all["position"].eq("QB")].copy()
    if qbs.empty:
        raise RuntimeError(f"No resolved QB rows for team={clean(team_all['team'].iloc[0])}")
    qbs["_eligible"] = eligible_mask(qbs)
    qbs["_rank"] = depth_rank_series(qbs)
    qbs["_starter"] = qbs["depth_starter_flag"].map(integer_flag)
    available = qbs.loc[qbs["_eligible"]].copy()
    team = clean(team_all["team"].iloc[0])
    if available.empty:
        raise RuntimeError(f"No eligible resolved QB for team={team}; team passing projection must fail.")

    ranked = available.loc[available["_rank"].notna()].copy()
    if ranked.empty:
        explicit = available.loc[available["_starter"].eq(1)].copy()
        if len(explicit) != 1:
            sample = available[["player_id", "player_name", "depth_rank", "depth_starter_flag"]].to_dict("records")
            raise RuntimeError(
                f"Unresolved QB starter for team={team}: no usable depth rank and not exactly one starter. sample={sample}"
            )
        chosen = explicit.iloc[0]
        confidence = 0.75
        reason = "qb_explicit_starter_without_depth_rank"
    else:
        best_rank = float(ranked["_rank"].min())
        tied = ranked.loc[ranked["_rank"].eq(best_rank)].copy()
        if len(tied) > 1:
            explicit = tied.loc[tied["_starter"].eq(1)].copy()
            if len(explicit) != 1:
                sample = tied[["player_id", "player_name", "depth_rank", "depth_starter_flag"]].to_dict("records")
                raise RuntimeError(
                    f"Unresolved QB starter for team={team}: multiple eligible QBs share best depth rank={best_rank}. sample={sample}"
                )
            chosen = explicit.iloc[0]
        else:
            chosen = tied.iloc[0]

        all_ranked = qbs.loc[qbs["_rank"].notna()].copy()
        original_best_rank = float(all_ranked["_rank"].min()) if not all_ranked.empty else best_rank
        promoted = best_rank > original_best_rank
        questionable = injury_is_questionable(chosen)
        if promoted:
            confidence = 0.90 if not questionable else 0.80
            reason = "qb_promoted_after_higher_depth_qb_ineligible"
        elif math.isclose(best_rank, 1.0):
            confidence = 0.85 if questionable else 1.00
            reason = "qb1_questionable_primary" if questionable else "healthy_depth_rank_1_qb"
        else:
            confidence = 0.82 if questionable else 0.90
            reason = "best_eligible_ranked_qb"

    summary = {
        "team": team,
        "player_id": clean(chosen["player_id"]),
        "player_name": clean(chosen["player_name"]),
        "chosen_depth_rank": None if pd.isna(chosen["_rank"]) else float(chosen["_rank"]),
        "role_confidence": confidence,
        "role_reason": reason,
    }
    return clean(chosen["player_id"]), float(confidence), reason, summary


def choose_column(frame: pd.DataFrame, aliases: list[str]) -> str | None:
    for column in aliases:
        if column in frame.columns:
            return column
    return None


def historical_kick_events(config: dict, player_ids: set[str], season: int, week: int) -> pd.DataFrame:
    path = config["paths"]["player_opportunity"]
    required = ["season", "week", "player_id", "field_goal_attempts", "extra_point_attempts"]
    history = common.read_parquet_required(path, required).copy()
    history["season"] = pd.to_numeric(history["season"], errors="coerce")
    history["week"] = pd.to_numeric(history["week"], errors="coerce")
    history["player_id"] = history["player_id"].map(common.normalize_player_id)
    history = history.loc[
        history["player_id"].isin(player_ids)
        & (
            history["season"].lt(season)
            | (history["season"].eq(season) & history["week"].lt(week))
        )
    ].copy()
    history["fg_att"] = pd.to_numeric(history["field_goal_attempts"], errors="coerce").fillna(0.0).clip(lower=0.0)
    history["pat_att"] = pd.to_numeric(history["extra_point_attempts"], errors="coerce").fillna(0.0).clip(lower=0.0)
    return history[["season", "week", "player_id", "fg_att", "pat_att"]]


def current_kick_events(config: dict, player_ids: set[str], season: int, week: int) -> tuple[pd.DataFrame, str]:
    source_root = common.repo_root() / config["paths"]["current_source_root"]
    path = source_root / f"stats_player_week_{season}.parquet"
    empty = pd.DataFrame(columns=["season", "week", "player_id", "fg_att", "pat_att"])
    if not path.exists() or week <= 1:
        return empty, "not_available_or_not_needed"
    frame = pd.read_parquet(path)
    id_col = choose_column(frame, ["player_id", "gsis_id", "nflverse_player_id"])
    week_col = choose_column(frame, ["week"])
    season_col = choose_column(frame, ["season"])
    fg_col = choose_column(frame, ["fg_att", "field_goal_attempts", "field_goals_attempted"])
    pat_col = choose_column(frame, ["pat_att", "extra_point_attempts", "extra_points_attempted"])
    if not id_col or not week_col or not fg_col or not pat_col:
        return empty, "present_but_schema_not_usable"
    out = pd.DataFrame()
    out["player_id"] = frame[id_col].map(common.normalize_player_id)
    out["week"] = pd.to_numeric(frame[week_col], errors="coerce")
    out["season"] = pd.to_numeric(frame[season_col], errors="coerce") if season_col else season
    out["fg_att"] = pd.to_numeric(frame[fg_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    out["pat_att"] = pd.to_numeric(frame[pat_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    out = out.loc[
        out["player_id"].isin(player_ids)
        & out["season"].eq(season)
        & out["week"].lt(week)
    ].copy()
    return out[["season", "week", "player_id", "fg_att", "pat_att"]], "used"


def build_recent_kick_usage(config: dict, universe: pd.DataFrame, season: int, week: int) -> tuple[dict[str, dict[str, float]], str]:
    kicker_ids = set(
        universe.loc[universe["position"].isin(KICKER_POSITIONS), "player_id"].map(common.normalize_player_id)
    )
    if not kicker_ids:
        return {}, "no_kickers"
    historical = historical_kick_events(config, kicker_ids, season, week)
    current, current_status = current_kick_events(config, kicker_ids, season, week)
    events = pd.concat([historical, current], ignore_index=True)
    if events.empty:
        return {pid: {"recent_fg_attempts": 0.0, "recent_pat_attempts": 0.0, "recent_kick_attempts": 0.0, "recent_games": 0.0} for pid in kicker_ids}, current_status
    events = events.sort_values(["player_id", "season", "week"], ascending=[True, False, False], kind="mergesort")
    recent = events.groupby("player_id", sort=False, group_keys=False).head(5)
    grouped = recent.groupby("player_id", as_index=False).agg(
        recent_fg_attempts=("fg_att", "sum"),
        recent_pat_attempts=("pat_att", "sum"),
        recent_games=("week", "size"),
    )
    grouped["recent_kick_attempts"] = grouped["recent_fg_attempts"] + grouped["recent_pat_attempts"]
    usage: dict[str, dict[str, float]] = {}
    for row in grouped.itertuples(index=False):
        usage[clean(row.player_id)] = {
            "recent_fg_attempts": float(row.recent_fg_attempts),
            "recent_pat_attempts": float(row.recent_pat_attempts),
            "recent_kick_attempts": float(row.recent_kick_attempts),
            "recent_games": float(row.recent_games),
        }
    for pid in kicker_ids:
        usage.setdefault(pid, {"recent_fg_attempts": 0.0, "recent_pat_attempts": 0.0, "recent_kick_attempts": 0.0, "recent_games": 0.0})
    return usage, current_status


def kicker_sort_score(rank: float, recent_attempts: float, starter: int) -> tuple[float, float, float]:
    rank_value = rank if math.isfinite(rank) else 999.0
    return (-rank_value, float(starter), float(recent_attempts))


def select_primary_kicker(team_all: pd.DataFrame, usage: dict[str, dict[str, float]]) -> tuple[str, float, str, set[str], dict[str, Any]]:
    team = clean(team_all["team"].iloc[0])
    kickers = team_all.loc[
        team_all["position"].isin(KICKER_POSITIONS) & eligible_mask(team_all)
    ].copy()
    if kickers.empty:
        raise RuntimeError(f"No eligible K/PK candidate for team={team}; kicking projection cannot be assigned.")
    kickers["_rank"] = depth_rank_series(kickers)
    kickers["_starter"] = kickers["depth_starter_flag"].map(integer_flag)
    kickers["_recent"] = kickers["player_id"].map(
        lambda pid: usage.get(clean(pid), {}).get("recent_kick_attempts", 0.0)
    ).astype(float)
    kickers["_rank_sort"] = kickers["_rank"].fillna(999.0)
    kickers = kickers.sort_values(
        ["_rank_sort", "_starter", "_recent", "player_id"],
        ascending=[True, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    chosen = kickers.iloc[0]
    ambiguous = False
    committee_ids: set[str] = set()
    ambiguity_reason = ""

    if len(kickers) >= 2:
        second = kickers.iloc[1]
        chosen_rank = numeric(chosen["_rank"], 999.0)
        second_rank = numeric(second["_rank"], 999.0)
        chosen_recent = float(chosen["_recent"])
        second_recent = float(second["_recent"])
        same_rank = math.isclose(chosen_rank, second_rank)
        conflicting_usage = (
            second_rank <= chosen_rank + 1.0
            and second_recent >= chosen_recent + 3.0
            and second_recent > 0.0
        )
        duplicate_starters = int(chosen["_starter"]) == 1 and int(second["_starter"]) == 1
        if same_rank or conflicting_usage or duplicate_starters:
            ambiguous = True
            committee_ids = {clean(chosen["player_id"]), clean(second["player_id"])}
            pieces = []
            if same_rank:
                pieces.append("same_depth_rank")
            if conflicting_usage:
                pieces.append("recent_attempts_conflict_with_depth")
            if duplicate_starters:
                pieces.append("multiple_depth_starters")
            ambiguity_reason = "+".join(pieces)

    if ambiguous:
        confidence = KICKER_AMBIGUITY_CONFIDENCE
        reason = f"kicker_ambiguous_{ambiguity_reason}_widen_uncertainty_{KICKER_AMBIGUITY_UNCERTAINTY_MULTIPLIER:.2f}x"
    else:
        rank = numeric(chosen["_rank"], 999.0)
        recent = float(chosen["_recent"])
        if math.isclose(rank, 1.0) and recent > 0.0:
            confidence = 0.97
            reason = "kicker_depth_rank_1_with_recent_attempts"
        elif math.isclose(rank, 1.0):
            confidence = 0.88
            reason = "kicker_depth_rank_1_without_recent_attempts"
        elif recent > 0.0:
            confidence = 0.82
            reason = "kicker_best_depth_with_recent_attempts"
        else:
            confidence = 0.72
            reason = "kicker_best_available_depth_no_recent_attempts"

    summary = {
        "team": team,
        "player_id": clean(chosen["player_id"]),
        "player_name": clean(chosen["player_name"]),
        "depth_rank": None if pd.isna(chosen["_rank"]) else float(chosen["_rank"]),
        "recent_kick_attempts": float(chosen["_recent"]),
        "ambiguous": ambiguous,
        "committee_player_ids": sorted(committee_ids),
        "role_confidence": float(confidence),
        "role_reason": reason,
        "uncertainty_widening_multiplier": (
            KICKER_AMBIGUITY_UNCERTAINTY_MULTIPLIER if ambiguous else 1.0
        ),
    }
    return clean(chosen["player_id"]), float(confidence), reason, committee_ids, summary


def generic_role(row: pd.Series) -> tuple[int, int, float, str]:
    position = normalize_position(row["position"])
    depth_starter = integer_flag(row.get("depth_starter_flag", 0))
    rank = numeric(row.get("depth_rank"), 999.0)
    questionable = injury_is_questionable(row)
    if depth_starter:
        confidence = 0.82 if questionable else 0.92
        return 1, 0, confidence, "current_depth_starter_questionable" if questionable else "current_depth_starter"
    if position in COMMITTEE_POSITIONS and math.isfinite(rank) and rank <= 2.0:
        confidence = 0.68 if questionable else 0.78
        return 0, 1, confidence, "depth_committee_questionable" if questionable else "depth_committee_role"
    if clean(row.get("role_status", "")):
        confidence = 0.55 if questionable else 0.65
        return 0, 0, confidence, "eligible_current_role_questionable" if questionable else "eligible_current_role"
    return 0, 0, 0.50, "eligible_role_depth_uncertain"


def build_roles(universe: pd.DataFrame, config: dict, season: int, week: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    eligible = universe.loc[eligible_mask(universe)].copy()
    if eligible.empty:
        raise RuntimeError("Issue 29 universe contains no eligible players.")

    usage, current_usage_status = build_recent_kick_usage(config, universe, season, week)

    team_groups = {team: group.copy() for team, group in universe.groupby("team", sort=True)}
    qb_assignments: dict[str, tuple[str, float, str]] = {}
    kicker_assignments: dict[str, tuple[str, float, str, set[str]]] = {}
    qb_summary = []
    kicker_summary = []
    failures: list[str] = []

    for team, group in team_groups.items():
        try:
            qid, qconf, qreason, qsum = select_primary_qb(group)
            qb_assignments[team] = (qid, qconf, qreason)
            qb_summary.append(qsum)
        except Exception as exc:
            failures.append(str(exc))

        try:
            kid, kconf, kreason, committee, ksum = select_primary_kicker(group, usage)
            kicker_assignments[team] = (kid, kconf, kreason, committee)
            kicker_summary.append(ksum)
        except Exception as exc:
            failures.append(str(exc))

    if failures:
        raise RuntimeError(
            "Current role selection failed; unresolved team role(s) must not be guessed. "
            f"Count={len(failures)} sample={failures[:10]}"
        )

    rows: list[dict[str, Any]] = []
    for _, row in eligible.iterrows():
        team = clean(row["team"])
        pid = clean(row["player_id"])
        position = normalize_position(row["position"])
        generic_primary, generic_committee, confidence, reason = generic_role(row)
        starter_flag = integer_flag(row.get("depth_starter_flag", 0))
        primary_qb_flag = 0
        primary_kicker_flag = 0
        primary_role_flag = generic_primary
        committee_role_flag = generic_committee

        qid, qconf, qreason = qb_assignments[team]
        if position == "QB":
            if pid == qid:
                primary_qb_flag = 1
                primary_role_flag = 1
                committee_role_flag = 0
                starter_flag = 1
                confidence = qconf
                reason = qreason
            else:
                primary_role_flag = 0
                committee_role_flag = 0
                starter_flag = 0
                confidence = 0.75 if integer_flag(row.get("depth_backup_flag", 0)) else 0.60
                reason = "qb_backup"

        kid, kconf, kreason, kicker_committee = kicker_assignments[team]
        if position in KICKER_POSITIONS:
            if pid == kid:
                primary_kicker_flag = 1
                primary_role_flag = 1
                starter_flag = 1
                confidence = kconf
                reason = kreason
            else:
                primary_role_flag = 0
                starter_flag = 0
                confidence = KICKER_AMBIGUITY_CONFIDENCE if pid in kicker_committee else 0.55
                reason = (
                    f"kicker_committee_ambiguity_widen_uncertainty_{KICKER_AMBIGUITY_UNCERTAINTY_MULTIPLIER:.2f}x"
                    if pid in kicker_committee else "kicker_backup"
                )
            committee_role_flag = int(pid in kicker_committee)

        rows.append({
            "season": int(row["season"]),
            "week": int(row["week"]),
            "game_id": clean(row["game_id"]),
            "team": team,
            "player_id": pid,
            "player_name": clean(row["player_name"]),
            "position": position,
            "depth_rank": (np.nan if pd.isna(row["depth_rank"]) else float(row["depth_rank"])),
            "starter_flag": int(starter_flag),
            "primary_qb_flag": int(primary_qb_flag),
            "primary_kicker_flag": int(primary_kicker_flag),
            "primary_role_flag": int(primary_role_flag),
            "committee_role_flag": int(committee_role_flag),
            "role_confidence": float(round(float(confidence), 4)),
            "role_reason": reason,
        })

    output = pd.DataFrame.from_records(rows, columns=OUTPUT_COLUMNS)
    common.ensure_unique(output, ["season", "week", "game_id", "team", "player_id"], "current roles")

    for column in ["starter_flag", "primary_qb_flag", "primary_kicker_flag", "primary_role_flag", "committee_role_flag"]:
        if not set(output[column].dropna().astype(int).unique()).issubset({0, 1}):
            raise ValueError(f"{column} must be binary.")
    if output["role_confidence"].isna().any() or not output["role_confidence"].between(0.0, 1.0).all():
        raise ValueError("role_confidence must be finite within [0,1].")

    team_count = int(universe["team"].nunique())
    qb_counts = output.groupby("team")["primary_qb_flag"].sum()
    kicker_counts = output.groupby("team")["primary_kicker_flag"].sum()
    if len(qb_counts) != team_count or not qb_counts.eq(1).all():
        raise RuntimeError(f"Each scheduled team must have exactly one primary QB: {qb_counts.to_dict()}")
    if len(kicker_counts) != team_count or not kicker_counts.eq(1).all():
        raise RuntimeError(f"Each scheduled team must have exactly one primary kicker: {kicker_counts.to_dict()}")

    audit = {
        "current_kick_stats_status": current_usage_status,
        "qb_assignments": qb_summary,
        "kicker_assignments": kicker_summary,
        "ambiguous_kicker_teams": [row["team"] for row in kicker_summary if row["ambiguous"]],
        "kicker_ambiguity_uncertainty_multiplier": KICKER_AMBIGUITY_UNCERTAINTY_MULTIPLIER,
    }
    return output, audit


def main() -> int:
    args = parse_args()
    config = common.load_config()
    season = resolve_season(args, config)
    week = int(args.week)
    if not 1 <= week <= 25:
        raise ValueError(f"Invalid week: {week}")

    market = run_market_preflight()
    universe = load_universe(season, week)
    roles, audit = build_roles(universe, config, season, week)

    destination = output_path(season, week)
    common.write_parquet_atomic(roles[OUTPUT_COLUMNS], destination)

    payload = {
        "script": Path(__file__).name,
        "status": "passed",
        "season": season,
        "week": week,
        "rows": int(len(roles)),
        "teams": int(roles["team"].nunique()),
        "primary_qbs": int(roles["primary_qb_flag"].sum()),
        "primary_kickers": int(roles["primary_kicker_flag"].sum()),
        "primary_roles": int(roles["primary_role_flag"].sum()),
        "committee_roles": int(roles["committee_role_flag"].sum()),
        "ambiguous_kicker_teams": audit["ambiguous_kicker_teams"],
        "market_exclusion_passed": bool(market["passed"]),
        "market_features_used": False,
        "output": repo_relative(destination),
        "log": repo_relative(log_path(season, week)),
    }
    log_payload = {
        **payload,
        "policy": {
            "qb1_available_is_primary": True,
            "qb1_ineligible_promotes_next_eligible": True,
            "unresolved_qb_starter_fails": True,
            "kicker_uses_depth_rank": True,
            "kicker_uses_strictly_prior_attempts": True,
            "ambiguous_kicker_reduces_confidence": True,
            "ambiguous_kicker_widens_uncertainty": True,
            "kicker_ambiguity_uncertainty_multiplier": KICKER_AMBIGUITY_UNCERTAINTY_MULTIPLIER,
            "market_exclusion_preflight": True,
        },
        "market_preflight": market,
        "audit": audit,
    }
    write_json_atomic(log_payload, log_path(season, week))
    common.log_run(Path(__file__).name, payload)

    print(json.dumps({"script": Path(__file__).name, "payload": payload}, sort_keys=True, separators=(",", ":")))
    print("CURRENT ROLE SELECTION: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
