#!/usr/bin/env python3
"""
Build leakage-safe current-week NFL Prop Engine features.

Issue 31 contract
-----------------
READS:
    data/current/{season}_week_{week}_universe.parquet
    data/current/{season}_week_{week}_roles.parquet
    data/current/source/stats_player_week_{season}.parquet
    data/current/source/snap_counts_{season}.parquet
    data/current/source/pbp_participation_{season}.parquet
    00_intake/pbp/{season}_pbp.csv.gz
    00_intake/team_stats/{season}_team_stats.csv
    data/weather/week_{week}_NFL_weekly_weather.csv
    data/travel/{season}_week_{week}_travel.csv
    data/historical/features/player_game_features.parquet
    data/historical/opportunity/position_allowed_week.parquet

WRITES:
    data/current/features/{season}_week_{week}_features.parquet
    data/current/features/{season}_week_{week}_feature_manifest.json
    logs/current_features_{season}_week_{week}.json

POLICY:
    - Current output exposes the canonical historical leading columns plus the
      canonical historical model-feature columns, but no target_* or audit_*.
    - Week 1 uses explicit prior-season priors. Current-season season_to_date
      fields are reset because no current-season observation exists yet.
    - For week > 1, realized current-season sources are required and only rows
      with source week < target week are eligible.
    - Weather and travel are joined by game_id only.
    - Current role/injury/depth state overrides stale historical role state.
    - Team/opponent/position-allowed priors are keyed to the current matchup;
      they are never copied from a player's final prior-season opponent.
    - No sportsbook/market/target field is permitted.
    - Every selected model/dependency manifest is checked for required feature
      presence and feature order. Stored direct-model feature hashes are
      independently recomputed and must match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
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



_CONFIG_CONTRACT = common.load_config()
TARGETS = list(_CONFIG_CONTRACT["targets"].keys())

GRAIN = ["season", "week", "game_id", "player_id"]
PLAYER_PRIOR_FAMILIES = ("role", "player", "history")
TEAM_HISTORY_ALIASES = {"SD": "LAC", "OAK": "LV", "STL": "LAR"}

FRONT7 = {
    "DL", "DE", "LDE", "RDE", "DT", "LDT", "RDT", "NT", "EDGE",
    "LB", "ILB", "OLB", "MLB", "WLB", "SLB",
}
SECONDARY = {"DB", "CB", "LCB", "RCB", "NB", "S", "FS", "SS"}
DEFENSE = FRONT7 | SECONDARY

DIVISION = {
    "BUF": "AFC_E", "MIA": "AFC_E", "NE": "AFC_E", "NYJ": "AFC_E",
    "BAL": "AFC_N", "CIN": "AFC_N", "CLE": "AFC_N", "PIT": "AFC_N",
    "HOU": "AFC_S", "IND": "AFC_S", "JAX": "AFC_S", "TEN": "AFC_S",
    "DEN": "AFC_W", "KC": "AFC_W", "LV": "AFC_W", "LAC": "AFC_W",
    "DAL": "NFC_E", "NYG": "NFC_E", "PHI": "NFC_E", "WSH": "NFC_E",
    "CHI": "NFC_N", "DET": "NFC_N", "GB": "NFC_N", "MIN": "NFC_N",
    "ATL": "NFC_S", "CAR": "NFC_S", "NO": "NFC_S", "TB": "NFC_S",
    "ARI": "NFC_W", "LAR": "NFC_W", "SF": "NFC_W", "SEA": "NFC_W",
}

CURRENT_PLAYER_METRIC_ALIASES: dict[str, list[str]] = {
    "pass_attempts": ["attempts", "pass_attempts", "passing_attempts"],
    "dropbacks": ["dropbacks", "passing_dropbacks"],
    "completions": ["completions"],
    "passing_yards": ["passing_yards"],
    "passing_tds": ["passing_tds", "passing_touchdowns"],
    "passing_air_yards": ["passing_air_yards"],
    "carries": ["carries", "rushing_attempts"],
    "rushing_yards": ["rushing_yards"],
    "rushing_tds": ["rushing_tds", "rushing_touchdowns"],
    "targets": ["targets"],
    "receptions": ["receptions"],
    "receiving_yards": ["receiving_yards"],
    "receiving_tds": ["receiving_tds", "receiving_touchdowns"],
    "field_goal_attempts": ["fg_att", "field_goal_attempts"],
    "field_goals_made": ["fg_made", "field_goals_made"],
    "extra_point_attempts": ["pat_att", "extra_point_attempts"],
    "extra_points_made": ["pat_made", "extra_points_made"],
    "tackles": ["tackles", "def_tackles_combined"],
    "sacks": ["sacks", "def_sacks"],
    "qb_hits": ["qb_hits", "def_qb_hits"],
}

CURRENT_TEAM_METRIC_ALIASES: dict[str, list[str]] = {
    "offensive_plays": ["offensive_plays", "plays", "total_plays"],
    "drives": ["drives"],
    "dropbacks": ["dropbacks"],
    "pass_attempts": ["pass_attempts", "passing_attempts", "attempts"],
    "rush_attempts": ["rush_attempts", "rushing_attempts", "carries"],
    "pass_rate": ["pass_rate"],
    "rush_rate": ["rush_rate"],
    "points_per_drive": ["points_per_drive"],
    "red_zone_drives": ["red_zone_drives"],
    "red_zone_pass_attempts": ["red_zone_pass_attempts"],
    "red_zone_rush_attempts": ["red_zone_rush_attempts"],
    "goal_line_rush_attempts": ["goal_line_rush_attempts"],
    "field_goal_attempts": ["field_goal_attempts", "fg_att"],
    "extra_point_attempts": ["extra_point_attempts", "pat_att"],
    "off_epa_per_play": ["off_epa_per_play", "epa_per_play"],
    "off_success_rate": ["off_success_rate", "success_rate"],
    "yards_per_play": ["yards_per_play"],
    "red_zone_td_rate": ["red_zone_td_rate"],
    "early_down_epa": ["early_down_epa"],
    "third_down_conversion_rate": ["third_down_conversion_rate"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build current-week Prop Engine features.")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week", type=int, required=True)
    return parser.parse_args()


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
    team = common.normalize_team(value)
    return TEAM_HISTORY_ALIASES.get(team, team)


def norm_position(value: Any) -> str:
    return clean(value).upper().replace(" ", "")


def norm_game_id(value: Any) -> str:
    return common.normalize_player_id(value)


def norm_injury(value: Any) -> str:
    text = clean(value).casefold().replace("-", " ").replace("_", " ")
    text = " ".join(text.split())
    if text in {"o", "out", "ir", "injured reserve"} or text.startswith("out "):
        return "out"
    if text in {"d", "doubtful"} or "doubt" in text:
        return "doubtful"
    if text in {"q", "questionable"} or "question" in text:
        return "questionable"
    if text in {"p", "probable"} or "probable" in text:
        return "probable"
    return ""


def as_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def safe_mean_pair(left: pd.Series, right: pd.Series) -> pd.Series:
    return pd.concat([as_num(left), as_num(right)], axis=1).mean(axis=1, skipna=True)


def safe_divide(left: pd.Series, right: pd.Series) -> pd.Series:
    a = as_num(left)
    b = as_num(right)
    out = pd.Series(np.nan, index=a.index, dtype="float64")
    ok = a.notna() & b.notna() & b.ne(0)
    out.loc[ok] = a.loc[ok] / b.loc[ok]
    return out


def safe_product(left: pd.Series, right: pd.Series) -> pd.Series:
    a = as_num(left)
    b = as_num(right)
    out = a * b
    out.loc[a.isna() | b.isna()] = np.nan
    return out


def first_existing(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    available = set(columns)
    for name in aliases:
        if name in available:
            return name
    return None


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON missing: {path}")
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path = path.resolve()
    prop = common.prop_root().resolve()
    try:
        path.relative_to(prop)
    except ValueError as exc:
        raise ValueError(f"Refusing write outside Prop Engine: {path}") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temp = Path(handle.name)
    try:
        with handle:
            json.dump(value, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def run_market_audit(repo: Path) -> dict[str, Any]:
    script = (
        repo
        / "docs/win/football/nfl/prop_engine/scripts/validate/audit_market_exclusion.py"
    )
    if not script.is_file():
        raise FileNotFoundError(f"Issue 28 market audit missing: {script}")
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Market exclusion audit failed before current feature build. "
            f"stdout={completed.stdout[-3000:]} stderr={completed.stderr[-3000:]}"
        )
    audit_path = (
        repo
        / "docs/win/football/nfl/prop_engine/evaluation/market_exclusion_audit.json"
    )
    audit = read_json(audit_path)
    if audit.get("passed") is not True:
        raise RuntimeError("Issue 28 market exclusion audit did not pass.")
    return audit


def schema_payload(numeric: list[str], categorical: list[str]) -> list[dict[str, str]]:
    return [
        *[{"name": x, "type": "numeric"} for x in numeric],
        *[{"name": x, "type": "categorical"} for x in categorical],
    ]


def manifest_feature_hash(numeric: list[str], categorical: list[str]) -> str:
    payload = json.dumps(
        schema_payload(numeric, categorical),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_lgb_feature_names(path: Path) -> list[str] | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith("feature_names="):
                    return line.rstrip("\r\n").split("=", 1)[1].split()
    except OSError:
        return None
    return None


def selected_manifest_specs(
    repo: Path,
    available_columns: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model_root = repo / "docs/win/football/nfl/prop_engine/models"
    specs: list[dict[str, Any]] = []
    proxy_checks: list[dict[str, Any]] = []
    seen: set[Path] = set()

    def add_manifest(path: Path, model_path: Path, owner: str) -> None:
        resolved = path.resolve()
        if resolved in seen:
            return
        if not path.is_file():
            raise FileNotFoundError(f"Selected model feature manifest missing: {path}")
        seen.add(resolved)
        manifest = read_json(path)
        numeric = list(manifest.get("numeric_features", []))
        categorical = list(manifest.get("categorical_features", []))
        if not numeric and not categorical:
            raise ValueError(f"Manifest has no model features: {path}")
        expected_hash = manifest.get("feature_hash")
        calculated_hash = manifest_feature_hash(numeric, categorical)
        if expected_hash is not None and str(expected_hash) != calculated_hash:
            raise ValueError(
                f"Stored feature hash mismatch in {path}: "
                f"stored={expected_hash} calculated={calculated_hash}"
            )
        model_names = parse_lgb_feature_names(model_path)
        if model_names is not None and model_names != numeric + categorical:
            raise ValueError(
                f"Model feature order differs from manifest: {model_path}"
            )
        derived = [clean(value) for value in manifest.get("derived_features", []) if clean(value)]
        canonical = [clean(value) for value in manifest.get("canonical_features", []) if clean(value)]
        ordered = numeric + categorical
        if derived:
            if not canonical:
                raise ValueError(
                    f"Manifest declares derived_features without canonical_features: {path}"
                )
            if set(canonical) & set(derived):
                raise ValueError(f"Manifest canonical/derived overlap: {path}")
            if set(canonical + derived) != set(ordered):
                raise ValueError(
                    f"Manifest canonical+derived features do not cover model schema: {path}"
                )
        else:
            canonical = ordered.copy()

        specs.append(
            {
                "owner": owner,
                "manifest_path": path,
                "model_path": model_path,
                "manifest": manifest,
                "numeric": numeric,
                "categorical": categorical,
                "canonical": canonical,
                "derived": derived,
                "calculated_hash": calculated_hash,
                "stored_hash": expected_hash,
                "model_feature_order_checked": model_names is not None,
            }
        )

    for target in TARGETS:
        selected_path = model_root / target / "selected_model.json"
        if not selected_path.is_file():
            raise FileNotFoundError(f"Selected architecture missing: {selected_path}")
        selected = read_json(selected_path)
        architecture = clean(selected.get("selected_architecture") or selected.get("selected_candidate"))
        if architecture in {"direct", "direct_component_blend"}:
            add_manifest(
                model_root / target / "feature_manifest.json",
                model_root / target / "direct_model.txt",
                f"{target}:direct",
            )
        if architecture in {"component", "direct_component_blend"}:
            deps = selected.get("component_dependencies") or []
            if not isinstance(deps, list) or not deps:
                raise ValueError(f"Selected component architecture has no dependencies: {selected_path}")

            proxy_features: set[str] = set()
            for proxy_key in ("goal_line_volume_proxy_order", "red_zone_volume_proxy_order"):
                proxy_values = selected.get(proxy_key) or []
                if not isinstance(proxy_values, list):
                    raise ValueError(
                        f"{selected_path}: {proxy_key} must be a list when present"
                    )
                proxy_features.update(clean(value) for value in proxy_values if clean(value))

            for dep in deps:
                dep_name = clean(dep)
                component = model_root / "components" / dep_name / "feature_manifest.json"
                efficiency = model_root / "efficiency" / dep_name / "feature_manifest.json"
                if component.is_file():
                    add_manifest(
                        component,
                        component.with_name("model.txt"),
                        f"{target}:component:{dep_name}",
                    )
                elif efficiency.is_file():
                    add_manifest(
                        efficiency,
                        efficiency.with_name("model.txt"),
                        f"{target}:efficiency:{dep_name}",
                    )
                elif dep_name in proxy_features:
                    present = available_columns is None or dep_name in available_columns
                    if not present:
                        raise ValueError(
                            f"Missing selected component proxy feature {dep_name} for {target}"
                        )
                    proxy_checks.append(
                        {
                            "target": target,
                            "feature": dep_name,
                            "dependency_type": "deterministic_pregame_volume_proxy",
                            "present": True,
                        }
                    )
                else:
                    raise FileNotFoundError(
                        f"No component/efficiency manifest for selected dependency {dep_name}"
                    )
    return specs, proxy_checks


def resolve_current_season(args: argparse.Namespace, config: dict[str, Any]) -> int:
    season = int(args.season if args.season is not None else config["seasons"]["current"])
    if season != int(config["seasons"]["current"]):
        raise ValueError(
            f"Issue 31 current builder season must equal configured seasons.current; "
            f"configured={config['seasons']['current']} requested={season}"
        )
    if args.week < 1 or args.week > 30:
        raise ValueError(f"Invalid NFL week: {args.week}")
    return season


def current_realized_sources(
    repo: Path,
    season: int,
    week: int,
    config: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    source_root = repo / config["paths"]["current_source_root"]
    paths = {
        "player_stats": source_root / f"stats_player_week_{season}.parquet",
        "snap_counts": source_root / f"snap_counts_{season}.parquet",
        "participation": source_root / f"pbp_participation_{season}.parquet",
        "pbp": repo / config["paths"]["pbp_pattern"].format(season=season),
        "team_stats": repo / config["paths"]["team_stats_pattern"].format(season=season),
    }
    frames: dict[str, pd.DataFrame] = {}
    audit: dict[str, Any] = {}

    for name, path in paths.items():
        exists = path.is_file()
        item: dict[str, Any] = {
            "path": str(path.relative_to(repo)),
            "exists": exists,
            "required_for_week": week > 1,
            "rows_before_filter": 0,
            "rows_used": 0,
            "max_source_week_used": None,
            "status": None,
        }
        if not exists:
            if week > 1:
                raise FileNotFoundError(
                    f"Issue 31 week {week} requires completed-current-season source: {path}"
                )
            frames[name] = pd.DataFrame()
            item["status"] = "not_applicable_no_completed_weeks"
            audit[name] = item
            continue

        # Week 1 may have pre-created current-season source files that are
        # intentionally empty because no 2026 game has been completed yet.
        # Treat both zero-byte files and compressed/headerless empty CSVs as
        # no completed-week data. From Week 2 onward, an empty required
        # realized source is a hard failure.
        empty_source = False
        if path.stat().st_size == 0:
            empty_source = True
            df = pd.DataFrame()
        else:
            try:
                if path.suffix.casefold() == ".parquet":
                    df = pd.read_parquet(path)
                else:
                    df = pd.read_csv(path, low_memory=False)
            except (pd.errors.EmptyDataError, EOFError):
                empty_source = True
                df = pd.DataFrame()

        if empty_source:
            if week > 1:
                raise RuntimeError(
                    f"Issue 31 week {week} requires nonempty completed-current-season source: {path}"
                )
            frames[name] = df
            item["status"] = "empty_source_no_completed_weeks"
            audit[name] = item
            continue

        item["rows_before_filter"] = int(len(df))

        week_col = first_existing(df.columns, ["week", "week_num", "game_week"])
        season_col = first_existing(df.columns, ["season", "season_year"])
        if season_col is not None:
            df = df.loc[as_num(df[season_col]).eq(season)].copy()
        if week_col is None:
            if week > 1 and not df.empty:
                raise ValueError(f"Current realized source lacks week column: {path}")
            filtered = df.iloc[0:0].copy() if week == 1 else df.copy()
        else:
            source_week = as_num(df[week_col])
            if week == 1:
                filtered = df.iloc[0:0].copy()
            else:
                filtered = df.loc[source_week.lt(week)].copy()
                if not filtered.empty:
                    used_week = as_num(filtered[week_col]).dropna()
                    if not used_week.empty:
                        max_week = int(used_week.max())
                        if max_week >= week:
                            raise ValueError(
                                f"Same/future current-season source leaked: {path} max_week={max_week}"
                            )
                        item["max_source_week_used"] = max_week

        forbidden = config.get("forbidden_features")
        if not isinstance(forbidden, list) or not forbidden:
            raise ValueError(
                "Config section 'forbidden_features' must be a non-empty list."
            )
        forbidden_tokens = [
            str(value).strip().casefold()
            for value in forbidden
            if str(value).strip()
        ]
        forbidden_columns = [
            column
            for column in filtered.columns
            if any(
                token in str(column).casefold()
                for token in forbidden_tokens
            )
        ]
        if forbidden_columns:
            filtered = filtered.drop(columns=forbidden_columns)

        common.reject_forbidden_feature_columns(filtered.columns, config)
        item["rows_used"] = int(len(filtered))
        item["status"] = "read_filtered_strictly_prior" if week > 1 else "read_but_no_completed_weeks"
        frames[name] = filtered
        audit[name] = item

    return frames, audit


def load_prior_reference(
    historical_path: Path,
    prior_season: int,
    columns: list[str],
) -> pd.DataFrame:
    try:
        frame = pd.read_parquet(
            historical_path,
            columns=columns,
            filters=[("season", "==", prior_season)],
        )
    except Exception:
        # Some parquet engines cannot push filters through every local file.
        frame = pd.read_parquet(historical_path, columns=columns)
        frame = frame.loc[as_num(frame["season"]).eq(prior_season)].copy()
    if frame.empty:
        raise RuntimeError(f"No historical feature priors for season {prior_season}: {historical_path}")
    frame["game_id"] = frame["game_id"].map(norm_game_id)
    frame["player_id"] = frame["player_id"].map(common.normalize_player_id)
    frame["team"] = frame["team"].map(norm_team)
    frame["opponent"] = frame["opponent"].map(norm_team)
    frame["position"] = frame["position"].map(norm_position)
    frame["position_group"] = frame["position_group"].map(norm_position)
    frame["_kickoff_sort"] = pd.to_datetime(frame["kickoff_timestamp"], errors="coerce", utc=True)
    return frame


def latest_rows(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    ordered = frame.sort_values(
        [*keys, "_kickoff_sort", "week", "game_id"],
        kind="mergesort",
    )
    return ordered.groupby(keys, sort=False, dropna=False).tail(1).copy()


def fallback_by_position(
    out: pd.DataFrame,
    reference: pd.DataFrame,
    columns: list[str],
) -> None:
    if not columns:
        return
    for column in columns:
        if column not in out.columns or column not in reference.columns:
            continue
        missing = out[column].isna()
        if not missing.any():
            continue
        if pd.api.types.is_numeric_dtype(reference[column].dtype):
            pos_value = reference.groupby("position_group", dropna=False)[column].median()
            fill = out.loc[missing, "position_group"].map(pos_value)
            out.loc[missing, column] = fill.to_numpy()
            still = out[column].isna()
            if still.any():
                league = as_num(reference[column]).median()
                if pd.notna(league):
                    out.loc[still, column] = league
        else:
            def mode_or_blank(series: pd.Series) -> str:
                x = series.dropna().astype(str)
                x = x.loc[x.str.strip().ne("")]
                return x.mode().iloc[0] if not x.empty and not x.mode().empty else ""
            pos_value = reference.groupby("position_group", dropna=False)[column].agg(mode_or_blank)
            fill = out.loc[missing, "position_group"].map(pos_value)
            out.loc[missing, column] = fill.to_numpy()


def overlay_player_priors(
    out: pd.DataFrame,
    reference: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    source = latest_rows(reference, ["player_id"])[["player_id", "team", *columns]].copy()
    source = source.rename(columns={"team": "_prior_player_team", **{c: f"_prior_{c}" for c in columns}})
    out = out.merge(source, on="player_id", how="left", validate="many_to_one", sort=False)
    # Materialize the wide player-prior block in one rename operation instead
    # of hundreds of per-column inserts, which heavily fragments pandas frames.
    existing = [column for column in columns if column in out.columns]
    if existing:
        out = out.drop(columns=existing)
    out = out.rename(columns={f"_prior_{column}": column for column in columns})
    out["_prior_player_team"] = out["_prior_player_team"].fillna("")
    return out.copy()


def overlay_context_priors(
    out: pd.DataFrame,
    reference: pd.DataFrame,
    family_columns: dict[str, list[str]],
) -> pd.DataFrame:
    team_cols = family_columns.get("team", [])
    if team_cols:
        team_latest = latest_rows(reference, ["team"])[["team", *team_cols]].copy()
        team_ref = team_latest.rename(columns={c: f"_ctx_{c}" for c in team_cols})
        out = out.merge(team_ref, on="team", how="left", validate="many_to_one", sort=False)
        # Replace the stale player-row team context as one block. Repeated
        # per-column assignment fragments wide DataFrames and is needlessly slow.
        existing_team_cols = [c for c in team_cols if c in out.columns]
        if existing_team_cols:
            out = out.drop(columns=existing_team_cols)
        out = out.rename(columns={f"_ctx_{c}": c for c in team_cols})

        # Opponent offensive priors are needed for defensive-player matchup
        # volume. They are keyed from the opponent's own team-form history,
        # never from the current player's prior opponent.
        opp_team_cols = [
            c for c in ["team_offensive_plays_roll3_mean", "team_dropbacks_roll3_mean"]
            if c in team_latest.columns
        ]
        if opp_team_cols:
            rename = {
                "team_offensive_plays_roll3_mean": "_opp_offensive_plays_roll3",
                "team_dropbacks_roll3_mean": "_opp_offense_dropbacks_roll3",
            }
            opp_team = team_latest[["team", *opp_team_cols]].rename(
                columns={"team": "opponent", **{c: rename[c] for c in opp_team_cols}}
            )
            out = out.merge(
                opp_team, on="opponent", how="left", validate="many_to_one", sort=False
            )

    opp_cols = family_columns.get("opponent", [])
    if opp_cols:
        opp_latest = latest_rows(reference, ["opponent"])[["opponent", *opp_cols]].copy()
        opp_ref = opp_latest.rename(columns={c: f"_ctx_{c}" for c in opp_cols})
        out = out.merge(opp_ref, on="opponent", how="left", validate="many_to_one", sort=False)
        existing_opp_cols = [c for c in opp_cols if c in out.columns]
        if existing_opp_cols:
            out = out.drop(columns=existing_opp_cols)
        out = out.rename(columns={f"_ctx_{c}": c for c in opp_cols})

        own_def_cols = [
            c for c in ["opponent_defensive_plays_roll3_mean", "opponent_opponent_dropbacks_roll3_mean"]
            if c in opp_latest.columns
        ]
        if own_def_cols:
            rename = {
                "opponent_defensive_plays_roll3_mean": "_team_defensive_plays_roll3",
                "opponent_opponent_dropbacks_roll3_mean": "_team_defense_dropbacks_roll3",
            }
            own_def = opp_latest[["opponent", *own_def_cols]].rename(
                columns={"opponent": "team", **{c: rename[c] for c in own_def_cols}}
            )
            out = out.merge(
                own_def, on="team", how="left", validate="many_to_one", sort=False
            )

    position_cols = [
        c for c in family_columns.get("matchup", [])
        if c.startswith("matchup_position_allowed_")
    ]
    if position_cols:
        pos_ref = latest_rows(reference, ["opponent", "position_group"])[
            ["opponent", "position_group", *position_cols]
        ].copy()
        pos_ref = pos_ref.rename(columns={c: f"_ctx_{c}" for c in position_cols})
        out = out.merge(
            pos_ref,
            on=["opponent", "position_group"],
            how="left",
            validate="many_to_one",
            sort=False,
        )
        existing_position_cols = [c for c in position_cols if c in out.columns]
        if existing_position_cols:
            out = out.drop(columns=existing_position_cols)
        out = out.rename(columns={f"_ctx_{c}": c for c in position_cols})
    # Consolidate blocks before the remaining current-week overlays.
    return out.copy()


def overlay_position_allowed_final_week(
    out: pd.DataFrame,
    position_allowed: pd.DataFrame,
    prior_season: int,
    matchup_columns: list[str],
) -> pd.DataFrame:
    if position_allowed.empty:
        return out
    needed_keys = {"season", "week", "defense_team", "offense_position_group"}
    if not needed_keys.issubset(position_allowed.columns):
        raise ValueError("Position-allowed source missing required grain columns.")
    pa = position_allowed.loc[as_num(position_allowed["season"]).eq(prior_season)].copy()
    if pa.empty:
        raise ValueError(f"Position-allowed source has no {prior_season} rows.")
    pa["week"] = as_num(pa["week"])
    pa["_defense"] = pa["defense_team"].map(norm_team)
    pa["_position"] = pa["offense_position_group"].map(norm_position)
    pa = pa.sort_values(["_defense", "_position", "week"], kind="mergesort")
    pa = pa.groupby(["_defense", "_position"], sort=False).tail(1)
    source_cols: dict[str, str] = {}
    for target_col in matchup_columns:
        prefix = "matchup_position_allowed_"
        suffix = "_lag1"
        if target_col.startswith(prefix) and target_col.endswith(suffix):
            raw = target_col[len(prefix):-len(suffix)]
            if raw in pa.columns:
                source_cols[raw] = target_col
    if not source_cols:
        return out
    join = pa[["_defense", "_position", *source_cols.keys()]].rename(columns=source_cols)
    join = join.rename(columns={c: f"_pa_{c}" for c in source_cols.values()})
    out = out.merge(
        join,
        left_on=["opponent", "position_group"],
        right_on=["_defense", "_position"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    for target_col in source_cols.values():
        value_col = f"_pa_{target_col}"
        replace = out[value_col].notna()
        out.loc[replace, target_col] = out.loc[replace, value_col]
        out = out.drop(columns=[value_col])
    out = out.drop(columns=[c for c in ["_defense", "_position"] if c in out.columns])
    return out


def current_role_overrides(out: pd.DataFrame, universe_all: pd.DataFrame, roles: pd.DataFrame) -> None:
    # Current depth/injury data overrides historical role state.
    depth = as_num(out["depth_rank"])
    if "role_depth_rank_pregame" in out:
        out["role_depth_rank_pregame"] = depth
    if "role_depth_starter_flag_pregame" in out:
        out["role_depth_starter_flag_pregame"] = as_num(out["depth_starter_flag"]).fillna(0)

    injury = out["injury_game_status"].map(norm_injury)
    if "role_injury_status_pregame" in out:
        out["role_injury_status_pregame"] = injury
    for status in ["out", "doubtful", "questionable"]:
        column = f"role_injury_{status}_flag"
        if column in out:
            out[column] = injury.eq(status).astype("int8")

    if "role_primary_kicker_flag" in out:
        out["role_primary_kicker_flag"] = as_num(out["primary_kicker_flag"]).fillna(0).astype("int8")

    if "role_starter_promotion_flag" in out:
        out["role_starter_promotion_flag"] = (
            as_num(out["starter_flag"]).fillna(0).eq(1)
            & as_num(out["depth_starter_flag"]).fillna(0).eq(0)
        ).astype("int8")
    if "role_starter_demotion_flag" in out:
        out["role_starter_demotion_flag"] = (
            as_num(out["starter_flag"]).fillna(0).eq(0)
            & as_num(out["depth_starter_flag"]).fillna(0).eq(1)
        ).astype("int8")
    if "role_role_missing_flag" in out:
        out["role_role_missing_flag"] = depth.isna().astype("int8")

    pos = out["position"].map(norm_position)
    if "role_defensive_starter_flag" in out:
        out["role_defensive_starter_flag"] = (
            pos.isin(DEFENSE) & as_num(out["starter_flag"]).fillna(0).eq(1)
        ).astype("int8")
    if "role_front7_flag" in out:
        out["role_front7_flag"] = pos.isin(FRONT7).astype("int8")
    if "role_secondary_flag" in out:
        out["role_secondary_flag"] = pos.isin(SECONDARY).astype("int8")

    if "role_team_change_flag" in out:
        prior_team = out["_prior_player_team"].map(norm_team)
        out["role_team_change_flag"] = (
            prior_team.ne("") & prior_team.ne(out["team"])
        ).astype("int8")
    if "history_new_team_flag" in out:
        prior_team = out["_prior_player_team"].map(norm_team)
        history_exists = (
            as_num(out.get("history_history_games", pd.Series(0, index=out.index)))
            .fillna(0)
            .gt(0)
        )
        out["history_new_team_flag"] = (
            history_exists & prior_team.ne("") & prior_team.ne(out["team"])
        ).astype("int8")
    if "history_no_nfl_history_flag" in out:
        no_hist = out["_prior_player_team"].eq("")
        out.loc[no_hist, "history_no_nfl_history_flag"] = 1
        if "history_history_games" in out:
            out.loc[no_hist, "history_history_games"] = 0

    # Teammate Out count by current team + normalized position group.
    if "role_teammate_out_count_position" in out:
        u = universe_all.copy()
        u["team"] = u["team"].map(norm_team)
        u["position_group"] = u["position_group"].map(norm_position)
        out_flag = u["injury_game_status"].map(norm_injury).eq("out")
        counts = (
            u.loc[out_flag]
            .groupby(["team", "position_group"], dropna=False)
            .size()
            .rename("_out_count")
            .reset_index()
        )
        probe = out[["team", "position_group"]].merge(
            counts, on=["team", "position_group"], how="left", validate="many_to_one"
        )
        own_out = injury.eq("out").astype(int)
        teammate_out = (
            probe["_out_count"].fillna(0).to_numpy(dtype="float64")
            - own_out.to_numpy(dtype="float64")
        )
        # NumPy arrays do not support pandas Series.clip(lower=...).
        out["role_teammate_out_count_position"] = np.maximum(teammate_out, 0.0)


def overlay_environment(
    out: pd.DataFrame,
    weather: pd.DataFrame,
    travel: pd.DataFrame,
    reference: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    w = weather.copy()
    t = travel.copy()
    game_col_w = first_existing(w.columns, ["game_id"])
    game_col_t = first_existing(t.columns, ["game_id"])
    if game_col_w is None or game_col_t is None:
        raise ValueError("Weather/travel must contain game_id.")
    w["game_id"] = w[game_col_w].map(norm_game_id)
    t["game_id"] = t[game_col_t].map(norm_game_id)
    if w["game_id"].duplicated().any():
        raise ValueError("Weather contains duplicate game_id rows.")
    if t["game_id"].duplicated().any():
        raise ValueError("Travel contains duplicate game_id rows.")

    w_keep = ["game_id"] + [
        c for c in ["stadium", "temperature", "wind_speed", "roof_type"] if c in w.columns
    ]
    t_keep = ["game_id"] + [
        c for c in [
            "miles_traveled", "time_zones_crossed", "east_to_west", "west_to_east",
            "international_flag", "neutral_site_flag",
        ] if c in t.columns
    ]
    out = out.merge(
        w[w_keep].rename(columns={c: f"_w_{c}" for c in w_keep if c != "game_id"}),
        on="game_id", how="left", validate="many_to_one", sort=False,
    )
    out = out.merge(
        t[t_keep].rename(columns={c: f"_t_{c}" for c in t_keep if c != "game_id"}),
        on="game_id", how="left", validate="many_to_one", sort=False,
    )

    # Clear target-game environment features before populating current values;
    # stale environment from a prior player game is not a valid fallback.
    env_cols = [c for c in out.columns if c.startswith("environment_")]
    for c in env_cols:
        out[c] = np.nan if pd.api.types.is_numeric_dtype(reference[c].dtype) else ""

    if "environment_stadium" in out:
        out["environment_stadium"] = out.get("_w_stadium", "")
    if "environment_temperature" in out:
        out["environment_temperature"] = as_num(out.get("_w_temperature", pd.Series(np.nan, index=out.index)))
    if "environment_wind" in out:
        out["environment_wind"] = as_num(out.get("_w_wind_speed", pd.Series(np.nan, index=out.index)))

    # Historical categorical vocabulary uses normalized broad roof values.
    if "environment_roof" in out:
        raw_roof = out.get("_w_roof_type", pd.Series("", index=out.index)).fillna("").astype(str).str.casefold()
        roof = raw_roof.map(
            {
                "fixed_roof": "dome",
                "dome": "dome",
                "retractable": "closed",
                "retractable_roof": "closed",
                "open_air": "outdoors",
                "open": "open",
                "outdoors": "outdoors",
            }
        ).fillna("")
        out["environment_roof"] = roof

    # Surface/stadium_id are recovered by current stadium name from historical
    # environment metadata only; never by player or opponent.
    if "environment_stadium" in out and "environment_surface" in out:
        stadium_ref = reference[[
            c for c in ["environment_stadium", "environment_surface", "environment_stadium_id"]
            if c in reference.columns
        ]].copy()
        if "environment_stadium" in stadium_ref:
            stadium_ref["environment_stadium"] = stadium_ref["environment_stadium"].fillna("").astype(str)
            stadium_ref = stadium_ref.loc[stadium_ref["environment_stadium"].str.strip().ne("")]
            stadium_ref = stadium_ref.drop_duplicates("environment_stadium", keep="last")
            mapping_surface = (
                stadium_ref.set_index("environment_stadium")["environment_surface"]
                if "environment_surface" in stadium_ref else pd.Series(dtype=object)
            )
            out["environment_surface"] = out["environment_stadium"].map(mapping_surface).fillna("")
            if "environment_stadium_id" in out and "environment_stadium_id" in stadium_ref:
                out["environment_stadium_id"] = out["environment_stadium"].map(
                    stadium_ref.set_index("environment_stadium")["environment_stadium_id"]
                )

    home = as_num(out["home_flag"]).fillna(0).eq(1)
    away = ~home
    miles = as_num(out.get("_t_miles_traveled", pd.Series(np.nan, index=out.index)))
    zones = as_num(out.get("_t_time_zones_crossed", pd.Series(np.nan, index=out.index)))
    east = as_num(out.get("_t_east_to_west", pd.Series(np.nan, index=out.index))).fillna(0)
    west = as_num(out.get("_t_west_to_east", pd.Series(np.nan, index=out.index))).fillna(0)

    assignments = {
        "environment_team_miles_traveled": np.where(away, miles, 0.0),
        "environment_opponent_miles_traveled": np.where(home, miles, 0.0),
        "environment_team_time_zones_crossed": np.where(away, zones, 0.0),
        "environment_opponent_time_zones_crossed": np.where(home, zones, 0.0),
        "environment_team_east_to_west_flag": np.where(away, east, 0),
        "environment_opponent_east_to_west_flag": np.where(home, east, 0),
        "environment_team_west_to_east_flag": np.where(away, west, 0),
        "environment_opponent_west_to_east_flag": np.where(home, west, 0),
    }
    for c, values in assignments.items():
        if c in out:
            out[c] = values

    if "environment_neutral_site_flag" in out:
        out["environment_neutral_site_flag"] = as_num(
            out.get("_t_neutral_site_flag", pd.Series(np.nan, index=out.index))
        ).fillna(0)
    if "environment_international_flag" in out:
        out["environment_international_flag"] = as_num(
            out.get("_t_international_flag", pd.Series(np.nan, index=out.index))
        ).fillna(0)
    if "environment_divisional_game_flag" in out:
        out["environment_divisional_game_flag"] = [
            int(DIVISION.get(a) != "" and DIVISION.get(a) == DIVISION.get(b))
            for a, b in zip(out["team"], out["opponent"])
        ]
    if "environment_weather_missing_flag" in out:
        weather_missing = out.get("_w_temperature", pd.Series(np.nan, index=out.index)).isna()
        out["environment_weather_missing_flag"] = weather_missing.astype("int8")
    if "environment_travel_missing_flag" in out:
        travel_missing = out.get("_t_miles_traveled", pd.Series(np.nan, index=out.index)).isna()
        out["environment_travel_missing_flag"] = travel_missing.astype("int8")

    # Current target-game rest is not provided in the Issue 31 inputs. Do not
    # retain unrelated prior-game rest values.
    for c in ["environment_team_rest_days", "environment_opponent_rest_days"]:
        if c in out:
            out[c] = np.nan

    helper = [c for c in out.columns if c.startswith("_w_") or c.startswith("_t_")]
    out = out.drop(columns=helper)
    return out, {
        "weather_join_key": "game_id",
        "travel_join_key": "game_id",
        "weather_rows": int(len(w)),
        "travel_rows": int(len(t)),
        "weather_games_matched": int(out["game_id"].isin(set(w["game_id"])).sum()),
        "travel_games_matched": int(out["game_id"].isin(set(t["game_id"])).sum()),
    }


def recompute_matchups(out: pd.DataFrame) -> None:
    def exists(*cols: str) -> bool:
        return all(c in out.columns for c in cols)

    if exists("matchup_expected_team_plays", "team_offensive_plays_roll3_mean", "opponent_defensive_plays_roll3_mean"):
        out["matchup_expected_team_plays"] = safe_mean_pair(
            out["team_offensive_plays_roll3_mean"], out["opponent_defensive_plays_roll3_mean"]
        )
    if exists("matchup_expected_team_dropbacks", "team_dropbacks_roll3_mean", "opponent_opponent_dropbacks_roll3_mean"):
        out["matchup_expected_team_dropbacks"] = safe_mean_pair(
            out["team_dropbacks_roll3_mean"], out["opponent_opponent_dropbacks_roll3_mean"]
        )
    if exists("matchup_expected_team_rush_attempts", "team_rush_attempts_roll3_mean", "opponent_opponent_rush_attempts_roll3_mean"):
        out["matchup_expected_team_rush_attempts"] = safe_mean_pair(
            out["team_rush_attempts_roll3_mean"], out["opponent_opponent_rush_attempts_roll3_mean"]
        )
    if exists("matchup_expected_opponent_plays", "_opp_offensive_plays_roll3", "_team_defensive_plays_roll3"):
        out["matchup_expected_opponent_plays"] = safe_mean_pair(
            out["_opp_offensive_plays_roll3"], out["_team_defensive_plays_roll3"]
        )
    if exists("matchup_expected_opponent_dropbacks", "_opp_offense_dropbacks_roll3", "_team_defense_dropbacks_roll3"):
        out["matchup_expected_opponent_dropbacks"] = safe_mean_pair(
            out["_opp_offense_dropbacks_roll3"], out["_team_defense_dropbacks_roll3"]
        )
    if exists("matchup_player_target_share_x_opp_targets", "player_target_share_roll3_mean", "matchup_position_allowed_targets_allowed_lag1"):
        out["matchup_player_target_share_x_opp_targets"] = safe_product(
            out["player_target_share_roll3_mean"], out["matchup_position_allowed_targets_allowed_lag1"]
        )
    if exists("matchup_player_carry_share_x_opp_rushes", "player_carry_share_roll3_mean", "matchup_position_allowed_carries_allowed_lag1"):
        out["matchup_player_carry_share_x_opp_rushes"] = safe_product(
            out["player_carry_share_roll3_mean"], out["matchup_position_allowed_carries_allowed_lag1"]
        )
    if exists("matchup_player_tackle_rate_x_opp_plays", "player_tackle_rate_per_def_play_roll3_mean", "matchup_expected_opponent_plays"):
        out["matchup_player_tackle_rate_x_opp_plays"] = safe_product(
            out["player_tackle_rate_per_def_play_roll3_mean"], out["matchup_expected_opponent_plays"]
        )
    if exists("matchup_player_sack_rate_x_opp_plays", "player_sack_rate_per_def_play_roll5_mean", "matchup_expected_opponent_plays"):
        out["matchup_player_sack_rate_x_opp_plays"] = safe_product(
            out["player_sack_rate_per_def_play_roll5_mean"], out["matchup_expected_opponent_plays"]
        )
    if exists("matchup_off_epa_vs_def_epa", "team_off_epa_per_play_roll3_mean", "opponent_def_epa_per_play_roll3_mean"):
        out["matchup_off_epa_vs_def_epa"] = (
            as_num(out["team_off_epa_per_play_roll3_mean"])
            - as_num(out["opponent_def_epa_per_play_roll3_mean"])
        )
    if exists("matchup_pass_rate_vs_opponent", "team_pass_rate_roll3_mean", "opponent_opponent_pass_attempts_roll3_mean", "opponent_defensive_plays_roll3_mean"):
        out["matchup_pass_rate_vs_opponent"] = (
            as_num(out["team_pass_rate_roll3_mean"])
            - safe_divide(out["opponent_opponent_pass_attempts_roll3_mean"], out["opponent_defensive_plays_roll3_mean"])
        )
    if exists("matchup_rush_rate_vs_opponent", "team_rush_rate_roll3_mean", "opponent_opponent_rush_attempts_roll3_mean", "opponent_defensive_plays_roll3_mean"):
        out["matchup_rush_rate_vs_opponent"] = (
            as_num(out["team_rush_rate_roll3_mean"])
            - safe_divide(out["opponent_opponent_rush_attempts_roll3_mean"], out["opponent_defensive_plays_roll3_mean"])
        )


def overlay_current_player_stats(out: pd.DataFrame, stats: pd.DataFrame, week: int) -> dict[str, Any]:
    if stats.empty or week <= 1:
        return {"metrics_updated": 0, "rows_used": int(len(stats))}
    id_col = first_existing(stats.columns, ["player_id", "gsis_id"])
    week_col = first_existing(stats.columns, ["week", "week_num"])
    if id_col is None or week_col is None:
        return {"metrics_updated": 0, "rows_used": int(len(stats)), "status": "schema_not_player_week"}
    x = stats.copy()
    x["_player_id"] = x[id_col].map(common.normalize_player_id)
    x["_week"] = as_num(x[week_col])
    x = x.loc[x["_player_id"].ne("") & x["_week"].lt(week)].copy()
    if x.empty:
        return {"metrics_updated": 0, "rows_used": 0}
    x = x.sort_values(["_player_id", "_week"], kind="mergesort")
    updated = 0
    for metric, aliases in CURRENT_PLAYER_METRIC_ALIASES.items():
        source_col = first_existing(x.columns, aliases)
        if source_col is None:
            continue
        values = as_num(x[source_col])
        work = pd.DataFrame({"player_id": x["_player_id"], "week": x["_week"], "value": values})
        work = work.loc[work["value"].notna()]
        if work.empty:
            continue
        latest = work.groupby("player_id", sort=False).tail(1).set_index("player_id")["value"]
        season_mean = work.groupby("player_id", sort=False)["value"].mean()
        lag_col = f"player_{metric}_lag1"
        std_col = f"player_{metric}_season_to_date"
        if lag_col in out:
            mapped = out["player_id"].map(latest)
            mask = mapped.notna()
            out.loc[mask, lag_col] = mapped.loc[mask]
            updated += 1
        if std_col in out:
            mapped = out["player_id"].map(season_mean)
            mask = mapped.notna()
            out.loc[mask, std_col] = mapped.loc[mask]
            updated += 1
    return {"metrics_updated": updated, "rows_used": int(len(x))}


def overlay_current_snaps(out: pd.DataFrame, snaps: pd.DataFrame, week: int) -> dict[str, Any]:
    if snaps.empty or week <= 1:
        return {"fields_updated": 0, "rows_used": int(len(snaps))}
    id_col = first_existing(snaps.columns, ["player_id", "gsis_id"])
    week_col = first_existing(snaps.columns, ["week", "week_num"])
    if id_col is None or week_col is None:
        return {"fields_updated": 0, "rows_used": int(len(snaps)), "status": "no_gsis_player_week"}
    x = snaps.copy()
    x["_player_id"] = x[id_col].map(common.normalize_player_id)
    x["_week"] = as_num(x[week_col])
    x = x.loc[x["_week"].lt(week)].sort_values(["_player_id", "_week"], kind="mergesort")
    updated = 0
    for source_aliases, output_col in [
        (["offense_pct", "offense_snap_pct"], "role_prior_offense_snap_pct"),
        (["defense_pct", "defense_snap_pct"], "role_prior_defense_snap_pct"),
    ]:
        source_col = first_existing(x.columns, source_aliases)
        if source_col is None or output_col not in out:
            continue
        latest = x.assign(_v=as_num(x[source_col])).groupby("_player_id", sort=False).tail(1).set_index("_player_id")["_v"]
        mapped = out["player_id"].map(latest)
        mask = mapped.notna()
        out.loc[mask, output_col] = mapped.loc[mask]
        updated += 1
    return {"fields_updated": updated, "rows_used": int(len(x))}


def overlay_current_team_stats(out: pd.DataFrame, team_stats: pd.DataFrame, week: int) -> dict[str, Any]:
    if team_stats.empty or week <= 1:
        return {"metrics_updated": 0, "rows_used": int(len(team_stats))}
    team_col = first_existing(team_stats.columns, ["team", "team_abbr", "recent_team"])
    week_col = first_existing(team_stats.columns, ["week", "week_num"])
    if team_col is None or week_col is None:
        return {"metrics_updated": 0, "rows_used": int(len(team_stats)), "status": "schema_not_team_week"}
    x = team_stats.copy()
    x["_team"] = x[team_col].map(norm_team)
    x["_week"] = as_num(x[week_col])
    x = x.loc[x["_week"].lt(week)].sort_values(["_team", "_week"], kind="mergesort")
    updated = 0
    for metric, aliases in CURRENT_TEAM_METRIC_ALIASES.items():
        source_col = first_existing(x.columns, aliases)
        if source_col is None:
            continue
        values = as_num(x[source_col])
        work = pd.DataFrame({"team": x["_team"], "week": x["_week"], "value": values})
        work = work.loc[work["value"].notna()]
        if work.empty:
            continue
        latest = work.groupby("team", sort=False).tail(1).set_index("team")["value"]
        season_mean = work.groupby("team", sort=False)["value"].mean()
        lag_col = f"team_{metric}_lag1"
        std_col = f"team_{metric}_season_to_date"
        if lag_col in out:
            mapped = out["team"].map(latest)
            mask = mapped.notna()
            out.loc[mask, lag_col] = mapped.loc[mask]
            updated += 1
        if std_col in out:
            mapped = out["team"].map(season_mean)
            mask = mapped.notna()
            out.loc[mask, std_col] = mapped.loc[mask]
            updated += 1
    return {"metrics_updated": updated, "rows_used": int(len(x))}


def cast_like_reference(out: pd.DataFrame, reference: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        dtype = reference[column].dtype
        try:
            if pd.api.types.is_datetime64_any_dtype(dtype):
                out[column] = pd.to_datetime(out[column], errors="coerce", utc=True)
                # Historical may be tz-aware or naive; preserve as closely as possible.
                if getattr(dtype, "tz", None) is None and getattr(out[column].dtype, "tz", None) is not None:
                    out[column] = out[column].dt.tz_convert(None)
            elif pd.api.types.is_integer_dtype(dtype):
                numeric = as_num(out[column])
                if numeric.isna().any() and not pd.api.types.is_extension_array_dtype(dtype):
                    # Do not invent values merely to satisfy a non-nullable dtype.
                    # Historical integer feature columns should be complete; current
                    # missing integer context falls back to zero only for flag-like fields.
                    if column.endswith("_flag") or "_flag_" in column or column in {"season", "week", "home_flag"}:
                        numeric = numeric.fillna(0)
                    else:
                        numeric = numeric.fillna(0)
                out[column] = numeric.astype(dtype)
            elif pd.api.types.is_float_dtype(dtype):
                out[column] = as_num(out[column]).astype(dtype)
            elif pd.api.types.is_bool_dtype(dtype):
                out[column] = out[column].fillna(False).astype(dtype)
            else:
                out[column] = out[column].where(out[column].notna(), "").astype(dtype)
        except Exception as exc:
            raise ValueError(f"Failed to cast current feature {column} to historical dtype {dtype}") from exc
    return out


def validate_model_slices(
    out: pd.DataFrame,
    specs: list[dict[str, Any]],
    config: dict[str, Any],
    repo: Path,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for spec in specs:
        manifest = spec["manifest"]
        numeric = spec["numeric"]
        categorical = spec["categorical"]
        ordered = numeric + categorical
        canonical = list(spec.get("canonical", ordered))
        derived = list(spec.get("derived", []))

        # Efficiency manifests deliberately reuse generic eff_* names across
        # multiple models. Those values are model-local shrinkage features and
        # cannot be represented correctly as one global current-feature column.
        # Issue 31 must provide every canonical input; the efficiency inference
        # adapter materializes the derived block for each model separately.
        missing = [c for c in canonical if c not in out.columns]
        leaked_derived = [c for c in derived if c in out.columns]

        required = [clean(value) for value in manifest.get("required_features", []) if clean(value)]
        unknown_required = [c for c in required if c not in ordered]
        required_canonical = [c for c in required if c not in derived]
        missing_required = [c for c in required_canonical if c not in out.columns]

        if unknown_required:
            raise ValueError(
                f"Manifest required_features outside model schema for {spec['owner']}: "
                f"{unknown_required[:20]}"
            )
        if missing or missing_required:
            raise ValueError(
                f"Missing selected-model canonical feature(s) for {spec['owner']}: "
                f"missing={missing[:20]} missing_required={missing_required[:20]}"
            )
        if leaked_derived:
            raise ValueError(
                f"Model-local derived efficiency feature(s) leaked into global current table "
                f"for {spec['owner']}: {leaked_derived[:20]}"
            )

        common.reject_forbidden_feature_columns(ordered, config)

        canonical_numeric = [c for c in numeric if c in canonical]
        canonical_categorical = [c for c in categorical if c in canonical]
        bad_numeric = [
            c for c in canonical_numeric
            if not pd.api.types.is_numeric_dtype(out[c].dtype)
            and not pd.api.types.is_bool_dtype(out[c].dtype)
        ]
        bad_categorical = [
            c for c in canonical_categorical
            if not (
                pd.api.types.is_object_dtype(out[c].dtype)
                or pd.api.types.is_string_dtype(out[c].dtype)
                or isinstance(out[c].dtype, pd.CategoricalDtype)
            )
        ]
        if bad_numeric or bad_categorical:
            raise ValueError(
                f"Selected-model canonical feature type mismatch for {spec['owner']}: "
                f"numeric={bad_numeric[:10]} categorical={bad_categorical[:10]}"
            )

        checks.append(
            {
                "owner": spec["owner"],
                "manifest": str(spec["manifest_path"].relative_to(repo)),
                "model": str(spec["model_path"].relative_to(repo)),
                "feature_count": len(ordered),
                "canonical_feature_count": len(canonical),
                "derived_feature_count": len(derived),
                "derived_features": derived,
                "derived_feature_policy": (
                    "model_local_at_efficiency_inference" if derived else "none"
                ),
                "required_feature_count": len(required),
                "calculated_feature_hash": spec["calculated_hash"],
                "stored_feature_hash": spec["stored_hash"],
                "stored_hash_matches": spec["stored_hash"] in {None, spec["calculated_hash"]},
                "model_feature_order_checked": spec["model_feature_order_checked"],
                "missing_features": [],
                "missing_required_features": [],
            }
        )
    return checks


def main() -> int:
    args = parse_args()
    config = common.load_config()
    repo = common.repo_root()
    prop = common.prop_root()
    season = resolve_current_season(args, config)
    week = int(args.week)
    prior_season = season - 1

    market_audit = run_market_audit(repo)

    universe_path = prop / f"data/current/{season}_week_{week}_universe.parquet"
    roles_path = prop / f"data/current/{season}_week_{week}_roles.parquet"
    historical_path = repo / config["paths"]["historical_features"]
    historical_manifest_path = historical_path.with_name("feature_manifest.json")
    position_allowed_path = repo / config["paths"]["position_allowed"]
    weather_path = repo / config["paths"]["current_weather"].format(week=week)
    travel_path = repo / config["paths"]["current_travel"].format(season=season, week=week)
    output_path = prop / f"data/current/features/{season}_week_{week}_features.parquet"
    current_manifest_path = prop / f"data/current/features/{season}_week_{week}_feature_manifest.json"
    log_path = prop / f"logs/current_features_{season}_week_{week}.json"

    for path in [
        universe_path, roles_path, historical_path, historical_manifest_path,
        position_allowed_path, weather_path, travel_path,
    ]:
        if not path.is_file():
            raise FileNotFoundError(f"Issue 31 required input missing: {path}")

    historical_manifest = read_json(historical_manifest_path)
    leading = list(historical_manifest["leading_columns"])
    families = historical_manifest["column_families"]
    feature_columns = list(historical_manifest["feature_columns"])
    target_columns = set(historical_manifest.get("target_columns", []))
    audit_columns = set(families.get("audit", []))

    if any(c.startswith("target_") for c in feature_columns):
        raise ValueError("Historical feature manifest itself contains target feature leakage.")
    if any(c in audit_columns for c in feature_columns):
        raise ValueError("Historical feature manifest itself contains audit feature leakage.")
    common.reject_forbidden_feature_columns(feature_columns, config)

    # Output schema is historical leading columns + all canonical model features,
    # with no target or audit families.
    output_columns = leading + [c for c in feature_columns if c not in leading]
    if len(output_columns) != len(set(output_columns)):
        raise ValueError("Issue 31 output schema contains duplicate columns.")

    reference = load_prior_reference(historical_path, prior_season, output_columns)

    universe_all = pd.read_parquet(universe_path)
    roles = pd.read_parquet(roles_path)
    common.require_columns(
        universe_all,
        [
            "season", "week", "game_id", "game_date", "kickoff_timestamp",
            "player_id", "player_name", "team", "opponent", "position",
            "position_group", "home_flag", "depth_rank", "depth_starter_flag",
            "injury_game_status", "eligibility_status",
        ],
        "Issue 29 current universe",
    )
    common.require_columns(
        roles,
        [
            "season", "week", "game_id", "team", "player_id", "player_name",
            "position", "depth_rank", "starter_flag", "primary_qb_flag",
            "primary_kicker_flag", "primary_role_flag", "committee_role_flag",
            "role_confidence", "role_reason",
        ],
        "Issue 30 current roles",
    )
    common.ensure_unique(universe_all, GRAIN, "Issue 29 current universe")
    common.ensure_unique(roles, GRAIN, "Issue 30 current roles")

    universe_all = universe_all.copy()
    roles = roles.copy()
    for frame in [universe_all, roles]:
        frame["season"] = as_num(frame["season"]).astype(int)
        frame["week"] = as_num(frame["week"]).astype(int)
        frame["game_id"] = frame["game_id"].map(norm_game_id)
        frame["player_id"] = frame["player_id"].map(common.normalize_player_id)
        frame["team"] = frame["team"].map(norm_team)
        frame["position"] = frame["position"].map(norm_position)
    universe_all["opponent"] = universe_all["opponent"].map(norm_team)
    universe_all["position_group"] = universe_all["position_group"].map(norm_position)

    base = universe_all.merge(
        roles[[
            "season", "week", "game_id", "team", "player_id",
            "starter_flag", "primary_qb_flag", "primary_kicker_flag",
            "primary_role_flag", "committee_role_flag", "role_confidence", "role_reason",
        ]],
        on=["season", "week", "game_id", "team", "player_id"],
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    if len(base) != len(roles):
        raise ValueError(
            f"Current roles do not map one-to-one to universe: roles={len(roles)} mapped={len(base)}"
        )
    if not base["eligibility_status"].astype(str).str.casefold().eq("eligible").all():
        raise ValueError("Issue 30 roles include an ineligible Issue 29 universe row.")

    out = pd.DataFrame(index=base.index)
    out["season"] = season
    out["season_type"] = "REG"
    out["week"] = week
    out["game_id"] = base["game_id"]
    out["gameday"] = base["game_date"]
    out["kickoff_timestamp"] = base["kickoff_timestamp"]
    out["player_id"] = base["player_id"]
    out["player_name"] = base["player_name"]
    out["team"] = base["team"]
    out["opponent"] = base["opponent"]
    out["position"] = base["position"]
    out["position_group"] = base["position_group"]
    out["home_flag"] = base["home_flag"]

    # Temporary current-state columns used to build role/environment features.
    temp_cols = [
        "depth_rank", "depth_starter_flag", "injury_game_status", "starter_flag",
        "primary_qb_flag", "primary_kicker_flag", "primary_role_flag",
        "committee_role_flag", "role_confidence", "role_reason",
    ]
    # Add current-state helper columns in one block to avoid DataFrame
    # fragmentation in this very wide feature table.
    out = pd.concat(
        [out, base[temp_cols].reset_index(drop=True)],
        axis=1,
    )

    family_columns: dict[str, list[str]] = {
        name: list(families.get(name, []))
        for name in ["role", "player", "team", "opponent", "matchup", "environment", "history"]
    }
    player_prior_columns = (
        family_columns["role"] + family_columns["player"] + family_columns["history"]
    )
    out = overlay_player_priors(out, reference, player_prior_columns)
    out = overlay_context_priors(out, reference, family_columns)

    # Team-share history resets on a franchise change. Career/non-share
    # efficiency history survives the change. The missing share block is then
    # filled from prior-season position priors below, matching Issue 12 policy.
    prior_team = out["_prior_player_team"].map(norm_team)
    changed_team = prior_team.ne("") & prior_team.ne(out["team"])
    share_prefixes = (
        "player_carry_share_",
        "player_target_share_",
        "player_air_yards_share_",
        "player_red_zone_target_share_",
    )
    share_columns = [
        c for c in player_prior_columns
        if c.startswith(share_prefixes)
    ]
    if share_columns and changed_team.any():
        out.loc[changed_team, share_columns] = np.nan

    # Non-position matchup columns start from no stale player-game value; the
    # deterministic formulas below rebuild what can be derived from current
    # team/opponent/player priors.
    missing_matchup = [
        c for c in family_columns["matchup"]
        if c not in out.columns
    ]
    if missing_matchup:
        out = pd.concat(
            [
                out,
                pd.DataFrame(
                    np.nan,
                    index=out.index,
                    columns=missing_matchup,
                ),
            ],
            axis=1,
        )

    position_allowed = pd.read_parquet(position_allowed_path)
    common.reject_forbidden_feature_columns(position_allowed.columns, config)
    out = overlay_position_allowed_final_week(
        out,
        position_allowed,
        prior_season,
        family_columns["matchup"],
    )

    # Position/league fallback is only for genuinely missing historical player
    # context (rookies/new IDs). It does not replace valid established-player NaN.
    fallback_by_position(out, reference, player_prior_columns)

    # Week 1 must not reuse prior-season season_to_date as if it were current
    # season. Team/player/opponent season-to-date values reset before games.
    if week == 1:
        season_to_date_columns = [
            c for c in feature_columns
            if c.endswith("_season_to_date") and c in out.columns
        ]
        if season_to_date_columns:
            out.loc[:, season_to_date_columns] = np.nan

    current_role_overrides(out, universe_all, roles)

    # The historical player/team/opponent prior overlays intentionally do not
    # carry target-game environment forward. Materialize the canonical
    # environment family as an empty schema block first, then populate it from
    # the required current weather/travel inputs by game_id.
    environment_columns = list(family_columns.get("environment", []))
    missing_environment = [c for c in environment_columns if c not in out.columns]
    if missing_environment:
        environment_block: dict[str, pd.Series] = {}
        for column in missing_environment:
            dtype = reference[column].dtype
            if (
                pd.api.types.is_numeric_dtype(dtype)
                or pd.api.types.is_bool_dtype(dtype)
            ):
                environment_block[column] = pd.Series(
                    np.nan, index=out.index, dtype="float64"
                )
            else:
                environment_block[column] = pd.Series(
                    "", index=out.index, dtype="object"
                )
        out = pd.concat(
            [out, pd.DataFrame(environment_block, index=out.index)],
            axis=1,
        )

    weather = pd.read_csv(weather_path, low_memory=False)
    travel = pd.read_csv(travel_path, low_memory=False)
    common.reject_forbidden_feature_columns(weather.columns, config)
    common.reject_forbidden_feature_columns(travel.columns, config)
    out, env_audit = overlay_environment(out, weather, travel, reference)

    current_frames, current_source_audit = current_realized_sources(
        repo, season, week, config
    )
    player_overlay = overlay_current_player_stats(out, current_frames["player_stats"], week)
    snap_overlay = overlay_current_snaps(out, current_frames["snap_counts"], week)
    team_overlay = overlay_current_team_stats(out, current_frames["team_stats"], week)

    recompute_matchups(out)
    context_helpers = [
        c for c in [
            "_opp_offensive_plays_roll3", "_opp_offense_dropbacks_roll3",
            "_team_defensive_plays_roll3", "_team_defense_dropbacks_roll3",
        ]
        if c in out.columns
    ]
    if context_helpers:
        out = out.drop(columns=context_helpers)

    # Ensure every historical model feature exists before final ordering.
    missing_features = [c for c in feature_columns if c not in out.columns]
    if missing_features:
        raise ValueError(f"Missing canonical current feature columns: {missing_features[:50]}")

    # Strip builder-only current state and enforce exact output schema.
    out = out[output_columns].copy()
    if any(c.startswith("target_") for c in out.columns):
        raise ValueError("Current features contain target column(s).")
    if any(c.startswith("audit_") for c in out.columns):
        raise ValueError("Current features contain audit column(s).")
    common.reject_forbidden_feature_columns(out.columns, config)

    # Match historical dtypes exactly using the prior-season canonical table.
    out = cast_like_reference(out, reference, output_columns)
    dtype_mismatches = {
        c: {"historical": str(reference[c].dtype), "current": str(out[c].dtype)}
        for c in output_columns
        if str(reference[c].dtype) != str(out[c].dtype)
    }
    if dtype_mismatches:
        raise ValueError(f"Current feature dtype mismatch: {dict(list(dtype_mismatches.items())[:20])}")

    common.ensure_unique(out, GRAIN, "Issue 31 current feature table")
    if len(out) != len(roles):
        raise ValueError(f"Current feature row count changed: roles={len(roles)} features={len(out)}")

    numeric_block = out[[c for c in feature_columns if pd.api.types.is_numeric_dtype(out[c].dtype)]]
    if not numeric_block.empty and np.isinf(numeric_block.to_numpy(dtype="float64", copy=False)).any():
        raise ValueError("Current features contain infinity.")

    selected_specs, proxy_checks = selected_manifest_specs(repo, set(out.columns))
    model_checks = validate_model_slices(out, selected_specs, config, repo)

    # Current output schema hash (not equal to the historical full-table hash
    # because target/audit families are intentionally absent).
    schema_text = "\n".join(f"{c}:{out[c].dtype}" for c in out.columns)
    current_schema_hash = hashlib.sha256(schema_text.encode("utf-8")).hexdigest()

    common.write_parquet_atomic(out, output_path)

    source_records = {
        "universe": str(universe_path.relative_to(repo)),
        "roles": str(roles_path.relative_to(repo)),
        "historical_features": str(historical_path.relative_to(repo)),
        "position_allowed": str(position_allowed_path.relative_to(repo)),
        "weather": str(weather_path.relative_to(repo)),
        "travel": str(travel_path.relative_to(repo)),
    }

    manifest_payload = {
        "schema_version": 1,
        "season": season,
        "week": week,
        "canonical_grain": GRAIN,
        "leading_columns": leading,
        "feature_columns": feature_columns,
        "output_columns": output_columns,
        "feature_count": len(feature_columns),
        "column_count": len(output_columns),
        "row_count": len(out),
        "historical_schema_hash": historical_manifest.get("schema_hash"),
        "current_schema_hash_without_targets_audit": current_schema_hash,
        "historical_dtype_match": True,
        "target_columns_present": [],
        "audit_columns_present": [],
        "market_features_used": False,
        "week1_prior_policy": "prior-season rolling/career/team/opponent/position priors; current-season season_to_date reset",
        "strict_current_source_week_filter": "source week < projection week",
        "environment_join_contract": env_audit,
        "current_realized_sources": current_source_audit,
        "selected_model_schema_checks": model_checks,
        "selected_proxy_feature_checks": proxy_checks,
        "source_paths": source_records,
    }
    write_json_atomic(current_manifest_path, manifest_payload)

    log_payload = {
        "script": Path(__file__).name,
        "status": "passed",
        "season": season,
        "week": week,
        "rows": int(len(out)),
        "games": int(out["game_id"].nunique()),
        "teams": int(out["team"].nunique()),
        "players": int(out["player_id"].nunique()),
        "features": int(len(feature_columns)),
        "columns": int(len(output_columns)),
        "prior_season": prior_season,
        "week1_prior_used": week == 1,
        "current_season_completed_weeks_allowed": list(range(1, week)),
        "same_or_future_current_week_rows_allowed": False,
        "weather_join_key": "game_id",
        "travel_join_key": "game_id",
        "no_target_columns": True,
        "no_forbidden_columns": True,
        "historical_dtype_match": True,
        "selected_model_manifest_count": len(model_checks),
        "selected_model_schema_hashes_valid": all(c["stored_hash_matches"] for c in model_checks),
        "market_exclusion_passed": market_audit.get("passed") is True,
        "current_source_audit": current_source_audit,
        "current_overlays": {
            "player_stats": player_overlay,
            "snap_counts": snap_overlay,
            "team_stats": team_overlay,
        },
        "environment_audit": env_audit,
        "output": str(output_path.relative_to(repo)),
        "manifest": str(current_manifest_path.relative_to(repo)),
    }
    write_json_atomic(log_path, log_payload)
    common.log_run("build_current_features.py", log_payload)

    print(json.dumps({"script": Path(__file__).name, "payload": log_payload}, sort_keys=True, default=str))
    print("CURRENT FEATURES BUILD: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
