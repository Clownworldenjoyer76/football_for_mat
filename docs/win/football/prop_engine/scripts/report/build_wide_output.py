#!/usr/bin/env python3
"""Build the Issue 37 wide per-player weekly projection report.

READS
  output/{season}/week_{week}_player_projections.csv  (accepted Issue 36 audit output)
  logs/week_projections_{season}_week_{week}.json     (Issue 36 sequence gate)

WRITES
  output/{season}/week_{week}_player_projections_wide.csv
  logs/wide_output_{season}_week_{week}.json

POLICY
  - Reporting-only reshape. No model scoring, blending, calibration, or new rounding.
  - Preserve the full Issue 36 audit player population, including players whose
    target-specific audit values are zero because they were ineligible.
  - Exactly one output row per season/week/game_id/player_id.
  - No sportsbook/market-derived input is permitted.
"""
from __future__ import annotations

import argparse
import json
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


_CONFIG_CONTRACT = common.load_config()
GRAIN = ["season", "week", "game_id", "player_id"]
TARGETS = list(_CONFIG_CONTRACT["targets"].keys())
# SIX_TARGET_PRODUCTION_REGISTRY_MODE

OUTPUT_COLUMNS = [
    "season",
    "week",
    "game_id",
    "player_id",
    "player_name",
    "team",
    "opponent",
    "position",
    "passing_yards",
    "passing_yards_low",
    "passing_yards_high",
    "passing_tds",
    "passing_tds_prob_1plus",
    "rushing_yards",
    "rushing_yards_low",
    "rushing_yards_high",
    "rushing_tds",
    "rushing_tds_prob_1plus",
    "receiving_yards",
    "receiving_yards_low",
    "receiving_yards_high",
    "receiving_tds",
    "receiving_tds_prob_1plus",
    "kicking_points",
    "kicking_points_low",
    "kicking_points_high",
    "tackles",
    "tackles_low",
    "tackles_high",
    "sacks",
    "sacks_prob_1plus",
    "injury_game_status",
    "role_status",
    "generated_at",
]

SOURCE_REQUIRED = [
    *GRAIN,
    "player_name",
    "team",
    "opponent",
    "position",
    "target",
    "projection",
    "low",
    "high",
    "probability_1_plus",
    "injury_game_status",
    "role_status",
    "generated_at",
]

META_COLUMNS = [
    "player_name",
    "team",
    "opponent",
    "position",
    "injury_game_status",
    "role_status",
    "generated_at",
]

TARGET_MAP: dict[str, dict[str, str]] = {
    "passing_yards": {
        "projection": "passing_yards",
        "low": "passing_yards_low",
        "high": "passing_yards_high",
    },
    "passing_tds": {
        "projection": "passing_tds",
        "probability_1_plus": "passing_tds_prob_1plus",
    },
    "rushing_yards": {
        "projection": "rushing_yards",
        "low": "rushing_yards_low",
        "high": "rushing_yards_high",
    },
    "rushing_tds": {
        "projection": "rushing_tds",
        "probability_1_plus": "rushing_tds_prob_1plus",
    },
    "receiving_yards": {
        "projection": "receiving_yards",
        "low": "receiving_yards_low",
        "high": "receiving_yards_high",
    },
    "receiving_tds": {
        "projection": "receiving_tds",
        "probability_1_plus": "receiving_tds_prob_1plus",
    },
    "kicking_points": {
        "projection": "kicking_points",
        "low": "kicking_points_low",
        "high": "kicking_points_high",
    },
    "tackles": {
        "projection": "tackles",
        "low": "tackles_low",
        "high": "tackles_high",
    },
    "sacks": {
        "projection": "sacks",
        "probability_1_plus": "sacks_prob_1plus",
    },
}


def production_target_state(prop: Path) -> tuple[list[str], list[str]]:
    registry = read_json(prop / "models" / "production_registry.json")
    if list(registry.keys()) != TARGETS:
        raise ValueError("Issue 37 production registry target set/order mismatch.")
    approved: list[str] = []
    deferred: list[str] = []
    for target in TARGETS:
        entry = registry[target]
        if not isinstance(entry, dict):
            raise ValueError(f"{target}: invalid registry entry")
        if entry.get("production_approved") is True:
            version = entry.get("version")
            if not isinstance(version, str) or not version.strip():
                raise ValueError(f"{target}: approved registry target has blank version")
            approved.append(target)
        elif entry.get("production_approved") is False and entry.get("version") is None:
            deferred.append(target)
        else:
            raise ValueError(f"{target}: invalid production registry state")
    if not approved:
        raise ValueError("Issue 37 requires at least one approved production target")
    return approved, deferred


def output_columns_for(targets: list[str]) -> list[str]:
    fixed = {
        "season", "week", "game_id", "player_id", "player_name", "team",
        "opponent", "position", "injury_game_status", "role_status", "generated_at",
    }
    mapped = {
        output
        for target in targets
        for output in TARGET_MAP[target].values()
    }
    return [column for column in OUTPUT_COLUMNS if column in fixed or column in mapped]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Issue 37 wide weekly player output.")
    p.add_argument("--season", type=int, default=None)
    p.add_argument("--week", type=int, required=True)
    return p.parse_args()


def repo_relative(path: Path) -> str:
    root = common.repo_root().resolve()
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON does not exist: {path}")
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = path.resolve()
    root = common.prop_root().resolve()
    try:
        path.relative_to(root)
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
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path = path.resolve()
    root = common.prop_root().resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Refusing write outside Prop Engine: {path}") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temp = Path(handle.name)
    handle.close()
    try:
        frame.to_csv(temp, index=False, lineterminator="\n")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def run_market_preflight() -> dict[str, Any]:
    path = SCRIPTS_ROOT / "validate" / "audit_market_exclusion.py"
    if not path.is_file():
        raise FileNotFoundError(f"Issue 28 market validator missing: {path}")
    cp = subprocess.run(
        [sys.executable, str(path)],
        cwd=common.repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    if cp.returncode != 0 or "MARKET EXCLUSION AUDIT: PASS" not in cp.stdout:
        raise RuntimeError(
            "Market-exclusion preflight failed before Issue 37 reporting. "
            f"stdout={cp.stdout[-2000:]!r} stderr={cp.stderr[-2000:]!r}"
        )
    return {"passed": True, "validator": repo_relative(path)}


def normalize_grain(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    out = frame.copy()
    out["season"] = pd.to_numeric(out["season"], errors="raise").astype(int)
    out["week"] = pd.to_numeric(out["week"], errors="raise").astype(int)
    for column in ["game_id", "player_id"]:
        out[column] = out[column].fillna("").astype(str).str.strip()
        if out[column].eq("").any():
            raise ValueError(f"{label}: blank {column}")
    return out


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")


def assert_metadata_invariant(source: pd.DataFrame) -> None:
    grouped = source.groupby(GRAIN, sort=False, dropna=False)
    for column in META_COLUMNS:
        counts = grouped[column].nunique(dropna=False)
        if counts.gt(1).any():
            sample = counts[counts.gt(1)].head(10).index.tolist()
            raise ValueError(
                f"Issue 37 source metadata varies across targets: {column}; sample={sample}"
            )


def build_wide(source: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    assert_metadata_invariant(source)
    output_columns = output_columns_for(targets)

    counts = source.groupby(GRAIN, sort=False, dropna=False)["target"].agg(list)
    bad_count = counts.map(len).ne(len(targets))
    if bad_count.any():
        raise ValueError(
            "Issue 37 requires exactly one row per approved production target per player-game; "
            f"sample={counts[bad_count].head(10).to_dict()}"
        )
    expected_targets = set(targets)
    bad_set = counts.map(lambda values: set(map(str, values)) != expected_targets)
    if bad_set.any():
        raise ValueError(
            "Issue 37 target coverage mismatch per player-game; "
            f"sample={counts[bad_set].head(10).to_dict()}"
        )

    base = (
        source[[*GRAIN, *META_COLUMNS]]
        .drop_duplicates(GRAIN, keep="first")
        .copy()
    )
    if base.duplicated(GRAIN).any():
        raise ValueError("Issue 37 base player-game grain is not unique")

    for target in targets:
        subset = source.loc[source["target"].astype(str).eq(target)].copy()
        if subset.duplicated(GRAIN).any():
            raise ValueError(f"Issue 37 duplicate source row for target={target}")
        use = subset[GRAIN].copy()
        for source_column, output_column in TARGET_MAP[target].items():
            use[output_column] = numeric(subset[source_column]).to_numpy()
        base = base.merge(use, on=GRAIN, how="left", validate="one_to_one", sort=False)

    mapped_numeric = [
        column
        for column in output_columns
        if column not in {
            *GRAIN,
            "player_name",
            "team",
            "opponent",
            "position",
            "injury_game_status",
            "role_status",
            "generated_at",
        }
    ]
    if base[mapped_numeric].isna().any().any():
        missing = base[mapped_numeric].isna().sum()
        missing = missing[missing.gt(0)].to_dict()
        raise ValueError(f"Issue 37 mapped wide values contain nulls: {missing}")

    probability_columns = [
        column
        for column in [
            "passing_tds_prob_1plus",
            "rushing_tds_prob_1plus",
            "receiving_tds_prob_1plus",
            "sacks_prob_1plus",
        ]
        if column in output_columns
    ]
    for column in probability_columns:
        values = numeric(base[column])
        if values.lt(0.0).any() or values.gt(1.0).any():
            raise ValueError(f"Issue 37 probability outside [0,1]: {column}")

    base = base[output_columns].sort_values(
        ["season", "week", "game_id", "team", "position", "player_name", "player_id"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)

    if list(base.columns) != output_columns:
        raise RuntimeError("Issue 37 exact output header/order mismatch")
    if base.duplicated(GRAIN).any():
        raise RuntimeError("Issue 37 output contains duplicate player-game rows")
    return base


def main() -> int:
    args = parse_args()
    config = common.load_config()
    season = int(args.season if args.season is not None else config["seasons"]["current"])
    week = int(args.week)
    if week < 1:
        raise ValueError("week must be >= 1")

    prop = common.prop_root()
    source_path = prop / "output" / str(season) / f"week_{week}_player_projections.csv"
    issue36_log = prop / "logs" / f"week_projections_{season}_week_{week}.json"
    output_path = prop / "output" / str(season) / f"week_{week}_player_projections_wide.csv"
    log_path = prop / "logs" / f"wide_output_{season}_week_{week}.json"

    market = run_market_preflight()
    production_targets, deferred_targets = production_target_state(prop)

    if not source_path.is_file():
        raise FileNotFoundError(f"Issue 37 required Issue 36 long output missing: {source_path}")
    issue36 = read_json(issue36_log)
    if issue36.get("status") != "passed":
        raise ValueError("Issue 37 requires a passed Issue 36 weekly projection log")
    if issue36.get("market_features_used") is not False:
        raise ValueError("Issue 37 sequence gate: Issue 36 market policy violation")

    source = pd.read_csv(
        source_path,
        dtype={"game_id": "string", "player_id": "string"},
        keep_default_na=False,
    )
    missing = [column for column in SOURCE_REQUIRED if column not in source.columns]
    if missing:
        raise ValueError(f"Issue 37 source missing required columns: {missing}")
    source = normalize_grain(source, "Issue 36 long output")
    if set(source["season"].unique()) != {season} or set(source["week"].unique()) != {week}:
        raise ValueError("Issue 37 source season/week mismatch")
    if set(source["target"].astype(str).unique()) != set(production_targets):
        raise ValueError("Issue 37 source target set differs from approved production registry")

    wide = build_wide(source, production_targets)
    write_csv_atomic(wide, output_path)

    generated_values = sorted({str(v).strip() for v in wide["generated_at"].tolist() if str(v).strip()})
    payload = {
        "status": "passed",
        "script": Path(__file__).name,
        "season": season,
        "week": week,
        "players": int(len(wide)),
        "games": int(wide["game_id"].nunique()),
        "teams": int(wide["team"].nunique()),
        "columns": int(len(wide.columns)),
        "source_long_rows": int(len(source)),
        "targets_reshaped": len(production_targets),
        "production_targets": list(production_targets),
        "deferred_targets": list(deferred_targets),
        "additional_rounding_applied": False,
        "probabilities_in_unit_interval": True,
        "market_exclusion_passed": bool(market["passed"]),
        "market_features_used": False,
        "source": repo_relative(source_path),
        "output": repo_relative(output_path),
        "log": repo_relative(log_path),
        "generated_at_values": generated_values,
        "policy": {
            "issue36_audit_output_is_authoritative": True,
            "full_audit_player_population_preserved": True,
            "one_row_per_player_game": True,
            "no_model_scoring": True,
            "no_recalibration": True,
            "no_additional_rounding": True,
            "market_exclusion_preflight": True,
        },
    }
    write_json_atomic(log_path, payload)

    print(json.dumps({"script": Path(__file__).name, "payload": payload}, sort_keys=True, separators=(",", ":")))
    print("WIDE PLAYER PROJECTIONS BUILD: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
