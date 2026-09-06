#!/usr/bin/env python3
"""Independent acceptance validator for Issue 40 source-quality monitoring."""
from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SCRIPTS_ROOT = HERE / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
import common

OUTPUT_COLUMNS = [
    "run_date", "season", "week", "source", "expected_rows", "actual_rows",
    "missing_player_id_pct", "duplicate_key_count", "missing_team_pct",
    "missing_game_id_pct", "latest_source_week", "freshness_status",
    "quality_status", "notes",
]
SOURCES = [
    "player_stats", "weekly_roster", "current_espn_roster", "depth_charts",
    "injuries", "snap_counts", "participation", "pbp", "team_stats",
    "schedule", "weather", "travel",
]
LAGGED = {"player_stats", "snap_counts", "participation", "pbp", "team_stats"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, default=None)
    p.add_argument("--week", type=int, required=True)
    return p.parse_args()


def static_contract(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assignments = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"OUTPUT_COLUMNS", "SOURCES"}:
                    assignments[target.id] = ast.literal_eval(node.value)
    if assignments.get("OUTPUT_COLUMNS") != OUTPUT_COLUMNS:
        raise AssertionError("Issue 40 OUTPUT_COLUMNS contract mismatch")
    if assignments.get("SOURCES") != SOURCES:
        raise AssertionError("Issue 40 required source registry mismatch")
    for marker in ["source_quality.csv", "not_required_week1", "duplicate_key_count", "run_market_preflight"]:
        if marker not in source:
            raise AssertionError(f"Issue 40 builder missing policy marker: {marker}")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise AssertionError(f"Missing Issue 40 output: {path}")
    df = pd.read_csv(path, low_memory=False)
    if list(df.columns) != OUTPUT_COLUMNS:
        raise AssertionError("source_quality.csv exact header/order mismatch")
    return df


def current_rows(df: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    date = datetime.now(timezone.utc).date().isoformat()
    out = df.loc[
        df["run_date"].astype(str).eq(date)
        & pd.to_numeric(df["season"], errors="coerce").eq(season)
        & pd.to_numeric(df["week"], errors="coerce").eq(week)
    ].copy()
    if len(out) != len(SOURCES):
        raise AssertionError(f"Expected exactly 12 current-run source rows; found {len(out)}")
    if set(out["source"].astype(str)) != set(SOURCES):
        raise AssertionError("Current-run source set mismatch")
    if out["source"].duplicated().any():
        raise AssertionError("Duplicate source row in current run")
    return out


def main() -> int:
    args = parse_args(); config = common.load_config()
    season = int(args.season if args.season is not None else config["seasons"]["current"]); week = int(args.week)
    prop = common.prop_root(); repo = common.repo_root()
    builder = prop / "scripts" / "validate" / "validate_source_quality.py"
    output = prop / "evaluation" / "source_quality.csv"

    print("CHECK 01: required builder, output, exact headers, and static source registry")
    if not builder.is_file(): raise AssertionError(f"Missing Issue 40 builder: {builder}")
    static_contract(builder); df = read_csv(output); cur = current_rows(df, season, week)

    print("CHECK 02: numeric quality fields, statuses, and one row per required source")
    for c in ["expected_rows", "actual_rows", "duplicate_key_count"]:
        x = pd.to_numeric(cur[c], errors="coerce")
        if x.isna().any() or x.lt(0).any(): raise AssertionError(f"Invalid nonnegative numeric field: {c}")
    for c in ["missing_player_id_pct", "missing_team_pct", "missing_game_id_pct"]:
        x = pd.to_numeric(cur[c], errors="coerce")
        finite = x.dropna()
        if ((finite < 0) | (finite > 100)).any(): raise AssertionError(f"Percentage outside [0,100]: {c}")
    if not cur["quality_status"].astype(str).isin({"pass", "warn", "fail"}).all():
        raise AssertionError("Unexpected quality_status")
    if cur["freshness_status"].astype(str).str.strip().eq("").any():
        raise AssertionError("Blank freshness_status")
    if cur["notes"].astype(str).str.strip().eq("").any(): raise AssertionError("Blank source-quality notes")

    print("CHECK 03: Week-1 lagged-source freshness contract and snapshot availability")
    if week == 1:
        lag = cur.loc[cur["source"].isin(LAGGED)]
        if not lag["freshness_status"].astype(str).eq("not_required_week1").all():
            raise AssertionError("Week 1 lagged sources must be explicitly not_required_week1")
    snap = cur.loc[~cur["source"].isin(LAGGED)]
    if snap["freshness_status"].astype(str).eq("missing").any():
        bad = snap.loc[snap["freshness_status"].astype(str).eq("missing"), "source"].tolist()
        raise AssertionError(f"Required current snapshot source missing: {bad}")

    print("CHECK 04: deterministic schedule/weather/travel expectations agree with current universe")
    universe = prop / "data" / "current" / f"{season}_week_{week}_universe.parquet"
    if not universe.is_file(): raise AssertionError("Current universe required for source-quality acceptance")
    u = pd.read_parquet(universe, columns=["game_id", "team"])
    games = int(u["game_id"].astype(str).nunique())
    checks = {"schedule": games, "weather": games, "travel": games}
    for source, expected in checks.items():
        row = cur.loc[cur["source"].eq(source)].iloc[0]
        if int(row["expected_rows"]) != expected:
            raise AssertionError(f"{source}: expected_rows != current universe game count")
        if int(row["actual_rows"]) != expected:
            raise AssertionError(f"{source}: actual_rows != scheduled game count")

    print("CHECK 05: no duplicate source keys and no failed current source rows")
    if pd.to_numeric(cur["duplicate_key_count"], errors="raise").gt(0).any():
        bad = cur.loc[pd.to_numeric(cur["duplicate_key_count"]).gt(0), ["source", "duplicate_key_count"]].to_dict("records")
        raise AssertionError(f"Duplicate source keys detected: {bad}")
    failed = cur.loc[cur["quality_status"].astype(str).eq("fail"), "source"].tolist()
    if failed: raise AssertionError(f"Current source-quality run contains failed source rows: {failed}")

    print("CHECK 06: market-exclusion preflight independently passes")
    audit = prop / "scripts" / "validate" / "audit_market_exclusion.py"
    cp = subprocess.run([sys.executable, str(audit)], cwd=repo, capture_output=True, text=True, check=False)
    if cp.returncode != 0 or "MARKET EXCLUSION AUDIT: PASS" not in cp.stdout:
        raise AssertionError("Independent market-exclusion audit failed")

    counts = cur["quality_status"].value_counts().to_dict()
    print(f"season={season}")
    print(f"week={week}")
    print(f"sources={len(cur)}")
    print(f"pass_rows={int(counts.get('pass',0))}")
    print(f"warn_rows={int(counts.get('warn',0))}")
    print(f"fail_rows={int(counts.get('fail',0))}")
    print(f"scheduled_games={games}")
    print("market_features_used=false")
    print("ISSUE 40 ACCEPTANCE: PASS")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
