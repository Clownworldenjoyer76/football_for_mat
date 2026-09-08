#!/usr/bin/env python3
"""Issue 40: source-quality monitoring for current-week NFL Prop Engine inputs."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
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
DETERMINISTIC_ROWS = {"schedule", "weather", "travel", "team_stats"}
PLAYER_ID_ALIASES = ["player_id", "gsis_id", "nflverse_player_id", "id", "espn_id", "pfr_player_id", "pfr_id"]
TEAM_ALIASES = ["team", "recent_team", "club_code", "team_abbr", "team_id", "posteam", "defteam"]
GAME_ID_ALIASES = ["game_id", "nflverse_game_id", "gameId", "event_id"]
WEEK_ALIASES = ["week", "week_number"]
SEASON_ALIASES = ["season", "season_year"]
PLAY_ID_ALIASES = ["play_id", "playId"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate current-week source quality.")
    p.add_argument("--season", type=int, default=None)
    p.add_argument("--week", type=int, required=True)
    return p.parse_args()


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.casefold() in {"", "nan", "none", "null", "<na>", "nat"} else text


def first_col(df: pd.DataFrame, aliases: list[str]) -> str | None:
    for c in aliases:
        if c in df.columns:
            return c
    lower = {str(c).casefold(): str(c) for c in df.columns}
    for c in aliases:
        if c.casefold() in lower:
            return lower[c.casefold()]
    return None


def best_col(df: pd.DataFrame, aliases: list[str]) -> str | None:
    candidates = [c for c in aliases if c in df.columns]
    if not candidates:
        lower = {str(c).casefold(): str(c) for c in df.columns}
        candidates = [lower[c.casefold()] for c in aliases if c.casefold() in lower]
    if not candidates:
        return None
    return max(candidates, key=lambda c: int(df[c].map(clean).ne("").sum()))

def pct_missing(df: pd.DataFrame, column: str | None) -> float | None:
    if column is None or len(df) == 0:
        return None
    return float(100.0 * df[column].map(clean).eq("").mean())


def read_tabular(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        return pd.read_parquet(path)
    if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        # Current-season realized feeds can legitimately be materialized as
        # zero-byte/headerless placeholders before Week 1. Treat those as an
        # empty monitored source rather than aborting the entire quality run.
        try:
            if path.stat().st_size == 0:
                return pd.DataFrame()
            return pd.read_csv(path, low_memory=False)
        except EmptyDataError:
            return pd.DataFrame()
    raise ValueError(f"Unsupported source format: {path}")


def source_paths(repo: Path, prop: Path, config: dict[str, Any], season: int, week: int) -> dict[str, list[Path]]:
    current_source = prop / "data" / "current" / "source"
    depth_root = (repo / config["paths"]["current_depth_root"]).resolve()
    depth_files = sorted(depth_root.glob("*/*_depth.csv")) if depth_root.is_dir() else []
    return {
        "player_stats": [current_source / f"stats_player_week_{season}.parquet"],
        "weekly_roster": [current_source / f"roster_weekly_{season}.parquet"],
        "current_espn_roster": [(repo / config["paths"]["current_roster"]).resolve()],
        "depth_charts": depth_files,
        "injuries": [(repo / str(config["paths"]["current_injuries"]).format(season=season, week=week)).resolve()],
        "snap_counts": [current_source / f"snap_counts_{season}.parquet"],
        "participation": [current_source / f"pbp_participation_{season}.parquet"],
        "pbp": [(repo / str(config["paths"]["pbp_pattern"]).format(season=season, week=week)).resolve()],
        "team_stats": [(repo / str(config["paths"]["team_stats_pattern"]).format(season=season, week=week)).resolve()],
        "schedule": [(repo / str(config["paths"]["current_schedule"]).format(season=season, week=week)).resolve()],
        "weather": [(repo / str(config["paths"]["current_weather"]).format(season=season, week=week)).resolve()],
        "travel": [(repo / str(config["paths"]["current_travel"]).format(season=season, week=week)).resolve()],
    }


def load_source(source: str, paths: list[Path]) -> tuple[pd.DataFrame, list[Path]]:
    existing = [p for p in paths if p.is_file()]
    if not existing:
        return pd.DataFrame(), []
    frames: list[pd.DataFrame] = []
    for p in existing:
        frame = read_tabular(p)
        if source == "depth_charts" and first_col(frame, TEAM_ALIASES) is None:
            frame = frame.copy()
            frame["_monitor_team"] = p.parent.name.strip().upper()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False), existing


def nums(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def filter_season(df: pd.DataFrame, season: int) -> pd.DataFrame:
    c = first_col(df, SEASON_ALIASES)
    if c is None:
        return df
    v = nums(df[c])
    return df.loc[v.eq(season)].copy() if v.notna().any() else df


def week_col(df: pd.DataFrame) -> str | None:
    return first_col(df, WEEK_ALIASES)


def relevant_slice(source: str, df: pd.DataFrame, season: int, week: int) -> tuple[pd.DataFrame, int | None]:
    if df.empty:
        return df, None
    df = filter_season(df, season)
    wc = week_col(df)
    if wc is None:
        return df, None
    wv = nums(df[wc])
    finite = wv.dropna()
    latest = int(finite.max()) if not finite.empty else None
    if source in LAGGED:
        return (df.iloc[0:0].copy(), latest) if week <= 1 else (df.loc[wv.eq(week - 1)].copy(), latest)
    return df.loc[wv.eq(week)].copy(), latest


def universe_counts(prop: Path, season: int, week: int) -> tuple[int, int, int]:
    p = prop / "data" / "current" / f"{season}_week_{week}_universe.parquet"
    if not p.is_file():
        return 0, 0, 0
    u = pd.read_parquet(p, columns=["game_id", "team", "player_id"])
    return int(u["game_id"].astype(str).nunique()), int(u["team"].astype(str).nunique()), int(len(u))


# WEEKLY_ROSTER_PRODUCTION_UNIVERSE_ID_GATE
# Raw nflverse weekly rosters can contain developmental/unresolved backup rows
# with no GSIS ID. Production identity is enforced separately by the current
# universe builder. Raw weekly-roster ID incompleteness is nonblocking only
# when the current production universe exists and every universe player_id is
# canonical/nonblank.
def production_universe_identity_status(
    prop: Path,
    season: int,
    week: int,
) -> tuple[int, int]:
    p = prop / "data" / "current" / f"{season}_week_{week}_universe.parquet"
    if not p.is_file():
        return 0, 0
    u = pd.read_parquet(p, columns=["player_id"])
    missing = int(u["player_id"].map(clean).eq("").sum())
    return int(len(u)), missing


def prior_schedule_team_games(schedule_df: pd.DataFrame, season: int, week: int) -> int:
    if schedule_df.empty or week <= 1:
        return 0
    sc, wc, gid = first_col(schedule_df, SEASON_ALIASES), week_col(schedule_df), first_col(schedule_df, GAME_ID_ALIASES)
    if wc is None:
        return 0
    mask = nums(schedule_df[wc]).eq(week - 1)
    if sc is not None:
        mask &= nums(schedule_df[sc]).eq(season)
    rows = schedule_df.loc[mask]
    return 2 * (int(rows[gid].astype(str).nunique()) if gid else int(len(rows)))


def trailing_expected(df: pd.DataFrame, week: int) -> int:
    wc = week_col(df)
    if wc is None or df.empty:
        return int(len(df))
    w = nums(df[wc])
    mask = w.lt(max(week, 1))
    counts = df.loc[mask].groupby(w.loc[mask]).size()
    counts = counts[counts > 0].tail(4)
    return int(round(float(counts.median()))) if len(counts) else 0


def duplicate_key_count(source: str, df: pd.DataFrame) -> tuple[int, str]:
    if df.empty:
        return 0, "empty"
    pid, team, gid = best_col(df, PLAYER_ID_ALIASES), best_col(df, TEAM_ALIASES), best_col(df, GAME_ID_ALIASES)
    wc, sc, play = week_col(df), first_col(df, SEASON_ALIASES), first_col(df, PLAY_ID_ALIASES)
    keys: list[str] = []
    if source in {"player_stats", "weekly_roster", "injuries", "snap_counts"} and pid:
        keys = [c for c in (sc, wc, gid, pid) if c]
    elif source == "current_espn_roster" and pid:
        keys = [c for c in (team, pid) if c]
    elif source == "depth_charts" and pid:
        t = team or ("_monitor_team" if "_monitor_team" in df.columns else None)
        pos = first_col(df, ["position_abb", "position", "position_abbreviation"])
        keys = [c for c in (t, pos, pid) if c]
    elif source in {"participation", "pbp"}:
        keys = [c for c in (gid, play) if c]
    elif source == "team_stats":
        keys = [c for c in (sc, wc, team) if c]
    elif source in {"schedule", "weather", "travel"}:
        keys = [c for c in (gid,) if c]
    if not keys:
        return 0, "not_applicable"
    usable = df.copy()
    for c in keys:
        usable = usable.loc[usable[c].map(clean).ne("")]
    return int(usable.duplicated(keys, keep=False).sum()), "+".join(keys)


def expected_rows_for(source: str, raw: pd.DataFrame, relevant: pd.DataFrame, week: int,
                      games: int, teams: int, universe_rows: int,
                      schedule_raw: pd.DataFrame, season: int) -> tuple[int, str]:
    if source == "schedule": return games, "current-universe unique game count"
    if source == "weather": return games, "one weather row per scheduled game"
    if source == "travel": return games, "one travel row per scheduled game"
    if source == "team_stats": return prior_schedule_team_games(schedule_raw, season, week), "prior-week scheduled team-games"
    if source in LAGGED and week <= 1: return 0, "Week 1 has no required current-season realized rows"
    if source in {"player_stats", "snap_counts", "participation", "pbp"}: return trailing_expected(raw, week), "trailing median weekly source rows"
    if source in {"weekly_roster", "current_espn_roster"}: return universe_rows, "current universe player rows"
    if source == "depth_charts": return int(len(relevant)), "snapshot bootstrap; depth row volume is source-defined"
    if source == "injuries": return int(len(relevant)), "sparse event feed; no fixed row expectation"
    return int(len(relevant)), "snapshot bootstrap"


def freshness(source: str, raw: pd.DataFrame, exists: bool, latest: int | None, week: int) -> tuple[str, bool]:
    if source in LAGGED:
        if week <= 1: return "not_required_week1", True
        if not exists: return "missing", False
        if latest is None: return "unknown", False
        return ("current", True) if latest >= week - 1 else ("stale", False)
    if not exists: return "missing", False
    if week_col(raw) is None: return "current_snapshot", True
    if latest is None: return "unknown", False
    return ("current", True) if latest >= week else ("stale", False)


def run_market_preflight() -> bool:
    path = SCRIPTS_ROOT / "validate" / "audit_market_exclusion.py"
    if not path.is_file():
        return False
    cp = subprocess.run([sys.executable, str(path)], cwd=common.repo_root(), capture_output=True, text=True, check=False)
    return cp.returncode == 0 and "MARKET EXCLUSION AUDIT: PASS" in cp.stdout


def write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    common.write_csv_atomic(df, path)


def main() -> int:
    args = parse_args(); config = common.load_config()
    season = int(args.season if args.season is not None else config["seasons"]["current"]); week = int(args.week)
    repo, prop = common.repo_root().resolve(), common.prop_root().resolve()
    run_date = datetime.now(timezone.utc).date().isoformat(); market_ok = run_market_preflight()
    paths = source_paths(repo, prop, config, season, week)
    games, teams, universe_rows = universe_counts(prop, season, week)
    universe_identity_rows, universe_missing_player_ids = (
        production_universe_identity_status(prop, season, week)
    )
    schedule_raw, _ = load_source("schedule", paths["schedule"])
    rows: list[dict[str, Any]] = []
    for source in SOURCES:
        raw, existing_paths = load_source(source, paths[source]); relevant, latest = relevant_slice(source, raw, season, week)
        expected, basis = expected_rows_for(source, raw, relevant, week, games, teams, universe_rows, schedule_raw, season)
        pid = best_col(relevant, PLAYER_ID_ALIASES); team = best_col(relevant, TEAM_ALIASES)
        if source == "depth_charts" and team is None and "_monitor_team" in relevant.columns: team = "_monitor_team"
        gid = best_col(relevant, GAME_ID_ALIASES); dupes, key_note = duplicate_key_count(source, relevant)
        fresh_status, fresh_ok = freshness(source, raw, bool(existing_paths), latest, week)
        pid_pct, team_pct, gid_pct = pct_missing(relevant, pid), pct_missing(relevant, team), pct_missing(relevant, gid)
        quality, reasons = "pass", []
        if not fresh_ok: quality = "fail"; reasons.append(f"freshness={fresh_status}")
        if dupes > 0: quality = "fail"; reasons.append(f"duplicate_key_rows={dupes}")
        for label, pct in (("player_id", pid_pct), ("team", team_pct), ("game_id", gid_pct)):
            if pct is None or pct <= 0:
                continue
            if (
                source == "weekly_roster"
                and label == "player_id"
                and universe_identity_rows > 0
                and universe_missing_player_ids == 0
            ):
                reasons.append(
                    f"raw_missing_player_id_pct={pct:.4f}_nonblocking;"
                    f"production_universe_player_ids_complete=true;"
                    f"production_universe_rows={universe_identity_rows}"
                )
                continue
            quality = "fail"
            reasons.append(f"missing_{label}_pct={pct:.4f}")
        actual = int(len(relevant))
        if source in DETERMINISTIC_ROWS and expected > 0:
            ratio = actual / expected
            if ratio < 0.5: quality = "fail"; reasons.append(f"row_ratio={ratio:.3f}")
            elif ratio < 0.75 and quality != "fail": quality = "warn"; reasons.append(f"row_ratio={ratio:.3f}")
        if not market_ok: quality = "fail"; reasons.append("market_exclusion_preflight_failed")
        if not existing_paths and source in LAGGED and week <= 1:
            quality = "pass" if market_ok else "fail"; reasons.append("source absence allowed before Week 1")
        rows.append({
            "run_date": run_date, "season": season, "week": week, "source": source,
            "expected_rows": int(expected), "actual_rows": actual,
            "missing_player_id_pct": pid_pct, "duplicate_key_count": dupes,
            "missing_team_pct": team_pct, "missing_game_id_pct": gid_pct,
            "latest_source_week": latest, "freshness_status": fresh_status,
            "quality_status": quality,
            "notes": "; ".join([f"paths={len(existing_paths)}/{len(paths[source])}", f"expected_basis={basis}", f"duplicate_key={key_note}", *(reasons or ["no quality exceptions"])])
        })
    current = pd.DataFrame(rows, columns=OUTPUT_COLUMNS); out = prop / "evaluation" / "source_quality.csv"
    if out.is_file():
        old = pd.read_csv(out, low_memory=False)
        if list(old.columns) != OUTPUT_COLUMNS: raise ValueError("Existing source_quality.csv header contract mismatch")
        keep = ~(old["run_date"].astype(str).eq(run_date) & pd.to_numeric(old["season"], errors="coerce").eq(season) & pd.to_numeric(old["week"], errors="coerce").eq(week) & old["source"].astype(str).isin(SOURCES))
        current = pd.concat([old.loc[keep], current], ignore_index=True)
    current = current.sort_values(["run_date", "season", "week", "source"], kind="mergesort").reset_index(drop=True); write_csv_atomic(current, out)
    latest_run = current.loc[current["run_date"].astype(str).eq(run_date) & pd.to_numeric(current["season"], errors="coerce").eq(season) & pd.to_numeric(current["week"], errors="coerce").eq(week)]
    counts = latest_run["quality_status"].value_counts().to_dict(); failed = bool(latest_run["quality_status"].eq("fail").any())
    payload = {"script": Path(__file__).name, "season": season, "week": week, "run_date": run_date, "sources": int(len(latest_run)), "quality_status_counts": {str(k): int(v) for k,v in counts.items()}, "market_exclusion_passed": bool(market_ok), "output": str(out.relative_to(repo)), "status": "failed" if failed else "passed"}
    print(json.dumps({"script": Path(__file__).name, "payload": payload}, sort_keys=True))
    print("SOURCE QUALITY VALIDATION: FAIL" if failed else "SOURCE QUALITY VALIDATION: PASS")
    return 1 if failed else 0

if __name__ == "__main__":
    raise SystemExit(main())

