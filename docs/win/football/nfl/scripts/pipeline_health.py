#!/usr/bin/env python3
"""Build the NFL 01 pipeline health snapshot.

NFL-owned inputs are read only from:
    docs/win/football/nfl/

Writes:
    docs/win/football/nfl/pipeline_health.json
    docs/win/football/nfl/errors/pipeline_health.txt
    frontend/data/pipeline_health/nfl.json

The workflow supplies NFL_PIPELINE_WORKFLOW, NFL_PIPELINE_JOB_STATUS, and
NFL_PROJECTION_MODE. Structured PipelineReporter JSON files are considered
current only when their GitHub run ID and attempt match this execution.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
NFL_ROOT = SCRIPT_DIR.parent
REPO_ROOT = NFL_ROOT.parents[3]

ERRORS = NFL_ROOT / "errors"
SETTINGS_PATH = NFL_ROOT / "config" / "settings.yaml"
OUTPUT = NFL_ROOT / "pipeline_health.json"
LOG = ERRORS / "pipeline_health.txt"
FRONTEND_OUTPUT = REPO_ROOT / "frontend" / "data" / "pipeline_health" / "nfl.json"
DASHBOARD_OUTPUT = REPO_ROOT / "frontend" / "nfl_dashboard.html"

NY = ZoneInfo("America/New_York")
WORKFLOW_KEY = "nfl_01"
WORKFLOW_LABEL = "NFL 01 Pipeline"
VALID_JOB_STATUSES = {"success", "failure", "cancelled", "unknown"}
VALID_SEASON_TYPES = {"pre", "reg", "post"}
VALID_REPORT_STATUSES = {"SUCCESS", "WARNING", "FAILED"}
EPSILON = 1e-9

EXPECTED_REPORT_FILES = {
    "clean_drat": ERRORS / "00_intake" / "clean_drat.json",
    "finalize_pred": ERRORS / "00_intake" / "finalize_pred.json",
    "enrich_moneyline": ERRORS / "00_intake" / "enrich_moneyline.json",
    "enrich_spread": ERRORS / "00_intake" / "enrich_spread.json",
    "enrich_totals": ERRORS / "00_intake" / "enrich_totals.json",
    "enrich_combine": ERRORS / "00_intake" / "enrich_combine.json",
    "selections": ERRORS / "02_select" / "selections.json",
    "picks": ERRORS / "03_picks" / "picks.json",
    "all_games_picks": ERRORS / "03_picks" / "all_games_picks.json",
    "final_picks": ERRORS / "03_picks" / "final_picks.json",
    "pull_final_scores": ERRORS / "04_final_results" / "pull_final_scores.json",
    "01_nfl_results_grade": ERRORS / "04_final_results" / "01_nfl_results_grade.json",
    "02_nfl_results_analyze": ERRORS / "04_final_results" / "02_nfl_results_analyze.json",
    "03_nfl_results_reports": ERRORS / "04_final_results" / "03_nfl_results_reports.json",
    "04_nfl_results_dashboard": ERRORS / "04_final_results" / "04_nfl_results_dashboard.json",
    "survivor": ERRORS / "03_picks" / "survivor.json",
    "nmbets": ERRORS / "03_picks" / "nmbets.json",
}

PROJECTION_REPORTS = {
    "week1": ERRORS / "01_merge" / "projection_week1.json",
    "inseason": ERRORS / "01_merge" / "projection.json",
}
REFRESH_REPORT = ERRORS / "00_intake" / "refresh_projection_sources.json"

REPORT_OVERVIEW_FILES = [
    "nfl_report_metric_definitions.csv",
    "nfl_summary_overall.csv",
    "nfl_summary_by_market.csv",
    "nfl_summary_by_side_group.csv",
    "nfl_summary_by_season_type.csv",
    "nfl_summary_by_week.csv",
    "nfl_summary_by_date.csv",
    "nfl_summary_by_day_night.csv",
    "nfl_bet_log.csv",
]

REPORT_DIMENSIONS = {
    "moneyline": (
        "moneyline",
        ["ev", "odds", "kelly", "win_prob", "week"],
    ),
    "spread": (
        "spread",
        ["ev", "odds", "kelly", "win_prob", "spread_range", "line", "week", "side"],
    ),
    "totals": (
        "total",
        ["ev", "odds", "kelly", "win_prob", "total_range", "line", "week", "side"],
    ),
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.casefold() in {"", "nan", "none", "null", "<na>", "nat"}:
        return ""
    return text


def clean_id(value: Any) -> str:
    text = clean(value)
    if not text:
        return ""
    try:
        number = float(text)
        if math.isfinite(number) and number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return text


def parse_positive_int(value: Any, *, label: str) -> int:
    text = clean(value)
    try:
        number = float(text)
    except (TypeError, ValueError):
        fail(f"{label} must be a positive integer; found {value!r}")
    if not math.isfinite(number) or not number.is_integer() or number <= 0:
        fail(f"{label} must be a positive integer; found {value!r}")
    return int(number)


def read_json_object(path: Path, *, required: bool = False) -> dict[str, Any]:
    if not path.is_file():
        if required:
            fail(f"Required JSON file not found: {path}")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"Invalid JSON {path}: {type(exc).__name__}: {exc}")
    if not isinstance(payload, dict):
        fail(f"JSON root must be an object: {path}")
    return payload


def read_settings() -> dict[str, Any]:
    if not SETTINGS_PATH.is_file():
        fail(f"NFL settings file not found: {SETTINGS_PATH}")

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        SETTINGS_PATH.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line or raw_line[0].isspace():
            continue
        stripped = raw_line.split("#", 1)[0].strip()
        if not stripped:
            continue
        match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*?)\s*", stripped)
        if not match:
            continue
        key, value = match.groups()
        value = value.strip().strip("'\"")
        if key in values:
            fail(f"{SETTINGS_PATH}: duplicate top-level key {key!r} at line {line_number}")
        values[key] = value

    missing = [key for key in ("season", "week", "season_type", "sportsbook") if not clean(values.get(key))]
    if missing:
        fail(f"{SETTINGS_PATH}: missing required setting(s): {missing}")

    season = parse_positive_int(values["season"], label="settings season")
    week = parse_positive_int(values["week"], label="settings week")
    season_type = clean(values["season_type"]).lower()
    sportsbook = clean(values["sportsbook"]).lower()

    if season_type not in VALID_SEASON_TYPES:
        fail(f"settings season_type must be one of {sorted(VALID_SEASON_TYPES)}; found {season_type!r}")
    if not sportsbook:
        fail("settings sportsbook must not be blank")

    return {
        "season": season,
        "week": week,
        "season_type": season_type,
        "sportsbook": sportsbook,
    }


def environment_target(settings: dict[str, Any]) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    fatals: list[str] = []

    expected = {
        "NFL_SEASON": str(settings["season"]),
        "NFL_WEEK": str(settings["week"]),
        "NFL_SEASON_TYPE_NAME": str(settings["season_type"]),
        "NFL_SPORTSBOOK": str(settings["sportsbook"]),
    }

    for name, target in expected.items():
        actual = clean(os.getenv(name))
        if not actual:
            warnings.append(f"{name} is not set; settings.yaml value {target!r} is authoritative")
            continue
        if actual.lower() != target.lower():
            fatals.append(
                f"{name}={actual!r} does not match settings.yaml value {target!r}"
            )

    return warnings, fatals


def validate_csv_header(path: Path, required_columns: set[str]) -> list[str]:
    if not path.is_file():
        fail(f"CSV file not found: {path}")
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            header = next(csv.reader(handle), None)
    except Exception as exc:
        fail(f"Unable to read CSV header {path}: {type(exc).__name__}: {exc}")

    if not header:
        fail(f"CSV has no header: {path}")

    normalized = [clean(column) for column in header]
    if any(not column for column in normalized):
        fail(f"CSV contains blank header name: {path}")

    duplicates = sorted(
        {column for column in normalized if normalized.count(column) > 1}
    )
    if duplicates:
        fail(f"CSV contains duplicate header names {duplicates}: {path}")

    missing = sorted(required_columns - set(normalized))
    if missing:
        fail(f"CSV missing required columns {missing}: {path}")

    return normalized


def read_csv_rows(
    path: Path,
    *,
    required_columns: set[str],
    allow_empty: bool,
) -> tuple[list[str], list[dict[str, str]]]:
    header = validate_csv_header(path, required_columns)

    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
    except Exception as exc:
        fail(f"Unable to read CSV {path}: {type(exc).__name__}: {exc}")

    if not rows and not allow_empty:
        fail(f"CSV contains no data rows: {path}")

    for row_number, row in enumerate(rows, start=2):
        if None in row:
            fail(f"CSV row {row_number} has fields beyond the header: {path}")

    return header, rows


def ids_and_integrity(
    rows: list[dict[str, str]],
    *,
    path: Path,
) -> tuple[set[str], int, list[str]]:
    ids: set[str] = set()
    duplicates: set[str] = set()
    blank = 0

    for row_number, row in enumerate(rows, start=2):
        game_id = clean_id(row.get("game_id"))
        if not game_id:
            blank += 1
            continue
        if game_id in ids:
            duplicates.add(game_id)
        ids.add(game_id)

    if blank:
        fail(f"{path}: {blank} blank game_id value(s)")
    if duplicates:
        fail(f"{path}: duplicate game_id value(s): {sorted(duplicates)[:10]}")

    return ids, blank, sorted(duplicates)


def validate_target_rows(
    rows: list[dict[str, str]],
    *,
    path: Path,
    settings: dict[str, Any],
) -> None:
    for row_number, row in enumerate(rows, start=2):
        if "season" in row:
            season = parse_positive_int(
                row["season"],
                label=f"{path} row {row_number} season",
            )
            if season != settings["season"]:
                fail(
                    f"{path} row {row_number}: season={season} does not match configured season={settings['season']}"
                )
        if "week" in row:
            week = parse_positive_int(
                row["week"],
                label=f"{path} row {row_number} week",
            )
            if week != settings["week"]:
                fail(
                    f"{path} row {row_number}: week={week} does not match configured week={settings['week']}"
                )
        if "season_type" in row:
            season_type = clean(row["season_type"]).lower()
            if season_type and season_type != settings["season_type"]:
                fail(
                    f"{path} row {row_number}: season_type={season_type!r} does not match configured season_type={settings['season_type']!r}"
                )


def artifact_specs(settings: dict[str, Any]) -> dict[str, dict[str, Any]]:
    season = settings["season"]
    week = settings["week"]
    season_type = settings["season_type"]

    return {
        "weekly_schedule": {
            "path": NFL_ROOT / "00_intake" / "schedule" / "weekly" / f"week_{week}_NFL_weekly_schedule.csv",
            "required": {"season", "season_type", "week", "game_id", "away_team", "home_team"},
            "allow_empty": False,
            "identity": "schedule_full",
        },
        "projection": {
            "path": NFL_ROOT / "01_merge" / f"week_{week}_NFL_enriched.csv",
            "required": {"season", "season_type", "week", "game_id", "away_team", "home_team"},
            "allow_empty": False,
            "identity": "schedule_subset",
        },
        "selected_candidates": {
            "path": NFL_ROOT / "02_select" / f"week_{week}_NFL_selected.csv",
            "required": {"season", "season_type", "week", "game_id", "away_team", "home_team"},
            "allow_empty": False,
            "identity": "projection_exact",
        },
        "root_picks": {
            "path": NFL_ROOT / "03_picks" / f"week_{week}_NFL_picks.csv",
            "required": {"season", "season_type", "week", "game_id", "away_team", "home_team"},
            "allow_empty": False,
            "identity": "projection_exact",
        },
        "all_games": {
            "path": NFL_ROOT / "03_picks" / "all_games" / f"all_week_{week}_NFL_picks.csv",
            "required": {"season", "week", "game_id", "away_team", "home_team"},
            "allow_empty": False,
            "identity": "projection_exact",
        },
        "selected_bets": {
            "path": NFL_ROOT / "03_picks" / "selected" / f"week_{week}_NFL_select_picks.csv",
            "required": {"season", "season_type", "week", "game_id", "away_team", "home_team"},
            "allow_empty": True,
            "identity": "schedule_subset",
        },
        "projection_picks": {
            "path": NFL_ROOT / "03_picks" / "projection" / f"week_{week}_NFL_projection.csv",
            "required": {"season", "season_type", "week", "game_id", "away_team", "home_team"},
            "allow_empty": False,
            "identity": "projection_exact",
        },
        "survivor": {
            "path": NFL_ROOT / "03_picks" / "survivor" / f"{week}_survivor_picks.csv",
            "required": {"week", "game_id", "pick", "pt_diff", "away_team", "home_team"},
            "allow_empty": False,
            "identity": "projection_exact",
        },
        "nmbets": {
            "path": NFL_ROOT / "03_picks" / "nmbets" / f"week_{week}_NM_NFL_picks.csv",
            "required": {
                "Date", "Time", "Away_Team", "Home_Team",
                "Projected_Score", "Predicted_Margin", "Predicted_Total",
            },
            "allow_empty": False,
            "identity": "root_picks_row_count",
        },
        "final_scores": {
            "path": NFL_ROOT / "04_final_results" / "results" / f"{season}_{season_type}_{week}.csv",
            "required": {
                "season", "season_type", "week", "game_id", "away_team",
                "home_team", "away_score", "home_score", "status",
            },
            "allow_empty": False,
            "identity": "schedule_full",
        },
    }


def collect_artifact_health(
    settings: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[str]]:
    fatals: list[str] = []
    warnings: list[str] = []
    artifacts: dict[str, Any] = {}
    id_sets: dict[str, set[str]] = {}

    for name, spec in artifact_specs(settings).items():
        path = spec["path"]
        try:
            header, rows = read_csv_rows(
                path,
                required_columns=set(spec["required"]),
                allow_empty=bool(spec["allow_empty"]),
            )
            validate_target_rows(rows, path=path, settings=settings)

            item: dict[str, Any] = {
                "path": str(path),
                "exists": True,
                "rows": len(rows),
                "columns": len(header),
                "identity_mode": spec["identity"],
            }

            if spec["identity"] != "root_picks_row_count":
                ids, blank, duplicates = ids_and_integrity(rows, path=path)
                id_sets[name] = ids
                item.update({
                    "unique_game_ids": len(ids),
                    "blank_game_ids": blank,
                    "duplicate_game_ids": duplicates,
                })

            artifacts[name] = item
        except Exception as exc:
            fatals.append(f"{name}: {exc}")
            artifacts[name] = {
                "path": str(path),
                "exists": path.is_file(),
                "error": f"{type(exc).__name__}: {exc}",
                "identity_mode": spec["identity"],
            }

    scheduled = id_sets.get("weekly_schedule")
    projection = id_sets.get("projection")

    if scheduled is not None:
        final_scores = id_sets.get("final_scores")
        if final_scores is not None:
            missing = sorted(scheduled - final_scores)
            extra = sorted(final_scores - scheduled)
            artifacts["final_scores"]["missing_scheduled_game_ids"] = missing
            artifacts["final_scores"]["extra_game_ids"] = extra
            if missing or extra:
                fatals.append(
                    "final_scores: game_id set does not match weekly_schedule "
                    f"(missing={len(missing)} extra={len(extra)})"
                )

        if projection is not None:
            not_projected = sorted(scheduled - projection)
            extra_projection = sorted(projection - scheduled)
            artifacts["projection"]["not_projected_game_ids"] = not_projected
            artifacts["projection"]["extra_game_ids"] = extra_projection
            if extra_projection:
                fatals.append(
                    "projection: "
                    f"{len(extra_projection)} game_id(s) are not in weekly_schedule"
                )

        selected = id_sets.get("selected_bets")
        if selected is not None:
            extra_selected = sorted(selected - scheduled)
            artifacts["selected_bets"]["extra_game_ids"] = extra_selected
            if extra_selected:
                fatals.append(
                    f"selected_bets: {len(extra_selected)} game_id(s) are not in weekly_schedule"
                )

    if projection is not None:
        for name in (
            "selected_candidates",
            "root_picks",
            "all_games",
            "projection_picks",
            "survivor",
        ):
            ids = id_sets.get(name)
            if ids is None:
                continue
            missing = sorted(projection - ids)
            extra = sorted(ids - projection)
            artifacts[name]["missing_projection_game_ids"] = missing
            artifacts[name]["extra_projection_game_ids"] = extra
            if missing or extra:
                fatals.append(
                    f"{name}: game_id set does not match projection "
                    f"(missing={len(missing)} extra={len(extra)})"
                )

    root_picks = artifacts.get("root_picks", {})
    nmbets = artifacts.get("nmbets", {})
    if "rows" in root_picks and "rows" in nmbets:
        expected_nm_rows = int(root_picks["rows"])
        actual_nm_rows = int(nmbets["rows"])
        nmbets["expected_root_pick_rows"] = expected_nm_rows
        if actual_nm_rows != expected_nm_rows:
            fatals.append(
                f"nmbets: rows={actual_nm_rows} "
                f"does not match root_picks_rows={expected_nm_rows}"
            )

    return artifacts, warnings, fatals


def expected_report_paths() -> list[Path]:
    paths = [
        NFL_ROOT / "04_final_results" / "reports" / "overview" / name
        for name in REPORT_OVERVIEW_FILES
    ]

    report_root = NFL_ROOT / "04_final_results" / "reports"
    for directory, (file_key, dimensions) in REPORT_DIMENSIONS.items():
        for dimension in dimensions:
            base = report_root / directory / f"nfl_{file_key}_by_{dimension}"
            paths.append(Path(f"{base}.csv"))
            paths.append(Path(f"{base}_side_summary.csv"))

    return paths


def collect_final_results_health() -> tuple[dict[str, Any], list[str], list[str]]:
    fatals: list[str] = []
    warnings: list[str] = []

    summary = NFL_ROOT / "04_final_results" / "nfl_summary_overall.csv"
    work = NFL_ROOT / "04_final_results" / "intermediate" / "work_nfl.csv"
    expected_reports = expected_report_paths()

    data: dict[str, Any] = {
        "summary": str(summary),
        "summary_exists": summary.is_file(),
        "work_nfl": str(work),
        "work_nfl_exists": work.is_file(),
        "dashboard": str(DASHBOARD_OUTPUT),
        "dashboard_exists": DASHBOARD_OUTPUT.is_file(),
        "expected_report_files": len(expected_reports),
        "present_report_files": sum(path.is_file() for path in expected_reports),
        "missing_report_files": [str(path) for path in expected_reports if not path.is_file()],
    }

    for label, path in (
        ("summary", summary),
        ("work_nfl", work),
    ):
        if not path.is_file():
            fatals.append(f"final_results {label} missing: {path}")
            continue
        try:
            header, rows = read_csv_rows(
                path,
                required_columns=set(),
                allow_empty=(label == "work_nfl"),
            )
            data[f"{label}_rows"] = len(rows)
            data[f"{label}_columns"] = len(header)
        except Exception as exc:
            fatals.append(f"final_results {label}: {exc}")

    if data["missing_report_files"]:
        fatals.append(
            f"final_results reports: {len(data['missing_report_files'])} expected report file(s) missing"
        )

    if not DASHBOARD_OUTPUT.is_file():
        fatals.append(f"NFL dashboard missing: {DASHBOARD_OUTPUT}")
    elif DASHBOARD_OUTPUT.stat().st_size <= 0:
        fatals.append(f"NFL dashboard is empty: {DASHBOARD_OUTPUT}")
    else:
        data["dashboard_bytes"] = DASHBOARD_OUTPUT.stat().st_size

    return data, warnings, fatals


def model_health(settings: dict[str, Any]) -> tuple[dict[str, Any], list[str], list[str]]:
    fatals: list[str] = []
    warnings: list[str] = []

    manifest_path = NFL_ROOT / "models" / "production_model.json"
    manifest = read_json_object(manifest_path, required=True)
    active_model = clean(manifest.get("active_model"))
    active_files = manifest.get("active_model_files")

    if not active_model:
        fatals.append("production_model.json active_model is blank")
    if not isinstance(active_files, dict) or not active_files:
        fatals.append("production_model.json active_model_files is missing or invalid")
        active_files = {}

    files: dict[str, Any] = {}
    for key, relative in active_files.items():
        relative_text = clean(relative)
        path = NFL_ROOT / "models" / relative_text
        exists = bool(relative_text) and path.is_file()
        files[str(key)] = {
            "path": str(path),
            "exists": exists,
            "bytes": path.stat().st_size if exists else None,
        }
        if not exists:
            fatals.append(f"active model file missing for {key}: {path}")

    manifest_season = manifest.get("season")
    if manifest_season is not None:
        try:
            parsed_manifest_season = parse_positive_int(
                manifest_season,
                label="production_model season",
            )
            if parsed_manifest_season > settings["season"]:
                warnings.append(
                    f"production_model season={parsed_manifest_season} is later than configured season={settings['season']}"
                )
        except Exception as exc:
            fatals.append(str(exc))

    return {
        "manifest_path": str(manifest_path),
        "manifest_exists": manifest_path.is_file(),
        "active_model": active_model,
        "manifest_season": manifest_season,
        "active_model_files": files,
    }, warnings, fatals


def expected_structured_reports(projection_mode: str) -> dict[str, Path]:
    reports = dict(EXPECTED_REPORT_FILES)
    projection_path = PROJECTION_REPORTS.get(projection_mode)
    if projection_path is not None:
        reports[f"projection_{projection_mode}"] = projection_path
    if projection_mode == "inseason":
        reports["refresh_projection_sources"] = REFRESH_REPORT
    return reports


def report_is_current(
    payload: dict[str, Any],
    *,
    github_run_id: str,
    github_run_attempt: str,
) -> bool:
    environment = payload.get("environment")
    if not isinstance(environment, dict):
        return False

    if github_run_id and clean(environment.get("github_run_id")) != github_run_id:
        return False
    if github_run_attempt and clean(environment.get("github_run_attempt")) != github_run_attempt:
        return False
    return bool(github_run_id or github_run_attempt)


def report_row(name: str, path: Path, payload: dict[str, Any], current: bool) -> dict[str, Any]:
    status = clean(payload.get("status")).upper() if payload else ""
    return {
        "name": name,
        "path": str(path),
        "exists": path.is_file(),
        "current_run": current,
        "status": status or None,
        "started_at_utc": payload.get("started_at_utc") if payload else None,
        "finished_at_utc": payload.get("finished_at_utc") if payload else None,
        "rows_in": payload.get("rows_in") if payload else None,
        "rows_out": payload.get("rows_out") if payload else None,
        "warning_count": payload.get("warning_count") if payload else None,
        "error_count": payload.get("error_count") if payload else None,
    }


def collect_structured_report_health(
    *,
    projection_mode: str,
    github_run_id: str,
    github_run_attempt: str,
    job_status: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str], list[str]]:
    warnings: list[str] = []
    fatals: list[str] = []
    stage_rows: list[dict[str, Any]] = []
    expected = expected_structured_reports(projection_mode)
    expected_resolved = {path.resolve() for path in expected.values()}

    current_count = 0
    failed_count = 0
    warning_count = 0

    for name, path in expected.items():
        payload: dict[str, Any] = {}
        if path.is_file():
            try:
                payload = read_json_object(path, required=True)
            except Exception as exc:
                fatals.append(f"structured report {name}: {exc}")
        current = bool(payload) and report_is_current(
            payload,
            github_run_id=github_run_id,
            github_run_attempt=github_run_attempt,
        )
        stage_rows.append(report_row(name, path, payload, current))

        if current:
            current_count += 1
            status = clean(payload.get("status")).upper()
            if status not in VALID_REPORT_STATUSES:
                fatals.append(f"structured report {name} has invalid status {status!r}")
            elif status == "FAILED":
                failed_count += 1
                fatals.append(f"structured report {name} status is FAILED")
            elif status == "WARNING":
                warning_count += 1
                warnings.append(f"structured report {name} status is WARNING")
        elif job_status == "success":
            fatals.append(f"structured report {name} is missing for the current GitHub run")

    if ERRORS.is_dir() and github_run_id:
        for path in sorted(ERRORS.rglob("*.json")):
            if path.resolve() in expected_resolved:
                continue
            try:
                payload = read_json_object(path, required=True)
            except Exception:
                continue
            if not report_is_current(
                payload,
                github_run_id=github_run_id,
                github_run_attempt=github_run_attempt,
            ):
                continue

            name = clean(payload.get("script")) or path.stem
            stage_rows.append(report_row(name, path, payload, True))
            current_count += 1

            status = clean(payload.get("status")).upper()
            if status == "FAILED":
                failed_count += 1
                fatals.append(f"current structured report {path} status is FAILED")
            elif status == "WARNING":
                warning_count += 1
                warnings.append(f"current structured report {path} status is WARNING")
            elif status not in VALID_REPORT_STATUSES:
                fatals.append(f"current structured report {path} has invalid status {status!r}")

    summary = {
        "expected_reports": len(expected),
        "current_expected_reports": sum(1 for row in stage_rows[:len(expected)] if row["current_run"]),
        "current_reports_total": current_count,
        "failed_reports": failed_count,
        "warning_reports": warning_count,
        "github_run_id": github_run_id or None,
        "github_run_attempt": github_run_attempt or None,
    }

    if not github_run_id:
        warnings.append("GITHUB_RUN_ID is unavailable; structured reports cannot be tied to the current run")

    return summary, stage_rows, warnings, fatals


def workflow_snapshot(
    *,
    job_status: str,
    projection_mode: str,
    generated_at: str,
    stage_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "workflow": WORKFLOW_KEY,
        "label": WORKFLOW_LABEL,
        "status": job_status,
        "projection_mode": projection_mode,
        "generated_at_utc": generated_at,
        "game_date_new_york": datetime.now(NY).strftime("%Y_%m_%d"),
        "run_id": clean(os.getenv("GITHUB_RUN_ID")),
        "run_attempt": clean(os.getenv("GITHUB_RUN_ATTEMPT")),
        "sha": clean(os.getenv("GITHUB_SHA")),
        "ref_name": clean(os.getenv("GITHUB_REF_NAME")),
        "stages": stage_rows,
    }


def build_payload() -> tuple[dict[str, Any], str]:
    workflow = clean(os.getenv("NFL_PIPELINE_WORKFLOW")).lower() or WORKFLOW_KEY
    if workflow != WORKFLOW_KEY:
        fail(f"NFL_PIPELINE_WORKFLOW must be {WORKFLOW_KEY!r}; found {workflow!r}")

    job_status = clean(os.getenv("NFL_PIPELINE_JOB_STATUS")).lower() or "unknown"
    if job_status not in VALID_JOB_STATUSES:
        fail(
            f"NFL_PIPELINE_JOB_STATUS must be one of {sorted(VALID_JOB_STATUSES)}; found {job_status!r}"
        )

    projection_mode = clean(os.getenv("NFL_PROJECTION_MODE")).lower()
    if projection_mode not in PROJECTION_REPORTS:
        fail(
            f"NFL_PROJECTION_MODE must be one of {sorted(PROJECTION_REPORTS)}; found {projection_mode!r}"
        )

    settings = read_settings()
    warnings, fatals = environment_target(settings)

    artifact_data, artifact_warnings, artifact_fatals = collect_artifact_health(settings)
    warnings.extend(artifact_warnings)
    fatals.extend(artifact_fatals)

    final_results, final_warnings, final_fatals = collect_final_results_health()
    warnings.extend(final_warnings)
    fatals.extend(final_fatals)

    model, model_warnings, model_fatals = model_health(settings)
    warnings.extend(model_warnings)
    fatals.extend(model_fatals)

    github_run_id = clean(os.getenv("GITHUB_RUN_ID"))
    github_run_attempt = clean(os.getenv("GITHUB_RUN_ATTEMPT"))

    report_health, stage_rows, report_warnings, report_fatals = collect_structured_report_health(
        projection_mode=projection_mode,
        github_run_id=github_run_id,
        github_run_attempt=github_run_attempt,
        job_status=job_status,
    )
    warnings.extend(report_warnings)
    fatals.extend(report_fatals)

    if job_status != "success":
        fatals.append(f"{WORKFLOW_LABEL} job status is {job_status.upper()}")

    status = "failed" if fatals else ("warning" if warnings else "healthy")
    now_utc = datetime.now(UTC)
    now_ny = datetime.now(NY)
    generated_at = now_utc.isoformat()

    scheduled_count = (
        artifact_data.get("weekly_schedule", {}).get("unique_game_ids")
        or artifact_data.get("weekly_schedule", {}).get("rows")
        or 0
    )

    league = {
        "in_season": True,
        "configured_target": settings,
        "counts": {
            "scheduled_games": scheduled_count,
            "projection_games": artifact_data.get("projection", {}).get("rows"),
            "selected_candidate_games": artifact_data.get("selected_candidates", {}).get("rows"),
            "root_pick_games": artifact_data.get("root_picks", {}).get("rows"),
            "all_games_pick_games": artifact_data.get("all_games", {}).get("rows"),
            "selected_bets": artifact_data.get("selected_bets", {}).get("rows"),
            "survivor_games": artifact_data.get("survivor", {}).get("rows"),
            "nm_bet_games": artifact_data.get("nmbets", {}).get("rows"),
            "final_score_games": artifact_data.get("final_scores", {}).get("rows"),
        },
        "artifacts": artifact_data,
        "final_results": final_results,
        "issues": list(warnings),
        "critical_failures": list(fatals),
    }

    workflow_data = workflow_snapshot(
        job_status=job_status,
        projection_mode=projection_mode,
        generated_at=generated_at,
        stage_rows=stage_rows,
    )

    payload = {
        "schema_version": 2,
        "generated_at_utc": generated_at,
        "game_date_new_york": now_ny.strftime("%Y_%m_%d"),
        "status": status,
        "current_workflow": WORKFLOW_KEY,
        "fatal_errors": fatals,
        "warnings": warnings,
        "leagues": {"nfl": league},
        "workflows": {WORKFLOW_KEY: workflow_data},
        "stage_status": stage_rows,
        "model_health": model,
        "report_health": report_health,
        "paths": {
            "nfl_root": str(NFL_ROOT),
            "settings": str(SETTINGS_PATH),
            "pipeline_health": str(OUTPUT),
            "pipeline_health_log": str(LOG),
            "frontend_pipeline_health": str(FRONTEND_OUTPUT),
            "nfl_dashboard": str(DASHBOARD_OUTPUT),
        },
    }

    return payload, job_status


def log_text(payload: dict[str, Any]) -> str:
    lines = [
        f"=== NFL PIPELINE HEALTH {payload['generated_at_utc']} ===",
        f"workflow={payload.get('current_workflow')}",
        f"status={payload.get('status')}",
        f"game_date_new_york={payload.get('game_date_new_york')}",
        f"fatal_errors={len(payload.get('fatal_errors', []))}",
        f"warnings={len(payload.get('warnings', []))}",
    ]
    for error in payload.get("fatal_errors", []):
        lines.append(f"FATAL: {error}")
    for warning in payload.get("warnings", []):
        lines.append(f"WARNING: {warning}")
    return "\n".join(lines) + "\n"


def stage_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{path.name}.stage.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(descriptor)
    staged = Path(raw_path)
    try:
        with staged.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if staged.read_text(encoding="utf-8") != text:
            fail(f"staged output reread mismatch: {path}")
        return staged
    except Exception:
        staged.unlink(missing_ok=True)
        raise


def publish_outputs(payload: dict[str, Any]) -> None:
    json_text = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    log_output = log_text(payload)

    outputs = {
        OUTPUT: json_text,
        FRONTEND_OUTPUT: json_text,
        LOG: log_output,
    }
    staged: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    modified: set[Path] = set()

    try:
        for path, text in outputs.items():
            staged[path] = stage_text(path, text)

        for path in outputs:
            if not path.exists():
                continue
            descriptor, raw_backup = tempfile.mkstemp(
                prefix=f".{path.name}.backup.",
                suffix=".tmp",
                dir=str(path.parent),
            )
            os.close(descriptor)
            backup = Path(raw_backup)
            shutil.copy2(path, backup)
            backups[path] = backup

        for path, staged_path in staged.items():
            os.replace(staged_path, path)
            modified.add(path)

        if OUTPUT.read_text(encoding="utf-8") != json_text:
            fail("published pipeline_health.json reread mismatch")
        if FRONTEND_OUTPUT.read_text(encoding="utf-8") != json_text:
            fail("published frontend NFL pipeline-health reread mismatch")
        if LOG.read_text(encoding="utf-8") != log_output:
            fail("published pipeline health log reread mismatch")

        reread = read_json_object(OUTPUT, required=True)
        if reread != payload:
            fail("published pipeline_health.json content validation failed")
    except Exception:
        for path in modified:
            backup = backups.get(path)
            if backup is not None and backup.exists():
                os.replace(backup, path)
            else:
                path.unlink(missing_ok=True)
        raise
    finally:
        for staged_path in staged.values():
            staged_path.unlink(missing_ok=True)
        for backup in backups.values():
            backup.unlink(missing_ok=True)


def main() -> int:
    try:
        payload, job_status = build_payload()
        publish_outputs(payload)
    except Exception as exc:
        ERRORS.mkdir(parents=True, exist_ok=True)
        fallback = (
            f"=== NFL PIPELINE HEALTH ERROR {datetime.now(UTC).isoformat()} ===\n"
            f"{type(exc).__name__}: {exc}\n"
        )
        try:
            LOG.write_text(fallback, encoding="utf-8")
        except Exception:
            pass
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1

    print(f"NFL pipeline health: {payload['status']}")
    print(f"workflow: {WORKFLOW_KEY} ({job_status})")
    print(f"output: {OUTPUT}")
    print(f"frontend: {FRONTEND_OUTPUT}")
    for warning in payload["warnings"]:
        print(f"WARNING: {warning}")
    for fatal in payload["fatal_errors"]:
        print(f"FATAL: {fatal}")

    return 1 if payload["fatal_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
