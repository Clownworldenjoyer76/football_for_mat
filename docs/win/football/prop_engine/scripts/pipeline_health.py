#!/usr/bin/env python3
"""Build the NFL Prop Engine pipeline health snapshot.

Writes:
    docs/win/football/prop_engine/pipeline_health.json
    docs/win/football/prop_engine/logs/pipeline_health.txt
    docs/win/football/prop_engine/logs/pipeline_reports/health/pipeline_health.json
    frontend/data/pipeline_health/prop_engine.json

The workflow supplies PROP_PIPELINE_WORKFLOW and PROP_PIPELINE_JOB_STATUS.
PipelineReporter JSON files are tied to the current GitHub run/attempt when
that metadata is available.
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
from typing import Any, Never
from zoneinfo import ZoneInfo

from pipeline_reporter import PipelineReporter


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
PROP_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PROP_ROOT.parents[3]

SETTINGS_PATH = PROP_ROOT / "config" / "settings.yaml"
PROP_CONFIG_PATH = PROP_ROOT / "config" / "prop_engine.yaml"
REGISTRY_PATH = PROP_ROOT / "models" / "production_registry.json"
REPORT_ROOT = PROP_ROOT / "logs" / "pipeline_reports"

OUTPUT = PROP_ROOT / "pipeline_health.json"
LOG = PROP_ROOT / "logs" / "pipeline_health.txt"
FRONTEND_OUTPUT = REPO_ROOT / "frontend" / "data" / "pipeline_health" / "prop_engine.json"

NY = ZoneInfo("America/New_York")
WORKFLOW_KEY = "prop_01"
WORKFLOW_LABEL = "Prop 01 Pipeline"
VALID_JOB_STATUSES = {"success", "failure", "cancelled", "unknown"}
VALID_SEASON_TYPES = {"pre", "reg", "post"}
VALID_REPORT_STATUSES = {"SUCCESS", "WARNING", "FAILED"}


def fail(message: str) -> Never:
    raise RuntimeError(message)


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.casefold() in {"", "nan", "none", "null", "<na>", "nat"}:
        return ""
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


def read_top_level_yaml_scalars(path: Path) -> dict[str, str]:
    if not path.is_file():
        fail(f"Settings file not found: {path}")

    values: dict[str, str] = {}
    pattern = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*?)\s*$")

    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line or raw_line[0].isspace():
            continue
        stripped = raw_line.split("#", 1)[0].strip()
        if not stripped:
            continue
        match = pattern.fullmatch(stripped)
        if not match:
            continue
        key, value = match.groups()
        value = value.strip().strip("'\"")
        if key in values:
            fail(f"{path}: duplicate top-level key {key!r} at line {line_number}")
        values[key] = value

    return values


def read_settings() -> dict[str, Any]:
    values = read_top_level_yaml_scalars(SETTINGS_PATH)
    missing = [
        key
        for key in ("season", "week", "season_type")
        if not clean(values.get(key))
    ]
    if missing:
        fail(f"{SETTINGS_PATH}: missing required setting(s): {missing}")

    season = parse_positive_int(values["season"], label="settings season")
    week = parse_positive_int(values["week"], label="settings week")
    season_type = clean(values["season_type"]).lower()

    if not 1900 <= season <= 2200:
        fail(f"settings season is outside supported bounds: {season}")
    if not 1 <= week <= 22:
        fail(f"settings week must be in 1..22; found {week}")
    if season_type not in VALID_SEASON_TYPES:
        fail(
            "settings season_type must be one of "
            f"{sorted(VALID_SEASON_TYPES)}; found {season_type!r}"
        )

    return {
        "season": season,
        "week": week,
        "season_type": season_type,
    }


def read_prop_config_contract() -> dict[str, Any]:
    if not PROP_CONFIG_PATH.is_file():
        fail(f"Prop Engine config not found: {PROP_CONFIG_PATH}")

    section = ""
    market_data_allowed: bool | None = None
    current_season: int | None = None
    targets: list[str] = []

    for raw_line in PROP_CONFIG_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        top = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*):\s*", line)
        if top:
            section = top.group(1)
            continue

        if section == "system":
            match = re.fullmatch(r"\s{2}market_data_allowed:\s*(\S+)\s*", line)
            if match:
                value = match.group(1).casefold()
                if value == "true":
                    market_data_allowed = True
                elif value == "false":
                    market_data_allowed = False
                else:
                    fail(f"Invalid market_data_allowed value: {match.group(1)!r}")

        elif section == "seasons":
            match = re.fullmatch(r"\s{2}current:\s*(\S+)\s*", line)
            if match:
                current_season = parse_positive_int(
                    match.group(1),
                    label="prop_engine.yaml seasons.current",
                )

        elif section == "targets":
            match = re.fullmatch(r"\s{2}([A-Za-z_][A-Za-z0-9_]*):\s*", line)
            if match:
                targets.append(match.group(1))

    if market_data_allowed is None:
        fail("prop_engine.yaml is missing system.market_data_allowed")
    if current_season is None:
        fail("prop_engine.yaml is missing seasons.current")
    if not targets:
        fail("prop_engine.yaml contains no configured targets")

    return {
        "market_data_allowed": market_data_allowed,
        "current_season": current_season,
        "targets": targets,
    }


def environment_target(settings: dict[str, Any]) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    fatals: list[str] = []

    expected = {
        "NFL_SEASON": str(settings["season"]),
        "NFL_WEEK": str(settings["week"]),
    }

    for name, target in expected.items():
        actual = clean(os.getenv(name))
        if not actual:
            warnings.append(f"{name} is not set; settings.yaml is authoritative")
        elif actual != target:
            fatals.append(
                f"{name}={actual!r} does not match settings.yaml value {target!r}"
            )

    configured_root = clean(os.getenv("PROP_ENGINE_ROOT"))
    if configured_root:
        try:
            if Path(configured_root).resolve() != PROP_ROOT.resolve():
                fatals.append(
                    f"PROP_ENGINE_ROOT={configured_root!r} does not resolve to {PROP_ROOT}"
                )
        except Exception as exc:
            fatals.append(f"Unable to resolve PROP_ENGINE_ROOT: {exc}")

    configured_settings = clean(os.getenv("SETTINGS_FILE"))
    if configured_settings:
        try:
            if Path(configured_settings).resolve() != SETTINGS_PATH.resolve():
                fatals.append(
                    f"SETTINGS_FILE={configured_settings!r} does not resolve to {SETTINGS_PATH}"
                )
        except Exception as exc:
            fatals.append(f"Unable to resolve SETTINGS_FILE: {exc}")

    return warnings, fatals


def inspect_csv(
    path: Path,
    *,
    settings: dict[str, Any],
    require_current_target: bool,
) -> dict[str, Any]:
    if not path.is_file():
        fail(f"CSV file not found: {path}")
    if path.stat().st_size <= 0:
        fail(f"CSV file is empty: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            fail(f"CSV has no header: {path}")
        if any(not clean(name) for name in fieldnames):
            fail(f"CSV contains a blank header: {path}")
        if len(fieldnames) != len(set(fieldnames)):
            fail(f"CSV contains duplicate headers: {path}")

        rows = 0
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                fail(f"CSV row {row_number} has fields beyond the header: {path}")
            rows += 1

            if not require_current_target:
                continue

            if "season" in row and clean(row.get("season")):
                season = parse_positive_int(
                    row["season"],
                    label=f"{path} row {row_number} season",
                )
                if season != settings["season"]:
                    fail(
                        f"{path} row {row_number}: season={season} "
                        f"does not match configured season={settings['season']}"
                    )

            if "week" in row and clean(row.get("week")):
                week = parse_positive_int(
                    row["week"],
                    label=f"{path} row {row_number} week",
                )
                if week != settings["week"]:
                    fail(
                        f"{path} row {row_number}: week={week} "
                        f"does not match configured week={settings['week']}"
                    )

    if rows <= 0:
        fail(f"CSV contains no data rows: {path}")

    return {
        "path": str(path),
        "exists": True,
        "bytes": path.stat().st_size,
        "rows": rows,
        "columns": len(fieldnames),
    }


def count_csv_files(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for item in path.rglob("*.csv") if item.is_file() and item.stat().st_size > 0)


def collect_artifact_health(
    settings: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[str], list[str]]:
    warnings: list[str] = []
    fatals: list[str] = []
    season = settings["season"]
    week = settings["week"]

    output_root = PROP_ROOT / "output" / str(season)
    props_root = output_root / f"week_{week}_props"
    stage_root = PROP_ROOT / "prop_picks_final" / str(season)

    files = {
        "player_projections": (
            output_root / f"week_{week}_player_projections.csv",
            True,
        ),
        "active_player_projections": (
            output_root / f"week_{week}_active_player_projections.csv",
            True,
        ),
        "player_projections_wide": (
            output_root / f"week_{week}_player_projections_wide.csv",
            True,
        ),
        "stage_3": (
            stage_root / "stage_3" / f"{season}_{week}_all_props.csv",
            True,
        ),
        "graded_all_props": (
            PROP_ROOT / "05_final" / "graded" / str(season) / f"{season}_all_props_graded.csv",
            False,
        ),
    }

    artifacts: dict[str, Any] = {}

    for name, (path, require_current) in files.items():
        try:
            artifacts[name] = inspect_csv(
                path,
                settings=settings,
                require_current_target=require_current,
            )
        except Exception as exc:
            fatals.append(f"{name}: {exc}")
            artifacts[name] = {
                "path": str(path),
                "exists": path.is_file(),
                "error": f"{type(exc).__name__}: {exc}",
            }

    directories = {
        "prop_odds": props_root,
        "prop_selections": props_root / "selections",
        "stage_1": stage_root / "stage_1" / f"week_{week}",
        "stage_2": stage_root / "stage_2" / f"week_{week}",
        "dashboard_reports": PROP_ROOT / "05_final" / "reports" / "dashboard",
    }

    for name, path in directories.items():
        csv_files = count_csv_files(path)
        artifacts[name] = {
            "path": str(path),
            "exists": path.is_dir(),
            "csv_files": csv_files,
        }
        if not path.is_dir():
            fatals.append(f"{name}: directory not found: {path}")
        elif csv_files <= 0:
            fatals.append(f"{name}: no non-empty CSV outputs found: {path}")

    required_report_files = (
        "overall.csv",
        "by_prop_type.csv",
        "by_probability.csv",
        "by_pick_direction.csv",
        "by_week.csv",
        "calibration.csv",
    )
    reports_root = PROP_ROOT / "05_final" / "reports" / str(season)
    missing_reports = [
        name
        for name in required_report_files
        if not (reports_root / name).is_file()
        or (reports_root / name).stat().st_size <= 0
    ]
    artifacts["graded_reports"] = {
        "path": str(reports_root),
        "expected_files": len(required_report_files),
        "missing_files": missing_reports,
    }
    if missing_reports:
        fatals.append(f"graded_reports: missing/empty files: {missing_reports}")

    validation_path = output_root / f"week_{week}_validation.json"
    manifest_path = output_root / f"week_{week}_run_manifest.json"

    validation: dict[str, Any] = {}
    manifest: dict[str, Any] = {}

    try:
        validation = read_json_object(validation_path, required=True)
        if int(validation.get("season", -1)) != season:
            fatals.append("validation: season does not match settings")
        if int(validation.get("week", -1)) != week:
            fatals.append("validation: week does not match settings")
        if validation.get("status") != "passed":
            fatals.append(f"validation: status={validation.get('status')!r}")
        if int(validation.get("checks_failed", -1)) != 0:
            fatals.append("validation: checks_failed is not zero")
        if validation.get("market_features_used") is not False:
            fatals.append("validation: market_features_used is not false")
    except Exception as exc:
        fatals.append(f"validation: {exc}")

    try:
        manifest = read_json_object(manifest_path, required=True)
        if int(manifest.get("season", -1)) != season:
            fatals.append("run_manifest: season does not match settings")
        if int(manifest.get("week", -1)) != week:
            fatals.append("run_manifest: week does not match settings")
        if manifest.get("status") != "success":
            fatals.append(f"run_manifest: status={manifest.get('status')!r}")
        # noinspection PySimplifyBooleanCheck
        if manifest.get("validation_passed") is not True:
            fatals.append("run_manifest: validation_passed is not true")
        if manifest.get("market_data_used") is not False:
            fatals.append("run_manifest: market_data_used is not false")
        if manifest.get("allow_unapproved_models") is not False:
            fatals.append("run_manifest: allow_unapproved_models is not false")

        steps = manifest.get("steps")
        if not isinstance(steps, list) or not steps:
            fatals.append("run_manifest: steps is missing or empty")
        else:
            bad_steps: list[str] = []
            for step in steps:
                if not isinstance(step, dict):
                    bad_steps.append("<invalid-step>")
                    continue
                if clean(step.get("status")).lower() not in {"success", "skipped"}:
                    bad_steps.append(clean(step.get("script")) or "<unknown>")
            if bad_steps:
                fatals.append(f"run_manifest: failed/invalid step(s): {bad_steps}")
    except Exception as exc:
        fatals.append(f"run_manifest: {exc}")

    artifacts["validation"] = {
        "path": str(validation_path),
        "exists": validation_path.is_file(),
        "status": validation.get("status") if validation else None,
        "checks_failed": validation.get("checks_failed") if validation else None,
    }
    artifacts["run_manifest"] = {
        "path": str(manifest_path),
        "exists": manifest_path.is_file(),
        "status": manifest.get("status") if manifest else None,
        "production_targets": manifest.get("production_targets") if manifest else None,
        "deferred_targets": manifest.get("deferred_targets") if manifest else None,
    }

    return artifacts, validation, manifest, warnings, fatals


def model_health(
    settings: dict[str, Any],
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[str]]:
    warnings: list[str] = []
    fatals: list[str] = []

    try:
        config = read_prop_config_contract()
    except Exception as exc:
        return {}, warnings, [f"prop_engine config: {exc}"]

    try:
        registry = read_json_object(REGISTRY_PATH, required=True)
    except Exception as exc:
        return {"config": config}, warnings, [f"production registry: {exc}"]

    if config["current_season"] != settings["season"]:
        fatals.append(
            "prop_engine.yaml seasons.current does not match settings season"
        )
    if config["market_data_allowed"] is not False:
        fatals.append("prop_engine.yaml market_data_allowed must be false")

    configured_targets = list(config["targets"])
    if set(registry) != set(configured_targets):
        fatals.append("production registry target set does not match prop_engine.yaml")

    approved: list[str] = []
    deferred: list[str] = []
    invalid: list[str] = []

    for target in configured_targets:
        entry = registry.get(target)
        if not isinstance(entry, dict):
            invalid.append(target)
            continue

        approved_flag = entry.get("production_approved")
        version = entry.get("version")

        if approved_flag is True and isinstance(version, str) and version.strip():
            approved.append(target)
        elif approved_flag is False and version is None:
            deferred.append(target)
        else:
            invalid.append(target)

    if invalid:
        fatals.append(f"invalid production registry target(s): {invalid}")
    if not approved:
        fatals.append("production registry has no approved targets")

    manifest_production = manifest.get("production_targets") if manifest else None
    manifest_deferred = manifest.get("deferred_targets") if manifest else None
    manifest_versions = manifest.get("production_model_versions") if manifest else None

    if manifest:
        if not isinstance(manifest_production, list) or set(manifest_production) != set(approved):
            fatals.append("run_manifest production_targets do not match production registry")
        if not isinstance(manifest_deferred, list) or set(manifest_deferred) != set(deferred):
            fatals.append("run_manifest deferred_targets do not match production registry")
        if not isinstance(manifest_versions, dict) or set(manifest_versions) != set(approved):
            fatals.append("run_manifest production_model_versions coverage mismatch")

    return {
        "config_path": str(PROP_CONFIG_PATH),
        "registry_path": str(REGISTRY_PATH),
        "configured_targets": configured_targets,
        "approved_targets": approved,
        "deferred_targets": deferred,
        "invalid_targets": invalid,
        "market_data_allowed": config["market_data_allowed"],
        "configured_season": config["current_season"],
    }, warnings, fatals


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


def collect_structured_report_health() -> tuple[dict[str, Any], list[dict[str, Any]], list[str], list[str]]:
    warnings: list[str] = []
    fatals: list[str] = []
    rows: list[dict[str, Any]] = []

    github_run_id = clean(os.getenv("GITHUB_RUN_ID"))
    github_run_attempt = clean(os.getenv("GITHUB_RUN_ATTEMPT"))

    if not REPORT_ROOT.is_dir():
        return {
            "report_root": str(REPORT_ROOT),
            "current_reports_total": 0,
            "failed_reports": 0,
            "warning_reports": 0,
            "github_run_id": github_run_id or None,
            "github_run_attempt": github_run_attempt or None,
        }, rows, warnings, fatals

    failed_count = 0
    warning_count = 0

    for path in sorted(REPORT_ROOT.rglob("*.json")):
        try:
            payload = read_json_object(path, required=True)
        except Exception as exc:
            warnings.append(f"Unreadable pipeline report ignored: {path}: {exc}")
            continue

        if not report_is_current(
            payload,
            github_run_id=github_run_id,
            github_run_attempt=github_run_attempt,
        ):
            continue

        status = clean(payload.get("status")).upper()
        row = {
            "stage": clean(payload.get("stage")) or None,
            "script": clean(payload.get("script")) or path.stem,
            "path": str(path),
            "status": status or None,
            "warning_count": payload.get("warning_count"),
            "error_count": payload.get("error_count"),
            "duration_seconds": payload.get("duration_seconds"),
        }
        rows.append(row)

        if status not in VALID_REPORT_STATUSES:
            fatals.append(f"pipeline report {path} has invalid status {status!r}")
        elif status == "FAILED":
            failed_count += 1
            fatals.append(f"pipeline report {path} status is FAILED")
        elif status == "WARNING":
            warning_count += 1
            warnings.append(f"pipeline report {path} status is WARNING")

    if not github_run_id:
        warnings.append(
            "GITHUB_RUN_ID is unavailable; reporter files cannot be tied to the current run"
        )

    return {
        "report_root": str(REPORT_ROOT),
        "current_reports_total": len(rows),
        "failed_reports": failed_count,
        "warning_reports": warning_count,
        "github_run_id": github_run_id or None,
        "github_run_attempt": github_run_attempt or None,
    }, rows, warnings, fatals


def build_payload() -> tuple[dict[str, Any], str]:
    workflow = clean(os.getenv("PROP_PIPELINE_WORKFLOW")).lower() or WORKFLOW_KEY
    if workflow != WORKFLOW_KEY:
        fail(
            f"PROP_PIPELINE_WORKFLOW must be {WORKFLOW_KEY!r}; found {workflow!r}"
        )

    job_status = clean(os.getenv("PROP_PIPELINE_JOB_STATUS")).lower() or "unknown"
    if job_status not in VALID_JOB_STATUSES:
        fail(
            "PROP_PIPELINE_JOB_STATUS must be one of "
            f"{sorted(VALID_JOB_STATUSES)}; found {job_status!r}"
        )

    settings = read_settings()

    warnings, fatals = environment_target(settings)

    artifacts, validation, manifest, artifact_warnings, artifact_fatals = (
        collect_artifact_health(settings)
    )
    warnings.extend(artifact_warnings)
    fatals.extend(artifact_fatals)

    model, model_warnings, model_fatals = model_health(settings, manifest)
    warnings.extend(model_warnings)
    fatals.extend(model_fatals)

    report_health, stage_rows, report_warnings, report_fatals = (
        collect_structured_report_health()
    )
    warnings.extend(report_warnings)
    fatals.extend(report_fatals)

    if job_status != "success":
        fatals.append(f"{WORKFLOW_LABEL} job status is {job_status.upper()}")

    status = "failed" if fatals else ("warning" if warnings else "healthy")
    now_utc = datetime.now(UTC)
    now_ny = datetime.now(NY)
    generated_at = now_utc.isoformat()

    payload = {
        "schema_version": 2,
        "generated_at_utc": generated_at,
        "game_date_new_york": now_ny.strftime("%Y_%m_%d"),
        "status": status,
        "current_workflow": WORKFLOW_KEY,
        "configured_target": settings,
        "fatal_errors": fatals,
        "warnings": warnings,
        "artifacts": artifacts,
        "validation": {
            "status": validation.get("status") if validation else None,
            "checks_total": validation.get("checks_total") if validation else None,
            "checks_failed": validation.get("checks_failed") if validation else None,
            "market_features_used": (
                validation.get("market_features_used") if validation else None
            ),
        },
        "model_health": model,
        "report_health": report_health,
        "stage_status": stage_rows,
        "workflow": {
            "workflow": WORKFLOW_KEY,
            "label": WORKFLOW_LABEL,
            "status": job_status,
            "generated_at_utc": generated_at,
            "run_id": clean(os.getenv("GITHUB_RUN_ID")),
            "run_attempt": clean(os.getenv("GITHUB_RUN_ATTEMPT")),
            "sha": clean(os.getenv("GITHUB_SHA")),
            "ref_name": clean(os.getenv("GITHUB_REF_NAME")),
        },
        "paths": {
            "prop_engine_root": str(PROP_ROOT),
            "settings": str(SETTINGS_PATH),
            "prop_config": str(PROP_CONFIG_PATH),
            "production_registry": str(REGISTRY_PATH),
            "pipeline_health": str(OUTPUT),
            "pipeline_health_log": str(LOG),
            "pipeline_reports": str(REPORT_ROOT),
            "frontend_pipeline_health": str(FRONTEND_OUTPUT),
        },
    }

    return payload, job_status


def log_text(payload: dict[str, Any]) -> str:
    lines = [
        f"=== NFL PROP ENGINE PIPELINE HEALTH {payload['generated_at_utc']} ===",
        f"workflow={payload.get('current_workflow')}",
        f"status={payload.get('status')}",
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
            fail("published frontend pipeline-health reread mismatch")
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


def write_fallback_log(exc: BaseException) -> None:
    # noinspection PyBroadException
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        LOG.write_text(
            (
                f"=== NFL PROP ENGINE PIPELINE HEALTH ERROR "
                f"{datetime.now(UTC).isoformat()} ===\n"
                f"{type(exc).__name__}: {exc}\n"
            ),
            encoding="utf-8",
        )
    except Exception:
        pass


def main() -> int:
    reporter = PipelineReporter(
        script=SCRIPT_PATH,
        stage="health",
        report_root=REPORT_ROOT,
    )
    reporter.add_input(SETTINGS_PATH)
    reporter.add_input(PROP_CONFIG_PATH)
    reporter.add_input(REGISTRY_PATH)

    try:
        payload, job_status = build_payload()
        publish_outputs(payload)

        reporter.add_output(OUTPUT)
        reporter.add_output(LOG)
        reporter.add_output(FRONTEND_OUTPUT)
        reporter.update_details(
            {
                "health_status": payload["status"],
                "workflow_status": job_status,
                "fatal_count": len(payload["fatal_errors"]),
                "warning_count": len(payload["warnings"]),
                "current_report_count": payload["report_health"][
                    "current_reports_total"
                ],
            }
        )

        for warning in payload["warnings"]:
            reporter.warning(warning)

        for fatal in payload["fatal_errors"]:
            reporter.error(fatal)

        final_status = (
            "FAILED"
            if payload["fatal_errors"]
            else ("WARNING" if payload["warnings"] else "SUCCESS")
        )
        exit_code = 1 if payload["fatal_errors"] else 0
        reporter.write_report(status=final_status, exit_code=exit_code)

    except Exception as exc:
        reporter.record_exception(exc)
        # noinspection PyBroadException
        try:
            reporter.write_report(status="FAILED", exit_code=1)
        except Exception:
            pass
        write_fallback_log(exc)
        print(
            f"Prop Engine pipeline health ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1

    print(
        "Prop Engine pipeline health: "
        f"{payload['status']} "
        f"(fatal={len(payload['fatal_errors'])}, "
        f"warnings={len(payload['warnings'])})"
    )
    return 1 if payload["fatal_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
