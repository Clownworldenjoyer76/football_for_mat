#!/usr/bin/env python3
# docs/win/baseball/mlb/scripts/pipeline_health.py
"""MLB pipeline health reporter used by the morning and pregame workflows.

Writes:
    docs/win/baseball/mlb/pipeline_health.json
    docs/win/baseball/mlb/errors/pipeline_health.txt
    frontend/data/pipeline_health/mlb.json

The current workflow identity/status is supplied by the workflow through
MLB_PIPELINE_WORKFLOW and MLB_PIPELINE_JOB_STATUS. The latest Morning and
Pregame workflow snapshots are preserved in one shared JSON file so the
existing frontend/pipeline_health.html page can report both pipelines.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

BASE = Path("docs/win/baseball/mlb")
ERRORS = BASE / "errors"
OUTPUT = BASE / "pipeline_health.json"
LOG = ERRORS / "pipeline_health.txt"
FRONTEND_OUTPUT = Path("frontend/data/pipeline_health/mlb.json")
NY = ZoneInfo("America/New_York")

WORKFLOW_LABELS = {
    "morning": "MLB 01 Morning",
    "pregame": "MLB 02 Pregame",
}

WORKFLOW_STAGE_LOGS = {
    "morning": [
        ("Build Run Training Set", ERRORS / "modeling/build_run_training_set.txt"),
        ("Train Run Model", ERRORS / "modeling/train_run_model.txt"),
        ("Evaluate Run Model", ERRORS / "modeling/evaluate_run_model.txt"),
        ("Odds Parse", ERRORS / "00_intake/odds_parse.txt"),
        ("Name Normalization", ERRORS / "00_intake/name_normalization.txt"),
        ("DRatings Scraper", ERRORS / "00_intake/baseball_drat_scraper.txt"),
        ("Transform Baseball", ERRORS / "00_intake/transform_baseball.txt"),
        ("Scrape MLB Raw", ERRORS / "00_intake/scrape_mlb_raw.txt"),
        ("Build Games List", ERRORS / "00_intake/build_games_list.txt"),
        ("SportsDataverse Features", ERRORS / "00_intake/sportsdataverse_mlb.txt"),
        ("Fetch Park Weather", ERRORS / "00_intake/fetch_park_weather.txt"),
        ("Build Game Weather", ERRORS / "00_intake/build_game_weather.txt"),
        ("Enrich Game Context", ERRORS / "00_intake/enrich_game_context.txt"),
        ("Game ID Predictions", ERRORS / "00_intake/game_id_pred.txt"),
        ("Build Run Projection", ERRORS / "00_intake/build_run_projection.txt"),
        ("Merge Intake", ERRORS / "01_merge/merge_intake.txt"),
        ("Build Juice Files", ERRORS / "01_merge/build_juice_files.txt"),
        ("Compute Edges", ERRORS / "03_edges/compute_edges.txt"),
        ("Compute EV Kelly", ERRORS / "03_edges/compute_ev_kelly.txt"),
        ("Select Bets", ERRORS / "04_select/select_bets.txt"),
        ("Select Morning Bets", ERRORS / "04_select/select_bets_AM.txt"),
    ],
    "pregame": [
        ("Odds Parse", ERRORS / "00_intake/odds_parse.txt"),
        ("Name Normalization", ERRORS / "00_intake/name_normalization.txt"),
        ("DRatings Scraper", ERRORS / "00_intake/baseball_drat_scraper.txt"),
        ("Transform Baseball", ERRORS / "00_intake/transform_baseball.txt"),
        ("Scrape MLB Raw", ERRORS / "00_intake/scrape_mlb_raw.txt"),
        ("Build Games List", ERRORS / "00_intake/build_games_list.txt"),
        ("SportsDataverse Features", ERRORS / "00_intake/sportsdataverse_mlb.txt"),
        ("Fetch Park Weather", ERRORS / "00_intake/fetch_park_weather.txt"),
        ("Build Game Weather", ERRORS / "00_intake/build_game_weather.txt"),
        ("Enrich Game Context", ERRORS / "00_intake/enrich_game_context.txt"),
        ("Game ID Predictions", ERRORS / "00_intake/game_id_pred.txt"),
        ("Build Run Projection", ERRORS / "00_intake/build_run_projection.txt"),
        ("Merge Intake", ERRORS / "01_merge/merge_intake.txt"),
        ("Build Juice Files", ERRORS / "01_merge/build_juice_files.txt"),
        ("Compute Edges", ERRORS / "03_edges/compute_edges.txt"),
        ("Compute EV Kelly", ERRORS / "03_edges/compute_ev_kelly.txt"),
        ("Select Bets", ERRORS / "04_select/select_bets.txt"),
    ],
}

BAD_STATUS_TEXT = (
    "STATUS: FAILED",
    "STATUS: PARTIAL",
    "STATUS: COMPLETED WITH ERRORS",
    "FATAL ERROR",
)


def clean(value) -> str:
    return "" if value is None else str(value).strip()


def clean_id(value) -> str:
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


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def ids_and_integrity(rows: list[dict]) -> tuple[set[str], int, list[str]]:
    ids: set[str] = set()
    duplicate_ids: set[str] = set()
    blank = 0
    for row in rows:
        gid = clean_id(row.get("game_id"))
        if not gid:
            blank += 1
            continue
        if gid in ids:
            duplicate_ids.add(gid)
        ids.add(gid)
    return ids, blank, sorted(duplicate_ids)


def last_status(path: Path) -> str | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    statuses = [line.strip() for line in text.splitlines() if "STATUS:" in line]
    if statuses:
        return statuses[-1]
    if "FATAL ERROR" in text.upper():
        return "STATUS: FAILED"
    return None


def snapshot_stages(workflow: str) -> list[dict]:
    stages: list[dict] = []
    label = WORKFLOW_LABELS[workflow]
    for stage_name, path in WORKFLOW_STAGE_LOGS[workflow]:
        status = last_status(path)
        stages.append({
            "name": stage_name,
            "path": str(path),
            "exists": path.exists(),
            "status": status or ("STATUS: LOG PRESENT" if path.exists() else "STATUS: NO LOG"),
            "display_path": f"{label} - {stage_name}",
        })
    return stages


def read_existing() -> dict:
    if not OUTPUT.exists():
        return {}
    try:
        payload = json.loads(OUTPUT.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def current_counts(now: datetime) -> tuple[dict, dict, list[str], list[str]]:
    date = now.strftime("%Y_%m_%d")
    paths = {
        "daily_games": BASE / f"00_intake/games/{date}_games.csv",
        "predictions": BASE / f"00_intake/predictions/model_projection/{date}_MLB.csv",
        "sportsbook": BASE / f"00_intake/sportsbook/{date}_MLB.csv",
        "merged": BASE / f"01_merge/{date}_mlb_moneyline.csv",
        "selected": BASE / f"04_select/{date}_MLB.csv",
        "morning_selected": BASE / f"04_select/morning/{date}_MLB.csv",
    }

    rows = {key: read_rows(path) for key, path in paths.items()}
    scheduled_ids, scheduled_blank, scheduled_dupes = ids_and_integrity(rows["daily_games"])
    prediction_ids, prediction_blank, prediction_dupes = ids_and_integrity(rows["predictions"])
    sportsbook_ids, sportsbook_blank, sportsbook_dupes = ids_and_integrity(rows["sportsbook"])
    merged_ids, merged_blank, merged_dupes = ids_and_integrity(rows["merged"])

    warnings: list[str] = []
    fatals: list[str] = []

    integrity = {
        "blank_daily_game_ids": scheduled_blank,
        "blank_prediction_game_ids": prediction_blank,
        "blank_sportsbook_game_ids": sportsbook_blank,
        "blank_merged_game_ids": merged_blank,
        "daily_duplicate_game_ids": scheduled_dupes,
        "prediction_duplicate_game_ids": prediction_dupes,
        "sportsbook_duplicate_game_ids": sportsbook_dupes,
        "merged_duplicate_game_ids": merged_dupes,
    }

    if any((scheduled_blank, prediction_blank, sportsbook_blank, merged_blank)):
        fatals.append("MLB: blank game_id found in current-day pipeline data")
    if any((scheduled_dupes, prediction_dupes, sportsbook_dupes, merged_dupes)):
        fatals.append("MLB: duplicate game_id found in current-day pipeline data")

    scheduled_missing_predictions = sorted(scheduled_ids - prediction_ids)
    scheduled_missing_sportsbook = sorted(scheduled_ids - sportsbook_ids)
    sportsbook_not_merged = sorted(sportsbook_ids - merged_ids)

    if scheduled_missing_predictions:
        warnings.append(
            f"MLB: {len(scheduled_missing_predictions)} scheduled game(s) have no model projection"
        )
    if scheduled_missing_sportsbook:
        warnings.append(
            f"MLB: {len(scheduled_missing_sportsbook)} scheduled game(s) have no sportsbook row"
        )
    if sportsbook_not_merged:
        warnings.append(
            f"MLB: {len(sportsbook_not_merged)} sportsbook game(s) did not reach the merged moneyline output"
        )

    league = {
        "in_season": True,
        "paths": {key: str(path) for key, path in paths.items()},
        "counts": {
            "scheduled_games": len(scheduled_ids),
            "prediction_games": len(prediction_ids),
            "sportsbook_games": len(sportsbook_ids),
            "merged_games": len(merged_ids),
            "selected_bets": len(rows["selected"]),
            "locked_bets": 0,
            "morning_selected_bets": len(rows["morning_selected"]),
        },
        "identity": integrity,
        "coverage": {
            "scheduled_missing_predictions": scheduled_missing_predictions,
            "scheduled_missing_sportsbook": scheduled_missing_sportsbook,
            "sportsbook_not_merged": sportsbook_not_merged,
        },
        "issues": [],
        "critical_failures": [],
    }
    return league, {key: str(path) for key, path in paths.items()}, warnings, fatals


def model_health() -> dict:
    validation_path = BASE / "modeling/reports/run_model_validation.json"
    promotion_path = BASE / "modeling/reports/run_model_promotion.json"
    home_meta_path = BASE / "models/run_projection/home_runs_model_metadata.json"
    away_meta_path = BASE / "models/run_projection/away_runs_model_metadata.json"

    validation = read_json(validation_path)
    promotion = read_json(promotion_path)
    home_meta = read_json(home_meta_path)
    away_meta = read_json(away_meta_path)

    return {
        "validation_path": str(validation_path),
        "validation_exists": validation_path.exists(),
        "validation_status": validation.get("status"),
        "validated_at": validation.get("validated_at"),
        "promotion_path": str(promotion_path),
        "promotion_exists": promotion_path.exists(),
        "promotion_status": promotion.get("status"),
        "home_model_metadata_exists": home_meta_path.exists(),
        "away_model_metadata_exists": away_meta_path.exists(),
        "home_model_created_at": home_meta.get("created_at"),
        "away_model_created_at": away_meta.get("created_at"),
    }


def workflow_snapshot(workflow: str, job_status: str, generated_at: str) -> dict:
    normalized = job_status.strip().lower() or "unknown"
    status = "success" if normalized == "success" else normalized
    if status not in {"success", "failure", "cancelled", "unknown"}:
        status = normalized

    return {
        "workflow": workflow,
        "label": WORKFLOW_LABELS[workflow],
        "status": status,
        "generated_at_utc": generated_at,
        "game_date_new_york": datetime.now(NY).strftime("%Y_%m_%d"),
        "run_id": clean(os.getenv("GITHUB_RUN_ID")),
        "run_attempt": clean(os.getenv("GITHUB_RUN_ATTEMPT")),
        "sha": clean(os.getenv("GITHUB_SHA")),
        "ref_name": clean(os.getenv("GITHUB_REF_NAME")),
        "stages": snapshot_stages(workflow),
    }


def workflow_stage_rows(workflows: dict) -> list[dict]:
    rows: list[dict] = []
    for key in ("morning", "pregame"):
        item = workflows.get(key)
        if not isinstance(item, dict):
            rows.append({
                "path": WORKFLOW_LABELS[key] + " Workflow",
                "exists": False,
                "status": "STATUS: NOT RUN",
            })
            continue

        raw_status = clean(item.get("status")).lower()
        if raw_status == "success":
            status = "STATUS: SUCCESS"
        elif raw_status == "failure":
            status = "STATUS: FAILED"
        elif raw_status == "cancelled":
            status = "STATUS: CANCELLED"
        else:
            status = "STATUS: UNKNOWN"

        stamp = clean(item.get("generated_at_utc"))
        if stamp:
            status = f"{status} | {stamp}"

        rows.append({
            "path": WORKFLOW_LABELS[key] + " Workflow",
            "exists": True,
            "status": status,
        })

        for stage in item.get("stages", []):
            if not isinstance(stage, dict):
                continue
            rows.append({
                "path": clean(stage.get("display_path")) or clean(stage.get("path")),
                "exists": bool(stage.get("exists")),
                "status": clean(stage.get("status")) or None,
                "source_path": clean(stage.get("path")),
            })
    return rows


def write_log(payload: dict) -> None:
    ERRORS.mkdir(parents=True, exist_ok=True)
    lines = [
        f"=== MLB PIPELINE HEALTH {payload['generated_at_utc']} ===",
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
    LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    workflow = clean(os.getenv("MLB_PIPELINE_WORKFLOW")).lower()
    if workflow not in WORKFLOW_LABELS:
        raise SystemExit(
            "MLB_PIPELINE_WORKFLOW must be 'morning' or 'pregame'"
        )

    job_status = clean(os.getenv("MLB_PIPELINE_JOB_STATUS")) or "unknown"
    now_utc = datetime.now(UTC)
    now_ny = datetime.now(NY)
    generated_at = now_utc.isoformat()

    existing = read_existing()
    workflows = existing.get("workflows")
    if not isinstance(workflows, dict):
        workflows = {}

    workflows[workflow] = workflow_snapshot(workflow, job_status, generated_at)

    league, paths, warnings, fatals = current_counts(now_ny)

    if job_status.lower() != "success":
        fatals.append(
            f"{WORKFLOW_LABELS[workflow]} workflow status is {job_status.upper()}"
        )

    current_stages = workflows[workflow].get("stages", [])
    bad_current_stages = []
    for stage in current_stages:
        status = clean(stage.get("status")).upper()
        if any(token in status for token in BAD_STATUS_TEXT):
            bad_current_stages.append(
                f"{stage.get('name')}: {stage.get('status')}"
            )
    if bad_current_stages and job_status.lower() == "success":
        fatals.append(
            f"{WORKFLOW_LABELS[workflow]} has stage log failure(s): "
            + "; ".join(bad_current_stages)
        )

    status = "failed" if fatals else ("warning" if warnings else "healthy")

    payload = {
        "schema_version": 1,
        "generated_at_utc": generated_at,
        "game_date_new_york": now_ny.strftime("%Y_%m_%d"),
        "status": status,
        "current_workflow": workflow,
        "fatal_errors": fatals,
        "warnings": warnings,
        "leagues": {"mlb": league},
        "workflows": workflows,
        "stage_status": workflow_stage_rows(workflows),
        "model_health": model_health(),
        "paths": paths,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    FRONTEND_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    OUTPUT.write_text(text, encoding="utf-8")
    FRONTEND_OUTPUT.write_text(text, encoding="utf-8")
    write_log(payload)

    print(f"MLB pipeline health: {status}")
    print(f"workflow: {workflow} ({job_status})")
    print(f"output: {OUTPUT}")
    print(f"frontend: {FRONTEND_OUTPUT}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    for fatal in fatals:
        print(f"FATAL: {fatal}")

    return 1 if fatals else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        ERRORS.mkdir(parents=True, exist_ok=True)
        LOG.write_text(
            f"=== MLB PIPELINE HEALTH ERROR {datetime.now(UTC).isoformat()} ===\n"
            f"{type(exc).__name__}: {exc}\n",
            encoding="utf-8",
        )
        raise
