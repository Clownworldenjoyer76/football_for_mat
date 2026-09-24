#!/usr/bin/env python3
"""Production NFL Week 2+ projection using market-independent v4 models."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
NFL_ROOT = SCRIPT_PATH.parents[2]
REPORT_ROOT = NFL_ROOT / "errors"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_reporter import PipelineReporter
from projection_common import (
    load_compatibility_schema,
    produce_and_publish_projection,
)

SCRIPT_VERSION = "2026-09-21-settings-season-week-hardened"
EXPECTED_FEATURE_COUNT = 260
MODEL_NAME = "v4_market_independent_outcomes"


def parse_configured_integer(value: Any, *, key: str, path: Path) -> int:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{path}: {key} must be an integer; found {value!r}")

    if isinstance(value, int):
        return value

    if isinstance(value, str):
        text = value.strip()
        if text and text.lstrip("+-").isdigit():
            return int(text)

    raise ValueError(f"{path}: {key} must be an integer; found {value!r}")


def load_runtime_settings(root: Path) -> tuple[int, int, Path]:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("Could not import PyYAML required for NFL settings") from exc

    path = root / "config/settings.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Missing NFL settings file: {path}")

    with path.open("r", encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)

    if not isinstance(settings, dict):
        raise ValueError(f"{path}: expected a YAML mapping")

    if "season" not in settings:
        raise ValueError(f"{path}: missing season")
    if "week" not in settings:
        raise ValueError(f"{path}: missing week")

    season = parse_configured_integer(
        settings["season"],
        key="season",
        path=path,
    )
    week = parse_configured_integer(
        settings["week"],
        key="week",
        path=path,
    )

    if season < 1:
        raise ValueError(f"{path}: season must be >= 1; found {season}")
    if week < 2:
        raise ValueError(
            f"{path}: projection.py is the Week 2+ projection path; "
            f"configured week must be >= 2; found {week}"
        )

    return season, week, path


def run_projection(*, reporter: PipelineReporter) -> list[Path]:
    root = NFL_ROOT
    settings_path = root / "config/settings.yaml"
    reporter.add_input(settings_path)

    season, week, loaded_settings_path = load_runtime_settings(root)
    if loaded_settings_path != settings_path:
        raise RuntimeError(
            "Resolved NFL settings path changed unexpectedly: "
            f"{loaded_settings_path}"
        )

    reporter.season = season
    reporter.week = week
    reporter.update_details(
        {
            "configured_season": season,
            "configured_week": week,
            "week1_mode": False,
            "settings_source": str(settings_path),
            "model": MODEL_NAME,
            "script_version": SCRIPT_VERSION,
            "expected_feature_count": EXPECTED_FEATURE_COUNT,
            "compatibility_schema_validated": False,
            "dependency_imports_ok": False,
            "staged_roundtrip_verified": False,
            "publication_completed": False,
            "post_publish_validation": False,
            "rollback_performed": False,
        }
    )

    import projection_feature_builder_legacy_inseason as helper
    from v4_production import (
        OUTPUT_COLUMNS,
        apply_v4_production_models,
    )
    reporter.set_detail("dependency_imports_ok", True)

    helper_root = helper.nfl_root()
    if helper_root.resolve() != root.resolve():
        raise RuntimeError(
            f"Resolved NFL root mismatch: helper={helper_root} script={root}"
        )

    combined_path = (
        root
        / "00_intake/predictions/enriched/combined"
        / f"week_{week}_NFL_enriched.csv"
    )
    output_path = root / "01_merge" / f"week_{week}_NFL_enriched.csv"

    reporter.add_input(combined_path)
    reporter.add_output(output_path)

    source_frame = helper.read_csv(combined_path)
    source_rows = len(source_frame)
    reporter.set_rows(rows_in=source_rows)
    reporter.set_detail("source_rows", source_rows)

    compatibility_schema, compatibility_path = load_compatibility_schema(root, expected_feature_count=EXPECTED_FEATURE_COUNT)
    reporter.add_input(compatibility_path)
    reporter.update_details(
        {
            "compatibility_schema_validated": True,
            "compatibility_schema_features": len(
                compatibility_schema["feature_order"]
            ),
            "compatibility_schema_numeric_features": len(
                compatibility_schema["numeric_features"]
            ),
            "compatibility_schema_categorical_features": len(
                compatibility_schema["categorical_features"]
            ),
        }
    )

    original, full_features = helper.prepare_week(
        root,
        season,
        week,
        False,
        compatibility_schema,
    )

    eligible_rows = len(original)
    reporter.update_details(
        {
            "eligible_rows": eligible_rows,
            "started_games_skipped": source_rows - eligible_rows,
            "output_prediction_columns": list(OUTPUT_COLUMNS),
        }
    )

    projected = produce_and_publish_projection(
        root=root,
        original=original,
        full_features=full_features,
        apply_models=apply_v4_production_models,
        output_path=output_path,
        output_columns=list(OUTPUT_COLUMNS),
        helper=helper,
        reporter=reporter,
        stage_prefix=".projection_stage_",
        stage_error_label="Staged in-season projection",
        rollback_failure_label=(
            "In-season projection publication failed and rollback "
            "also failed"
        ),
        backup_cleanup_warning=(
            "In-season projection temporary backup cleanup failed"
        ),
        cleanup_backup_if_unpublished=True,
    )
    reporter.update_details(
        {
            "output_rows": len(projected),
            "output_columns": len(projected.columns),
        }
    )

    print(
        f"WROTE {output_path} | games={len(projected)} | "
        f"model={MODEL_NAME} | columns={len(projected.columns)}"
    )

    return [output_path]


def main() -> int:
    print(f"projection.py version={SCRIPT_VERSION}")

    try:
        with PipelineReporter(
            script=SCRIPT_PATH,
            stage="01_merge",
            report_root=REPORT_ROOT,
            pipeline="NFL",
            league="NFL",
            extra_context={
                "component": "Week 2+ production projection",
                "model": MODEL_NAME,
                "runtime_config_source": "config/settings.yaml",
            },
        ) as reporter:
            run_projection(reporter=reporter)

        return 0
    except Exception as exc:
        print(
            f"ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
