#!/usr/bin/env python3
"""Production Week 1 NFL projection using market-independent v4 models."""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
NFL_ROOT = SCRIPT_PATH.parents[2]
REPORT_ROOT = NFL_ROOT / "errors"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_reporter import PipelineReporter
from projection_common import (
    load_reported_compatibility_schema,
    produce_and_publish_projection,
)

SEASON = 2026
WEEK = 1
SCRIPT_VERSION = "2026-09-21-v4-production-hardened"
EXPECTED_FEATURE_COUNT = 260
MODEL_NAME = "v4_market_independent_outcomes"


def run(reporter: PipelineReporter) -> None:
    try:
        import projection_feature_builder_legacy_week1 as helper
        from v4_production import OUTPUT_COLUMNS, apply_v4_production_models
    except Exception:
        reporter.set_detail("dependency_imports_ok", False)
        raise

    reporter.set_detail("dependency_imports_ok", True)

    root = helper.nfl_root()
    if root.resolve() != NFL_ROOT.resolve():
        raise RuntimeError(
            f"Resolved NFL root mismatch: helper={root} script={NFL_ROOT}"
        )

    combined_path = (
        root
        / "00_intake/predictions/enriched/combined/week_1_NFL_enriched.csv"
    )
    output_path = root / "01_merge/week_1_NFL_enriched.csv"

    reporter.add_input(combined_path)
    reporter.add_output(output_path)
    reporter.update_details(
        {
            "model": MODEL_NAME,
            "script_version": SCRIPT_VERSION,
            "expected_feature_count": EXPECTED_FEATURE_COUNT,
            "output_prediction_columns": list(OUTPUT_COLUMNS),
            "staged_roundtrip_verified": False,
            "compatibility_schema_validated": False,
        }
    )

    original = helper.read_csv(combined_path)
    source_rows = len(original)
    reporter.set_rows(rows_in=source_rows)

    helper.validate_week1_base(
        original,
        SEASON,
        str(combined_path),
    )

    eligible_rows = len(original)
    reporter.set_detail(
        "started_games_skipped",
        source_rows - eligible_rows,
    )
    reporter.set_detail("eligible_games", eligible_rows)

    collisions = [
        column
        for column in OUTPUT_COLUMNS
        if column in original.columns
    ]
    if collisions:
        raise ValueError(
            f"{combined_path}: prediction columns already exist and "
            f"would be overwritten: {collisions}"
        )

    compatibility_schema = load_reported_compatibility_schema(
        root,
        reporter,
        expected_feature_count=EXPECTED_FEATURE_COUNT,
    )

    full_features = helper.prepare_model_features(
        root,
        original.copy(),
        compatibility_schema,
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
        stage_prefix=".projection_week1_stage_",
        stage_error_label="Staged Week 1 projection",
        rollback_failure_label=(
            "Week 1 projection publication failed and rollback "
            "also failed"
        ),
        backup_cleanup_warning=(
            "Week 1 projection published but temporary backup "
            "cleanup failed"
        ),
        cleanup_backup_if_unpublished=False,
    )
    reporter.set_detail("output_columns", len(projected.columns))

    print(
        f"WROTE {output_path} | games={len(projected)} | "
        f"model={MODEL_NAME} | columns={len(projected.columns)}"
    )


def main() -> int:
    print(f"projection_week1.py version={SCRIPT_VERSION}")

    try:
        with PipelineReporter(
            script=SCRIPT_PATH,
            stage="01_merge",
            report_root=REPORT_ROOT,
            pipeline="NFL",
            league="NFL",
            season=SEASON,
            week=WEEK,
            extra_context={
                "component": "Week 1 production projection",
                "model": MODEL_NAME,
            },
        ) as reporter:
            run(reporter)
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
