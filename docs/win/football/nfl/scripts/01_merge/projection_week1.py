#!/usr/bin/env python3
"""Production Week 1 NFL projection using market-independent v4 models."""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
NFL_ROOT = SCRIPT_PATH.parents[2]
REPORT_ROOT = NFL_ROOT / "errors"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_reporter import PipelineReporter

SEASON = 2026
WEEK = 1
SCRIPT_VERSION = "2026-09-21-v4-production-hardened"
EXPECTED_FEATURE_COUNT = 260
MODEL_NAME = "v4_market_independent_outcomes"


def load_compatibility_schema(root: Path) -> tuple[dict[str, Any], Path]:
    path = root / "models/archive/legacy_260_feature_model/step11_feature_schema.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing feature-builder compatibility schema: {path}"
        )

    with path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)

    if not isinstance(schema, dict):
        raise ValueError(f"{path}: schema root must be a JSON object")

    required_keys = (
        "feature_order",
        "numeric_features",
        "categorical_features",
    )
    missing = [key for key in required_keys if key not in schema]
    if missing:
        raise ValueError(f"{path}: missing schema keys: {missing}")

    feature_order = schema["feature_order"]
    numeric_features = schema["numeric_features"]
    categorical_features = schema["categorical_features"]

    for key, values in (
        ("feature_order", feature_order),
        ("numeric_features", numeric_features),
        ("categorical_features", categorical_features),
    ):
        if not isinstance(values, list):
            raise ValueError(f"{path}: {key} must be a JSON array")
        if not all(isinstance(value, str) and value.strip() for value in values):
            raise ValueError(f"{path}: {key} contains a blank or non-string feature name")
        if len(values) != len(set(values)):
            raise ValueError(f"{path}: duplicate names in {key}")

    if len(feature_order) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"{path}: expected {EXPECTED_FEATURE_COUNT} features; "
            f"found {len(feature_order)}"
        )

    numeric_set = set(numeric_features)
    categorical_set = set(categorical_features)
    feature_set = set(feature_order)

    overlap = sorted(numeric_set & categorical_set)
    if overlap:
        raise ValueError(
            f"{path}: features classified as both numeric and categorical: "
            f"{overlap}"
        )

    if numeric_set | categorical_set != feature_set:
        missing_partition = sorted(feature_set - (numeric_set | categorical_set))
        extra_partition = sorted((numeric_set | categorical_set) - feature_set)
        raise ValueError(
            f"{path}: numeric/categorical features do not exactly cover "
            f"feature_order; missing={missing_partition} extra={extra_partition}"
        )

    return schema, path


def _first_mismatch(expected: list[str], actual: list[str]) -> int | None:
    for index, (expected_value, actual_value) in enumerate(
        zip(expected, actual, strict=True)
    ):
        if expected_value != actual_value:
            return index
    return None


def validate_serialized_output(
    path: Path,
    *,
    projected: Any,
    original: Any,
    output_columns: list[str],
    helper: Any,
) -> None:
    staged = helper.read_csv(path)
    expected_columns = [*original.columns.tolist(), *output_columns]

    if staged.columns.tolist() != expected_columns:
        raise RuntimeError(
            f"{path}: serialized output columns differ from the exact "
            "input-plus-production-predictions contract"
        )

    if len(staged) != len(projected):
        raise RuntimeError(
            f"{path}: serialized row count {len(staged)} differs from "
            f"validated in-memory row count {len(projected)}"
        )

    helper.require_unique_game_id(staged, str(path))

    if staged["game_id"].tolist() != original["game_id"].tolist():
        raise RuntimeError(f"{path}: serialized game_id row order changed")

    if staged["home_team"].tolist() != original["home_team"].tolist():
        raise RuntimeError(f"{path}: serialized home_team values changed")

    if staged["away_team"].tolist() != original["away_team"].tolist():
        raise RuntimeError(f"{path}: serialized away_team values changed")

    for column in original.columns:
        expected_values = [
            helper.clean(value)
            for value in projected[column].tolist()
        ]
        actual_values = [
            helper.clean(value)
            for value in staged[column].tolist()
        ]
        mismatch = _first_mismatch(expected_values, actual_values)
        if mismatch is not None:
            game_id = staged.iloc[mismatch]["game_id"]
            raise RuntimeError(
                f"{path}: serialized source column {column!r} changed "
                f"for game_id={game_id}"
            )

    numeric_values: dict[str, list[float]] = {}
    for column in output_columns:
        values: list[float] = []
        for row_index, value in enumerate(staged[column].tolist()):
            number = helper.parse_float(value)
            if number is None:
                game_id = staged.iloc[row_index]["game_id"]
                raise RuntimeError(
                    f"{path}: serialized prediction {column!r} is "
                    f"blank/non-finite for game_id={game_id}"
                )
            values.append(number)
        numeric_values[column] = values

        expected_numbers = [
            float(value)
            for value in projected[column].tolist()
        ]
        for row_index, (expected_value, actual_value) in enumerate(
            zip(expected_numbers, values, strict=True)
        ):
            if not math.isclose(
                expected_value,
                actual_value,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                game_id = staged.iloc[row_index]["game_id"]
                raise RuntimeError(
                    f"{path}: serialized prediction {column!r} differs "
                    f"from validated in-memory value for game_id={game_id}"
                )

    for first, second, label in (
        ("home_win_probability", "away_win_probability", "moneyline"),
        ("home_cover_probability", "away_cover_probability", "spread"),
        ("over_probability", "under_probability", "total"),
    ):
        for row_index, (first_value, second_value) in enumerate(
            zip(
                numeric_values[first],
                numeric_values[second],
                strict=True,
            )
        ):
            game_id = staged.iloc[row_index]["game_id"]
            if not (
                0.0 <= first_value <= 1.0
                and 0.0 <= second_value <= 1.0
            ):
                raise RuntimeError(
                    f"{path}: serialized {label} probabilities outside "
                    f"[0, 1] for game_id={game_id}"
                )
            if not math.isclose(
                first_value + second_value,
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RuntimeError(
                    f"{path}: serialized {label} probabilities do not "
                    f"sum to 1 for game_id={game_id}"
                )


def stage_output(output_dir: Path, projected: Any) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=".projection_week1_stage_",
        suffix=".csv",
        dir=str(output_dir),
    )
    os.close(descriptor)
    staged_path = Path(raw_path)

    try:
        projected.to_csv(
            staged_path,
            index=False,
            encoding="utf-8-sig",
        )
        if not staged_path.is_file() or staged_path.stat().st_size == 0:
            raise RuntimeError(
                f"Staged Week 1 projection was not written correctly: "
                f"{staged_path}"
            )
        return staged_path
    except Exception:
        staged_path.unlink(missing_ok=True)
        raise


def publish_staged_output(
    staged_path: Path,
    output_path: Path,
    *,
    projected: Any,
    original: Any,
    output_columns: list[str],
    helper: Any,
    reporter: PipelineReporter,
) -> None:
    backup_path = output_path.parent / (
        f".{output_path.name}.backup.{uuid.uuid4().hex}"
    )
    had_existing_output = output_path.exists()
    published = False

    reporter.update_details(
        {
            "publication_mode": "atomic_replace_with_backup_rollback",
            "publication_completed": False,
            "post_publish_validation": False,
            "rollback_performed": False,
        }
    )

    try:
        if had_existing_output:
            shutil.copy2(output_path, backup_path)

        os.replace(staged_path, output_path)
        published = True

        validate_serialized_output(
            output_path,
            projected=projected,
            original=original,
            output_columns=output_columns,
            helper=helper,
        )

        reporter.update_details(
            {
                "publication_completed": True,
                "post_publish_validation": True,
            }
        )
    except Exception as publish_exc:
        if published:
            try:
                if had_existing_output and backup_path.exists():
                    os.replace(backup_path, output_path)
                elif not had_existing_output and output_path.exists():
                    output_path.unlink()

                reporter.update_details(
                    {
                        "publication_completed": False,
                        "post_publish_validation": False,
                        "rollback_performed": True,
                    }
                )
            except Exception as rollback_exc:
                reporter.update_details(
                    {
                        "publication_completed": False,
                        "post_publish_validation": False,
                        "rollback_performed": False,
                        "rollback_error_type": type(rollback_exc).__name__,
                        "rollback_error": str(rollback_exc),
                    }
                )
                raise RuntimeError(
                    "Week 1 projection publication failed and rollback "
                    f"also failed: publication_error={publish_exc}; "
                    f"rollback_error={rollback_exc}"
                ) from rollback_exc
        raise
    finally:
        staged_path.unlink(missing_ok=True)

    if backup_path.exists():
        try:
            backup_path.unlink()
        except Exception as exc:
            reporter.warning(
                "Week 1 projection published but temporary backup cleanup failed",
                backup_path=str(backup_path),
                error_type=type(exc).__name__,
                error=str(exc),
            )


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

    compatibility_schema, compatibility_path = load_compatibility_schema(root)
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

    full_features = helper.prepare_model_features(
        root,
        original.copy(),
        compatibility_schema,
    )
    projected = apply_v4_production_models(
        root,
        original,
        full_features,
    )

    reporter.set_detail("in_memory_output_validated", True)

    staged_path = stage_output(
        output_path.parent,
        projected,
    )

    try:
        validate_serialized_output(
            staged_path,
            projected=projected,
            original=original,
            output_columns=list(OUTPUT_COLUMNS),
            helper=helper,
        )
        reporter.set_detail("staged_roundtrip_verified", True)

        publish_staged_output(
            staged_path,
            output_path,
            projected=projected,
            original=original,
            output_columns=list(OUTPUT_COLUMNS),
            helper=helper,
            reporter=reporter,
        )
    finally:
        staged_path.unlink(missing_ok=True)

    reporter.set_rows(rows_out=len(projected))
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
