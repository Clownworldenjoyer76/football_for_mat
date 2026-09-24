from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable


def load_compatibility_schema(
    root: Path,
    *,
    expected_feature_count: int,
) -> tuple[dict[str, Any], Path]:
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

        if not all(
            isinstance(value, str) and value.strip()
            for value in values
        ):
            raise ValueError(
                f"{path}: {key} contains a blank or non-string feature name"
            )

        if len(values) != len(set(values)):
            raise ValueError(f"{path}: duplicate names in {key}")

    if len(feature_order) != expected_feature_count:
        raise ValueError(
            f"{path}: expected {expected_feature_count} features; "
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
        missing_partition = sorted(
            feature_set - (numeric_set | categorical_set)
        )
        extra_partition = sorted(
            (numeric_set | categorical_set) - feature_set
        )
        raise ValueError(
            f"{path}: numeric/categorical features do not exactly cover "
            f"feature_order; missing={missing_partition} "
            f"extra={extra_partition}"
        )

    return schema, path

def load_reported_compatibility_schema(
    root: Path,
    reporter: Any,
    *,
    expected_feature_count: int,
) -> dict[str, Any]:
    schema, path = load_compatibility_schema(
        root,
        expected_feature_count=expected_feature_count,
    )
    reporter.add_input(path)
    reporter.update_details(
        {
            "compatibility_schema_validated": True,
            "compatibility_schema_features": len(
                schema["feature_order"]
            ),
            "compatibility_schema_numeric_features": len(
                schema["numeric_features"]
            ),
            "compatibility_schema_categorical_features": len(
                schema["categorical_features"]
            ),
        }
    )
    return schema


def _first_mismatch(
    expected: list[str],
    actual: list[str],
) -> int | None:
    for index, (expected_value, actual_value) in enumerate(
        zip(expected, actual, strict=True)
    ):
        if expected_value != actual_value:
            return index

    return None


def _validate_serialized_output(
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
        raise RuntimeError(
            f"{path}: serialized game_id row order changed"
        )

    if staged["home_team"].tolist() != original["home_team"].tolist():
        raise RuntimeError(
            f"{path}: serialized home_team values changed"
        )

    if staged["away_team"].tolist() != original["away_team"].tolist():
        raise RuntimeError(
            f"{path}: serialized away_team values changed"
        )

    for column in original.columns:
        expected_values = [
            helper.clean(value)
            for value in projected[column].tolist()
        ]
        actual_values = [
            helper.clean(value)
            for value in staged[column].tolist()
        ]

        mismatch = _first_mismatch(
            expected_values,
            actual_values,
        )
        if mismatch is not None:
            game_id = staged.iloc[mismatch]["game_id"]
            raise RuntimeError(
                f"{path}: serialized source column {column!r} changed "
                f"for game_id={game_id}"
            )

    numeric_values: dict[str, list[float]] = {}

    for column in output_columns:
        values: list[float] = []

        for row_index, value in enumerate(
            staged[column].tolist()
        ):
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

        for row_index, (
            expected_value,
            actual_value,
        ) in enumerate(
            zip(
                expected_numbers,
                values,
                strict=True,
            )
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
                    "from validated in-memory value "
                    f"for game_id={game_id}"
                )

    for first, second, label in (
        (
            "home_win_probability",
            "away_win_probability",
            "moneyline",
        ),
        (
            "home_cover_probability",
            "away_cover_probability",
            "spread",
        ),
        (
            "over_probability",
            "under_probability",
            "total",
        ),
    ):
        for row_index, (
            first_value,
            second_value,
        ) in enumerate(
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


def _stage_output(
    output_dir: Path,
    projected: Any,
    *,
    prefix: str,
    error_label: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    descriptor, raw_path = tempfile.mkstemp(
        prefix=prefix,
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

        if (
            not staged_path.is_file()
            or staged_path.stat().st_size == 0
        ):
            raise RuntimeError(
                f"{error_label} was not written correctly: "
                f"{staged_path}"
            )

        return staged_path
    except Exception:
        staged_path.unlink(missing_ok=True)
        raise


def _publish_staged_output(
    staged_path: Path,
    output_path: Path,
    *,
    projected: Any,
    original: Any,
    output_columns: list[str],
    helper: Any,
    reporter: Any,
    rollback_failure_label: str,
    backup_cleanup_warning: str,
    cleanup_backup_if_unpublished: bool,
) -> None:
    backup_path = output_path.parent / (
        f".{output_path.name}.backup.{uuid.uuid4().hex}"
    )
    had_existing_output = output_path.exists()
    published = False
    rollback_failed = False

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
            shutil.copy2(
                output_path,
                backup_path,
            )

        os.replace(
            staged_path,
            output_path,
        )
        published = True

        _validate_serialized_output(
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
                if (
                    had_existing_output
                    and backup_path.exists()
                ):
                    os.replace(
                        backup_path,
                        output_path,
                    )
                elif (
                    not had_existing_output
                    and output_path.exists()
                ):
                    output_path.unlink()

                reporter.update_details(
                    {
                        "publication_completed": False,
                        "post_publish_validation": False,
                        "rollback_performed": True,
                    }
                )
            except Exception as rollback_exc:
                rollback_failed = True
                reporter.update_details(
                    {
                        "publication_completed": False,
                        "post_publish_validation": False,
                        "rollback_performed": False,
                        "rollback_error_type": type(
                            rollback_exc
                        ).__name__,
                        "rollback_error": str(
                            rollback_exc
                        ),
                    }
                )
                raise RuntimeError(
                    f"{rollback_failure_label}: "
                    f"publication_error={publish_exc}; "
                    f"rollback_error={rollback_exc}"
                ) from rollback_exc

        raise
    finally:
        staged_path.unlink(missing_ok=True)

        should_cleanup_backup = (
            cleanup_backup_if_unpublished
            and not published
        ) or (
            published
            and output_path.exists()
        )

        if (
            backup_path.exists()
            and not rollback_failed
            and should_cleanup_backup
        ):
            try:
                backup_path.unlink()
            except Exception as exc:
                reporter.warning(
                    backup_cleanup_warning,
                    backup_path=str(backup_path),
                    error_type=type(exc).__name__,
                    error=str(exc),
                )


def produce_and_publish_projection(
    *,
    root: Path,
    original: Any,
    full_features: Any,
    apply_models: Callable[[Path, Any, Any], Any],
    output_path: Path,
    output_columns: list[str],
    helper: Any,
    reporter: Any,
    stage_prefix: str,
    stage_error_label: str,
    rollback_failure_label: str,
    backup_cleanup_warning: str,
    cleanup_backup_if_unpublished: bool,
) -> Any:
    projected = apply_models(
        root,
        original,
        full_features,
    )

    reporter.set_detail(
        "in_memory_output_validated",
        True,
    )

    staged_path = _stage_output(
        output_path.parent,
        projected,
        prefix=stage_prefix,
        error_label=stage_error_label,
    )

    try:
        _validate_serialized_output(
            staged_path,
            projected=projected,
            original=original,
            output_columns=output_columns,
            helper=helper,
        )
        reporter.set_detail(
            "staged_roundtrip_verified",
            True,
        )

        _publish_staged_output(
            staged_path,
            output_path,
            projected=projected,
            original=original,
            output_columns=output_columns,
            helper=helper,
            reporter=reporter,
            rollback_failure_label=rollback_failure_label,
            backup_cleanup_warning=backup_cleanup_warning,
            cleanup_backup_if_unpublished=(
                cleanup_backup_if_unpublished
            ),
        )
    finally:
        staged_path.unlink(missing_ok=True)

    reporter.set_rows(
        rows_out=len(projected),
    )

    return projected
