#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PROP = Path(__file__).resolve().parent

TARGET = "rushing_tds"
VALIDATION_SEASON = 2024
TEST_SEASON = 2025
GRAIN = ["season", "week", "game_id", "player_id"]

AUDIT = PROP / "evaluation/model_selection_predictions.parquet"
HURDLE = PROP / "evaluation/issue56_rushing_tds_hurdle_predictions.csv"
CALIBRATION = PROP / "models/calibration/rushing_tds_calibration.json"
SELECTED = PROP / "models/rushing_tds/selected_model.json"
THRESHOLDS = PROP / "config/acceptance_thresholds.yaml"

OUT_CSV = (
    PROP
    / "evaluation/issue56_rushing_tds_component_hurdle_blend_diagnostic.csv"
)
OUT_JSON = (
    PROP
    / "evaluation/issue56_rushing_tds_component_hurdle_blend_diagnostic.json"
)
FREEZE_JSON = (
    PROP
    / "evaluation/issue56_rushing_tds_component_hurdle_blend_freeze.json"
)

HURDLE_COLUMNS = [
    "unit_positive_severity_expected_count",
    "training_positive_mean_severity_expected_count",
]
WEIGHTS = np.arange(0.0, 1.0001, 0.001, dtype="float64")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise AssertionError(f"Expected JSON object: {path}")
    return value


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise AssertionError(f"Expected YAML mapping: {path}")
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def normalize_grain(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    out = frame.copy()
    for column in GRAIN:
        if column not in out.columns:
            raise AssertionError(f"{label}: missing grain column {column}")

    out["season"] = pd.to_numeric(
        out["season"],
        errors="raise",
    ).astype("int64")
    out["week"] = pd.to_numeric(
        out["week"],
        errors="raise",
    ).astype("int64")
    out["game_id"] = out["game_id"].astype("string").str.strip()
    out["player_id"] = out["player_id"].astype("string").str.strip()

    if out[["game_id", "player_id"]].isna().any().any():
        raise AssertionError(f"{label}: null canonical key")
    if out["game_id"].eq("").any() or out["player_id"].eq("").any():
        raise AssertionError(f"{label}: blank canonical key")
    if out.duplicated(GRAIN).any():
        sample = (
            out.loc[out.duplicated(GRAIN, keep=False), GRAIN]
            .head(10)
            .to_dict(orient="records")
        )
        raise AssertionError(
            f"{label}: duplicate canonical grain; sample={sample}"
        )
    return out


def read_audit_split(split: str, season: int) -> pd.DataFrame:
    columns = [
        "split",
        *GRAIN,
        "target",
        "actual",
        "baseline_projection",
        "component_projection",
    ]
    try:
        frame = pd.read_parquet(
            AUDIT,
            columns=columns,
            filters=[
                ("target", "==", TARGET),
                ("split", "==", split),
            ],
        )
    except Exception:
        frame = pd.read_parquet(AUDIT, columns=columns)
        frame = frame.loc[
            frame["target"].astype(str).eq(TARGET)
            & frame["split"].astype(str).eq(split)
        ].copy()

    frame = frame.loc[
        frame["target"].astype(str).eq(TARGET)
        & frame["split"].astype(str).eq(split)
    ].copy()
    frame = normalize_grain(frame, f"audit_{split}")

    found = set(frame["season"].unique().tolist())
    if found != {season}:
        raise AssertionError(
            f"audit_{split}: expected season={season}; found={sorted(found)}"
        )
    if frame.empty:
        raise AssertionError(f"audit_{split}: no rows")

    for column in [
        "actual",
        "baseline_projection",
        "component_projection",
    ]:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any():
            raise AssertionError(
                f"audit_{split}: nonnumeric/null {column}"
            )
        frame[column] = values.astype("float64")

    return frame.reset_index(drop=True)


def hurdle_columns() -> list[str]:
    columns = list(pd.read_csv(HURDLE, nrows=0).columns)
    required = {
        "split_label",
        *GRAIN,
        "actual",
        "hurdle_p1",
        *HURDLE_COLUMNS,
    }
    missing = sorted(required - set(columns))
    if missing:
        raise AssertionError(
            f"Hurdle prediction CSV missing columns: {missing}"
        )
    return columns


def read_hurdle_split(label: str, season: int) -> pd.DataFrame:
    available = hurdle_columns()
    usecols = [
        "split_label",
        *GRAIN,
        "actual",
        "hurdle_p1",
        *HURDLE_COLUMNS,
    ]
    if "baseline_projection" in available:
        usecols.append("baseline_projection")

    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        HURDLE,
        usecols=usecols,
        chunksize=100_000,
        dtype={
            "game_id": "string",
            "player_id": "string",
            "split_label": "string",
        },
    ):
        selected = chunk.loc[
            chunk["split_label"].astype(str).eq(label)
        ].copy()
        if not selected.empty:
            chunks.append(selected)

    if not chunks:
        raise AssertionError(f"hurdle_{label}: no rows")

    frame = pd.concat(chunks, ignore_index=True)
    frame = normalize_grain(frame, f"hurdle_{label}")
    found = set(frame["season"].unique().tolist())
    if found != {season}:
        raise AssertionError(
            f"hurdle_{label}: expected season={season}; "
            f"found={sorted(found)}"
        )

    numeric_columns = [
        "actual",
        "hurdle_p1",
        *HURDLE_COLUMNS,
    ]
    if "baseline_projection" in frame.columns:
        numeric_columns.append("baseline_projection")

    for column in numeric_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any():
            raise AssertionError(
                f"hurdle_{label}: nonnumeric/null {column}"
            )
        frame[column] = values.astype("float64")

    return frame.reset_index(drop=True)


def reconcile(
    audit: pd.DataFrame,
    hurdle: pd.DataFrame,
    label: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    akeys = audit[GRAIN].copy()
    hkeys = hurdle[GRAIN].copy()
    key_check = akeys.merge(
        hkeys,
        on=GRAIN,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    missing_hurdle = int(key_check["_merge"].eq("left_only").sum())
    extra_hurdle = int(key_check["_merge"].eq("right_only").sum())
    if missing_hurdle or extra_hurdle:
        raise AssertionError(
            f"{label}: canonical grain mismatch; "
            f"missing_hurdle={missing_hurdle} "
            f"extra_hurdle={extra_hurdle}"
        )

    hurdle_columns_for_join = [
        *GRAIN,
        "actual",
        "hurdle_p1",
        *HURDLE_COLUMNS,
    ]
    if "baseline_projection" in hurdle.columns:
        hurdle_columns_for_join.append("baseline_projection")

    rename = {
        "actual": "hurdle_actual",
    }
    if "baseline_projection" in hurdle.columns:
        rename["baseline_projection"] = "copied_hurdle_baseline"

    merged = audit.merge(
        hurdle[hurdle_columns_for_join].rename(columns=rename),
        on=GRAIN,
        how="inner",
        validate="one_to_one",
    )

    canonical_actual = merged["actual"].to_numpy(dtype="float64")
    hurdle_actual = merged["hurdle_actual"].to_numpy(dtype="float64")
    if not np.array_equal(canonical_actual, hurdle_actual):
        diff = np.abs(canonical_actual - hurdle_actual)
        bad = np.flatnonzero(diff != 0.0)
        sample = (
            merged.iloc[
                bad[:10]
            ][
                GRAIN + ["actual", "hurdle_actual"]
            ].to_dict(orient="records")
        )
        raise AssertionError(
            f"{label}: actual-value reconciliation failed; "
            f"mismatches={len(bad)} sample={sample}"
        )

    info: dict[str, Any] = {
        "rows": int(len(merged)),
        "canonical_grain_exact": True,
        "actual_exact_match": True,
        "baseline_source": "evaluation/model_selection_predictions.parquet",
        "copied_hurdle_baseline_used_for_metrics": False,
    }

    if "copied_hurdle_baseline" in merged.columns:
        canonical_baseline = merged["baseline_projection"].to_numpy(
            dtype="float64"
        )
        copied = merged["copied_hurdle_baseline"].to_numpy(
            dtype="float64"
        )
        mismatch = canonical_baseline != copied
        info["copied_hurdle_baseline_present"] = True
        info["copied_hurdle_baseline_mismatch_rows"] = int(
            mismatch.sum()
        )
        info["copied_hurdle_baseline_max_abs_difference"] = (
            float(np.max(np.abs(canonical_baseline - copied)))
            if len(canonical_baseline)
            else 0.0
        )
    else:
        info["copied_hurdle_baseline_present"] = False
        info["copied_hurdle_baseline_mismatch_rows"] = None
        info["copied_hurdle_baseline_max_abs_difference"] = None

    return merged.reset_index(drop=True), info


def apply_mapping(values: np.ndarray, mapping: dict[str, Any]) -> np.ndarray:
    xp = np.asarray(mapping["knots_x"], dtype="float64")
    fp = np.asarray(mapping["knots_y"], dtype="float64")
    if len(xp) == 0 or len(xp) != len(fp):
        raise AssertionError("Invalid calibration mapping")

    out = np.interp(
        np.asarray(values, dtype="float64"),
        xp,
        fp,
        left=float(mapping["left_value"]),
        right=float(mapping["right_value"]),
    )
    bounds = mapping.get("output_bounds", [None, None])
    if bounds[0] is not None:
        out = np.maximum(out, float(bounds[0]))
    if bounds[1] is not None:
        out = np.minimum(out, float(bounds[1]))
    return out


def frozen_probability_path(
    raw_component: np.ndarray,
    calibration: dict[str, Any],
) -> np.ndarray:
    count = calibration.get("count_calibration", {})
    expected_map = count.get("expected_count", {}).get("mapping")
    p1_map = count.get("probability_1_plus", {}).get("mapping")
    if not isinstance(expected_map, dict) or not isinstance(p1_map, dict):
        raise AssertionError("Missing rushing_tds count-calibration mapping")

    expected = np.maximum(
        apply_mapping(
            np.maximum(
                np.asarray(raw_component, dtype="float64"),
                0.0,
            ),
            expected_map,
        ),
        0.0,
    )
    poisson_p1 = 1.0 - np.exp(-expected)
    return np.clip(
        apply_mapping(poisson_p1, p1_map),
        0.0,
        1.0,
    )


def metrics(
    actual: np.ndarray,
    prediction: np.ndarray,
    canonical_baseline: np.ndarray,
    probability_1plus: np.ndarray,
) -> dict[str, float | int]:
    y = np.asarray(actual, dtype="float64")
    p = np.maximum(
        np.asarray(prediction, dtype="float64"),
        0.0,
    )
    b = np.asarray(canonical_baseline, dtype="float64")
    q = np.clip(
        np.asarray(probability_1plus, dtype="float64"),
        0.0,
        1.0,
    )

    valid = (
        np.isfinite(y)
        & np.isfinite(p)
        & np.isfinite(b)
        & np.isfinite(q)
    )
    if not valid.all():
        raise AssertionError(
            f"Nonfinite metric rows={int((~valid).sum())}"
        )

    mae = float(np.mean(np.abs(y - p)))
    bias = float(np.mean(p - y))
    baseline_mae = float(np.mean(np.abs(y - b)))
    if baseline_mae <= 0.0:
        raise AssertionError("Canonical baseline MAE must be positive")
    improvement = (
        100.0
        * (baseline_mae - mae)
        / baseline_mae
    )

    lam = np.maximum(p, 1e-12)
    terms = np.empty_like(y)
    zero = y <= 0.0
    terms[zero] = lam[zero]
    positive = ~zero
    terms[positive] = (
        y[positive] * np.log(y[positive] / lam[positive])
        - (y[positive] - lam[positive])
    )

    event = (y >= 1.0).astype("float64")
    return {
        "rows": int(len(y)),
        "mae": mae,
        "bias": bias,
        "abs_bias": abs(bias),
        "baseline_mae": baseline_mae,
        "improvement_vs_baseline_pct": float(improvement),
        "brier_1plus": float(np.mean(np.square(q - event))),
        "poisson_deviance": float(2.0 * np.mean(terms)),
        "mean_actual": float(np.mean(y)),
        "mean_prediction": float(np.mean(p)),
    }


def failures(
    metric: dict[str, float | int],
    threshold: dict[str, Any],
) -> list[str]:
    failed: list[str] = []
    if float(metric["mae"]) > float(
        threshold["maximum_validation_mae"]
    ):
        failed.append("mae")
    if float(metric["abs_bias"]) > float(
        threshold["maximum_allowed_bias"]
    ):
        failed.append("bias")
    if float(metric["improvement_vs_baseline_pct"]) < float(
        threshold["minimum_improvement_vs_baseline_pct"]
    ):
        failed.append("improvement")
    if float(metric["brier_1plus"]) > float(
        threshold["maximum_brier_1plus"]
    ):
        failed.append("brier_1plus")
    if float(metric["poisson_deviance"]) > float(
        threshold["maximum_poisson_deviance"]
    ):
        failed.append("poisson_deviance")
    return failed


def validate_frozen_contract() -> tuple[dict[str, Any], dict[str, Any]]:
    for path in [
        AUDIT,
        HURDLE,
        CALIBRATION,
        SELECTED,
        THRESHOLDS,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    selected = load_json(SELECTED)
    if selected.get("selected_architecture") != "component":
        raise AssertionError(
            "Expected current rushing_tds architecture=component "
            f"before this diagnostic; found="
            f"{selected.get('selected_architecture')}"
        )
    if int(selected.get("validation_season", -1)) != VALIDATION_SEASON:
        raise AssertionError("selected_model validation season mismatch")
    if int(selected.get("test_season", -1)) != TEST_SEASON:
        raise AssertionError("selected_model test season mismatch")
    if selected.get("test_used_for_selection") is not False:
        raise AssertionError(
            "selected_model test_used_for_selection must be false"
        )
    if selected.get("market_features_used") is not False:
        raise AssertionError(
            "selected_model market_features_used must be false"
        )

    calibration = load_json(CALIBRATION)
    source = calibration.get("calibration_source", {})
    reporting = calibration.get("reporting_test_policy", {})
    if int(source.get("season", -1)) != VALIDATION_SEASON:
        raise AssertionError(
            "rushing_tds calibration source must be 2024"
        )
    if reporting.get("test_rows_used_for_calibration") is not False:
        raise AssertionError("calibration used 2025 rows")
    if calibration.get("market_features_used") is not False:
        raise AssertionError("calibration market_features_used must be false")

    thresholds = load_yaml(THRESHOLDS)
    threshold = thresholds.get(TARGET)
    if not isinstance(threshold, dict):
        raise AssertionError("Missing rushing_tds acceptance thresholds")

    required_thresholds = [
        "maximum_allowed_bias",
        "maximum_validation_mae",
        "minimum_improvement_vs_baseline_pct",
        "maximum_brier_1plus",
        "maximum_poisson_deviance",
    ]
    missing = [
        key for key in required_thresholds
        if key not in threshold
    ]
    if missing:
        raise AssertionError(
            f"Missing locked rushing_tds gates: {missing}"
        )

    return calibration, dict(threshold)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preflight-only",
        action="store_true",
    )
    args = parser.parse_args()

    calibration, threshold = validate_frozen_contract()

    # Remove a stale freeze artifact before starting a new 2024 search.
    if FREEZE_JSON.exists():
        FREEZE_JSON.unlink()

    # SELECTION PHASE: 2024 ONLY.
    validation_audit = read_audit_split(
        "validation",
        VALIDATION_SEASON,
    )
    validation_hurdle = read_hurdle_split(
        "validation_2024",
        VALIDATION_SEASON,
    )
    validation, reconciliation_2024 = reconcile(
        validation_audit,
        validation_hurdle,
        "validation_2024",
    )

    if args.preflight_only:
        print("RUSHING TDS HURDLE/AUDIT RECONCILIATION: PASS")
        print(
            json.dumps(
                {
                    "validation_2024": reconciliation_2024,
                    "canonical_baseline_used": True,
                    "test_metrics_read": False,
                },
                sort_keys=True,
            )
        )
        return 0

    vy = validation["actual"].to_numpy(dtype="float64")
    vb = validation["baseline_projection"].to_numpy(dtype="float64")
    raw_component = validation["component_projection"].to_numpy(
        dtype="float64"
    )
    vp1 = frozen_probability_path(
        raw_component,
        calibration,
    )

    rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    for hurdle_column in HURDLE_COLUMNS:
        hurdle_expected = validation[hurdle_column].to_numpy(
            dtype="float64"
        )

        for weight_value in WEIGHTS:
            weight = float(weight_value)
            point = (
                (1.0 - weight) * raw_component
                + weight * hurdle_expected
            )
            metric = metrics(vy, point, vb, vp1)
            failed = failures(metric, threshold)
            candidate = {
                "hurdle_path": hurdle_column,
                "hurdle_weight": weight,
                "raw_component_weight": 1.0 - weight,
                "validation_metrics": metric,
                "validation_failed": failed,
                "validation_pass": not failed,
            }
            candidates.append(candidate)
            rows.append(
                {
                    "hurdle_path": hurdle_column,
                    "hurdle_weight": weight,
                    "raw_component_weight": 1.0 - weight,
                    "validation_mae": metric["mae"],
                    "validation_bias": metric["bias"],
                    "validation_abs_bias": metric["abs_bias"],
                    "validation_baseline_mae": metric["baseline_mae"],
                    "validation_improvement_pct": (
                        metric["improvement_vs_baseline_pct"]
                    ),
                    "validation_brier_1plus": metric["brier_1plus"],
                    "validation_poisson_deviance": (
                        metric["poisson_deviance"]
                    ),
                    "validation_failed": ";".join(failed),
                    "validation_pass": not failed,
                }
            )

    table = pd.DataFrame(rows)
    table = table.sort_values(
        [
            "validation_pass",
            "validation_mae",
            "validation_poisson_deviance",
            "validation_abs_bias",
            "hurdle_path",
            "hurdle_weight",
        ],
        ascending=[False, True, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    table.to_csv(OUT_CSV, index=False, lineterminator="\n")

    passing = [candidate for candidate in candidates if candidate["validation_pass"]]

    base_payload: dict[str, Any] = {
        "status": "complete",
        "target": TARGET,
        "diagnostic_only": True,
        "production_files_modified": False,
        "market_data_used": False,
        "canonical_grain": GRAIN,
        "actual_reconciliation_required_exact": True,
        "canonical_baseline_source": (
            "evaluation/model_selection_predictions.parquet"
        ),
        "copied_hurdle_baseline_used_for_metrics": False,
        "selection_season": VALIDATION_SEASON,
        "reporting_season": TEST_SEASON,
        "test_used_for_selection": False,
        "thresholds_unchanged": threshold,
        "validation_reconciliation": reconciliation_2024,
        "candidate_count": int(len(table)),
        "candidate_family": (
            "raw_component/hurdle_expected_count_convex_blend"
        ),
        "probability_screening_policy": (
            "existing_frozen_2024_component_probability_path; "
            "final uncertainty calibration must be rerun after repair"
        ),
        "candidate_table": str(OUT_CSV),
        "freeze_artifact": str(FREEZE_JSON),
    }

    if not passing:
        payload = {
            **base_payload,
            "selection_status": "no_candidate_passed_2024_gates",
            "blend_family_decision": "STOP",
            "test_2025_metrics_read": False,
            "chosen_candidate": None,
            "top_20_validation": table.head(20).to_dict(
                orient="records"
            ),
        }
        write_json(OUT_JSON, payload)

        best = table.iloc[0]
        print("RUSHING TDS HURDLE/AUDIT RECONCILIATION: PASS")
        print("CANONICAL BASELINE: model_selection_predictions.parquet")
        print("2024 CANDIDATE SELECTION: NO PASS")
        print("BLEND FAMILY DECISION: STOP")
        print("2025 REPORTING METRICS READ: false")
        print(
            "BEST_2024_NONPASS: "
            f"path={best['hurdle_path']} "
            f"weight={best['hurdle_weight']:.3f} "
            f"mae={best['validation_mae']:.6f} "
            f"bias={best['validation_bias']:.6f} "
            f"improvement={best['validation_improvement_pct']:.6f}% "
            f"brier={best['validation_brier_1plus']:.6f} "
            f"poisson={best['validation_poisson_deviance']:.6f} "
            f"failed={best['validation_failed']}"
        )
        print(f"csv={OUT_CSV}")
        print(f"json={OUT_JSON}")
        return 0

    chosen = sorted(
        passing,
        key=lambda candidate: (
            float(candidate["validation_metrics"]["mae"]),
            float(candidate["validation_metrics"]["poisson_deviance"]),
            float(candidate["validation_metrics"]["abs_bias"]),
            str(candidate["hurdle_path"]),
            float(candidate["hurdle_weight"]),
        ),
    )[0]

    # Freeze the candidate BEFORE any 2025 reporting metrics are read.
    freeze_payload = {
        "status": "frozen",
        "target": TARGET,
        "frozen_from_season": VALIDATION_SEASON,
        "test_season": TEST_SEASON,
        "test_used_for_selection": False,
        "market_data_used": False,
        "hurdle_path": chosen["hurdle_path"],
        "hurdle_weight": chosen["hurdle_weight"],
        "raw_component_weight": chosen["raw_component_weight"],
        "validation_metrics": chosen["validation_metrics"],
        "validation_failed": chosen["validation_failed"],
        "locked_thresholds": threshold,
        "canonical_baseline_source": (
            "evaluation/model_selection_predictions.parquet"
        ),
    }
    write_json(FREEZE_JSON, freeze_payload)

    # REPORTING PHASE: only now may 2025 metrics be evaluated.
    test_audit = read_audit_split("test", TEST_SEASON)
    test_hurdle = read_hurdle_split(
        "test_2025_reporting_only",
        TEST_SEASON,
    )
    test, reconciliation_2025 = reconcile(
        test_audit,
        test_hurdle,
        "test_2025_reporting_only",
    )

    ty = test["actual"].to_numpy(dtype="float64")
    tb = test["baseline_projection"].to_numpy(dtype="float64")
    traw = test["component_projection"].to_numpy(dtype="float64")
    thurdle = test[str(chosen["hurdle_path"])].to_numpy(
        dtype="float64"
    )
    tp1 = frozen_probability_path(traw, calibration)
    test_point = (
        float(chosen["raw_component_weight"]) * traw
        + float(chosen["hurdle_weight"]) * thurdle
    )
    test_metric = metrics(ty, test_point, tb, tp1)

    payload = {
        **base_payload,
        "selection_status": "validation_gate_pass_candidate_frozen",
        "blend_family_decision": "VALID_2024_CANDIDATE",
        "candidate_frozen_before_2025_reporting": True,
        "chosen_candidate": chosen,
        "test_2025_metrics_read": True,
        "test_2025_reporting_only": {
            "metrics": test_metric,
            "failed_if_gates_were_applied_for_reporting": failures(
                test_metric,
                threshold,
            ),
            "reconciliation": reconciliation_2025,
        },
        "top_20_validation": table.head(20).to_dict(
            orient="records"
        ),
    }
    write_json(OUT_JSON, payload)

    metric = chosen["validation_metrics"]
    print("RUSHING TDS HURDLE/AUDIT RECONCILIATION: PASS")
    print("CANONICAL BASELINE: model_selection_predictions.parquet")
    print("2024 CANDIDATE SELECTION: PASS")
    print(
        "FROZEN_2024: "
        f"path={chosen['hurdle_path']} "
        f"hurdle_weight={chosen['hurdle_weight']:.3f} "
        f"raw_weight={chosen['raw_component_weight']:.3f} "
        f"mae={metric['mae']:.6f} "
        f"bias={metric['bias']:.6f} "
        f"improvement={metric['improvement_vs_baseline_pct']:.6f}% "
        f"brier={metric['brier_1plus']:.6f} "
        f"poisson={metric['poisson_deviance']:.6f}"
    )
    print("CANDIDATE FROZEN BEFORE 2025 REPORTING: true")
    print(
        "2025 REPORTING ONLY: "
        f"mae={test_metric['mae']:.6f} "
        f"bias={test_metric['bias']:.6f} "
        f"improvement={test_metric['improvement_vs_baseline_pct']:.6f}% "
        f"brier={test_metric['brier_1plus']:.6f} "
        f"poisson={test_metric['poisson_deviance']:.6f}"
    )
    print(f"freeze={FREEZE_JSON}")
    print(f"csv={OUT_CSV}")
    print(f"json={OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
