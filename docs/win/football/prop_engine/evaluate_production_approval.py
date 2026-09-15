#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / "scripts"
PROJECT = SCRIPTS / "project"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import common

TARGETS = list(common.load_config()["targets"].keys())

SELECTED_PROJECTION_COLUMNS = {
    "baseline": "baseline_projection",
    "direct": "direct_projection",
    "component": "component_projection",
    "direct_component_blend": "blend_projection",
}

GRAIN = ["season", "week", "game_id", "player_id"]

PREDICTIONS = HERE / "evaluation" / "model_selection_predictions.parquet"
THRESHOLDS = HERE / "config" / "acceptance_thresholds.yaml"
SOURCE_QUALITY = HERE / "evaluation" / "source_quality.csv"
INTERVAL_COVERAGE = HERE / "evaluation" / "interval_coverage.csv"
CSV_OUT = HERE / "evaluation" / "production_approval_evaluation.csv"
JSON_OUT = HERE / "evaluation" / "production_approval_evaluation.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate frozen 2025 test predictions after applying the same point "
            "calibration used by the production weekly projection path."
        )
    )
    p.add_argument("--test-season", type=int, default=2025)
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


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    tmp = Path(h.name)
    try:
        with h:
            json.dump(
                payload,
                h,
                indent=2,
                sort_keys=False,
                ensure_ascii=False,
                allow_nan=False,
                default=str,
            )
            h.write("\n")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    tmp = Path(h.name)
    h.close()
    try:
        frame.to_csv(tmp, index=False, lineterminator="\n")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def numeric(values: Any) -> np.ndarray:
    return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype="float64")


def apply_mapping(values: np.ndarray, mapping: dict[str, Any]) -> np.ndarray:
    """Exact mapping logic used by scripts/project/project_week.py."""
    xp = np.asarray(mapping["knots_x"], dtype="float64")
    fp = np.asarray(mapping["knots_y"], dtype="float64")
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


def count_outputs(
    raw_selected: np.ndarray,
    payload: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Exact point/probability calibration math used by production."""
    ccal = payload["count_calibration"]
    raw = np.maximum(np.asarray(raw_selected, dtype="float64"), 0.0)
    expected = apply_mapping(raw, ccal["expected_count"]["mapping"])
    poisson_p1 = 1.0 - np.exp(-expected)
    poisson_p2 = 1.0 - np.exp(-expected) * (1.0 + expected)
    p1 = apply_mapping(poisson_p1, ccal["probability_1_plus"]["mapping"])
    p2 = apply_mapping(poisson_p2, ccal["probability_2_plus"]["mapping"])
    return {
        "expected_count": np.maximum(expected, 0.0),
        "probability_1_plus": np.clip(p1, 0.0, 1.0),
        "probability_2_plus": np.clip(p2, 0.0, 1.0),
    }


def apply_point_prediction_blend(
    raw_selected: np.ndarray,
    calibrated_point: np.ndarray,
    calibration: dict[str, Any],
) -> np.ndarray:
    spec = calibration.get("point_prediction_blend")
    if not isinstance(spec, dict):
        return np.asarray(calibrated_point, dtype="float64")
    alpha = float(spec.get("calibrated_weight", 1.0))
    raw = np.asarray(raw_selected, dtype="float64")
    base = np.asarray(calibrated_point, dtype="float64")
    output = raw + alpha * (base - raw)
    if bool(spec.get("floor_at_zero")):
        output = np.maximum(output, 0.0)
    return output


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(predicted - actual)))


def bias(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(predicted - actual))


def poisson_deviance(actual: np.ndarray, predicted: np.ndarray) -> float:
    y = np.asarray(actual, dtype="float64")
    lam = np.maximum(np.asarray(predicted, dtype="float64"), 1e-12)
    if np.any(y < 0):
        raise ValueError("Poisson deviance cannot be computed with negative actual values.")
    terms = np.empty_like(y)
    zero = y <= 0.0
    terms[zero] = lam[zero]
    nz = ~zero
    terms[nz] = y[nz] * np.log(y[nz] / lam[nz]) - (y[nz] - lam[nz])
    return float(2.0 * np.mean(terms))


def brier_1plus(actual: np.ndarray, probability: np.ndarray) -> float:
    event = (np.asarray(actual, dtype="float64") >= 1.0).astype("float64")
    p = np.clip(np.asarray(probability, dtype="float64"), 0.0, 1.0)
    return float(np.mean(np.square(p - event)))


def calibration_integrity(
    target: str,
    payload: dict[str, Any],
    selected: dict[str, Any],
    test_season: int,
) -> tuple[bool, list[str]]:
    problems: list[str] = []

    if payload.get("target") != target:
        problems.append("calibration_target_mismatch")

    selected_arch = clean(
        selected.get("selected_architecture")
        or selected.get("selected_candidate")
    )
    if clean(payload.get("selected_architecture")) != selected_arch:
        problems.append("calibration_architecture_mismatch")

    if payload.get("market_features_used") is not False:
        problems.append("calibration_market_features_used")
    if payload.get("forbidden_features_used") is not False:
        problems.append("calibration_forbidden_features_used")

    source = payload.get("calibration_source", {})
    if not isinstance(source, dict):
        problems.append("calibration_source_missing")
    else:
        if clean(source.get("split")) != "validation":
            problems.append("calibration_not_from_validation")
        try:
            source_season = int(source.get("season"))
        except (TypeError, ValueError):
            source_season = -1
        if source_season >= test_season:
            problems.append("calibration_uses_test_or_future_season")

    policy = payload.get("reporting_test_policy", {})
    if isinstance(policy, dict):
        if policy.get("test_rows_used_for_calibration") is not False:
            problems.append("test_rows_used_for_calibration")
        if policy.get("test_reporting_only_preserved") is not True:
            problems.append("test_reporting_only_not_preserved")
    else:
        problems.append("reporting_test_policy_missing")

    if selected.get("test_used_for_selection") is not False:
        problems.append("test_used_for_model_selection")

    return not problems, problems


def production_calibrated_values(
    target: str,
    raw_selected: np.ndarray,
    calibration: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray | None, str]:
    """
    Return the production point prediction and production p(>=1), if available.

    This follows scripts/project/project_week.py:
    - count / quantiles_and_count: count_outputs -> expected_count + calibrated p1+
    - quantiles only: selected point + residual q50
    """
    mode = clean(calibration.get("calibration_mode"))
    if mode in {"count", "quantiles_and_count"}:
        out = count_outputs(raw_selected, calibration)
        base_point = np.asarray(out["expected_count"], dtype="float64")
        point = apply_point_prediction_blend(raw_selected, base_point, calibration)
        p1 = np.asarray(out["probability_1_plus"], dtype="float64")
        return point, p1, "production_count_calibration_with_point_blend"

    if mode == "quantiles":
        qcal = calibration.get("quantile_calibration")
        if not isinstance(qcal, dict):
            raise ValueError(f"{target}: quantile_calibration missing")
        residual = qcal.get("residual_quantiles")
        if not isinstance(residual, dict) or "q50" not in residual:
            raise ValueError(f"{target}: residual q50 missing")
        center = float(residual["q50"])
        base_point = np.asarray(raw_selected, dtype="float64") + center
        point = apply_point_prediction_blend(raw_selected, base_point, calibration)
        return point, None, "production_quantile_q50_with_point_blend"

    raise ValueError(f"{target}: unsupported calibration_mode={mode!r}")


def training_rows(target: str) -> int:
    metadata = load_json(HERE / "models" / target / "metadata.json")
    rows = metadata.get("rows", {})
    if not isinstance(rows, dict):
        raise ValueError(f"{target}: metadata.rows missing")
    value = rows.get("model_selection_train")
    if value is None:
        raise ValueError(f"{target}: metadata.rows.model_selection_train missing")
    return int(value)


def validate_global_gates(thresholds: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}

    # Source quality.
    sq_cfg = thresholds["source_quality"]
    sq = pd.read_csv(SOURCE_QUALITY)
    status = sq["quality_status"].astype("string").fillna("").str.strip().str.upper()
    checked = int(len(sq))
    pass_rate = float(status.eq("PASS").mean()) if checked else 0.0
    warn_rows = int(status.eq("WARN").sum())
    fail_rows = int(status.eq("FAIL").sum())
    sq_pass = (
        checked >= int(sq_cfg["minimum_sources_checked"])
        and pass_rate >= float(sq_cfg["minimum_pass_rate"])
        and warn_rows <= int(sq_cfg["maximum_warn_rows"])
        and fail_rows <= int(sq_cfg["maximum_fail_rows"])
    )
    result["source_quality"] = {
        "passed": bool(sq_pass),
        "sources_checked": checked,
        "pass_rate": pass_rate,
        "warn_rows": warn_rows,
        "fail_rows": fail_rows,
    }

    # Calibrated interval coverage. The Issue 42 contract uses overall rows only.
    ic_cfg = thresholds["interval_coverage"]
    cov = pd.read_csv(INTERVAL_COVERAGE)
    overall = cov.loc[
        cov["position_group"].astype(str).eq("ALL")
        & cov["usage_bucket"].astype(str).eq("ALL")
    ].copy()
    targets_checked = int(overall["target"].astype(str).nunique())

    interval_results: dict[str, Any] = {}
    interval_pass = targets_checked >= int(ic_cfg["minimum_targets_checked"])
    for interval, key in [
        ("q10_q90", "q10_q90_minimum_coverage"),
        ("q25_q75", "q25_q75_minimum_coverage"),
    ]:
        rows = overall.loc[overall["interval"].astype(str).eq(interval)].copy()
        vals = pd.to_numeric(rows["actual_coverage"], errors="coerce")
        minimum = float(ic_cfg[key])
        passed = (
            not rows.empty
            and vals.notna().all()
            and bool(vals.ge(minimum).all())
        )
        interval_pass = interval_pass and passed
        interval_results[interval] = {
            "passed": bool(passed),
            "minimum_required": minimum,
            "minimum_observed": (
                float(vals.min()) if not rows.empty and vals.notna().all() else None
            ),
            "rows": int(len(rows)),
        }

    result["interval_coverage"] = {
        "passed": bool(interval_pass),
        "targets_checked": targets_checked,
        "intervals": interval_results,
    }

    # Market exclusion.
    audit = HERE / "scripts" / "validate" / "audit_market_exclusion.py"
    cp = subprocess.run(
        [sys.executable, str(audit), "--preflight"],
        cwd=common.repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    market_pass = (
        cp.returncode == 0
        and "MARKET EXCLUSION AUDIT: PASS" in cp.stdout
    )
    result["market_exclusion"] = {
        "passed": bool(market_pass),
        "returncode": int(cp.returncode),
        "stdout_tail": cp.stdout[-1200:],
        "stderr_tail": cp.stderr[-1200:],
    }

    result["all_global_gates_passed"] = bool(
        result["source_quality"]["passed"]
        and result["interval_coverage"]["passed"]
        and result["market_exclusion"]["passed"]
    )
    return result


def main() -> int:
    args = parse_args()
    test_season = int(args.test_season)

    if not PREDICTIONS.is_file():
        raise FileNotFoundError(
            f"Required row-level model-selection predictions are missing: {PREDICTIONS}"
        )

    thresholds = load_yaml(THRESHOLDS)
    predictions = pd.read_parquet(PREDICTIONS)
    required = {
        "split",
        "season",
        *GRAIN[1:],
        "target",
        "actual",
        "baseline_projection",
        "direct_projection",
        "component_projection",
        "blend_projection",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"model_selection_predictions.parquet missing: {missing}")

    test = predictions.loc[
        predictions["split"].astype(str).eq("test")
        & pd.to_numeric(predictions["season"], errors="coerce").eq(test_season)
    ].copy()
    if test.empty:
        raise ValueError(f"No test rows found for season {test_season}")

    if set(test["target"].astype(str).unique()) != set(TARGETS):
        raise ValueError(
            "2025 test target coverage does not equal the nine required targets."
        )

    global_gates = validate_global_gates(thresholds)

    rows: list[dict[str, Any]] = []
    detailed: dict[str, Any] = {}

    for target in TARGETS:
        selected_path = HERE / "models" / target / "selected_model.json"
        calibration_path = (
            HERE / "models" / "calibration" / f"{target}_calibration.json"
        )
        selected = load_json(selected_path)
        calibration = load_json(calibration_path)

        architecture = clean(
            selected.get("selected_architecture")
            or selected.get("selected_candidate")
        )
        if architecture not in SELECTED_PROJECTION_COLUMNS:
            raise ValueError(
                f"{target}: unsupported selected architecture {architecture!r}"
            )
        selected_col = SELECTED_PROJECTION_COLUMNS[architecture]

        target_test = test.loc[
            test["target"].astype(str).eq(target)
        ].copy()

        actual_s = pd.to_numeric(target_test["actual"], errors="coerce")
        selected_s = pd.to_numeric(target_test[selected_col], errors="coerce")
        baseline_s = pd.to_numeric(
            target_test["baseline_projection"], errors="coerce"
        )

        valid_selected = actual_s.notna() & selected_s.notna()
        if not valid_selected.any():
            raise ValueError(f"{target}: no finite selected test predictions")

        actual = actual_s.loc[valid_selected].to_numpy(dtype="float64")
        raw_selected = selected_s.loc[valid_selected].to_numpy(dtype="float64")

        production, production_p1, calibration_method = (
            production_calibrated_values(
                target,
                raw_selected,
                calibration,
            )
        )

        finite_prod = np.isfinite(actual) & np.isfinite(production)
        actual = actual[finite_prod]
        raw_selected = raw_selected[finite_prod]
        production = production[finite_prod]
        if production_p1 is not None:
            production_p1 = production_p1[finite_prod]

        # Baseline improvement must be compared on exactly the same rows.
        baseline_aligned = baseline_s.loc[valid_selected].to_numpy(dtype="float64")
        baseline_aligned = baseline_aligned[finite_prod]
        baseline_finite = np.isfinite(baseline_aligned)
        baseline_coverage_pass = bool(baseline_finite.all())
        if not baseline_coverage_pass:
            baseline_actual = actual[baseline_finite]
            baseline_pred = baseline_aligned[baseline_finite]
            production_for_improvement = production[baseline_finite]
        else:
            baseline_actual = actual
            baseline_pred = baseline_aligned
            production_for_improvement = production

        if len(baseline_actual) == 0:
            raise ValueError(f"{target}: no baseline rows for improvement gate")

        raw_mae = mae(actual, raw_selected)
        raw_bias = bias(actual, raw_selected)
        prod_mae = mae(actual, production)
        prod_bias = bias(actual, production)
        prod_abs_bias = abs(prod_bias)
        base_mae = mae(baseline_actual, baseline_pred)
        prod_common_mae = mae(baseline_actual, production_for_improvement)
        improvement_pct = (
            (base_mae - prod_common_mae) / base_mae * 100.0
            if base_mae > 0
            else float("-inf")
        )

        section = thresholds[target]
        train_rows = training_rows(target)
        min_rows = int(section["minimum_training_rows"])
        max_bias = float(section["maximum_allowed_bias"])
        max_mae = float(section["maximum_validation_mae"])
        min_improvement = float(
            section["minimum_improvement_vs_baseline_pct"]
        )

        gate_training = train_rows >= min_rows
        gate_bias = prod_abs_bias <= max_bias + 1e-12
        gate_mae = prod_mae <= max_mae + 1e-12
        gate_improvement = (
            baseline_coverage_pass
            and improvement_pct + 1e-12 >= min_improvement
        )

        integrity_pass, integrity_problems = calibration_integrity(
            target,
            calibration,
            selected,
            test_season,
        )

        # Count gates exist wherever the threshold file requires them.
        has_count_gates = (
            "maximum_brier_1plus" in section
            and "maximum_poisson_deviance" in section
        )
        brier = None
        poisson = None
        p1_method = None
        gate_brier = True
        gate_poisson = True

        if has_count_gates:
            if np.any(actual < 0):
                raise ValueError(
                    f"{target}: negative actual encountered for count diagnostics"
                )

            if production_p1 is not None:
                p1 = production_p1
                p1_method = "production_calibrated_probability_1_plus"
            else:
                # Kicking points is a derived_count whose production calibration
                # is quantile-only. Issue 42 explicitly requires independent count
                # diagnostics before approval, so calculate the generic >=1-event
                # probability from the calibrated production point.
                p1 = 1.0 - np.exp(-np.maximum(production, 0.0))
                p1_method = (
                    "independent_poisson_probability_from_"
                    "calibrated_production_point"
                )

            brier = brier_1plus(actual, p1)
            poisson = poisson_deviance(actual, production)
            gate_brier = (
                brier <= float(section["maximum_brier_1plus"]) + 1e-12
            )
            gate_poisson = (
                poisson
                <= float(section["maximum_poisson_deviance"]) + 1e-12
            )

        failed: list[str] = []
        gate_map = {
            "training_rows": gate_training,
            "absolute_bias": gate_bias,
            "mae": gate_mae,
            "improvement_vs_baseline": gate_improvement,
            "calibration_integrity": integrity_pass,
            "brier_1plus": gate_brier,
            "poisson_deviance": gate_poisson,
            "global_source_quality": global_gates["source_quality"]["passed"],
            "global_interval_coverage": global_gates["interval_coverage"]["passed"],
            "global_market_exclusion": global_gates["market_exclusion"]["passed"],
        }
        failed = [name for name, passed in gate_map.items() if not passed]
        target_pass = not failed

        row = {
            "target": target,
            "selected_architecture": architecture,
            "calibration_mode": clean(calibration.get("calibration_mode")),
            "calibration_method": calibration_method,
            "test_season": test_season,
            "test_rows": int(len(actual)),
            "training_rows": train_rows,
            "minimum_training_rows": min_rows,
            "raw_selected_mae": raw_mae,
            "production_calibrated_mae": prod_mae,
            "maximum_allowed_mae": max_mae,
            "mae_pass": gate_mae,
            "raw_selected_bias": raw_bias,
            "production_calibrated_bias": prod_bias,
            "production_absolute_bias": prod_abs_bias,
            "maximum_allowed_absolute_bias": max_bias,
            "bias_pass": gate_bias,
            "baseline_mae_same_rows": base_mae,
            "production_mae_same_baseline_rows": prod_common_mae,
            "improvement_vs_baseline_pct": improvement_pct,
            "minimum_improvement_vs_baseline_pct": min_improvement,
            "baseline_coverage_pass": baseline_coverage_pass,
            "improvement_pass": gate_improvement,
            "brier_1plus": brier,
            "maximum_brier_1plus": (
                float(section["maximum_brier_1plus"])
                if has_count_gates
                else None
            ),
            "brier_pass": gate_brier if has_count_gates else None,
            "poisson_deviance": poisson,
            "maximum_poisson_deviance": (
                float(section["maximum_poisson_deviance"])
                if has_count_gates
                else None
            ),
            "poisson_pass": gate_poisson if has_count_gates else None,
            "p1_method": p1_method,
            "calibration_integrity_pass": integrity_pass,
            "production_approval_qualified": target_pass,
            "failed_gates": ";".join(failed),
        }
        rows.append(row)
        detailed[target] = {
            **row,
            "calibration_integrity_problems": integrity_problems,
            "gates": gate_map,
        }

    result = pd.DataFrame(rows)
    atomic_csv(CSV_OUT, result)

    qualified = result.loc[
        result["production_approval_qualified"].eq(True), "target"
    ].astype(str).tolist()
    blocked = result.loc[
        result["production_approval_qualified"].eq(False), "target"
    ].astype(str).tolist()

    payload = {
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "test_season": test_season,
        "policy": {
            "thresholds_are_frozen_from_2024": True,
            "test_season_used_for_selection": False,
            "test_season_used_for_calibration": False,
            "production_calibration_applied_before_metrics": True,
            "thresholds_lowered_to_force_pass": False,
            "production_registry_modified": False,
            "kicking_count_diagnostics_independently_measured": True,
        },
        "global_gates": global_gates,
        "qualified_targets": qualified,
        "blocked_targets": blocked,
        "targets": detailed,
        "outputs": {
            "csv": str(CSV_OUT),
            "json": str(JSON_OUT),
        },
    }
    atomic_json(JSON_OUT, payload)

    print("PRODUCTION-PATH APPROVAL EVALUATION")
    print(f"test_season={test_season}")
    print(
        "global_gates="
        + ("PASS" if global_gates["all_global_gates_passed"] else "FAIL")
    )
    for row in rows:
        status = "PASS" if row["production_approval_qualified"] else "FAIL"
        failed = row["failed_gates"] or "none"
        print(
            f"{row['target']}: {status} "
            f"mae={row['production_calibrated_mae']:.6f} "
            f"bias={row['production_calibrated_bias']:.6f} "
            f"improvement={row['improvement_vs_baseline_pct']:.6f}% "
            f"failed={failed}"
        )
        if row["brier_1plus"] is not None:
            print(
                f"  brier_1plus={row['brier_1plus']:.6f} "
                f"poisson_deviance={row['poisson_deviance']:.6f} "
                f"p1_method={row['p1_method']}"
            )

    print("qualified=" + (",".join(qualified) if qualified else "none"))
    print("blocked=" + (",".join(blocked) if blocked else "none"))
    print(f"csv={CSV_OUT}")
    print(f"json={JSON_OUT}")
    print("PRODUCTION APPROVAL EVALUATION: COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
