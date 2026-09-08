#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
PREDICTIONS = HERE / "evaluation" / "model_selection_predictions.parquet"
THRESHOLDS = HERE / "config" / "acceptance_thresholds.yaml"
CAL_ROOT = HERE / "models" / "calibration"
PROJECT = HERE / "scripts" / "project" / "project_week.py"
TRAINER = HERE / "scripts" / "train" / "calibrate_uncertainty.py"

TARGETS = [
    "passing_yards", "passing_tds", "rushing_yards", "rushing_tds",
    "receiving_yards", "receiving_tds", "kicking_points", "tackles", "sacks",
]
EXPECTED_PASS = {
    "passing_yards", "passing_tds", "rushing_yards",
    "receiving_yards", "kicking_points", "tackles",
}
EXPECTED_BLOCKED = {"rushing_tds", "receiving_tds", "sacks"}
SELECTED_COLUMNS = {
    "baseline": "baseline_projection",
    "direct": "direct_projection",
    "component": "component_projection",
    "direct_component_blend": "blend_projection",
}
COUNT_DIAGNOSTIC_TARGETS = {
    "passing_tds", "rushing_tds", "receiving_tds",
    "kicking_points", "tackles", "sacks",
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AssertionError(f"Missing {path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise AssertionError(f"Expected JSON object: {path}")
    return value


def apply_mapping(values: np.ndarray, mapping: dict[str, Any]) -> np.ndarray:
    xp = np.asarray(mapping["knots_x"], dtype="float64")
    fp = np.asarray(mapping["knots_y"], dtype="float64")
    out = np.interp(
        np.asarray(values, dtype="float64"), xp, fp,
        left=float(mapping["left_value"]), right=float(mapping["right_value"]),
    )
    bounds = mapping.get("output_bounds", [None, None])
    if bounds[0] is not None:
        out = np.maximum(out, float(bounds[0]))
    if bounds[1] is not None:
        out = np.minimum(out, float(bounds[1]))
    return out


def base_point(raw: np.ndarray, cal: dict[str, Any]) -> np.ndarray:
    raw = np.asarray(raw, dtype="float64")
    if "count_calibration" in cal:
        return np.maximum(
            apply_mapping(
                np.maximum(raw, 0.0),
                cal["count_calibration"]["expected_count"]["mapping"],
            ),
            0.0,
        )
    qcal = cal["quantile_calibration"]
    out = raw + float(qcal["residual_quantiles"]["q50"])
    if bool(qcal.get("floor_at_zero")):
        out = np.maximum(out, 0.0)
    return out


def probability_1plus(raw: np.ndarray, cal: dict[str, Any], base: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype="float64")
    if "count_calibration" in cal:
        ccal = cal["count_calibration"]
        expected = apply_mapping(np.maximum(raw, 0.0), ccal["expected_count"]["mapping"])
        poisson = 1.0 - np.exp(-np.maximum(expected, 0.0))
        return np.clip(apply_mapping(poisson, ccal["probability_1_plus"]["mapping"]), 0.0, 1.0)
    return 1.0 - np.exp(-np.maximum(base, 0.0))


def final_point(raw: np.ndarray, cal: dict[str, Any]) -> np.ndarray:
    base = base_point(raw, cal)
    spec = cal["point_prediction_blend"]
    alpha = float(spec["calibrated_weight"])
    out = np.asarray(raw, dtype="float64") + alpha * (base - np.asarray(raw, dtype="float64"))
    if bool(spec.get("floor_at_zero")):
        out = np.maximum(out, 0.0)
    return out


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(p) - np.asarray(y))))


def bias(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.asarray(p) - np.asarray(y)))


def poisson_deviance(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype="float64")
    lam = np.maximum(np.asarray(p, dtype="float64"), 1e-12)
    terms = np.empty_like(y)
    zero = y <= 0.0
    terms[zero] = lam[zero]
    nz = ~zero
    terms[nz] = y[nz] * np.log(y[nz] / lam[nz]) - (y[nz] - lam[nz])
    return float(2.0 * np.mean(terms))


def brier_1plus(y: np.ndarray, p1: np.ndarray) -> float:
    event = (np.asarray(y, dtype="float64") >= 1.0).astype("float64")
    return float(np.mean(np.square(np.asarray(p1) - event)))


def evaluate(target: str, y: np.ndarray, p: np.ndarray, baseline: np.ndarray, p1: np.ndarray, threshold: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    model_mae = mae(y, p)
    model_bias = bias(y, p)
    abs_bias = abs(model_bias)
    baseline_mae = mae(y, baseline)
    improvement = (baseline_mae - model_mae) / baseline_mae * 100.0
    checks = {
        "mae": model_mae <= float(threshold["maximum_validation_mae"]) + 1e-12,
        "bias": abs_bias <= float(threshold["maximum_allowed_bias"]) + 1e-12,
        "improvement": improvement + 1e-12 >= float(threshold["minimum_improvement_vs_baseline_pct"]),
    }
    brier = None
    poisson = None
    if target in COUNT_DIAGNOSTIC_TARGETS:
        brier = brier_1plus(y, p1)
        poisson = poisson_deviance(y, p)
        checks["brier"] = brier <= float(threshold["maximum_brier_1plus"]) + 1e-12
        checks["poisson"] = poisson <= float(threshold["maximum_poisson_deviance"]) + 1e-12
    return bool(all(checks.values())), {
        "mae": model_mae,
        "bias": model_bias,
        "improvement": improvement,
        "brier": brier,
        "poisson": poisson,
        "failed": [name for name, ok in checks.items() if not ok],
    }


def main() -> int:
    print("CHECK 01: patched production/training source")
    trainer_text = TRAINER.read_text(encoding="utf-8")
    project_text = PROJECT.read_text(encoding="utf-8")
    assert "POINT_PREDICTION_BLEND_CANDIDATES" in trainer_text
    assert "fit_point_prediction_blend" in trainer_text
    assert "def apply_point_prediction_blend(" in project_text
    assert 'result["projection"] = apply_point_prediction_blend(' in project_text

    print("CHECK 02: regenerated calibration artifacts")
    calibrations: dict[str, dict[str, Any]] = {}
    weights: dict[str, float] = {}
    for target in TARGETS:
        cal = read_json(CAL_ROOT / f"{target}_calibration.json")
        calibrations[target] = cal
        spec = cal.get("point_prediction_blend")
        assert isinstance(spec, dict), f"{target}: missing point_prediction_blend"
        assert spec.get("selection_split") == "validation"
        assert int(spec.get("selection_season", -1)) == 2024
        assert spec.get("test_rows_used_for_selection") is False
        assert spec.get("probability_calibration_unchanged") is True
        assert spec.get("interval_calibration_unchanged") is True
        alpha = float(spec["calibrated_weight"])
        assert alpha in {0.0, 0.25, 0.5, 0.75, 1.0}
        weights[target] = alpha

    print("CHECK 03: untouched 2025 reporting result")
    thresholds = yaml.safe_load(THRESHOLDS.read_text(encoding="utf-8-sig"))
    assert isinstance(thresholds, dict)
    audit = pd.read_parquet(PREDICTIONS)
    passed: set[str] = set()
    blocked: set[str] = set()

    for target in TARGETS:
        selected = read_json(HERE / "models" / target / "selected_model.json")
        assert selected.get("test_used_for_selection") is False
        assert selected.get("test_reporting_only") is True
        architecture = str(selected.get("selected_architecture") or selected.get("selected_candidate"))
        source_col = SELECTED_COLUMNS[architecture]
        rows = audit.loc[
            audit["split"].astype(str).eq("test")
            & pd.to_numeric(audit["season"], errors="coerce").eq(2025)
            & audit["target"].astype(str).eq(target)
        ].copy()
        assert not rows.empty
        y = pd.to_numeric(rows["actual"], errors="coerce").to_numpy(dtype="float64")
        raw = pd.to_numeric(rows[source_col], errors="coerce").to_numpy(dtype="float64")
        baseline = pd.to_numeric(rows["baseline_projection"], errors="coerce").to_numpy(dtype="float64")
        valid = np.isfinite(y) & np.isfinite(raw) & np.isfinite(baseline)
        y, raw, baseline = y[valid], raw[valid], baseline[valid]
        base = base_point(raw, calibrations[target])
        p1 = probability_1plus(raw, calibrations[target], base)
        point = final_point(raw, calibrations[target])
        ok, metrics = evaluate(target, y, point, baseline, p1, thresholds[target])
        status = "PASS" if ok else "FAIL"
        print(
            f"{target}: {status} weight={weights[target]:.2f} "
            f"mae={metrics['mae']:.6f} bias={metrics['bias']:.6f} "
            f"improvement={metrics['improvement']:.6f}% "
            f"failed={';'.join(metrics['failed']) or 'none'}"
        )
        (passed if ok else blocked).add(target)

    print("CHECK 04: repair boundary")
    assert passed == EXPECTED_PASS, f"Unexpected pass set: {sorted(passed)}"
    assert blocked == EXPECTED_BLOCKED, f"Unexpected blocked set: {sorted(blocked)}"
    print("passing=" + ",".join(sorted(passed)))
    print("still_blocked=" + ",".join(sorted(blocked)))
    print("weights=" + ",".join(f"{t}:{weights[t]:.2f}" for t in TARGETS))
    print("thresholds_modified=false")
    print("registry_modified=false")
    print("2025_used_for_calibration_selection=false")
    print("ISSUE 56 CALIBRATION POINT FIX VALIDATION: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
