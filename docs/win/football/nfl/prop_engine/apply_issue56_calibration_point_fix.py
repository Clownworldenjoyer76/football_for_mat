#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAINER = HERE / "scripts" / "train" / "calibrate_uncertainty.py"
PROJECT = HERE / "scripts" / "project" / "project_week.py"
EVALUATOR = HERE / "evaluate_production_approval.py"


def fail(message: str) -> None:
    raise RuntimeError(message)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        fail(f"{label}: expected exactly one replacement site, found {count}.")
    return text.replace(old, new, 1)


def patch_trainer() -> None:
    text = TRAINER.read_text(encoding="utf-8")
    if "POINT_PREDICTION_BLEND_CANDIDATES" in text:
        print("trainer=already_patched")
        return

    text = replace_once(
        text,
        "import tempfile\n\nimport numpy as np",
        "import tempfile\n\nimport numpy as np\nimport yaml",
        "trainer yaml import",
    )

    text = replace_once(
        text,
        '''CALIBRATION_ROOT = Path(\n    "docs/win/football/nfl/prop_engine/models/calibration"\n)\n\nQUANTILE_TARGETS = [''',
        '''CALIBRATION_ROOT = Path(\n    "docs/win/football/nfl/prop_engine/models/calibration"\n)\nACCEPTANCE_THRESHOLDS_PATH = Path(\n    "docs/win/football/nfl/prop_engine/config/acceptance_thresholds.yaml"\n)\n\n# Validation-only strength for the displayed point prediction.\n# 0.0 = raw selected point; 1.0 = full existing calibrated point.\nPOINT_PREDICTION_BLEND_CANDIDATES = (0.0, 0.25, 0.50, 0.75, 1.0)\n\nQUANTILE_TARGETS = [''',
        "trainer constants",
    )

    old = '''        actual = pd.to_numeric(frame["actual"], errors="coerce")\n        point = pd.to_numeric(frame[source_column], errors="coerce")\n        valid = actual.notna() & point.notna() & np.isfinite(actual) & np.isfinite(point)\n        frame = frame.loc[valid, [*GRAIN, "fold_id", "actual"]].copy()\n        frame["target"] = target\n        frame["selected_architecture"] = contract.selected_architecture\n        frame["selected_point_prediction"] = point.loc[valid].to_numpy(dtype="float64")\n'''
    new = '''        actual = pd.to_numeric(frame["actual"], errors="coerce")\n        point = pd.to_numeric(frame[source_column], errors="coerce")\n        baseline = pd.to_numeric(frame["baseline_projection"], errors="coerce")\n        valid = (\n            actual.notna()\n            & point.notna()\n            & baseline.notna()\n            & np.isfinite(actual)\n            & np.isfinite(point)\n            & np.isfinite(baseline)\n        )\n        frame = frame.loc[\n            valid,\n            [*GRAIN, "fold_id", "actual", "baseline_projection"],\n        ].copy()\n        frame["target"] = target\n        frame["selected_architecture"] = contract.selected_architecture\n        frame["selected_point_prediction"] = point.loc[valid].to_numpy(dtype="float64")\n        frame["baseline_projection"] = baseline.loc[valid].to_numpy(dtype="float64")\n'''
    text = replace_once(text, old, new, "trainer validation cohort")

    anchor = '''def target_is_nonnegative(config: dict[str, Any], target: str) -> bool:\n    return str(config["targets"][target].get("type", "")) != "continuous_signed"\n\n\ndef build_target_calibration(\n'''
    helpers = '''def target_is_nonnegative(config: dict[str, Any], target: str) -> bool:\n    return str(config["targets"][target].get("type", "")) != "continuous_signed"\n\n\ndef _point_mae(actual: np.ndarray, prediction: np.ndarray) -> float:\n    return float(np.mean(np.abs(prediction - actual)))\n\n\ndef _point_bias(actual: np.ndarray, prediction: np.ndarray) -> float:\n    return float(np.mean(prediction - actual))\n\n\ndef _point_poisson_deviance(actual: np.ndarray, prediction: np.ndarray) -> float:\n    y = np.asarray(actual, dtype="float64")\n    lam = np.maximum(np.asarray(prediction, dtype="float64"), 1e-12)\n    if np.any(y < 0.0):\n        raise ValueError("Negative actual in point-calibration Poisson gate.")\n    terms = np.empty_like(y)\n    zero = y <= 0.0\n    terms[zero] = lam[zero]\n    nz = ~zero\n    terms[nz] = y[nz] * np.log(y[nz] / lam[nz]) - (y[nz] - lam[nz])\n    return float(2.0 * np.mean(terms))\n\n\ndef _point_brier_1plus(actual: np.ndarray, probability: np.ndarray) -> float:\n    event = (np.asarray(actual, dtype="float64") >= 1.0).astype("float64")\n    p = np.clip(np.asarray(probability, dtype="float64"), 0.0, 1.0)\n    return float(np.mean(np.square(p - event)))\n\n\ndef _base_calibrated_point(\n    frame: pd.DataFrame,\n    payload: dict[str, Any],\n    *,\n    floor_at_zero: bool,\n) -> np.ndarray:\n    raw = frame["selected_point_prediction"].to_numpy(dtype="float64")\n    if "count_calibration" in payload:\n        mapping = payload["count_calibration"]["expected_count"]["mapping"]\n        return np.maximum(apply_mapping(np.maximum(raw, 0.0), mapping), 0.0)\n    qcal = payload.get("quantile_calibration")\n    if not isinstance(qcal, dict):\n        raise ValueError("Point calibration has no count or quantile calibration.")\n    output = raw + float(qcal["residual_quantiles"]["q50"])\n    if floor_at_zero:\n        output = np.maximum(output, 0.0)\n    return output\n\n\ndef _base_probability_1plus(\n    frame: pd.DataFrame,\n    payload: dict[str, Any],\n    base_point: np.ndarray,\n) -> np.ndarray:\n    raw = frame["selected_point_prediction"].to_numpy(dtype="float64")\n    if "count_calibration" in payload:\n        ccal = payload["count_calibration"]\n        expected = apply_mapping(\n            np.maximum(raw, 0.0),\n            ccal["expected_count"]["mapping"],\n        )\n        poisson_p1 = 1.0 - np.exp(-np.maximum(expected, 0.0))\n        return np.clip(\n            apply_mapping(poisson_p1, ccal["probability_1_plus"]["mapping"]),\n            0.0,\n            1.0,\n        )\n    return 1.0 - np.exp(-np.maximum(base_point, 0.0))\n\n\ndef fit_point_prediction_blend(\n    config: dict[str, Any],\n    target: str,\n    frame: pd.DataFrame,\n    payload: dict[str, Any],\n    acceptance: dict[str, Any],\n) -> dict[str, Any]:\n    # Selection uses only the 2024 validation rows already loaded into frame.\n    actual = frame["actual"].to_numpy(dtype="float64")\n    raw = frame["selected_point_prediction"].to_numpy(dtype="float64")\n    baseline = frame["baseline_projection"].to_numpy(dtype="float64")\n    floor_at_zero = target_is_nonnegative(config, target)\n    base = _base_calibrated_point(frame, payload, floor_at_zero=floor_at_zero)\n    p1 = _base_probability_1plus(frame, payload, base)\n\n    baseline_mae = _point_mae(actual, baseline)\n    if not math.isfinite(baseline_mae) or baseline_mae <= 0.0:\n        raise ValueError(f"{target}: invalid baseline MAE for point calibration.")\n\n    candidates: list[dict[str, Any]] = []\n    passing: list[tuple[float, float, float]] = []\n    for alpha in POINT_PREDICTION_BLEND_CANDIDATES:\n        prediction = raw + float(alpha) * (base - raw)\n        if floor_at_zero:\n            prediction = np.maximum(prediction, 0.0)\n\n        candidate_mae = _point_mae(actual, prediction)\n        candidate_bias = _point_bias(actual, prediction)\n        abs_bias = abs(candidate_bias)\n        improvement = (baseline_mae - candidate_mae) / baseline_mae * 100.0\n        gates: dict[str, bool] = {\n            "mae": candidate_mae <= float(acceptance["maximum_validation_mae"]) + 1e-12,\n            "bias": abs_bias <= float(acceptance["maximum_allowed_bias"]) + 1e-12,\n            "improvement": improvement + 1e-12 >= float(acceptance["minimum_improvement_vs_baseline_pct"]),\n        }\n\n        brier = None\n        poisson = None\n        if "maximum_brier_1plus" in acceptance and "maximum_poisson_deviance" in acceptance:\n            brier = _point_brier_1plus(actual, p1)\n            poisson = _point_poisson_deviance(actual, prediction)\n            gates["brier_1plus"] = brier <= float(acceptance["maximum_brier_1plus"]) + 1e-12\n            gates["poisson_deviance"] = poisson <= float(acceptance["maximum_poisson_deviance"]) + 1e-12\n\n        passed = bool(all(gates.values()))\n        candidates.append({\n            "calibrated_weight": float(alpha),\n            "raw_weight": float(1.0 - alpha),\n            "validation_mae": candidate_mae,\n            "validation_bias": candidate_bias,\n            "validation_absolute_bias": abs_bias,\n            "validation_improvement_vs_baseline_pct": improvement,\n            "validation_brier_1plus": brier,\n            "validation_poisson_deviance": poisson,\n            "passed_all_configured_gates": passed,\n            "failed_gates": [name for name, ok in gates.items() if not ok],\n        })\n        if passed:\n            passing.append((candidate_mae, abs_bias, float(alpha)))\n\n    if passing:\n        passing.sort(key=lambda item: (item[0], item[1], item[2]))\n        chosen_alpha = float(passing[0][2])\n        status = "validation_candidate_passed_all_gates"\n    else:\n        chosen_alpha = 1.0\n        status = "no_validation_candidate_passed_all_gates"\n\n    chosen = next(\n        item for item in candidates\n        if abs(float(item["calibrated_weight"]) - chosen_alpha) <= 1e-12\n    )\n    return {\n        "method": "validation_only_blend_raw_with_existing_calibrated_point",\n        "selection_split": "validation",\n        "selection_season": int(frame["season"].iloc[0]),\n        "test_rows_used_for_selection": False,\n        "candidate_calibrated_weights": [float(v) for v in POINT_PREDICTION_BLEND_CANDIDATES],\n        "calibrated_weight": chosen_alpha,\n        "raw_weight": float(1.0 - chosen_alpha),\n        "floor_at_zero": bool(floor_at_zero),\n        "selection_status": status,\n        "selected_validation_metrics": chosen,\n        "candidate_metrics": candidates,\n        "probability_calibration_unchanged": True,\n        "interval_calibration_unchanged": True,\n    }\n\n\ndef build_target_calibration(\n'''
    text = replace_once(text, anchor, helpers, "trainer helpers")

    text = replace_once(
        text,
        '''def build_target_calibration(\n    config: dict[str, Any],\n    contract: SelectedContract,\n    frame: pd.DataFrame,\n    usage_source: dict[str, Any],\n) -> tuple[dict[str, Any], list[dict[str, Any]]]:\n''',
        '''def build_target_calibration(\n    config: dict[str, Any],\n    contract: SelectedContract,\n    frame: pd.DataFrame,\n    usage_source: dict[str, Any],\n    acceptance: dict[str, Any],\n) -> tuple[dict[str, Any], list[dict[str, Any]]]:\n''',
        "trainer signature",
    )

    text = replace_once(
        text,
        '''    if target == "tackles":\n        payload["calibration_mode"] = "quantiles_and_count"\n    elif target in QUANTILE_TARGETS:\n        payload["calibration_mode"] = "quantiles"\n    else:\n        payload["calibration_mode"] = "count"\n\n    return payload, coverage_output\n''',
        '''    if target == "tackles":\n        payload["calibration_mode"] = "quantiles_and_count"\n    elif target in QUANTILE_TARGETS:\n        payload["calibration_mode"] = "quantiles"\n    else:\n        payload["calibration_mode"] = "count"\n\n    payload["point_prediction_blend"] = fit_point_prediction_blend(\n        config,\n        target,\n        frame,\n        payload,\n        acceptance,\n    )\n\n    return payload, coverage_output\n''',
        "trainer build point blend",
    )

    text = replace_once(
        text,
        '''    config = common.load_config()\n    targets = list(config["targets"].keys())\n''',
        '''    config = common.load_config()\n\n    acceptance_path = common.repo_root() / ACCEPTANCE_THRESHOLDS_PATH\n    if not acceptance_path.is_file():\n        raise FileNotFoundError(f"Missing acceptance thresholds: {acceptance_path}")\n    with acceptance_path.open("r", encoding="utf-8-sig") as handle:\n        acceptance_thresholds = yaml.safe_load(handle)\n    if not isinstance(acceptance_thresholds, dict):\n        raise ValueError("acceptance_thresholds.yaml must be a YAML mapping.")\n\n    targets = list(config["targets"].keys())\n''',
        "trainer acceptance load",
    )

    text = replace_once(
        text,
        '''        payload, target_coverage = build_target_calibration(\n            config,\n            contracts[target],\n            frame,\n            usage_sources[target],\n        )\n''',
        '''        if target not in acceptance_thresholds:\n            raise ValueError(f"Missing acceptance thresholds for {target}.")\n        payload, target_coverage = build_target_calibration(\n            config,\n            contracts[target],\n            frame,\n            usage_sources[target],\n            acceptance_thresholds[target],\n        )\n''',
        "trainer build call",
    )

    TRAINER.write_text(text, encoding="utf-8", newline="\n")
    print("trainer=patched")


def patch_project() -> None:
    text = PROJECT.read_text(encoding="utf-8")
    if "def apply_point_prediction_blend(" in text:
        print("project_week=already_patched")
        return

    text = replace_once(
        text,
        '''def calibrate_current_target(\n    target: str,\n''',
        '''def apply_point_prediction_blend(\n    frame: pd.DataFrame,\n    calibrated_point: np.ndarray,\n    calibration: dict[str, Any],\n) -> np.ndarray:\n    spec = calibration.get("point_prediction_blend")\n    if not isinstance(spec, dict):\n        return np.asarray(calibrated_point, dtype="float64")\n    alpha = float(spec.get("calibrated_weight", 1.0))\n    if not 0.0 <= alpha <= 1.0:\n        raise ValueError(f"Invalid point calibration weight: {alpha}")\n    raw = numeric(frame["selected_point_prediction"]).to_numpy(dtype="float64")\n    base = np.asarray(calibrated_point, dtype="float64")\n    output = raw + alpha * (base - raw)\n    if bool(spec.get("floor_at_zero")):\n        output = np.maximum(output, 0.0)\n    return output\n\n\ndef calibrate_current_target(\n    target: str,\n''',
        "project helper",
    )

    text = replace_once(
        text,
        '''    if cout is not None:\n        result["projection"] = cout["expected_count"]\n    elif qout is not None:\n        result["projection"] = qout["q50"]\n    else:\n        raise ValueError(f"{target}: no calibration output")\n''',
        '''    if cout is not None:\n        base_point = cout["expected_count"]\n    elif qout is not None:\n        base_point = qout["q50"]\n    else:\n        raise ValueError(f"{target}: no calibration output")\n\n    result["projection"] = apply_point_prediction_blend(\n        frame,\n        np.asarray(base_point, dtype="float64"),\n        calibration,\n    )\n''',
        "project use",
    )

    PROJECT.write_text(text, encoding="utf-8", newline="\n")
    print("project_week=patched")


def patch_evaluator() -> None:
    if not EVALUATOR.is_file():
        print("evaluator=not_present")
        return
    text = EVALUATOR.read_text(encoding="utf-8")
    if "def apply_point_prediction_blend(" in text:
        print("evaluator=already_patched")
        return

    text = replace_once(
        text,
        '''def mae(actual: np.ndarray, predicted: np.ndarray) -> float:\n''',
        '''def apply_point_prediction_blend(\n    raw_selected: np.ndarray,\n    calibrated_point: np.ndarray,\n    calibration: dict[str, Any],\n) -> np.ndarray:\n    spec = calibration.get("point_prediction_blend")\n    if not isinstance(spec, dict):\n        return np.asarray(calibrated_point, dtype="float64")\n    alpha = float(spec.get("calibrated_weight", 1.0))\n    raw = np.asarray(raw_selected, dtype="float64")\n    base = np.asarray(calibrated_point, dtype="float64")\n    output = raw + alpha * (base - raw)\n    if bool(spec.get("floor_at_zero")):\n        output = np.maximum(output, 0.0)\n    return output\n\n\ndef mae(actual: np.ndarray, predicted: np.ndarray) -> float:\n''',
        "evaluator helper",
    )
    text = replace_once(
        text,
        '''        point = np.asarray(out["expected_count"], dtype="float64")\n        p1 = np.asarray(out["probability_1_plus"], dtype="float64")\n        return point, p1, "production_count_calibration"\n''',
        '''        base_point = np.asarray(out["expected_count"], dtype="float64")\n        point = apply_point_prediction_blend(raw_selected, base_point, calibration)\n        p1 = np.asarray(out["probability_1_plus"], dtype="float64")\n        return point, p1, "production_count_calibration_with_point_blend"\n''',
        "evaluator count",
    )
    text = replace_once(
        text,
        '''        center = float(residual["q50"])\n        point = np.asarray(raw_selected, dtype="float64") + center\n        return point, None, "production_quantile_q50"\n''',
        '''        center = float(residual["q50"])\n        base_point = np.asarray(raw_selected, dtype="float64") + center\n        point = apply_point_prediction_blend(raw_selected, base_point, calibration)\n        return point, None, "production_quantile_q50_with_point_blend"\n''',
        "evaluator quantile",
    )
    EVALUATOR.write_text(text, encoding="utf-8", newline="\n")
    print("evaluator=patched")


def main() -> int:
    for path in (TRAINER, PROJECT):
        if not path.is_file():
            fail(f"Required file missing: {path}")
    patch_trainer()
    patch_project()
    patch_evaluator()

    import py_compile
    py_compile.compile(str(TRAINER), doraise=True)
    py_compile.compile(str(PROJECT), doraise=True)
    if EVALUATOR.is_file():
        py_compile.compile(str(EVALUATOR), doraise=True)

    print("syntax=PASS")
    print("thresholds_modified=false")
    print("registry_modified=false")
    print("model_files_modified=false")
    print("ISSUE 56 CALIBRATION POINT FIX PATCH: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
