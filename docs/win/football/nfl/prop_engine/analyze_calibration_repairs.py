#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent

PREDICTIONS = HERE / "evaluation" / "model_selection_predictions.parquet"
THRESHOLDS = HERE / "config" / "acceptance_thresholds.yaml"
CSV_OUT = HERE / "evaluation" / "calibration_repair_candidates.csv"
JSON_OUT = HERE / "evaluation" / "calibration_repair_analysis.json"

TARGETS = [
    "passing_yards",
    "passing_tds",
    "rushing_yards",
    "rushing_tds",
    "receiving_yards",
    "receiving_tds",
    "kicking_points",
    "tackles",
    "sacks",
]

SELECTED_PROJECTION_COLUMNS = {
    "baseline": "baseline_projection",
    "direct": "direct_projection",
    "component": "component_projection",
    "direct_component_blend": "blend_projection",
}

COUNT_TARGETS = {
    "passing_tds",
    "rushing_tds",
    "receiving_tds",
    "kicking_points",
    "tackles",
    "sacks",
}


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


def numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")


def apply_mapping(values: np.ndarray, mapping: dict[str, Any]) -> np.ndarray:
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


def current_production_point(
    raw: np.ndarray,
    calibration: dict[str, Any],
) -> np.ndarray:
    raw = np.asarray(raw, dtype="float64")
    mode = clean(calibration.get("calibration_mode"))

    if mode in {"count", "quantiles_and_count"}:
        mapping = calibration["count_calibration"]["expected_count"]["mapping"]
        return np.maximum(
            apply_mapping(np.maximum(raw, 0.0), mapping),
            0.0,
        )

    if mode == "quantiles":
        q50 = float(
            calibration["quantile_calibration"]["residual_quantiles"]["q50"]
        )
        out = raw + q50
        if bool(calibration["quantile_calibration"].get("floor_at_zero")):
            out = np.maximum(out, 0.0)
        return out

    raise ValueError(f"Unsupported calibration mode: {mode!r}")


def current_probability_1plus(
    raw: np.ndarray,
    calibration: dict[str, Any],
) -> np.ndarray:
    raw = np.asarray(raw, dtype="float64")
    mode = clean(calibration.get("calibration_mode"))

    if mode in {"count", "quantiles_and_count"}:
        ccal = calibration["count_calibration"]
        expected = apply_mapping(
            np.maximum(raw, 0.0),
            ccal["expected_count"]["mapping"],
        )
        poisson_p1 = 1.0 - np.exp(-np.maximum(expected, 0.0))
        return np.clip(
            apply_mapping(
                poisson_p1,
                ccal["probability_1_plus"]["mapping"],
            ),
            0.0,
            1.0,
        )

    # Kicking points has quantile-only production calibration but Issue 42
    # requires an independent >=1 diagnostic.
    point = current_production_point(raw, calibration)
    return 1.0 - np.exp(-np.maximum(point, 0.0))


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(p) - np.asarray(y))))


def bias(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.asarray(p) - np.asarray(y)))


def poisson_deviance(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype="float64")
    lam = np.maximum(np.asarray(p, dtype="float64"), 1e-12)
    if np.any(y < 0):
        raise ValueError("Negative actual in Poisson diagnostic.")
    terms = np.empty_like(y)
    zero = y <= 0
    terms[zero] = lam[zero]
    nz = ~zero
    terms[nz] = y[nz] * np.log(y[nz] / lam[nz]) - (y[nz] - lam[nz])
    return float(2.0 * np.mean(terms))


def brier_1plus(y: np.ndarray, probability: np.ndarray) -> float:
    event = (np.asarray(y, dtype="float64") >= 1.0).astype("float64")
    p = np.clip(np.asarray(probability, dtype="float64"), 0.0, 1.0)
    return float(np.mean(np.square(p - event)))


def point_gates(
    target: str,
    y: np.ndarray,
    p: np.ndarray,
    baseline: np.ndarray,
    threshold: dict[str, Any],
    *,
    p1: np.ndarray | None,
) -> dict[str, Any]:
    valid = (
        np.isfinite(y)
        & np.isfinite(p)
        & np.isfinite(baseline)
    )
    y = y[valid]
    p = p[valid]
    baseline = baseline[valid]
    if p1 is not None:
        p1 = p1[valid]

    model_mae = mae(y, p)
    model_bias = bias(y, p)
    abs_bias = abs(model_bias)
    baseline_mae = mae(y, baseline)
    improvement = (
        (baseline_mae - model_mae) / baseline_mae * 100.0
        if baseline_mae > 0
        else float("-inf")
    )

    gates = {
        "mae": model_mae <= float(threshold["maximum_validation_mae"]) + 1e-12,
        "bias": abs_bias <= float(threshold["maximum_allowed_bias"]) + 1e-12,
        "improvement": (
            improvement + 1e-12
            >= float(threshold["minimum_improvement_vs_baseline_pct"])
        ),
    }

    brier = None
    poisson = None
    if target in COUNT_TARGETS:
        poisson = poisson_deviance(y, p)
        gates["poisson"] = (
            poisson
            <= float(threshold["maximum_poisson_deviance"]) + 1e-12
        )
        if p1 is None:
            p1 = 1.0 - np.exp(-np.maximum(p, 0.0))
        brier = brier_1plus(y, p1)
        gates["brier"] = (
            brier <= float(threshold["maximum_brier_1plus"]) + 1e-12
        )

    return {
        "mae": model_mae,
        "bias": model_bias,
        "abs_bias": abs_bias,
        "baseline_mae": baseline_mae,
        "improvement_pct": improvement,
        "poisson_deviance": poisson,
        "brier_1plus": brier,
        "passed": bool(all(gates.values())),
        "failed_gates": [k for k, v in gates.items() if not v],
        "gates": gates,
    }


def fit_mean_shift(
    y: np.ndarray,
    raw: np.ndarray,
    *,
    nonnegative: bool,
) -> Callable[[np.ndarray], np.ndarray]:
    delta = float(np.mean(y - raw))

    def transform(x: np.ndarray) -> np.ndarray:
        out = np.asarray(x, dtype="float64") + delta
        return np.maximum(out, 0.0) if nonnegative else out

    transform.delta = delta  # type: ignore[attr-defined]
    return transform


def fit_scale_to_mean(
    y: np.ndarray,
    raw: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    denom = float(np.mean(np.maximum(raw, 0.0)))
    numer = float(np.mean(np.maximum(y, 0.0)))
    ratio = 1.0 if denom <= 1e-12 else numer / denom
    ratio = float(np.clip(ratio, 0.25, 4.0))

    def transform(x: np.ndarray) -> np.ndarray:
        return np.maximum(np.asarray(x, dtype="float64") * ratio, 0.0)

    transform.ratio = ratio  # type: ignore[attr-defined]
    return transform


def fit_partial_mean_shift(
    y: np.ndarray,
    raw: np.ndarray,
    alpha: float,
    *,
    nonnegative: bool,
) -> Callable[[np.ndarray], np.ndarray]:
    full_delta = float(np.mean(y - raw))
    delta = float(alpha * full_delta)

    def transform(x: np.ndarray) -> np.ndarray:
        out = np.asarray(x, dtype="float64") + delta
        return np.maximum(out, 0.0) if nonnegative else out

    transform.delta = delta  # type: ignore[attr-defined]
    return transform


def choose_pre2025_candidate(
    target: str,
    y_val: np.ndarray,
    raw_val: np.ndarray,
    baseline_val: np.ndarray,
    calibration: dict[str, Any],
    threshold: dict[str, Any],
) -> tuple[str, Callable[[np.ndarray], np.ndarray], list[dict[str, Any]]]:
    nonnegative = target not in {"passing_yards", "rushing_yards", "receiving_yards"}

    existing_val = current_production_point(raw_val, calibration)
    p1_val = (
        current_probability_1plus(raw_val, calibration)
        if target in COUNT_TARGETS
        else None
    )

    candidates: list[tuple[str, Callable[[np.ndarray], np.ndarray], dict[str, Any]]] = []

    def identity(x: np.ndarray) -> np.ndarray:
        out = np.asarray(x, dtype="float64")
        return np.maximum(out, 0.0) if nonnegative else out

    candidates.append(("raw_identity", identity, {"complexity": 0}))

    # Mean-bias corrections. These directly address the acceptance bias gate
    # without fitting a flexible nonlinear mapping.
    for alpha in (0.25, 0.50, 0.75, 1.00):
        fn = fit_partial_mean_shift(
            y_val,
            raw_val,
            alpha,
            nonnegative=nonnegative,
        )
        candidates.append((
            f"mean_shift_{alpha:.2f}",
            fn,
            {
                "complexity": 1,
                "delta": float(getattr(fn, "delta")),
            },
        ))

    if nonnegative:
        fn = fit_scale_to_mean(y_val, raw_val)
        candidates.append((
            "scale_to_mean",
            fn,
            {
                "complexity": 1,
                "ratio": float(getattr(fn, "ratio")),
            },
        ))

    # Shrink the existing production calibration toward the raw model.
    # alpha=1 is current production; alpha=0 is identity.
    for alpha in (0.25, 0.50, 0.75, 1.00):
        def make_blend(a: float):
            def transform(x: np.ndarray) -> np.ndarray:
                x = np.asarray(x, dtype="float64")
                current = current_production_point(x, calibration)
                out = x + a * (current - x)
                return np.maximum(out, 0.0) if nonnegative else out
            return transform

        candidates.append((
            f"existing_calibration_blend_{alpha:.2f}",
            make_blend(alpha),
            {"complexity": 2, "alpha": alpha},
        ))

    evaluated: list[dict[str, Any]] = []
    passing: list[tuple[float, float, int, str, Callable[[np.ndarray], np.ndarray]]] = []

    for name, fn, meta in candidates:
        pred = fn(raw_val)
        metrics = point_gates(
            target,
            y_val,
            pred,
            baseline_val,
            threshold,
            p1=p1_val,
        )
        record = {
            "candidate": name,
            **meta,
            **{k: v for k, v in metrics.items() if k not in {"gates"}},
        }
        evaluated.append(record)
        if metrics["passed"]:
            passing.append((
                metrics["mae"],
                metrics["abs_bias"],
                int(meta["complexity"]),
                name,
                fn,
            ))

    if passing:
        passing.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
        _, _, _, chosen_name, chosen_fn = passing[0]
        return chosen_name, chosen_fn, evaluated

    # No candidate clears every 2024 gate. Choose nothing rather than silently
    # forcing a repair.
    return "NO_PRE2025_CANDIDATE_PASSES", identity, evaluated


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    tmp = Path(handle.name)
    handle.close()
    try:
        frame.to_csv(tmp, index=False, lineterminator="\n")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    tmp = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, indent=2, allow_nan=False, default=str)
            handle.write("\n")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def main() -> int:
    if not PREDICTIONS.is_file():
        raise FileNotFoundError(PREDICTIONS)

    thresholds = load_yaml(THRESHOLDS)
    frame = pd.read_parquet(PREDICTIONS)

    required = {
        "split", "season", "target", "actual",
        "baseline_projection",
        "direct_projection",
        "component_projection",
        "blend_projection",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction audit missing columns: {missing}")

    all_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}

    print("CALIBRATION REPAIR ANALYSIS")
    print("selection_data=2024_only")
    print("reporting_data=2025")
    print("production_files_modified=false")

    for target in TARGETS:
        selected = load_json(HERE / "models" / target / "selected_model.json")
        calibration = load_json(
            HERE / "models" / "calibration" / f"{target}_calibration.json"
        )
        architecture = clean(
            selected.get("selected_architecture")
            or selected.get("selected_candidate")
        )
        selected_col = SELECTED_PROJECTION_COLUMNS[architecture]

        val = frame.loc[
            frame["split"].astype(str).eq("validation")
            & pd.to_numeric(frame["season"], errors="coerce").eq(2024)
            & frame["target"].astype(str).eq(target)
        ].copy()

        test = frame.loc[
            frame["split"].astype(str).eq("test")
            & pd.to_numeric(frame["season"], errors="coerce").eq(2025)
            & frame["target"].astype(str).eq(target)
        ].copy()

        if val.empty or test.empty:
            raise ValueError(f"{target}: missing 2024 validation or 2025 test rows")

        def arrays(piece: pd.DataFrame):
            y = numeric(piece["actual"])
            raw = numeric(piece[selected_col])
            baseline = numeric(piece["baseline_projection"])
            valid = np.isfinite(y) & np.isfinite(raw) & np.isfinite(baseline)
            return y[valid], raw[valid], baseline[valid]

        yv, rv, bv = arrays(val)
        yt, rt, bt = arrays(test)

        threshold = thresholds[target]
        chosen_name, chosen_fn, candidates = choose_pre2025_candidate(
            target,
            yv,
            rv,
            bv,
            calibration,
            threshold,
        )

        raw_test_p1 = (
            current_probability_1plus(rt, calibration)
            if target in COUNT_TARGETS
            else None
        )
        chosen_test = chosen_fn(rt)
        test_metrics = point_gates(
            target,
            yt,
            chosen_test,
            bt,
            threshold,
            p1=raw_test_p1,
        )

        current_test = current_production_point(rt, calibration)
        current_metrics = point_gates(
            target,
            yt,
            current_test,
            bt,
            threshold,
            p1=raw_test_p1,
        )
        raw_metrics = point_gates(
            target,
            yt,
            rt,
            bt,
            threshold,
            p1=raw_test_p1,
        )

        chosen_2024 = next(
            (
                row for row in candidates
                if row["candidate"] == chosen_name
            ),
            None,
        )

        for row in candidates:
            all_rows.append({
                "target": target,
                "selection_window": 2024,
                **row,
            })

        summary[target] = {
            "selected_architecture": architecture,
            "chosen_from_2024_only": chosen_name,
            "chosen_2024_metrics": chosen_2024,
            "raw_2025": raw_metrics,
            "current_production_2025": current_metrics,
            "chosen_repair_2025": test_metrics,
            "calibration_only_repair_clears_2025": bool(test_metrics["passed"]),
        }

        status = "PASS" if test_metrics["passed"] else "FAIL"
        print(
            f"{target}: {status} "
            f"repair={chosen_name} "
            f"mae={test_metrics['mae']:.6f} "
            f"bias={test_metrics['bias']:.6f} "
            f"improvement={test_metrics['improvement_pct']:.6f}% "
            f"failed={';'.join(test_metrics['failed_gates']) or 'none'}"
        )

    out_frame = pd.DataFrame(all_rows)
    atomic_csv(CSV_OUT, out_frame)

    cleared = [
        target for target in TARGETS
        if summary[target]["calibration_only_repair_clears_2025"]
    ]
    still_blocked = [
        target for target in TARGETS
        if not summary[target]["calibration_only_repair_clears_2025"]
    ]

    payload = {
        "status": "complete",
        "policy": {
            "repair_method_selected_using_2024_only": True,
            "2025_used_for_repair_selection": False,
            "2025_used_for_reporting_only": True,
            "production_files_modified": False,
            "thresholds_modified": False,
            "registry_modified": False,
        },
        "targets": summary,
        "cleared_by_pre2025_calibration_repair": cleared,
        "still_needs_model_or_feature_work": still_blocked,
        "outputs": {
            "csv": str(CSV_OUT),
            "json": str(JSON_OUT),
        },
    }
    atomic_json(JSON_OUT, payload)

    print("cleared=" + (",".join(cleared) if cleared else "none"))
    print("still_blocked=" + (",".join(still_blocked) if still_blocked else "none"))
    print(f"csv={CSV_OUT}")
    print(f"json={JSON_OUT}")
    print("CALIBRATION REPAIR ANALYSIS: COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
