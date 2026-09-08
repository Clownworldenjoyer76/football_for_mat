#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
PRED = HERE / "evaluation" / "model_selection_predictions.parquet"
THRESHOLDS = HERE / "config" / "acceptance_thresholds.yaml"
OUT_CSV = HERE / "evaluation" / "final_three_repair_candidates.csv"
OUT_JSON = HERE / "evaluation" / "final_three_repair_analysis.json"

TARGETS = ["rushing_tds", "receiving_tds", "sacks"]

# Candidate architecture blends are chosen on 2024 only.
DIRECT_WEIGHTS = [i / 20.0 for i in range(21)]  # 0.00 .. 1.00

# Small calibration families, also chosen on 2024 only.
SHIFT_STRENGTHS = [0.0, 0.25, 0.50, 0.75, 1.0]
SCALE_STRENGTHS = [0.0, 0.25, 0.50, 0.75, 1.0]


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(p) - np.asarray(y))))


def bias(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.asarray(p) - np.asarray(y)))


def poisson_deviance(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype="float64")
    lam = np.maximum(np.asarray(p, dtype="float64"), 1e-12)
    if np.any(y < 0):
        raise ValueError("Negative actual encountered.")
    terms = np.empty_like(y)
    zero = y <= 0.0
    terms[zero] = lam[zero]
    nz = ~zero
    terms[nz] = y[nz] * np.log(y[nz] / lam[nz]) - (y[nz] - lam[nz])
    return float(2.0 * np.mean(terms))


def brier_1plus(y: np.ndarray, p: np.ndarray) -> float:
    event = (np.asarray(y, dtype="float64") >= 1.0).astype("float64")
    prob = 1.0 - np.exp(-np.maximum(np.asarray(p, dtype="float64"), 0.0))
    return float(np.mean((prob - event) ** 2))


def evaluate(
    y: np.ndarray,
    p: np.ndarray,
    baseline: np.ndarray,
    gate: dict[str, Any],
) -> dict[str, Any]:
    p = np.maximum(np.asarray(p, dtype="float64"), 0.0)
    m = mae(y, p)
    b = bias(y, p)
    ab = abs(b)
    bm = mae(y, baseline)
    improvement = (bm - m) / bm * 100.0 if bm > 0 else float("-inf")
    br = brier_1plus(y, p)
    pdv = poisson_deviance(y, p)

    checks = {
        "mae": m <= float(gate["maximum_validation_mae"]) + 1e-12,
        "bias": ab <= float(gate["maximum_allowed_bias"]) + 1e-12,
        "improvement": (
            improvement + 1e-12
            >= float(gate["minimum_improvement_vs_baseline_pct"])
        ),
        "brier": br <= float(gate["maximum_brier_1plus"]) + 1e-12,
        "poisson": pdv <= float(gate["maximum_poisson_deviance"]) + 1e-12,
    }

    return {
        "mae": m,
        "bias": b,
        "absolute_bias": ab,
        "baseline_mae": bm,
        "improvement_vs_baseline_pct": improvement,
        "brier_1plus": br,
        "poisson_deviance": pdv,
        "passed": bool(all(checks.values())),
        "failed_gates": [k for k, v in checks.items() if not v],
    }


def transform_candidate(
    direct: np.ndarray,
    component: np.ndarray,
    *,
    direct_weight: float,
    shift: float,
    scale: float,
) -> np.ndarray:
    base = (
        float(direct_weight) * np.asarray(direct, dtype="float64")
        + (1.0 - float(direct_weight)) * np.asarray(component, dtype="float64")
    )
    out = np.maximum(base * float(scale) + float(shift), 0.0)
    return out


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
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


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    tmp = Path(h.name)
    try:
        with h:
            json.dump(payload, h, indent=2, allow_nan=False, default=str)
            h.write("\n")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def arrays(frame: pd.DataFrame) -> tuple[np.ndarray, ...]:
    cols = [
        "actual",
        "baseline_projection",
        "direct_projection",
        "component_projection",
    ]
    series = [
        pd.to_numeric(frame[c], errors="coerce").to_numpy(dtype="float64")
        for c in cols
    ]
    valid = np.ones(len(frame), dtype=bool)
    for arr in series:
        valid &= np.isfinite(arr)
    return tuple(arr[valid] for arr in series)


def choose_candidate(
    y: np.ndarray,
    baseline: np.ndarray,
    direct: np.ndarray,
    component: np.ndarray,
    gate: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    # Derive small correction ranges from 2024 only.
    component_bias = bias(y, component)
    direct_bias = bias(y, direct)

    base_candidates: list[tuple[float, float, float]] = []

    for w in DIRECT_WEIGHTS:
        blend = w * direct + (1.0 - w) * component
        blend_bias = bias(y, blend)

        # Shift candidates: fractions of the exact 2024 bias correction.
        shift_target = -blend_bias
        shifts = [s * shift_target for s in SHIFT_STRENGTHS]

        # Scale candidates: fractions of a 2024 mean-ratio correction.
        blend_mean = float(np.mean(np.maximum(blend, 0.0)))
        actual_mean = float(np.mean(np.maximum(y, 0.0)))
        full_ratio = 1.0 if blend_mean <= 1e-12 else actual_mean / blend_mean
        scales = [
            1.0 + s * (full_ratio - 1.0)
            for s in SCALE_STRENGTHS
        ]

        # Identity + shift-only + scale-only + modest scale/shift combinations.
        for shift in shifts:
            base_candidates.append((w, shift, 1.0))
        for scale in scales:
            base_candidates.append((w, 0.0, scale))
        for shift in shifts:
            for scale in scales:
                base_candidates.append((w, shift, scale))

    # Deduplicate rounded candidate specs.
    seen: set[tuple[float, float, float]] = set()
    candidates: list[dict[str, Any]] = []
    passing: list[dict[str, Any]] = []

    for w, shift, scale in base_candidates:
        key = (round(w, 8), round(shift, 12), round(scale, 12))
        if key in seen:
            continue
        seen.add(key)

        pred = transform_candidate(
            direct,
            component,
            direct_weight=w,
            shift=shift,
            scale=scale,
        )
        m = evaluate(y, pred, baseline, gate)
        row = {
            "direct_weight": float(w),
            "component_weight": float(1.0 - w),
            "shift": float(shift),
            "scale": float(scale),
            **m,
        }
        candidates.append(row)
        if m["passed"]:
            passing.append(row)

    if not passing:
        return None, candidates

    # 2024-only selection: lowest MAE, then abs bias, then smallest correction,
    # then smallest direct weight for deterministic tie-breaking.
    passing.sort(
        key=lambda r: (
            float(r["mae"]),
            float(r["absolute_bias"]),
            abs(float(r["shift"])) + abs(float(r["scale"]) - 1.0),
            float(r["direct_weight"]),
        )
    )
    return passing[0], candidates


def main() -> int:
    if not PRED.is_file():
        raise FileNotFoundError(PRED)

    thresholds = load_yaml(THRESHOLDS)
    pred = pd.read_parquet(PRED)

    required = {
        "split",
        "season",
        "target",
        "actual",
        "baseline_projection",
        "direct_projection",
        "component_projection",
    }
    missing = sorted(required - set(pred.columns))
    if missing:
        raise ValueError(f"Prediction audit missing columns: {missing}")

    print("FINAL THREE REPAIR ANALYSIS")
    print("candidate_selection_data=2024_only")
    print("2025_used_for_reporting_only=true")
    print("production_files_modified=false")

    all_candidates: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}

    for target in TARGETS:
        gate = thresholds[target]

        val = pred.loc[
            pred["split"].astype(str).eq("validation")
            & pd.to_numeric(pred["season"], errors="coerce").eq(2024)
            & pred["target"].astype(str).eq(target)
        ].copy()
        test = pred.loc[
            pred["split"].astype(str).eq("test")
            & pd.to_numeric(pred["season"], errors="coerce").eq(2025)
            & pred["target"].astype(str).eq(target)
        ].copy()

        if val.empty or test.empty:
            raise ValueError(f"{target}: missing 2024 validation or 2025 test rows")

        yv, bv, dv, cv = arrays(val)
        yt, bt, dt, ct = arrays(test)

        chosen, candidates = choose_candidate(yv, bv, dv, cv, gate)

        for row in candidates:
            all_candidates.append(
                {
                    "target": target,
                    "selection_season": 2024,
                    **row,
                }
            )

        raw_component_2025 = evaluate(yt, ct, bt, gate)
        raw_direct_2025 = evaluate(yt, dt, bt, gate)

        if chosen is None:
            test_result = None
            status = "NO_2024_CANDIDATE_PASSES"
            print(
                f"{target}: MODEL_WORK_REQUIRED "
                f"component_2025_mae={raw_component_2025['mae']:.6f} "
                f"direct_2025_mae={raw_direct_2025['mae']:.6f}"
            )
        else:
            pt = transform_candidate(
                dt,
                ct,
                direct_weight=float(chosen["direct_weight"]),
                shift=float(chosen["shift"]),
                scale=float(chosen["scale"]),
            )
            test_result = evaluate(yt, pt, bt, gate)
            status = "PASS_2025" if test_result["passed"] else "FAIL_2025"
            print(
                f"{target}: {status} "
                f"w_direct={chosen['direct_weight']:.2f} "
                f"shift={chosen['shift']:.6f} "
                f"scale={chosen['scale']:.6f} "
                f"mae={test_result['mae']:.6f} "
                f"bias={test_result['bias']:.6f} "
                f"improvement={test_result['improvement_vs_baseline_pct']:.6f}% "
                f"failed={';'.join(test_result['failed_gates']) or 'none'}"
            )

        summary[target] = {
            "chosen_using_2024_only": chosen,
            "2025_result_of_chosen_candidate": test_result,
            "raw_component_2025": raw_component_2025,
            "raw_direct_2025": raw_direct_2025,
            "status": status,
        }

    atomic_csv(OUT_CSV, pd.DataFrame(all_candidates))
    atomic_json(
        OUT_JSON,
        {
            "status": "complete",
            "policy": {
                "candidate_selection_data": 2024,
                "2025_used_for_selection": False,
                "2025_reporting_only": True,
                "thresholds_modified": False,
                "registry_modified": False,
                "models_modified": False,
            },
            "targets": summary,
            "csv": str(OUT_CSV),
            "json": str(OUT_JSON),
        },
    )

    print(f"csv={OUT_CSV}")
    print(f"json={OUT_JSON}")
    print("FINAL THREE REPAIR ANALYSIS: COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
