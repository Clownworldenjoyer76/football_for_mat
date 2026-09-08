#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROP = Path(__file__).resolve().parent
AUDIT = PROP / "evaluation/model_selection_predictions.parquet"
THRESHOLDS = PROP / "config/acceptance_thresholds.yaml"
CALIBRATION = PROP / "models/calibration/sacks_calibration.json"
SELECTED = PROP / "models/sacks/selected_model.json"
OUT_CSV = PROP / "evaluation/issue56_sacks_point_diagnostic.csv"
OUT_JSON = PROP / "evaluation/issue56_sacks_point_diagnostic.json"

TARGET = "sacks"
VALIDATION_SEASON = 2024
TEST_SEASON = 2025

REQ_COLUMNS = [
    "split",
    "fold_id",
    "season",
    "week",
    "game_id",
    "player_id",
    "target",
    "actual",
    "baseline_projection",
    "direct_projection",
    "component_projection",
    "blend_projection",
    "blend_direct_weight",
    "blend_component_weight",
]


def finite(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
    return values


def apply_mapping(x: np.ndarray, mapping: dict) -> np.ndarray:
    xp = np.asarray(mapping["knots_x"], dtype="float64")
    fp = np.asarray(mapping["knots_y"], dtype="float64")
    if len(xp) == 0 or len(fp) == 0 or len(xp) != len(fp):
        raise AssertionError("Invalid calibration mapping knots")
    out = np.interp(
        np.asarray(x, dtype="float64"),
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
    return np.maximum(out, 0.0)


def metrics(actual: np.ndarray, pred: np.ndarray, baseline: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype="float64")
    pred = np.maximum(np.asarray(pred, dtype="float64"), 0.0)
    baseline = np.asarray(baseline, dtype="float64")
    ok = np.isfinite(actual) & np.isfinite(pred) & np.isfinite(baseline)
    if not bool(ok.all()):
        raise AssertionError(f"Nonfinite metric input rows: {int((~ok).sum())}")
    mae = float(np.mean(np.abs(actual - pred)))
    bias = float(np.mean(pred - actual))
    baseline_mae = float(np.mean(np.abs(actual - baseline)))
    improvement = (
        100.0 * (baseline_mae - mae) / baseline_mae
        if baseline_mae > 0
        else float("nan")
    )
    lam = np.maximum(pred, 1e-12)
    terms = np.where(
        actual > 0.0,
        actual * np.log(actual / lam) - (actual - lam),
        lam,
    )
    poisson = float(2.0 * np.mean(terms))
    return {
        "rows": int(len(actual)),
        "mae": mae,
        "bias": bias,
        "abs_bias": abs(bias),
        "baseline_mae": baseline_mae,
        "improvement_vs_baseline_pct": improvement,
        "poisson_deviance": poisson,
    }


def gates(m: dict, t: dict) -> list[str]:
    failed = []
    if m["mae"] > float(t["maximum_validation_mae"]):
        failed.append("mae")
    if m["abs_bias"] > float(t["maximum_allowed_bias"]):
        failed.append("bias")
    if m["improvement_vs_baseline_pct"] < float(t["minimum_improvement_vs_baseline_pct"]):
        failed.append("improvement")
    if "maximum_poisson_deviance" in t:
        if m["poisson_deviance"] > float(t["maximum_poisson_deviance"]):
            failed.append("poisson_deviance")
    return failed


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as h:
        v = json.load(h)
    if not isinstance(v, dict):
        raise AssertionError(f"Expected JSON object: {path}")
    return v


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as h:
        v = yaml.safe_load(h)
    if not isinstance(v, dict):
        raise AssertionError(f"Expected YAML mapping: {path}")
    return v


def frame_for(audit: pd.DataFrame, split: str, season: int) -> pd.DataFrame:
    f = audit.loc[
        audit["target"].astype(str).eq(TARGET)
        & audit["split"].astype(str).eq(split)
    ].copy()
    if f.empty:
        raise AssertionError(f"No {TARGET} rows for split={split}")
    seasons = set(pd.to_numeric(f["season"], errors="raise").astype(int).unique())
    if seasons != {season}:
        raise AssertionError(f"{split}: expected season {season}, found {sorted(seasons)}")
    for col in ["actual", "baseline_projection", "direct_projection", "component_projection"]:
        vals = pd.to_numeric(f[col], errors="coerce")
        if vals.isna().any() or not np.isfinite(vals.to_numpy(dtype="float64")).all():
            raise AssertionError(f"{split}: nonfinite {col}")
        f[col] = vals.astype("float64")
    return f.reset_index(drop=True)


def build_candidate_predictions(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    calibration: dict,
) -> list[dict]:
    va = validation
    te = test

    v_direct = va["direct_projection"].to_numpy(dtype="float64")
    t_direct = te["direct_projection"].to_numpy(dtype="float64")
    v_component = va["component_projection"].to_numpy(dtype="float64")
    t_component = te["component_projection"].to_numpy(dtype="float64")

    expected_mapping = (
        calibration.get("count_calibration", {})
        .get("expected_count", {})
        .get("mapping")
    )
    if not isinstance(expected_mapping, dict):
        raise AssertionError("sacks calibration expected_count mapping missing")

    v_cal = apply_mapping(np.maximum(v_direct, 0.0), expected_mapping)
    t_cal = apply_mapping(np.maximum(t_direct, 0.0), expected_mapping)

    candidates: list[dict] = []

    def add(name: str, family: str, params: dict, vp: np.ndarray, tp: np.ndarray):
        candidates.append({
            "name": name,
            "family": family,
            "params": params,
            "validation_prediction": np.maximum(np.asarray(vp, dtype="float64"), 0.0),
            "test_prediction": np.maximum(np.asarray(tp, dtype="float64"), 0.0),
        })

    add("identity_direct", "identity", {}, v_direct, t_direct)
    add("component", "component", {}, v_component, t_component)
    add("existing_expected_count_calibration", "existing_calibration", {}, v_cal, t_cal)

    # Convex direct/component candidates. This is the same architectural family
    # already permitted by Issue 25, evaluated here at finer granularity for diagnosis.
    for i in range(0, 101):
        w = i / 100.0
        add(
            f"direct_component_w{w:.2f}",
            "direct_component_blend",
            {"direct_weight": w, "component_weight": 1.0 - w},
            w * v_direct + (1.0 - w) * v_component,
            w * t_direct + (1.0 - w) * t_component,
        )

    # Point-calibration blend between the frozen selected direct point and the
    # existing 2024-fitted expected-count calibration mapping.
    for i in range(0, 101):
        w = i / 100.0
        add(
            f"direct_calibrated_w{w:.2f}",
            "direct_calibration_blend",
            {"raw_direct_weight": w, "calibrated_weight": 1.0 - w},
            w * v_direct + (1.0 - w) * v_cal,
            w * t_direct + (1.0 - w) * t_cal,
        )

    # Multiplicative shrink/expand family. The scale is selected strictly on 2024.
    for i in range(500, 1501, 5):
        scale = i / 1000.0
        add(
            f"direct_scale_{scale:.3f}",
            "multiplicative_scale",
            {"scale": scale},
            scale * v_direct,
            scale * t_direct,
        )

    # Additive correction family. Again selected strictly on 2024.
    for i in range(-100, 101, 2):
        offset = i / 1000.0
        add(
            f"direct_offset_{offset:+.3f}",
            "additive_offset",
            {"offset": offset},
            v_direct + offset,
            t_direct + offset,
        )

    return candidates


def main() -> int:
    for path in [AUDIT, THRESHOLDS, CALIBRATION, SELECTED]:
        if not path.is_file():
            raise FileNotFoundError(path)

    selected = load_json(SELECTED)
    if selected.get("target") != TARGET:
        raise AssertionError("selected_model target mismatch")
    if selected.get("selected_architecture") != "direct":
        raise AssertionError(
            f"Expected current sacks architecture=direct, found {selected.get('selected_architecture')}"
        )
    if selected.get("validation_season") != VALIDATION_SEASON:
        raise AssertionError("selected_model validation_season mismatch")
    if selected.get("test_season") != TEST_SEASON:
        raise AssertionError("selected_model test_season mismatch")
    if selected.get("test_used_for_selection") is not False:
        raise AssertionError("selected_model says test was used for selection")
    if selected.get("selection_frozen_before_test_reporting") is not True:
        raise AssertionError("selected_model selection freeze flag is not true")
    if selected.get("market_features_used") is not False:
        raise AssertionError("selected_model market_features_used must be false")

    calibration = load_json(CALIBRATION)
    source = calibration.get("calibration_source", {})
    reporting = calibration.get("reporting_test_policy", {})
    if source.get("season") != VALIDATION_SEASON:
        raise AssertionError("calibration source is not 2024")
    if reporting.get("test_rows_used_for_calibration") is not False:
        raise AssertionError("calibration says test rows were used")
    if calibration.get("market_features_used") is not False:
        raise AssertionError("calibration market_features_used must be false")

    threshold_map = load_yaml(THRESHOLDS)
    threshold = dict(threshold_map[TARGET])

    audit = pd.read_parquet(AUDIT, columns=REQ_COLUMNS)
    validation = frame_for(audit, "validation", VALIDATION_SEASON)
    test = frame_for(audit, "test", TEST_SEASON)

    # Diagnostic candidates are built from frozen audit/calibration artifacts only.
    candidates = build_candidate_predictions(validation, test, calibration)

    v_actual = validation["actual"].to_numpy(dtype="float64")
    v_baseline = validation["baseline_projection"].to_numpy(dtype="float64")
    t_actual = test["actual"].to_numpy(dtype="float64")
    t_baseline = test["baseline_projection"].to_numpy(dtype="float64")

    rows = []
    candidate_payload = []

    for candidate in candidates:
        vm = metrics(v_actual, candidate["validation_prediction"], v_baseline)
        vf = gates(vm, threshold)
        tm = metrics(t_actual, candidate["test_prediction"], t_baseline)
        tf = gates(tm, threshold)

        row = {
            "name": candidate["name"],
            "family": candidate["family"],
            "params_json": json.dumps(candidate["params"], sort_keys=True),
            "validation_mae": vm["mae"],
            "validation_bias": vm["bias"],
            "validation_improvement_pct": vm["improvement_vs_baseline_pct"],
            "validation_poisson_deviance": vm["poisson_deviance"],
            "validation_failed": ";".join(vf) if vf else "",
            "validation_pass": len(vf) == 0,
            "test_mae_reporting_only": tm["mae"],
            "test_bias_reporting_only": tm["bias"],
            "test_improvement_pct_reporting_only": tm["improvement_vs_baseline_pct"],
            "test_poisson_deviance_reporting_only": tm["poisson_deviance"],
            "test_failed_reporting_only": ";".join(tf) if tf else "",
            "test_pass_reporting_only": len(tf) == 0,
        }
        rows.append(row)
        candidate_payload.append({
            "name": candidate["name"],
            "family": candidate["family"],
            "params": candidate["params"],
            "validation_metrics": vm,
            "validation_failed": vf,
            "test_metrics_reporting_only": tm,
            "test_failed_reporting_only": tf,
        })

    table = pd.DataFrame(rows)

    # Selection is STRICTLY validation-only:
    # 1) Prefer candidates passing every point gate on 2024.
    # 2) Otherwise rank by number of failed gates, MAE, abs bias, then name.
    passing = table.loc[table["validation_pass"]].copy()
    if not passing.empty:
        chosen_row = passing.sort_values(
            ["validation_mae", "name"],
            kind="mergesort",
        ).iloc[0]
        selection_status = "validation_gate_pass_candidate_found"
    else:
        ranked = table.copy()
        ranked["_fail_count"] = ranked["validation_failed"].map(
            lambda x: 0 if not x else len(str(x).split(";"))
        )
        ranked["_abs_bias"] = ranked["validation_bias"].abs()
        chosen_row = ranked.sort_values(
            ["_fail_count", "validation_mae", "_abs_bias", "name"],
            kind="mergesort",
        ).iloc[0]
        selection_status = "no_candidate_passed_2024_point_gates"

    chosen_name = str(chosen_row["name"])
    chosen = next(x for x in candidate_payload if x["name"] == chosen_name)

    table = table.sort_values(
        ["validation_pass", "validation_mae", "name"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    table.to_csv(OUT_CSV, index=False)

    payload = {
        "status": "complete",
        "target": TARGET,
        "diagnostic_is_read_only": True,
        "market_data_used": False,
        "selection_policy": {
            "selection_season": VALIDATION_SEASON,
            "test_season": TEST_SEASON,
            "test_used_for_candidate_selection": False,
            "candidate_selection": (
                "prefer all 2024 point-gate passers; otherwise fewest failed gates, "
                "then lowest 2024 MAE, then lowest absolute 2024 bias"
            ),
        },
        "thresholds_unchanged": threshold,
        "selected_model_contract": {
            "selected_architecture": selected.get("selected_architecture"),
            "test_used_for_selection": selected.get("test_used_for_selection"),
            "selection_frozen_before_test_reporting": selected.get(
                "selection_frozen_before_test_reporting"
            ),
        },
        "candidate_families": sorted(table["family"].unique().tolist()),
        "candidate_count": int(len(table)),
        "selection_status": selection_status,
        "chosen_from_2024_only": chosen,
        "top_10_validation": table.head(10).to_dict(orient="records"),
        "outputs": {
            "csv": OUT_CSV.as_posix(),
            "json": OUT_JSON.as_posix(),
        },
    }
    OUT_JSON.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("ISSUE 56 SACKS READ-ONLY POINT DIAGNOSTIC")
    print(f"validation_rows={len(validation)} test_rows={len(test)}")
    print(
        "thresholds: "
        f"mae<={float(threshold['maximum_validation_mae']):.6f} "
        f"abs_bias<={float(threshold['maximum_allowed_bias']):.6f} "
        f"improvement>={float(threshold['minimum_improvement_vs_baseline_pct']):.6f}% "
        f"poisson<={float(threshold['maximum_poisson_deviance']):.6f}"
    )
    print(f"candidates={len(table)}")
    print(f"selection_status={selection_status}")
    print(
        "chosen_2024_only: "
        f"name={chosen['name']} family={chosen['family']} params={json.dumps(chosen['params'], sort_keys=True)}"
    )
    vm = chosen["validation_metrics"]
    tm = chosen["test_metrics_reporting_only"]
    print(
        "2024: "
        f"mae={vm['mae']:.6f} bias={vm['bias']:.6f} "
        f"improvement={vm['improvement_vs_baseline_pct']:.6f}% "
        f"poisson={vm['poisson_deviance']:.6f} "
        f"failed={';'.join(chosen['validation_failed']) if chosen['validation_failed'] else 'none'}"
    )
    print(
        "2025_REPORTING_ONLY: "
        f"mae={tm['mae']:.6f} bias={tm['bias']:.6f} "
        f"improvement={tm['improvement_vs_baseline_pct']:.6f}% "
        f"poisson={tm['poisson_deviance']:.6f} "
        f"failed={';'.join(chosen['test_failed_reporting_only']) if chosen['test_failed_reporting_only'] else 'none'}"
    )

    print("TOP 10 BY 2024 VALIDATION:")
    for row in table.head(10).itertuples(index=False):
        print(
            f"{row.name}: family={row.family} "
            f"mae={row.validation_mae:.6f} "
            f"bias={row.validation_bias:.6f} "
            f"improvement={row.validation_improvement_pct:.6f}% "
            f"poisson={row.validation_poisson_deviance:.6f} "
            f"failed={row.validation_failed or 'none'}"
        )

    print(f"csv={OUT_CSV}")
    print(f"json={OUT_JSON}")
    print("ISSUE 56 SACKS READ-ONLY POINT DIAGNOSTIC: COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
