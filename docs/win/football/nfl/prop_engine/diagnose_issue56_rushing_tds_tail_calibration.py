#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROP = Path(__file__).resolve().parent
AUDIT = PROP / "evaluation/model_selection_predictions.parquet"
THRESHOLDS = PROP / "config/acceptance_thresholds.yaml"
CALIBRATION = PROP / "models/calibration/rushing_tds_calibration.json"
SELECTED = PROP / "models/rushing_tds/selected_model.json"
OUT_CSV = PROP / "evaluation/issue56_rushing_tds_tail_calibration_diagnostic.csv"
OUT_JSON = PROP / "evaluation/issue56_rushing_tds_tail_calibration_diagnostic.json"

TARGET = "rushing_tds"
VALIDATION_SEASON = 2024
TEST_SEASON = 2025

QUANTILES = [
    0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90,
    0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99,
]
TAIL_ADD_DELTAS = np.arange(0.005, 0.505, 0.005)
TAIL_SCALES = np.arange(1.05, 6.05, 0.05)
CONTINUOUS_SCALES = np.arange(1.10, 10.10, 0.10)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as h:
        value = json.load(h)
    if not isinstance(value, dict):
        raise AssertionError(f"Expected JSON object: {path}")
    return value


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as h:
        value = yaml.safe_load(h)
    if not isinstance(value, dict):
        raise AssertionError(f"Expected YAML mapping: {path}")
    return value


def apply_mapping(x: np.ndarray, mapping: dict) -> np.ndarray:
    xp = np.asarray(mapping["knots_x"], dtype="float64")
    fp = np.asarray(mapping["knots_y"], dtype="float64")
    if len(xp) == 0 or len(fp) == 0 or len(xp) != len(fp):
        raise AssertionError("Invalid calibration mapping")
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
    return out


def production_p1(raw_component: np.ndarray, calibration: dict) -> np.ndarray:
    cc = calibration.get("count_calibration", {})
    expected_map = cc.get("expected_count", {}).get("mapping")
    p1_map = cc.get("probability_1_plus", {}).get("mapping")
    if not isinstance(expected_map, dict) or not isinstance(p1_map, dict):
        raise AssertionError("Missing count probability calibration mappings")
    lam = np.maximum(apply_mapping(np.maximum(raw_component, 0.0), expected_map), 0.0)
    poisson_p1 = 1.0 - np.exp(-lam)
    return np.clip(apply_mapping(poisson_p1, p1_map), 0.0, 1.0)


def preflight() -> dict:
    for p in [AUDIT, THRESHOLDS, CALIBRATION, SELECTED]:
        if not p.is_file():
            raise FileNotFoundError(p)

    selected = load_json(SELECTED)
    if selected.get("selected_architecture") != "component":
        raise AssertionError(
            f"Expected rushing_tds architecture=component; found "
            f"{selected.get('selected_architecture')}"
        )
    if selected.get("validation_season") != VALIDATION_SEASON:
        raise AssertionError("selected validation season mismatch")
    if selected.get("test_season") != TEST_SEASON:
        raise AssertionError("selected test season mismatch")
    if selected.get("test_used_for_selection") is not False:
        raise AssertionError("selected model uses test for selection")
    if selected.get("market_features_used") is not False:
        raise AssertionError("selected model market_features_used must be false")

    calibration = load_json(CALIBRATION)
    source = calibration.get("calibration_source", {})
    reporting = calibration.get("reporting_test_policy", {})
    if source.get("season") != VALIDATION_SEASON:
        raise AssertionError("count calibration source is not 2024")
    if reporting.get("test_rows_used_for_calibration") is not False:
        raise AssertionError("count calibration used test rows")
    if calibration.get("market_features_used") is not False:
        raise AssertionError("calibration market_features_used must be false")

    audit = pd.read_parquet(
        AUDIT,
        columns=["split", "season", "target", "actual", "baseline_projection", "component_projection"],
    )
    rt = audit.loc[audit["target"].astype(str).eq(TARGET)].copy()
    counts = {}
    for split, season in [("validation", VALIDATION_SEASON), ("test", TEST_SEASON)]:
        f = rt.loc[rt["split"].astype(str).eq(split)].copy()
        if f.empty:
            raise AssertionError(f"No {TARGET} rows for {split}")
        found = set(pd.to_numeric(f["season"], errors="raise").astype(int).unique())
        if found != {season}:
            raise AssertionError(f"{split}: unexpected seasons {sorted(found)}")
        for c in ["actual", "baseline_projection", "component_projection"]:
            v = pd.to_numeric(f[c], errors="coerce")
            if v.isna().any() or not np.isfinite(v.to_numpy(dtype=float)).all():
                raise AssertionError(f"{split}: invalid {c}")
        counts[split] = int(len(f))

    threshold = dict(load_yaml(THRESHOLDS)[TARGET])
    info = {
        "validation_rows": counts["validation"],
        "test_rows": counts["test"],
        "validation_season": VALIDATION_SEASON,
        "test_season": TEST_SEASON,
        "test_used_for_selection": False,
        "point_only_candidate": True,
        "production_probability_path_unchanged": True,
        "market_data_used": False,
        "thresholds": threshold,
    }
    print("PREFLIGHT PASS: tail-calibration contracts verified")
    print(json.dumps(info, sort_keys=True))
    return info


def split_frame(audit: pd.DataFrame, split: str, season: int) -> pd.DataFrame:
    f = audit.loc[
        audit["target"].astype(str).eq(TARGET)
        & audit["split"].astype(str).eq(split)
    ].copy()
    if f.empty:
        raise AssertionError(f"No {split} rows")
    found = set(pd.to_numeric(f["season"], errors="raise").astype(int).unique())
    if found != {season}:
        raise AssertionError(f"{split}: unexpected seasons {sorted(found)}")
    for c in ["actual", "baseline_projection", "component_projection"]:
        f[c] = pd.to_numeric(f[c], errors="raise").astype(float)
    return f.reset_index(drop=True)


def metrics(
    actual: np.ndarray,
    point: np.ndarray,
    baseline: np.ndarray,
    p1: np.ndarray,
) -> dict:
    y = np.asarray(actual, dtype=float)
    p = np.maximum(np.asarray(point, dtype=float), 0.0)
    b = np.asarray(baseline, dtype=float)
    q = np.clip(np.asarray(p1, dtype=float), 0.0, 1.0)
    ok = np.isfinite(y) & np.isfinite(p) & np.isfinite(b) & np.isfinite(q)
    if not ok.all():
        raise AssertionError(f"Nonfinite metric rows: {int((~ok).sum())}")

    mae = float(np.mean(np.abs(y - p)))
    bias = float(np.mean(p - y))
    baseline_mae = float(np.mean(np.abs(y - b)))
    improvement = 100.0 * (baseline_mae - mae) / baseline_mae

    lam = np.maximum(p, 1e-12)
    terms = np.empty_like(y)
    zero = y <= 0.0
    terms[zero] = lam[zero]
    pos = ~zero
    terms[pos] = y[pos] * np.log(y[pos] / lam[pos]) - (y[pos] - lam[pos])
    poisson = float(2.0 * np.mean(terms))

    event = (y >= 1.0).astype(float)
    brier = float(np.mean(np.square(q - event)))

    return {
        "rows": int(len(y)),
        "mae": mae,
        "bias": bias,
        "abs_bias": abs(bias),
        "baseline_mae": baseline_mae,
        "improvement_vs_baseline_pct": improvement,
        "poisson_deviance": poisson,
        "brier_1plus": brier,
        "mean_actual": float(np.mean(y)),
        "mean_prediction": float(np.mean(p)),
    }


def failures(m: dict, t: dict) -> list[str]:
    failed = []
    if m["mae"] > float(t["maximum_validation_mae"]):
        failed.append("mae")
    if m["abs_bias"] > float(t["maximum_allowed_bias"]):
        failed.append("bias")
    if m["improvement_vs_baseline_pct"] < float(t["minimum_improvement_vs_baseline_pct"]):
        failed.append("improvement")
    if m["poisson_deviance"] > float(t["maximum_poisson_deviance"]):
        failed.append("poisson_deviance")
    if m["brier_1plus"] > float(t["maximum_brier_1plus"]):
        failed.append("brier_1plus")
    return failed


def apply_candidate(raw: np.ndarray, family: str, cutoff: float, parameter: float) -> np.ndarray:
    p = np.maximum(np.asarray(raw, dtype=float), 0.0)
    high = p >= cutoff
    out = p.copy()

    if family == "tail_add":
        out[high] = p[high] + parameter
    elif family == "tail_scale":
        out[high] = p[high] * parameter
    elif family == "continuous_tail_scale":
        # Continuous, monotone, leaves all values <= cutoff unchanged.
        out[high] = cutoff + parameter * (p[high] - cutoff)
    else:
        raise KeyError(family)
    return np.maximum(out, 0.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    preflight_info = preflight()
    if args.preflight_only:
        print("ISSUE 56 RUSHING TDS TAIL CALIBRATION PREFLIGHT: PASS")
        return 0

    threshold = dict(load_yaml(THRESHOLDS)[TARGET])
    calibration = load_json(CALIBRATION)
    audit = pd.read_parquet(AUDIT)
    validation = split_frame(audit, "validation", VALIDATION_SEASON)
    test = split_frame(audit, "test", TEST_SEASON)

    vr = validation["component_projection"].to_numpy(dtype=float)
    tr = test["component_projection"].to_numpy(dtype=float)
    vy = validation["actual"].to_numpy(dtype=float)
    ty = test["actual"].to_numpy(dtype=float)
    vb = validation["baseline_projection"].to_numpy(dtype=float)
    tb = test["baseline_projection"].to_numpy(dtype=float)

    # Probability path is intentionally frozen and unchanged by this point-only diagnostic.
    vp1 = production_p1(vr, calibration)
    tp1 = production_p1(tr, calibration)

    raw_v = metrics(vy, vr, vb, vp1)
    raw_t = metrics(ty, tr, tb, tp1)
    raw_v_failed = failures(raw_v, threshold)
    raw_t_failed = failures(raw_t, threshold)

    expected_map = calibration["count_calibration"]["expected_count"]["mapping"]
    vfull = np.maximum(apply_mapping(np.maximum(vr, 0.0), expected_map), 0.0)
    tfull = np.maximum(apply_mapping(np.maximum(tr, 0.0), expected_map), 0.0)
    full_v = metrics(vy, vfull, vb, vp1)
    full_t = metrics(ty, tfull, tb, tp1)

    # Freeze numeric cutoffs from the 2024 raw prediction distribution.
    cutoff_rows = []
    seen = set()
    for q in QUANTILES:
        cutoff = float(np.quantile(vr, q))
        key = round(cutoff, 15)
        if key in seen:
            continue
        seen.add(key)
        cutoff_rows.append((q, cutoff))

    rows = []
    payload_candidates = []

    def evaluate(family: str, q: float, cutoff: float, parameter: float):
        vp = apply_candidate(vr, family, cutoff, parameter)
        tp = apply_candidate(tr, family, cutoff, parameter)
        vm = metrics(vy, vp, vb, vp1)
        tm = metrics(ty, tp, tb, tp1)
        vf = failures(vm, threshold)
        tf = failures(tm, threshold)
        high_v = int(np.sum(vr >= cutoff))
        high_t = int(np.sum(tr >= cutoff))
        row = {
            "family": family,
            "quantile_source_2024": q,
            "frozen_cutoff": cutoff,
            "parameter": parameter,
            "validation_high_rows": high_v,
            "test_high_rows_reporting_only": high_t,
            "validation_mae": vm["mae"],
            "validation_bias": vm["bias"],
            "validation_improvement_pct": vm["improvement_vs_baseline_pct"],
            "validation_poisson_deviance": vm["poisson_deviance"],
            "validation_brier_1plus_unchanged_probability_path": vm["brier_1plus"],
            "validation_failed": ";".join(vf),
            "validation_pass": not vf,
            "test_mae_reporting_only": tm["mae"],
            "test_bias_reporting_only": tm["bias"],
            "test_improvement_pct_reporting_only": tm["improvement_vs_baseline_pct"],
            "test_poisson_deviance_reporting_only": tm["poisson_deviance"],
            "test_brier_1plus_reporting_only": tm["brier_1plus"],
            "test_failed_reporting_only": ";".join(tf),
            "test_pass_reporting_only": not tf,
        }
        rows.append(row)
        payload_candidates.append({
            "family": family,
            "quantile_source_2024": q,
            "frozen_cutoff": cutoff,
            "parameter": parameter,
            "validation_high_rows": high_v,
            "test_high_rows_reporting_only": high_t,
            "validation_metrics": vm,
            "validation_failed": vf,
            "test_metrics_reporting_only": tm,
            "test_failed_reporting_only": tf,
        })

    for q, cutoff in cutoff_rows:
        for delta in TAIL_ADD_DELTAS:
            evaluate("tail_add", q, cutoff, float(delta))
        for scale in TAIL_SCALES:
            evaluate("tail_scale", q, cutoff, float(scale))
        for scale in CONTINUOUS_SCALES:
            evaluate("continuous_tail_scale", q, cutoff, float(scale))

    table = pd.DataFrame(rows)
    passing = table.loc[table["validation_pass"]].copy()
    if not passing.empty:
        chosen_row = passing.sort_values(
            [
                "validation_mae",
                "validation_poisson_deviance",
                "validation_bias",
                "family",
                "quantile_source_2024",
                "parameter",
            ],
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
            ["_fail_count", "validation_mae", "_abs_bias", "validation_poisson_deviance"],
            kind="mergesort",
        ).iloc[0]
        selection_status = "no_candidate_passed_2024_gates"

    chosen = next(
        c for c in payload_candidates
        if c["family"] == chosen_row["family"]
        and abs(c["frozen_cutoff"] - float(chosen_row["frozen_cutoff"])) < 1e-15
        and abs(c["parameter"] - float(chosen_row["parameter"])) < 1e-12
    )

    table = table.sort_values(
        ["validation_pass", "validation_mae", "validation_poisson_deviance", "validation_bias"],
        ascending=[False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    table.to_csv(OUT_CSV, index=False)

    payload = {
        "status": "complete",
        "target": TARGET,
        "diagnostic_only": True,
        "production_files_modified": False,
        "market_data_used": False,
        "point_only_tail_calibration": True,
        "production_probability_path_unchanged": True,
        "selection_policy": {
            "cutoffs_derived_from": "2024 raw component prediction quantiles only",
            "candidate_metrics_selected_on": 2024,
            "test_season": 2025,
            "test_used_for_selection": False,
            "frozen_numeric_cutoff_applied_to_2025": True,
        },
        "rationale": (
            "raw component passes 2024 MAE/improvement but underpredicts mean; "
            "generic isotonic expected-count calibration raises low-risk zero rows via a positive floor"
        ),
        "generic_expected_count_left_floor": float(expected_map["left_value"]),
        "thresholds_unchanged": threshold,
        "preflight": preflight_info,
        "raw_component": {
            "validation_2024": raw_v,
            "validation_failed": raw_v_failed,
            "test_2025_reporting_only": raw_t,
            "test_failed_reporting_only": raw_t_failed,
        },
        "generic_full_expected_count_calibration": {
            "validation_2024": full_v,
            "test_2025_reporting_only": full_t,
        },
        "candidate_count": int(len(table)),
        "unique_2024_cutoffs": int(len(cutoff_rows)),
        "selection_status": selection_status,
        "chosen_from_2024_only": chosen,
        "top_20_validation": table.head(20).to_dict(orient="records"),
        "outputs": {"csv": str(OUT_CSV), "json": str(OUT_JSON)},
    }
    OUT_JSON.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("ISSUE 56 RUSHING TDS TAIL CALIBRATION DIAGNOSTIC")
    print(f"generic_expected_count_left_floor={float(expected_map['left_value']):.9f}")
    print(
        "RAW 2024: "
        f"mae={raw_v['mae']:.6f} bias={raw_v['bias']:.6f} "
        f"improvement={raw_v['improvement_vs_baseline_pct']:.6f}% "
        f"poisson={raw_v['poisson_deviance']:.6f} brier={raw_v['brier_1plus']:.6f} "
        f"failed={';'.join(raw_v_failed) if raw_v_failed else 'none'}"
    )
    print(
        "FULL GENERIC 2024: "
        f"mae={full_v['mae']:.6f} bias={full_v['bias']:.6f} "
        f"improvement={full_v['improvement_vs_baseline_pct']:.6f}% "
        f"poisson={full_v['poisson_deviance']:.6f} brier={full_v['brier_1plus']:.6f}"
    )
    print(
        f"candidates={len(table)} unique_2024_cutoffs={len(cutoff_rows)} "
        f"selection_status={selection_status}"
    )
    print(
        "chosen_2024_only: "
        f"family={chosen['family']} q={chosen['quantile_source_2024']:.2f} "
        f"frozen_cutoff={chosen['frozen_cutoff']:.9f} parameter={chosen['parameter']:.6f} "
        f"validation_high_rows={chosen['validation_high_rows']}"
    )
    vm = chosen["validation_metrics"]
    tm = chosen["test_metrics_reporting_only"]
    print(
        "2024 CHOSEN: "
        f"mae={vm['mae']:.6f} bias={vm['bias']:.6f} "
        f"improvement={vm['improvement_vs_baseline_pct']:.6f}% "
        f"poisson={vm['poisson_deviance']:.6f} brier={vm['brier_1plus']:.6f} "
        f"failed={';'.join(chosen['validation_failed']) if chosen['validation_failed'] else 'none'}"
    )
    print(
        "2025 REPORTING ONLY: "
        f"mae={tm['mae']:.6f} bias={tm['bias']:.6f} "
        f"improvement={tm['improvement_vs_baseline_pct']:.6f}% "
        f"poisson={tm['poisson_deviance']:.6f} brier={tm['brier_1plus']:.6f} "
        f"failed={';'.join(chosen['test_failed_reporting_only']) if chosen['test_failed_reporting_only'] else 'none'}"
    )
    print("TOP 10 BY 2024 VALIDATION:")
    for row in table.head(10).itertuples(index=False):
        print(
            f"{row.family} q={row.quantile_source_2024:.2f} cutoff={row.frozen_cutoff:.6f} "
            f"param={row.parameter:.4f}: mae={row.validation_mae:.6f} "
            f"bias={row.validation_bias:.6f} improvement={row.validation_improvement_pct:.6f}% "
            f"poisson={row.validation_poisson_deviance:.6f} "
            f"failed={row.validation_failed or 'none'}"
        )
    print(f"csv={OUT_CSV}")
    print(f"json={OUT_JSON}")
    print("ISSUE 56 RUSHING TDS TAIL CALIBRATION DIAGNOSTIC: COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
