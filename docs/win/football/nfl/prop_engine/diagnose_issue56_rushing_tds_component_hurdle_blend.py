#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

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

OUT_CSV = PROP / "evaluation/issue56_rushing_tds_component_hurdle_blend_diagnostic.csv"
OUT_JSON = PROP / "evaluation/issue56_rushing_tds_component_hurdle_blend_diagnostic.json"

HURDLE_COLUMNS = [
    "unit_positive_severity_expected_count",
    "training_positive_mean_severity_expected_count",
]

WEIGHTS = np.arange(0.0, 1.0001, 0.001)


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
        raise AssertionError("Missing production probability calibration mappings")

    calibrated_lambda = np.maximum(
        apply_mapping(np.maximum(raw_component, 0.0), expected_map),
        0.0,
    )
    poisson_p1 = 1.0 - np.exp(-calibrated_lambda)

    return np.clip(
        apply_mapping(poisson_p1, p1_map),
        0.0,
        1.0,
    )


def metrics(
    actual: np.ndarray,
    point: np.ndarray,
    baseline: np.ndarray,
    p1: np.ndarray,
) -> dict:
    y = np.asarray(actual, dtype="float64")
    p = np.maximum(np.asarray(point, dtype="float64"), 0.0)
    b = np.asarray(baseline, dtype="float64")
    q = np.clip(np.asarray(p1, dtype="float64"), 0.0, 1.0)

    ok = np.isfinite(y) & np.isfinite(p) & np.isfinite(b) & np.isfinite(q)
    if not ok.all():
        raise AssertionError(f"Nonfinite metric rows: {int((~ok).sum())}")

    mae = float(np.mean(np.abs(y - p)))
    bias = float(np.mean(p - y))

    baseline_mae = float(np.mean(np.abs(y - b)))
    improvement = (
        100.0 * (baseline_mae - mae) / baseline_mae
        if baseline_mae > 0.0 else float("nan")
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
    poisson = float(2.0 * np.mean(terms))

    event = (y >= 1.0).astype(float)
    brier = float(np.mean(np.square(q - event)))

    return {
        "rows": int(len(y)),
        "mae": mae,
        "bias": bias,
        "abs_bias": abs(bias),
        "baseline_mae": baseline_mae,
        "improvement_vs_baseline_pct": float(improvement),
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
    if m["improvement_vs_baseline_pct"] < float(
        t["minimum_improvement_vs_baseline_pct"]
    ):
        failed.append("improvement")
    if m["brier_1plus"] > float(t["maximum_brier_1plus"]):
        failed.append("brier_1plus")
    if m["poisson_deviance"] > float(t["maximum_poisson_deviance"]):
        failed.append("poisson_deviance")

    return failed


def preflight() -> dict:
    for path in [AUDIT, HURDLE, CALIBRATION, SELECTED, THRESHOLDS]:
        if not path.is_file():
            raise FileNotFoundError(path)

    selected = load_json(SELECTED)
    if selected.get("selected_architecture") != "component":
        raise AssertionError(
            f"Expected current rushing_tds architecture=component; "
            f"found {selected.get('selected_architecture')}"
        )
    if selected.get("validation_season") != VALIDATION_SEASON:
        raise AssertionError("selected validation season mismatch")
    if selected.get("test_season") != TEST_SEASON:
        raise AssertionError("selected test season mismatch")
    if selected.get("test_used_for_selection") is not False:
        raise AssertionError("selected_model test_used_for_selection must be false")
    if selected.get("market_features_used") is not False:
        raise AssertionError("selected_model market_features_used must be false")

    calibration = load_json(CALIBRATION)
    source = calibration.get("calibration_source", {})
    reporting = calibration.get("reporting_test_policy", {})

    if source.get("season") != VALIDATION_SEASON:
        raise AssertionError("calibration source season is not 2024")
    if reporting.get("test_rows_used_for_calibration") is not False:
        raise AssertionError("calibration used 2025 test rows")
    if calibration.get("market_features_used") is not False:
        raise AssertionError("calibration market_features_used must be false")

    hurdle = pd.read_csv(HURDLE)
    required_hurdle = {
        "split_label",
        *GRAIN,
        "actual",
        "baseline_projection",
        "hurdle_p1",
        *HURDLE_COLUMNS,
    }
    missing = sorted(required_hurdle - set(hurdle.columns))
    if missing:
        raise AssertionError(f"Hurdle prediction CSV missing columns: {missing}")

    if hurdle.duplicated(["split_label", *GRAIN]).any():
        raise AssertionError("Duplicate hurdle prediction grain")

    expected_labels = {
        "validation_2024": VALIDATION_SEASON,
        "test_2025_reporting_only": TEST_SEASON,
    }
    hurdle_counts = {}

    for label, season in expected_labels.items():
        f = hurdle.loc[hurdle["split_label"].astype(str).eq(label)].copy()
        if f.empty:
            raise AssertionError(f"No hurdle rows for {label}")

        found = set(pd.to_numeric(f["season"], errors="raise").astype(int).unique())
        if found != {season}:
            raise AssertionError(f"{label}: unexpected seasons {sorted(found)}")

        for c in ["actual", "baseline_projection", "hurdle_p1", *HURDLE_COLUMNS]:
            v = pd.to_numeric(f[c], errors="coerce")
            if v.isna().any() or not np.isfinite(v.to_numpy(dtype=float)).all():
                raise AssertionError(f"{label}: invalid {c}")

        hurdle_counts[label] = int(len(f))

    audit = pd.read_parquet(
        AUDIT,
        columns=[
            "split",
            "season",
            "week",
            "game_id",
            "player_id",
            "target",
            "actual",
            "baseline_projection",
            "component_projection",
        ],
    )

    audit_counts = {}
    for split, season in [("validation", VALIDATION_SEASON), ("test", TEST_SEASON)]:
        f = audit.loc[
            audit["target"].astype(str).eq(TARGET)
            & audit["split"].astype(str).eq(split)
        ].copy()

        if f.empty:
            raise AssertionError(f"No audit rows for {split}")

        found = set(pd.to_numeric(f["season"], errors="raise").astype(int).unique())
        if found != {season}:
            raise AssertionError(f"{split}: unexpected seasons {sorted(found)}")

        if f.duplicated(GRAIN).any():
            raise AssertionError(f"{split}: duplicate audit grain")

        audit_counts[split] = int(len(f))

    if hurdle_counts["validation_2024"] != audit_counts["validation"]:
        raise AssertionError("Validation row count mismatch: hurdle vs audit")
    if hurdle_counts["test_2025_reporting_only"] != audit_counts["test"]:
        raise AssertionError("Test row count mismatch: hurdle vs audit")

    info = {
        "validation_rows": audit_counts["validation"],
        "test_rows": audit_counts["test"],
        "validation_season": VALIDATION_SEASON,
        "test_season": TEST_SEASON,
        "test_used_for_selection": False,
        "candidate_hurdle_paths": HURDLE_COLUMNS,
        "blend_weights": int(len(WEIGHTS)),
        "production_probability_path_unchanged": True,
        "market_data_used": False,
    }

    print("PREFLIGHT PASS: raw-component/hurdle blend inputs and chronology verified")
    print(json.dumps(info, sort_keys=True))
    return info


def aligned_split(
    audit: pd.DataFrame,
    hurdle: pd.DataFrame,
    split: str,
    hurdle_label: str,
    season: int,
) -> pd.DataFrame:
    a = audit.loc[
        audit["target"].astype(str).eq(TARGET)
        & audit["split"].astype(str).eq(split)
    ].copy()

    h = hurdle.loc[
        hurdle["split_label"].astype(str).eq(hurdle_label)
    ].copy()

    merged = a.merge(
        h[
            GRAIN
            + [
                "actual",
                "baseline_projection",
                "hurdle_p1",
                *HURDLE_COLUMNS,
            ]
        ].rename(
            columns={
                "actual": "hurdle_actual",
                "baseline_projection": "hurdle_baseline_projection",
            }
        ),
        on=GRAIN,
        how="left",
        validate="one_to_one",
    )

    if merged["hurdle_actual"].isna().any():
        n = int(merged["hurdle_actual"].isna().sum())
        raise AssertionError(f"{split}: missing hurdle rows={n}")

    for c in [
        "actual",
        "baseline_projection",
        "component_projection",
        "hurdle_actual",
        "hurdle_baseline_projection",
        "hurdle_p1",
        *HURDLE_COLUMNS,
    ]:
        merged[c] = pd.to_numeric(merged[c], errors="raise").astype(float)

    if not np.allclose(
        merged["actual"].to_numpy(),
        merged["hurdle_actual"].to_numpy(),
        rtol=0.0,
        atol=0.0,
    ):
        raise AssertionError(f"{split}: hurdle actual != audit actual")

    if not np.allclose(
        merged["baseline_projection"].to_numpy(),
        merged["hurdle_baseline_projection"].to_numpy(),
        rtol=0.0,
        atol=0.0,
    ):
        raise AssertionError(f"{split}: hurdle baseline != audit baseline")

    found = set(pd.to_numeric(merged["season"], errors="raise").astype(int).unique())
    if found != {season}:
        raise AssertionError(f"{split}: aligned season mismatch {sorted(found)}")

    return merged.reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    preflight_info = preflight()

    if args.preflight_only:
        print("ISSUE 56 RUSHING TDS COMPONENT/HURDLE BLEND PREFLIGHT: PASS")
        return 0

    threshold = dict(load_yaml(THRESHOLDS)[TARGET])
    calibration = load_json(CALIBRATION)

    audit = pd.read_parquet(AUDIT)
    hurdle = pd.read_csv(HURDLE)

    validation = aligned_split(
        audit,
        hurdle,
        "validation",
        "validation_2024",
        VALIDATION_SEASON,
    )
    test = aligned_split(
        audit,
        hurdle,
        "test",
        "test_2025_reporting_only",
        TEST_SEASON,
    )

    vr = validation["component_projection"].to_numpy(dtype=float)
    tr = test["component_projection"].to_numpy(dtype=float)

    vy = validation["actual"].to_numpy(dtype=float)
    ty = test["actual"].to_numpy(dtype=float)

    vb = validation["baseline_projection"].to_numpy(dtype=float)
    tb = test["baseline_projection"].to_numpy(dtype=float)

    vp1 = production_p1(vr, calibration)
    tp1 = production_p1(tr, calibration)

    raw_v = metrics(vy, vr, vb, vp1)
    raw_t = metrics(ty, tr, tb, tp1)

    rows = []
    payload_candidates = []

    for hurdle_column in HURDLE_COLUMNS:
        vh = validation[hurdle_column].to_numpy(dtype=float)
        th = test[hurdle_column].to_numpy(dtype=float)

        for weight in WEIGHTS:
            w = float(weight)

            vp = (1.0 - w) * vr + w * vh
            tp = (1.0 - w) * tr + w * th

            vm = metrics(vy, vp, vb, vp1)
            tm = metrics(ty, tp, tb, tp1)

            vf = failures(vm, threshold)
            tf = failures(tm, threshold)

            name = f"{hurdle_column}__hurdle_weight_{w:.3f}"

            rows.append({
                "name": name,
                "hurdle_path": hurdle_column,
                "hurdle_weight": w,
                "raw_component_weight": 1.0 - w,
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
            })

            payload_candidates.append({
                "name": name,
                "hurdle_path": hurdle_column,
                "hurdle_weight": w,
                "raw_component_weight": 1.0 - w,
                "validation_metrics": vm,
                "validation_failed": vf,
                "test_metrics_reporting_only": tm,
                "test_failed_reporting_only": tf,
            })

    table = pd.DataFrame(rows)

    passing = table.loc[table["validation_pass"]].copy()

    if not passing.empty:
        chosen_row = passing.sort_values(
            [
                "validation_mae",
                "validation_poisson_deviance",
                "hurdle_path",
                "hurdle_weight",
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
            [
                "_fail_count",
                "validation_mae",
                "_abs_bias",
                "validation_poisson_deviance",
                "hurdle_path",
                "hurdle_weight",
            ],
            kind="mergesort",
        ).iloc[0]
        selection_status = "no_candidate_passed_2024_gates"

    chosen_name = str(chosen_row["name"])
    chosen = next(
        c for c in payload_candidates
        if c["name"] == chosen_name
    )

    table = table.sort_values(
        [
            "validation_pass",
            "validation_mae",
            "validation_poisson_deviance",
            "validation_bias",
            "name",
        ],
        ascending=[False, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    table.to_csv(OUT_CSV, index=False)

    payload = {
        "status": "complete",
        "target": TARGET,
        "diagnostic_only": True,
        "production_files_modified": False,
        "market_data_used": False,
        "architecture_family": (
            "convex point blend of current raw component and frozen hurdle expected count"
        ),
        "production_probability_path_unchanged": True,
        "selection_policy": {
            "validation_season": VALIDATION_SEASON,
            "test_season": TEST_SEASON,
            "test_used_for_selection": False,
            "weight_grid": {
                "minimum": 0.0,
                "maximum": 1.0,
                "step": 0.001,
            },
            "hurdle_paths": HURDLE_COLUMNS,
        },
        "thresholds_unchanged": threshold,
        "preflight": preflight_info,
        "raw_component": {
            "validation_2024": raw_v,
            "validation_failed": failures(raw_v, threshold),
            "test_2025_reporting_only": raw_t,
            "test_failed_reporting_only": failures(raw_t, threshold),
        },
        "candidate_count": int(len(table)),
        "selection_status": selection_status,
        "chosen_from_2024_only": chosen,
        "top_20_validation": table.head(20).to_dict(orient="records"),
        "outputs": {
            "csv": str(OUT_CSV),
            "json": str(OUT_JSON),
        },
    }

    OUT_JSON.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("ISSUE 56 RUSHING TDS COMPONENT/HURDLE BLEND DIAGNOSTIC")
    print(
        f"candidates={len(table)} "
        f"selection_status={selection_status}"
    )

    print(
        "RAW 2024: "
        f"mae={raw_v['mae']:.6f} "
        f"bias={raw_v['bias']:.6f} "
        f"improvement={raw_v['improvement_vs_baseline_pct']:.6f}% "
        f"poisson={raw_v['poisson_deviance']:.6f} "
        f"brier={raw_v['brier_1plus']:.6f} "
        f"failed={';'.join(failures(raw_v, threshold)) or 'none'}"
    )

    vm = chosen["validation_metrics"]
    tm = chosen["test_metrics_reporting_only"]

    print(
        "CHOSEN_2024_ONLY: "
        f"path={chosen['hurdle_path']} "
        f"hurdle_weight={chosen['hurdle_weight']:.3f} "
        f"raw_weight={chosen['raw_component_weight']:.3f}"
    )

    print(
        "2024 CHOSEN: "
        f"mae={vm['mae']:.6f} "
        f"bias={vm['bias']:.6f} "
        f"improvement={vm['improvement_vs_baseline_pct']:.6f}% "
        f"poisson={vm['poisson_deviance']:.6f} "
        f"brier={vm['brier_1plus']:.6f} "
        f"failed={';'.join(chosen['validation_failed']) or 'none'}"
    )

    print(
        "2025 REPORTING ONLY: "
        f"mae={tm['mae']:.6f} "
        f"bias={tm['bias']:.6f} "
        f"improvement={tm['improvement_vs_baseline_pct']:.6f}% "
        f"poisson={tm['poisson_deviance']:.6f} "
        f"brier={tm['brier_1plus']:.6f} "
        f"failed={';'.join(chosen['test_failed_reporting_only']) or 'none'}"
    )

    print("TOP 10 BY 2024 VALIDATION:")
    for row in table.head(10).itertuples(index=False):
        print(
            f"{row.hurdle_path} w={row.hurdle_weight:.3f}: "
            f"mae={row.validation_mae:.6f} "
            f"bias={row.validation_bias:.6f} "
            f"improvement={row.validation_improvement_pct:.6f}% "
            f"poisson={row.validation_poisson_deviance:.6f} "
            f"failed={row.validation_failed or 'none'}"
        )

    print(f"csv={OUT_CSV}")
    print(f"json={OUT_JSON}")
    print("ISSUE 56 RUSHING TDS COMPONENT/HURDLE BLEND DIAGNOSTIC: COMPLETE")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
