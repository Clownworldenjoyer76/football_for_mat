#!/usr/bin/env python3
"""Independent acceptance validation for Prop Engine Issue 25."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import common


SELECTION = HERE / "evaluation/model_selection.csv"
AUDIT = HERE / "evaluation/model_selection_predictions.parquet"
BASELINE = HERE / "evaluation/baseline_oof_predictions.parquet"
FOLDS = HERE / "evaluation/backtest_folds.parquet"
THRESHOLDS = HERE / "config/acceptance_thresholds.yaml"

# RUSHING_YARDS_ROBUST_GATE_SELECTION
RUSHING_YARDS_ROBUST_GATE_SELECTION = True
POINT_PREDICTION_BLEND_CANDIDATES = (0.0, 0.25, 0.50, 0.75, 1.0)

CANDIDATES = [
    "baseline",
    "direct",
    "component",
    "direct_component_blend",
]
EXPECTED_HEADERS = [
    "target",
    "candidate",
    "validation_mae",
    "validation_rmse",
    "validation_median_ae",
    "validation_poisson_deviance",
    "validation_brier_1plus",
    "validation_logloss_1plus",
    "test_mae",
    "test_rmse",
    "test_poisson_deviance",
    "selected_flag",
    "selection_reason",
]
GRAIN = ["season", "week", "game_id", "player_id"]


def numeric(series: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .astype("float64")
    )


def is_count(config: dict[str, Any], target: str) -> bool:
    return str(config["targets"][target].get("type", "")) == "count_nonnegative"


def metrics(
    actual: pd.Series,
    prediction: pd.Series,
    *,
    count_target: bool,
) -> dict[str, float | None]:
    y = numeric(actual).to_numpy(dtype="float64")
    p = numeric(prediction).to_numpy(dtype="float64")
    if len(y) == 0 or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise AssertionError("Metric inputs must be complete and finite.")
    err = np.abs(y - p)
    result: dict[str, float | None] = {
        "mae": float(np.mean(err)),
        "rmse": float(np.sqrt(np.mean(np.square(y - p)))),
        "median_ae": float(np.median(err)),
        "poisson_deviance": None,
        "brier_1plus": None,
        "logloss_1plus": None,
    }
    if count_target:
        if np.any(y < 0.0):
            raise AssertionError("Count target contains negative actual.")
        lam = np.maximum(p, 1e-12)
        terms = np.where(
            y > 0.0,
            y * np.log(y / lam) - (y - lam),
            lam,
        )
        result["poisson_deviance"] = float(2.0 * np.mean(terms))
        prob = np.clip(
            1.0 - np.exp(-np.maximum(p, 0.0)),
            1e-12,
            1.0 - 1e-12,
        )
        event = (y >= 1.0).astype(float)
        result["brier_1plus"] = float(np.mean(np.square(prob - event)))
        result["logloss_1plus"] = float(
            -np.mean(
                event * np.log(prob)
                + (1.0 - event) * np.log(1.0 - prob)
            )
        )
    return result


def close(actual: Any, expected: Any, tol: float = 1e-9) -> bool:
    if expected is None:
        return pd.isna(actual)
    if pd.isna(actual):
        return False
    return math.isclose(float(actual), float(expected), rel_tol=tol, abs_tol=tol)


def resolve_policy(folds: pd.DataFrame) -> dict[str, Any]:
    test = folds.loc[folds["test_flag"].eq(1)]
    assert len(test) == 1, "Expected exactly one test fold"
    test_row = test.iloc[0]
    validation_season = int(test_row["train_end_season"])
    development = folds.loc[
        folds["test_flag"].eq(0)
        & folds["validation_start_season"].eq(validation_season)
        & folds["validation_end_season"].eq(validation_season)
    ]
    assert len(development) == 1, "Expected exactly one final development fold"
    dev = development.iloc[0]
    return {
        "validation_fold_id": str(dev["fold_id"]),
        "validation_season": validation_season,
        "selection_train_end_season": int(dev["train_end_season"]),
        "test_fold_id": str(test_row["fold_id"]),
        "test_season": int(test_row["validation_start_season"]),
    }


def best_blend_weight(
    frame: pd.DataFrame,
    *,
    target: str,
    acceptance: dict[str, Any],
) -> float:
    y = frame["actual"].to_numpy(dtype=float)
    d = frame["direct_projection"].to_numpy(dtype=float)
    c = frame["component_projection"].to_numpy(dtype=float)

    if target != "rushing_yards":
        best_weight = 0.0
        best_key = None
        for weight in np.linspace(0.0, 1.0, 101):
            pred = weight * d + (1.0 - weight) * c
            err = np.abs(y - pred)
            key = (
                float(np.mean(err)),
                float(np.sqrt(np.mean(np.square(y - pred)))),
                float(np.median(err)),
                abs(float(weight) - 0.5),
                float(weight),
            )
            if best_key is None or key < best_key:
                best_key = key
                best_weight = float(weight)
        return best_weight

    baseline = frame["baseline_projection"].to_numpy(dtype=float)
    baseline_mae = float(np.mean(np.abs(baseline - y)))
    max_bias = float(acceptance["maximum_allowed_bias"])
    max_mae = float(acceptance["maximum_validation_mae"])
    min_improvement = float(
        acceptance["minimum_improvement_vs_baseline_pct"]
    )

    passing = []
    for weight in np.linspace(0.0, 1.0, 101):
        raw = weight * d + (1.0 - weight) * c
        q50 = float(np.quantile(y - raw, 0.50))
        base = raw + q50
        for alpha in POINT_PREDICTION_BLEND_CANDIDATES:
            pred = raw + float(alpha) * (base - raw)
            candidate_mae = float(np.mean(np.abs(pred - y)))
            candidate_bias = float(np.mean(pred - y))
            abs_bias = abs(candidate_bias)
            improvement = (
                (baseline_mae - candidate_mae) / baseline_mae * 100.0
            )
            if (
                candidate_mae > max_mae + 1e-12
                or abs_bias > max_bias + 1e-12
                or improvement + 1e-12 < min_improvement
            ):
                continue
            robust_margin = min(
                (max_bias - abs_bias) / max_bias,
                (max_mae - candidate_mae) / max_mae,
                (improvement - min_improvement)
                / max(abs(min_improvement), 1.0),
            )
            passing.append(
                (
                    -robust_margin,
                    candidate_mae,
                    abs_bias,
                    float(alpha),
                    abs(float(weight) - 0.5),
                    float(weight),
                )
            )

    assert passing, "rushing_yards: no locked-gate passing 2024 candidate"
    passing.sort()
    return float(passing[0][5])


def main() -> int:
    config = common.load_config()
    thresholds = yaml.safe_load(THRESHOLDS.read_text(encoding="utf-8-sig"))
    assert isinstance(thresholds, dict), "Acceptance thresholds must be a mapping"
    targets = list(config["targets"].keys())

    assert SELECTION.is_file(), f"Missing {SELECTION}"
    assert AUDIT.is_file(), f"Missing {AUDIT}"

    selection = pd.read_csv(SELECTION)
    assert selection.columns.tolist() == EXPECTED_HEADERS, (
        f"Header mismatch: {selection.columns.tolist()}"
    )
    assert len(selection) == len(targets) * len(CANDIDATES), "Unexpected selection row count"
    assert set(selection["target"]) == set(targets), "Target coverage mismatch"

    for target in targets:
        group = selection.loc[selection["target"].eq(target)]
        assert group["candidate"].tolist() == CANDIDATES, f"{target}: candidate order mismatch"
        assert int(group["selected_flag"].sum()) == 1, f"{target}: selected_flag count != 1"
        assert group["selection_reason"].fillna("").str.len().gt(0).all(), f"{target}: blank reason"

    folds = pd.read_parquet(FOLDS)
    policy = resolve_policy(folds)
    baseline = pd.read_parquet(BASELINE)
    audit = pd.read_parquet(AUDIT)

    required_audit = {
        "split",
        "fold_id",
        *GRAIN,
        "target",
        "actual",
        "baseline_projection",
        "direct_projection",
        "component_projection",
        "blend_projection",
        "blend_direct_weight",
        "blend_component_weight",
        "direct_variant",
    }
    assert required_audit.issubset(audit.columns), "Audit columns missing"
    common.ensure_unique(audit, ["split", *GRAIN, "target"], "Issue 25 audit")

    for split, fold_id, season in [
        ("validation", policy["validation_fold_id"], policy["validation_season"]),
        ("test", policy["test_fold_id"], policy["test_season"]),
    ]:
        part = audit.loc[audit["split"].eq(split)].copy()
        assert set(part["season"].astype(int)) == {season}, f"{split}: wrong season"
        assert set(part["fold_id"].astype(str)) == {fold_id}, f"{split}: wrong fold id"
        assert set(part["target"].astype(str)) == set(targets), f"{split}: targets incomplete"

        source = baseline.loc[baseline["fold_id"].astype(str).eq(fold_id)].copy()
        source["actual"] = numeric(source["actual"])
        source["baseline_projection"] = numeric(source["baseline_projection"])
        source = source.loc[
            source["actual"].notna()
            & source["baseline_projection"].notna()
        ].copy()
        check = part.merge(
            source[[*GRAIN, "target", "actual", "baseline_projection"]],
            on=[*GRAIN, "target"],
            how="outer",
            validate="one_to_one",
            suffixes=("_audit", "_source"),
            indicator=True,
        )
        assert check["_merge"].eq("both").all(), (
            f"{split}: audit/evaluable-baseline row mismatch"
        )
        assert np.allclose(
            check["actual_audit"].to_numpy(dtype=float),
            check["actual_source"].to_numpy(dtype=float),
            rtol=0.0,
            atol=0.0,
        ), f"{split}: actual differs from baseline source"
        assert np.allclose(
            check["baseline_projection_audit"].to_numpy(dtype=float),
            check["baseline_projection_source"].to_numpy(dtype=float),
            rtol=0.0,
            atol=0.0,
        ), f"{split}: baseline projection differs from source"

    projection_column = {
        "baseline": "baseline_projection",
        "direct": "direct_projection",
        "component": "component_projection",
        "direct_component_blend": "blend_projection",
    }

    for target in targets:
        valid = audit.loc[
            audit["split"].eq("validation")
            & audit["target"].eq(target)
        ].copy()
        test = audit.loc[
            audit["split"].eq("test")
            & audit["target"].eq(target)
        ].copy()
        assert not valid.empty and not test.empty, f"{target}: empty audit window"

        for column in [
            "actual",
            "baseline_projection",
            "direct_projection",
            "component_projection",
            "blend_projection",
        ]:
            assert numeric(valid[column]).notna().all(), f"{target}: validation {column} missing"
            assert numeric(test[column]).notna().all(), f"{target}: test {column} missing"

        weights = valid["blend_direct_weight"].drop_duplicates().tolist()
        assert len(weights) == 1, f"{target}: validation blend weight not constant"
        weight = float(weights[0])
        assert target in thresholds, f"{target}: missing acceptance thresholds"
        expected_weight = best_blend_weight(
            valid,
            target=target,
            acceptance=thresholds[target],
        )
        assert math.isclose(weight, expected_weight, abs_tol=1e-12), (
            f"{target}: blend weight {weight} != validation optimum {expected_weight}"
        )
        test_weights = test["blend_direct_weight"].drop_duplicates().tolist()
        assert test_weights == weights, f"{target}: test changed blend weight"
        assert np.allclose(
            valid["blend_projection"].to_numpy(dtype=float),
            weight * valid["direct_projection"].to_numpy(dtype=float)
            + (1.0 - weight) * valid["component_projection"].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        ), f"{target}: validation blend formula mismatch"
        assert np.allclose(
            test["blend_projection"].to_numpy(dtype=float),
            weight * test["direct_projection"].to_numpy(dtype=float)
            + (1.0 - weight) * test["component_projection"].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        ), f"{target}: test blend formula mismatch"

        recomputed_valid: dict[str, dict[str, float | None]] = {}
        for candidate in CANDIDATES:
            v = metrics(
                valid["actual"],
                valid[projection_column[candidate]],
                count_target=is_count(config, target),
            )
            t = metrics(
                test["actual"],
                test[projection_column[candidate]],
                count_target=is_count(config, target),
            )
            recomputed_valid[candidate] = v
            row = selection.loc[
                selection["target"].eq(target)
                & selection["candidate"].eq(candidate)
            ].iloc[0]
            comparisons = {
                "validation_mae": v["mae"],
                "validation_rmse": v["rmse"],
                "validation_median_ae": v["median_ae"],
                "validation_poisson_deviance": v["poisson_deviance"],
                "validation_brier_1plus": v["brier_1plus"],
                "validation_logloss_1plus": v["logloss_1plus"],
                "test_mae": t["mae"],
                "test_rmse": t["rmse"],
                "test_poisson_deviance": t["poisson_deviance"],
            }
            for column, expected in comparisons.items():
                assert close(row[column], expected), (
                    f"{target}/{candidate}: {column} mismatch; "
                    f"csv={row[column]} expected={expected}"
                )

        selected_expected = min(
            CANDIDATES,
            key=lambda candidate: (
                float(recomputed_valid[candidate]["mae"]),
                float(recomputed_valid[candidate]["rmse"]),
                float(recomputed_valid[candidate]["median_ae"]),
                CANDIDATES.index(candidate),
            ),
        )
        selected_row = selection.loc[
            selection["target"].eq(target)
            & selection["selected_flag"].eq(1)
        ].iloc[0]
        assert selected_row["candidate"] == selected_expected, (
            f"{target}: selected candidate not validation optimum"
        )

        selected_path = HERE / "models" / target / "selected_model.json"
        assert selected_path.is_file(), f"{target}: missing selected_model.json"
        payload = json.loads(selected_path.read_text(encoding="utf-8"))
        assert payload["target"] == target
        assert payload["selected_architecture"] == selected_expected
        assert payload["test_reporting_only"] is True
        assert payload["test_used_for_selection"] is False
        assert payload["selection_frozen_before_test_reporting"] is True
        assert payload["comparison_row_policy"] == (
            "common_evaluable_rows; missing baseline projections excluded; "
            "no missing projection is imputed to zero"
        )
        assert payload["market_features_used"] is False
        assert int(payload["validation_season"]) == policy["validation_season"]
        assert int(payload["test_season"]) == policy["test_season"]
        assert math.isclose(
            float(payload["blend_weights"]["direct"]),
            weight,
            abs_tol=1e-12,
        )
        assert payload["blend_weights"]["selected_from_validation_only"] is True

        variants = set(valid["direct_variant"].astype(str))
        assert len(variants) == 1, f"{target}: direct variant changed within validation"
        assert set(test["direct_variant"].astype(str)) == variants, (
            f"{target}: direct variant changed for test"
        )
        assert payload["direct_variant"]["variant"] in variants
        assert payload["direct_variant"]["selected_on_validation_only"] is True

    print("ISSUE 25 ACCEPTANCE: PASS")
    print(
        json.dumps(
            {
                "targets": targets,
                "rows": int(len(selection)),
                "validation_season": policy["validation_season"],
                "test_season": policy["test_season"],
                "audit_rows": int(len(audit)),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
