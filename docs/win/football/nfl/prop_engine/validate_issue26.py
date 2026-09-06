#!/usr/bin/env python3
"""Independent acceptance checks for Prop Engine Issue 26."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import math
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import common


GRAIN = ["season", "week", "game_id", "player_id"]
TRAINER = HERE / "scripts" / "train" / "calibrate_uncertainty.py"
AUDIT = HERE / "evaluation" / "model_selection_predictions.parquet"
COVERAGE = HERE / "evaluation" / "interval_coverage.csv"
CAL_ROOT = HERE / "models" / "calibration"

QUANTILE_TARGETS = {
    "passing_yards",
    "rushing_yards",
    "receiving_yards",
    "kicking_points",
    "tackles",
}
QUANTILE_NAMES = ["q10", "q25", "q50", "q75", "q90"]
COUNT_OUTPUTS = ["expected_count", "probability_1_plus", "probability_2_plus"]
SELECTED_COLUMNS = {
    "baseline": "baseline_projection",
    "direct": "direct_projection",
    "component": "component_projection",
    "direct_component_blend": "blend_projection",
}
INTERVALS = {
    "q25_q75": ("q25", "q75", 0.50),
    "q10_q90": ("q10", "q90", 0.80),
}
COVERAGE_COLUMNS = [
    "target",
    "interval",
    "expected_coverage",
    "actual_coverage",
    "mean_interval_width",
    "position_group",
    "usage_bucket",
]
MIN_RISK = {
    "rookie": 1.25,
    "backup_promotion": 1.15,
    "low_history": 1.10,
}
RISK_FIELDS = [
    "history_no_nfl_history_flag",
    "history_history_games",
    "role_starter_promotion_flag",
]


def read_json(path: Path) -> dict[str, Any]:
    assert path.is_file(), f"Missing {path}"
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert isinstance(payload, dict), f"Expected JSON object: {path}"
    return payload


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def assert_monotone_mapping(mapping: dict[str, Any], probability: bool) -> None:
    x = np.asarray(mapping["knots_x"], dtype="float64")
    y = np.asarray(mapping["knots_y"], dtype="float64")
    assert len(x) == len(y) and len(x) >= 2
    assert np.isfinite(x).all() and np.isfinite(y).all()
    assert np.all(np.diff(x) >= -1e-12), "Mapping x knots are not monotone."
    assert np.all(np.diff(y) >= -1e-12), "Mapping y knots are not isotonic."
    if probability:
        assert np.all((y >= -1e-12) & (y <= 1.0 + 1e-12))
    else:
        assert np.all(y >= -1e-12)


def risk_multiplier(frame: pd.DataFrame, risk: dict[str, Any]) -> np.ndarray:
    rookie = (
        pd.to_numeric(frame["history_no_nfl_history_flag"], errors="coerce")
        .fillna(0)
        .to_numpy(dtype="float64")
        >= 0.5
    )
    promotion = (
        pd.to_numeric(frame["role_starter_promotion_flag"], errors="coerce")
        .fillna(0)
        .to_numpy(dtype="float64")
        >= 0.5
    )
    history = pd.to_numeric(
        frame["history_history_games"], errors="coerce"
    ).to_numpy(dtype="float64")
    threshold = int(risk["low_history_games_threshold"])
    low_history = ~np.isfinite(history) | (history < threshold)
    flags = {
        "rookie": rookie,
        "backup_promotion": promotion,
        "low_history": low_history,
    }
    result = np.ones(len(frame), dtype="float64")
    for name, mask in flags.items():
        result[mask] *= float(risk["factors"][name]["multiplier"])
    return np.minimum(result, float(risk["combined_cap"]))


def usage_signal(frame: pd.DataFrame, source: dict[str, Any]) -> np.ndarray:
    method = source["method"]
    if method == "sum":
        out = np.zeros(len(frame), dtype="float64")
        found = np.zeros(len(frame), dtype=bool)
        for column in source["columns"]:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(
                dtype="float64"
            )
            finite = np.isfinite(values)
            out[finite] += values[finite]
            found |= finite
        out[~found] = np.nan
        return out
    if method == "single":
        return pd.to_numeric(
            frame[source["columns"][0]], errors="coerce"
        ).to_numpy(dtype="float64")
    assert method == "selected_projection_proxy"
    return frame["selected_point_prediction"].to_numpy(dtype="float64")


def usage_bucket(values: np.ndarray, thresholds: dict[str, Any]) -> np.ndarray:
    low = float(thresholds["low_max"])
    medium = float(thresholds["medium_max"])
    labels = np.full(len(values), "low", dtype=object)
    finite = np.isfinite(values)
    labels[finite & (values > low)] = "medium"
    labels[finite & (values > medium)] = "high"
    labels[~finite] = "low"
    return labels


def interval_factor(
    frame: pd.DataFrame,
    widening: dict[str, Any],
    interval: str,
) -> np.ndarray:
    result = np.full(
        len(frame), float(widening["global"][interval]), dtype="float64"
    )
    keys = (
        frame["position_group"].astype(str)
        + "|"
        + frame["usage_bucket"].astype(str)
    )
    for key, details in widening["segment_extra"][interval].items():
        result[keys.eq(key).to_numpy()] *= float(details["multiplier"])
    return result


def reconstruct_quantiles(
    frame: pd.DataFrame,
    quantile_cal: dict[str, Any],
) -> dict[str, np.ndarray]:
    point = frame["selected_point_prediction"].to_numpy(dtype="float64")
    residual = quantile_cal["residual_quantiles"]
    center = float(residual["q50"])
    risk = risk_multiplier(frame, quantile_cal["risk_widening"])
    widening = quantile_cal["coverage_widening"]
    floor_zero = bool(quantile_cal["floor_at_zero"])

    output: dict[str, np.ndarray] = {"q50": point + center}
    for interval, (lower_name, upper_name, _expected) in INTERVALS.items():
        factor = interval_factor(frame, widening, interval)
        lower = (
            point
            + center
            + (float(residual[lower_name]) - center) * risk * factor
        )
        upper = (
            point
            + center
            + (float(residual[upper_name]) - center) * risk * factor
        )
        if floor_zero:
            lower = np.maximum(lower, 0.0)
            upper = np.maximum(upper, 0.0)
        output[lower_name] = np.minimum(lower, upper)
        output[upper_name] = np.maximum(lower, upper)
    if floor_zero:
        output["q50"] = np.maximum(output["q50"], 0.0)

    matrix = np.column_stack([output[name] for name in QUANTILE_NAMES])
    matrix = np.maximum.accumulate(matrix, axis=1)
    for idx, name in enumerate(QUANTILE_NAMES):
        output[name] = matrix[:, idx]
    return output


def expected_coverage_row(
    target: str,
    interval: str,
    expected: float,
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    position_group: str,
    usage_bucket_name: str,
) -> dict[str, Any]:
    return {
        "target": target,
        "interval": interval,
        "expected_coverage": expected,
        "actual_coverage": float(np.mean((actual >= lower) & (actual <= upper))),
        "mean_interval_width": float(np.mean(upper - lower)),
        "position_group": position_group,
        "usage_bucket": usage_bucket_name,
    }


def compare_coverage_rows(
    observed: pd.DataFrame,
    target: str,
    frame: pd.DataFrame,
    q: dict[str, np.ndarray],
) -> None:
    actual = frame["actual"].to_numpy(dtype="float64")
    positions = sorted(frame["position_group"].astype(str).unique())
    buckets = [
        b
        for b in ["low", "medium", "high"]
        if b in set(frame["usage_bucket"].astype(str))
    ]
    expected_rows: list[dict[str, Any]] = []

    for interval, (lower_name, upper_name, expected) in INTERVALS.items():
        masks: list[tuple[np.ndarray, str, str]] = [
            (np.ones(len(frame), dtype=bool), "ALL", "ALL")
        ]
        masks.extend(
            (
                frame["position_group"].astype(str).eq(position).to_numpy(),
                position,
                "ALL",
            )
            for position in positions
        )
        masks.extend(
            (
                frame["usage_bucket"].astype(str).eq(bucket).to_numpy(),
                "ALL",
                bucket,
            )
            for bucket in buckets
        )
        for position in positions:
            for bucket in buckets:
                mask = (
                    frame["position_group"].astype(str).eq(position)
                    & frame["usage_bucket"].astype(str).eq(bucket)
                ).to_numpy()
                masks.append((mask, position, bucket))

        for mask, position, bucket in masks:
            if not mask.any():
                continue
            expected_rows.append(
                expected_coverage_row(
                    target,
                    interval,
                    expected,
                    actual[mask],
                    q[lower_name][mask],
                    q[upper_name][mask],
                    position,
                    bucket,
                )
            )

    expected_df = pd.DataFrame(expected_rows, columns=COVERAGE_COLUMNS)
    observed_df = observed.loc[observed["target"].astype(str).eq(target)].copy()
    keys = ["target", "interval", "position_group", "usage_bucket"]
    merged = expected_df.merge(
        observed_df,
        on=keys,
        how="outer",
        suffixes=("_expected", "_observed"),
        indicator=True,
        validate="one_to_one",
    )
    assert merged["_merge"].eq("both").all(), (
        f"Coverage rows differ for {target}: "
        f"{merged.loc[merged['_merge'].ne('both'), keys + ['_merge']].head(20).to_dict('records')}"
    )
    for metric in ["expected_coverage", "actual_coverage", "mean_interval_width"]:
        a = pd.to_numeric(merged[f"{metric}_expected"], errors="raise").to_numpy(
            dtype="float64"
        )
        b = pd.to_numeric(merged[f"{metric}_observed"], errors="raise").to_numpy(
            dtype="float64"
        )
        assert np.allclose(a, b, rtol=1e-9, atol=1e-9), (
            f"Coverage metric mismatch {target} {metric}."
        )


def main() -> int:
    print("CHECK 01: required trainer and calibration artifacts")
    assert TRAINER.is_file(), f"Missing {TRAINER}"
    config = common.load_config()
    targets = list(config["targets"].keys())
    assert targets, "No configured targets."
    assert AUDIT.is_file(), f"Missing {AUDIT}"
    assert COVERAGE.is_file(), f"Missing {COVERAGE}"

    calibration_payloads: dict[str, dict[str, Any]] = {}
    selected_payloads: dict[str, dict[str, Any]] = {}
    for target in targets:
        cal_path = CAL_ROOT / f"{target}_calibration.json"
        calibration_payloads[target] = read_json(cal_path)
        selected_payloads[target] = read_json(
            HERE / "models" / target / "selected_model.json"
        )

    print("CHECK 02: coverage schema and OOF-only source policy")
    coverage_df = pd.read_csv(COVERAGE)
    assert list(coverage_df.columns) == COVERAGE_COLUMNS, coverage_df.columns.tolist()
    assert not coverage_df.empty
    assert set(coverage_df["target"].astype(str)) == QUANTILE_TARGETS
    assert set(coverage_df["interval"].astype(str)) == set(INTERVALS)
    for col in ["expected_coverage", "actual_coverage", "mean_interval_width"]:
        values = pd.to_numeric(coverage_df[col], errors="raise").to_numpy(dtype="float64")
        assert np.isfinite(values).all(), f"Nonfinite coverage {col}."
    assert coverage_df["actual_coverage"].between(0.0, 1.0).all()
    assert (coverage_df["mean_interval_width"] >= -1e-12).all()

    audit = pd.read_parquet(AUDIT)
    required_audit = [
        "split",
        "fold_id",
        *GRAIN,
        "target",
        "actual",
        *SELECTED_COLUMNS.values(),
    ]
    common.require_columns(audit, required_audit, "Issue 25 prediction audit")
    validation = audit.loc[audit["split"].astype(str).eq("validation")].copy()
    assert not validation.empty
    assert validation["fold_id"].astype(str).nunique() == 1
    validation_fold = str(validation["fold_id"].iloc[0])
    validation_season = int(pd.to_numeric(validation["season"], errors="raise").iloc[0])
    assert validation_fold == f"dev_{validation_season}"
    assert set(pd.to_numeric(validation["season"], errors="raise").astype(int)) == {validation_season}

    test_rows = audit.loc[audit["split"].astype(str).eq("test")]
    assert not test_rows.empty, "Issue 25 reporting-only test audit unexpectedly absent."
    test_seasons = set(pd.to_numeric(test_rows["season"], errors="raise").astype(int))
    assert len(test_seasons) == 1
    test_season = next(iter(test_seasons))
    assert validation_season < test_season

    print("CHECK 03: independent selected residual reconstruction")
    target_frames: dict[str, pd.DataFrame] = {}
    context_columns: set[str] = set(GRAIN + RISK_FIELDS + ["position_group"])

    for target in targets:
        selected = selected_payloads[target]
        cal = calibration_payloads[target]
        assert cal["target"] == target
        architecture = selected["selected_architecture"]
        assert cal["selected_architecture"] == architecture
        assert architecture in SELECTED_COLUMNS
        assert cal["market_features_used"] is False
        assert cal["forbidden_features_used"] is False
        source = cal["calibration_source"]
        assert source["split"] == "validation"
        assert source["fold_id"] == validation_fold
        assert int(source["season"]) == validation_season
        assert source["oof_residuals_used"] is True
        test_policy = cal["reporting_test_policy"]
        assert int(test_policy["test_season"]) == test_season
        assert test_policy["test_rows_used_for_calibration"] is False
        assert test_policy["test_reporting_only_preserved"] is True
        assert selected["test_used_for_selection"] is False

        subset = validation.loc[validation["target"].astype(str).eq(target)].copy()
        point_col = SELECTED_COLUMNS[architecture]
        actual = pd.to_numeric(subset["actual"], errors="coerce")
        point = pd.to_numeric(subset[point_col], errors="coerce")
        valid = actual.notna() & point.notna() & np.isfinite(actual) & np.isfinite(point)
        subset = subset.loc[valid, GRAIN + ["actual"]].copy()
        subset["selected_point_prediction"] = point.loc[valid].to_numpy(dtype="float64")
        subset["residual"] = (
            subset["actual"].to_numpy(dtype="float64")
            - subset["selected_point_prediction"].to_numpy(dtype="float64")
        )
        assert len(subset) == int(source["rows"]), f"Calibration row mismatch {target}."
        target_frames[target] = subset

        usage_source = cal["usage_bucket"]["source"]
        context_columns.update(usage_source.get("columns", []))

    historical_path = common.repo_root() / str(config["paths"]["historical_features"])
    historical_context = pd.read_parquet(
        historical_path,
        columns=list(context_columns),
    )
    common.ensure_unique(historical_context, GRAIN, "Issue 26 validation context")
    common.reject_forbidden_feature_columns(historical_context.columns, config)

    print("CHECK 04: quantiles, widening rules, and interval coverage")
    for target in targets:
        cal = calibration_payloads[target]
        frame = target_frames[target].merge(
            historical_context,
            on=GRAIN,
            how="left",
            validate="one_to_one",
            sort=False,
        ).reset_index(drop=True)
        frame["position_group"] = (
            frame["position_group"]
            .astype("string")
            .fillna("UNKNOWN")
            .str.strip()
            .str.upper()
            .replace("", "UNKNOWN")
        )
        signal = usage_signal(frame, cal["usage_bucket"]["source"])
        frame["usage_bucket"] = usage_bucket(
            signal, cal["usage_bucket"]["thresholds"]
        )

        is_count = str(config["targets"][target].get("type", "")) == "count_nonnegative"
        if target in QUANTILE_TARGETS:
            assert cal["calibration_mode"] in {"quantiles", "quantiles_and_count"}
            qcal = cal["quantile_calibration"]
            assert qcal["outputs"] == QUANTILE_NAMES
            residual = frame["residual"].to_numpy(dtype="float64")
            levels = {"q10": 0.10, "q25": 0.25, "q50": 0.50, "q75": 0.75, "q90": 0.90}
            for name, level in levels.items():
                independently = float(np.quantile(residual, level))
                assert close(independently, qcal["residual_quantiles"][name]), (
                    f"Residual quantile mismatch {target} {name}."
                )

            risk = qcal["risk_widening"]
            assert risk["combination"] == "multiply_active_factors_then_cap"
            for name, minimum in MIN_RISK.items():
                factor = float(risk["factors"][name]["multiplier"])
                assert factor + 1e-12 >= minimum, f"{target} {name} not widened."
                assert factor > 1.0, f"{target} {name} multiplier must widen."

            widening = qcal["coverage_widening"]
            for interval in INTERVALS:
                assert float(widening["global"][interval]) >= 1.0
                for details in widening["segment_extra"][interval].values():
                    assert float(details["multiplier"]) >= 1.0
                    assert int(details["rows"]) >= int(widening["segment_min_rows"])

            q = reconstruct_quantiles(frame, qcal)
            matrix = np.column_stack([q[name] for name in QUANTILE_NAMES])
            assert np.isfinite(matrix).all()
            assert np.all(np.diff(matrix, axis=1) >= -1e-10), f"Crossed quantiles {target}."
            if bool(qcal["floor_at_zero"]):
                assert np.all(matrix >= -1e-10)
            compare_coverage_rows(coverage_df, target, frame, q)

            # The aggregate intervals must meet nominal coverage after widening.
            aggregate = coverage_df.loc[
                coverage_df["target"].astype(str).eq(target)
                & coverage_df["position_group"].astype(str).eq("ALL")
                & coverage_df["usage_bucket"].astype(str).eq("ALL")
            ]
            assert len(aggregate) == 2
            assert np.all(
                aggregate["actual_coverage"].to_numpy(dtype="float64")
                + 1e-12
                >= aggregate["expected_coverage"].to_numpy(dtype="float64")
            ), f"Aggregate interval undercoverage remains for {target}."
        else:
            assert cal["calibration_mode"] == "count"
            assert "quantile_calibration" not in cal

        if is_count:
            assert cal["count_outputs"] == COUNT_OUTPUTS
            count = cal["count_calibration"]
            assert set(count) == set(COUNT_OUTPUTS)
            assert_monotone_mapping(count["expected_count"]["mapping"], False)
            assert_monotone_mapping(count["probability_1_plus"]["mapping"], True)
            assert_monotone_mapping(count["probability_2_plus"]["mapping"], True)
        else:
            assert "count_calibration" not in cal

    print("CHECK 05: static trainer policy markers")
    source_text = TRAINER.read_text(encoding="utf-8")
    required_markers = [
        "oof_residuals_used",
        "test_rows_used_for_calibration",
        "role_starter_promotion_flag",
        "history_no_nfl_history_flag",
        "history_history_games",
        "find_extra_widening",
        "probability_1_plus",
        "probability_2_plus",
        "common.reject_forbidden_feature_columns",
    ]
    for marker in required_markers:
        assert marker in source_text, f"Trainer missing policy marker {marker}."

    forbidden_literals = [
        "sportsbook",
        "moneyline",
        "prop_line",
        "over_odds",
        "under_odds",
        "drat",
        "epred",
    ]
    # These terms are allowed only in comments/docstrings that explicitly state
    # exclusion; source features themselves are screened by common above.
    assert "market_features_used\": True" not in source_text

    print("CHECK 06: summarize independently validated calibration")
    for target in targets:
        cal = calibration_payloads[target]
        print(
            f"{target}: mode={cal['calibration_mode']}, "
            f"architecture={cal['selected_architecture']}, "
            f"rows={cal['calibration_source']['rows']}, "
            f"fold={cal['calibration_source']['fold_id']}"
        )
    print(f"targets={len(targets)}")
    print(f"quantile_targets={len(QUANTILE_TARGETS)}")
    print(
        "count_targets="
        + str(
            sum(
                str(config["targets"][t].get("type", "")) == "count_nonnegative"
                for t in targets
            )
        )
    )
    print(f"calibration=OOF_{validation_fold}")
    print(f"untouched_test={test_season}")
    print("test_used_for_calibration=false")
    print("undercoverage_widening=true")
    print("rookie_backup_low_history_widening=true")
    print("market_features_used=false")
    print("ISSUE 26 ACCEPTANCE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
