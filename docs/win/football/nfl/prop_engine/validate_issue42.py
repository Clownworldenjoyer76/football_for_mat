#!/usr/bin/env python3
"""Independent acceptance validator for NFL Prop Engine Issue 42."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config" / "acceptance_thresholds.yaml"
MODEL_SELECTION = HERE / "evaluation" / "model_selection.csv"
SOURCE_QUALITY = HERE / "evaluation" / "source_quality.csv"
INTERVAL_COVERAGE = HERE / "evaluation" / "interval_coverage.csv"
MODEL_ROOT = HERE / "models"

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

TOP_LEVEL_HEADERS = [
    *TARGETS,
    "source_quality",
    "interval_coverage",
]

REQUIRED_TARGET_FIELDS = [
    "minimum_training_rows",
    "maximum_allowed_bias",
    "maximum_validation_mae",
    "minimum_improvement_vs_baseline_pct",
]

# The engine's canonical target config calls kicking_points a derived_count.
COUNT_TARGETS = {
    "passing_tds",
    "rushing_tds",
    "receiving_tds",
    "kicking_points",
    "tackles",
    "sacks",
}

PURE_COUNT_TARGETS_WITH_RECORDED_SELECTION_METRICS = {
    "passing_tds",
    "rushing_tds",
    "receiving_tds",
    "tackles",
    "sacks",
}

FORBIDDEN_TOKENS = (
    "sportsbook",
    "moneyline",
    "spread",
    "prop_line",
    "drat",
    "epred",
)

TOL = 5e-7


def fail(message: str) -> None:
    raise AssertionError(message)


def numeric(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{label} must be numeric; got {value!r}.")
    result = float(value)
    if not math.isfinite(result):
        fail(f"{label} must be finite; got {value!r}.")
    return result


def close(actual: Any, expected: float, label: str, tol: float = TOL) -> None:
    value = numeric(actual, label)
    if not math.isclose(value, float(expected), rel_tol=0.0, abs_tol=tol):
        fail(f"{label} mismatch: expected {expected!r}, got {value!r}.")


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        fail(f"Missing required model metadata: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        fail(f"Expected JSON object: {path}")
    return data


def selected_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "target",
        "candidate",
        "validation_mae",
        "validation_poisson_deviance",
        "validation_brier_1plus",
        "selected_flag",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        fail(f"model_selection.csv missing columns: {missing}")

    selected = frame.loc[
        pd.to_numeric(frame["selected_flag"], errors="coerce").fillna(0).astype(int).eq(1)
    ].copy()
    baseline = frame.loc[frame["candidate"].astype(str).eq("baseline")].copy()

    if set(selected["target"].astype(str)) != set(TARGETS):
        fail("Selected model rows do not cover exactly the nine required targets.")
    if set(baseline["target"].astype(str)) != set(TARGETS):
        fail("Baseline rows do not cover exactly the nine required targets.")
    if selected["target"].astype(str).duplicated().any():
        fail("More than one selected architecture row exists for a target.")
    if baseline["target"].astype(str).duplicated().any():
        fail("More than one baseline row exists for a target.")
    return selected, baseline


def main() -> int:
    if not CONFIG.is_file():
        fail(f"Missing required Issue 42 config: {CONFIG}")

    raw = CONFIG.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        fail("acceptance_thresholds.yaml must be a YAML mapping.")

    if list(data.keys()) != TOP_LEVEL_HEADERS:
        fail(
            "Top-level headers/order mismatch.\n"
            f"Expected: {TOP_LEVEL_HEADERS}\n"
            f"Actual:   {list(data.keys())}"
        )

    # Structural and numeric contract.
    for target in TARGETS:
        section = data[target]
        if not isinstance(section, dict):
            fail(f"{target} must be a YAML mapping.")

        for field in REQUIRED_TARGET_FIELDS:
            if field not in section:
                fail(f"{target} missing required field: {field}")
            value = numeric(section[field], f"{target}.{field}")
            if value < 0:
                fail(f"{target}.{field} must be nonnegative.")

        rows = section["minimum_training_rows"]
        if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
            fail(f"{target}.minimum_training_rows must be a positive integer.")

        if target in COUNT_TARGETS:
            for field in ("maximum_brier_1plus", "maximum_poisson_deviance"):
                if field not in section:
                    fail(f"Count target {target} missing required field: {field}")
                value = numeric(section[field], f"{target}.{field}")
                if value <= 0:
                    fail(f"{target}.{field} must be positive.")

        if section.get("production_approval") is not False:
            fail(
                f"{target}.production_approval must remain false until all "
                "configured acceptance gates are independently passed."
            )

    for name in ("source_quality", "interval_coverage"):
        if not isinstance(data[name], dict):
            fail(f"{name} must be a YAML mapping.")
        if data[name].get("production_approval") is not False:
            fail(f"{name}.production_approval must be false in Issue 42.")

    if not MODEL_SELECTION.is_file():
        fail(f"Missing chronological backtest artifact: {MODEL_SELECTION}")
    selection = pd.read_csv(MODEL_SELECTION)
    selected, baseline = selected_rows(selection)
    selected = selected.set_index(selected["target"].astype(str), drop=False)
    baseline = baseline.set_index(baseline["target"].astype(str), drop=False)

    # Confirm chronology/provenance from the frozen selected-model records.
    for target in TARGETS:
        selected_model = read_json(MODEL_ROOT / target / "selected_model.json")
        if int(selected_model.get("validation_season", -1)) != 2024:
            fail(f"{target} validation_season is not the first chronological 2024 validation.")
        if int(selected_model.get("test_season", -1)) != 2025:
            fail(f"{target} test_season is not 2025.")
        if selected_model.get("test_used_for_selection") is not False:
            fail(f"{target} improperly used 2025 test data for model selection.")
        if selected_model.get("selection_frozen_before_test_reporting") is not True:
            fail(f"{target} selection was not frozen before 2025 reporting.")

    # Reconstruct the configured numeric gates from the first chronological backtest.
    backtest_metric_passes = 0
    for target in TARGETS:
        section = data[target]
        sel = selected.loc[target]
        base = baseline.loc[target]

        sel_mae = float(sel["validation_mae"])
        base_mae = float(base["validation_mae"])
        if not (math.isfinite(sel_mae) and math.isfinite(base_mae) and base_mae > 0):
            fail(f"Invalid validation MAE evidence for {target}.")

        metadata = read_json(MODEL_ROOT / target / "metadata.json")
        model_selection_train = int(metadata["rows"]["model_selection_train"])
        if int(section["minimum_training_rows"]) != model_selection_train:
            fail(
                f"{target}.minimum_training_rows must equal first-backtest "
                f"model-selection rows {model_selection_train}."
            )

        expected_bias_gate = round(sel_mae * 0.20, 6)
        expected_mae_gate = round(sel_mae * 1.05, 6)
        observed_improvement = (base_mae - sel_mae) / base_mae * 100.0
        expected_improvement_gate = round(observed_improvement * 0.90, 6)

        close(
            section["maximum_allowed_bias"],
            expected_bias_gate,
            f"{target}.maximum_allowed_bias",
        )
        close(
            section["maximum_validation_mae"],
            expected_mae_gate,
            f"{target}.maximum_validation_mae",
        )
        close(
            section["minimum_improvement_vs_baseline_pct"],
            expected_improvement_gate,
            f"{target}.minimum_improvement_vs_baseline_pct",
        )

        # The selected model must clear the MAE and baseline-improvement gates
        # used to construct this initial threshold set.
        if sel_mae > float(section["maximum_validation_mae"]) + 1e-12:
            fail(f"{target} does not pass configured validation MAE threshold.")
        if observed_improvement + 1e-12 < float(section["minimum_improvement_vs_baseline_pct"]):
            fail(f"{target} does not pass configured baseline-improvement threshold.")
        backtest_metric_passes += 2

        if target in PURE_COUNT_TARGETS_WITH_RECORDED_SELECTION_METRICS:
            brier = float(sel["validation_brier_1plus"])
            poisson = float(sel["validation_poisson_deviance"])
            if not (math.isfinite(brier) and math.isfinite(poisson)):
                fail(f"Missing recorded count metrics for {target}.")
            expected_brier_gate = round(brier * 1.05, 6)
            expected_poisson_gate = round(poisson * 1.05, 6)
            close(
                section["maximum_brier_1plus"],
                expected_brier_gate,
                f"{target}.maximum_brier_1plus",
            )
            close(
                section["maximum_poisson_deviance"],
                expected_poisson_gate,
                f"{target}.maximum_poisson_deviance",
            )
            if brier > float(section["maximum_brier_1plus"]) + 1e-12:
                fail(f"{target} does not pass configured Brier threshold.")
            if poisson > float(section["maximum_poisson_deviance"]) + 1e-12:
                fail(f"{target} does not pass configured Poisson threshold.")
            backtest_metric_passes += 2

    # Kicking points is derived_count, but the first selection report recorded it
    # as regression and did not populate Brier/Poisson metrics. Numeric gates must
    # nevertheless exist, and approval must remain false until those metrics are
    # independently measured and passed.
    close(
        data["kicking_points"]["maximum_brier_1plus"],
        0.20,
        "kicking_points.maximum_brier_1plus",
    )
    close(
        data["kicking_points"]["maximum_poisson_deviance"],
        5.0,
        "kicking_points.maximum_poisson_deviance",
    )

    # Source-quality gates are anchored to the accepted 12-source monitor.
    sq = data["source_quality"]
    required_sq = {
        "minimum_sources_checked",
        "minimum_pass_rate",
        "maximum_warn_rows",
        "maximum_fail_rows",
        "production_approval",
    }
    missing_sq = sorted(required_sq - set(sq))
    if missing_sq:
        fail(f"source_quality missing keys: {missing_sq}")
    if int(sq["minimum_sources_checked"]) != 12:
        fail("source_quality.minimum_sources_checked must be 12.")
    close(sq["minimum_pass_rate"], 1.0, "source_quality.minimum_pass_rate")
    if int(sq["maximum_warn_rows"]) != 0 or int(sq["maximum_fail_rows"]) != 0:
        fail("source_quality warning/failure ceilings must both be zero.")

    if not SOURCE_QUALITY.is_file():
        fail(f"Missing accepted Issue 40 source-quality artifact: {SOURCE_QUALITY}")
    source = pd.read_csv(SOURCE_QUALITY)
    if len(source) < int(sq["minimum_sources_checked"]):
        fail("Current source-quality artifact checks too few sources.")
    quality = source["quality_status"].astype(str).str.strip().str.upper()
    pass_rate = float(quality.eq("PASS").mean())
    warn_rows = int(quality.eq("WARN").sum())
    fail_rows = int(quality.eq("FAIL").sum())
    if pass_rate + 1e-12 < float(sq["minimum_pass_rate"]):
        fail("Current source-quality pass rate is below threshold.")
    if warn_rows > int(sq["maximum_warn_rows"]):
        fail("Current source-quality WARN rows exceed threshold.")
    if fail_rows > int(sq["maximum_fail_rows"]):
        fail("Current source-quality FAIL rows exceed threshold.")

    # Interval gates use the calibrated overall rows only, not tiny position/usage cells.
    ic = data["interval_coverage"]
    required_ic = {
        "minimum_targets_checked",
        "q10_q90_expected_coverage",
        "q10_q90_minimum_coverage",
        "q25_q75_expected_coverage",
        "q25_q75_minimum_coverage",
        "production_approval",
    }
    missing_ic = sorted(required_ic - set(ic))
    if missing_ic:
        fail(f"interval_coverage missing keys: {missing_ic}")
    if int(ic["minimum_targets_checked"]) != 5:
        fail("interval_coverage.minimum_targets_checked must be 5.")
    close(ic["q10_q90_expected_coverage"], 0.80, "interval_coverage.q10_q90_expected_coverage")
    close(ic["q10_q90_minimum_coverage"], 0.78, "interval_coverage.q10_q90_minimum_coverage")
    close(ic["q25_q75_expected_coverage"], 0.50, "interval_coverage.q25_q75_expected_coverage")
    close(ic["q25_q75_minimum_coverage"], 0.48, "interval_coverage.q25_q75_minimum_coverage")

    if not INTERVAL_COVERAGE.is_file():
        fail(f"Missing accepted interval coverage artifact: {INTERVAL_COVERAGE}")
    coverage = pd.read_csv(INTERVAL_COVERAGE)
    required_cov = {
        "target",
        "interval",
        "actual_coverage",
        "position_group",
        "usage_bucket",
    }
    missing_cov = sorted(required_cov - set(coverage.columns))
    if missing_cov:
        fail(f"interval_coverage.csv missing columns: {missing_cov}")

    overall = coverage.loc[
        coverage["position_group"].astype(str).eq("ALL")
        & coverage["usage_bucket"].astype(str).eq("ALL")
    ].copy()
    targets_checked = overall["target"].astype(str).nunique()
    if targets_checked < int(ic["minimum_targets_checked"]):
        fail("Interval coverage has fewer overall targets than required.")

    for interval, threshold_key in (
        ("q10_q90", "q10_q90_minimum_coverage"),
        ("q25_q75", "q25_q75_minimum_coverage"),
    ):
        rows = overall.loc[overall["interval"].astype(str).eq(interval)]
        if rows.empty:
            fail(f"No overall interval coverage rows for {interval}.")
        values = pd.to_numeric(rows["actual_coverage"], errors="raise")
        if (values < float(ic[threshold_key]) - 1e-12).any():
            bad = rows.loc[values < float(ic[threshold_key]) - 1e-12, ["target", "actual_coverage"]]
            fail(f"{interval} overall coverage below threshold: {bad.to_dict('records')}")

    lower = raw.casefold()
    forbidden = [token for token in FORBIDDEN_TOKENS if token in lower]
    if forbidden:
        fail("Forbidden betting-derived tokens found in threshold config: " + ", ".join(forbidden))

    approvals = [
        data[name].get("production_approval")
        for name in TOP_LEVEL_HEADERS
    ]
    if any(value is not False for value in approvals):
        fail("All Issue 42 production approvals must remain false.")

    print(f"config={CONFIG.relative_to(HERE).as_posix()}")
    print(f"headers={len(TOP_LEVEL_HEADERS)}")
    print(f"targets={len(TARGETS)}")
    print(f"count_targets={len(COUNT_TARGETS)}")
    print(f"chronological_validation_season=2024")
    print(f"reporting_only_test_season=2025")
    print(f"backtest_metric_gates_passed={backtest_metric_passes}")
    print(f"source_quality_pass_rate={pass_rate:.6f}")
    print(f"source_quality_warn_rows={warn_rows}")
    print(f"source_quality_fail_rows={fail_rows}")
    print(f"interval_targets_checked={targets_checked}")
    print("thresholds_populated_after_chronological_backtest=true")
    print("production_approval=false")
    print("market_features_used=false")
    print("MODEL ACCEPTANCE THRESHOLDS VALIDATION: PASS")
    print("ISSUE 42 ACCEPTANCE: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"MODEL ACCEPTANCE THRESHOLDS VALIDATION: FAIL - {exc}", file=sys.stderr)
        raise SystemExit(1)
