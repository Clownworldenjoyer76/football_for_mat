#!/usr/bin/env python3
"""
Calibrate uncertainty for the selected NFL Prop Engine architectures.

Issue 26 contract
-----------------
READS
    docs/win/football/nfl/prop_engine/config/prop_engine.yaml
    docs/win/football/nfl/prop_engine/evaluation/model_selection_predictions.parquet
    docs/win/football/nfl/prop_engine/models/{target}/selected_model.json
    configured historical feature table

WRITES
    docs/win/football/nfl/prop_engine/models/calibration/{target}_calibration.json
    docs/win/football/nfl/prop_engine/evaluation/interval_coverage.csv

POLICY
    - Calibration uses only OOF validation residuals from the selected architecture.
    - The Issue 25 2025 reporting split is never used for calibration.
    - Quantile targets emit q10/q25/q50/q75/q90 calibration parameters.
    - Count targets emit expected_count / probability_1_plus / probability_2_plus.
    - Under-covering intervals are widened, globally and by position/usage segment.
    - Rookie, starter-promotion, and low-history rows receive explicit uncertainty
      multipliers learned from OOF residual scale with conservative minimum widening.
    - No sportsbook, market, DRAT, EPRED, or other forbidden feature may enter.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import json
import math
import os
import sys
import tempfile

import numpy as np
import yaml
import pandas as pd
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import common


GRAIN = ["season", "week", "game_id", "player_id"]
AUDIT_PATH = Path(
    "docs/win/football/nfl/prop_engine/evaluation/"
    "model_selection_predictions.parquet"
)
COVERAGE_PATH = Path(
    "docs/win/football/nfl/prop_engine/evaluation/interval_coverage.csv"
)
CALIBRATION_ROOT = Path(
    "docs/win/football/nfl/prop_engine/models/calibration"
)
ACCEPTANCE_THRESHOLDS_PATH = Path(
    "docs/win/football/nfl/prop_engine/config/acceptance_thresholds.yaml"
)

# Validation-only strength for the displayed point prediction.
# 0.0 = raw selected point; 1.0 = full existing calibrated point.
POINT_PREDICTION_BLEND_CANDIDATES = (0.0, 0.25, 0.50, 0.75, 1.0)

QUANTILE_TARGETS = [
    "passing_yards",
    "rushing_yards",
    "receiving_yards",
    "kicking_points",
    "tackles",
]

QUANTILE_LEVELS = {
    "q10": 0.10,
    "q25": 0.25,
    "q50": 0.50,
    "q75": 0.75,
    "q90": 0.90,
}

INTERVALS = {
    "q25_q75": ("q25", "q75", 0.50),
    "q10_q90": ("q10", "q90", 0.80),
}

SELECTED_PROJECTION_COLUMNS = {
    "baseline": "baseline_projection",
    "direct": "direct_projection",
    "component": "component_projection",
    "direct_component_blend": "blend_projection",
}

CONTEXT_REQUIRED = [
    "position_group",
    "history_no_nfl_history_flag",
    "history_history_games",
    "role_starter_promotion_flag",
]

# Ordered fallbacks. The first available pregame feature is used. Kicking has
# a special two-column sum when both FG and XP attempt form fields are present.
USAGE_CANDIDATES = {
    "passing_yards": [
        "player_pass_attempts_roll3_mean",
        "player_pass_attempts_roll5_mean",
        "player_pass_attempts_ewm5",
        "player_pass_attempts_career_prior",
    ],
    "passing_tds": [
        "player_pass_attempts_roll3_mean",
        "player_pass_attempts_roll5_mean",
        "player_pass_attempts_ewm5",
        "player_pass_attempts_career_prior",
    ],
    "rushing_yards": [
        "player_carries_roll3_mean",
        "player_carries_roll5_mean",
        "player_carries_ewm5",
        "player_carries_career_prior",
    ],
    "rushing_tds": [
        "player_goal_line_carries_roll3_mean",
        "player_goal_line_carries_roll5_mean",
        "player_carries_roll3_mean",
        "player_carries_career_prior",
    ],
    "receiving_yards": [
        "player_targets_roll3_mean",
        "player_targets_roll5_mean",
        "player_targets_ewm5",
        "player_targets_career_prior",
    ],
    "receiving_tds": [
        "player_red_zone_targets_roll3_mean",
        "player_red_zone_targets_roll5_mean",
        "player_targets_roll3_mean",
        "player_targets_career_prior",
    ],
    "kicking_points": [
        "player_field_goal_attempts_roll3_mean",
        "player_field_goal_attempts_roll5_mean",
        "player_field_goal_attempts_career_prior",
    ],
    "tackles": [
        "player_defense_participation_roll3_mean",
        "role_participation_roll3",
        "player_defense_participation_career_prior",
    ],
    "sacks": [
        "player_defense_participation_roll3_mean",
        "role_participation_roll3",
        "player_defense_participation_career_prior",
    ],
}

KICKING_USAGE_COMPONENTS = [
    "player_field_goal_attempts_roll3_mean",
    "player_extra_point_attempts_roll3_mean",
]

LOW_HISTORY_GAMES = 4
MIN_RISK_MULTIPLIERS = {
    "rookie": 1.25,
    "backup_promotion": 1.15,
    "low_history": 1.10,
}
MAX_RISK_MULTIPLIER = 2.50
MAX_COMBINED_RISK_MULTIPLIER = 3.00
MIN_RISK_SAMPLE = 20
MIN_SEGMENT_SAMPLE = 30
MAX_INTERVAL_FACTOR = 5.00
COUNT_CALIBRATION_BINS = 12

COVERAGE_COLUMNS = [
    "target",
    "interval",
    "expected_coverage",
    "actual_coverage",
    "mean_interval_width",
    "position_group",
    "usage_bucket",
]


@dataclass(frozen=True)
class SelectedContract:
    target: str
    selected_architecture: str
    validation_season: int
    test_season: int
    model_selection_train_end_season: int
    path: Path


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.casefold() in {"", "nan", "none", "null", "<na>", "nat"}:
        return ""
    return text


def finite_array(values: Iterable[Any], label: str) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(
        dtype="float64"
    )
    if len(arr) == 0 or not np.isfinite(arr).all():
        raise ValueError(f"{label}: expected non-empty finite numeric values.")
    return arr


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = path.resolve()
    root = common.prop_root().resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Refusing write outside Prop Engine: {path}") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    )
    temp = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temp, path)
    except Exception:
        temp.unlink(missing_ok=True)
        raise


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def load_selected_contracts(
    targets: list[str],
) -> dict[str, SelectedContract]:
    contracts: dict[str, SelectedContract] = {}
    for target in targets:
        path = common.prop_root() / "models" / target / "selected_model.json"
        payload = read_json(path)
        if payload.get("target") != target:
            raise ValueError(f"{path}: target mismatch.")
        architecture = clean_text(
            payload.get("selected_architecture")
            or payload.get("selected_candidate")
        )
        if architecture not in SELECTED_PROJECTION_COLUMNS:
            raise ValueError(
                f"{path}: unsupported selected architecture {architecture!r}."
            )
        if payload.get("test_used_for_selection") is not False:
            raise ValueError(f"{path}: 2025/test was used for selection.")
        if payload.get("test_reporting_only") is not True:
            raise ValueError(f"{path}: test_reporting_only must be true.")
        if payload.get("market_features_used") is not False:
            raise ValueError(f"{path}: market feature policy violation.")

        contracts[target] = SelectedContract(
            target=target,
            selected_architecture=architecture,
            validation_season=int(payload["validation_season"]),
            test_season=int(payload["test_season"]),
            model_selection_train_end_season=int(
                payload["model_selection_train_end_season"]
            ),
            path=path,
        )

    validation_seasons = {c.validation_season for c in contracts.values()}
    test_seasons = {c.test_season for c in contracts.values()}
    train_ends = {c.model_selection_train_end_season for c in contracts.values()}
    if len(validation_seasons) != 1 or len(test_seasons) != 1 or len(train_ends) != 1:
        raise ValueError("Selected-model split contracts disagree across targets.")
    validation_season = next(iter(validation_seasons))
    test_season = next(iter(test_seasons))
    train_end = next(iter(train_ends))
    if train_end >= validation_season or validation_season >= test_season:
        raise ValueError(
            "Expected chronological train < validation < reporting-test split."
        )
    return contracts


def load_oof_selected_predictions(
    targets: list[str],
    contracts: dict[str, SelectedContract],
) -> pd.DataFrame:
    required = [
        "split",
        "fold_id",
        *GRAIN,
        "target",
        "actual",
        *sorted(set(SELECTED_PROJECTION_COLUMNS.values())),
    ]
    audit = common.read_parquet_required(AUDIT_PATH, required)
    common.ensure_unique(audit, [*GRAIN, "target", "split"], "Issue 25 audit")

    # Calibration is deliberately restricted to the OOF validation split.
    calibration = audit.loc[audit["split"].astype(str).eq("validation")].copy()
    if calibration.empty:
        raise ValueError("Issue 25 audit has no validation rows for calibration.")

    if set(calibration["target"].astype(str).unique()) != set(targets):
        raise ValueError("Issue 25 validation audit target coverage mismatch.")

    expected_validation_season = next(
        iter({c.validation_season for c in contracts.values()})
    )
    seasons = set(pd.to_numeric(calibration["season"], errors="raise").astype(int))
    if seasons != {expected_validation_season}:
        raise ValueError(
            f"Calibration must use validation season {expected_validation_season}; "
            f"found {sorted(seasons)}."
        )

    if calibration["fold_id"].astype(str).nunique() != 1:
        raise ValueError("Calibration validation rows span multiple fold IDs.")
    expected_fold = f"dev_{expected_validation_season}"
    fold_id = str(calibration["fold_id"].iloc[0])
    if fold_id != expected_fold:
        raise ValueError(
            f"Expected OOF calibration fold {expected_fold}, found {fold_id}."
        )

    pieces: list[pd.DataFrame] = []
    for target in targets:
        frame = calibration.loc[calibration["target"].astype(str).eq(target)].copy()
        contract = contracts[target]
        source_column = SELECTED_PROJECTION_COLUMNS[
            contract.selected_architecture
        ]
        actual = pd.to_numeric(frame["actual"], errors="coerce")
        point = pd.to_numeric(frame[source_column], errors="coerce")
        baseline = pd.to_numeric(frame["baseline_projection"], errors="coerce")
        valid = (
            actual.notna()
            & point.notna()
            & baseline.notna()
            & np.isfinite(actual)
            & np.isfinite(point)
            & np.isfinite(baseline)
        )
        frame = frame.loc[
            valid,
            [*GRAIN, "fold_id", "actual", "baseline_projection"],
        ].copy()
        frame["target"] = target
        frame["selected_architecture"] = contract.selected_architecture
        frame["selected_point_prediction"] = point.loc[valid].to_numpy(dtype="float64")
        frame["baseline_projection"] = baseline.loc[valid].to_numpy(dtype="float64")
        if frame.empty:
            raise ValueError(f"No finite OOF calibration rows for {target}.")
        frame["residual"] = (
            frame["actual"].to_numpy(dtype="float64")
            - frame["selected_point_prediction"].to_numpy(dtype="float64")
        )
        pieces.append(frame)

    output = pd.concat(pieces, ignore_index=True)
    common.ensure_unique(output, [*GRAIN, "target"], "selected OOF calibration rows")
    return output


def parquet_columns(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Required historical feature table missing: {path}")
    return set(pq.ParquetFile(path).schema.names)


def context_columns_for_schema(
    schema: set[str],
    targets: list[str],
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    missing = [column for column in [*GRAIN, *CONTEXT_REQUIRED] if column not in schema]
    if missing:
        raise ValueError(
            "Historical feature table missing Issue 26 context fields: "
            + ", ".join(missing)
        )

    chosen: dict[str, dict[str, Any]] = {}
    columns = [*GRAIN, *CONTEXT_REQUIRED]

    for target in targets:
        if target == "kicking_points" and all(
            column in schema for column in KICKING_USAGE_COMPONENTS
        ):
            chosen[target] = {
                "method": "sum",
                "columns": list(KICKING_USAGE_COMPONENTS),
                "label": "+".join(KICKING_USAGE_COMPONENTS),
            }
            columns.extend(KICKING_USAGE_COMPONENTS)
            continue

        available = [
            column
            for column in USAGE_CANDIDATES[target]
            if column in schema
        ]
        if available:
            chosen[target] = {
                "method": "single",
                "columns": [available[0]],
                "label": available[0],
            }
            columns.append(available[0])
        else:
            # This is still pregame because the selected point prediction is
            # computed only from pregame features. The fallback is explicit.
            chosen[target] = {
                "method": "selected_projection_proxy",
                "columns": [],
                "label": "selected_point_prediction",
            }

    return list(dict.fromkeys(columns)), chosen


def load_context(
    config: dict[str, Any],
    targets: list[str],
    oof: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    root = common.repo_root()
    historical_path = root / str(config["paths"]["historical_features"])
    schema = parquet_columns(historical_path)
    columns, usage_sources = context_columns_for_schema(schema, targets)
    common.reject_forbidden_feature_columns(columns, config)
    context = pd.read_parquet(historical_path, columns=columns)
    common.ensure_unique(context, GRAIN, "historical calibration context")

    output = oof.merge(
        context,
        on=GRAIN,
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if output["position_group"].isna().all():
        raise ValueError("Calibration context failed to join position_group.")

    output["position_group"] = (
        output["position_group"]
        .astype("string")
        .fillna("UNKNOWN")
        .str.strip()
        .str.upper()
        .replace("", "UNKNOWN")
    )
    for column in [
        "history_no_nfl_history_flag",
        "history_history_games",
        "role_starter_promotion_flag",
    ]:
        output[column] = pd.to_numeric(output[column], errors="coerce")

    return output, usage_sources


def usage_signal(
    frame: pd.DataFrame,
    source: dict[str, Any],
) -> np.ndarray:
    method = source["method"]
    if method == "sum":
        total = np.zeros(len(frame), dtype="float64")
        any_finite = np.zeros(len(frame), dtype=bool)
        for column in source["columns"]:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(
                dtype="float64"
            )
            finite = np.isfinite(values)
            total[finite] += values[finite]
            any_finite |= finite
        total[~any_finite] = np.nan
        return total
    if method == "single":
        return pd.to_numeric(
            frame[source["columns"][0]], errors="coerce"
        ).to_numpy(dtype="float64")
    return pd.to_numeric(
        frame["selected_point_prediction"], errors="coerce"
    ).to_numpy(dtype="float64")


def fit_usage_buckets(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {"low_max": 0.0, "medium_max": 0.0}
    low, medium = np.quantile(finite, [1.0 / 3.0, 2.0 / 3.0])
    if medium < low:
        medium = low
    return {"low_max": float(low), "medium_max": float(medium)}


def apply_usage_buckets(
    values: np.ndarray,
    thresholds: dict[str, float],
) -> np.ndarray:
    low = float(thresholds["low_max"])
    medium = float(thresholds["medium_max"])
    labels = np.full(len(values), "low", dtype=object)
    finite = np.isfinite(values)
    labels[finite & (values > low)] = "medium"
    labels[finite & (values > medium)] = "high"
    labels[~finite] = "low"
    return labels


def centered_scale(residual: np.ndarray, center: float) -> float:
    if len(residual) == 0:
        return 0.0
    return float(np.quantile(np.abs(residual - center), 0.80))


def learned_risk_multiplier(
    residual: np.ndarray,
    mask: np.ndarray,
    center: float,
    minimum: float,
) -> tuple[float, int]:
    mask = np.asarray(mask, dtype=bool)
    flagged = residual[mask]
    reference = residual[~mask]
    if len(flagged) < MIN_RISK_SAMPLE or len(reference) < MIN_RISK_SAMPLE:
        return float(minimum), int(len(flagged))
    flag_scale = centered_scale(flagged, center)
    ref_scale = centered_scale(reference, center)
    ratio = 1.0 if ref_scale <= 1e-12 else flag_scale / ref_scale
    multiplier = min(
        MAX_RISK_MULTIPLIER,
        max(float(minimum), float(ratio)),
    )
    return float(multiplier), int(len(flagged))


def risk_flags(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    rookie = (
        pd.to_numeric(
            frame["history_no_nfl_history_flag"], errors="coerce"
        )
        .fillna(0)
        .to_numpy(dtype="float64")
        >= 0.5
    )
    promotion = (
        pd.to_numeric(
            frame["role_starter_promotion_flag"], errors="coerce"
        )
        .fillna(0)
        .to_numpy(dtype="float64")
        >= 0.5
    )
    history = pd.to_numeric(
        frame["history_history_games"], errors="coerce"
    ).to_numpy(dtype="float64")
    low_history = ~np.isfinite(history) | (history < LOW_HISTORY_GAMES)
    return {
        "rookie": rookie,
        "backup_promotion": promotion,
        "low_history": low_history,
    }


def fit_risk_widening(
    frame: pd.DataFrame,
    residual: np.ndarray,
    center: float,
) -> dict[str, Any]:
    flags = risk_flags(frame)
    payload: dict[str, Any] = {
        "combination": "multiply_active_factors_then_cap",
        "combined_cap": MAX_COMBINED_RISK_MULTIPLIER,
        "low_history_games_threshold": LOW_HISTORY_GAMES,
        "factors": {},
    }
    for name, mask in flags.items():
        factor, rows = learned_risk_multiplier(
            residual,
            mask,
            center,
            MIN_RISK_MULTIPLIERS[name],
        )
        payload["factors"][name] = {
            "multiplier": factor,
            "rows": rows,
            "minimum_required_multiplier": MIN_RISK_MULTIPLIERS[name],
        }
    return payload


def apply_risk_multiplier(
    frame: pd.DataFrame,
    risk_payload: dict[str, Any],
) -> np.ndarray:
    flags = risk_flags(frame)
    result = np.ones(len(frame), dtype="float64")
    for name, mask in flags.items():
        factor = float(risk_payload["factors"][name]["multiplier"])
        result[mask] *= factor
    return np.minimum(result, float(risk_payload["combined_cap"]))


def interval_bounds(
    point: np.ndarray,
    residual_quantiles: dict[str, float],
    lower_name: str,
    upper_name: str,
    risk: np.ndarray,
    interval_factor: np.ndarray | float,
    floor_at_zero: bool,
) -> tuple[np.ndarray, np.ndarray]:
    center = float(residual_quantiles["q50"])
    lo = float(residual_quantiles[lower_name])
    hi = float(residual_quantiles[upper_name])
    factor = np.asarray(interval_factor, dtype="float64")
    lower = point + center + (lo - center) * risk * factor
    upper = point + center + (hi - center) * risk * factor
    if floor_at_zero:
        lower = np.maximum(lower, 0.0)
        upper = np.maximum(upper, 0.0)
    lower, upper = np.minimum(lower, upper), np.maximum(lower, upper)
    return lower, upper


def coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean((actual >= lower) & (actual <= upper)))


def find_extra_widening(
    actual: np.ndarray,
    point: np.ndarray,
    residual_quantiles: dict[str, float],
    lower_name: str,
    upper_name: str,
    risk: np.ndarray,
    base_factor: np.ndarray | float,
    expected: float,
    floor_at_zero: bool,
) -> float:
    base = np.asarray(base_factor, dtype="float64")

    def cov(extra: float) -> float:
        lower, upper = interval_bounds(
            point,
            residual_quantiles,
            lower_name,
            upper_name,
            risk,
            base * extra,
            floor_at_zero,
        )
        return coverage(actual, lower, upper)

    if cov(1.0) + 1e-12 >= expected:
        return 1.0

    high = 1.25
    while high < MAX_INTERVAL_FACTOR and cov(high) + 1e-12 < expected:
        high *= 1.25
    high = min(high, MAX_INTERVAL_FACTOR)
    if cov(high) + 1e-12 < expected:
        return float(high)

    low = 1.0
    for _ in range(50):
        mid = (low + high) / 2.0
        if cov(mid) + 1e-12 >= expected:
            high = mid
        else:
            low = mid
    return float(high)


def fit_interval_widening(
    frame: pd.DataFrame,
    residual_quantiles: dict[str, float],
    risk: np.ndarray,
    floor_at_zero: bool,
) -> dict[str, Any]:
    actual = frame["actual"].to_numpy(dtype="float64")
    point = frame["selected_point_prediction"].to_numpy(dtype="float64")
    payload: dict[str, Any] = {
        "global": {},
        "segment_extra": {},
        "segment_min_rows": MIN_SEGMENT_SAMPLE,
        "segment_key": "position_group|usage_bucket",
    }

    for interval_name, (lower_name, upper_name, expected) in INTERVALS.items():
        global_factor = find_extra_widening(
            actual,
            point,
            residual_quantiles,
            lower_name,
            upper_name,
            risk,
            1.0,
            expected,
            floor_at_zero,
        )
        payload["global"][interval_name] = float(global_factor)
        payload["segment_extra"][interval_name] = {}

        grouped = frame.groupby(
            ["position_group", "usage_bucket"],
            sort=True,
            dropna=False,
        )
        for (position_group, usage_bucket), group in grouped:
            if len(group) < MIN_SEGMENT_SAMPLE:
                continue
            idx = group.index.to_numpy(dtype=int)
            extra = find_extra_widening(
                actual[idx],
                point[idx],
                residual_quantiles,
                lower_name,
                upper_name,
                risk[idx],
                global_factor,
                expected,
                floor_at_zero,
            )
            key = f"{position_group}|{usage_bucket}"
            payload["segment_extra"][interval_name][key] = {
                "multiplier": float(extra),
                "rows": int(len(group)),
            }
    return payload


def segment_factor_array(
    frame: pd.DataFrame,
    widening: dict[str, Any],
    interval_name: str,
) -> np.ndarray:
    global_factor = float(widening["global"][interval_name])
    result = np.full(len(frame), global_factor, dtype="float64")
    segment = widening["segment_extra"][interval_name]
    keys = (
        frame["position_group"].astype(str)
        + "|"
        + frame["usage_bucket"].astype(str)
    )
    for key, details in segment.items():
        mask = keys.eq(key).to_numpy()
        result[mask] *= float(details["multiplier"])
    return result


def quantile_values(
    frame: pd.DataFrame,
    residual_quantiles: dict[str, float],
    risk: np.ndarray,
    widening: dict[str, Any],
    floor_at_zero: bool,
) -> dict[str, np.ndarray]:
    point = frame["selected_point_prediction"].to_numpy(dtype="float64")
    center = float(residual_quantiles["q50"])
    output: dict[str, np.ndarray] = {
        "q50": point + center,
    }
    for interval_name, (lower_name, upper_name, _expected) in INTERVALS.items():
        factors = segment_factor_array(frame, widening, interval_name)
        lower, upper = interval_bounds(
            point,
            residual_quantiles,
            lower_name,
            upper_name,
            risk,
            factors,
            floor_at_zero,
        )
        output[lower_name] = lower
        output[upper_name] = upper
    if floor_at_zero:
        output["q50"] = np.maximum(output["q50"], 0.0)
    # Defensive monotonicity enforcement after flooring and independently
    # calibrated interval widening.
    matrix = np.column_stack(
        [output[name] for name in ["q10", "q25", "q50", "q75", "q90"]]
    )
    matrix = np.maximum.accumulate(matrix, axis=1)
    for i, name in enumerate(["q10", "q25", "q50", "q75", "q90"]):
        output[name] = matrix[:, i]
    return output


def coverage_rows(
    target: str,
    frame: pd.DataFrame,
    quantiles: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    actual = frame["actual"].to_numpy(dtype="float64")

    def append_group(
        interval_name: str,
        lower_name: str,
        upper_name: str,
        expected: float,
        mask: np.ndarray,
        position_group: str,
        usage_bucket: str,
    ) -> None:
        if not mask.any():
            return
        lower = quantiles[lower_name][mask]
        upper = quantiles[upper_name][mask]
        y = actual[mask]
        rows.append(
            {
                "target": target,
                "interval": interval_name,
                "expected_coverage": float(expected),
                "actual_coverage": coverage(y, lower, upper),
                "mean_interval_width": float(np.mean(upper - lower)),
                "position_group": position_group,
                "usage_bucket": usage_bucket,
            }
        )

    all_mask = np.ones(len(frame), dtype=bool)
    positions = sorted(frame["position_group"].astype(str).unique().tolist())
    buckets = [
        bucket
        for bucket in ["low", "medium", "high"]
        if bucket in set(frame["usage_bucket"].astype(str))
    ]

    for interval_name, (lower_name, upper_name, expected) in INTERVALS.items():
        append_group(
            interval_name,
            lower_name,
            upper_name,
            expected,
            all_mask,
            "ALL",
            "ALL",
        )
        for position_group in positions:
            mask = frame["position_group"].astype(str).eq(position_group).to_numpy()
            append_group(
                interval_name,
                lower_name,
                upper_name,
                expected,
                mask,
                position_group,
                "ALL",
            )
        for bucket in buckets:
            mask = frame["usage_bucket"].astype(str).eq(bucket).to_numpy()
            append_group(
                interval_name,
                lower_name,
                upper_name,
                expected,
                mask,
                "ALL",
                bucket,
            )
        for position_group in positions:
            for bucket in buckets:
                mask = (
                    frame["position_group"].astype(str).eq(position_group)
                    & frame["usage_bucket"].astype(str).eq(bucket)
                ).to_numpy()
                append_group(
                    interval_name,
                    lower_name,
                    upper_name,
                    expected,
                    mask,
                    position_group,
                    bucket,
                )
    return rows


def _pava(y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = [float(v) for v in y]
    w = [float(v) for v in weights]
    starts = list(range(len(values)))
    ends = list(range(len(values)))
    i = 0
    while i < len(values) - 1:
        if values[i] <= values[i + 1] + 1e-15:
            i += 1
            continue
        total_w = w[i] + w[i + 1]
        pooled = (values[i] * w[i] + values[i + 1] * w[i + 1]) / total_w
        values[i] = pooled
        w[i] = total_w
        ends[i] = ends[i + 1]
        del values[i + 1]
        del w[i + 1]
        del starts[i + 1]
        del ends[i + 1]
        if i > 0:
            i -= 1

    result = np.empty(len(y), dtype="float64")
    for value, start, end in zip(values, starts, ends):
        result[start : end + 1] = value
    return result


def fit_monotone_mapping(
    x: np.ndarray,
    y: np.ndarray,
    *,
    probability: bool,
) -> dict[str, Any]:
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) == 0:
        raise ValueError("Cannot fit count calibration with zero valid rows.")

    order = np.argsort(x, kind="mergesort")
    x = x[order]
    y = y[order]
    bins = min(COUNT_CALIBRATION_BINS, len(x))
    chunks = [chunk for chunk in np.array_split(np.arange(len(x)), bins) if len(chunk)]
    bx = np.array([float(np.mean(x[idx])) for idx in chunks], dtype="float64")
    by = np.array([float(np.mean(y[idx])) for idx in chunks], dtype="float64")
    bw = np.array([float(len(idx)) for idx in chunks], dtype="float64")

    # Combine identical x knots before isotonic pooling.
    grouped_x: list[float] = []
    grouped_y: list[float] = []
    grouped_w: list[float] = []
    for xv, yv, wv in zip(bx, by, bw):
        if grouped_x and abs(xv - grouped_x[-1]) <= 1e-15:
            total = grouped_w[-1] + wv
            grouped_y[-1] = (
                grouped_y[-1] * grouped_w[-1] + yv * wv
            ) / total
            grouped_w[-1] = total
        else:
            grouped_x.append(float(xv))
            grouped_y.append(float(yv))
            grouped_w.append(float(wv))

    gx = np.asarray(grouped_x, dtype="float64")
    gy = np.asarray(grouped_y, dtype="float64")
    gw = np.asarray(grouped_w, dtype="float64")
    calibrated = _pava(gy, gw)
    if probability:
        calibrated = np.clip(calibrated, 0.0, 1.0)
    else:
        calibrated = np.maximum(calibrated, 0.0)

    if len(gx) == 1:
        # np.interp accepts one point, but two equal-valued boundary knots make
        # the future mapping contract explicit.
        gx = np.array([gx[0], gx[0] + 1e-12], dtype="float64")
        calibrated = np.array([calibrated[0], calibrated[0]], dtype="float64")

    return {
        "method": "equal_frequency_binning_then_pava_isotonic",
        "bins_requested": COUNT_CALIBRATION_BINS,
        "knots_x": [float(v) for v in gx],
        "knots_y": [float(v) for v in calibrated],
        "left_value": float(calibrated[0]),
        "right_value": float(calibrated[-1]),
        "output_bounds": [0.0, 1.0] if probability else [0.0, None],
    }


def apply_mapping(x: np.ndarray, mapping: dict[str, Any]) -> np.ndarray:
    xp = np.asarray(mapping["knots_x"], dtype="float64")
    fp = np.asarray(mapping["knots_y"], dtype="float64")
    output = np.interp(
        np.asarray(x, dtype="float64"),
        xp,
        fp,
        left=float(mapping["left_value"]),
        right=float(mapping["right_value"]),
    )
    bounds = mapping.get("output_bounds", [None, None])
    if bounds[0] is not None:
        output = np.maximum(output, float(bounds[0]))
    if bounds[1] is not None:
        output = np.minimum(output, float(bounds[1]))
    return output


def fit_count_calibration(frame: pd.DataFrame) -> dict[str, Any]:
    raw = np.maximum(
        frame["selected_point_prediction"].to_numpy(dtype="float64"),
        0.0,
    )
    actual = np.maximum(frame["actual"].to_numpy(dtype="float64"), 0.0)
    expected_mapping = fit_monotone_mapping(raw, actual, probability=False)
    calibrated_lambda = apply_mapping(raw, expected_mapping)
    poisson_p1 = 1.0 - np.exp(-calibrated_lambda)
    poisson_p2 = 1.0 - np.exp(-calibrated_lambda) * (1.0 + calibrated_lambda)
    p1_mapping = fit_monotone_mapping(
        poisson_p1,
        (actual >= 1.0).astype("float64"),
        probability=True,
    )
    p2_mapping = fit_monotone_mapping(
        poisson_p2,
        (actual >= 2.0).astype("float64"),
        probability=True,
    )
    return {
        "expected_count": {
            "input": "selected_point_prediction_clipped_at_zero",
            "mapping": expected_mapping,
        },
        "probability_1_plus": {
            "input": "poisson_probability_1_plus_from_calibrated_expected_count",
            "mapping": p1_mapping,
        },
        "probability_2_plus": {
            "input": "poisson_probability_2_plus_from_calibrated_expected_count",
            "mapping": p2_mapping,
        },
    }


def target_is_nonnegative(config: dict[str, Any], target: str) -> bool:
    return str(config["targets"][target].get("type", "")) != "continuous_signed"


def _point_mae(actual: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.abs(prediction - actual)))


def _point_bias(actual: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(prediction - actual))


def _point_poisson_deviance(actual: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(actual, dtype="float64")
    lam = np.maximum(np.asarray(prediction, dtype="float64"), 1e-12)
    if np.any(y < 0.0):
        raise ValueError("Negative actual in point-calibration Poisson gate.")
    terms = np.empty_like(y)
    zero = y <= 0.0
    terms[zero] = lam[zero]
    nz = ~zero
    terms[nz] = y[nz] * np.log(y[nz] / lam[nz]) - (y[nz] - lam[nz])
    return float(2.0 * np.mean(terms))


def _point_brier_1plus(actual: np.ndarray, probability: np.ndarray) -> float:
    event = (np.asarray(actual, dtype="float64") >= 1.0).astype("float64")
    p = np.clip(np.asarray(probability, dtype="float64"), 0.0, 1.0)
    return float(np.mean(np.square(p - event)))


def _base_calibrated_point(
    frame: pd.DataFrame,
    payload: dict[str, Any],
    *,
    floor_at_zero: bool,
) -> np.ndarray:
    raw = frame["selected_point_prediction"].to_numpy(dtype="float64")
    if "count_calibration" in payload:
        mapping = payload["count_calibration"]["expected_count"]["mapping"]
        return np.maximum(apply_mapping(np.maximum(raw, 0.0), mapping), 0.0)
    qcal = payload.get("quantile_calibration")
    if not isinstance(qcal, dict):
        raise ValueError("Point calibration has no count or quantile calibration.")
    output = raw + float(qcal["residual_quantiles"]["q50"])
    if floor_at_zero:
        output = np.maximum(output, 0.0)
    return output


def _base_probability_1plus(
    frame: pd.DataFrame,
    payload: dict[str, Any],
    base_point: np.ndarray,
) -> np.ndarray:
    raw = frame["selected_point_prediction"].to_numpy(dtype="float64")
    if "count_calibration" in payload:
        ccal = payload["count_calibration"]
        expected = apply_mapping(
            np.maximum(raw, 0.0),
            ccal["expected_count"]["mapping"],
        )
        poisson_p1 = 1.0 - np.exp(-np.maximum(expected, 0.0))
        return np.clip(
            apply_mapping(poisson_p1, ccal["probability_1_plus"]["mapping"]),
            0.0,
            1.0,
        )
    return 1.0 - np.exp(-np.maximum(base_point, 0.0))


def fit_point_prediction_blend(
    config: dict[str, Any],
    target: str,
    frame: pd.DataFrame,
    payload: dict[str, Any],
    acceptance: dict[str, Any],
) -> dict[str, Any]:
    # Selection uses only the 2024 validation rows already loaded into frame.
    actual = frame["actual"].to_numpy(dtype="float64")
    raw = frame["selected_point_prediction"].to_numpy(dtype="float64")
    baseline = frame["baseline_projection"].to_numpy(dtype="float64")
    floor_at_zero = target_is_nonnegative(config, target)
    base = _base_calibrated_point(frame, payload, floor_at_zero=floor_at_zero)
    p1 = _base_probability_1plus(frame, payload, base)

    baseline_mae = _point_mae(actual, baseline)
    if not math.isfinite(baseline_mae) or baseline_mae <= 0.0:
        raise ValueError(f"{target}: invalid baseline MAE for point calibration.")

    candidates: list[dict[str, Any]] = []
    passing: list[tuple[float, float, float]] = []
    for alpha in POINT_PREDICTION_BLEND_CANDIDATES:
        prediction = raw + float(alpha) * (base - raw)
        if floor_at_zero:
            prediction = np.maximum(prediction, 0.0)

        candidate_mae = _point_mae(actual, prediction)
        candidate_bias = _point_bias(actual, prediction)
        abs_bias = abs(candidate_bias)
        improvement = (baseline_mae - candidate_mae) / baseline_mae * 100.0
        gates: dict[str, bool] = {
            "mae": candidate_mae <= float(acceptance["maximum_validation_mae"]) + 1e-12,
            "bias": abs_bias <= float(acceptance["maximum_allowed_bias"]) + 1e-12,
            "improvement": improvement + 1e-12 >= float(acceptance["minimum_improvement_vs_baseline_pct"]),
        }

        brier = None
        poisson = None
        if "maximum_brier_1plus" in acceptance and "maximum_poisson_deviance" in acceptance:
            brier = _point_brier_1plus(actual, p1)
            poisson = _point_poisson_deviance(actual, prediction)
            gates["brier_1plus"] = brier <= float(acceptance["maximum_brier_1plus"]) + 1e-12
            gates["poisson_deviance"] = poisson <= float(acceptance["maximum_poisson_deviance"]) + 1e-12

        passed = bool(all(gates.values()))
        candidates.append({
            "calibrated_weight": float(alpha),
            "raw_weight": float(1.0 - alpha),
            "validation_mae": candidate_mae,
            "validation_bias": candidate_bias,
            "validation_absolute_bias": abs_bias,
            "validation_improvement_vs_baseline_pct": improvement,
            "validation_brier_1plus": brier,
            "validation_poisson_deviance": poisson,
            "passed_all_configured_gates": passed,
            "failed_gates": [name for name, ok in gates.items() if not ok],
        })
        if passed:
            passing.append((candidate_mae, abs_bias, float(alpha)))

    if passing:
        if target == "rushing_yards":
            passing_candidates = [
                item
                for item in candidates
                if item["passed_all_configured_gates"]
            ]
            max_bias = float(acceptance["maximum_allowed_bias"])
            max_mae = float(acceptance["maximum_validation_mae"])
            min_improvement = float(
                acceptance["minimum_improvement_vs_baseline_pct"]
            )

            def rushing_yards_robust_margin(
                item: dict[str, Any],
            ) -> tuple[float, float, float, float]:
                abs_bias = float(item["validation_absolute_bias"])
                candidate_mae = float(item["validation_mae"])
                improvement = float(
                    item["validation_improvement_vs_baseline_pct"]
                )
                robust_margin = min(
                    (max_bias - abs_bias) / max_bias,
                    (max_mae - candidate_mae) / max_mae,
                    (improvement - min_improvement)
                    / max(abs(min_improvement), 1.0),
                )
                return (
                    -robust_margin,
                    candidate_mae,
                    abs_bias,
                    float(item["calibrated_weight"]),
                )

            chosen_item = min(
                passing_candidates,
                key=rushing_yards_robust_margin,
            )
            chosen_alpha = float(chosen_item["calibrated_weight"])
            status = (
                "validation_candidate_passed_all_gates_"
                "rushing_yards_robust_margin"
            )
        else:
            passing.sort(key=lambda item: (item[0], item[1], item[2]))
            chosen_alpha = float(passing[0][2])
            status = "validation_candidate_passed_all_gates"
    else:
        chosen_alpha = 1.0
        status = "no_validation_candidate_passed_all_gates"

    chosen = next(
        item for item in candidates
        if abs(float(item["calibrated_weight"]) - chosen_alpha) <= 1e-12
    )
    return {
        "method": "validation_only_blend_raw_with_existing_calibrated_point",
        "selection_split": "validation",
        "selection_season": int(frame["season"].iloc[0]),
        "test_rows_used_for_selection": False,
        "candidate_calibrated_weights": [float(v) for v in POINT_PREDICTION_BLEND_CANDIDATES],
        "calibrated_weight": chosen_alpha,
        "raw_weight": float(1.0 - chosen_alpha),
        "floor_at_zero": bool(floor_at_zero),
        "selection_status": status,
        "selected_validation_metrics": chosen,
        "candidate_metrics": candidates,
        "probability_calibration_unchanged": True,
        "interval_calibration_unchanged": True,
    }


def build_target_calibration(
    config: dict[str, Any],
    contract: SelectedContract,
    frame: pd.DataFrame,
    usage_source: dict[str, Any],
    acceptance: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target = contract.target
    signal = usage_signal(frame, usage_source)
    thresholds = fit_usage_buckets(signal)
    frame = frame.copy().reset_index(drop=True)
    frame["usage_bucket"] = apply_usage_buckets(signal, thresholds)

    payload: dict[str, Any] = {
        "target": target,
        "selected_architecture": contract.selected_architecture,
        "calibration_source": {
            "prediction_file": AUDIT_PATH.as_posix(),
            "split": "validation",
            "fold_id": f"dev_{contract.validation_season}",
            "season": contract.validation_season,
            "model_selection_train_end_season": contract.model_selection_train_end_season,
            "oof_residuals_used": True,
            "rows": int(len(frame)),
        },
        "reporting_test_policy": {
            "test_season": contract.test_season,
            "test_rows_used_for_calibration": False,
            "test_reporting_only_preserved": True,
        },
        "usage_bucket": {
            "source": usage_source,
            "thresholds": thresholds,
            "labels": ["low", "medium", "high"],
            "fitted_on_oof_validation_only": True,
        },
        "market_features_used": False,
        "forbidden_features_used": False,
    }

    coverage_output: list[dict[str, Any]] = []
    if target in QUANTILE_TARGETS:
        residual = frame["residual"].to_numpy(dtype="float64")
        residual_quantiles = {
            name: float(np.quantile(residual, level))
            for name, level in QUANTILE_LEVELS.items()
        }
        risk = fit_risk_widening(
            frame,
            residual,
            float(residual_quantiles["q50"]),
        )
        risk_array = apply_risk_multiplier(frame, risk)
        floor_at_zero = target_is_nonnegative(config, target)
        widening = fit_interval_widening(
            frame,
            residual_quantiles,
            risk_array,
            floor_at_zero,
        )
        qvalues = quantile_values(
            frame,
            residual_quantiles,
            risk_array,
            widening,
            floor_at_zero,
        )
        payload["quantile_calibration"] = {
            "outputs": list(QUANTILE_LEVELS.keys()),
            "method": "selected_point_plus_oof_residual_quantiles_with_interval_widening",
            "residual_quantiles": residual_quantiles,
            "intervals": {
                name: {
                    "lower": lower,
                    "upper": upper,
                    "expected_coverage": expected,
                }
                for name, (lower, upper, expected) in INTERVALS.items()
            },
            "floor_at_zero": floor_at_zero,
            "risk_widening": risk,
            "coverage_widening": widening,
        }
        coverage_output.extend(coverage_rows(target, frame, qvalues))

    if str(config["targets"][target].get("type", "")) == "count_nonnegative":
        payload["count_calibration"] = fit_count_calibration(frame)
        payload["count_outputs"] = [
            "expected_count",
            "probability_1_plus",
            "probability_2_plus",
        ]

    if target not in QUANTILE_TARGETS and "count_calibration" not in payload:
        raise ValueError(f"No Issue 26 uncertainty mode defined for target {target}.")

    if target == "tackles":
        payload["calibration_mode"] = "quantiles_and_count"
    elif target in QUANTILE_TARGETS:
        payload["calibration_mode"] = "quantiles"
    else:
        payload["calibration_mode"] = "count"

    payload["point_prediction_blend"] = fit_point_prediction_blend(
        config,
        target,
        frame,
        payload,
        acceptance,
    )

    return payload, coverage_output


def validate_coverage_table(table: pd.DataFrame) -> None:
    if list(table.columns) != COVERAGE_COLUMNS:
        raise ValueError(
            f"interval_coverage.csv header mismatch: {table.columns.tolist()}"
        )
    if table.empty:
        raise ValueError("interval_coverage.csv would be empty.")
    for column in [
        "expected_coverage",
        "actual_coverage",
        "mean_interval_width",
    ]:
        values = pd.to_numeric(table[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype="float64")).all():
            raise ValueError(f"Coverage table has nonfinite {column}.")
    if not table["actual_coverage"].between(0.0, 1.0).all():
        raise ValueError("actual_coverage outside [0,1].")
    if not table["expected_coverage"].between(0.0, 1.0).all():
        raise ValueError("expected_coverage outside [0,1].")
    if (table["mean_interval_width"] < -1e-12).any():
        raise ValueError("Negative interval width detected.")


def main() -> int:
    # ISSUE28_MARKET_EXCLUSION_PREFLIGHT
    _issue28_audit = common.prop_root() / "scripts" / "validate" / "audit_market_exclusion.py"
    _issue28_result = __import__("subprocess").run(
        [__import__("sys").executable, str(_issue28_audit), "--preflight"],
        check=False,
    )
    if _issue28_result.returncode != 0:
        raise RuntimeError("Issue 28 market-exclusion preflight failed.")

    config = common.load_config()

    acceptance_path = common.repo_root() / ACCEPTANCE_THRESHOLDS_PATH
    if not acceptance_path.is_file():
        raise FileNotFoundError(f"Missing acceptance thresholds: {acceptance_path}")
    with acceptance_path.open("r", encoding="utf-8-sig") as handle:
        acceptance_thresholds = yaml.safe_load(handle)
    if not isinstance(acceptance_thresholds, dict):
        raise ValueError("acceptance_thresholds.yaml must be a YAML mapping.")

    targets = list(config["targets"].keys())
    if set(QUANTILE_TARGETS) - set(targets):
        raise ValueError("Configured targets missing required quantile targets.")

    print("CHECK 01: selected architecture and OOF calibration contracts")
    contracts = load_selected_contracts(targets)
    # _CONFIG_ENFORCED_CALIBRATION_SPLIT
    training = config["training"]
    expected_validation = int(
        training["development_validation_season"]
    )
    expected_test = int(
        training["untouched_test_season"]
    )
    expected_train_end = int(
        training["model_selection_train_end_season"]
    )
    for target, contract in contracts.items():
        observed = (
            int(contract.validation_season),
            int(contract.test_season),
            int(contract.model_selection_train_end_season),
        )
        expected = (
            expected_validation,
            expected_test,
            expected_train_end,
        )
        if observed != expected:
            raise ValueError(
                f"{target}: selected-model split contract "
                f"{observed} != config.training {expected}"
            )
    oof = load_oof_selected_predictions(targets, contracts)

    print("CHECK 02: pregame calibration context and risk flags")
    context, usage_sources = load_context(config, targets, oof)

    coverage_records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    print("CHECK 03: calibrate quantiles and count probabilities")
    for target in targets:
        frame = context.loc[context["target"].astype(str).eq(target)].copy()
        frame = frame.reset_index(drop=True)
        if target not in acceptance_thresholds:
            raise ValueError(f"Missing acceptance thresholds for {target}.")
        payload, target_coverage = build_target_calibration(
            config,
            contracts[target],
            frame,
            usage_sources[target],
            acceptance_thresholds[target],
        )
        output_path = common.prop_root() / CALIBRATION_ROOT.relative_to(
            "docs/win/football/nfl/prop_engine"
        ) / f"{target}_calibration.json"
        write_json_atomic(output_path, payload)
        coverage_records.extend(target_coverage)
        summaries.append(
            {
                "target": target,
                "mode": payload["calibration_mode"],
                "rows": int(len(frame)),
                "architecture": contracts[target].selected_architecture,
                "calibration_path": str(output_path.relative_to(common.repo_root())).replace("\\", "/"),
            }
        )
        print(json.dumps({"status": "calibrated", **summaries[-1]}, sort_keys=True))

    print("CHECK 04: write interval coverage audit")
    coverage_table = pd.DataFrame(coverage_records, columns=COVERAGE_COLUMNS)
    coverage_table = coverage_table.sort_values(
        ["target", "interval", "position_group", "usage_bucket"],
        kind="mergesort",
    ).reset_index(drop=True)
    validate_coverage_table(coverage_table)
    common.write_csv_atomic(coverage_table, COVERAGE_PATH)

    validation_season = next(iter({c.validation_season for c in contracts.values()}))
    test_season = next(iter({c.test_season for c in contracts.values()}))
    run_payload = {
        "status": "passed",
        "targets": len(targets),
        "quantile_targets": QUANTILE_TARGETS,
        "count_targets": [
            target
            for target in targets
            if str(config["targets"][target].get("type", "")) == "count_nonnegative"
        ],
        "quantiles": list(QUANTILE_LEVELS.keys()),
        "count_outputs": [
            "expected_count",
            "probability_1_plus",
            "probability_2_plus",
        ],
        "calibration_fold_id": f"dev_{validation_season}",
        "calibration_season": validation_season,
        "oof_residuals_used": True,
        "test_season": test_season,
        "test_rows_used_for_calibration": False,
        "undercoverage_widening": True,
        "rookie_widening": True,
        "backup_promotion_widening": True,
        "low_history_widening": True,
        "market_features_used": False,
        "interval_coverage_rows": int(len(coverage_table)),
        "results": summaries,
    }
    try:
        common.log_run("calibrate_uncertainty.py", run_payload)
    except Exception:
        # Logging must not invalidate otherwise atomic calibration artifacts.
        pass
    print(json.dumps({"script": "calibrate_uncertainty.py", "payload": run_payload}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
