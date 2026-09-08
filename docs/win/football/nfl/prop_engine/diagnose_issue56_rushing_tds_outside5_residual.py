#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

PROP = Path(__file__).resolve().parent
TRAIN_DIR = PROP / "scripts/train"
SCRIPTS_DIR = PROP / "scripts"
for p in [TRAIN_DIR, SCRIPTS_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import common
import train_direct_models as direct

TARGET = "rushing_tds"
SELECTION_TRAIN_END = 2023
VALIDATION_SEASON = 2024
FINAL_TRAIN_END = 2024
TEST_SEASON = 2025

AUDIT_PATH = PROP / "evaluation/model_selection_predictions.parquet"
THRESHOLD_PATH = PROP / "config/acceptance_thresholds.yaml"
FEATURE_CONFIG_PATH = PROP / "config/features/rushing_tds.json"
CONFIG_PATH = PROP / "config/prop_engine.yaml"
OUT_JSON = PROP / "evaluation/issue56_rushing_tds_outside5_residual_diagnostic.json"
OUT_CSV = PROP / "evaluation/issue56_rushing_tds_outside5_residual_predictions.csv"

GRAIN = ["season", "week", "game_id", "player_id"]


def repo_root() -> Path:
    current = PROP.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists():
            return candidate
    raise AssertionError("Could not locate repository root")


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


def clean_id(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .replace({"nan": "", "None": "", "<NA>": ""})
    )


def build_outside5_labels(config: dict, seasons: list[int]) -> pd.DataFrame:
    root = repo_root()
    pattern = config["paths"]["pbp_pattern"]
    frames = []

    for season in seasons:
        path = root / pattern.format(season=season)
        if not path.is_file():
            raise FileNotFoundError(f"Required PBP missing: {path}")

        pbp = pd.read_csv(
            path,
            usecols=[
                "season_type",
                "week",
                "game_id",
                "yardline_100",
                "rush_attempt",
                "rusher_player_id",
                "rush_touchdown",
            ],
            low_memory=False,
        )
        pbp = pbp.loc[
            pbp["season_type"].astype(str).str.upper().eq("REG")
        ].copy()
        pbp["season"] = season
        pbp["week"] = pd.to_numeric(pbp["week"], errors="raise").astype(int)
        pbp["game_id"] = pbp["game_id"].astype(str).str.strip()
        pbp["player_id"] = clean_id(pbp["rusher_player_id"])
        pbp["yardline_100"] = pd.to_numeric(pbp["yardline_100"], errors="coerce")
        pbp["rush_attempt"] = pd.to_numeric(
            pbp["rush_attempt"], errors="coerce"
        ).fillna(0.0)
        pbp["rush_touchdown"] = pd.to_numeric(
            pbp["rush_touchdown"], errors="coerce"
        ).fillna(0.0)

        rush = pbp.loc[
            pbp["rush_attempt"].eq(1.0)
            & pbp["player_id"].ne("")
        ].copy()

        rush["_outside5_td"] = (
            rush["rush_touchdown"].eq(1.0)
            & (
                rush["yardline_100"].isna()
                | rush["yardline_100"].gt(5.0)
            )
        ).astype(float)
        rush["_all_td"] = rush["rush_touchdown"].eq(1.0).astype(float)
        rush["_inside5_td"] = (
            rush["rush_touchdown"].eq(1.0)
            & rush["yardline_100"].notna()
            & rush["yardline_100"].le(5.0)
        ).astype(float)

        agg = rush.groupby(
            GRAIN,
            as_index=False,
            dropna=False,
        ).agg(
            outside5_rushing_tds=("_outside5_td", "sum"),
            all_pbp_rushing_tds=("_all_td", "sum"),
            inside5_rushing_tds=("_inside5_td", "sum"),
        )
        frames.append(agg)

    out = pd.concat(frames, ignore_index=True)
    if out.duplicated(GRAIN).any():
        raise AssertionError("Duplicate outside5 PBP grain")
    return out


def prediction_metrics(
    actual: np.ndarray,
    pred: np.ndarray,
    baseline: np.ndarray,
) -> dict:
    y = np.asarray(actual, dtype="float64")
    p = np.maximum(np.asarray(pred, dtype="float64"), 0.0)
    b = np.asarray(baseline, dtype="float64")

    valid = np.isfinite(y) & np.isfinite(p) & np.isfinite(b)
    if not valid.all():
        raise AssertionError(f"Nonfinite metric rows: {int((~valid).sum())}")

    mae = float(np.mean(np.abs(y - p)))
    bias = float(np.mean(p - y))
    baseline_mae = float(np.mean(np.abs(y - b)))
    improvement = 100.0 * (baseline_mae - mae) / baseline_mae

    lam = np.maximum(p, 1e-12)
    terms = np.empty_like(y)
    zero = y <= 0.0
    terms[zero] = lam[zero]
    pos = ~zero
    terms[pos] = (
        y[pos] * np.log(y[pos] / lam[pos])
        - (y[pos] - lam[pos])
    )
    poisson = float(2.0 * np.mean(terms))

    p1 = np.clip(1.0 - np.exp(-np.maximum(p, 0.0)), 0.0, 1.0)
    event = (y >= 1.0).astype(float)
    brier = float(np.mean(np.square(p1 - event)))

    return {
        "rows": int(len(y)),
        "mae": mae,
        "bias": bias,
        "abs_bias": abs(bias),
        "baseline_mae": baseline_mae,
        "improvement_vs_baseline_pct": improvement,
        "poisson_deviance": poisson,
        "brier_1plus_independent_poisson": brier,
        "mean_actual": float(np.mean(y)),
        "mean_prediction": float(np.mean(p)),
    }


def gate_failures(metrics: dict, threshold: dict) -> list[str]:
    failed = []
    if metrics["mae"] > float(threshold["maximum_validation_mae"]):
        failed.append("mae")
    if metrics["abs_bias"] > float(threshold["maximum_allowed_bias"]):
        failed.append("bias")
    if metrics["improvement_vs_baseline_pct"] < float(
        threshold["minimum_improvement_vs_baseline_pct"]
    ):
        failed.append("improvement")
    if metrics["poisson_deviance"] > float(
        threshold["maximum_poisson_deviance"]
    ):
        failed.append("poisson_deviance")
    if metrics["brier_1plus_independent_poisson"] > float(
        threshold["maximum_brier_1plus"]
    ):
        failed.append("brier_1plus")
    return failed


def frame_for_model(
    features: pd.DataFrame,
    manifest: dict,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    positions = {
        str(x).strip().upper()
        for x in manifest["eligible_positions"]
    }

    frame = features.copy()
    frame["position"] = (
        frame["position"].fillna("").astype(str).str.strip().str.upper()
    )
    frame = frame.loc[frame["position"].isin(positions)].copy()

    frame = frame.merge(
        labels[GRAIN + ["outside5_rushing_tds"]],
        on=GRAIN,
        how="left",
        validate="one_to_one",
    )
    frame["outside5_rushing_tds"] = pd.to_numeric(
        frame["outside5_rushing_tds"], errors="coerce"
    ).fillna(0.0)

    # Match the direct-model target availability cohort.
    actual = pd.to_numeric(
        frame["target_rushing_tds"], errors="coerce"
    )
    frame = frame.loc[actual.notna()].copy()
    frame["_outside5_target"] = frame["outside5_rushing_tds"].astype(float)

    return frame


def train_selection_and_final(
    frame: pd.DataFrame,
    manifest: dict,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame, int, dict]:
    numeric_features = list(manifest["numeric_features"])
    categorical_features = list(manifest["categorical_features"])
    feature_names = [*numeric_features, *categorical_features]

    train = frame.loc[
        pd.to_numeric(frame["season"]).le(SELECTION_TRAIN_END)
    ].copy()
    valid = frame.loc[
        pd.to_numeric(frame["season"]).eq(VALIDATION_SEASON)
    ].copy()
    final_train = frame.loc[
        pd.to_numeric(frame["season"]).le(FINAL_TRAIN_END)
    ].copy()
    test = frame.loc[
        pd.to_numeric(frame["season"]).eq(TEST_SEASON)
    ].copy()

    if train.empty or valid.empty or final_train.empty or test.empty:
        raise AssertionError(
            f"Empty split: train={len(train)} valid={len(valid)} "
            f"final={len(final_train)} test={len(test)}"
        )

    levels_selection = direct.categorical_levels(
        train, categorical_features
    )
    X_train = direct.model_matrix(
        train,
        numeric_features,
        categorical_features,
        levels_selection,
    )
    X_valid = direct.model_matrix(
        valid,
        numeric_features,
        categorical_features,
        levels_selection,
    )

    y_train = pd.to_numeric(
        train["_outside5_target"], errors="raise"
    ).astype(float)
    y_valid = pd.to_numeric(
        valid["_outside5_target"], errors="raise"
    ).astype(float)

    train_set = lgb.Dataset(
        X_train,
        label=y_train,
        feature_name=feature_names,
        categorical_feature=categorical_features,
        free_raw_data=False,
    )
    valid_set = lgb.Dataset(
        X_valid,
        label=y_valid,
        feature_name=feature_names,
        categorical_feature=categorical_features,
        reference=train_set,
        free_raw_data=False,
    )

    params = direct.params_for("poisson")
    selected = lgb.train(
        params,
        train_set,
        num_boost_round=2500,
        valid_sets=[valid_set],
        valid_names=["validation_2024"],
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=100,
                first_metric_only=True,
                verbose=False,
            ),
            lgb.log_evaluation(period=0),
        ],
    )
    best_iteration = int(selected.best_iteration)
    if best_iteration < 1:
        raise AssertionError("Invalid selected best_iteration")

    valid_pred = selected.predict(
        X_valid,
        num_iteration=best_iteration,
    )
    valid_pred = direct.transform_prediction(valid_pred, "poisson")

    levels_final = direct.categorical_levels(
        final_train, categorical_features
    )
    X_final = direct.model_matrix(
        final_train,
        numeric_features,
        categorical_features,
        levels_final,
    )
    y_final = pd.to_numeric(
        final_train["_outside5_target"], errors="raise"
    ).astype(float)

    final_set = lgb.Dataset(
        X_final,
        label=y_final,
        feature_name=feature_names,
        categorical_feature=categorical_features,
        free_raw_data=False,
    )
    final_model = lgb.train(
        params,
        final_set,
        num_boost_round=best_iteration,
        callbacks=[lgb.log_evaluation(period=0)],
    )

    X_test = direct.model_matrix(
        test,
        numeric_features,
        categorical_features,
        levels_final,
    )
    test_pred = final_model.predict(
        X_test,
        num_iteration=best_iteration,
    )
    test_pred = direct.transform_prediction(test_pred, "poisson")

    meta = {
        "objective": "poisson",
        "params": params,
        "best_iteration_selected_on_2024": best_iteration,
        "selection_train_rows": int(len(train)),
        "validation_2024_rows": int(len(valid)),
        "final_train_through_2024_rows": int(len(final_train)),
        "test_2025_rows": int(len(test)),
        "test_used_for_selection": False,
        "market_features_used": False,
    }
    return (
        np.asarray(valid_pred, dtype="float64"),
        valid[GRAIN].copy(),
        np.asarray(test_pred, dtype="float64"),
        test[GRAIN].copy(),
        best_iteration,
        meta,
    )


def audit_split(audit: pd.DataFrame, split: str, season: int) -> pd.DataFrame:
    f = audit.loc[
        audit["target"].astype(str).eq(TARGET)
        & audit["split"].astype(str).eq(split)
    ].copy()
    if f.empty:
        raise AssertionError(f"No audit rows: {split}")
    seasons = set(pd.to_numeric(f["season"], errors="raise").astype(int))
    if seasons != {season}:
        raise AssertionError(f"{split}: unexpected seasons={sorted(seasons)}")
    return f


def main() -> int:
    for p in [
        AUDIT_PATH,
        THRESHOLD_PATH,
        FEATURE_CONFIG_PATH,
        CONFIG_PATH,
    ]:
        if not p.is_file():
            raise FileNotFoundError(p)

    config = load_yaml(CONFIG_PATH)
    manifest = load_json(FEATURE_CONFIG_PATH)
    thresholds = load_yaml(THRESHOLD_PATH)
    threshold = dict(thresholds[TARGET])

    historical_path = (
        repo_root()
        / config["paths"]["historical_features"]
    )
    if not historical_path.is_file():
        raise FileNotFoundError(historical_path)

    required_columns = list(dict.fromkeys([
        *GRAIN,
        "position",
        "target_rushing_tds",
        *manifest["numeric_features"],
        *manifest["categorical_features"],
    ]))
    features = pd.read_parquet(
        historical_path,
        columns=required_columns,
    )
    features["season"] = pd.to_numeric(
        features["season"], errors="raise"
    ).astype(int)
    features["week"] = pd.to_numeric(
        features["week"], errors="raise"
    ).astype(int)
    features["game_id"] = features["game_id"].astype(str).str.strip()
    features["player_id"] = clean_id(features["player_id"])

    seasons = sorted(
        s for s in features["season"].unique().tolist()
        if int(s) <= TEST_SEASON
    )
    labels = build_outside5_labels(config, [int(s) for s in seasons])

    model_frame = frame_for_model(features, manifest, labels)

    (
        valid_resid,
        valid_keys,
        test_resid,
        test_keys,
        best_iteration,
        training_meta,
    ) = train_selection_and_final(model_frame, manifest)

    valid_residual = valid_keys.copy()
    valid_residual["outside5_residual_prediction"] = valid_resid
    test_residual = test_keys.copy()
    test_residual["outside5_residual_prediction"] = test_resid

    audit = pd.read_parquet(AUDIT_PATH)
    valid_audit = audit_split(audit, "validation", VALIDATION_SEASON)
    test_audit = audit_split(audit, "test", TEST_SEASON)

    def combine(
        base: pd.DataFrame,
        residual: pd.DataFrame,
        split: str,
    ) -> tuple[pd.DataFrame, dict, dict]:
        merged = base.merge(
            residual,
            on=GRAIN,
            how="left",
            validate="one_to_one",
        )
        if merged["outside5_residual_prediction"].isna().any():
            n = int(merged["outside5_residual_prediction"].isna().sum())
            raise AssertionError(
                f"{split}: missing outside5 residual predictions on {n} audit rows"
            )

        for c in [
            "actual",
            "baseline_projection",
            "component_projection",
            "outside5_residual_prediction",
        ]:
            merged[c] = pd.to_numeric(
                merged[c], errors="coerce"
            ).astype(float)
            if merged[c].isna().any():
                raise AssertionError(f"{split}: NaN in {c}")

        merged["structural_projection"] = (
            merged["component_projection"]
            + merged["outside5_residual_prediction"]
        ).clip(lower=0.0)

        raw_metrics = prediction_metrics(
            merged["actual"].to_numpy(),
            merged["component_projection"].to_numpy(),
            merged["baseline_projection"].to_numpy(),
        )
        structural_metrics = prediction_metrics(
            merged["actual"].to_numpy(),
            merged["structural_projection"].to_numpy(),
            merged["baseline_projection"].to_numpy(),
        )
        return merged, raw_metrics, structural_metrics

    valid_combined, valid_raw, valid_structural = combine(
        valid_audit, valid_residual, "validation"
    )
    test_combined, test_raw, test_structural = combine(
        test_audit, test_residual, "test"
    )

    valid_failed = gate_failures(valid_structural, threshold)
    test_failed = gate_failures(test_structural, threshold)

    output = pd.concat([
        valid_combined.assign(split_label="validation_2024"),
        test_combined.assign(split_label="test_2025_reporting_only"),
    ], ignore_index=True)

    output[[
        "split_label",
        *GRAIN,
        "actual",
        "baseline_projection",
        "component_projection",
        "outside5_residual_prediction",
        "structural_projection",
    ]].to_csv(OUT_CSV, index=False)

    payload = {
        "status": "complete",
        "target": TARGET,
        "diagnostic_only": True,
        "production_files_modified": False,
        "market_data_used": False,
        "structural_candidate": (
            "existing inside5 goal-line component "
            "+ Poisson outside5 rushing-TD residual"
        ),
        "outside5_residual_definition": (
            "rush_touchdown == 1 and yardline_100 > 5 "
            "(or yardline unavailable on a recorded rush TD)"
        ),
        "weighting": {
            "inside5_component_weight": 1.0,
            "outside5_residual_weight": 1.0,
            "weights_tuned": False,
        },
        "training": training_meta,
        "thresholds_unchanged": threshold,
        "validation_2024": {
            "existing_component": valid_raw,
            "structural_candidate": valid_structural,
            "failed_gates": valid_failed,
            "passed_point_and_independent_poisson_gates": not valid_failed,
        },
        "test_2025_reporting_only": {
            "existing_component": test_raw,
            "structural_candidate": test_structural,
            "failed_gates": test_failed,
            "passed_point_and_independent_poisson_gates": not test_failed,
            "used_for_selection": False,
        },
        "outputs": {
            "predictions_csv": str(OUT_CSV),
            "summary_json": str(OUT_JSON),
        },
    }
    OUT_JSON.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("ISSUE 56 RUSHING TDS OUTSIDE-5 RESIDUAL DIAGNOSTIC")
    print(
        f"best_iteration_selected_on_2024={best_iteration} "
        f"selection_train_rows={training_meta['selection_train_rows']} "
        f"validation_rows={training_meta['validation_2024_rows']}"
    )
    print("STRUCTURE: full_projection = existing_inside5_component + outside5_residual")
    print("weights: inside5=1.0 outside5=1.0 tuned=false")
    print(
        "2024 EXISTING: "
        f"mae={valid_raw['mae']:.6f} "
        f"bias={valid_raw['bias']:.6f} "
        f"improvement={valid_raw['improvement_vs_baseline_pct']:.6f}% "
        f"poisson={valid_raw['poisson_deviance']:.6f} "
        f"brier={valid_raw['brier_1plus_independent_poisson']:.6f}"
    )
    print(
        "2024 STRUCTURAL: "
        f"mae={valid_structural['mae']:.6f} "
        f"bias={valid_structural['bias']:.6f} "
        f"improvement={valid_structural['improvement_vs_baseline_pct']:.6f}% "
        f"poisson={valid_structural['poisson_deviance']:.6f} "
        f"brier={valid_structural['brier_1plus_independent_poisson']:.6f} "
        f"failed={';'.join(valid_failed) if valid_failed else 'none'}"
    )
    print(
        "2025 REPORTING ONLY: "
        f"mae={test_structural['mae']:.6f} "
        f"bias={test_structural['bias']:.6f} "
        f"improvement={test_structural['improvement_vs_baseline_pct']:.6f}% "
        f"poisson={test_structural['poisson_deviance']:.6f} "
        f"brier={test_structural['brier_1plus_independent_poisson']:.6f} "
        f"failed={';'.join(test_failed) if test_failed else 'none'}"
    )
    print(f"csv={OUT_CSV}")
    print(f"json={OUT_JSON}")
    print("ISSUE 56 RUSHING TDS OUTSIDE-5 RESIDUAL DIAGNOSTIC: COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
