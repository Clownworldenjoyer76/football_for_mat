#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
SELECTED_PATH = PROP / "models/rushing_tds/selected_model.json"
OUT_JSON = PROP / "evaluation/issue56_rushing_tds_outside5_residual_diagnostic.json"
OUT_CSV = PROP / "evaluation/issue56_rushing_tds_outside5_residual_predictions.csv"

GRAIN = ["season", "week", "game_id", "player_id"]
PBP_REQUIRED = [
    "season_type",
    "week",
    "game_id",
    "yardline_100",
    "rush_attempt",
    "rusher_player_id",
    "rush_touchdown",
]


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


def required_pbp_seasons(config: dict) -> list[int]:
    rich_start = int(config["seasons"]["rich_feature_start"])
    if rich_start > SELECTION_TRAIN_END:
        raise AssertionError(
            f"rich_feature_start={rich_start} leaves no pre-2024 selection training seasons"
        )
    return list(range(rich_start, TEST_SEASON + 1))


def preflight(config: dict, manifest: dict) -> dict:
    root = repo_root()
    rich_start = int(config["seasons"]["rich_feature_start"])
    seasons = required_pbp_seasons(config)

    # Explicit chronological contract. Never fabricate labels before rich PBP begins.
    selection_seasons = [s for s in seasons if s <= SELECTION_TRAIN_END]
    if not selection_seasons:
        raise AssertionError("No PBP-backed selection-training seasons")
    if VALIDATION_SEASON not in seasons or TEST_SEASON not in seasons:
        raise AssertionError("Required validation/test PBP seasons are outside configured rich range")

    # Check every required PBP file and schema before any model training.
    pattern = config["paths"]["pbp_pattern"]
    checked = []
    for season in seasons:
        path = root / pattern.format(season=season)
        if not path.is_file():
            raise FileNotFoundError(f"Required rich-feature PBP missing: {path}")
        header = pd.read_csv(path, nrows=0)
        missing = [c for c in PBP_REQUIRED if c not in header.columns]
        if missing:
            raise AssertionError(f"{path}: missing PBP columns {missing}")
        checked.append(str(path))

    historical_path = root / config["paths"]["historical_features"]
    if not historical_path.is_file():
        raise FileNotFoundError(historical_path)

    required_features = list(dict.fromkeys([
        *GRAIN,
        "position",
        "target_rushing_tds",
        *manifest["numeric_features"],
        *manifest["categorical_features"],
    ]))
    schema = pd.read_parquet(historical_path, columns=None).head(0)
    missing_features = [c for c in required_features if c not in schema.columns]
    if missing_features:
        raise AssertionError(
            f"Historical feature table missing required columns: {missing_features[:20]}"
        )

    # Existing feature contract must remain market-free.
    common.reject_forbidden_feature_columns(
        [*manifest["numeric_features"], *manifest["categorical_features"]],
        config,
    )

    for p in [AUDIT_PATH, THRESHOLD_PATH, SELECTED_PATH]:
        if not p.is_file():
            raise FileNotFoundError(p)

    selected = load_json(SELECTED_PATH)
    if selected.get("selected_architecture") != "component":
        raise AssertionError(
            f"Expected current rushing_tds architecture=component; found "
            f"{selected.get('selected_architecture')}"
        )
    if selected.get("validation_season") != VALIDATION_SEASON:
        raise AssertionError("selected_model validation season mismatch")
    if selected.get("test_season") != TEST_SEASON:
        raise AssertionError("selected_model test season mismatch")
    if selected.get("test_used_for_selection") is not False:
        raise AssertionError("selected_model test_used_for_selection must be false")
    if selected.get("market_features_used") is not False:
        raise AssertionError("selected_model market_features_used must be false")

    audit = pd.read_parquet(
        AUDIT_PATH,
        columns=["split", "season", "target", "actual", "baseline_projection", "component_projection"],
    )
    rt = audit.loc[audit["target"].astype(str).eq(TARGET)].copy()
    for split, season in [("validation", VALIDATION_SEASON), ("test", TEST_SEASON)]:
        f = rt.loc[rt["split"].astype(str).eq(split)].copy()
        if f.empty:
            raise AssertionError(f"No {TARGET} audit rows for {split}")
        found = set(pd.to_numeric(f["season"], errors="raise").astype(int).unique())
        if found != {season}:
            raise AssertionError(f"{split}: unexpected seasons {sorted(found)}")

    info = {
        "rich_feature_start": rich_start,
        "pbp_seasons": seasons,
        "selection_training_seasons": selection_seasons,
        "validation_season": VALIDATION_SEASON,
        "final_fit_seasons": list(range(rich_start, FINAL_TRAIN_END + 1)),
        "test_season": TEST_SEASON,
        "pbp_files_checked": len(checked),
        "market_features_used": False,
    }
    print("PREFLIGHT PASS: PBP-backed chronology and schemas verified")
    print(json.dumps(info, sort_keys=True))
    return info


def build_td_labels(config: dict, seasons: list[int]) -> pd.DataFrame:
    root = repo_root()
    pattern = config["paths"]["pbp_pattern"]
    frames = []

    for season in seasons:
        path = root / pattern.format(season=season)
        pbp = pd.read_csv(path, usecols=PBP_REQUIRED, low_memory=False)
        pbp = pbp.loc[pbp["season_type"].astype(str).str.upper().eq("REG")].copy()
        pbp["season"] = season
        pbp["week"] = pd.to_numeric(pbp["week"], errors="raise").astype(int)
        pbp["game_id"] = pbp["game_id"].astype(str).str.strip()
        pbp["player_id"] = clean_id(pbp["rusher_player_id"])
        pbp["yardline_100"] = pd.to_numeric(pbp["yardline_100"], errors="coerce")
        pbp["rush_attempt"] = pd.to_numeric(pbp["rush_attempt"], errors="coerce").fillna(0.0)
        pbp["rush_touchdown"] = pd.to_numeric(pbp["rush_touchdown"], errors="coerce").fillna(0.0)

        rush = pbp.loc[pbp["rush_attempt"].eq(1.0) & pbp["player_id"].ne("")].copy()
        rush["_all_td"] = rush["rush_touchdown"].eq(1.0).astype(float)
        rush["_inside5_td"] = (
            rush["rush_touchdown"].eq(1.0)
            & rush["yardline_100"].notna()
            & rush["yardline_100"].le(5.0)
        ).astype(float)

        agg = rush.groupby(GRAIN, as_index=False, dropna=False).agg(
            all_pbp_rushing_tds=("_all_td", "sum"),
            inside5_rushing_tds=("_inside5_td", "sum"),
        )
        agg["outside5_residual_tds"] = (
            agg["all_pbp_rushing_tds"] - agg["inside5_rushing_tds"]
        )
        if (agg["outside5_residual_tds"] < 0).any():
            raise AssertionError(f"{season}: negative outside5 residual label")
        frames.append(agg)

    out = pd.concat(frames, ignore_index=True)
    if out.duplicated(GRAIN).any():
        raise AssertionError("Duplicate PBP TD-label grain")
    return out


def build_model_frame(
    features: pd.DataFrame,
    manifest: dict,
    labels: pd.DataFrame,
    rich_start: int,
) -> pd.DataFrame:
    positions = {str(x).strip().upper() for x in manifest["eligible_positions"]}
    frame = features.copy()
    frame["season"] = pd.to_numeric(frame["season"], errors="raise").astype(int)

    # Critical: pre-rich seasons are excluded, never filled as zero labels.
    frame = frame.loc[frame["season"].between(rich_start, TEST_SEASON)].copy()
    frame["position"] = frame["position"].fillna("").astype(str).str.strip().str.upper()
    frame = frame.loc[frame["position"].isin(positions)].copy()

    frame = frame.merge(labels, on=GRAIN, how="left", validate="one_to_one")
    for col in ["all_pbp_rushing_tds", "inside5_rushing_tds", "outside5_residual_tds"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)

    target = pd.to_numeric(frame["target_rushing_tds"], errors="coerce")
    frame = frame.loc[target.notna()].copy()
    frame["target_rushing_tds"] = target.loc[frame.index].astype(float)

    # Full weekly target must reconcile exactly to PBP before any fitting.
    mismatch = ~np.isclose(
        frame["target_rushing_tds"].to_numpy(dtype=float),
        frame["all_pbp_rushing_tds"].to_numpy(dtype=float),
        rtol=0.0,
        atol=0.0,
    )
    if mismatch.any():
        sample = frame.loc[
            mismatch,
            [*GRAIN, "target_rushing_tds", "all_pbp_rushing_tds", "inside5_rushing_tds"],
        ].head(20).to_dict(orient="records")
        raise AssertionError(
            f"PBP rushing-TD labels do not reconcile to canonical target; "
            f"mismatch_rows={int(mismatch.sum())} sample={sample}"
        )

    frame["_outside5_target"] = (
        frame["target_rushing_tds"] - frame["inside5_rushing_tds"]
    )
    if (frame["_outside5_target"] < 0).any():
        raise AssertionError("Canonical target minus inside5 TDs produced negative residual")
    if not np.allclose(
        frame["_outside5_target"].to_numpy(dtype=float),
        frame["outside5_residual_tds"].to_numpy(dtype=float),
        rtol=0.0,
        atol=0.0,
    ):
        raise AssertionError("Residual target does not match PBP outside5 decomposition")

    return frame


def prediction_metrics(actual: np.ndarray, pred: np.ndarray, baseline: np.ndarray) -> dict:
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
    terms[pos] = y[pos] * np.log(y[pos] / lam[pos]) - (y[pos] - lam[pos])
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
    if metrics["improvement_vs_baseline_pct"] < float(threshold["minimum_improvement_vs_baseline_pct"]):
        failed.append("improvement")
    if metrics["poisson_deviance"] > float(threshold["maximum_poisson_deviance"]):
        failed.append("poisson_deviance")
    if metrics["brier_1plus_independent_poisson"] > float(threshold["maximum_brier_1plus"]):
        failed.append("brier_1plus")
    return failed


def train_residual(frame: pd.DataFrame, manifest: dict):
    numeric_features = list(manifest["numeric_features"])
    categorical_features = list(manifest["categorical_features"])
    feature_names = [*numeric_features, *categorical_features]

    train = frame.loc[frame["season"].between(int(frame["season"].min()), SELECTION_TRAIN_END)].copy()
    valid = frame.loc[frame["season"].eq(VALIDATION_SEASON)].copy()
    final_train = frame.loc[frame["season"].le(FINAL_TRAIN_END)].copy()
    test = frame.loc[frame["season"].eq(TEST_SEASON)].copy()

    if train.empty or valid.empty or final_train.empty or test.empty:
        raise AssertionError(
            f"Empty split: train={len(train)} valid={len(valid)} final={len(final_train)} test={len(test)}"
        )

    levels_selection = direct.categorical_levels(train, categorical_features)
    X_train = direct.model_matrix(train, numeric_features, categorical_features, levels_selection)
    X_valid = direct.model_matrix(valid, numeric_features, categorical_features, levels_selection)
    y_train = pd.to_numeric(train["_outside5_target"], errors="raise").astype(float)
    y_valid = pd.to_numeric(valid["_outside5_target"], errors="raise").astype(float)

    params = direct.params_for("poisson")
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

    selected = lgb.train(
        params,
        train_set,
        num_boost_round=2500,
        valid_sets=[valid_set],
        valid_names=["validation_2024"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, first_metric_only=True, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    best_iteration = int(selected.best_iteration)
    if best_iteration < 1:
        raise AssertionError("Invalid best_iteration")

    valid_pred = direct.transform_prediction(
        selected.predict(X_valid, num_iteration=best_iteration), "poisson"
    )

    levels_final = direct.categorical_levels(final_train, categorical_features)
    X_final = direct.model_matrix(final_train, numeric_features, categorical_features, levels_final)
    y_final = pd.to_numeric(final_train["_outside5_target"], errors="raise").astype(float)
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

    X_test = direct.model_matrix(test, numeric_features, categorical_features, levels_final)
    test_pred = direct.transform_prediction(
        final_model.predict(X_test, num_iteration=best_iteration), "poisson"
    )

    valid_out = valid[GRAIN].copy()
    valid_out["outside5_residual_prediction"] = np.asarray(valid_pred, dtype=float)
    test_out = test[GRAIN].copy()
    test_out["outside5_residual_prediction"] = np.asarray(test_pred, dtype=float)

    meta = {
        "objective": "poisson",
        "params": params,
        "best_iteration_selected_on_2024": best_iteration,
        "selection_train_seasons": sorted(train["season"].unique().astype(int).tolist()),
        "selection_train_rows": int(len(train)),
        "validation_2024_rows": int(len(valid)),
        "final_train_seasons": sorted(final_train["season"].unique().astype(int).tolist()),
        "final_train_rows": int(len(final_train)),
        "test_2025_rows": int(len(test)),
        "test_used_for_selection": False,
        "market_features_used": False,
    }
    return valid_out, test_out, meta


def audit_split(audit: pd.DataFrame, split: str, season: int) -> pd.DataFrame:
    f = audit.loc[
        audit["target"].astype(str).eq(TARGET)
        & audit["split"].astype(str).eq(split)
    ].copy()
    if f.empty:
        raise AssertionError(f"No audit rows: {split}")
    found = set(pd.to_numeric(f["season"], errors="raise").astype(int).unique())
    if found != {season}:
        raise AssertionError(f"{split}: unexpected seasons {sorted(found)}")
    return f


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    config = load_yaml(CONFIG_PATH)
    manifest = load_json(FEATURE_CONFIG_PATH)
    preflight_info = preflight(config, manifest)
    if args.preflight_only:
        print("ISSUE 56 RUSHING TDS OUTSIDE5 RESIDUAL PREFLIGHT: PASS")
        return 0

    thresholds = load_yaml(THRESHOLD_PATH)
    threshold = dict(thresholds[TARGET])
    rich_start = int(config["seasons"]["rich_feature_start"])
    seasons = required_pbp_seasons(config)

    historical_path = repo_root() / config["paths"]["historical_features"]
    required_columns = list(dict.fromkeys([
        *GRAIN,
        "position",
        "target_rushing_tds",
        *manifest["numeric_features"],
        *manifest["categorical_features"],
    ]))
    features = pd.read_parquet(historical_path, columns=required_columns)
    features["season"] = pd.to_numeric(features["season"], errors="raise").astype(int)
    features["week"] = pd.to_numeric(features["week"], errors="raise").astype(int)
    features["game_id"] = features["game_id"].astype(str).str.strip()
    features["player_id"] = clean_id(features["player_id"])

    labels = build_td_labels(config, seasons)
    model_frame = build_model_frame(features, manifest, labels, rich_start)

    print(
        "LABEL CONTRACT PASS: "
        f"seasons={sorted(model_frame['season'].unique().astype(int).tolist())} "
        f"rows={len(model_frame)} target_pbp_mismatch=0 pre_rich_rows_used=0"
    )

    valid_residual, test_residual, training_meta = train_residual(model_frame, manifest)

    audit = pd.read_parquet(AUDIT_PATH)
    valid_audit = audit_split(audit, "validation", VALIDATION_SEASON)
    test_audit = audit_split(audit, "test", TEST_SEASON)

    def combine(base: pd.DataFrame, residual: pd.DataFrame, split: str):
        merged = base.merge(residual, on=GRAIN, how="left", validate="one_to_one")
        if merged["outside5_residual_prediction"].isna().any():
            n = int(merged["outside5_residual_prediction"].isna().sum())
            raise AssertionError(f"{split}: missing residual predictions on {n} audit rows")

        for c in ["actual", "baseline_projection", "component_projection", "outside5_residual_prediction"]:
            merged[c] = pd.to_numeric(merged[c], errors="coerce").astype(float)
            if merged[c].isna().any():
                raise AssertionError(f"{split}: NaN in {c}")

        merged["structural_projection"] = (
            merged["component_projection"] + merged["outside5_residual_prediction"]
        ).clip(lower=0.0)

        existing = prediction_metrics(
            merged["actual"].to_numpy(),
            merged["component_projection"].to_numpy(),
            merged["baseline_projection"].to_numpy(),
        )
        structural = prediction_metrics(
            merged["actual"].to_numpy(),
            merged["structural_projection"].to_numpy(),
            merged["baseline_projection"].to_numpy(),
        )
        return merged, existing, structural

    valid_combined, valid_existing, valid_structural = combine(
        valid_audit, valid_residual, "validation"
    )
    test_combined, test_existing, test_structural = combine(
        test_audit, test_residual, "test"
    )

    valid_failed = gate_failures(valid_structural, threshold)
    test_failed = gate_failures(test_structural, threshold)

    pd.concat([
        valid_combined.assign(split_label="validation_2024"),
        test_combined.assign(split_label="test_2025_reporting_only"),
    ], ignore_index=True)[[
        "split_label", *GRAIN, "actual", "baseline_projection",
        "component_projection", "outside5_residual_prediction", "structural_projection",
    ]].to_csv(OUT_CSV, index=False)

    payload = {
        "status": "complete",
        "target": TARGET,
        "diagnostic_only": True,
        "production_files_modified": False,
        "market_data_used": False,
        "preflight": preflight_info,
        "structural_candidate": (
            "existing inside5 goal-line component + Poisson outside5 rushing-TD residual"
        ),
        "residual_target": (
            "canonical target_rushing_tds minus exact PBP rushing TDs from yardline_100 <= 5"
        ),
        "weights": {
            "inside5_component_weight": 1.0,
            "outside5_residual_weight": 1.0,
            "weights_tuned": False,
        },
        "training": training_meta,
        "thresholds_unchanged": threshold,
        "validation_2024": {
            "existing_component": valid_existing,
            "structural_candidate": valid_structural,
            "failed_gates": valid_failed,
            "passed_diagnostic_gates": not valid_failed,
        },
        "test_2025_reporting_only": {
            "existing_component": test_existing,
            "structural_candidate": test_structural,
            "failed_gates": test_failed,
            "passed_diagnostic_gates": not test_failed,
            "used_for_selection": False,
        },
        "outputs": {"predictions_csv": str(OUT_CSV), "summary_json": str(OUT_JSON)},
    }
    OUT_JSON.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("ISSUE 56 RUSHING TDS OUTSIDE-5 RESIDUAL DIAGNOSTIC V2")
    print(
        f"selection_train_seasons={training_meta['selection_train_seasons']} "
        f"validation_season=2024 final_train_seasons={training_meta['final_train_seasons']} "
        f"test_season=2025 test_used_for_selection=false"
    )
    print(
        f"best_iteration_selected_on_2024={training_meta['best_iteration_selected_on_2024']} "
        f"selection_train_rows={training_meta['selection_train_rows']} "
        f"validation_rows={training_meta['validation_2024_rows']}"
    )
    print("STRUCTURE: full_projection = existing_inside5_component + outside5_residual")
    print("weights: inside5=1.0 outside5=1.0 tuned=false")
    print(
        "2024 EXISTING: "
        f"mae={valid_existing['mae']:.6f} bias={valid_existing['bias']:.6f} "
        f"improvement={valid_existing['improvement_vs_baseline_pct']:.6f}% "
        f"poisson={valid_existing['poisson_deviance']:.6f} "
        f"brier={valid_existing['brier_1plus_independent_poisson']:.6f}"
    )
    print(
        "2024 STRUCTURAL: "
        f"mae={valid_structural['mae']:.6f} bias={valid_structural['bias']:.6f} "
        f"improvement={valid_structural['improvement_vs_baseline_pct']:.6f}% "
        f"poisson={valid_structural['poisson_deviance']:.6f} "
        f"brier={valid_structural['brier_1plus_independent_poisson']:.6f} "
        f"failed={';'.join(valid_failed) if valid_failed else 'none'}"
    )
    print(
        "2025 REPORTING ONLY: "
        f"mae={test_structural['mae']:.6f} bias={test_structural['bias']:.6f} "
        f"improvement={test_structural['improvement_vs_baseline_pct']:.6f}% "
        f"poisson={test_structural['poisson_deviance']:.6f} "
        f"brier={test_structural['brier_1plus_independent_poisson']:.6f} "
        f"failed={';'.join(test_failed) if test_failed else 'none'}"
    )
    print(f"csv={OUT_CSV}")
    print(f"json={OUT_JSON}")
    print("ISSUE 56 RUSHING TDS OUTSIDE-5 RESIDUAL DIAGNOSTIC V2: COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
