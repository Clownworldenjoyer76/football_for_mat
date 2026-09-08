#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

PROP = Path(__file__).resolve().parent
TRAIN_DIR = PROP / "scripts/train"
SCRIPTS_DIR = PROP / "scripts"
for p in [TRAIN_DIR, SCRIPTS_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import train_direct_models as direct

CONFIG = PROP / "config/prop_engine.yaml"
FEATURE_CONFIG = PROP / "config/features/rushing_tds.json"
THRESHOLDS = PROP / "config/acceptance_thresholds.yaml"
SELECTED = PROP / "models/rushing_tds/selected_model.json"
AUDIT = PROP / "evaluation/model_selection_predictions.parquet"

OUT_JSON = PROP / "evaluation/issue56_rushing_tds_hurdle_diagnostic.json"
OUT_CSV = PROP / "evaluation/issue56_rushing_tds_hurdle_predictions.csv"

TARGET = "rushing_tds"
SELECTION_TRAIN_END = 2023
VALIDATION_SEASON = 2024
FINAL_TRAIN_END = 2024
TEST_SEASON = 2025
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


def binary_params() -> dict:
    # Inherit the deterministic Issue 24 direct-model parameter contract,
    # changing only the statistical objective/metric required by the hurdle gate.
    params = dict(direct.params_for("poisson"))
    params["objective"] = "binary"
    params["metric"] = "binary_logloss"
    params.pop("poisson_max_delta_step", None)
    return params


def preflight(config: dict, manifest: dict) -> dict:
    for p in [CONFIG, FEATURE_CONFIG, THRESHOLDS, SELECTED, AUDIT]:
        if not p.is_file():
            raise FileNotFoundError(p)

    # Exact chronology contract.
    expected_cutoffs = {
        "MODEL_SELECTION_TRAIN_END": SELECTION_TRAIN_END,
        "DEVELOPMENT_VALIDATION_SEASON": VALIDATION_SEASON,
        "FINAL_TRAIN_END": FINAL_TRAIN_END,
        "UNTOUCHED_TEST_SEASON": TEST_SEASON,
    }
    for name, expected in expected_cutoffs.items():
        actual = getattr(direct, name, None)
        if actual != expected:
            raise AssertionError(
                f"train_direct_models.{name}={actual!r}; expected {expected}"
            )

    selected = load_json(SELECTED)
    if selected.get("selected_architecture") != "component":
        raise AssertionError(
            f"Expected current rushing_tds architecture=component; "
            f"found {selected.get('selected_architecture')}"
        )
    if selected.get("validation_season") != VALIDATION_SEASON:
        raise AssertionError("selected_model validation season mismatch")
    if selected.get("test_season") != TEST_SEASON:
        raise AssertionError("selected_model test season mismatch")
    if selected.get("test_used_for_selection") is not False:
        raise AssertionError("selected_model test_used_for_selection must be false")
    if selected.get("market_features_used") is not False:
        raise AssertionError("selected_model market_features_used must be false")

    if manifest.get("target") != TARGET:
        raise AssertionError("rushing_tds feature config target mismatch")
    numeric_features = list(manifest["numeric_features"])
    categorical_features = list(manifest["categorical_features"])
    all_features = [*numeric_features, *categorical_features]
    if len(all_features) != len(set(all_features)):
        raise AssertionError("Duplicate rushing_tds model features")

    forbidden = [str(x).lower() for x in config.get("forbidden_features", [])]
    bad = [
        feature
        for feature in all_features
        if any(token in feature.lower() for token in forbidden)
    ]
    if bad:
        raise AssertionError(f"Forbidden/market features in hurdle schema: {bad}")

    hist = repo_root() / config["paths"]["historical_features"]
    if not hist.is_file():
        raise FileNotFoundError(hist)
    schema_names = set(pq.ParquetFile(hist).schema.names)
    required = {
        *GRAIN,
        "position",
        "target_rushing_tds",
        *all_features,
    }
    missing = sorted(required - schema_names)
    if missing:
        raise AssertionError(f"Historical feature table missing columns: {missing}")

    threshold = dict(load_yaml(THRESHOLDS)[TARGET])

    audit = pd.read_parquet(
        AUDIT,
        columns=["split", "season", "target", "actual", "baseline_projection"],
    )
    rt = audit.loc[audit["target"].astype(str).eq(TARGET)].copy()
    counts = {}
    for split, season in [("validation", VALIDATION_SEASON), ("test", TEST_SEASON)]:
        f = rt.loc[rt["split"].astype(str).eq(split)].copy()
        if f.empty:
            raise AssertionError(f"No {TARGET} audit rows for {split}")
        found = set(pd.to_numeric(f["season"], errors="raise").astype(int).unique())
        if found != {season}:
            raise AssertionError(f"{split}: unexpected seasons {sorted(found)}")
        counts[split] = int(len(f))

    params = binary_params()
    if params.get("objective") != "binary":
        raise AssertionError("Hurdle objective is not binary")
    if params.get("metric") != "binary_logloss":
        raise AssertionError("Hurdle metric is not binary_logloss")
    if params.get("deterministic") is not True:
        raise AssertionError("Hurdle LightGBM params are not deterministic")
    if int(params.get("seed", -1)) != int(direct.SEED):
        raise AssertionError("Hurdle seed does not match direct trainer seed")

    info = {
        "selection_train_end": SELECTION_TRAIN_END,
        "validation_season": VALIDATION_SEASON,
        "final_train_end": FINAL_TRAIN_END,
        "test_season": TEST_SEASON,
        "test_used_for_selection": False,
        "market_features_used": False,
        "feature_count": len(all_features),
        "audit_rows": counts,
        "binary_params": params,
        "thresholds": threshold,
    }
    print("PREFLIGHT PASS: hurdle chronology, schema, market exclusion and deterministic params")
    print(json.dumps(info, sort_keys=True))
    return info


def metrics(
    actual: np.ndarray,
    expected_count: np.ndarray,
    baseline: np.ndarray,
    p1: np.ndarray,
) -> dict:
    y = np.asarray(actual, dtype="float64")
    lam = np.maximum(np.asarray(expected_count, dtype="float64"), 0.0)
    b = np.asarray(baseline, dtype="float64")
    q = np.clip(np.asarray(p1, dtype="float64"), 0.0, 1.0)

    ok = np.isfinite(y) & np.isfinite(lam) & np.isfinite(b) & np.isfinite(q)
    if not ok.all():
        raise AssertionError(f"Nonfinite metric rows: {int((~ok).sum())}")

    mae = float(np.mean(np.abs(y - lam)))
    bias = float(np.mean(lam - y))
    baseline_mae = float(np.mean(np.abs(y - b)))
    improvement = 100.0 * (baseline_mae - mae) / baseline_mae

    safe_lam = np.maximum(lam, 1e-12)
    terms = np.empty_like(y)
    zero = y <= 0.0
    terms[zero] = safe_lam[zero]
    positive = ~zero
    terms[positive] = (
        y[positive] * np.log(y[positive] / safe_lam[positive])
        - (y[positive] - safe_lam[positive])
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
        "improvement_vs_baseline_pct": improvement,
        "brier_1plus": brier,
        "poisson_deviance": poisson,
        "mean_actual": float(np.mean(y)),
        "mean_expected_count": float(np.mean(lam)),
        "mean_p1": float(np.mean(q)),
        "event_rate": float(np.mean(event)),
    }


def failures(m: dict, threshold: dict) -> list[str]:
    failed = []
    if m["mae"] > float(threshold["maximum_validation_mae"]):
        failed.append("mae")
    if m["abs_bias"] > float(threshold["maximum_allowed_bias"]):
        failed.append("bias")
    if m["improvement_vs_baseline_pct"] < float(
        threshold["minimum_improvement_vs_baseline_pct"]
    ):
        failed.append("improvement")
    if m["brier_1plus"] > float(threshold["maximum_brier_1plus"]):
        failed.append("brier_1plus")
    if m["poisson_deviance"] > float(threshold["maximum_poisson_deviance"]):
        failed.append("poisson_deviance")
    return failed


def prepare_frame(
    features: pd.DataFrame,
    manifest: dict,
) -> pd.DataFrame:
    eligible_positions = {
        str(x).strip().upper()
        for x in manifest["eligible_positions"]
    }
    frame = features.copy()
    frame["season"] = pd.to_numeric(frame["season"], errors="raise").astype(int)
    frame["week"] = pd.to_numeric(frame["week"], errors="raise").astype(int)
    frame["game_id"] = frame["game_id"].astype(str).str.strip()
    frame["player_id"] = clean_id(frame["player_id"])
    frame["position"] = (
        frame["position"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )
    frame = frame.loc[frame["position"].isin(eligible_positions)].copy()

    target = pd.to_numeric(frame["target_rushing_tds"], errors="coerce")
    frame = frame.loc[target.notna()].copy()
    frame["_count_target"] = target.loc[frame.index].astype(float)
    if (frame["_count_target"] < 0).any():
        raise AssertionError("Negative rushing_tds target")
    frame["_event_target"] = frame["_count_target"].ge(1.0).astype(float)
    return frame


def fit_binary_hurdle(
    frame: pd.DataFrame,
    manifest: dict,
):
    numeric_features = list(manifest["numeric_features"])
    categorical_features = list(manifest["categorical_features"])
    feature_names = [*numeric_features, *categorical_features]

    train = frame.loc[frame["season"].le(SELECTION_TRAIN_END)].copy()
    valid = frame.loc[frame["season"].eq(VALIDATION_SEASON)].copy()
    final_train = frame.loc[frame["season"].le(FINAL_TRAIN_END)].copy()
    test = frame.loc[frame["season"].eq(TEST_SEASON)].copy()

    if train.empty or valid.empty or final_train.empty or test.empty:
        raise AssertionError(
            f"Empty split: train={len(train)} valid={len(valid)} "
            f"final_train={len(final_train)} test={len(test)}"
        )

    # Ensure the model sees both classes in every fit/evaluation split.
    for label, f in [
        ("selection_train", train),
        ("validation_2024", valid),
        ("final_train", final_train),
        ("test_2025", test),
    ]:
        classes = set(f["_event_target"].astype(int).unique().tolist())
        if classes != {0, 1}:
            raise AssertionError(f"{label}: binary target classes={sorted(classes)}")

    params = binary_params()

    selection_levels = direct.categorical_levels(train, categorical_features)
    X_train = direct.model_matrix(
        train, numeric_features, categorical_features, selection_levels
    )
    X_valid = direct.model_matrix(
        valid, numeric_features, categorical_features, selection_levels
    )
    y_train = train["_event_target"].astype(float)
    y_valid = valid["_event_target"].astype(float)

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
        raise AssertionError("Invalid binary best_iteration")

    valid_p1 = np.clip(
        selected.predict(X_valid, num_iteration=best_iteration),
        0.0,
        1.0,
    )

    final_levels = direct.categorical_levels(final_train, categorical_features)
    X_final = direct.model_matrix(
        final_train, numeric_features, categorical_features, final_levels
    )
    y_final = final_train["_event_target"].astype(float)
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
        test, numeric_features, categorical_features, final_levels
    )
    test_p1 = np.clip(
        final_model.predict(X_test, num_iteration=best_iteration),
        0.0,
        1.0,
    )

    return train, valid, final_train, test, valid_p1, test_p1, best_iteration, params


def positive_severity(frame: pd.DataFrame) -> float:
    positive = frame.loc[frame["_count_target"].gt(0.0), "_count_target"]
    if positive.empty:
        raise AssertionError("No positive rushing TD rows for severity estimate")
    value = float(positive.mean())
    if not np.isfinite(value) or value < 1.0:
        raise AssertionError(f"Invalid positive severity={value}")
    return value


def align_to_audit(
    scoring_frame: pd.DataFrame,
    p1: np.ndarray,
    audit: pd.DataFrame,
    split: str,
    season: int,
) -> pd.DataFrame:
    pred = scoring_frame[GRAIN + ["_count_target"]].copy()
    pred["hurdle_p1"] = np.asarray(p1, dtype=float)

    a = audit.loc[
        audit["target"].astype(str).eq(TARGET)
        & audit["split"].astype(str).eq(split)
    ].copy()
    found = set(pd.to_numeric(a["season"], errors="raise").astype(int).unique())
    if found != {season}:
        raise AssertionError(f"{split}: unexpected audit seasons {sorted(found)}")

    merged = a.merge(
        pred,
        on=GRAIN,
        how="left",
        validate="one_to_one",
    )
    if merged["hurdle_p1"].isna().any():
        n = int(merged["hurdle_p1"].isna().sum())
        raise AssertionError(f"{split}: missing hurdle predictions on {n} audit rows")

    actual = pd.to_numeric(merged["actual"], errors="raise").astype(float)
    target = pd.to_numeric(merged["_count_target"], errors="raise").astype(float)
    mismatch = ~np.isclose(
        actual.to_numpy(),
        target.to_numpy(),
        rtol=0.0,
        atol=0.0,
    )
    if mismatch.any():
        raise AssertionError(
            f"{split}: canonical target mismatch on {int(mismatch.sum())} rows"
        )

    merged["actual"] = actual
    merged["baseline_projection"] = pd.to_numeric(
        merged["baseline_projection"], errors="raise"
    ).astype(float)
    return merged


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    config = load_yaml(CONFIG)
    manifest = load_json(FEATURE_CONFIG)
    preflight_info = preflight(config, manifest)
    if args.preflight_only:
        print("ISSUE 56 RUSHING TDS HURDLE PREFLIGHT: PASS")
        return 0

    threshold = dict(load_yaml(THRESHOLDS)[TARGET])
    hist_path = repo_root() / config["paths"]["historical_features"]

    numeric_features = list(manifest["numeric_features"])
    categorical_features = list(manifest["categorical_features"])
    columns = list(dict.fromkeys([
        *GRAIN,
        "position",
        "target_rushing_tds",
        *numeric_features,
        *categorical_features,
    ]))
    features = pd.read_parquet(hist_path, columns=columns)
    frame = prepare_frame(features, manifest)

    (
        train,
        valid,
        final_train,
        test,
        valid_p1,
        test_p1,
        best_iteration,
        params,
    ) = fit_binary_hurdle(frame, manifest)

    # Severity is estimated strictly from data available before the scoring season.
    selection_global_severity = positive_severity(train)
    final_global_severity = positive_severity(final_train)

    audit = pd.read_parquet(AUDIT)
    valid_aligned = align_to_audit(
        valid, valid_p1, audit, "validation", VALIDATION_SEASON
    )
    test_aligned = align_to_audit(
        test, test_p1, audit, "test", TEST_SEASON
    )

    candidates = [
        {
            "name": "unit_positive_severity",
            "selection_severity": 1.0,
            "final_severity": 1.0,
        },
        {
            "name": "training_positive_mean_severity",
            "selection_severity": selection_global_severity,
            "final_severity": final_global_severity,
        },
    ]

    results = []
    for candidate in candidates:
        v_expected = (
            valid_aligned["hurdle_p1"].to_numpy(dtype=float)
            * float(candidate["selection_severity"])
        )
        t_expected = (
            test_aligned["hurdle_p1"].to_numpy(dtype=float)
            * float(candidate["final_severity"])
        )

        vm = metrics(
            valid_aligned["actual"].to_numpy(dtype=float),
            v_expected,
            valid_aligned["baseline_projection"].to_numpy(dtype=float),
            valid_aligned["hurdle_p1"].to_numpy(dtype=float),
        )
        tm = metrics(
            test_aligned["actual"].to_numpy(dtype=float),
            t_expected,
            test_aligned["baseline_projection"].to_numpy(dtype=float),
            test_aligned["hurdle_p1"].to_numpy(dtype=float),
        )
        vf = failures(vm, threshold)
        tf = failures(tm, threshold)

        results.append({
            **candidate,
            "validation_metrics": vm,
            "validation_failed": vf,
            "test_metrics_reporting_only": tm,
            "test_failed_reporting_only": tf,
        })

    passing = [r for r in results if not r["validation_failed"]]
    if passing:
        chosen = sorted(
            passing,
            key=lambda r: (
                r["validation_metrics"]["mae"],
                r["validation_metrics"]["poisson_deviance"],
                r["name"],
            ),
        )[0]
        selection_status = "validation_gate_pass_candidate_found"
    else:
        chosen = sorted(
            results,
            key=lambda r: (
                len(r["validation_failed"]),
                r["validation_metrics"]["mae"],
                r["validation_metrics"]["abs_bias"],
                r["validation_metrics"]["poisson_deviance"],
                r["name"],
            ),
        )[0]
        selection_status = "no_candidate_passed_2024_gates"

    # Save both candidate prediction paths for auditability.
    output = valid_aligned[GRAIN + ["actual", "baseline_projection", "hurdle_p1"]].copy()
    output["split_label"] = "validation_2024"
    output["unit_positive_severity_expected_count"] = output["hurdle_p1"]
    output["training_positive_mean_severity_expected_count"] = (
        output["hurdle_p1"] * selection_global_severity
    )

    test_output = test_aligned[GRAIN + ["actual", "baseline_projection", "hurdle_p1"]].copy()
    test_output["split_label"] = "test_2025_reporting_only"
    test_output["unit_positive_severity_expected_count"] = test_output["hurdle_p1"]
    test_output["training_positive_mean_severity_expected_count"] = (
        test_output["hurdle_p1"] * final_global_severity
    )

    pd.concat([output, test_output], ignore_index=True).to_csv(
        OUT_CSV, index=False
    )

    payload = {
        "status": "complete",
        "target": TARGET,
        "diagnostic_only": True,
        "production_files_modified": False,
        "market_data_used": False,
        "architecture": "binary_hurdle_probability_x_positive_count_severity",
        "binary_target": "target_rushing_tds >= 1",
        "training_policy": {
            "selection_train_end": SELECTION_TRAIN_END,
            "validation_season": VALIDATION_SEASON,
            "best_iteration_selected_on_2024": best_iteration,
            "final_train_end": FINAL_TRAIN_END,
            "test_season": TEST_SEASON,
            "test_used_for_selection": False,
            "binary_params": params,
            "selection_training_rows": int(len(train)),
            "selection_training_positive_rows": int(train["_event_target"].sum()),
            "validation_rows": int(len(valid)),
            "validation_positive_rows": int(valid["_event_target"].sum()),
            "final_training_rows": int(len(final_train)),
            "final_training_positive_rows": int(final_train["_event_target"].sum()),
            "test_rows": int(len(test)),
            "test_positive_rows": int(test["_event_target"].sum()),
        },
        "severity_policy": {
            "candidate_names": [r["name"] for r in results],
            "selection_global_positive_mean_through_2023": selection_global_severity,
            "final_global_positive_mean_through_2024": final_global_severity,
            "2025_outcomes_used_for_severity": False,
        },
        "thresholds_unchanged": threshold,
        "preflight": preflight_info,
        "selection_status": selection_status,
        "chosen_from_2024_only": chosen,
        "all_candidates": results,
        "outputs": {
            "csv": str(OUT_CSV),
            "json": str(OUT_JSON),
        },
    }
    OUT_JSON.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("ISSUE 56 RUSHING TDS HURDLE DIAGNOSTIC")
    print(
        f"selection_train_rows={len(train)} positives={int(train['_event_target'].sum())} "
        f"validation_rows={len(valid)} positives={int(valid['_event_target'].sum())}"
    )
    print(
        f"best_iteration_selected_on_2024={best_iteration} "
        f"selection_positive_severity={selection_global_severity:.6f} "
        f"final_positive_severity={final_global_severity:.6f}"
    )
    print(f"selection_status={selection_status}")
    for result in results:
        vm = result["validation_metrics"]
        tm = result["test_metrics_reporting_only"]
        print(
            f"{result['name']} 2024: "
            f"mae={vm['mae']:.6f} bias={vm['bias']:.6f} "
            f"improvement={vm['improvement_vs_baseline_pct']:.6f}% "
            f"brier={vm['brier_1plus']:.6f} poisson={vm['poisson_deviance']:.6f} "
            f"failed={';'.join(result['validation_failed']) if result['validation_failed'] else 'none'}"
        )
        print(
            f"{result['name']} 2025_REPORTING_ONLY: "
            f"mae={tm['mae']:.6f} bias={tm['bias']:.6f} "
            f"improvement={tm['improvement_vs_baseline_pct']:.6f}% "
            f"brier={tm['brier_1plus']:.6f} poisson={tm['poisson_deviance']:.6f} "
            f"failed={';'.join(result['test_failed_reporting_only']) if result['test_failed_reporting_only'] else 'none'}"
        )

    vm = chosen["validation_metrics"]
    tm = chosen["test_metrics_reporting_only"]
    print(
        f"CHOSEN_2024_ONLY={chosen['name']} "
        f"2024_mae={vm['mae']:.6f} 2024_bias={vm['bias']:.6f} "
        f"2024_improvement={vm['improvement_vs_baseline_pct']:.6f}%"
    )
    print(
        f"CHOSEN_2025_REPORTING_ONLY "
        f"mae={tm['mae']:.6f} bias={tm['bias']:.6f} "
        f"improvement={tm['improvement_vs_baseline_pct']:.6f}% "
        f"brier={tm['brier_1plus']:.6f} poisson={tm['poisson_deviance']:.6f} "
        f"failed={';'.join(chosen['test_failed_reporting_only']) if chosen['test_failed_reporting_only'] else 'none'}"
    )
    print(f"csv={OUT_CSV}")
    print(f"json={OUT_JSON}")
    print("ISSUE 56 RUSHING TDS HURDLE DIAGNOSTIC: COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
