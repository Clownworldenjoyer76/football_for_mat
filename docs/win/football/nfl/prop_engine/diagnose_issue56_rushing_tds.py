#!/usr/bin/env python3
from __future__ import annotations

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
CONFIG = PROP / "config/prop_engine.yaml"
OUT_CSV = PROP / "evaluation/issue56_rushing_tds_point_diagnostic.csv"
OUT_JSON = PROP / "evaluation/issue56_rushing_tds_point_diagnostic.json"

TARGET = "rushing_tds"
VALIDATION_SEASON = 2024
TEST_SEASON = 2025

REQ_COLUMNS = [
    "split", "fold_id", "season", "week", "game_id", "player_id", "target",
    "actual", "baseline_projection", "direct_projection",
    "component_projection", "blend_projection", "blend_direct_weight",
    "blend_component_weight",
]


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
        raise AssertionError("Invalid expected-count calibration mapping")
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
    y = np.asarray(actual, dtype="float64")
    p = np.maximum(np.asarray(pred, dtype="float64"), 0.0)
    b = np.asarray(baseline, dtype="float64")
    ok = np.isfinite(y) & np.isfinite(p) & np.isfinite(b)
    if not bool(ok.all()):
        raise AssertionError(f"Nonfinite metric input rows: {int((~ok).sum())}")

    mae = float(np.mean(np.abs(y - p)))
    bias = float(np.mean(p - y))
    baseline_mae = float(np.mean(np.abs(y - b)))
    improvement = 100.0 * (baseline_mae - mae) / baseline_mae if baseline_mae > 0 else np.nan

    lam = np.maximum(p, 1e-12)
    terms = np.empty_like(y, dtype="float64")
    zero = y <= 0.0
    terms[zero] = lam[zero]
    positive = ~zero
    terms[positive] = (
        y[positive] * np.log(y[positive] / lam[positive])
        - (y[positive] - lam[positive])
    )
    poisson = float(2.0 * np.mean(terms))

    return {
        "rows": int(len(y)),
        "mae": mae,
        "bias": bias,
        "abs_bias": abs(bias),
        "baseline_mae": baseline_mae,
        "improvement_vs_baseline_pct": float(improvement),
        "poisson_deviance": poisson,
        "mean_actual": float(np.mean(y)),
        "mean_prediction": float(np.mean(p)),
    }


def failed_gates(m: dict, threshold: dict) -> list[str]:
    failed = []
    if m["mae"] > float(threshold["maximum_validation_mae"]):
        failed.append("mae")
    if m["abs_bias"] > float(threshold["maximum_allowed_bias"]):
        failed.append("bias")
    if m["improvement_vs_baseline_pct"] < float(threshold["minimum_improvement_vs_baseline_pct"]):
        failed.append("improvement")
    if m["poisson_deviance"] > float(threshold["maximum_poisson_deviance"]):
        failed.append("poisson_deviance")
    return failed


def split_frame(audit: pd.DataFrame, split: str, season: int) -> pd.DataFrame:
    f = audit.loc[
        audit["target"].astype(str).eq(TARGET)
        & audit["split"].astype(str).eq(split)
    ].copy()
    if f.empty:
        raise AssertionError(f"No {TARGET} rows for split={split}")
    found = set(pd.to_numeric(f["season"], errors="raise").astype(int).unique())
    if found != {season}:
        raise AssertionError(f"{split}: expected season {season}, found {sorted(found)}")
    for col in ["actual", "baseline_projection", "direct_projection", "component_projection"]:
        f[col] = pd.to_numeric(f[col], errors="coerce").astype("float64")
        if f[col].isna().any() or not np.isfinite(f[col].to_numpy()).all():
            raise AssertionError(f"{split}: invalid {col}")
    return f.reset_index(drop=True)


def candidate_set(validation: pd.DataFrame, test: pd.DataFrame, calibration: dict) -> list[dict]:
    vd = validation["direct_projection"].to_numpy(dtype="float64")
    vc = validation["component_projection"].to_numpy(dtype="float64")
    td = test["direct_projection"].to_numpy(dtype="float64")
    tc = test["component_projection"].to_numpy(dtype="float64")

    mapping = (
        calibration.get("count_calibration", {})
        .get("expected_count", {})
        .get("mapping")
    )
    if not isinstance(mapping, dict):
        raise AssertionError("rushing_tds expected-count calibration mapping missing")

    vcal = apply_mapping(np.maximum(vc, 0.0), mapping)
    tcal = apply_mapping(np.maximum(tc, 0.0), mapping)

    out = []

    def add(name: str, family: str, params: dict, vp: np.ndarray, tp: np.ndarray):
        out.append({
            "name": name,
            "family": family,
            "params": params,
            "validation_prediction": np.maximum(np.asarray(vp, dtype="float64"), 0.0),
            "test_prediction": np.maximum(np.asarray(tp, dtype="float64"), 0.0),
        })

    add("identity_component", "identity_component", {}, vc, tc)
    add("identity_direct", "identity_direct", {}, vd, td)
    add("existing_expected_count_calibration", "existing_calibration", {}, vcal, tcal)

    # Architecture family already supported by Issue 25; fine grid is diagnostic only.
    for i in range(0, 101):
        w = i / 100.0
        add(
            f"direct_component_w{w:.2f}",
            "direct_component_blend",
            {"direct_weight": w, "component_weight": 1.0 - w},
            w * vd + (1.0 - w) * vc,
            w * td + (1.0 - w) * tc,
        )

    # Raw component versus existing 2024 expected-count calibration.
    for i in range(0, 101):
        w = i / 100.0
        add(
            f"component_calibrated_w{w:.2f}",
            "component_calibration_blend",
            {"raw_component_weight": w, "calibrated_weight": 1.0 - w},
            w * vc + (1.0 - w) * vcal,
            w * tc + (1.0 - w) * tcal,
        )

    # Multiplicative scale and additive correction selected only on 2024.
    for i in range(500, 2001, 5):
        scale = i / 1000.0
        add(
            f"component_scale_{scale:.3f}",
            "multiplicative_scale",
            {"scale": scale},
            scale * vc,
            scale * tc,
        )

    for i in range(-100, 201, 2):
        offset = i / 1000.0
        add(
            f"component_offset_{offset:+.3f}",
            "additive_offset",
            {"offset": offset},
            vc + offset,
            tc + offset,
        )

    return out


def repo_root() -> Path:
    current = PROP.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists():
            return candidate
    raise AssertionError("Could not locate repository root from Prop Engine path")


def exact_rushing_td_decomposition(config: dict, seasons: list[int]) -> pd.DataFrame:
    root = repo_root()
    pattern = config["paths"]["pbp_pattern"]
    frames = []

    for season in seasons:
        path = root / pattern.format(season=season)
        if not path.is_file():
            raise FileNotFoundError(path)
        pbp = pd.read_csv(
            path,
            usecols=[
                "season_type", "week", "game_id", "yardline_100",
                "rush_attempt", "rusher_player_id", "rush_touchdown",
            ],
            low_memory=False,
        )
        pbp = pbp.loc[pbp["season_type"].astype(str).str.upper().eq("REG")].copy()
        pbp["season"] = season
        pbp["week"] = pd.to_numeric(pbp["week"], errors="raise").astype(int)
        pbp["game_id"] = pbp["game_id"].astype(str).str.strip()
        pbp["yardline_100"] = pd.to_numeric(pbp["yardline_100"], errors="coerce")
        pbp["rush_attempt"] = pd.to_numeric(pbp["rush_attempt"], errors="coerce").fillna(0.0)
        pbp["rush_touchdown"] = pd.to_numeric(pbp["rush_touchdown"], errors="coerce").fillna(0.0)
        pbp["player_id"] = (
            pbp["rusher_player_id"].fillna("").astype(str).str.strip()
        )

        rush = pbp.loc[
            pbp["rush_attempt"].eq(1.0)
            & pbp["player_id"].ne("")
        ].copy()

        rush["_td"] = rush["rush_touchdown"].eq(1.0).astype(float)
        rush["_goal_line_attempt"] = (
            rush["yardline_100"].notna()
            & rush["yardline_100"].le(5.0)
        ).astype(float)
        rush["_goal_line_td"] = (
            rush["_td"].eq(1.0)
            & rush["_goal_line_attempt"].eq(1.0)
        ).astype(float)
        rush["_outside5_td"] = (
            rush["_td"].eq(1.0)
            & rush["_goal_line_attempt"].eq(0.0)
        ).astype(float)

        agg = rush.groupby(
            ["season", "week", "game_id", "player_id"],
            as_index=False,
            dropna=False,
        ).agg(
            pbp_rush_attempts=("_td", "size"),
            pbp_all_rushing_tds=("_td", "sum"),
            pbp_goal_line_rush_attempts=("_goal_line_attempt", "sum"),
            pbp_goal_line_rushing_tds=("_goal_line_td", "sum"),
            pbp_outside5_rushing_tds=("_outside5_td", "sum"),
        )
        frames.append(agg)

    return pd.concat(frames, ignore_index=True)


def decomposition_summary(frame: pd.DataFrame, season: int) -> dict:
    f = frame.loc[frame["season"].eq(season)].copy()
    total = float(f["pbp_all_rushing_tds"].sum())
    gl = float(f["pbp_goal_line_rushing_tds"].sum())
    outside = float(f["pbp_outside5_rushing_tds"].sum())
    return {
        "season": season,
        "rows": int(len(f)),
        "all_rushing_tds": total,
        "goal_line_rushing_tds": gl,
        "outside_5_rushing_tds": outside,
        "goal_line_share_of_rushing_tds": gl / total if total > 0 else None,
        "outside_5_share_of_rushing_tds": outside / total if total > 0 else None,
    }


def main() -> int:
    for path in [AUDIT, THRESHOLDS, CALIBRATION, SELECTED, CONFIG]:
        if not path.is_file():
            raise FileNotFoundError(path)

    selected = load_json(SELECTED)
    if selected.get("selected_architecture") != "component":
        raise AssertionError(
            f"Expected rushing_tds selected architecture=component; found "
            f"{selected.get('selected_architecture')}"
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
        raise AssertionError("calibration source is not 2024")
    if reporting.get("test_rows_used_for_calibration") is not False:
        raise AssertionError("calibration used test rows")
    if calibration.get("market_features_used") is not False:
        raise AssertionError("calibration market_features_used must be false")

    thresholds = load_yaml(THRESHOLDS)
    threshold = dict(thresholds[TARGET])
    config = load_yaml(CONFIG)

    audit = pd.read_parquet(AUDIT, columns=REQ_COLUMNS)
    validation = split_frame(audit, "validation", VALIDATION_SEASON)
    test = split_frame(audit, "test", TEST_SEASON)

    candidates = candidate_set(validation, test, calibration)
    vy = validation["actual"].to_numpy(dtype="float64")
    vb = validation["baseline_projection"].to_numpy(dtype="float64")
    ty = test["actual"].to_numpy(dtype="float64")
    tb = test["baseline_projection"].to_numpy(dtype="float64")

    rows = []
    payload_candidates = []
    for c in candidates:
        vm = metrics(vy, c["validation_prediction"], vb)
        vf = failed_gates(vm, threshold)
        tm = metrics(ty, c["test_prediction"], tb)
        tf = failed_gates(tm, threshold)
        rows.append({
            "name": c["name"],
            "family": c["family"],
            "params_json": json.dumps(c["params"], sort_keys=True),
            "validation_mae": vm["mae"],
            "validation_bias": vm["bias"],
            "validation_improvement_pct": vm["improvement_vs_baseline_pct"],
            "validation_poisson_deviance": vm["poisson_deviance"],
            "validation_failed": ";".join(vf),
            "validation_pass": not vf,
            "test_mae_reporting_only": tm["mae"],
            "test_bias_reporting_only": tm["bias"],
            "test_improvement_pct_reporting_only": tm["improvement_vs_baseline_pct"],
            "test_poisson_deviance_reporting_only": tm["poisson_deviance"],
            "test_failed_reporting_only": ";".join(tf),
            "test_pass_reporting_only": not tf,
        })
        payload_candidates.append({
            "name": c["name"],
            "family": c["family"],
            "params": c["params"],
            "validation_metrics": vm,
            "validation_failed": vf,
            "test_metrics_reporting_only": tm,
            "test_failed_reporting_only": tf,
        })

    table = pd.DataFrame(rows)
    passing = table.loc[table["validation_pass"]].copy()
    if not passing.empty:
        chosen_row = passing.sort_values(
            ["validation_mae", "name"], kind="mergesort"
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
    chosen = next(c for c in payload_candidates if c["name"] == chosen_name)

    table = table.sort_values(
        ["validation_pass", "validation_mae", "name"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    table.to_csv(OUT_CSV, index=False)

    pbp = exact_rushing_td_decomposition(config, [VALIDATION_SEASON, TEST_SEASON])

    # Join PBP decomposition to the exact model-evaluation cohorts.
    cohort_frames = []
    for split, f, season in [
        ("validation", validation, VALIDATION_SEASON),
        ("test", test, TEST_SEASON),
    ]:
        joined = f[
            ["season", "week", "game_id", "player_id", "actual", "component_projection"]
        ].merge(
            pbp,
            on=["season", "week", "game_id", "player_id"],
            how="left",
            validate="one_to_one",
        )
        for col in [
            "pbp_rush_attempts", "pbp_all_rushing_tds",
            "pbp_goal_line_rush_attempts", "pbp_goal_line_rushing_tds",
            "pbp_outside5_rushing_tds",
        ]:
            joined[col] = pd.to_numeric(joined[col], errors="coerce").fillna(0.0)

        target_mismatch = ~np.isclose(
            joined["actual"].to_numpy(dtype="float64"),
            joined["pbp_all_rushing_tds"].to_numpy(dtype="float64"),
            rtol=0.0,
            atol=1e-12,
        )
        joined["split"] = split
        joined["target_pbp_mismatch"] = target_mismatch
        cohort_frames.append(joined)

    cohort = pd.concat(cohort_frames, ignore_index=True)

    decomposition = {}
    for split, season in [("validation", VALIDATION_SEASON), ("test", TEST_SEASON)]:
        f = cohort.loc[cohort["split"].eq(split)].copy()
        total = float(f["pbp_all_rushing_tds"].sum())
        gl = float(f["pbp_goal_line_rushing_tds"].sum())
        outside = float(f["pbp_outside5_rushing_tds"].sum())
        component = f["component_projection"].to_numpy(dtype="float64")
        gl_actual = f["pbp_goal_line_rushing_tds"].to_numpy(dtype="float64")
        full_actual = f["actual"].to_numpy(dtype="float64")
        decomposition[split] = {
            "season": season,
            "evaluation_rows": int(len(f)),
            "target_pbp_mismatch_rows": int(f["target_pbp_mismatch"].sum()),
            "all_rushing_tds": total,
            "goal_line_rushing_tds": gl,
            "outside_5_rushing_tds": outside,
            "goal_line_share_of_rushing_tds": gl / total if total > 0 else None,
            "outside_5_share_of_rushing_tds": outside / total if total > 0 else None,
            "component_mae_vs_full_target": float(np.mean(np.abs(full_actual - component))),
            "component_bias_vs_full_target": float(np.mean(component - full_actual)),
            "component_mae_vs_goal_line_td_subset": float(np.mean(np.abs(gl_actual - component))),
            "component_bias_vs_goal_line_td_subset": float(np.mean(component - gl_actual)),
        }

    result = {
        "status": "complete",
        "target": TARGET,
        "diagnostic_is_read_only": True,
        "market_data_used": False,
        "structural_contract": {
            "production_target": "all weekly rushing touchdowns",
            "current_component_efficiency_numerator": "rushing touchdowns on rush attempts from yardline_100 <= 5",
            "current_component_efficiency_exposure": "rush attempts from yardline_100 <= 5",
            "current_component_scope_gap": (
                "rushing touchdowns scored from outside the 5 are not represented "
                "by the goal-line component"
            ),
        },
        "thresholds_unchanged": threshold,
        "selection_policy": {
            "selection_season": VALIDATION_SEASON,
            "test_season": TEST_SEASON,
            "test_used_for_candidate_selection": False,
        },
        "candidate_count": int(len(table)),
        "selection_status": selection_status,
        "chosen_from_2024_only": chosen,
        "top_10_validation": table.head(10).to_dict(orient="records"),
        "td_decomposition_on_evaluation_cohorts": decomposition,
        "outputs": {
            "csv": OUT_CSV.as_posix(),
            "json": OUT_JSON.as_posix(),
        },
    }
    OUT_JSON.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("ISSUE 56 RUSHING TDS READ-ONLY DIAGNOSTIC")
    print(
        "STRUCTURAL CONTRACT: target=all_rushing_tds; "
        "component=inside_5_rushing_tds_only"
    )
    print(
        "thresholds: "
        f"mae<={float(threshold['maximum_validation_mae']):.6f} "
        f"abs_bias<={float(threshold['maximum_allowed_bias']):.6f} "
        f"improvement>={float(threshold['minimum_improvement_vs_baseline_pct']):.6f}% "
        f"poisson<={float(threshold['maximum_poisson_deviance']):.6f}"
    )
    print(f"candidates={len(table)} selection_status={selection_status}")
    print(
        f"chosen_2024_only={chosen['name']} family={chosen['family']} "
        f"params={json.dumps(chosen['params'], sort_keys=True)}"
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

    for split in ["validation", "test"]:
        d = decomposition[split]
        print(
            f"{split.upper()} TD DECOMPOSITION: "
            f"all={d['all_rushing_tds']:.0f} "
            f"inside5={d['goal_line_rushing_tds']:.0f} "
            f"outside5={d['outside_5_rushing_tds']:.0f} "
            f"outside5_share={100.0*d['outside_5_share_of_rushing_tds']:.3f}% "
            f"target_pbp_mismatch_rows={d['target_pbp_mismatch_rows']}"
        )
        print(
            f"{split.upper()} COMPONENT: "
            f"mae_vs_full={d['component_mae_vs_full_target']:.6f} "
            f"bias_vs_full={d['component_bias_vs_full_target']:.6f} "
            f"mae_vs_inside5={d['component_mae_vs_goal_line_td_subset']:.6f} "
            f"bias_vs_inside5={d['component_bias_vs_goal_line_td_subset']:.6f}"
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
    print("ISSUE 56 RUSHING TDS READ-ONLY DIAGNOSTIC: COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
