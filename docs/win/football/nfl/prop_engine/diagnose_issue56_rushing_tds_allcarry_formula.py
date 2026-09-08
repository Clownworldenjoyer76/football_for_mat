#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

PROP = Path(__file__).resolve().parent
CONFIG = PROP / "config/prop_engine.yaml"
THRESHOLDS = PROP / "config/acceptance_thresholds.yaml"
AUDIT = PROP / "evaluation/model_selection_predictions.parquet"
SELECTED = PROP / "models/rushing_tds/selected_model.json"
OUT_CSV = PROP / "evaluation/issue56_rushing_tds_allcarry_formula_diagnostic.csv"
OUT_JSON = PROP / "evaluation/issue56_rushing_tds_allcarry_formula_diagnostic.json"

TARGET = "rushing_tds"
VALIDATION_SEASON = 2024
TEST_SEASON = 2025
GRAIN = ["season", "week", "game_id", "player_id"]

TEAM_VOLUME_COLUMNS = [
    "matchup_expected_team_rush_attempts",
    "team_rush_attempts_roll3_mean",
    "team_rush_attempts_roll5_mean",
    "team_rush_attempts_ewm5",
    "team_rush_attempts_season_to_date",
]
CARRY_SHARE_COLUMNS = [
    "player_carry_share_roll3_mean",
    "player_carry_share_roll5_mean",
    "player_carry_share_ewm5",
    "player_carry_share_season_to_date",
    "player_carry_share_career_prior",
]
PLAYER_CARRY_COLUMNS = [
    "player_carries_roll3_mean",
    "player_carries_roll5_mean",
    "player_carries_ewm5",
    "player_carries_season_to_date",
    "player_carries_career_prior",
]
RATE_PAIRS = {
    "roll3": ("player_rushing_tds_roll3_mean", "player_carries_roll3_mean"),
    "roll5": ("player_rushing_tds_roll5_mean", "player_carries_roll5_mean"),
    "ewm5": ("player_rushing_tds_ewm5", "player_carries_ewm5"),
    "season_to_date": (
        "player_rushing_tds_season_to_date",
        "player_carries_season_to_date",
    ),
    "career_prior": (
        "player_rushing_tds_career_prior",
        "player_carries_career_prior",
    ),
}


def repo_root() -> Path:
    current = PROP.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists():
            return candidate
    raise AssertionError("Could not locate repository root")


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as h:
        value = yaml.safe_load(h)
    if not isinstance(value, dict):
        raise AssertionError(f"Expected YAML mapping: {path}")
    return value


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as h:
        value = json.load(h)
    if not isinstance(value, dict):
        raise AssertionError(f"Expected JSON object: {path}")
    return value


def num(series: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .astype("float64")
    )


def coalesce(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    out = pd.Series(np.nan, index=frame.index, dtype="float64")
    for c in columns:
        values = num(frame[c])
        out = out.where(out.notna(), values)
    return out


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    n = num(numerator)
    d = num(denominator)
    out = pd.Series(np.nan, index=n.index, dtype="float64")
    ok = n.notna() & d.notna() & d.gt(0.0)
    out.loc[ok] = n.loc[ok] / d.loc[ok]
    return out.clip(lower=0.0, upper=1.0)


def league_prior_by_target_season(config: dict) -> dict[int, float]:
    root = repo_root()
    path = root / config["paths"]["player_opportunity"]
    if not path.is_file():
        raise FileNotFoundError(path)
    opp = pd.read_parquet(path, columns=["season", "carries", "rushing_tds"])
    opp["season"] = pd.to_numeric(opp["season"], errors="raise").astype(int)
    opp["carries"] = num(opp["carries"]).fillna(0.0).clip(lower=0.0)
    opp["rushing_tds"] = num(opp["rushing_tds"]).fillna(0.0).clip(lower=0.0)

    result = {}
    for target_season in [VALIDATION_SEASON, TEST_SEASON]:
        prior = opp.loc[opp["season"].lt(target_season)]
        carries = float(prior["carries"].sum())
        tds = float(prior["rushing_tds"].sum())
        if carries <= 0.0:
            raise AssertionError(f"No prior carries for target season {target_season}")
        result[target_season] = tds / carries
    return result


def feature_columns() -> list[str]:
    cols = [
        *GRAIN,
        "target_rushing_tds",
        *TEAM_VOLUME_COLUMNS,
        *CARRY_SHARE_COLUMNS,
        *PLAYER_CARRY_COLUMNS,
    ]
    for td_col, carry_col in RATE_PAIRS.values():
        cols.extend([td_col, carry_col])
    return list(dict.fromkeys(cols))


def preflight(config: dict) -> dict:
    for path in [AUDIT, THRESHOLDS, SELECTED]:
        if not path.is_file():
            raise FileNotFoundError(path)

    selected = load_json(SELECTED)
    if selected.get("selected_architecture") != "component":
        raise AssertionError(
            f"Expected current rushing_tds architecture=component; found "
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

    hist = repo_root() / config["paths"]["historical_features"]
    if not hist.is_file():
        raise FileNotFoundError(hist)
    schema_names = set(pq.ParquetFile(hist).schema.names)
    missing = [c for c in feature_columns() if c not in schema_names]
    if missing:
        raise AssertionError(f"Historical features missing columns: {missing}")

    # Explicit market exclusion on every candidate input feature name.
    forbidden = [str(x).lower() for x in config.get("forbidden_features", [])]
    candidate_inputs = feature_columns()
    bad = [
        c for c in candidate_inputs
        if any(token in c.lower() for token in forbidden)
    ]
    if bad:
        raise AssertionError(f"Forbidden candidate feature columns: {bad}")

    audit = pd.read_parquet(
        AUDIT,
        columns=["split", "season", "target", "actual", "baseline_projection"],
    )
    rt = audit.loc[audit["target"].astype(str).eq(TARGET)].copy()
    counts = {}
    for split, season in [("validation", VALIDATION_SEASON), ("test", TEST_SEASON)]:
        f = rt.loc[rt["split"].astype(str).eq(split)].copy()
        if f.empty:
            raise AssertionError(f"No audit rows for {split}")
        found = set(pd.to_numeric(f["season"], errors="raise").astype(int).unique())
        if found != {season}:
            raise AssertionError(f"{split}: unexpected seasons {sorted(found)}")
        counts[split] = int(len(f))

    info = {
        "validation_season": VALIDATION_SEASON,
        "test_season": TEST_SEASON,
        "test_used_for_selection": False,
        "market_features_used": False,
        "candidate_input_columns": len(candidate_inputs),
        "audit_rows": counts,
    }
    print("PREFLIGHT PASS: all-carry formula inputs and contracts verified")
    print(json.dumps(info, sort_keys=True))
    return info


def metrics(actual: np.ndarray, pred: np.ndarray, baseline: np.ndarray) -> dict:
    y = np.asarray(actual, dtype="float64")
    p = np.maximum(np.asarray(pred, dtype="float64"), 0.0)
    b = np.asarray(baseline, dtype="float64")
    ok = np.isfinite(y) & np.isfinite(p) & np.isfinite(b)
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

    p1 = np.clip(1.0 - np.exp(-p), 0.0, 1.0)
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


def failures(m: dict, threshold: dict) -> list[str]:
    out = []
    if m["mae"] > float(threshold["maximum_validation_mae"]):
        out.append("mae")
    if m["abs_bias"] > float(threshold["maximum_allowed_bias"]):
        out.append("bias")
    if m["improvement_vs_baseline_pct"] < float(threshold["minimum_improvement_vs_baseline_pct"]):
        out.append("improvement")
    if m["poisson_deviance"] > float(threshold["maximum_poisson_deviance"]):
        out.append("poisson_deviance")
    if m["brier_1plus_independent_poisson"] > float(threshold["maximum_brier_1plus"]):
        out.append("brier_1plus")
    return out


def prepare_split(
    features: pd.DataFrame,
    audit: pd.DataFrame,
    split: str,
    season: int,
) -> pd.DataFrame:
    a = audit.loc[
        audit["target"].astype(str).eq(TARGET)
        & audit["split"].astype(str).eq(split)
    ].copy()
    f = features.loc[features["season"].eq(season)].copy()
    merged = a.merge(f, on=GRAIN, how="left", validate="one_to_one")
    if merged["target_rushing_tds"].isna().any():
        n = int(merged["target_rushing_tds"].isna().sum())
        raise AssertionError(f"{split}: missing feature rows/targets for {n} audit rows")

    actual = num(merged["actual"])
    target = num(merged["target_rushing_tds"])
    mismatch = ~np.isclose(actual.to_numpy(), target.to_numpy(), rtol=0.0, atol=0.0)
    if mismatch.any():
        raise AssertionError(f"{split}: audit actual differs from canonical target on {int(mismatch.sum())} rows")
    return merged


def volume_estimators(frame: pd.DataFrame) -> dict[str, pd.Series]:
    result = {}

    for c in PLAYER_CARRY_COLUMNS:
        result[f"direct_{c.removeprefix('player_carries_')}"] = num(frame[c]).fillna(0.0).clip(lower=0.0)

    team_sources = {
        "matchup": num(frame["matchup_expected_team_rush_attempts"]),
        "team_roll3": num(frame["team_rush_attempts_roll3_mean"]),
        "team_roll5": num(frame["team_rush_attempts_roll5_mean"]),
        "team_ewm5": num(frame["team_rush_attempts_ewm5"]),
        "team_season": num(frame["team_rush_attempts_season_to_date"]),
        "team_coalesce": coalesce(frame, TEAM_VOLUME_COLUMNS),
    }
    share_sources = {
        "share_roll3": num(frame["player_carry_share_roll3_mean"]),
        "share_roll5": num(frame["player_carry_share_roll5_mean"]),
        "share_ewm5": num(frame["player_carry_share_ewm5"]),
        "share_season": num(frame["player_carry_share_season_to_date"]),
        "share_career": num(frame["player_carry_share_career_prior"]),
        "share_coalesce": coalesce(frame, CARRY_SHARE_COLUMNS),
    }

    for tname, team in team_sources.items():
        t = team.fillna(0.0).clip(lower=0.0)
        for sname, share in share_sources.items():
            s = share.fillna(0.0).clip(lower=0.0, upper=1.0)
            result[f"{tname}_x_{sname}"] = (t * s).clip(lower=0.0)

    return result


def rate_estimators(
    frame: pd.DataFrame,
    league_rate: float,
) -> dict[str, pd.Series]:
    result = {}
    for name, (td_col, carry_col) in RATE_PAIRS.items():
        r = safe_ratio(frame[td_col], frame[carry_col])
        result[name] = r.fillna(league_rate).clip(lower=0.0, upper=1.0)

    # Deterministic empirical-Bayes all-carry career rate. Prior exposure=100
    # matches the existing stronger TD-efficiency regularization constant and
    # is not tuned in this diagnostic.
    career_tds = num(frame["player_rushing_tds_career_prior"]).fillna(0.0).clip(lower=0.0)
    career_carries = num(frame["player_carries_career_prior"]).fillna(0.0).clip(lower=0.0)
    prior_exposure = 100.0
    result["career_eb100"] = (
        (career_tds + prior_exposure * league_rate)
        / (career_carries + prior_exposure)
    ).clip(lower=0.0, upper=1.0)

    # Pregame fallback ladder; no target-game outcomes are consulted.
    ladder = pd.Series(np.nan, index=frame.index, dtype="float64")
    for name in ["season_to_date", "ewm5", "roll5", "roll3", "career_prior"]:
        ladder = ladder.where(ladder.notna(), result[name])
    result["coalesced"] = ladder.fillna(league_rate).clip(lower=0.0, upper=1.0)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    config = load_yaml(CONFIG)
    preflight_info = preflight(config)
    if args.preflight_only:
        print("ISSUE 56 RUSHING TDS ALL-CARRY FORMULA PREFLIGHT: PASS")
        return 0

    threshold = dict(load_yaml(THRESHOLDS)[TARGET])
    league_rates = league_prior_by_target_season(config)

    hist_path = repo_root() / config["paths"]["historical_features"]
    features = pd.read_parquet(hist_path, columns=feature_columns())
    features["season"] = pd.to_numeric(features["season"], errors="raise").astype(int)
    features["week"] = pd.to_numeric(features["week"], errors="raise").astype(int)
    features["game_id"] = features["game_id"].astype(str).str.strip()
    features["player_id"] = features["player_id"].fillna("").astype(str).str.strip()
    features = features.loc[features["season"].isin([VALIDATION_SEASON, TEST_SEASON])].copy()

    audit = pd.read_parquet(AUDIT)
    validation = prepare_split(features, audit, "validation", VALIDATION_SEASON)
    test = prepare_split(features, audit, "test", TEST_SEASON)

    v_volumes = volume_estimators(validation)
    t_volumes = volume_estimators(test)
    if set(v_volumes) != set(t_volumes):
        raise AssertionError("Volume estimator families differ between 2024 and 2025")

    v_rates = rate_estimators(validation, league_rates[VALIDATION_SEASON])
    t_rates = rate_estimators(test, league_rates[TEST_SEASON])
    if set(v_rates) != set(t_rates):
        raise AssertionError("Rate estimator families differ between 2024 and 2025")

    vy = num(validation["actual"]).to_numpy()
    vb = num(validation["baseline_projection"]).to_numpy()
    ty = num(test["actual"]).to_numpy()
    tb = num(test["baseline_projection"]).to_numpy()

    rows = []
    payload_candidates = []
    for volume_name in sorted(v_volumes):
        for rate_name in sorted(v_rates):
            name = f"{volume_name}__rate_{rate_name}"
            vp = (v_volumes[volume_name] * v_rates[rate_name]).to_numpy(dtype=float)
            tp = (t_volumes[volume_name] * t_rates[rate_name]).to_numpy(dtype=float)
            vm = metrics(vy, vp, vb)
            tm = metrics(ty, tp, tb)
            vf = failures(vm, threshold)
            tf = failures(tm, threshold)
            rows.append({
                "name": name,
                "volume_estimator": volume_name,
                "rate_estimator": rate_name,
                "validation_mae": vm["mae"],
                "validation_bias": vm["bias"],
                "validation_improvement_pct": vm["improvement_vs_baseline_pct"],
                "validation_poisson_deviance": vm["poisson_deviance"],
                "validation_brier_1plus": vm["brier_1plus_independent_poisson"],
                "validation_failed": ";".join(vf),
                "validation_pass": not vf,
                "test_mae_reporting_only": tm["mae"],
                "test_bias_reporting_only": tm["bias"],
                "test_improvement_pct_reporting_only": tm["improvement_vs_baseline_pct"],
                "test_poisson_deviance_reporting_only": tm["poisson_deviance"],
                "test_brier_1plus_reporting_only": tm["brier_1plus_independent_poisson"],
                "test_failed_reporting_only": ";".join(tf),
                "test_pass_reporting_only": not tf,
            })
            payload_candidates.append({
                "name": name,
                "volume_estimator": volume_name,
                "rate_estimator": rate_name,
                "validation_metrics": vm,
                "validation_failed": vf,
                "test_metrics_reporting_only": tm,
                "test_failed_reporting_only": tf,
            })

    table = pd.DataFrame(rows)
    passing = table.loc[table["validation_pass"]].copy()
    if not passing.empty:
        chosen_row = passing.sort_values(
            ["validation_mae", "validation_poisson_deviance", "name"],
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
            ["_fail_count", "validation_mae", "_abs_bias", "validation_poisson_deviance", "name"],
            kind="mergesort",
        ).iloc[0]
        selection_status = "no_candidate_passed_2024_gates"

    chosen_name = str(chosen_row["name"])
    chosen = next(c for c in payload_candidates if c["name"] == chosen_name)

    table = table.sort_values(
        ["validation_pass", "validation_mae", "validation_poisson_deviance", "name"],
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
        "structural_family": "projected_carries * all-carry rushing_td_rate",
        "target_rate_definition": "rushing_tds / carries",
        "selection_policy": {
            "selection_season": VALIDATION_SEASON,
            "test_season": TEST_SEASON,
            "test_used_for_selection": False,
            "best_candidate_rule": (
                "prefer candidates passing all locked 2024 diagnostic gates; "
                "then lowest 2024 MAE and Poisson deviance"
            ),
        },
        "league_rate_prior": {
            "2024_uses_seasons_before_2024": league_rates[VALIDATION_SEASON],
            "2025_uses_seasons_before_2025": league_rates[TEST_SEASON],
        },
        "eb_prior_exposure": 100.0,
        "eb_prior_exposure_tuned": False,
        "candidate_count": int(len(table)),
        "selection_status": selection_status,
        "chosen_from_2024_only": chosen,
        "top_20_validation": table.head(20).to_dict(orient="records"),
        "thresholds_unchanged": threshold,
        "preflight": preflight_info,
        "outputs": {"csv": str(OUT_CSV), "json": str(OUT_JSON)},
    }
    OUT_JSON.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("ISSUE 56 RUSHING TDS ALL-CARRY FORMULA DIAGNOSTIC")
    print("STRUCTURE: projected_carries * all_carry_rushing_td_rate")
    print(
        f"league_rate_prior_2024={league_rates[VALIDATION_SEASON]:.8f} "
        f"league_rate_prior_2025={league_rates[TEST_SEASON]:.8f}"
    )
    print(f"candidates={len(table)} selection_status={selection_status}")
    print(
        f"chosen_2024_only={chosen['name']} "
        f"volume={chosen['volume_estimator']} rate={chosen['rate_estimator']}"
    )
    vm = chosen["validation_metrics"]
    tm = chosen["test_metrics_reporting_only"]
    print(
        "2024: "
        f"mae={vm['mae']:.6f} bias={vm['bias']:.6f} "
        f"improvement={vm['improvement_vs_baseline_pct']:.6f}% "
        f"poisson={vm['poisson_deviance']:.6f} "
        f"brier={vm['brier_1plus_independent_poisson']:.6f} "
        f"failed={';'.join(chosen['validation_failed']) if chosen['validation_failed'] else 'none'}"
    )
    print(
        "2025_REPORTING_ONLY: "
        f"mae={tm['mae']:.6f} bias={tm['bias']:.6f} "
        f"improvement={tm['improvement_vs_baseline_pct']:.6f}% "
        f"poisson={tm['poisson_deviance']:.6f} "
        f"brier={tm['brier_1plus_independent_poisson']:.6f} "
        f"failed={';'.join(chosen['test_failed_reporting_only']) if chosen['test_failed_reporting_only'] else 'none'}"
    )
    print("TOP 10 BY 2024 VALIDATION:")
    for row in table.head(10).itertuples(index=False):
        print(
            f"{row.name}: mae={row.validation_mae:.6f} bias={row.validation_bias:.6f} "
            f"improvement={row.validation_improvement_pct:.6f}% "
            f"poisson={row.validation_poisson_deviance:.6f} "
            f"brier={row.validation_brier_1plus:.6f} "
            f"failed={row.validation_failed or 'none'}"
        )
    print(f"csv={OUT_CSV}")
    print(f"json={OUT_JSON}")
    print("ISSUE 56 RUSHING TDS ALL-CARRY FORMULA DIAGNOSTIC: COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
