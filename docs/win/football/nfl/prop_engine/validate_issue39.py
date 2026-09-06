#!/usr/bin/env python3
"""Independent acceptance validator for Issue 39 model performance reports."""

from __future__ import annotations

import ast
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import common

GRAIN = ["season", "week", "game_id", "player_id"]
BUILDER = HERE / "scripts" / "report" / "build_model_report.py"
AUDIT = HERE / "evaluation" / "model_selection_predictions.parquet"
LOG = HERE / "logs" / "model_performance_report.json"

METRIC_HEADERS = [
    "target", "sample_size", "mae", "rmse", "median_absolute_error", "r2",
    "poisson_deviance", "brier_1plus", "logloss_1plus", "mean_actual",
    "mean_projection", "bias",
]
OUTPUTS = {
    "target": HERE / "evaluation" / "metrics_by_target.csv",
    "position": HERE / "evaluation" / "metrics_by_position.csv",
    "usage": HERE / "evaluation" / "metrics_by_usage.csv",
    "week": HERE / "evaluation" / "metrics_by_week.csv",
    "role": HERE / "evaluation" / "metrics_by_role.csv",
    "calibration": HERE / "evaluation" / "calibration_by_projection_range.csv",
}
SELECTED_PROJECTION_COLUMNS = {
    "baseline": "baseline_projection",
    "direct": "direct_projection",
    "component": "component_projection",
    "direct_component_blend": "blend_projection",
}
CALIBRATION_HEADERS = [
    "target", "projection_decile", "sample_size", "projection_min", "projection_max",
    "mean_projection", "mean_actual", "bias", "mean_probability_1plus",
    "actual_rate_1plus", "brier_1plus", "logloss_1plus",
]
USAGE_CANDIDATES = {
    "passing_yards": ["player_pass_attempts_roll3_mean", "player_pass_attempts_roll5_mean", "player_pass_attempts_ewm5", "player_pass_attempts_career_prior"],
    "passing_tds": ["player_pass_attempts_roll3_mean", "player_pass_attempts_roll5_mean", "player_pass_attempts_ewm5", "player_pass_attempts_career_prior"],
    "rushing_yards": ["player_carries_roll3_mean", "player_carries_roll5_mean", "player_carries_ewm5", "player_carries_career_prior"],
    "rushing_tds": ["player_goal_line_carries_roll3_mean", "player_goal_line_carries_roll5_mean", "player_carries_roll3_mean", "player_carries_career_prior"],
    "receiving_yards": ["player_targets_roll3_mean", "player_targets_roll5_mean", "player_targets_ewm5", "player_targets_career_prior"],
    "receiving_tds": ["player_red_zone_targets_roll3_mean", "player_red_zone_targets_roll5_mean", "player_targets_roll3_mean", "player_targets_career_prior"],
    "kicking_points": ["player_field_goal_attempts_roll3_mean", "player_field_goal_attempts_roll5_mean", "player_field_goal_attempts_career_prior"],
    "tackles": ["player_defense_participation_roll3_mean", "role_participation_roll3", "player_defense_participation_career_prior"],
    "sacks": ["player_defense_participation_roll3_mean", "role_participation_roll3", "player_defense_participation_career_prior"],
}
KICKING_USAGE_COMPONENTS = ["player_field_goal_attempts_roll3_mean", "player_extra_point_attempts_roll3_mean"]


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.casefold() in {"", "nan", "none", "null", "<na>", "nat"} else text


def num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).astype(float)


def target_is_count(config: dict[str, Any], target: str) -> bool:
    return str(config["targets"][target].get("type", "")) == "count_nonnegative"


def metrics(frame: pd.DataFrame, count: bool) -> dict[str, float | int | None]:
    y = num(frame["actual"]).to_numpy(float)
    p = num(frame["projection"]).to_numpy(float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    if not len(y):
        raise AssertionError("empty metric slice")
    e = p - y
    sst = float(np.sum((y - y.mean()) ** 2))
    out: dict[str, float | int | None] = {
        "sample_size": int(len(y)),
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "median_absolute_error": float(np.median(np.abs(e))),
        "r2": None if sst <= 0 else float(1 - np.sum(e ** 2) / sst),
        "poisson_deviance": None,
        "brier_1plus": None,
        "logloss_1plus": None,
        "mean_actual": float(y.mean()),
        "mean_projection": float(p.mean()),
        "bias": float(p.mean() - y.mean()),
    }
    if count:
        lam = np.maximum(p, 1e-12)
        terms = np.empty_like(y)
        z = y <= 0
        terms[z] = lam[z]
        nz = ~z
        terms[nz] = y[nz] * np.log(y[nz] / lam[nz]) - (y[nz] - lam[nz])
        out["poisson_deviance"] = float(2 * terms.mean())
        pr = np.clip(1 - np.exp(-np.maximum(p, 0)), 1e-12, 1 - 1e-12)
        ev = (y >= 1).astype(float)
        out["brier_1plus"] = float(np.mean((pr - ev) ** 2))
        out["logloss_1plus"] = float(-np.mean(ev * np.log(pr) + (1 - ev) * np.log(1 - pr)))
    return out


def close(a: Any, b: Any, tol: float = 1e-10) -> bool:
    if pd.isna(a) and (b is None or pd.isna(b)):
        return True
    if b is None and pd.isna(a):
        return True
    try:
        return math.isclose(float(a), float(b), rel_tol=0, abs_tol=tol)
    except (TypeError, ValueError):
        return False


def assert_metric_row(row: pd.Series, expected: dict[str, Any], label: str) -> None:
    if int(row["sample_size"]) != int(expected["sample_size"]):
        raise AssertionError(f"{label}: sample_size mismatch")
    for key in METRIC_HEADERS[2:]:
        if not close(row[key], expected[key]):
            raise AssertionError(f"{label}: {key} mismatch actual={row[key]!r} expected={expected[key]!r}")


def selected_test(config: dict[str, Any], audit: pd.DataFrame) -> pd.DataFrame:
    chunks = []
    for target in config["targets"]:
        selected = json.loads((HERE / "models" / target / "selected_model.json").read_text(encoding="utf-8-sig"))
        arch = clean(selected.get("selected_architecture") or selected.get("selected_candidate"))
        if arch not in SELECTED_PROJECTION_COLUMNS:
            raise AssertionError(f"{target}: invalid selected architecture")
        if selected.get("test_used_for_selection") is not False:
            raise AssertionError(f"{target}: untouched test selection contract violated")
        col = SELECTED_PROJECTION_COLUMNS[arch]
        sub = audit.loc[(audit["split"].astype(str) == "test") & (audit["target"].astype(str) == target), [*GRAIN, "target", "actual", col]].copy()
        sub = sub.rename(columns={col: "projection"})
        sub["actual"] = num(sub["actual"])
        sub["projection"] = num(sub["projection"])
        sub = sub.loc[sub["actual"].notna() & sub["projection"].notna()].copy()
        if sub.empty:
            raise AssertionError(f"{target}: empty untouched test rows")
        chunks.append(sub)
    frame = pd.concat(chunks, ignore_index=True)
    common.ensure_unique(frame, [*GRAIN, "target"], "Issue39 validator selected test")
    return frame


def rank_bucket(series: pd.Series, k: int, prefix: str) -> pd.Series:
    out = pd.Series(pd.NA, index=series.index, dtype="string")
    x = num(series)
    ok = x.notna()
    if ok.any():
        pct = x.loc[ok].rank(method="first", pct=True)
        labels = np.maximum(1, np.minimum(k, np.ceil(pct.to_numpy(float) * k).astype(int)))
        out.loc[ok] = [f"{prefix}{v}" for v in labels]
    return out


def load_enriched(config: dict[str, Any], selected: pd.DataFrame) -> pd.DataFrame:
    path = common.repo_root() / str(config["paths"]["historical_features"])
    manifest_path = path.with_name("feature_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    schema = set(manifest.get("leading_columns", []))
    families = manifest.get("column_families", {})
    if isinstance(families, dict):
        for values in families.values():
            if isinstance(values, list): schema.update(str(v) for v in values)
    required = [*GRAIN, "position", "role_depth_starter_flag_pregame", "history_no_nfl_history_flag", "history_new_team_flag"]
    missing = [c for c in required if c not in schema]
    if missing:
        raise AssertionError(f"missing historical context {missing}")
    opt = {"role_injury_status_pregame", "injury_game_status", "depth_injury", "role_injury_out_flag", "role_injury_doubtful_flag", "role_injury_questionable_flag", *KICKING_USAGE_COMPONENTS}
    for v in USAGE_CANDIDATES.values(): opt.update(v)
    cols = list(dict.fromkeys(required + sorted(opt & schema)))
    ctx = pd.read_parquet(path, columns=cols)
    years = set(pd.to_numeric(selected["season"], errors="raise").astype(int).unique())
    ctx = ctx.loc[pd.to_numeric(ctx["season"], errors="coerce").isin(years)].copy()
    e = selected.merge(ctx, on=GRAIN, how="left", validate="many_to_one")
    if e["position"].isna().any(): raise AssertionError("context join failed")

    e["starter_backup"] = np.where(num(e["role_depth_starter_flag_pregame"]).fillna(0).gt(0), "starter", "backup")
    e["history_status"] = np.where(num(e["history_no_nfl_history_flag"]).fillna(0).gt(0), "no_history", "has_history")
    e["team_status"] = np.where(num(e["history_new_team_flag"]).fillna(0).gt(0), "new_team", "returning_team")
    wk = pd.to_numeric(e["week"], errors="raise").astype(int)
    e["week_phase"] = np.where(wk <= 4, "week_1_4", "week_5_plus")
    e["week_exact"] = [f"week_{x:02d}" for x in wk]

    injury_col = next((c for c in ["role_injury_status_pregame", "injury_game_status", "depth_injury"] if c in e.columns), None)
    if injury_col:
        s = e[injury_col].fillna("").astype(str).str.strip().str.casefold().replace({"": "none", "nan": "none", "<na>": "none"})
        e["injury_status"] = s
    else:
        needed = ["role_injury_out_flag", "role_injury_doubtful_flag", "role_injury_questionable_flag"]
        if not all(c in e.columns for c in needed): raise AssertionError("no injury context")
        e["injury_status"] = "none"
        e.loc[num(e["role_injury_questionable_flag"]).fillna(0).gt(0), "injury_status"] = "questionable"
        e.loc[num(e["role_injury_doubtful_flag"]).fillna(0).gt(0), "injury_status"] = "doubtful"
        e.loc[num(e["role_injury_out_flag"]).fillna(0).gt(0), "injury_status"] = "out"

    e["usage_score"] = np.nan
    for target in config["targets"]:
        mask = e["target"].eq(target)
        sub = e.loc[mask]
        if target == "kicking_points" and all(c in e.columns for c in KICKING_USAGE_COMPONENTS):
            a, b = num(sub[KICKING_USAGE_COMPONENTS[0]]), num(sub[KICKING_USAGE_COMPONENTS[1]])
            usage = a + b
            usage = usage.where(a.notna() | b.notna())
        else:
            cand = [c for c in USAGE_CANDIDATES[target] if c in e.columns]
            if not cand: raise AssertionError(f"{target}: no usage context")
            usage = pd.Series(np.nan, index=sub.index)
            for c in cand: usage = usage.where(usage.notna(), num(sub[c]))
        e.loc[mask, "usage_score"] = usage.to_numpy()

    e["usage_quartile"] = pd.NA
    e["projection_decile"] = pd.NA
    for target, idx in e.groupby("target", sort=False).groups.items():
        idx = list(idx)
        e.loc[idx, "usage_quartile"] = rank_bucket(e.loc[idx, "usage_score"], 4, "Q").fillna("unknown").to_numpy()
        e.loc[idx, "projection_decile"] = rank_bucket(e.loc[idx, "projection"], 10, "D").to_numpy()
    return e


def validate_report_rows(frame: pd.DataFrame, enriched: pd.DataFrame, config: dict[str, Any], kind: str) -> float:
    max_error = 0.0
    for _, row in frame.iterrows():
        target = str(row["target"])
        sub = enriched.loc[enriched["target"].eq(target)]
        if kind == "position":
            sub = sub.loc[sub["position"].astype(str).eq(str(row["position"]))]
        elif kind == "usage":
            st, sv = str(row["slice_type"]), str(row["slice_value"])
            col = {"usage_quartile": "usage_quartile", "projection_decile": "projection_decile"}[st]
            sub = sub.loc[sub[col].astype(str).eq(sv)]
        elif kind == "week":
            st, sv = str(row["slice_type"]), str(row["slice_value"])
            col = {"week": "week_exact", "week_phase": "week_phase"}[st]
            sub = sub.loc[sub[col].astype(str).eq(sv)]
        elif kind == "role":
            st, sv = str(row["slice_type"]), str(row["slice_value"])
            col = {"starter_backup": "starter_backup", "rookie_no_history": "history_status", "new_team": "team_status", "injury_status": "injury_status"}[st]
            sub = sub.loc[sub[col].astype(str).eq(sv)]
        exp = metrics(sub, target_is_count(config, target))
        assert_metric_row(row, exp, f"{kind}:{target}")
        for k in METRIC_HEADERS[2:]:
            if exp[k] is not None and not pd.isna(row[k]): max_error = max(max_error, abs(float(row[k]) - float(exp[k])))
    return max_error


def main() -> int:
    config = common.load_config()
    targets = list(config["targets"])

    print("CHECK 01: required builder, six outputs, log, and static contracts")
    for p in [BUILDER, AUDIT, LOG, *OUTPUTS.values()]:
        if not p.is_file(): raise AssertionError(f"Missing Issue39 artifact/input: {p}")
    source = BUILDER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    if "MODEL PERFORMANCE REPORTS BUILD: PASS" not in source or 'default="test"' not in source:
        raise AssertionError("Builder missing untouched-test default/pass marker")
    log = json.loads(LOG.read_text(encoding="utf-8-sig"))
    if log.get("status") != "passed" or log.get("reporting_split") != "test":
        raise AssertionError("Issue39 log is not a passed untouched-test report")
    if bool(log.get("market_features_used", True)):
        raise AssertionError("Issue39 log reports market features")

    by_target = pd.read_csv(OUTPUTS["target"])
    by_position = pd.read_csv(OUTPUTS["position"])
    by_usage = pd.read_csv(OUTPUTS["usage"])
    by_week = pd.read_csv(OUTPUTS["week"])
    by_role = pd.read_csv(OUTPUTS["role"])
    calibration = pd.read_csv(OUTPUTS["calibration"])
    if list(by_target.columns) != METRIC_HEADERS: raise AssertionError("metrics_by_target header mismatch")
    if list(by_position.columns) != ["target", "position", *METRIC_HEADERS[1:]]: raise AssertionError("metrics_by_position header mismatch")
    for f, name in [(by_usage,"usage"),(by_week,"week"),(by_role,"role")]:
        if list(f.columns) != ["target", "slice_type", "slice_value", *METRIC_HEADERS[1:]]: raise AssertionError(f"metrics_by_{name} header mismatch")
    if list(calibration.columns) != CALIBRATION_HEADERS: raise AssertionError("calibration header mismatch")

    print("CHECK 02: independently resolve selected architectures on untouched 2025 test split")
    audit = pd.read_parquet(AUDIT)
    selected = selected_test(config, audit)
    seasons = sorted(pd.to_numeric(selected["season"], errors="raise").astype(int).unique().tolist())
    if seasons != [2025]: raise AssertionError(f"Expected untouched 2025 reporting only, got {seasons}")
    if set(selected["target"]) != set(targets): raise AssertionError("selected test target coverage mismatch")

    print("CHECK 03: independently reconstruct target metrics")
    if len(by_target) != len(targets) or set(by_target["target"]) != set(targets): raise AssertionError("metrics_by_target target coverage mismatch")
    max_err = 0.0
    for target in targets:
        sub = selected.loc[selected["target"].eq(target)]
        exp = metrics(sub, target_is_count(config, target))
        row = by_target.loc[by_target["target"].eq(target)].iloc[0]
        assert_metric_row(row, exp, f"target:{target}")
        for k in METRIC_HEADERS[2:]:
            if exp[k] is not None and not pd.isna(row[k]): max_err = max(max_err, abs(float(row[k]) - float(exp[k])))

    print("CHECK 04: required position, role, usage, week-phase, and projection-decile slices")
    enriched = load_enriched(config, selected)
    max_err = max(max_err, validate_report_rows(by_position, enriched, config, "position"))
    max_err = max(max_err, validate_report_rows(by_usage, enriched, config, "usage"))
    max_err = max(max_err, validate_report_rows(by_week, enriched, config, "week"))
    max_err = max(max_err, validate_report_rows(by_role, enriched, config, "role"))
    if set(by_usage["slice_type"]) != {"usage_quartile", "projection_decile"}: raise AssertionError("usage slice types incomplete")
    if not {"week_1_4", "week_5_plus"}.issubset(set(by_week.loc[by_week["slice_type"].eq("week_phase"), "slice_value"].astype(str))): raise AssertionError("week phase slices incomplete")
    required_role_types = {"starter_backup", "rookie_no_history", "new_team", "injury_status"}
    if set(by_role["slice_type"]) != required_role_types: raise AssertionError("role slice types incomplete")

    print("CHECK 05: independently verify calibration by projection decile")
    if set(calibration["target"]) != set(targets): raise AssertionError("calibration target coverage mismatch")
    for _, row in calibration.iterrows():
        target, decile = str(row["target"]), str(row["projection_decile"])
        sub = enriched.loc[enriched["target"].eq(target) & enriched["projection_decile"].astype(str).eq(decile)]
        y, p = num(sub["actual"]).to_numpy(float), num(sub["projection"]).to_numpy(float)
        if int(row["sample_size"]) != len(sub): raise AssertionError(f"calibration {target}/{decile}: sample mismatch")
        checks = {
            "projection_min": float(np.min(p)), "projection_max": float(np.max(p)),
            "mean_projection": float(np.mean(p)), "mean_actual": float(np.mean(y)),
            "bias": float(np.mean(p)-np.mean(y)),
        }
        for k,v in checks.items():
            if not close(row[k], v): raise AssertionError(f"calibration {target}/{decile}: {k} mismatch")
        if target_is_count(config, target):
            pr = np.clip(1-np.exp(-np.maximum(p,0)), 1e-12, 1-1e-12)
            ev = (y>=1).astype(float)
            extra = {
                "mean_probability_1plus": float(pr.mean()), "actual_rate_1plus": float(ev.mean()),
                "brier_1plus": float(np.mean((pr-ev)**2)),
                "logloss_1plus": float(-np.mean(ev*np.log(pr)+(1-ev)*np.log(1-pr))),
            }
            for k,v in extra.items():
                if not close(row[k], v): raise AssertionError(f"calibration {target}/{decile}: {k} mismatch")
        else:
            for k in ["mean_probability_1plus","actual_rate_1plus","brier_1plus","logloss_1plus"]:
                if not pd.isna(row[k]): raise AssertionError(f"calibration {target}/{decile}: {k} should be null")

    print("CHECK 06: metric sanity, sample accounting, and market policy")
    for frame in [by_target, by_position, by_usage, by_week, by_role]:
        if (pd.to_numeric(frame["sample_size"], errors="coerce") <= 0).any(): raise AssertionError("nonpositive metric sample_size")
        if (pd.to_numeric(frame["mae"], errors="coerce") < 0).any() or (pd.to_numeric(frame["rmse"], errors="coerce") < 0).any(): raise AssertionError("negative error metric")
    market = json.loads((HERE / "evaluation" / "market_exclusion_audit.json").read_text(encoding="utf-8-sig"))
    if not bool(market.get("passed")) or market.get("forbidden_source_references") or market.get("forbidden_feature_columns"):
        raise AssertionError("market exclusion audit not clean")

    print(f"reporting_season={seasons[0]}")
    print(f"targets={len(targets)}")
    print(f"rows_evaluated={len(selected)}")
    print(f"metrics_by_target_rows={len(by_target)}")
    print(f"metrics_by_position_rows={len(by_position)}")
    print(f"metrics_by_usage_rows={len(by_usage)}")
    print(f"metrics_by_week_rows={len(by_week)}")
    print(f"metrics_by_role_rows={len(by_role)}")
    print(f"calibration_rows={len(calibration)}")
    print(f"max_metric_abs_error={max_err:.12g}")
    print("untouched_test_only=true")
    print("market_features_used=false")
    print("ISSUE 39 ACCEPTANCE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
