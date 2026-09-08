#!/usr/bin/env python3
"""Independent acceptance validation for Issue 36 final weekly projections."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:
    import lightgbm as lgb
except ModuleNotFoundError as exc:
    raise SystemExit("Issue 36 validation requires LightGBM.") from exc

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / "scripts"
PROJECT = SCRIPTS / "project"
TRAIN = SCRIPTS / "train"
for p in (SCRIPTS, PROJECT, TRAIN):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import common
import project_components as pc
import train_opportunity_models as opportunity
import train_efficiency_models as efficiency

GRAIN = ["season", "week", "game_id", "player_id"]
TEAM_GRAIN = ["season", "week", "game_id", "team"]
TARGETS = [
    "passing_yards", "passing_tds", "rushing_yards", "rushing_tds",
    "receiving_yards", "receiving_tds", "kicking_points", "tackles", "sacks",
]
OUT_COLS = [
    "season", "week", "game_id", "game_date", "game_time", "player_id",
    "player_name", "team", "opponent", "position", "target", "projection",
    "low", "high", "probability_1_plus", "probability_2_plus",
    "eligibility_status", "eligibility_reason", "role_status",
    "injury_game_status", "depth_rank", "model_architecture", "model_version",
    "feature_asof", "generated_at",
]
DIRECT = {t: f"direct_{t}" for t in TARGETS}
ONE_DEC = {"passing_yards", "rushing_yards", "receiving_yards", "kicking_points", "tackles"}
THREE_DEC = {"passing_tds", "rushing_tds", "receiving_tds", "sacks"}
COUNT = {"passing_tds", "rushing_tds", "receiving_tds", "tackles", "sacks"}
EXTRA_OPP = [
    "player_red_zone_target_share", "player_goal_line_carry_share",
    "opponent_offensive_plays", "opponent_dropbacks", "player_defensive_participation",
]
EXTRA_EFF = [
    "passing_td_rate", "rushing_td_per_goal_line_carry",
    "receiving_td_per_red_zone_target", "extra_point_conversion",
    "tackle_rate_per_defensive_play", "sack_rate_per_defensive_play",
]


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--week", type=int, required=True)
    return p.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AssertionError(f"Missing required JSON: {path}")
    with path.open("r", encoding="utf-8-sig") as h:
        x = json.load(h)
    if not isinstance(x, dict):
        raise AssertionError(f"Expected JSON object: {path}")
    return x


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AssertionError(f"Missing required YAML: {path}")
    with path.open("r", encoding="utf-8-sig") as h:
        x = yaml.safe_load(h)
    if not isinstance(x, dict):
        raise AssertionError(f"Expected YAML mapping: {path}")
    return x


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")


def clean(v: Any) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    t = str(v).strip()
    return "" if t.casefold() in {"", "nan", "none", "null", "<na>", "nat"} else t


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def registry_version(reg: dict[str, Any], path: Path, target: str) -> str:
    def ev(x: Any) -> str:
        if isinstance(x, str):
            return clean(x)
        if isinstance(x, dict):
            for k in ("model_version", "production_version", "active_version", "version", "release"):
                z = clean(x.get(k))
                if z:
                    return z
        return ""
    for name in ("targets", "models", "production"):
        c = reg.get(name)
        if isinstance(c, dict) and target in c:
            z = ev(c[target])
            if z:
                return z
        if isinstance(c, list):
            for item in c:
                if isinstance(item, dict) and clean(item.get("target") or item.get("name")) == target:
                    z = ev(item)
                    if z:
                        return z
    if target in reg:
        z = ev(reg[target])
        if z:
            return z
    for k in ("model_version", "production_version", "active_version", "release_version"):
        z = clean(reg.get(k))
        if z:
            return z
    return "registry-" + sha(path)[:12]


def booster_manifest(prop: Path, family: str, name: str) -> tuple[lgb.Booster, list[str]]:
    d = prop / "models" / family / name
    m = read_json(d / "feature_manifest.json")
    names = list(m.get("numeric_features", [])) + list(m.get("categorical_features", []))
    b = lgb.Booster(model_file=str(d / "model.txt"))
    if list(b.feature_name()) != names:
        raise AssertionError(f"{family}/{name}: persisted feature order mismatch")
    return b, names


TEAM_OPP_SOURCE = {
    "player_defensive_opponent_plays_roll3": "team_offensive_plays_roll3_mean",
    "player_defensive_opponent_dropbacks_roll3": "team_dropbacks_roll3_mean",
    "player_defensive_opponent_rush_rate_roll3": "team_rush_rate_roll3_mean",
    "player_defensive_opponent_pass_rate_roll3": "team_pass_rate_roll3_mean",
}
TEAM_DEF_RATE = "player_defensive_team_def_sack_rate_roll3"


def strict_team_def_rate(config: dict[str, Any], repo: Path, season: int) -> pd.DataFrame:
    path = repo / str(config["paths"]["opponent_opportunity"])
    raw = common.read_parquet_required(
        path, ["season", "week", "team", "sacks", "opponent_dropbacks"]
    ).copy()
    raw["season"] = pd.to_numeric(raw["season"], errors="raise").astype(int)
    raw["week"] = pd.to_numeric(raw["week"], errors="raise").astype(int)
    raw = raw.loc[raw["season"].lt(season)].copy()
    raw["_team_key"] = raw["team"].map(opportunity.canonical_team)
    sacks = num(raw["sacks"]); drops = num(raw["opponent_dropbacks"])
    raw["_rate"] = np.where(sacks.notna() & drops.notna() & drops.ne(0), sacks / drops, np.nan)
    raw = raw.sort_values(["_team_key", "season", "week"], kind="mergesort")
    records = []
    for team, f in raw.groupby("_team_key", sort=False):
        rates = f["_rate"].dropna().to_numpy(dtype="float64")
        records.append({"_team_key": team, TEAM_DEF_RATE: float(np.mean(rates[-3:])) if len(rates) else np.nan})
    out = pd.DataFrame(records)
    common.ensure_unique(out, ["_team_key"], "Issue36 validator strict team defense rate")
    return out


def team_opp_rows(features: pd.DataFrame, names: list[str], team_def_rate: pd.DataFrame) -> pd.DataFrame:
    rebuilt = set(TEAM_OPP_SOURCE) | {TEAM_DEF_RATE}
    passthrough = [n for n in names if n not in rebuilt]
    common.require_columns(
        features, [*TEAM_GRAIN, "opponent", *passthrough, *TEAM_OPP_SOURCE.values()],
        "Issue36 validator team-opponent current context"
    )
    opportunity.check_team_feature_invariance(features, passthrough)
    rows = opportunity.team_rows_from_features(features, passthrough)
    source_cols = list(TEAM_OPP_SOURCE.values())
    opportunity.check_team_feature_invariance(features, source_cols)
    ctx = opportunity.team_rows_from_features(features, source_cols).rename(
        columns={"team": "_context_team", **{src: dst for dst, src in TEAM_OPP_SOURCE.items()}}
    )
    rows = rows.merge(
        ctx[["season", "week", "game_id", "_context_team", *TEAM_OPP_SOURCE.keys()]],
        left_on=["season", "week", "game_id", "opponent"],
        right_on=["season", "week", "game_id", "_context_team"],
        how="left", validate="one_to_one"
    )
    rows["_team_key"] = rows["team"].map(opportunity.canonical_team)
    rows = rows.merge(team_def_rate, on="_team_key", how="left", validate="many_to_one")
    common.ensure_unique(rows, TEAM_GRAIN, "Issue36 validator reconstructed team-opponent rows")
    return rows


def score_opp(
    prop: Path, features: pd.DataFrame, elig: dict[str, Any], name: str,
    team_def_rate: pd.DataFrame,
) -> pd.DataFrame:
    b, names = booster_manifest(prop, "components", name)
    spec = opportunity.COMPONENTS[name]
    scope = str(spec["scope"])
    if scope == "team":
        opportunity.check_team_feature_invariance(features, names)
        rows = opportunity.team_rows_from_features(features, names)
    elif scope == "team_opponent":
        rows = team_opp_rows(features, names, team_def_rate)
    elif scope == "player":
        rule = str(spec["eligible_rule"])
        positions = {str(x).strip().upper() for x in elig[rule]["eligible_positions"]}
        pos = features["position"].fillna("").astype(str).str.strip().str.upper()
        rows = features.loc[pos.isin(positions)].copy()
    else:
        raise AssertionError(f"{name}: unsupported opportunity scope {scope!r}")
    pred = opportunity.transform_prediction(b.predict(opportunity.numeric_frame(rows, names)), name)
    if not np.isfinite(pred).all():
        raise AssertionError(f"{name}: nonfinite prediction")
    key = GRAIN if scope == "player" else TEAM_GRAIN
    out = rows[key].copy(); out[name] = pred
    common.ensure_unique(out, key, f"Issue36 validator {name}")
    return out


def eff_cols(names: list[str]) -> list[str]:
    cols = [*GRAIN, "kickoff_timestamp", "position", "position_group"]
    for n in names:
        for f in efficiency.FEATURES[n]:
            if f not in efficiency.DERIVED_FEATURES and f not in cols:
                cols.append(f)
    return cols


def raw_histories(config: dict[str, Any], hist: pd.DataFrame, elig: dict[str, Any]) -> dict[str, pd.DataFrame]:
    base = efficiency.prepare_label_base(config, hist)
    out = {}
    for n in EXTRA_EFF:
        r = efficiency.build_component_label(base, n)
        out[n] = efficiency.apply_eligibility(r, n, elig)
    return out


def score_eff(prop: Path, features: pd.DataFrame, raw: pd.DataFrame, elig: dict[str, Any], name: str) -> pd.DataFrame:
    b, names = booster_manifest(prop, "efficiency", name)
    rows = pc.efficiency_inference_frame(features, raw, name, elig)
    if names != list(efficiency.FEATURES[name]):
        raise AssertionError(f"{name}: manifest/trainer order mismatch")
    pred = efficiency.transform_prediction(b.predict(efficiency.feature_matrix(rows, name)), name)
    if not np.isfinite(pred).all():
        raise AssertionError(f"{name}: nonfinite prediction")
    out = rows[GRAIN].copy(); out[name] = pred
    common.ensure_unique(out, GRAIN, f"Issue36 validator {name}")
    return out


def coalesce(frame: pd.DataFrame, cols: list[str]) -> pd.Series:
    out = pd.Series(np.nan, index=frame.index, dtype="float64")
    for c in cols:
        out = out.where(out.notna(), num(frame[c]))
    return out


def normalized_share(base: pd.DataFrame, pred: pd.DataFrame, name: str, mask: pd.Series) -> pd.Series:
    w = base[[*GRAIN, "team"]].merge(pred, on=GRAIN, how="left", validate="one_to_one")
    raw = num(w[name]).fillna(0).clip(0, 1)
    raw.loc[~mask.to_numpy(dtype=bool)] = 0.0
    totals = raw.groupby([w[c] for c in TEAM_GRAIN], sort=False).transform("sum")
    return raw.where(~totals.gt(0), raw / totals).clip(0, 1)


def component_points(
    config: dict[str, Any], prop: Path, repo: Path, season: int,
    component: pd.DataFrame, direct: pd.DataFrame, allocation: pd.DataFrame,
    features: pd.DataFrame, elig: dict[str, Any],
) -> dict[str, pd.Series]:
    # Issue 33 independently validates all persisted component-model scoring
    # and all nine target component formulas. Issue 36 must consume that
    # canonical implementation rather than duplicate it.
    _ = (config, prop, repo, season, elig)

    direct_index = direct.set_index(GRAIN)
    key = pd.MultiIndex.from_frame(component[GRAIN])

    receiving_td_eligible = pd.Series(
        direct_index[DIRECT["receiving_tds"]]
        .reindex(key)
        .notna()
        .to_numpy(),
        index=component.index,
    )
    rushing_td_eligible = pd.Series(
        direct_index[DIRECT["rushing_tds"]]
        .reindex(key)
        .notna()
        .to_numpy(),
        index=component.index,
    )

    points, audit = pc.final_component_points(
        component,
        allocation,
        features,
        receiving_td_eligible=receiving_td_eligible,
        rushing_td_eligible=rushing_td_eligible,
    )

    if audit.get("canonical_issue33_target_components_used") is not True:
        raise AssertionError(
            "Issue36 validator did not consume canonical Item33 components"
        )
    if audit.get("duplicate_component_model_scoring") is not False:
        raise AssertionError(
            "Issue36 validator reports duplicate component model scoring"
        )

    return points



def bucket(vals: np.ndarray, thr: dict[str, Any]) -> np.ndarray:
    low, med = float(thr["low_max"]), float(thr["medium_max"])
    out = np.full(len(vals), "low", dtype=object)
    finite = np.isfinite(vals)
    out[finite & (vals > low)] = "medium"
    out[finite & (vals > med)] = "high"
    return out


def usage(frame: pd.DataFrame, source: dict[str, Any]) -> np.ndarray:
    method = clean(source.get("method")); cols = list(source.get("columns", []))
    if method == "single":
        return num(frame[cols[0]]).to_numpy(dtype="float64")
    if method == "sum":
        total = np.zeros(len(frame), dtype="float64"); anyf = np.zeros(len(frame), dtype=bool)
        for c in cols:
            v = num(frame[c]).to_numpy(dtype="float64"); f = np.isfinite(v)
            total[f] += v[f]; anyf |= f
        total[~anyf] = np.nan
        return total
    return num(frame["selected_point_prediction"]).to_numpy(dtype="float64")


def risk(frame: pd.DataFrame, qcal: dict[str, Any]) -> np.ndarray:
    rp = qcal["risk_widening"]
    threshold = int(rp.get("low_history_games_threshold", 4))
    rook = num(frame["history_no_nfl_history_flag"]).fillna(0).to_numpy() >= .5
    promo = num(frame["role_starter_promotion_flag"]).fillna(0).to_numpy() >= .5
    h = num(frame["history_history_games"]).to_numpy(); low = ~np.isfinite(h) | (h < threshold)
    out = np.ones(len(frame), dtype="float64")
    for name, mask in (("rookie", rook), ("backup_promotion", promo), ("low_history", low)):
        out[mask] *= float(rp["factors"][name]["multiplier"])
    return np.minimum(out, float(rp.get("combined_cap", 3.0)))


def segfactor(frame: pd.DataFrame, qcal: dict[str, Any], interval: str) -> np.ndarray:
    w = qcal["coverage_widening"]
    out = np.full(len(frame), float(w["global"][interval]), dtype="float64")
    keys = frame["position_group"].astype(str) + "|" + frame["usage_bucket"].astype(str)
    for k, d in w.get("segment_extra", {}).get(interval, {}).items():
        out[keys.eq(k).to_numpy()] *= float(d["multiplier"])
    return out


def mapping(v: np.ndarray, m: dict[str, Any]) -> np.ndarray:
    out = np.interp(v, np.asarray(m["knots_x"], float), np.asarray(m["knots_y"], float), left=float(m["left_value"]), right=float(m["right_value"]))
    b = m.get("output_bounds", [None, None])
    if b[0] is not None: out = np.maximum(out, float(b[0]))
    if b[1] is not None: out = np.minimum(out, float(b[1]))
    return out


def calibrate(point: pd.Series, context: pd.DataFrame, cal: dict[str, Any]) -> pd.DataFrame:
    f = context.copy().reset_index(drop=True)
    f["selected_point_prediction"] = num(point).to_numpy(dtype="float64")
    f["position_group"] = f["position_group"].astype("string").fillna("UNKNOWN").str.strip().str.upper().replace("", "UNKNOWN")
    f["usage_bucket"] = bucket(usage(f, cal["usage_bucket"]["source"]), cal["usage_bucket"]["thresholds"])
    low = np.full(len(f), np.nan); high = np.full(len(f), np.nan)
    p1 = np.full(len(f), np.nan); p2 = np.full(len(f), np.nan)
    q50 = None; expected = None
    mode = str(cal["calibration_mode"])
    if mode in {"quantiles", "quantiles_and_count"}:
        q = cal["quantile_calibration"]; rq = {k: float(v) for k,v in q["residual_quantiles"].items()}; r = risk(f, q)
        pointv = num(f["selected_point_prediction"]).to_numpy(); center = rq["q50"]
        vals: dict[str,np.ndarray] = {"q50": pointv + center}
        for interval in ("q10_q90", "q25_q75"):
            spec = q["intervals"][interval]; lo_name, hi_name = str(spec["lower"]), str(spec["upper"]); sf = segfactor(f,q,interval)
            lo = pointv + center + (rq[lo_name]-center)*r*sf; hi = pointv + center + (rq[hi_name]-center)*r*sf
            if bool(q.get("floor_at_zero")): lo=np.maximum(lo,0); hi=np.maximum(hi,0)
            vals[lo_name]=np.minimum(lo,hi); vals[hi_name]=np.maximum(lo,hi)
        if bool(q.get("floor_at_zero")): vals["q50"] = np.maximum(vals["q50"],0)
        mat=np.column_stack([vals[x] for x in ["q10","q25","q50","q75","q90"]]); mat=np.maximum.accumulate(mat,axis=1)
        vals={n:mat[:,i] for i,n in enumerate(["q10","q25","q50","q75","q90"])}
        low, high, q50 = vals["q10"], vals["q90"], vals["q50"]
    if mode in {"count", "quantiles_and_count"}:
        c=cal["count_calibration"]; raw=np.maximum(num(f["selected_point_prediction"]).to_numpy(),0)
        expected=mapping(raw,c["expected_count"]["mapping"]); pp1=1-np.exp(-expected); pp2=1-np.exp(-expected)*(1+expected)
        p1=mapping(pp1,c["probability_1_plus"]["mapping"]); p2=mapping(pp2,c["probability_2_plus"]["mapping"])
    projection = expected if expected is not None else q50
    return pd.DataFrame({"projection":projection,"low":low,"high":high,"probability_1_plus":p1,"probability_2_plus":p2})


def main() -> int:
    a=args(); season, week = a.season, a.week
    repo=common.repo_root(); prop=common.prop_root(); config=common.load_config()
    builder=PROJECT/"project_week.py"
    up=prop/"data/current"/f"{season}_week_{week}_universe.parquet"
    cp=prop/"data/current"/f"{season}_week_{week}_component_projections.parquet"
    dp=prop/"data/current"/f"{season}_week_{week}_direct_projections.parquet"
    ap=prop/"data/current"/f"{season}_week_{week}_allocated_opportunity.parquet"
    fp=prop/"data/current/features"/f"{season}_week_{week}_features.parquet"
    rp=prop/"models/production_registry.json"
    ep=prop/"config/target_eligibility.yaml"
    out=prop/"output"/str(season)/f"week_{week}_player_projections.csv"
    activep=prop/"output"/str(season)/f"week_{week}_active_player_projections.csv"
    logp=prop/"logs"/f"week_projections_{season}_week_{week}.json"

    print("CHECK 01: required builder/inputs/registry/outputs/log and exact headers")
    for p in [builder,up,cp,dp,ap,fp,rp,ep,out,activep,logp]:
        if not p.is_file(): raise AssertionError(f"Missing Issue 36 artifact/input: {p}")
    audit=pd.read_csv(out); active=pd.read_csv(activep); log=read_json(logp)
    if list(audit.columns)!=OUT_COLS or list(active.columns)!=OUT_COLS: raise AssertionError("Issue 36 exact CSV header order mismatch")

    universe=pd.read_parquet(up); component=pd.read_parquet(cp); direct=pd.read_parquet(dp); allocation=pd.read_parquet(ap); features=pd.read_parquet(fp)

    # CSV inference can coerce numeric-looking game_id values to integers, while
    # the canonical parquet grain stores game_id as text. Normalize all grain
    # dtypes before MultiIndex construction/reindexing so validation compares
    # actual keys rather than dtype-mismatched keys that appear missing.
    for frame_name, frame in [
        ("audit", audit), ("active", active), ("universe", universe),
        ("component", component), ("direct", direct),
        ("allocation", allocation), ("features", features),
    ]:
        frame["season"] = pd.to_numeric(frame["season"], errors="raise").astype(int)
        frame["week"] = pd.to_numeric(frame["week"], errors="raise").astype(int)
        frame["game_id"] = frame["game_id"].astype("string").str.strip()
        frame["player_id"] = frame["player_id"].astype("string").str.strip()
        if frame[["game_id", "player_id"]].isna().any().any():
            raise AssertionError(f"{frame_name}: null canonical grain key after normalization")
        if frame["game_id"].eq("").any() or frame["player_id"].eq("").any():
            raise AssertionError(f"{frame_name}: blank canonical grain key after normalization")
    # Independently normalize current kickoff timestamps to the same UTC-aware
    # dtype used for historical efficiency history before strict-prior scoring.
    features=features.copy()
    features["kickoff_timestamp"]=pd.to_datetime(features["kickoff_timestamp"],errors="raise",utc=True)
    if features["kickoff_timestamp"].isna().any(): raise AssertionError("Invalid current kickoff_timestamp")
    common.ensure_unique(universe,GRAIN,"Issue36 validator universe")
    for name,frame in [("component",component),("direct",direct),("allocation",allocation),("features",features)]:
        common.ensure_unique(frame,GRAIN,f"Issue36 validator {name}")
    eligible_universe=universe.loc[universe["eligibility_status"].fillna("").astype(str).str.casefold().eq("eligible"),GRAIN].copy()
    if eligible_universe.empty: raise AssertionError("Universe has no eligible current players")
    eligible_keys=set(map(tuple,eligible_universe.itertuples(index=False,name=None)))
    for name,frame in [("component",component),("direct",direct),("allocation",allocation),("features",features)]:
        frame_keys=set(map(tuple,frame[GRAIN].itertuples(index=False,name=None)))
        if frame_keys!=eligible_keys:
            raise AssertionError(f"{name}: keys do not match eligible universe subset")
    if len(audit)!=len(universe)*len(TARGETS): raise AssertionError("Audit output is not full player x target grain")
    if audit.duplicated([*GRAIN,"target"]).any(): raise AssertionError("Duplicate player-target rows")

    print("CHECK 02: target eligibility, audit zeros, active-only exclusion, and current context")
    dindex=direct.set_index(GRAIN); ukey=pd.MultiIndex.from_frame(universe[GRAIN])
    eligible_counts={t:int(dindex[DIRECT[t]].reindex(ukey).notna().sum()) for t in TARGETS}
    expected_active=sum(eligible_counts.values())
    if len(active)!=expected_active: raise AssertionError(f"Active rows={len(active)} expected={expected_active}")
    if active["eligibility_status"].astype(str).str.casefold().ne("eligible").any(): raise AssertionError("Active output contains ineligible status")
    inel=audit["eligibility_status"].astype(str).str.casefold().eq("ineligible")
    for c in ["projection","low","high","probability_1_plus","probability_2_plus"]:
        v=num(audit.loc[inel,c]).fillna(0)
        if v.abs().gt(1e-12).any(): raise AssertionError(f"Ineligible audit rows have nonzero {c}")
    if audit.loc[inel,"eligibility_reason"].fillna("").astype(str).str.strip().eq("").any(): raise AssertionError("Ineligible audit row missing explicit reason")

    print("CHECK 03: independently reconstruct selected component/direct/blend point predictions")
    elig=read_yaml(ep); cpoints=component_points(config,prop,repo,season,component,direct,allocation,features,elig)
    selected={t:read_json(prop/"models"/t/"selected_model.json") for t in TARGETS}
    calibrations={t:read_json(prop/"models/calibration"/f"{t}_calibration.json") for t in TARGETS}
    reg=read_json(rp)
    compkey=pd.MultiIndex.from_frame(component[GRAIN])
    context=universe.merge(features,on=GRAIN,how="left",validate="one_to_one",suffixes=("","_feature"))
    if "position_group_feature" in context.columns: context["position_group"]=context["position_group_feature"]
    for c in ["history_no_nfl_history_flag","history_history_games","role_starter_promotion_flag"]:
        cf=f"{c}_feature"
        if cf in context.columns: context[c]=context[cf]

    print("CHECK 04: independently apply Issue 26 calibration and display rounding")
    expected_rows=[]; max_error=0.0
    for t in TARGETS:
        dv=dindex[DIRECT[t]].reindex(ukey).reset_index(drop=True); mask=dv.notna()
        cmap=pd.Series(num(cpoints[t]).to_numpy(),index=compkey)
        cv=cmap.reindex(ukey).reset_index(drop=True)
        arch=str(selected[t]["selected_architecture"])
        if arch=="direct": point=dv
        elif arch=="component": point=cv
        elif arch=="direct_component_blend":
            point=float(selected[t]["blend_weights"]["direct"])*dv+float(selected[t]["blend_weights"]["component"])*cv
        else: raise AssertionError(f"Unsupported selected architecture {arch}")
        if point.loc[mask].isna().any(): raise AssertionError(f"{t}: eligible selected point missing")
        cal=calibrate(point,context,calibrations[t])
        calc=cal.copy(); calc.loc[~mask,["projection","low","high","probability_1_plus","probability_2_plus"]]=0.0
        # Issue 38 production support rule: final weekly projections are
        # nonnegative even for historically signed yardage calibration.
        calc["projection"] = num(calc["projection"]).clip(lower=0.0)
        bounded_high = num(calc["high"]).notna()
        if bounded_high.any():
            calc.loc[bounded_high,"high"] = np.maximum(
                num(calc.loc[bounded_high,"high"]).to_numpy(dtype="float64"),
                num(calc.loc[bounded_high,"projection"]).to_numpy(dtype="float64"),
            )
        if t in ONE_DEC:
            for c in ["projection","low","high"]: calc[c]=num(calc[c]).round(1)
        if t in THREE_DEC: calc["projection"]=num(calc["projection"]).round(3)
        actual_t=audit.loc[audit["target"].eq(t)].copy().set_index(GRAIN).reindex(ukey).reset_index()
        if num(actual_t["projection"]).lt(-1e-12).any(): raise AssertionError(f"{t}: negative final projection")
        for c in ["projection","low","high","probability_1_plus","probability_2_plus"]:
            av=num(actual_t[c]); ev=num(calc[c])
            both=av.notna() & ev.notna()
            if both.any():
                err=float(np.max(np.abs(av.loc[both].to_numpy()-ev.loc[both].to_numpy())))
                max_error=max(max_error,err)
                if err>1e-10: raise AssertionError(f"{t} {c}: max abs error {err}")
            if not av.isna().equals(ev.isna()): raise AssertionError(f"{t} {c}: null mask mismatch")
        if set(actual_t["model_architecture"].astype(str))!={arch}: raise AssertionError(f"{t}: model_architecture mismatch")
        version=registry_version(reg,rp,t)
        if set(actual_t["model_version"].astype(str))!={version}: raise AssertionError(f"{t}: model_version mismatch")
        eligible=actual_t.loc[mask.to_numpy()]
        if t in COUNT:
            probs=eligible[["probability_1_plus","probability_2_plus"]].apply(pd.to_numeric,errors="coerce")
            if probs.isna().any().any() or ((probs<0)|(probs>1)).any().any(): raise AssertionError(f"{t}: probability bounds/null failure")
        else:
            if actual_t.loc[mask.to_numpy(),["probability_1_plus","probability_2_plus"]].notna().any().any(): raise AssertionError(f"{t}: non-count probabilities should be null")
        expected_rows.append(calc)

    print("CHECK 05: active output exact subset, timestamps, market/log policy")
    keys=[*GRAIN,"target"]
    audit_active=audit.loc[audit["eligibility_status"].astype(str).str.casefold().eq("eligible")].sort_values(keys,kind="mergesort").reset_index(drop=True)
    active_cmp=active.sort_values(keys,kind="mergesort").reset_index(drop=True)
    pd.testing.assert_frame_equal(active_cmp,audit_active,check_dtype=False,check_like=False)
    for c in ["feature_asof","generated_at"]:
        if audit[c].isna().any() or audit[c].astype(str).str.strip().eq("").any(): raise AssertionError(f"Missing {c}")
        for v in audit[c].astype(str).unique():
            datetime.fromisoformat(v.replace("Z","+00:00"))
    if log.get("status")!="passed" or log.get("market_features_used") is not False or log.get("market_exclusion_passed") is not True: raise AssertionError("Issue 36 log market/status policy failed")
    if int(log.get("audit_rows",-1))!=len(audit) or int(log.get("active_rows",-1))!=len(active): raise AssertionError("Issue 36 log row totals mismatch")
    if log.get("registry_sha256")!=sha(rp): raise AssertionError("Registry SHA mismatch in log")

    print(f"season={season}")
    print(f"week={week}")
    print(f"players={len(universe)}")
    print(f"audit_rows={len(audit)}")
    print(f"active_rows={len(active)}")
    print(f"targets={len(TARGETS)}")
    print(f"max_final_display_abs_error={max_error:.12g}")
    for t in TARGETS: print(f"eligible_{t}={eligible_counts[t]}")
    print("ineligible_audit_zero=true")
    print("active_only_ineligible_rows=0")
    print("probabilities_in_unit_interval=true")
    print("market_features_used=false")
    print("ISSUE 36 ACCEPTANCE: PASS")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
