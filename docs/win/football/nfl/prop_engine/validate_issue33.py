#!/usr/bin/env python3
"""Independent acceptance validator for Issue 33 component projections."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
try:
    import lightgbm as lgb
except ModuleNotFoundError as exc:
    raise SystemExit("Issue 33 validation requires LightGBM.") from exc

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / "scripts"
TRAIN = SCRIPTS / "train"
for p in (SCRIPTS, TRAIN):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import common
import train_opportunity_models as opportunity
import train_efficiency_models as efficiency

GRAIN=["season","week","game_id","player_id"]
TEAM_GRAIN=["season","week","game_id","team"]
HEADERS=[
"season","week","game_id","player_id","team","opponent","position",
"projected_team_pass_attempts","projected_qb_pass_attempts","projected_team_rush_attempts",
"projected_player_carries","projected_target_share","projected_targets",
"projected_yards_per_attempt","projected_yards_per_carry","projected_yards_per_target",
"projected_red_zone_targets","projected_goal_line_carries","projected_fg_attempts",
"projected_fg_make_probability","projected_pat_attempts"]
OPP=["qb_pass_attempts","team_pass_attempts","team_rush_attempts","player_carry_share","player_target_share","player_red_zone_target_share","player_goal_line_carry_share","field_goal_attempts","extra_point_attempts"]
EFF=["passing_yards_per_attempt","rushing_yards_per_carry","receiving_yards_per_target","field_goal_conversion"]
RZ=["team_red_zone_pass_attempts_roll3_mean","team_red_zone_pass_attempts_roll5_mean","team_red_zone_pass_attempts_ewm5","team_red_zone_pass_attempts_season_to_date"]
GL=["team_goal_line_rush_attempts_roll3_mean","team_goal_line_rush_attempts_roll5_mean","team_goal_line_rush_attempts_ewm5","team_goal_line_rush_attempts_season_to_date"]


def args():
    p=argparse.ArgumentParser(); p.add_argument("--season",type=int,default=None); p.add_argument("--week",type=int,required=True); return p.parse_args()

def num(s): return pd.to_numeric(s,errors="coerce").replace([np.inf,-np.inf],np.nan).astype("float64")
def load_yaml(path):
    with path.open("r",encoding="utf-8-sig") as h: v=yaml.safe_load(h)
    if not isinstance(v,dict): raise AssertionError(f"Expected YAML mapping: {path}")
    return v

def load_json(path):
    with path.open("r",encoding="utf-8-sig") as h: return json.load(h)
def assert_close(a,b,label,atol=1e-9):
    x=num(a); y=num(b); same_nan=x.isna().eq(y.isna())
    if not same_nan.all(): raise AssertionError(f"{label}: null-mask mismatch")
    ok=x.isna() | np.isclose(x,y,rtol=1e-9,atol=atol,equal_nan=True)
    if not ok.all():
        idx=ok[~ok].index[:10].tolist(); raise AssertionError(f"{label}: numeric mismatch at rows {idx}")
def coalesce(frame,cols):
    out=pd.Series(np.nan,index=frame.index,dtype="float64")
    for c in cols: out=out.where(out.notna(),num(frame[c]))
    return out

def booster(root,family,name):
    d=root/"models"/family/name; m=load_json(d/"feature_manifest.json"); b=lgb.Booster(model_file=str(d/"model.txt")); names=list(m.get("numeric_features",[]))+list(m.get("categorical_features",[]))
    if list(b.feature_name())!=names: raise AssertionError(f"{family}/{name}: persisted feature order mismatch")
    return b,m,names

def opp_rows(features,name,elig):
    spec=opportunity.COMPONENTS[name]; f=list(spec["features"])
    missing=[c for c in f if c not in features.columns]
    if missing: raise AssertionError(f"{name}: missing features {missing[:20]}")
    if spec["scope"]=="team":
        opportunity.check_team_feature_invariance(features,f); return opportunity.team_rows_from_features(features,f)
    positions={str(x).strip().upper() for x in elig[str(spec["eligible_rule"])]["eligible_positions"]}; pos=features.position.fillna("").astype(str).str.strip().str.upper(); return features.loc[pos.isin(positions)].copy()
def score_opp(root,features,elig):
    out={}
    for name in OPP:
        b,m,names=booster(root,"components",name); rows=opp_rows(features,name,elig); p=opportunity.transform_prediction(b.predict(opportunity.numeric_frame(rows,names)),name); key=GRAIN if opportunity.COMPONENTS[name]["scope"]=="player" else TEAM_GRAIN; q=rows[key].copy(); q[name]=p; out[name]=q
    return out

def eff_cols():
    c=[*GRAIN,"kickoff_timestamp","position","position_group"]
    for n in EFF:
        for f in efficiency.FEATURES[n]:
            if f not in efficiency.DERIVED_FEATURES and f not in c: c.append(f)
    return c

def eff_inference(current,raw,name,elig):
    rule=efficiency.ELIGIBILITY_RULE[name]; positions={str(x).strip().upper() for x in elig[rule]["eligible_positions"]}; pos=current.position.fillna("").astype(str).str.strip().str.upper(); target=current.loc[pos.isin(positions)].copy()
    pc=[*GRAIN,"kickoff_timestamp","position","position_group","_prior_position_group","_numerator","_exposure","_label"]
    hist=raw[pc].copy(); hist["_mark"]=0
    ph=target[[*GRAIN,"kickoff_timestamp","position","position_group"]].copy(); ph["_prior_position_group"]=efficiency.normalize_position_group(ph.position,ph.position_group); ph["_numerator"]=np.nan; ph["_exposure"]=np.nan; ph["_label"]=np.nan; ph["_mark"]=1
    e=efficiency.add_strict_prior_features(pd.concat([hist,ph],ignore_index=True,sort=False),name); e=e.loc[e["_mark"].eq(1)].copy(); canonical=[f for f in efficiency.FEATURES[name] if f not in efficiency.DERIVED_FEATURES]; return e.merge(target[[*GRAIN,*canonical]],on=GRAIN,how="left",validate="one_to_one",suffixes=("","_canonical"))
def score_eff(root,current,history,elig,config):
    base=efficiency.prepare_label_base(config,history); out={}
    for name in EFF:
        raw=efficiency.apply_eligibility(efficiency.build_component_label(base,name),name,elig); rows=eff_inference(current,raw,name,elig); b,m,names=booster(root,"efficiency",name)
        if names!=list(efficiency.FEATURES[name]): raise AssertionError(f"{name}: manifest/trainer order mismatch")
        p=efficiency.transform_prediction(b.predict(efficiency.feature_matrix(rows,name)),name); q=rows[GRAIN].copy(); q[name]=p; out[name]=q
    return out


def main():
    a=args(); config=common.load_config(); season=int(a.season) if a.season is not None else int(config["seasons"]["current"]); week=int(a.week); repo=common.repo_root(); prop=common.prop_root()
    builder=prop/"scripts"/"project"/"project_components.py"; features_path=prop/"data"/"current"/"features"/f"{season}_week_{week}_features.parquet"; roles_path=prop/"data"/"current"/f"{season}_week_{week}_roles.parquet"; output_path=prop/"data"/"current"/f"{season}_week_{week}_component_projections.parquet"; log_path=prop/"logs"/f"component_projections_{season}_week_{week}.json"; elig_path=prop/"config"/"target_eligibility.yaml"; hist_path=repo/config["paths"]["historical_features"]
    print("CHECK 01: required builder/output/log, sequence inputs, and exact headers")
    req=[builder,features_path,roles_path,output_path,log_path,elig_path,hist_path]
    if week==1: req.append(prop/"data"/"current"/f"{season}_week_1_priors.parquet")
    for p in req:
        if not p.is_file(): raise AssertionError(f"Missing Issue 33 artifact/input: {p}")
    out=pd.read_parquet(output_path)
    if list(out.columns)!=HEADERS: raise AssertionError(f"Issue 33 headers mismatch: {list(out.columns)}")
    common.ensure_unique(out,GRAIN,"Issue 33 validator output")

    print("CHECK 02: player grain/current context and role assignments")
    features=pd.read_parquet(features_path); roles=pd.read_parquet(roles_path); common.ensure_unique(features,GRAIN,"Issue 33 validator features"); expected=features[[*GRAIN,"team","opponent","position"]].merge(out,on=[*GRAIN,"team","opponent","position"],how="outer",indicator=True)
    if not expected["_merge"].eq("both").all(): raise AssertionError("Issue 33 output does not exactly match current-feature player grain")
    role=roles[[*GRAIN,"primary_qb_flag","primary_kicker_flag"]]; check=out.merge(role,on=GRAIN,how="left",validate="one_to_one"); qb=num(check.primary_qb_flag).fillna(0).gt(0); k=num(check.primary_kicker_flag).fillna(0).gt(0)
    if not num(check.loc[~qb,"projected_qb_pass_attempts"]).eq(0).all(): raise AssertionError("Non-primary QB rows received QB attempts")
    if not num(check.loc[~k,"projected_fg_attempts"]).eq(0).all() or not num(check.loc[~k,"projected_pat_attempts"]).eq(0).all(): raise AssertionError("Non-primary kicker rows received kicking volume")
    if check.loc[~k,"projected_fg_make_probability"].notna().any(): raise AssertionError("Non-primary kicker received FG make probability")

    print("CHECK 03: independently rescore persisted Issue 22 opportunity models")
    eligibility=load_yaml(elig_path); features["kickoff_timestamp"]=pd.to_datetime(features["kickoff_timestamp"],utc=True,errors="raise"); opp=score_opp(prop,features,eligibility); ctx=out[[*GRAIN,"team"]].copy()
    for name,col in [("team_pass_attempts","projected_team_pass_attempts"),("team_rush_attempts","projected_team_rush_attempts")]:
        z=ctx.merge(opp[name],on=TEAM_GRAIN,how="left",validate="many_to_one"); assert_close(out[col],z[name],col)
    z=ctx.merge(opp["qb_pass_attempts"],on=GRAIN,how="left",validate="one_to_one"); expected_qb=num(z.qb_pass_attempts).where(qb,0.0); assert_close(out.projected_qb_pass_attempts,expected_qb,"projected_qb_pass_attempts")
    carry=ctx.merge(opp["player_carry_share"],on=GRAIN,how="left",validate="one_to_one"); carry_share=num(carry.player_carry_share).fillna(0).clip(0,1); assert_close(out.projected_player_carries,num(out.projected_team_rush_attempts)*carry_share,"projected_player_carries")
    targ=ctx.merge(opp["player_target_share"],on=GRAIN,how="left",validate="one_to_one"); ts=num(targ.player_target_share).fillna(0).clip(0,1); assert_close(out.projected_target_share,ts,"projected_target_share"); assert_close(out.projected_targets,num(out.projected_team_pass_attempts)*ts,"projected_targets")
    rz=ctx.merge(opp["player_red_zone_target_share"],on=GRAIN,how="left",validate="one_to_one"); rs=num(rz.player_red_zone_target_share).fillna(0).clip(0,1); assert_close(out.projected_red_zone_targets,coalesce(features,RZ).clip(lower=0)*rs,"projected_red_zone_targets")
    gl=ctx.merge(opp["player_goal_line_carry_share"],on=GRAIN,how="left",validate="one_to_one"); gs=num(gl.player_goal_line_carry_share).fillna(0).clip(0,1); assert_close(out.projected_goal_line_carries,coalesce(features,GL).clip(lower=0)*gs,"projected_goal_line_carries")
    for name,col in [("field_goal_attempts","projected_fg_attempts"),("extra_point_attempts","projected_pat_attempts")]:
        z=ctx.merge(opp[name],on=TEAM_GRAIN,how="left",validate="many_to_one"); exp=num(z[name]).where(k,0.0); assert_close(out[col],exp,col)

    print("CHECK 04: independently reconstruct model-local efficiency priors and rescore Issue 23 models")
    history=pd.read_parquet(hist_path,columns=eff_cols()); history["season"]=pd.to_numeric(history.season,errors="raise").astype(int); history["week"]=pd.to_numeric(history.week,errors="raise").astype(int); history["kickoff_timestamp"]=pd.to_datetime(history.kickoff_timestamp,utc=True,errors="raise"); history=history.loc[history.season.lt(season)].copy(); eff=score_eff(prop,features,history,eligibility,config)
    for name,col in [("passing_yards_per_attempt","projected_yards_per_attempt"),("rushing_yards_per_carry","projected_yards_per_carry"),("receiving_yards_per_target","projected_yards_per_target")]:
        z=ctx.merge(eff[name],on=GRAIN,how="left",validate="one_to_one"); assert_close(out[col],z[name],col)
    z=ctx.merge(eff["field_goal_conversion"],on=GRAIN,how="left",validate="one_to_one"); fp=num(z.field_goal_conversion).where(k,np.nan).clip(0,1); assert_close(out.projected_fg_make_probability,fp,"projected_fg_make_probability")

    print("CHECK 05: bounds, no premature share reconciliation, market exclusion, and log policy")
    for c in ["projected_team_pass_attempts","projected_qb_pass_attempts","projected_team_rush_attempts","projected_player_carries","projected_target_share","projected_targets","projected_red_zone_targets","projected_goal_line_carries","projected_fg_attempts","projected_pat_attempts"]:
        if num(out[c]).isna().any() or num(out[c]).lt(0).any(): raise AssertionError(f"Invalid nonnegative output: {c}")
    if num(out.projected_target_share).gt(1).any(): raise AssertionError("Target share outside [0,1]")
    if num(out.loc[k,"projected_fg_make_probability"]).isna().any() or not num(out.loc[k,"projected_fg_make_probability"]).between(0,1).all(): raise AssertionError("Primary kicker FG probability invalid")
    common.reject_forbidden_feature_columns(out.columns,config)
    audit=SCRIPTS/"validate"/"audit_market_exclusion.py"; cp=subprocess.run([sys.executable,str(audit)],cwd=repo,capture_output=True,text=True,check=False)
    if cp.returncode!=0 or "MARKET EXCLUSION AUDIT: PASS" not in cp.stdout: raise AssertionError("Market exclusion audit failed")
    log=load_json(log_path)
    if log.get("shares_reconciled") is not False or log.get("share_reconciliation_stage")!="current_week_allocation": raise AssertionError("Issue 33 share-allocation boundary not logged correctly")
    if log.get("market_features_used") is not False: raise AssertionError("Issue 33 log says market features were used")
    print(f"season={season}\nweek={week}\ngames={out.game_id.nunique()}\nteams={out.team.nunique()}\nrows={len(out)}\ncolumns={len(out.columns)}\nopportunity_models_rescored={len(OPP)}\nefficiency_models_rescored={len(EFF)}\nprimary_qbs={int(qb.sum())}\nprimary_kickers={int(k.sum())}\nshares_reconciled=false\nmarket_features_used=false")
    print("ISSUE 33 ACCEPTANCE: PASS")
    return 0

if __name__=="__main__": raise SystemExit(main())
