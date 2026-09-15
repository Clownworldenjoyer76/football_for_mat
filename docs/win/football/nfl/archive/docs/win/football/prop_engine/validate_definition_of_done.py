#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, subprocess, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import common

TARGETS = [
    "passing_yards","passing_tds","rushing_yards","rushing_tds",
    "receiving_yards","receiving_tds","kicking_points","tackles","sacks",
]
EVENT_TARGETS = ["passing_tds","rushing_tds","receiving_tds","sacks"]
OPP = [
    "qb_pass_attempts","team_pass_attempts","team_rush_attempts",
    "player_carry_share","player_target_share","player_red_zone_target_share",
    "player_goal_line_carry_share","field_goal_attempts","extra_point_attempts",
    "opponent_offensive_plays","opponent_dropbacks","player_defensive_participation",
]
EFF = [
    "passing_yards_per_attempt","passing_td_rate","rushing_yards_per_carry",
    "rushing_td_per_goal_line_carry","receiving_yards_per_target",
    "receiving_td_per_red_zone_target","field_goal_conversion",
    "extra_point_conversion","tackle_rate_per_defensive_play",
    "sack_rate_per_defensive_play",
]
TARGET_VALUES = [
    "passing_yards","passing_tds","rushing_yards","rushing_tds",
    "receiving_yards","receiving_tds","field_goals_made","extra_points_made",
    "kicking_points","solo_tackles","assisted_tackles","tackles","sacks",
]

def args():
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, default=None)
    p.add_argument("--week", type=int, default=1)
    return p.parse_args()

def run(cmd, cwd):
    cp = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    return {"returncode": cp.returncode, "stdout": cp.stdout, "stderr": cp.stderr}

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    h = tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n",
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False)
    temp = Path(h.name)
    try:
        with h:
            json.dump(payload, h, indent=2)
            h.write("\n")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()

def add(checks, n, name, passed, details=None):
    checks.append({"number": n, "name": name, "passed": bool(passed), "details": details or {}})
    print(f"CHECK {n:02d}: {'PASS' if passed else 'BLOCKED'} - {name}")

def model_set(base, model_name):
    return all((base / x).is_file() for x in (model_name,"feature_manifest.json","metadata.json"))

def weekly_check(report, phrase):
    return any(
        isinstance(x, dict)
        and x.get("passed") is True
        and phrase.casefold() in str(x.get("name","")).casefold()
        for x in report.get("checks", [])
    )

def main():
    a = args()
    config = common.load_config()
    repo = common.repo_root().resolve()
    prop = common.prop_root().resolve()
    season = int(a.season if a.season is not None else config["seasons"]["current"])
    week = int(a.week)
    checks = []

    # 1
    roots = [prop/x for x in ("config","scripts","models","data","evaluation","output","logs","tests")]
    add(checks,1,"Every Prop Engine implementation file/deliverable is rooted under prop_engine",
        all(x.exists() for x in roots), {"root": str(prop)})

    # 2
    watched = [
        "docs/win/football/nfl/scripts","docs/win/football/nfl/models",
        "docs/win/football/nfl/training","docs/win/football/nfl/01_merge",
    ]
    d = run(["git","diff","--name-only","HEAD","--",*watched], repo)
    outside = [x.strip().replace("\\","/") for x in d["stdout"].splitlines()
               if x.strip() and "/prop_engine/" not in x.replace("\\","/")]
    add(checks,2,"Existing game-prediction files remain unchanged",
        d["returncode"] == 0 and not outside, {"changed_outside_prop_engine": outside})

    # 3
    cw_path = repo / config["paths"]["identity_crosswalk"]
    resolved_ok = False
    resolved_rows = unresolved_rows = 0
    if cw_path.is_file():
        cw = pd.read_parquet(cw_path, columns=["player_id","gsis_id","resolution_status"])
        status = cw["resolution_status"].astype("string").fillna("").str.strip().str.lower()
        resolved = cw.loc[status.eq("resolved")]
        unresolved = cw.loc[status.eq("unresolved")]
        resolved_rows, unresolved_rows = len(resolved), len(unresolved)
        p = resolved["player_id"].astype("string").fillna("").str.strip()
        g = resolved["gsis_id"].astype("string").fillna("").str.strip()
        resolved_ok = resolved_rows > 0 and p.ne("").all() and g.ne("").all() and p.eq(g).all()
    canonical = str(config.get("system",{}).get("canonical_player_id","")).casefold()
    add(checks,3,"GSIS ID is canonical", canonical == "gsis_id" and resolved_ok,
        {"resolved_rows": resolved_rows, "unresolved_rows_allowed": unresolved_rows})

    # 4-6
    grain = ["season","week","game_id","player_id"]
    up = repo / config["paths"]["historical_universe"]
    tp = repo / config["paths"]["historical_targets"]
    targets_exist = zero_ok = non_ok = False
    zero_rows = non_rows = 0
    if up.is_file() and tp.is_file():
        universe = pd.read_parquet(up, columns=[*grain,"played_game_flag"])
        target_cols = list(pd.read_parquet(tp).columns)
        targets_exist = set(TARGETS).issubset(target_cols)
        targets = pd.read_parquet(tp, columns=[*grain,"target_source_present",*TARGET_VALUES])
        joined = universe.merge(targets, on=grain, how="inner", validate="one_to_one")
        played = pd.to_numeric(joined["played_game_flag"], errors="coerce")
        src = pd.to_numeric(joined["target_source_present"], errors="coerce")
        vals = joined[TARGET_VALUES].apply(pd.to_numeric, errors="coerce")
        zm = played.eq(1) & src.eq(0)
        nm = played.eq(0) & src.eq(0)
        zero_rows, non_rows = int(zm.sum()), int(nm.sum())
        zero_ok = zero_rows > 0 and vals.loc[zm].notna().all().all() and vals.loc[zm].eq(0).all().all()
        non_ok = non_rows > 0 and vals.loc[nm].isna().all().all()
    add(checks,4,"All nine historical targets exist", targets_exist)
    add(checks,5,"Zero-stat participants remain represented", zero_ok, {"rows": zero_rows})
    add(checks,6,"Nonparticipants are not false zeros", non_ok, {"rows": non_rows})

    # 7
    fc = prop / "validate_final_data_contracts.py"
    r = run([sys.executable,str(fc)], repo) if fc.is_file() else {"returncode":2,"stdout":"","stderr":"missing"}
    pregame_ok = r["returncode"] == 0 and "FINAL DATA CONTRACTS VALIDATION: PASS" in r["stdout"]
    add(checks,7,"All model features are pregame-safe", pregame_ok)

    # 8-10
    pri_path = prop/"data/current"/f"{season}_week_1_priors.parquet"
    pri_ok = False; rookies = new_team = 0
    if pri_path.is_file():
        pri = pd.read_parquet(pri_path)
        pri_ok = len(pri)>0 and {"week1_role_projection","week1_uncertainty_multiplier"}.issubset(pri.columns)
        if "career_games" in pri.columns:
            rookies = int(pd.to_numeric(pri["career_games"],errors="coerce").fillna(0).eq(0).sum())
        if "new_team_flag" in pri.columns:
            new_team = int(pd.to_numeric(pri["new_team_flag"],errors="coerce").fillna(0).gt(0).sum())
    add(checks,8,"Week 1 fallback exists", pri_ok)
    add(checks,9,"Rookie fallback exists", pri_ok and rookies>0, {"rookie_rows": rookies})
    add(checks,10,"Trade logic exists", pri_ok and pregame_ok and new_team>0, {"new_team_rows": new_team})

    weekly_path = prop/"output"/str(season)/f"week_{week}_validation.json"
    weekly_report = read_json(weekly_path) if weekly_path.is_file() else {}

    role_test = run([sys.executable,"-m","pytest","-q",str(prop/"tests/test_role_selection.py")], repo)
    role_ok = role_test["returncode"] == 0
    add(checks,11,"Backup-promotion logic exists",
        role_ok and (prop/"scripts/project/select_roles.py").is_file(),
        {"role_test_returncode": role_test["returncode"]})
    add(checks,12,"QB starter selection exists",
        role_ok and weekly_check(weekly_report,"no two QBs receive full QB1 volume"))
    add(checks,13,"Kicker selection exists",
        weekly_check(weekly_report,"one primary kicker receives majority kicking opportunity"))
    add(checks,14,"Offensive opportunity allocation exists",
        (prop/"scripts/project/allocate_team_opportunity.py").is_file()
        and weekly_check(weekly_report,"target and carry shares are logically bounded and reconciled"))
    add(checks,15,"Defensive participation modeling exists",
        (prop/"models/components/player_defensive_participation/model.txt").is_file()
        and weekly_check(weekly_report,"defensive projections limited to plausible participants"))

    # 16-17
    missing_direct = [t for t in TARGETS if not model_set(prop/"models"/t,"direct_model.txt")]
    add(checks,16,"Direct models exist for all nine targets", not missing_direct, {"missing": missing_direct})
    missing_comp = []
    for x in OPP:
        if not model_set(prop/"models/components"/x,"model.txt"):
            missing_comp.append("opportunity:"+x)
    for x in EFF:
        if not model_set(prop/"models/efficiency"/x,"model.txt"):
            missing_comp.append("efficiency:"+x)
    add(checks,17,"All required opportunity and efficiency component models exist",
        not missing_comp, {"missing": missing_comp})

    # 18-19
    tr = config.get("training",{})
    split = {
        "model_selection_train_end_season": tr.get("model_selection_train_end_season"),
        "development_validation_season": tr.get("development_validation_season"),
        "final_train_end_season": tr.get("final_train_end_season"),
        "untouched_test_season": tr.get("untouched_test_season"),
    }
    split_ok = split == {
        "model_selection_train_end_season":2023,
        "development_validation_season":2024,
        "final_train_end_season":2024,
        "untouched_test_season":2025,
    }
    selection_ok = True
    for t in TARGETS:
        p = prop/"models"/t/"selected_model.json"
        if not p.is_file() or read_json(p).get("test_used_for_selection") is not False:
            selection_ok = False
    add(checks,18,"Chronological validation exists", split_ok and selection_ok, {"split": split})
    add(checks,19,"2025 is untouched during initial model selection", split_ok and selection_ok)

    # 20
    bad_cal = []
    for t in TARGETS:
        p = prop/"models/calibration"/f"{t}_calibration.json"
        if not p.is_file():
            bad_cal.append(t); continue
        j = read_json(p)
        if j.get("target") != t or j.get("market_features_used") is not False:
            bad_cal.append(t)
    add(checks,20,"Uncertainty is calibrated", not bad_cal, {"invalid_or_missing": bad_cal})

    # 21-22
    active_path = prop/"output"/str(season)/f"week_{week}_active_player_projections.csv"
    interval_ok = prob_ok = False
    active_rows = 0
    if active_path.is_file():
        active = pd.read_csv(active_path, low_memory=False)
        active_rows = len(active)
        req = {"target","projection","low","high","probability_1_plus"}
        if active_rows>0 and req.issubset(active.columns):
            point = pd.to_numeric(active["projection"],errors="coerce")
            low = pd.to_numeric(active["low"],errors="coerce")
            high = pd.to_numeric(active["high"],errors="coerce")
            applicable = low.notna() & high.notna()
            interval_ok = (
                point.notna().all() and applicable.any()
                and (low.loc[applicable] <= point.loc[applicable]).all()
                and (point.loc[applicable] <= high.loc[applicable]).all()
                and weekly_check(weekly_report,"nonnegative projections, ordered intervals, and probabilities in [0,1]")
            )
            ev = active.loc[active["target"].isin(EVENT_TARGETS)]
            probs = pd.to_numeric(ev["probability_1_plus"],errors="coerce")
            prob_ok = len(ev)>0 and probs.notna().all() and probs.between(0,1).all()
    add(checks,21,"Final output includes point, low, and high estimates where applicable",
        interval_ok, {"active_rows": active_rows})
    add(checks,22,"TD and sack outputs include supported event probabilities", prob_ok)

    # 23
    m = run([sys.executable,str(prop/"scripts/validate/audit_market_exclusion.py"),"--preflight"], repo)
    add(checks,23,"Market-exclusion audit passes",
        m["returncode"]==0 and "MARKET EXCLUSION AUDIT: PASS" in m["stdout"])

    # 24
    h = run([sys.executable,str(prop/"scripts/validate/validate_historical_data.py")], repo)
    hpath = prop/"evaluation/historical_validation.json"
    hok = False; hdetails={"returncode":h["returncode"]}
    if hpath.is_file():
        hj = read_json(hpath)
        hdetails.update({"status":hj.get("status"),"checks_failed":hj.get("checks_failed")})
        hok = h["returncode"]==0 and hj.get("status")=="passed" and int(hj.get("checks_failed",0))==0
    add(checks,24,"Historical validation passes", hok, hdetails)

    # 25
    w = run([sys.executable,str(prop/"scripts/validate/validate_week.py"),
             "--season",str(season),"--week",str(week)], repo)
    wok = False
    if weekly_path.is_file():
        wj = read_json(weekly_path)
        wok = w["returncode"]==0 and wj.get("status")=="passed" and int(wj.get("checks_failed",0))==0
    add(checks,25,"Weekly validation passes", wok, {"returncode":w["returncode"]})

    # 26
    u = run([sys.executable,"-m","pytest","-q",str(prop/"tests"),
             "--ignore",str(prop/"tests/test_end_to_end.py")], repo)
    add(checks,26,"Unit tests pass", u["returncode"]==0,
        {"returncode":u["returncode"],"stdout_tail":u["stdout"][-1800:],"stderr_tail":u["stderr"][-1800:]})

    # 27
    s = run([sys.executable,"-m","pytest","-q",str(prop/"tests/test_end_to_end.py")], repo)
    add(checks,27,"End-to-end historical-as-current smoke test passes",
        s["returncode"]==0, {"returncode":s["returncode"]})

    # 28
    reg_path = prop/"models/production_registry.json"
    reg_ok = False; state = {}
    if reg_path.is_file():
        reg = read_json(reg_path)
        reg_ok = set(reg)==set(TARGETS)
        for t in TARGETS:
            e = reg.get(t)
            approved = isinstance(e,dict) and e.get("production_approved") is True
            version = e.get("version") if isinstance(e,dict) else None
            state[t]={"production_approved":approved,"version":version}
            if not approved or version in (None,""):
                reg_ok=False
    add(checks,28,"Production registry contains explicit approved versions", reg_ok, {"targets":state})

    # 29
    longp = prop/"output"/str(season)/f"week_{week}_player_projections.csv"
    widep = prop/"output"/str(season)/f"week_{week}_player_projections_wide.csv"
    add(checks,29,"Weekly runner creates long and wide outputs",
        (prop/"scripts/run_weekly.py").is_file()
        and longp.is_file() and longp.stat().st_size>0
        and widep.is_file() and widep.stat().st_size>0)

    # 30
    manp = prop/"output"/str(season)/f"week_{week}_run_manifest.json"
    man_ok=False; mdetails={"path":str(manp)}
    if manp.is_file():
        j=read_json(manp)
        keys={"source_hashes","model_versions","as_of","feature_schema_hash","validation_passed","market_data_used"}
        missing=sorted(keys-set(j))
        mdetails.update({
            "missing_keys":missing,"status":j.get("status"),
            "allow_unapproved_models":j.get("allow_unapproved_models"),
            "validation_passed":j.get("validation_passed"),
            "market_data_used":j.get("market_data_used"),
        })
        versions=j.get("model_versions")
        hashes=j.get("source_hashes")
        man_ok=(
            not missing and isinstance(hashes,dict) and bool(hashes)
            and isinstance(versions,dict) and set(versions)==set(TARGETS)
            and all(v not in (None,"") for v in versions.values())
            and bool(j.get("as_of")) and bool(j.get("feature_schema_hash"))
            and j.get("validation_passed") is True
            and j.get("market_data_used") is False
            and j.get("allow_unapproved_models") is False
            and j.get("status")=="success"
        )
    add(checks,30,"Run manifest contains complete production run metadata", man_ok, mdetails)

    blocked=[c for c in checks if not c["passed"]]
    payload={
        "status":"passed" if not blocked else "blocked",
        "generated_at":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
        "season":season,"week":week,"checks_total":len(checks),
        "checks_passed":len(checks)-len(blocked),"checks_blocked":len(blocked),
        "blocked_checks":[{"number":c["number"],"name":c["name"]} for c in blocked],
        "checks":checks,
    }
    report=prop/"evaluation/final_definition_of_done.json"
    write_json(report,payload)
    print(f"checks_total={payload['checks_total']}")
    print(f"checks_passed={payload['checks_passed']}")
    print(f"checks_blocked={payload['checks_blocked']}")
    if blocked:
        print("blocked="+",".join(str(c["number"]) for c in blocked))
        print(f"report={report}")
        print("FINAL DEFINITION OF DONE: BLOCKED")
        return 1
    print(f"report={report}")
    print("FINAL DEFINITION OF DONE: PASS")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
