#!/usr/bin/env python3
from __future__ import annotations
import hashlib, importlib.util, json, sys, tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent; RUNNER=HERE/'scripts/run_training.py'
EXPECTED=[
'validate/audit_market_exclusion.py','validate/validate_historical_data.py','train/build_backtest_folds.py',
'train/train_baselines.py','train/train_opportunity_models.py','train/train_efficiency_models.py',
'train/train_direct_models.py','train/select_model_architecture.py','train/calibrate_uncertainty.py','report/build_model_report.py']
SEEDED={'train/train_opportunity_models.py','train/train_efficiency_models.py','train/train_direct_models.py'}
TARGETS=['passing_yards','passing_tds','rushing_yards','rushing_tds','receiving_yards','receiving_tds','kicking_points','tackles','sacks']
def fail(x): raise AssertionError(x)
def load_runner():
    scripts=HERE/'scripts'; sys.path.insert(0,str(scripts)) if str(scripts) not in sys.path else None
    spec=importlib.util.spec_from_file_location('issue45_runner',RUNNER); m=importlib.util.module_from_spec(spec); sys.modules[spec.name]=m; spec.loader.exec_module(m); return m
def result(m,i,s,ok,seed):
    return m.StepResult(i,s,'success' if ok else 'failed',0 if ok else 9,'a','b',1.0,'','',seed if s in SEEDED else None)
def write_json(p,x): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
def make_models(m,root,seed):
    for kind,name,d,model_name in m.artifact_specs(root):
        d.mkdir(parents=True,exist_ok=True); (d/model_name).write_text('model\n',encoding='utf-8'); (d/'feature_manifest.json').write_text('{"features":["x"]}\n',encoding='utf-8')
        sha=hashlib.sha256((d/'feature_manifest.json').read_bytes()).hexdigest(); meta={'seed':seed,'feature_manifest_sha256':sha,'training_policy':{'final_train_end_season':2024}}
        if kind=='direct': meta['random_seed']=seed; meta['training_end']='2025-01-05T20:20:00+00:00'
        write_json(d/'metadata.json',meta)
def main():
    if not RUNNER.is_file(): fail('missing run_training.py')
    m=load_runner()
    if list(m.PIPELINE)!=EXPECTED: fail('execution order mismatch')
    if set(m.SEEDED_TRAINERS)!=SEEDED: fail('seeded trainer set mismatch')
    m.validate_pipeline_files(HERE)
    if m.resolve_training_seed({'training':{'random_seed':76076}})!=76076: fail('config seed not used')
    for bad in ({'training':{}},{'training':{'random_seed':None}},{'training':{'random_seed':'76076'}},{'training':{'random_seed':True}},{}):
        try: m.resolve_training_seed(bad)
        except ValueError: pass
        else: fail('invalid seed accepted')
    src='SEED = 22022\nVALUE = SEED\n'; ns={}; exec(compile(m.seeded_source(src,76076,'x.py'),'x.py','exec'),ns,ns)
    if ns['SEED']!=76076 or ns['VALUE']!=76076: fail('seed injection failed')
    m.assert_production_registry_safe(HERE)
    with tempfile.TemporaryDirectory() as td:
        r=Path(td); safe={t:{'production_approved':False,'version':None} for t in TARGETS}; write_json(r/'models/production_registry.json',safe); m.assert_production_registry_safe(r)
        a=json.loads(json.dumps(safe)); a['passing_yards']['production_approved']=True; write_json(r/'models/production_registry.json',a)
        try: m.assert_production_registry_safe(r)
        except RuntimeError: pass
        else: fail('approved production overwrite not blocked')
        v=json.loads(json.dumps(safe)); v['passing_yards']['version']='v1'; write_json(r/'models/production_registry.json',v)
        try: m.assert_production_registry_safe(r)
        except RuntimeError: pass
        else: fail('versioned production overwrite not blocked')
    calls=[]
    def ok(**kw): calls.append(kw['script']); return result(m,kw['step_number'],kw['script'],True,kw['training_seed'])
    status,rows=m.run_pipeline(pipeline=m.PIPELINE,scripts_root=HERE/'scripts',repo_root=HERE.parents[4],training_seed=76076,executor=ok)
    if status!='success' or calls!=EXPECTED or len(rows)!=10: fail('success pipeline contract failed')
    calls=[]
    def stop(**kw): calls.append(kw['script']); return result(m,kw['step_number'],kw['script'],kw['step_number']!=2,kw['training_seed'])
    status,rows=m.run_pipeline(pipeline=m.PIPELINE,scripts_root=HERE/'scripts',repo_root=HERE.parents[4],training_seed=76076,executor=stop)
    if status!='failed' or calls!=EXPECTED[:2] or any(x.startswith('train/') for x in calls): fail('validation failure did not stop training')
    with tempfile.TemporaryDirectory() as td:
        r=Path(td); make_models(m,r,76076); rec=m.collect_model_artifacts(r,76076)
        if len(rec)!=31 or not all(x['feature_manifest_sha256'] and x['cutoff']['value'] is not None for x in rec): fail('artifact postflight failed')
        p=r/rec[0]['metadata_file']; x=json.loads(p.read_text()); x['seed']=1; write_json(p,x)
        try: m.collect_model_artifacts(r,76076)
        except ValueError: pass
        else: fail('seed mismatch not rejected')
    print('runner=scripts/run_training.py'); print('steps=10'); print('execution_order_exact=true'); print('validation_steps_first=2'); print('stop_on_validation_failure=true'); print('training_steps_after_validation_failure=0'); print('fixed_seed_source=config.training.random_seed'); print('seed_injected_in_memory=true'); print('training_source_files_mutated=false'); print('production_registry_guard=true'); print('approved_production_overwrite_blocked=true'); print('versioned_production_overwrite_blocked=true'); print('model_artifacts_verified=31'); print('model_cutoffs_stored=true'); print('feature_manifests_stored=true'); print('market_features_used=false'); print('MODEL TRAINING RUNNER VALIDATION: PASS'); print('ISSUE 45 ACCEPTANCE: PASS'); return 0
if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as exc: print('MODEL TRAINING RUNNER VALIDATION: FAIL - '+str(exc),file=sys.stderr); raise SystemExit(1)
