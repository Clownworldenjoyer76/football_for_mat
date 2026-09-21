#!/usr/bin/env python3
from __future__ import annotations
import hashlib, io, json, os, re, runpy, sys, tempfile, time, traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import common


_CONFIG_CONTRACT = common.load_config()
PIPELINE = (
    'validate/audit_market_exclusion.py',
    'validate/validate_historical_data.py',
    'train/build_backtest_folds.py',
    'train/train_baselines.py',
    'train/train_opportunity_models.py',
    'train/train_efficiency_models.py',
    'train/train_direct_models.py',
    'train/select_model_architecture.py',
    'train/calibrate_uncertainty.py',
    'report/build_model_report.py',
)
SEEDED_TRAINERS = frozenset({
    'train/train_opportunity_models.py',
    'train/train_efficiency_models.py',
    'train/train_direct_models.py',
})
TARGETS = list(_CONFIG_CONTRACT["targets"].keys())
OPPORTUNITY_COMPONENTS = (
    'qb_pass_attempts','team_pass_attempts','team_rush_attempts',
    'player_carry_share','player_target_share','player_red_zone_target_share',
    'player_goal_line_carry_share','field_goal_attempts','extra_point_attempts',
    'opponent_offensive_plays','opponent_dropbacks','player_defensive_participation',
)
EFFICIENCY_COMPONENTS = (
    'passing_yards_per_attempt','passing_td_rate','rushing_yards_per_carry',
    'rushing_td_per_goal_line_carry','receiving_yards_per_target',
    'receiving_td_per_red_zone_target','field_goal_conversion','extra_point_conversion',
    'tackle_rate_per_defensive_play','sack_rate_per_defensive_play',
)
SEED_RE = re.compile(r'(?m)^SEED\s*=\s*\d+\s*$')
OUTPUT_TAIL = 12000

@dataclass
class StepResult:
    step_number: int
    script: str
    status: str
    exit_code: int
    started_at: str
    ended_at: str
    duration_seconds: float
    stdout_tail: str
    stderr_tail: str
    seed_override: int | None

class Tee(io.TextIOBase):
    def __init__(self, primary):
        self.primary = primary
        self.buffer = io.StringIO()
    def write(self, text):
        self.primary.write(text); self.primary.flush(); self.buffer.write(text); return len(text)
    def flush(self): self.primary.flush()
    def getvalue(self): return self.buffer.getvalue()

def now(): return datetime.now(timezone.utc)
def iso(dt=None): return (dt or now()).isoformat().replace('+00:00','Z')
def token(dt=None): return (dt or now()).strftime('%Y%m%dT%H%M%S%fZ')
def log_path(log_dir, dt=None): return log_dir / f'training_{token(dt)}.json'

def write_json_atomic(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    h = tempfile.NamedTemporaryFile('w', encoding='utf-8', newline='\n', dir=path.parent,
                                    prefix='.'+path.name+'.', suffix='.tmp', delete=False)
    tmp = Path(h.name)
    try:
        with h:
            json.dump(payload, h, indent=2, sort_keys=True); h.write('\n')
        os.replace(tmp, path)
    finally:
        if tmp.exists(): tmp.unlink()

def resolve_training_seed(config: dict[str, Any]) -> int:
    training = config.get('training')
    if not isinstance(training, dict):
        raise ValueError('Config section training must be a mapping.')
    seed = training.get('random_seed')
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('Issue 45 requires config training.random_seed to be a nonnegative integer; no fallback seed is permitted.')
    return int(seed)

def load_json(path):
    if not path.is_file(): raise FileNotFoundError(f'Required JSON missing: {path}')
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(data, dict): raise ValueError(f'Expected JSON object: {path}')
    return data

def assert_production_registry_safe(prop_root):
    registry = load_json(prop_root / 'models/production_registry.json')
    if list(registry) != list(TARGETS):
        raise ValueError('Production registry must contain exactly the nine target keys.')
    protected = []
    for target in TARGETS:
        entry = registry[target]
        if not isinstance(entry, dict): raise ValueError(f'Invalid registry entry: {target}')
        if entry.get('production_approved') is not False or entry.get('version') is not None:
            protected.append(target)
    if protected:
        raise RuntimeError('Refusing training because registered production version(s) could be overwritten: ' + ', '.join(protected))
    return registry

def validate_pipeline_files(prop_root, pipeline=PIPELINE):
    missing = [x for x in pipeline if not (prop_root/'scripts'/x).is_file()]
    if missing: raise FileNotFoundError('Missing required training script(s): ' + ', '.join(missing))

def seeded_source(source, seed, script):
    matches = list(SEED_RE.finditer(source))
    if len(matches) != 1:
        raise ValueError(f'{script}: expected exactly one numeric module-level SEED assignment; found {len(matches)}')
    return SEED_RE.sub(f'SEED = {int(seed)}', source, count=1)

def _exit_code(exc):
    if exc.code is None: return 0
    if isinstance(exc.code, int): return int(exc.code)
    return 1

def execute_script(*, step_number, script, scripts_root, repo_root, training_seed):
    path = (scripts_root/script).resolve(); started = now(); clock = time.perf_counter()
    out, err = Tee(sys.stdout), Tee(sys.stderr)
    old_argv, old_cwd = list(sys.argv), Path.cwd()
    status, code = 'success', 0
    seed_override = training_seed if script in SEEDED_TRAINERS else None
    try:
        sys.argv = [str(path)]; os.chdir(repo_root)
        from contextlib import redirect_stdout, redirect_stderr
        with redirect_stdout(out), redirect_stderr(err):
            try:
                if script in SEEDED_TRAINERS:
                    src = seeded_source(path.read_text(encoding='utf-8-sig'), training_seed, script)
                    g = {'__name__':'__main__','__file__':str(path),'__package__':None,'__cached__':None}
                    exec(compile(src, str(path), 'exec'), g, g)
                else:
                    runpy.run_path(str(path), run_name='__main__')
            except SystemExit as exc:
                code = _exit_code(exc)
                if code != 0: status = 'failed'
            except BaseException:
                status, code = 'failed', 1
                traceback.print_exc(file=err)
    finally:
        sys.argv = old_argv; os.chdir(old_cwd)
    ended = now()
    return StepResult(step_number, script, status, code, iso(started), iso(ended),
                      round(time.perf_counter()-clock,6), out.getvalue()[-OUTPUT_TAIL:],
                      err.getvalue()[-OUTPUT_TAIL:], seed_override)

Executor = Callable[..., StepResult]
def run_pipeline(*, pipeline, scripts_root, repo_root, training_seed, executor=execute_script):
    results=[]
    for i, script in enumerate(pipeline,1):
        print(f'[{i:02d}/{len(pipeline):02d}] {script}')
        r = executor(step_number=i, script=script, scripts_root=scripts_root,
                     repo_root=repo_root, training_seed=training_seed)
        results.append(r)
        if r.status != 'success' or r.exit_code != 0:
            print(f'TRAINING STOPPED: step={i} script={script} exit_code={r.exit_code}', file=sys.stderr)
            return 'failed', results
    return 'success', results

def sha256_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def metadata_seed(meta):
    for key in ('seed','random_seed'):
        v=meta.get(key)
        if isinstance(v,int) and not isinstance(v,bool): return int(v)
    params=meta.get('params')
    if isinstance(params,dict):
        v=params.get('seed')
        if isinstance(v,int) and not isinstance(v,bool): return int(v)
    return None

def cutoff_record(meta):
    for key in ('training_end','training_cutoff','cutoff_date'):
        if meta.get(key) not in (None,''): return {'kind':'timestamp','value':meta[key]}
    policy=meta.get('training_policy')
    if isinstance(policy,dict) and policy.get('final_train_end_season') is not None:
        return {'kind':'season','value':int(policy['final_train_end_season'])}
    label=meta.get('label')
    if isinstance(label,dict) and label.get('last_training_season') is not None:
        return {'kind':'season','value':int(label['last_training_season'])}
    raise ValueError('Model metadata does not store a training cutoff.')

def artifact_specs(prop_root):
    rows=[]
    for x in OPPORTUNITY_COMPONENTS: rows.append(('opportunity',x,prop_root/'models/components'/x,'model.txt'))
    for x in EFFICIENCY_COMPONENTS: rows.append(('efficiency',x,prop_root/'models/efficiency'/x,'model.txt'))
    for x in TARGETS: rows.append(('direct',x,prop_root/'models'/x,'direct_model.txt'))
    return rows

def collect_model_artifacts(prop_root, expected_seed):
    records=[]
    for kind,name,directory,model_name in artifact_specs(prop_root):
        model=directory/model_name; manifest=directory/'feature_manifest.json'; meta_path=directory/'metadata.json'
        for p in (model,manifest,meta_path):
            if not p.is_file(): raise FileNotFoundError(f'{kind}/{name}: missing {p}')
        meta=load_json(meta_path); observed=metadata_seed(meta)
        if observed != expected_seed:
            raise ValueError(f'{kind}/{name}: persisted seed {observed!r} != config seed {expected_seed}')
        manifest_sha=sha256_file(manifest)
        recorded=meta.get('feature_manifest_sha256')
        if recorded is None and isinstance(meta.get('feature_manifest'),dict): recorded=meta['feature_manifest'].get('sha256')
        if recorded is not None and recorded != manifest_sha: raise ValueError(f'{kind}/{name}: manifest SHA mismatch')
        records.append({'kind':kind,'name':name,'model_file':str(model.relative_to(prop_root)).replace('\\','/'),
                        'metadata_file':str(meta_path.relative_to(prop_root)).replace('\\','/'),
                        'feature_manifest_file':str(manifest.relative_to(prop_root)).replace('\\','/'),
                        'feature_manifest_sha256':manifest_sha,'model_sha256':sha256_file(model),
                        'training_seed':observed,'cutoff':cutoff_record(meta)})
    if len(records) != 31: raise AssertionError(f'Expected 31 model artifacts; got {len(records)}')
    return records

def main():
    if len(sys.argv) != 1: raise SystemExit('run_training.py takes no CLI arguments.')
    config=common.load_config(); seed=resolve_training_seed(config)
    repo=common.repo_root().resolve(); prop=common.prop_root().resolve(); scripts=(prop/'scripts').resolve(); logs=(prop/'logs').resolve()
    validate_pipeline_files(prop); assert_production_registry_safe(prop)
    started=now(); dest=log_path(logs,started)
    status, results = run_pipeline(pipeline=PIPELINE,scripts_root=scripts,repo_root=repo,training_seed=seed)
    failed=next((r for r in results if r.status!='success'),None)
    artifacts=[]; postflight_error=None
    if status=='success':
        try: artifacts=collect_model_artifacts(prop,seed)
        except Exception as exc:
            status='failed'; postflight_error=f'{type(exc).__name__}: {exc}'
            print('TRAINING POSTFLIGHT FAILED: '+postflight_error,file=sys.stderr)
    payload={'status':status,'started_at':iso(started),'ended_at':iso(),'training_seed':seed,
             'seed_source':'config.training.random_seed','pipeline':list(PIPELINE),'steps_total':len(PIPELINE),
             'steps_executed':len(results),'failed_step_number':failed.step_number if failed else None,
             'failed_script':failed.script if failed else None,'postflight_error':postflight_error,
             'steps':[asdict(r) for r in results],'model_artifact_count':len(artifacts),'model_artifacts':artifacts,
             'production_registry_overwrite_allowed':False,'market_features_used':False}
    write_json_atomic(payload,dest); print(f'log={dest}')
    if status!='success':
        print('MODEL TRAINING: FAIL',file=sys.stderr); return max(1,failed.exit_code) if failed else 1
    print('MODEL TRAINING: PASS'); return 0

if __name__=='__main__': raise SystemExit(main())
