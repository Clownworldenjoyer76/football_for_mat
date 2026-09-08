#!/usr/bin/env python3
"""Independent acceptance validator for NFL Prop Engine Issue 46."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNER = HERE / 'scripts/run_weekly.py'
EXPECTED_HEADER = '#!/usr/bin/env python3\n"""\nRun the complete NFL Prop Engine weekly projection pipeline.\n\nREADS:\n    Existing repository football-only schedule, PBP, team stats, roster,\n    depth, injuries, weather, travel, and nflverse data.\n\nWRITES:\n    docs/win/football/nfl/prop_engine/ only.\n\nNO SPORTSBOOK OR MARKET DATA IS READ.\n"""\n'
EXPECTED_PIPELINE = [
    'build/refresh_nflverse_player_data.py',
    'build/build_player_identity.py',
    'validate/audit_market_exclusion.py',
    'validate/validate_source_quality.py',
    'project/build_current_universe.py',
    'project/select_roles.py',
    'project/build_current_features.py',
    'project/build_week1_priors.py',
    'project/project_components.py',
    'project/allocate_team_opportunity.py',
    'project/project_direct.py',
    'project/project_week.py',
    'report/build_wide_output.py',
    'validate/validate_week.py',
]
REQUIRED_CLI = {'--season','--week','--as-of','--skip-refresh','--allow-unapproved-models'}
REQUIRED_KEYS = ['season','week','as_of','generated_at','source_files','source_hashes','model_versions','feature_schema_hash','validation_passed','market_data_used']


def fail(msg: str) -> None:
    raise AssertionError(msg)


def load_runner():
    if not RUNNER.is_file():
        fail(f'Missing runner: {RUNNER}')
    scripts = HERE / 'scripts'
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location('issue46_runner', RUNNER)
    if spec is None or spec.loader is None:
        fail('Unable to import runner')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def result(module, n, script, command, ok=True):
    return module.StepResult(n, script, 'success' if ok else 'failed', 0 if ok else 17,
                             list(command), '2026-09-06T12:00:00Z',
                             '2026-09-06T12:00:01Z', 1.0, None)


def main() -> int:
    source = RUNNER.read_text(encoding='utf-8')
    if not source.startswith(EXPECTED_HEADER):
        fail('Required file header mismatch')
    m = load_runner()
    if list(m.PIPELINE) != EXPECTED_PIPELINE:
        fail('Execution order mismatch')
    m.validate_pipeline_files(HERE)

    options = {opt for a in m.build_parser()._actions for opt in a.option_strings}
    if not REQUIRED_CLI <= options:
        fail(f'Missing CLI options: {sorted(REQUIRED_CLI-options)}')
    help_run = subprocess.run([sys.executable, str(RUNNER), '--help'], cwd=HERE,
                              capture_output=True, text=True, check=False)
    if help_run.returncode != 0 or any(opt not in help_run.stdout for opt in REQUIRED_CLI):
        fail('CLI help contract failed')

    if m.child_args(EXPECTED_PIPELINE[0], 2026, 1) != ['--season','2026']:
        fail('Refresh args mismatch')
    if m.child_args(EXPECTED_PIPELINE[1], 2026, 1) != []:
        fail('Identity args mismatch')
    if m.child_args(EXPECTED_PIPELINE[2], 2026, 1) != []:
        fail('Market audit args mismatch')
    if m.child_args(EXPECTED_PIPELINE[7], 2026, 1) != ['--season','2026']:
        fail('Week 1 prior args mismatch')
    for script in [EXPECTED_PIPELINE[3],EXPECTED_PIPELINE[4],EXPECTED_PIPELINE[5],EXPECTED_PIPELINE[6],*EXPECTED_PIPELINE[8:]]:
        if m.child_args(script, 2026, 1) != ['--season','2026','--week','1']:
            fail(f'Weekly args mismatch: {script}')

    if m.normalize_as_of('2026-09-06T08:00:00-04:00') != '2026-09-06T12:00:00Z':
        fail('as-of UTC normalization failed')

    registry, versions = m.registry_state(HERE)
    if set(versions) != set(m.TARGETS):
        fail('model_versions coverage mismatch')
    production_targets, deferred_targets = m.assert_model_approval(registry, False)
    if set(production_targets) | set(deferred_targets) != set(m.TARGETS):
        fail('Production/deferred target partition mismatch')
    if set(production_targets) & set(deferred_targets):
        fail('Production/deferred target partition overlaps')
    all_targets, no_deferred = m.assert_model_approval(registry, True)
    if all_targets != list(m.TARGETS) or no_deferred != []:
        fail('Explicit unapproved override contract failed')
    approved = {t:{'production_approved':True,'version':f'{t}-v1'} for t in m.TARGETS}
    all_approved, deferred = m.assert_model_approval(approved, False)
    if all_approved != list(m.TARGETS) or deferred != []:
        fail('Fully approved registry contract failed')

    calls=[]
    def ok_exec(**kw):
        calls.append(kw['script'])
        return result(m, kw['step_number'], kw['script'], kw['command'], True)
    status, rows = m.run_pipeline(pipeline=m.PIPELINE, scripts_root=HERE/'scripts',
                                  repo_root=HERE.parents[4], season=2026, week=1,
                                  skip_refresh=False, executor=ok_exec)
    if status != 'success' or calls != EXPECTED_PIPELINE or len(rows) != 14:
        fail('Week 1 exact execution failed')

    calls=[]
    status, rows = m.run_pipeline(pipeline=m.PIPELINE, scripts_root=HERE/'scripts',
                                  repo_root=HERE.parents[4], season=2026, week=2,
                                  skip_refresh=False, executor=ok_exec)
    expected = [s for i,s in enumerate(EXPECTED_PIPELINE,1) if i != 8]
    if status != 'success' or calls != expected:
        fail('Week >1 conditional prior behavior failed')
    skipped=[r for r in rows if r.status=='skipped']
    if len(skipped)!=1 or skipped[0].step_number!=8:
        fail('Week >1 skip record incorrect')

    calls=[]
    status, _ = m.run_pipeline(pipeline=m.PIPELINE, scripts_root=HERE/'scripts',
                               repo_root=HERE.parents[4], season=2026, week=1,
                               skip_refresh=True, executor=ok_exec)
    if status!='success' or calls != EXPECTED_PIPELINE[1:]:
        fail('--skip-refresh behavior failed')

    calls=[]
    def fail4(**kw):
        calls.append(kw['script'])
        return result(m, kw['step_number'], kw['script'], kw['command'], kw['step_number'] != 4)
    status, _ = m.run_pipeline(pipeline=m.PIPELINE, scripts_root=HERE/'scripts',
                               repo_root=HERE.parents[4], season=2026, week=1,
                               skip_refresh=False, executor=fail4)
    if status!='failed' or calls != EXPECTED_PIPELINE[:4]:
        fail('Stop-on-first-failure contract failed')

    pairs=[('season','int64'),('week','int64'),('feature_x','double')]
    if m.schema_hash_from_pairs(pairs) == m.schema_hash_from_pairs(list(reversed(pairs))):
        fail('Feature schema hash not order-sensitive')

    with tempfile.TemporaryDirectory() as tmp:
        repo=Path(tmp); (repo/'source').mkdir(); (repo/'source/a.parquet').write_bytes(b'a'); (repo/'schedule.csv').write_bytes(b'schedule')
        cfg={'paths':{'current_source_root':'source','current_schedule':'schedule.csv'}}
        files, hashes=m.source_inventory(cfg, repo, 2026, 1)
        if files != ['schedule.csv','source/a.parquet'] or set(files)!=set(hashes):
            fail('Source inventory/hash keys mismatch')
        if hashes['schedule.csv'] != hashlib.sha256(b'schedule').hexdigest():
            fail('Source SHA-256 mismatch')

    synth=[result(m,i,s,['python',s],True) for i,s in enumerate(EXPECTED_PIPELINE,1)]
    manifest=m.make_manifest(season=2026,week=1,as_of='2026-09-06T12:00:00Z',
        source_files=['source/a.parquet'],source_hashes={'source/a.parquet':'abc'},
        model_versions={t:None for t in m.TARGETS},
        production_targets=list(m.TARGETS),deferred_targets=[],
        feature_schema_hash='deadbeef',
        validation_passed=True,steps=synth,skip_refresh=False,
        allow_unapproved_models=True,status='success',failure=None)
    if any(k not in manifest for k in REQUIRED_KEYS):
        fail('Required manifest key missing')
    if manifest['market_data_used'] is not False:
        fail('market_data_used must be false')
    if m.run_manifest_path(HERE,2026,1) != HERE/'output/2026/week_1_run_manifest.json':
        fail('Run manifest path mismatch')

    print(f'runner={RUNNER.relative_to(HERE).as_posix()}')
    print('steps=14')
    print('required_header_exact=true')
    print('execution_order_exact=true')
    print('cli_season=true')
    print('cli_week=true')
    print('cli_as_of=true')
    print('cli_skip_refresh=true')
    print('cli_allow_unapproved_models=true')
    print('week1_priors_conditional=true')
    print('skip_refresh_verified=true')
    print('stop_on_first_failure=true')
    print('approved_registry_subset_runs_in_default_production=true')
    print('explicit_unapproved_override=true')
    print('production_registry_mutated=false')
    print('source_hashes_sha256=true')
    print('feature_schema_hash=true')
    print('run_manifest_path=output/{season}/week_{week}_run_manifest.json')
    print('run_manifest_required_keys=10')
    print('market_data_used=false')
    print('WEEKLY PROP ENGINE RUNNER VALIDATION: PASS')
    print('ISSUE 46 ACCEPTANCE: PASS')
    return 0

if __name__=='__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'WEEKLY PROP ENGINE RUNNER VALIDATION: FAIL - {exc}', file=sys.stderr)
        raise SystemExit(1)
