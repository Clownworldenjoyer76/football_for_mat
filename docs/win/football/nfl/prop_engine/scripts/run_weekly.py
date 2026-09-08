#!/usr/bin/env python3
"""
Run the complete NFL Prop Engine weekly projection pipeline.

READS:
    Existing repository football-only schedule, PBP, team stats, roster,
    depth, injuries, weather, travel, and nflverse data.

WRITES:
    docs/win/football/nfl/prop_engine/ only.

NO SPORTSBOOK OR MARKET DATA IS READ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
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
    'build/refresh_nflverse_player_data.py',
    'build/build_player_identity.py',
    'validate/audit_market_exclusion.py',
    'validate/validate_source_quality.py',
    'project/build_current_universe.py',
    'project/select_roles.py',
    # WEEK1_FEATURES_BEFORE_PRIORS_SEQUENCE
    # Week 1 priors consume the freshly materialized current-feature frame,
    # so features must be rebuilt from the current universe/roles first.
    'project/build_current_features.py',
    'project/build_week1_priors.py',
    'project/project_components.py',
    'project/allocate_team_opportunity.py',
    'project/project_direct.py',
    'project/project_week.py',
    'report/build_wide_output.py',
    'validate/validate_week.py',
)

TARGETS = list(_CONFIG_CONTRACT["targets"].keys())

# SIX_TARGET_PRODUCTION_REGISTRY_MODE
REQUIRED_MANIFEST_KEYS = (
    'season', 'week', 'as_of', 'generated_at', 'source_files', 'source_hashes',
    'model_versions', 'production_model_versions', 'production_targets',
    'deferred_targets', 'feature_schema_hash', 'validation_passed',
    'market_data_used', 'allow_unapproved_models', 'status',
)

SEASON_ONLY = frozenset({
    'build/refresh_nflverse_player_data.py',
    'project/build_week1_priors.py',
})
WEEKLY = frozenset({
    'validate/validate_source_quality.py',
    'project/build_current_universe.py',
    'project/select_roles.py',
    'project/build_current_features.py',
    'project/project_components.py',
    'project/allocate_team_opportunity.py',
    'project/project_direct.py',
    'project/project_week.py',
    'report/build_wide_output.py',
    'validate/validate_week.py',
})

@dataclass
class StepResult:
    step_number: int
    script: str
    status: str
    exit_code: int | None
    command: list[str]
    started_at: str | None
    ended_at: str | None
    duration_seconds: float | None
    skip_reason: str | None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(value: datetime | None = None) -> str:
    return (value or utc_now()).astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')


def normalize_as_of(value: str | None) -> str:
    if value is None:
        return iso_utc()
    text = str(value).strip()
    if not text:
        raise ValueError('--as-of cannot be blank.')
    parsed_text = text[:-1] + '+00:00' if text.endswith(('Z', 'z')) else text
    try:
        parsed = datetime.fromisoformat(parsed_text)
    except ValueError as exc:
        raise ValueError('--as-of must be an ISO-8601 date/time.') from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return iso_utc(parsed)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Run the complete NFL Prop Engine weekly projection pipeline.')
    p.add_argument('--season', type=int, required=True)
    p.add_argument('--week', type=int, required=True)
    p.add_argument('--as-of', default=None)
    p.add_argument('--skip-refresh', action='store_true')
    p.add_argument('--allow-unapproved-models', action='store_true')
    return p


def validate_season_week(season: int, week: int) -> None:
    if not 1900 <= int(season) <= 2200:
        raise ValueError(f'Invalid season: {season}')
    if not 1 <= int(week) <= 22:
        raise ValueError(f'Invalid NFL week: {week}')


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f'Required JSON missing: {path}')
    with path.open('r', encoding='utf-8-sig') as fh:
        value = json.load(fh)
    if not isinstance(value, dict):
        raise ValueError(f'Expected JSON object: {path}')
    return value


def registry_state(prop_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    registry = load_json(prop_root / 'models/production_registry.json')
    if list(registry.keys()) != list(TARGETS):
        raise ValueError('Production registry must contain exactly the nine Prop Engine targets.')
    versions: dict[str, Any] = {}
    for target in TARGETS:
        entry = registry[target]
        if not isinstance(entry, dict) or 'production_approved' not in entry or 'version' not in entry:
            raise ValueError(f'Invalid production registry entry for {target}.')
        versions[target] = entry['version']
    return registry, versions


def assert_model_approval(
    registry: dict[str, Any],
    allow_unapproved_models: bool,
) -> tuple[list[str], list[str]]:
    if allow_unapproved_models:
        return list(TARGETS), []

    approved: list[str] = []
    deferred: list[str] = []
    for target in TARGETS:
        entry = registry[target]
        approved_flag = entry.get('production_approved')
        version = entry.get('version')
        if approved_flag is True and isinstance(version, str) and version.strip():
            approved.append(target)
        elif approved_flag is False and version is None:
            deferred.append(target)
        else:
            raise RuntimeError(
                f'Invalid production registry state for {target}: '
                f'production_approved={approved_flag!r} version={version!r}'
            )
    if not approved:
        raise RuntimeError('Production weekly run requires at least one approved/versioned target.')
    return approved, deferred


def validate_pipeline_files(prop_root: Path, pipeline: Sequence[str] = PIPELINE) -> None:
    missing = [s for s in pipeline if not (prop_root / 'scripts' / s).is_file()]
    if missing:
        raise FileNotFoundError('Weekly runner missing required script(s): ' + ', '.join(missing))


def child_args(script: str, season: int, week: int) -> list[str]:
    if script in SEASON_ONLY:
        return ['--season', str(int(season))]
    if script in WEEKLY:
        return ['--season', str(int(season)), '--week', str(int(week))]
    return []


def command_for_step(script: str, scripts_root: Path, season: int, week: int) -> list[str]:
    return [sys.executable, str((scripts_root / script).resolve()), *child_args(script, season, week)]


def should_skip_step(script: str, week: int, skip_refresh: bool) -> str | None:
    if script == 'build/refresh_nflverse_player_data.py' and skip_refresh:
        return '--skip-refresh'
    if script == 'project/build_week1_priors.py' and int(week) != 1:
        return 'week1_priors_not_required'
    return None


def execute_step(*, step_number: int, script: str, command: list[str], repo_root: Path) -> StepResult:
    started = utc_now()
    clock = time.perf_counter()
    completed = subprocess.run(command, cwd=repo_root, check=False)
    ended = utc_now()
    return StepResult(
        step_number, script, 'success' if completed.returncode == 0 else 'failed',
        int(completed.returncode), command, iso_utc(started), iso_utc(ended),
        round(time.perf_counter() - clock, 6), None,
    )

Executor = Callable[..., StepResult]


def run_pipeline(*, pipeline: Sequence[str], scripts_root: Path, repo_root: Path,
                 season: int, week: int, skip_refresh: bool,
                 executor: Executor = execute_step) -> tuple[str, list[StepResult]]:
    results: list[StepResult] = []
    for number, script in enumerate(pipeline, start=1):
        command = command_for_step(script, scripts_root, season, week)
        reason = should_skip_step(script, week, skip_refresh)
        if reason is not None:
            print(f'[{number:02d}/{len(pipeline):02d}] {script} SKIP ({reason})')
            results.append(StepResult(number, script, 'skipped', None, command, None, None, None, reason))
            continue
        print(f'[{number:02d}/{len(pipeline):02d}] {script}')
        result = executor(step_number=number, script=script, command=command, repo_root=repo_root)
        results.append(result)
        if result.status != 'success' or result.exit_code != 0:
            print(f'WEEKLY PIPELINE STOPPED: step={number} script={script} exit_code={result.exit_code}', file=sys.stderr)
            return 'failed', results
    return 'success', results


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _fmt(value: str, season: int, week: int) -> str:
    return str(value).format(season=season, week=week)


def collect_source_paths(config: dict[str, Any], repo_root: Path, season: int, week: int) -> list[Path]:
    paths = config.get('paths')
    if not isinstance(paths, dict):
        raise ValueError('Config paths section must be a mapping.')
    files: set[Path] = set()
    for key in ('pbp_pattern', 'team_stats_pattern', 'current_schedule', 'current_roster',
                'current_injuries', 'current_weather', 'current_travel'):
        value = paths.get(key)
        if value:
            candidate = repo_root / _fmt(str(value), season, week)
            if candidate.is_file():
                files.add(candidate.resolve())
    for key in ('current_source_root', 'current_depth_root'):
        value = paths.get(key)
        if value:
            directory = repo_root / _fmt(str(value), season, week)
            if directory.is_dir():
                for candidate in directory.rglob('*'):
                    if candidate.is_file() and '__pycache__' not in candidate.parts:
                        files.add(candidate.resolve())
    return sorted(files, key=lambda p: str(p).casefold())


def source_inventory(config: dict[str, Any], repo_root: Path, season: int, week: int) -> tuple[list[str], dict[str, str]]:
    relative: list[str] = []
    hashes: dict[str, str] = {}
    for path in collect_source_paths(config, repo_root, season, week):
        rel = str(path.relative_to(repo_root)).replace('\\', '/')
        relative.append(rel)
        hashes[rel] = sha256_file(path)
    return relative, hashes


def schema_hash_from_pairs(pairs: Sequence[tuple[str, str]]) -> str:
    payload = [{'name': str(name), 'type': str(dtype)} for name, dtype in pairs]
    return hashlib.sha256(json.dumps(payload, separators=(',', ':'), ensure_ascii=False).encode('utf-8')).hexdigest()


def current_feature_schema_hash(prop_root: Path, season: int, week: int) -> str | None:
    path = prop_root / 'data/current/features' / f'{season}_week_{week}_features.parquet'
    if not path.is_file():
        return None
    import pyarrow.parquet as pq
    schema = pq.read_schema(path)
    return schema_hash_from_pairs([(field.name, str(field.type)) for field in schema])


def validation_report_path(prop_root: Path, season: int, week: int) -> Path:
    return prop_root / 'output' / str(season) / f'week_{week}_validation.json'


def run_manifest_path(prop_root: Path, season: int, week: int) -> Path:
    return prop_root / 'output' / str(season) / f'week_{week}_run_manifest.json'


def write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    destination = path.resolve()
    root = common.prop_root().resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError(f'Refusing write outside Prop Engine: {destination}') from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    fh = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n',
                                     prefix=f'.{destination.name}.', suffix='.tmp',
                                     dir=destination.parent, delete=False)
    temp = Path(fh.name)
    try:
        with fh:
            json.dump(payload, fh, indent=2, sort_keys=False, ensure_ascii=False)
            fh.write('\n')
        os.replace(temp, destination)
    finally:
        if temp.exists():
            temp.unlink()


def make_manifest(*, season: int, week: int, as_of: str, source_files: list[str],
                  source_hashes: dict[str, str], model_versions: dict[str, Any],
                  production_targets: list[str], deferred_targets: list[str],
                  feature_schema_hash: str | None, validation_passed: bool,
                  steps: Sequence[StepResult], skip_refresh: bool,
                  allow_unapproved_models: bool, status: str,
                  failure: str | None) -> dict[str, Any]:
    production_model_versions = {
        target: model_versions[target]
        for target in production_targets
    }
    manifest = {
        'season': int(season),
        'week': int(week),
        'as_of': as_of,
        'generated_at': iso_utc(),
        'source_files': source_files,
        'source_hashes': source_hashes,
        'model_versions': model_versions,
        'production_model_versions': production_model_versions,
        'production_targets': list(production_targets),
        'deferred_targets': list(deferred_targets),
        'feature_schema_hash': feature_schema_hash,
        'validation_passed': bool(validation_passed),
        'market_data_used': False,
        'status': status,
        'skip_refresh': bool(skip_refresh),
        'allow_unapproved_models': bool(allow_unapproved_models),
        'failure': failure,
        'steps': [asdict(step) for step in steps],
    }
    missing = [key for key in REQUIRED_MANIFEST_KEYS if key not in manifest]
    if missing:
        raise AssertionError(f'Run manifest missing required key(s): {missing}')
    if manifest['market_data_used'] is not False:
        raise AssertionError('market_data_used must be false.')
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_season_week(args.season, args.week)
    as_of = normalize_as_of(args.as_of)
    config = common.load_config()
    repo_root = common.repo_root().resolve()
    prop_root = common.prop_root().resolve()
    scripts_root = (prop_root / 'scripts').resolve()
    destination = run_manifest_path(prop_root, args.season, args.week)

    steps: list[StepResult] = []
    validation_passed = False
    failure: str | None = None
    model_versions = {target: None for target in TARGETS}
    production_targets: list[str] = []
    deferred_targets: list[str] = []

    try:
        validate_pipeline_files(prop_root)
        registry, model_versions = registry_state(prop_root)
        production_targets, deferred_targets = assert_model_approval(
            registry, args.allow_unapproved_models
        )
        status, steps = run_pipeline(
            pipeline=PIPELINE, scripts_root=scripts_root, repo_root=repo_root,
            season=args.season, week=args.week, skip_refresh=args.skip_refresh,
        )
        if status == 'success':
            report = validation_report_path(prop_root, args.season, args.week)
            if not report.is_file():
                raise FileNotFoundError(f'Required weekly validation report missing: {report}')
            report_payload = load_json(report)
            if report_payload.get('status') != 'passed':
                raise RuntimeError(
                    f'Weekly validation report status is not passed: {report_payload.get("status")!r}'
                )
            validation_passed = True
        else:
            failed = next((s for s in steps if s.status == 'failed'), None)
            failure = f'step {failed.step_number} {failed.script} exit_code={failed.exit_code}' if failed else 'weekly pipeline failed'
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
        print(f'WEEKLY PIPELINE: FAIL - {failure}', file=sys.stderr)
        try:
            _, model_versions = registry_state(prop_root)
        except Exception:
            pass

    try:
        source_files, source_hashes = source_inventory(config, repo_root, args.season, args.week)
    except Exception as exc:
        source_files, source_hashes = [], {}
        validation_passed = False
        failure = failure or f'source inventory failed: {type(exc).__name__}: {exc}'

    try:
        feature_hash = current_feature_schema_hash(prop_root, args.season, args.week)
    except Exception as exc:
        feature_hash = None
        validation_passed = False
        failure = failure or f'feature schema hash failed: {type(exc).__name__}: {exc}'

    manifest = make_manifest(
        season=args.season, week=args.week, as_of=as_of,
        source_files=source_files, source_hashes=source_hashes,
        model_versions=model_versions,
        production_targets=production_targets,
        deferred_targets=deferred_targets,
        feature_schema_hash=feature_hash,
        validation_passed=validation_passed, steps=steps,
        skip_refresh=args.skip_refresh,
        allow_unapproved_models=args.allow_unapproved_models,
        status='success' if validation_passed else 'failed', failure=failure,
    )
    write_json_atomic(manifest, destination)
    print(f'run_manifest={destination}')
    if validation_passed:
        print('WEEKLY PROP ENGINE: PASS')
        return 0
    print('WEEKLY PROP ENGINE: FAIL', file=sys.stderr)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
