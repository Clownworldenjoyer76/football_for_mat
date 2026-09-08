# NFL Prop Engine Operations Runbook

## Scope

Run all commands from:

`C:\Users\Mat\Documents\GitHub\football_for_mat`

Prop Engine root:

`docs/win/football/nfl/prop_engine/`

The Prop Engine is football-data-only. Do not bypass market-exclusion checks. Do not use sportsbook, odds, moneyline, spread, totals, market-implied projections, DRAT, EPRED, consensus projections, or other prohibited market-derived inputs.

Production runs fail closed on source, identity, feature, leakage, model-approval, or validation failures.

## Environment

Install isolated dependencies:

```powershell
python -m pip install -r docs/win/football/nfl/prop_engine/requirements.txt
```

Configuration:

`docs/win/football/nfl/prop_engine/config/prop_engine.yaml`

Data contract:

`docs/win/football/nfl/prop_engine/docs/DATA_CONTRACT.md`

Feature dictionary:

`docs/win/football/nfl/prop_engine/docs/FEATURE_DICTIONARY.md`

## Historical build

Historical runner:

`docs/win/football/nfl/prop_engine/scripts/run_historical_build.py`

Issue 51 accepted command:

```powershell
python docs/win/football/nfl/prop_engine/scripts/run_historical_build.py --start-season 2021 --end-season 2025
```

Full configured rebuild:

```powershell
python docs/win/football/nfl/prop_engine/scripts/run_historical_build.py --start-season 2012 --end-season 2025 --force
```

Runner log:

`docs/win/football/nfl/prop_engine/logs/historical_build_{timestamp}.json`

Primary historical outputs:

- `docs/win/football/nfl/prop_engine/data/identity/player_crosswalk.parquet`
- `docs/win/football/nfl/prop_engine/data/historical/universe/player_game_universe.parquet`
- `docs/win/football/nfl/prop_engine/data/historical/targets/player_game_targets.parquet`
- `docs/win/football/nfl/prop_engine/data/historical/features/player_game_features.parquet`
- `docs/win/football/nfl/prop_engine/data/historical/features/feature_manifest.json`

A failed historical build blocks training.

## Validators

Run cheap/static validators before expensive training.

Config enforcement:

```powershell
python docs/win/football/nfl/prop_engine/validate_config_enforcement.py
```

Common-utility enforcement:

```powershell
python docs/win/football/nfl/prop_engine/validate_common_usage.py
```

Final data contracts:

```powershell
python docs/win/football/nfl/prop_engine/validate_final_data_contracts.py
```

Market-exclusion preflight:

```powershell
python docs/win/football/nfl/prop_engine/scripts/validate/audit_market_exclusion.py --preflight
```

Full market audit:

```powershell
python docs/win/football/nfl/prop_engine/scripts/validate/audit_market_exclusion.py
```

Market audit output:

`docs/win/football/nfl/prop_engine/evaluation/market_exclusion_audit.json`

Historical validation:

```powershell
python docs/win/football/nfl/prop_engine/scripts/validate/validate_historical_data.py
```

Historical validation output:

`docs/win/football/nfl/prop_engine/evaluation/historical_validation.json`

Never weaken the leakage criterion.

Unit/integration/E2E acceptance:

```powershell
python docs/win/football/nfl/prop_engine/validate_issue47.py
```

Runbook acceptance:

```powershell
python docs/win/football/nfl/prop_engine/validate_issue51.py
```

Expected Item 51 result:

```text
OPERATIONS RUNBOOK VALIDATION: PASS
ISSUE 51 ACCEPTANCE: PASS
```

## Training

Training runner:

`docs/win/football/nfl/prop_engine/scripts/run_training.py`

Exact command:

```powershell
python docs/win/football/nfl/prop_engine/scripts/run_training.py
```

The runner takes no CLI arguments and executes the configured training pipeline fail-fast.

Training log:

`docs/win/football/nfl/prop_engine/logs/training_{timestamp}.json`

The training runner must finish with `status = success`.

Do not start full training until all cheap/static preflights pass.

## Production approval

Evaluator:

`docs/win/football/nfl/prop_engine/evaluate_production_approval.py`

Exact command:

```powershell
python docs/win/football/nfl/prop_engine/evaluate_production_approval.py --test-season 2025
```

Acceptance thresholds:

`docs/win/football/nfl/prop_engine/config/acceptance_thresholds.yaml`

Inputs:

- `docs/win/football/nfl/prop_engine/evaluation/model_selection_predictions.parquet`
- `docs/win/football/nfl/prop_engine/evaluation/source_quality.csv`
- `docs/win/football/nfl/prop_engine/evaluation/interval_coverage.csv`

Outputs:

- `docs/win/football/nfl/prop_engine/evaluation/production_approval_evaluation.csv`
- `docs/win/football/nfl/prop_engine/evaluation/production_approval_evaluation.json`

All nine targets must pass before production registry approval.

Do not lower acceptance thresholds to manufacture a pass.

Production registry:

`docs/win/football/nfl/prop_engine/models/production_registry.json`

A true production run requires all nine entries to have `production_approved = true` and an explicit nonblank `version`.

## Weekly projection

Weekly runner:

`docs/win/football/nfl/prop_engine/scripts/run_weekly.py`

Exact standard command:

```powershell
python docs/win/football/nfl/prop_engine/scripts/run_weekly.py --season 2026 --week 1
```

Accepted no-refresh development rerun:

```powershell
python docs/win/football/nfl/prop_engine/scripts/run_weekly.py --season 2026 --week 1 --skip-refresh
```

`--allow-unapproved-models` is development-only. Do not use an unapproved-model override for a production run.

Weekly source-quality output:

`docs/win/football/nfl/prop_engine/evaluation/source_quality.csv`

Weekly validation:

`docs/win/football/nfl/prop_engine/output/2026/week_1_validation.json`

Weekly run manifest:

`docs/win/football/nfl/prop_engine/output/2026/week_1_run_manifest.json`

Current-week operational logs:

- `docs/win/football/nfl/prop_engine/logs/current_universe_2026_week_1.json`
- `docs/win/football/nfl/prop_engine/logs/current_roles_2026_week_1.json`
- `docs/win/football/nfl/prop_engine/logs/current_features_2026_week_1.json`

A final production manifest must report:

- `validation_passed = true`
- `market_data_used = false`
- `allow_unapproved_models = false`
- `status = success`
- explicit model versions for all nine targets

## Troubleshooting

### Missing current player stats

**Rerun script**

`docs/win/football/nfl/prop_engine/scripts/build/refresh_nflverse_player_data.py`

**Inspect exact output/log**

`docs/win/football/nfl/prop_engine/data/current/source/stats_player_week_2026.parquet`

### Missing snap counts

**Rerun script**

`docs/win/football/nfl/prop_engine/scripts/build/refresh_nflverse_player_data.py`

**Inspect exact output/log**

`docs/win/football/nfl/prop_engine/data/current/source/snap_counts_2026.parquet`

### Missing participation

**Rerun script**

`docs/win/football/nfl/prop_engine/scripts/build/refresh_nflverse_player_data.py`

**Inspect exact output/log**

`docs/win/football/nfl/prop_engine/data/current/source/pbp_participation_2026.parquet`

### Unresolved player ID

**Rerun script**

`docs/win/football/nfl/prop_engine/scripts/build/build_player_identity.py`

**Inspect exact output/log**

`docs/win/football/nfl/prop_engine/logs/current_universe_2026_week_1.json`

### Missing depth chart

**Rerun script**

`docs/win/football/nfl/prop_engine/scripts/project/build_current_universe.py`

**Inspect exact output/log**

`docs/win/football/nfl/prop_engine/logs/current_universe_2026_week_1.json`

### Missing injury file

**Rerun script**

`docs/win/football/nfl/prop_engine/scripts/project/build_current_universe.py`

**Inspect exact output/log**

`docs/win/football/nfl/prop_engine/evaluation/source_quality.csv`

### Missing weather

**Rerun script**

`docs/win/football/nfl/prop_engine/scripts/project/build_current_features.py`

**Inspect exact output/log**

`docs/win/football/nfl/prop_engine/logs/current_features_2026_week_1.json`

### Missing travel

**Rerun script**

`docs/win/football/nfl/prop_engine/scripts/project/build_current_features.py`

**Inspect exact output/log**

`docs/win/football/nfl/prop_engine/logs/current_features_2026_week_1.json`

### Feature schema mismatch

**Rerun script**

`docs/win/football/nfl/prop_engine/scripts/project/build_current_features.py`

**Inspect exact output/log**

`docs/win/football/nfl/prop_engine/data/current/features/2026_week_1_feature_manifest.json`

### Out player still projected

**Rerun script**

`docs/win/football/nfl/prop_engine/scripts/validate/validate_week.py`

**Inspect exact output/log**

`docs/win/football/nfl/prop_engine/output/2026/week_1_validation.json`

### Multiple starting quarterbacks

**Rerun script**

`docs/win/football/nfl/prop_engine/scripts/project/select_roles.py`

**Inspect exact output/log**

`docs/win/football/nfl/prop_engine/logs/current_roles_2026_week_1.json`

If multiple quarterbacks remain unresolved, do not choose one arbitrarily.

### Multiple kickers

**Rerun script**

`docs/win/football/nfl/prop_engine/scripts/project/select_roles.py`

**Inspect exact output/log**

`docs/win/football/nfl/prop_engine/logs/current_roles_2026_week_1.json`

### Market-feature rejection

**Rerun script**

`docs/win/football/nfl/prop_engine/scripts/validate/audit_market_exclusion.py`

**Inspect exact output/log**

`docs/win/football/nfl/prop_engine/evaluation/market_exclusion_audit.json`

Do not bypass market-exclusion checks.

### Historical leakage failure

**Rerun script**

`docs/win/football/nfl/prop_engine/scripts/validate/validate_historical_data.py`

**Inspect exact output/log**

`docs/win/football/nfl/prop_engine/evaluation/historical_validation.json`

Never weaken the leakage criterion.
