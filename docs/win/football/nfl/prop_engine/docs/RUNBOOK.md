# NFL Prop Engine Operations Runbook

This runbook covers normal historical, training, and weekly operations for the NFL Prop Engine. Commands are intended to be run from the repository root.

The Prop Engine is football-data-only. Do not bypass market-exclusion checks, leakage checks, feature-schema checks, source-quality checks, or role-resolution hard failures to force a projection run through.

## Standard Commands

### Historical rebuild

```powershell
python docs/win/football/nfl/prop_engine/scripts/run_historical_build.py --start-season 2021 --end-season 2025
```

Inspect the newest structured runner log:

```text
docs/win/football/nfl/prop_engine/logs/historical_build_{timestamp}.json
```

A successful run completes all historical build steps and both final validation steps.

### Training

```powershell
python docs/win/football/nfl/prop_engine/scripts/run_training.py
```

Inspect:

```text
docs/win/football/nfl/prop_engine/logs/training_{timestamp}.json
```

The training runner fails closed if its required configured seed is unavailable, if validation fails, or if the production registry indicates an existing production version that may not be silently overwritten.

### Weekly

```powershell
python docs/win/football/nfl/prop_engine/scripts/run_weekly.py --season 2026 --week 1
```

Inspect:

```text
docs/win/football/nfl/prop_engine/output/2026/week_1_run_manifest.json
docs/win/football/nfl/prop_engine/output/2026/week_1_validation.json
```

The default weekly run requires production-approved model registry entries. Do not use an unapproved-model override for a production run.

### Weekly without refresh

```powershell
python docs/win/football/nfl/prop_engine/scripts/run_weekly.py --season 2026 --week 1 --skip-refresh
```

Use `--skip-refresh` only when the current-source files have already been refreshed and their freshness has been independently verified. The weekly runner still executes source-quality validation.

## Troubleshooting

The examples below use season `2026`, week `1`. Substitute the intended season/week consistently when operating another week.

### Missing current player stats

**Rerun script**

```powershell
python docs/win/football/nfl/prop_engine/scripts/build/refresh_nflverse_player_data.py --season 2026
python docs/win/football/nfl/prop_engine/scripts/validate/validate_source_quality.py --season 2026 --week 1
```

**Inspect exact output/log**

```text
docs/win/football/nfl/prop_engine/data/current/source/stats_player_week_2026.parquet
docs/win/football/nfl/prop_engine/evaluation/source_quality.csv
```

In `source_quality.csv`, inspect the `player_stats` row for season 2026/week 1. For Week 1, zero current-season realized player-stat rows are allowed; for later weeks the lagged source must be fresh through the required prior week.

### Missing snap counts

**Rerun script**

```powershell
python docs/win/football/nfl/prop_engine/scripts/build/refresh_nflverse_player_data.py --season 2026
python docs/win/football/nfl/prop_engine/scripts/validate/validate_source_quality.py --season 2026 --week 1
```

**Inspect exact output/log**

```text
docs/win/football/nfl/prop_engine/data/current/source/snap_counts_2026.parquet
docs/win/football/nfl/prop_engine/evaluation/source_quality.csv
```

Inspect the `snap_counts` row. Week N features may use only snap observations from weeks `< N`.

### Missing participation

**Rerun script**

```powershell
python docs/win/football/nfl/prop_engine/scripts/build/refresh_nflverse_player_data.py --season 2026
python docs/win/football/nfl/prop_engine/scripts/validate/validate_source_quality.py --season 2026 --week 1
```

**Inspect exact output/log**

```text
docs/win/football/nfl/prop_engine/data/current/source/pbp_participation_2026.parquet
docs/win/football/nfl/prop_engine/evaluation/source_quality.csv
```

Inspect the `participation` row. Participation is a lagged realized source; target-week participation is forbidden.

### Unresolved player ID

**Rerun script**

```powershell
python docs/win/football/nfl/prop_engine/scripts/build/build_player_identity.py
python docs/win/football/nfl/prop_engine/scripts/project/build_current_universe.py --season 2026 --week 1
```

**Inspect exact output/log**

```text
docs/win/football/nfl/prop_engine/data/identity/player_crosswalk.parquet
docs/win/football/nfl/prop_engine/logs/build_player_identity.json
docs/win/football/nfl/prop_engine/logs/current_universe_2026_week_1.json
```

GSIS is authoritative. Do not resolve an ambiguous player by display-name guess. The current-universe log identifies critical unresolved players that prevent safe projection.

### Missing depth chart

**Rerun script**

```powershell
python docs/win/football/nfl/prop_engine/scripts/validate/validate_source_quality.py --season 2026 --week 1
python docs/win/football/nfl/prop_engine/scripts/project/build_current_universe.py --season 2026 --week 1
```

**Inspect exact output/log**

```text
docs/win/football/nfl/prop_engine/evaluation/source_quality.csv
docs/win/football/nfl/prop_engine/logs/current_universe_2026_week_1.json
docs/win/football/nfl/prop_engine/data/current/2026_week_1_universe.parquet
```

The configured external depth source is `docs/win/football/nfl/data/master/depth_charts/`. Restore/refresh the affected team depth file before rerunning the current universe. Missing depth must remain explicit; do not invent starter rank.

### Missing injury file

**Rerun script**

```powershell
python docs/win/football/nfl/prop_engine/scripts/validate/validate_source_quality.py --season 2026 --week 1
python docs/win/football/nfl/prop_engine/scripts/project/build_current_universe.py --season 2026 --week 1
```

**Inspect exact output/log**

```text
docs/win/football/nfl/00_intake/injuries/2026_injuries.csv
docs/win/football/nfl/prop_engine/evaluation/source_quality.csv
docs/win/football/nfl/prop_engine/logs/current_universe_2026_week_1.json
```

A missing injury source is not equivalent to healthy. Restore the source, rerun source-quality validation, then rebuild the current universe.

### Missing weather

**Rerun script**

```powershell
python docs/win/football/nfl/prop_engine/scripts/validate/validate_source_quality.py --season 2026 --week 1
python docs/win/football/nfl/prop_engine/scripts/project/build_current_features.py --season 2026 --week 1
```

**Inspect exact output/log**

```text
docs/win/football/nfl/data/weather/week_1_NFL_weekly_weather.csv
docs/win/football/nfl/prop_engine/evaluation/source_quality.csv
docs/win/football/nfl/prop_engine/logs/current_features_2026_week_1.json
docs/win/football/nfl/prop_engine/data/current/features/2026_week_1_feature_manifest.json
```

Missing weather must remain explicit through `environment_weather_missing_flag`; do not copy weather from another game or a previous week.

### Missing travel

**Rerun script**

```powershell
python docs/win/football/nfl/prop_engine/scripts/validate/validate_source_quality.py --season 2026 --week 1
python docs/win/football/nfl/prop_engine/scripts/project/build_current_features.py --season 2026 --week 1
```

**Inspect exact output/log**

```text
docs/win/football/nfl/data/travel/2026_week_1_travel.csv
docs/win/football/nfl/prop_engine/evaluation/source_quality.csv
docs/win/football/nfl/prop_engine/logs/current_features_2026_week_1.json
docs/win/football/nfl/prop_engine/data/current/features/2026_week_1_feature_manifest.json
```

Missing travel must remain explicit through `environment_travel_missing_flag`; do not fabricate miles or time-zone crossings.

### Feature schema mismatch

**Rerun script**

```powershell
python docs/win/football/nfl/prop_engine/scripts/project/build_current_features.py --season 2026 --week 1
```

**Inspect exact output/log**

```text
docs/win/football/nfl/prop_engine/logs/current_features_2026_week_1.json
docs/win/football/nfl/prop_engine/data/current/features/2026_week_1_feature_manifest.json
docs/win/football/nfl/prop_engine/data/historical/features/feature_manifest.json
docs/win/football/nfl/prop_engine/data/current/features/2026_week_1_features.parquet
```

The current feature schema must match the canonical historical feature contract. Do not drop a required model feature or add an undocumented feature merely to make the projection script run.

### Out player still projected

**Rerun script**

```powershell
python docs/win/football/nfl/prop_engine/scripts/project/project_week.py --season 2026 --week 1
python docs/win/football/nfl/prop_engine/scripts/validate/validate_week.py --season 2026 --week 1
```

**Inspect exact output/log**

```text
docs/win/football/nfl/prop_engine/output/2026/week_1_player_projections.csv
docs/win/football/nfl/prop_engine/output/2026/week_1_active_player_projections.csv
docs/win/football/nfl/prop_engine/output/2026/week_1_validation.json
```

The long audit output may retain ineligible rows for traceability. The active-only output must contain no ineligible/Out projections. If the active file still contains the player, fix the upstream eligibility/role state rather than filtering the validator.

### Multiple starting quarterbacks

**Rerun script**

```powershell
python docs/win/football/nfl/prop_engine/scripts/project/select_roles.py --season 2026 --week 1
```

**Inspect exact output/log**

```text
docs/win/football/nfl/prop_engine/logs/current_roles_2026_week_1.json
docs/win/football/nfl/prop_engine/data/current/2026_week_1_roles.parquet
```

Exactly one primary QB is required per scheduled team. A real unresolved best-depth-rank tie is a hard failure. Resolve the depth/eligibility input; do not choose one arbitrarily.

### Multiple kickers

**Rerun script**

```powershell
python docs/win/football/nfl/prop_engine/scripts/project/select_roles.py --season 2026 --week 1
```

**Inspect exact output/log**

```text
docs/win/football/nfl/prop_engine/logs/current_roles_2026_week_1.json
docs/win/football/nfl/prop_engine/data/current/2026_week_1_roles.parquet
```

Exactly one primary kicker is emitted per scheduled team. Kicker selection uses current depth plus strictly prior realized kick attempts. Genuine two-kicker ambiguity is retained in the log with reduced confidence/uncertainty widening; it must not use target-game attempts.

### Market-feature rejection

**Rerun script**

```powershell
python docs/win/football/nfl/prop_engine/scripts/validate/audit_market_exclusion.py
```

**Inspect exact output/log**

```text
docs/win/football/nfl/prop_engine/evaluation/market_exclusion_audit.json
```

Remove the forbidden feature/source reference at its origin. Do not weaken the deny list, rename a sportsbook/market feature to conceal it, or bypass the audit. The expected successful marker is `MARKET EXCLUSION AUDIT: PASS`.

### Historical leakage failure

**Rerun script**

```powershell
python docs/win/football/nfl/prop_engine/scripts/validate/validate_historical_data.py
```

**Inspect exact output/log**

```text
docs/win/football/nfl/prop_engine/evaluation/historical_validation.json
docs/win/football/nfl/prop_engine/data/historical/features/feature_manifest.json
docs/win/football/nfl/prop_engine/data/historical/features/player_game_features.parquet
```

Trace the failed feature family to its upstream builder. Week N player, snap, participation, team/opponent, and position-allowed features must exclude Week N realized information; depth and injury state must be pre-kickoff. Fix the upstream builder and rerun the minimum necessary historical dependency chain before validating again. Never weaken the leakage criterion.

## Weekly Failure Triage Order

For a failed weekly run, inspect in this order:

1. `output/{season}/week_{week}_run_manifest.json` for the failed numbered step.
2. `evaluation/source_quality.csv` for missing/stale current inputs.
3. `evaluation/market_exclusion_audit.json` for forbidden-source/feature findings.
4. `logs/current_universe_{season}_week_{week}.json` for identity/eligibility problems.
5. `logs/current_roles_{season}_week_{week}.json` for QB/kicker role problems.
6. `logs/current_features_{season}_week_{week}.json` and the current feature manifest for schema/as-of problems.
7. `output/{season}/week_{week}_validation.json` for final projection constraints.

The weekly runner stops on the first failed stage. Correct that stage or its upstream source, then rerun only the required dependency chain.
