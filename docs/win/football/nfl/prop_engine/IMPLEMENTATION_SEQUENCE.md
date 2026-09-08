# Prop Engine exact implementation execution sequence

This document is the authoritative implementation and acceptance order for Issue 55.

## Rules

1. Execute the numbered stages in ascending order.
2. A later stage must not be used to bypass an earlier validation or approval gate.
3. Historical feature building must use football-only verified sources and the Issue 54 forbidden-input contract.
4. Training and model selection must remain chronological and leakage-safe.
5. Development weekly projection may use `--allow-unapproved-models` only at stage 55.54.
6. Production weekly projection at stage 55.57 must run without the unapproved-model override.
7. Source-dependent 2026 nflverse families may remain explicitly unavailable until nflverse publishes them; no unavailable rows may be fabricated.

## Exact implementation sequence

- **55.1** Create directory structure.
- **55.2** Create `config/prop_engine.yaml`.
- **55.3** Create `scripts/common.py`.
- **55.4** Create `config/target_eligibility.yaml`.
- **55.5** Create `config/fallback_rules.yaml`.
- **55.6** Create `config/acceptance_thresholds.yaml`.
- **55.7** Create all nine `config/features/*.json`.
- **55.8** Create `build/build_player_identity.py`.
- **55.9** Create `build/refresh_nflverse_player_data.py`.
- **55.10** Create `build/build_historical_universe.py`.
- **55.11** Create `build/build_targets.py`.
- **55.12** Create `build/build_player_opportunity.py`.
- **55.13** Create `build/build_team_opportunity.py`.
- **55.14** Create `build/build_position_allowed.py`.
- **55.15** Create `build/build_role_history.py`.
- **55.16** Create `build/build_player_form.py`.
- **55.17** Create `build/build_team_form.py`.
- **55.18** Create `build/build_environment_history.py`.
- **55.19** Create `build/build_defensive_features.py`.
- **55.20** Create `build/build_kicking_features.py`.
- **55.21** Create `build/build_historical_features.py`.
- **55.22** Create market/source/historical validators.
- **55.23** Create `run_historical_build.py`.
- **55.24** Run historical build and resolve every validation failure.
- **55.25** Create backtest folds.
- **55.26** Create baseline training.
- **55.27** Create opportunity training.
- **55.28** Create efficiency training.
- **55.29** Create direct training.
- **55.30** Create architecture selection.
- **55.31** Create uncertainty calibration.
- **55.32** Create model report.
- **55.33** Create `run_training.py`.
- **55.34** Run chronological backtests.
- **55.35** Populate numerical acceptance thresholds.
- **55.36** Approve only models passing baseline and calibration gates.
- **55.37** Create production registry.
- **55.38** Create Week 1 priors.
- **55.39** Create current universe.
- **55.40** Create role selection.
- **55.41** Create current features.
- **55.42** Create component projections.
- **55.43** Create opportunity allocation.
- **55.44** Create direct projections.
- **55.45** Create final projection assembly.
- **55.46** Create wide output.
- **55.47** Create weekly validation.
- **55.48** Create `run_weekly.py`.
- **55.49** Create all tests.
- **55.50** Create requirements and documentation.
- **55.51** Run unit tests.
- **55.52** Run end-to-end historical smoke test.
- **55.53** Run market exclusion audit.
- **55.54** Run one development weekly projection with `--allow-unapproved-models`.
- **55.55** Inspect generated QB, kicker, offensive-role, and defensive-role tables.
- **55.56** Approve individual targets only after thresholds pass.
- **55.57** Run first production weekly projection without unapproved-model override.

## Historical build runner order

`docs/win/football/nfl/prop_engine/scripts/run_historical_build.py` must execute:

1. `build/build_player_identity.py`
2. `build/build_historical_universe.py`
3. `build/build_targets.py`
4. `build/build_player_opportunity.py`
5. `build/build_team_opportunity.py`
6. `build/build_position_allowed.py`
7. `build/build_role_history.py`
8. `build/build_player_form.py`
9. `build/build_team_form.py`
10. `build/build_environment_history.py`
11. `build/build_defensive_features.py`
12. `build/build_kicking_features.py`
13. `build/build_historical_features.py`
14. `validate/audit_market_exclusion.py`
15. `validate/validate_historical_data.py`

## Training runner order

`docs/win/football/nfl/prop_engine/scripts/run_training.py` must execute:

1. `validate/audit_market_exclusion.py`
2. `validate/validate_historical_data.py`
3. `train/build_backtest_folds.py`
4. `train/train_baselines.py`
5. `train/train_opportunity_models.py`
6. `train/train_efficiency_models.py`
7. `train/train_direct_models.py`
8. `train/select_model_architecture.py`
9. `train/calibrate_uncertainty.py`
10. `report/build_model_report.py`

Training remains fail-closed on its configured deterministic-training requirements. Defining the sequence does not waive those requirements.

## Weekly runner order

`docs/win/football/nfl/prop_engine/scripts/run_weekly.py` must execute:

1. `build/refresh_nflverse_player_data.py`
2. `build/build_player_identity.py`
3. `validate/audit_market_exclusion.py`
4. `validate/validate_source_quality.py`
5. `project/build_current_universe.py`
6. `project/select_roles.py`
7. `project/build_week1_priors.py`
8. `project/build_current_features.py`
9. `project/project_components.py`
10. `project/allocate_team_opportunity.py`
11. `project/project_direct.py`
12. `project/project_week.py`
13. `report/build_wide_output.py`
14. `validate/validate_week.py`

## Acceptance boundary

Stages 55.1–55.50 define and implement the system. Stages 55.51–55.57 are execution and acceptance gates and require real run evidence; file existence alone is not proof that those gates passed.
