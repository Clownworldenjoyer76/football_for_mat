# NFL Prop Engine Data Contract

## 1. Purpose and authority

This document defines the data contract for:

`docs/win/football/nfl/prop_engine/`

The authoritative machine-readable configuration is:

`docs/win/football/nfl/prop_engine/config/prop_engine.yaml`

If this document and `prop_engine.yaml` disagree on an executable configuration value, the discrepancy must be resolved before build, training, or production inference proceeds.

The Prop Engine is a football-only projection system. Sportsbook, betting-market, market-implied, DRAT, EPRED, consensus-projection, and similar external prediction inputs are prohibited.

---

## 2. Canonical player-game grain

The canonical player-game grain is:

`season + week + game_id + player_id`

Every canonical player-game table must contain these four fields.

### 2.1 Grain fields

| Field | Contract |
|---|---|
| `season` | NFL season as an integer |
| `week` | NFL week as an integer |
| `game_id` | Stable game identifier for the scheduled NFL game |
| `player_id` | Canonical GSIS player identifier |

A canonical player-game table must contain no duplicate rows at this grain.

Production builders and validators must fail on duplicate canonical grain rather than silently deduplicating rows.

### 2.2 Team-level grain

Where a table is explicitly team-game rather than player-game, its grain must be declared by that builder and must not be silently joined as if it were player-game data.

Typical team-level joins use game/team dimensions such as:

`season + week + game_id + team`

Any team-level source expanded to player-game grain must preserve the canonical player-game uniqueness constraint after the join.

---

## 3. Canonical player identity

The canonical player identifier is:

`GSIS ID`

Configuration key:

`system.canonical_player_id: gsis_id`

Canonical model tables expose the normalized GSIS identity as:

`player_id`

### 3.1 Identity requirements

Every projected or trained player row must have a nonblank canonical `player_id`.

Alternate IDs may be retained as metadata or crosswalk fields, including source IDs such as ESPN or PFR IDs, but they may not replace the canonical GSIS identifier.

Canonical identities must come from authoritative football data sources or the canonical identity crosswalk.

The Prop Engine must not fabricate a GSIS identifier.

The canonical identity crosswalk is:

`docs/win/football/nfl/prop_engine/data/identity/player_crosswalk.parquet`

The primary identity builder is:

`docs/win/football/nfl/prop_engine/scripts/build/build_player_identity.py`

Unresolved identity rows must remain unresolved and be logged. Names, team, position, jersey number, or alternate IDs must not be used to invent a GSIS identifier.

---

## 4. Historical source paths

Historical source paths are configured in:

`docs/win/football/nfl/prop_engine/config/prop_engine.yaml`

### 4.1 Historical player/game sources

| Source | Configured path |
|---|---|
| Player weekly stats | `docs/win/football/nfl/data/historic_data/player_stats/stats_player_week_{season}.parquet` |
| Historical games | `docs/win/football/nfl/data/historic_data/games/games_2010_2025.csv` |
| Historical players | `docs/win/football/nfl/data/historic_data/players/players.parquet` |
| Historical weekly rosters | `docs/win/football/nfl/data/historic_data/weekly_rosters/roster_weekly_{season}.parquet` |
| Historical depth charts | `docs/win/football/nfl/data/historic_data/depth_charts/depth_charts_{season}.parquet` |
| Historical injuries | `docs/win/football/nfl/data/historic_data/injuries/injuries_{season}.parquet` |
| Historical snap counts | `docs/win/football/nfl/data/historic_data/snap_counts/snap_counts_{season}.parquet` |
| Historical participation | `docs/win/football/nfl/data/historic_data/participation/pbp_participation_{season}.parquet` |
| Play-by-play | `docs/win/football/nfl/00_intake/pbp/{season}_pbp.csv.gz` |
| Team stats | `docs/win/football/nfl/00_intake/team_stats/{season}_team_stats.csv` |
| QB stats | `docs/win/football/nfl/00_intake/qb/{season}_qb_stats.csv` |

Historical source files outside `prop_engine/` are read-only inputs to the Prop Engine unless a separate repository workflow explicitly owns those files.

---

## 5. Current-season source paths

The current projection pipeline may read the following football-only sources configured in `prop_engine.yaml`.

| Source | Configured path |
|---|---|
| Current schedule | `docs/win/football/nfl/00_intake/schedule/{season}_schedule.csv` |
| Current roster | `docs/win/football/nfl/data/master/roster_master.csv` |
| Current depth charts | `docs/win/football/nfl/data/master/depth_charts` |
| Current injuries | `docs/win/football/nfl/00_intake/injuries/{season}_injuries.csv` |
| Current weather | `docs/win/football/nfl/data/weather/week_{week}_NFL_weekly_weather.csv` |
| Current travel | `docs/win/football/nfl/data/travel/{season}_week_{week}_travel.csv` |
| Team master | `docs/win/football/nfl/data/master/team_master.csv` |

Isolated refreshed nflverse current-season inputs are written under:

`docs/win/football/nfl/prop_engine/data/current/source/`

That directory may contain source-family files such as:

- `stats_player_week_{season}.parquet`
- `roster_weekly_{season}.parquet`
- `snap_counts_{season}.parquet`
- `pbp_participation_{season}.parquet`
- `players.parquet`

Current-source refreshes must preserve native source identifiers and must not fabricate missing football records.

---

## 6. Canonical derived data paths

The canonical Prop Engine derived datasets are:

| Dataset | Path |
|---|---|
| Identity crosswalk | `docs/win/football/nfl/prop_engine/data/identity/player_crosswalk.parquet` |
| Historical universe | `docs/win/football/nfl/prop_engine/data/historical/universe/player_game_universe.parquet` |
| Historical targets | `docs/win/football/nfl/prop_engine/data/historical/targets/player_game_targets.parquet` |
| Player opportunity | `docs/win/football/nfl/prop_engine/data/historical/opportunity/player_week_opportunity.parquet` |
| Team opportunity | `docs/win/football/nfl/prop_engine/data/historical/opportunity/team_week_opportunity.parquet` |
| Opponent opportunity | `docs/win/football/nfl/prop_engine/data/historical/opportunity/opponent_week_opportunity.parquet` |
| Position allowed | `docs/win/football/nfl/prop_engine/data/historical/opportunity/position_allowed_week.parquet` |
| Role history | `docs/win/football/nfl/prop_engine/data/historical/features/player_role_history.parquet` |
| Player form | `docs/win/football/nfl/prop_engine/data/historical/features/player_form.parquet` |
| Team form | `docs/win/football/nfl/prop_engine/data/historical/features/team_form.parquet` |
| Opponent form | `docs/win/football/nfl/prop_engine/data/historical/features/opponent_form.parquet` |
| Environment | `docs/win/football/nfl/prop_engine/data/historical/features/environment.parquet` |
| Defensive features | `docs/win/football/nfl/prop_engine/data/historical/features/defensive_features.parquet` |
| Kicking features | `docs/win/football/nfl/prop_engine/data/historical/features/kicking_features.parquet` |
| Final historical model features | `docs/win/football/nfl/prop_engine/data/historical/features/player_game_features.parquet` |

Current feature artifacts are rooted at:

`docs/win/football/nfl/prop_engine/data/current/features/`

Models are rooted at:

`docs/win/football/nfl/prop_engine/models/`

Evaluation artifacts are rooted at:

`docs/win/football/nfl/prop_engine/evaluation/`

Production outputs are rooted at:

`docs/win/football/nfl/prop_engine/output/`

Run logs are rooted at:

`docs/win/football/nfl/prop_engine/logs/`

---

## 7. Season coverage contract

Configured season boundaries are:

| Purpose | Season |
|---|---:|
| Historical start | 2012 |
| Participation-feature start | 2016 |
| Rich/PBP feature start | 2021 |
| Historical end | 2025 |
| Current season | 2026 |

Participation-dependent features must not be populated before the configured participation-feature start unless an authoritative equivalent source exists and the configuration is intentionally changed.

PBP-derived rich features may only be enabled for a season when the configured PBP source exists and validates successfully.

Absence of PBP before the rich-feature era must not be converted into fabricated zero PBP events.

---

## 8. Pregame and as-of contract

All production model inputs are pregame features.

For a target player-game with kickoff time `T`, a feature is valid only if every observation used to construct that feature was knowable before `T`.

### 8.1 Historical rows

Historical training features must emulate information available before the historical game's kickoff.

The following are prohibited as model inputs for the game being predicted:

- same-game final player statistics
- same-game target values
- same-game final team statistics
- same-game final score
- same-game snap counts
- same-game participation
- same-game realized target share
- same-game realized carry share
- same-game realized route/role outcomes
- post-kickoff injury updates
- post-kickoff depth-chart updates
- any feature derived from the result of the game being predicted

Rolling, expanding, career, team-form, opponent-form, opportunity, efficiency, and position-allowed features must use strict prior observations.

Week N historical predictors must not use Week N realized outcomes.

### 8.2 Historical audit fields

The historical feature table must retain sufficient audit metadata to prove chronological correctness.

This includes audit fields such as:

- `audit_feature_asof`
- maximum contributing prior player source game
- maximum contributing prior team source game
- depth snapshot timestamp when available
- injury snapshot timestamp when available

For historical model rows, `audit_feature_asof` represents the prediction cutoff associated with the target game's kickoff.

Any contributing timestamped depth or injury snapshot must precede kickoff.

### 8.3 Current-week projections

A weekly production run must have a single explicit `as_of` cutoff.

Current features must use only source data available at or before that cutoff.

The production `as_of` cutoff must be before the kickoff of any game whose prediction is being treated as a true pregame projection.

If a source is updated after the run cutoff, that later information belongs to a later rerun and must not be retroactively included in the earlier run.

Production output must retain:

- `feature_asof`
- `generated_at`

The run manifest must retain the requested/effective `as_of` value and source hashes.

### 8.4 Final-test chronology

Current configured training chronology is:

- model-selection training ends after 2023
- development validation season is 2024
- final training ends after 2024
- final reporting/test season is 2025

New architecture, calibration, or hyperparameter choices must not be selected using 2025 performance.

---

## 9. Target definitions

The canonical prediction targets are defined in `config/prop_engine.yaml`.

Target-building implementation:

`docs/win/football/nfl/prop_engine/scripts/build/build_targets.py`

Canonical target table:

`docs/win/football/nfl/prop_engine/data/historical/targets/player_game_targets.parquet`

### 9.1 Passing yards

Logical target:

`passing_yards`

Source column:

`passing_yards`

Type:

`continuous_signed`

Negative realized passing yardage is valid and must not be clipped to zero in the historical target.

### 9.2 Passing touchdowns

Logical target:

`passing_tds`

Source column:

`passing_tds`

Type:

`count_nonnegative`

### 9.3 Rushing yards

Logical target:

`rushing_yards`

Source column:

`rushing_yards`

Type:

`continuous_signed`

Negative realized rushing yardage is valid and must not be clipped to zero in the historical target.

### 9.4 Rushing touchdowns

Logical target:

`rushing_tds`

Source column:

`rushing_tds`

Type:

`count_nonnegative`

This target means all credited weekly rushing touchdowns for the player.

A component model based on a narrower field-position subset does not redefine the canonical target.

### 9.5 Receiving yards

Logical target:

`receiving_yards`

Source column:

`receiving_yards`

Type:

`continuous_signed`

Negative realized receiving yardage is valid and must not be clipped to zero in the historical target.

### 9.6 Receiving touchdowns

Logical target:

`receiving_tds`

Source column:

`receiving_tds`

Type:

`count_nonnegative`

This target means all credited weekly receiving touchdowns for the player.

A component model based on red-zone targets does not redefine the canonical target.

### 9.7 Kicking points

Logical target:

`kicking_points`

Type:

`derived_count`

Exact formula:

`3 * field_goals_made + extra_points_made`

No sportsbook scoring convention or fantasy scoring convention may replace this formula.

### 9.8 Tackles

Logical target:

`tackles`

Type:

`count_nonnegative`

Exact definition:

`solo_tackles + assisted_tackles`

`def_tackles_with_assist` must not be substituted for the credited assisted-tackle field when its source semantics differ.

### 9.9 Sacks

Logical target:

`sacks`

Source column:

`sacks`

Type:

`count_nonnegative`

---

## 10. Target-source presence and zero semantics

Zero and missing are not interchangeable.

### 10.1 Observed participant with zero statistics

If an authoritative player-game source proves that the player participated in the game and the applicable realized stat is zero, the target may legitimately be `0`.

A zero-stat participant must remain in the target dataset.

### 10.2 Nonparticipant or missing source row

A player-game without authoritative evidence supporting a realized target must not be converted into a false zero merely because the player exists in the universe.

The target builder uses source-presence information, including:

`target_source_present`

to distinguish observed source rows from absent source rows.

### 10.3 Missing player identity

A source row without a canonical GSIS player ID must not be assigned to a player by guesswork.

Unidentifiable nonzero source records must be logged as source-quality anomalies and excluded from canonical player-game targets until identity can be resolved authoritatively.

### 10.4 Missing feature values

There is no global rule that converts missing feature values to zero.

Missing source data must remain missing unless a specific feature definition explicitly establishes a mathematically valid fallback.

A model-specific missing-value strategy may handle nulls at model time, but that does not change the underlying semantic meaning of the source value.

Examples:

- missing snap data is not zero snaps
- missing participation is not zero participation
- missing target share is not automatically zero target share
- missing injury status is not automatically healthy
- missing depth rank is not automatically a backup rank

### 10.5 No fabricated historical data

If a source family is unavailable for an older season, builders must either:

1. leave the dependent feature unavailable;
2. use an explicitly documented prior-only fallback; or
3. exclude that feature/model cohort when the contract requires it.

They must not create synthetic zero rows to simulate unavailable source history.

---

## 11. Feature/target separation

Columns beginning with or representing canonical targets are evaluation labels, not predictors.

Production feature manifests must not include:

- `target_passing_yards`
- `target_passing_tds`
- `target_rushing_yards`
- `target_rushing_tds`
- `target_receiving_yards`
- `target_receiving_tds`
- `target_kicking_points`
- `target_tackles`
- `target_sacks`

Targets may be joined after predictions are constructed for training/evaluation scoring.

The same-game target must never be used to construct its own prediction.

---

## 12. Market-data prohibition

Configuration states:

`system.market_data_allowed: false`

The Prop Engine must not read, derive features from, train on, select models from, calibrate from, or otherwise use sportsbook or market-derived prediction inputs.

### 12.1 Forbidden feature concepts

The configured forbidden feature tokens include:

- `odds`
- `moneyline`
- `spread`
- `total_line`
- `over_odds`
- `under_odds`
- `market`
- `sportsbook`
- `prop_line`
- `drat`
- `epred`
- `projected_pts`
- `win_probability`
- `cover_probability`

Feature manifests, parquet model-feature schemas, and persisted model feature names must not contain forbidden market-derived predictors.

### 12.2 Forbidden input paths

The current configured forbidden input paths are:

- `docs/win/football/nfl/data/historic_data/odds/`
- `docs/win/football/nfl/scripts/00_intake/pull_odds.py`
- `docs/win/football/nfl/scripts/00_intake/pull_opening_odds.py`
- `docs/win/football/nfl/scripts/00_intake/enrich_moneyline.py`
- `docs/win/football/nfl/scripts/00_intake/enrich_spread.py`
- `docs/win/football/nfl/scripts/00_intake/enrich_totals.py`
- `docs/win/football/nfl/scripts/00_intake/pull_market_futures.py`
- `docs/win/football/nfl/data/historic_data/predictions/drat/`
- `docs/win/football/nfl/data/historic_data/predictions/epred/`
- `docs/win/football/nfl/training/`
- `docs/win/football/nfl/01_merge/`

`config.forbidden_input_paths` is the executable deny-list and must remain synchronized with this document.

Adding another forbidden legacy path to configuration requires updating this section in the same change.

### 12.3 Enforcement

Market exclusion is enforced by:

`docs/win/football/nfl/prop_engine/scripts/validate/audit_market_exclusion.py`

The audit output is:

`docs/win/football/nfl/prop_engine/evaluation/market_exclusion_audit.json`

Training and production inference must not proceed when the market-exclusion preflight fails.

Production manifests and logs must report:

`market_data_used: false`

---

## 13. Data validation requirements

Before historical training data is accepted, validators must prove at minimum:

- canonical grain uniqueness
- nonblank canonical GSIS `player_id`
- valid game IDs
- target-domain validity
- exact kicking-points formula
- exact tackles definition
- zero-stat participant retention
- nonparticipant/missing-source rows are not false-zero targets
- strict prior-game feature chronology
- pre-kickoff depth/injury chronology where timestamps exist
- no same-game target leakage
- no forbidden market feature columns
- no forbidden market source references

Historical validation entry point:

`docs/win/football/nfl/prop_engine/scripts/validate/validate_historical_data.py`

Market exclusion entry point:

`docs/win/football/nfl/prop_engine/scripts/validate/audit_market_exclusion.py`

Current-week validation entry point:

`docs/win/football/nfl/prop_engine/scripts/validate/validate_week.py`

---

## 14. Fail-closed behavior

The Prop Engine must fail rather than silently repair a contract violation when any of the following occur:

- duplicate canonical grain
- blank required canonical GSIS ID
- incompatible target source semantics
- missing required model feature
- same-game leakage
- post-cutoff data use
- forbidden market source use
- forbidden market feature use
- malformed required source schema
- impossible or invalid count target
- target formula mismatch

Logging a violation does not make the violating row acceptable for training or production inference.

---

## 15. Change-control rule

Any change to one of the following is a data-contract change:

- canonical grain
- canonical player ID
- source paths
- season availability boundaries
- target definition
- target units
- target missing/zero semantics
- as-of/pregame rules
- feature leakage rules
- forbidden input paths
- forbidden feature definitions

A data-contract change requires:

1. an intentional source/configuration change;
2. corresponding builder changes where needed;
3. validator/test updates;
4. regeneration of affected derived artifacts;
5. rerunning the relevant historical/current validation before production use.

No production model or weekly projection run may silently redefine this contract.
