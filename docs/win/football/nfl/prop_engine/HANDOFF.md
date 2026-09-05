# NFL PROP ENGINE — PROJECT HANDOFF ENGRAM

## 1. Mission

Continue development of an NFL individual-player projection system inside:

`Clownworldenjoyer76/football_for_mat`

The system projects individual NFL game/week statistics:

* passing touchdowns
* passing yards
* receiving yards
* rushing yards
* receiving touchdowns
* rushing touchdowns
* kicking points
* tackles
* sacks

This must be a genuine football-data statistical/modeling system.

### Absolutely forbidden

Do not use:

* sportsbook player props
* prop lines
* sportsbook odds
* betting probabilities
* consensus betting data
* market-implied projections
* external betting projections

The Prop Engine config explicitly prohibits market-derived features.

---

# 2. Working environment

Local repository root:

`C:\Users\Mat\Documents\GitHub\football_for_mat`

All Prop Engine files belong under:

`docs/win/football/nfl/prop_engine/`

The user performs local implementation through **PowerShell**.

When giving implementation instructions:

1. Briefly explain what the command does.
2. Give ready-to-run PowerShell.
3. Work one numbered issue at a time.
4. Do not advance until the current issue passes acceptance validation.

---

# 3. GitHub policy

GitHub is **strictly read-only**.

Allowed:

* search repository
* inspect files
* inspect schemas
* inspect historical implementation
* read code

Forbidden:

* commits
* branches
* pull requests
* file writes
* file deletes
* GitHub modifications of any kind

All actual edits happen locally through PowerShell.

Never invent:

* repository paths
* schemas
* columns
* functions
* dependencies
* source availability

Inspect first when uncertain.

---

# 4. Core data contract

Canonical player-game grain:

`season + week + game_id + player_id`

Canonical player ID:

`GSIS`

Historical modeled seasons:

`2012–2025`

Current season:

`2026`

Rich PBP feature start:

`2021`

Participation-dependent features begin no earlier than:

`2016`

### Leakage rule

Every feature intended for a pregame model must be derived only from information available **strictly before the target game's kickoff**.

Same-game realized values may exist in raw historical tables but must never enter target-game model features.

`played_game_flag` is outcome/target metadata and must never be used as a model feature.

---

# 5. Important configuration

Main config:

`docs/win/football/nfl/prop_engine/config/prop_engine.yaml`

Important configured paths include:

* `historical_player_stats_pattern`
* `historical_games`
* `historical_players`
* `historical_rosters_pattern`
* `historical_depth_pattern`
* `historical_injuries_pattern`
* `historical_snaps_pattern`
* `historical_participation_pattern`
* `pbp_pattern`
* `team_stats_pattern`
* `qb_stats_pattern`
* `current_schedule`
* `current_roster`
* `current_depth_root`
* `current_injuries`
* `current_weather`
* `current_travel`
* `team_master`
* `identity_crosswalk`
* `historical_universe`
* `historical_targets`
* `player_opportunity`
* `team_opportunity`
* `opponent_opportunity`
* `position_allowed`
* `role_history`
* `player_form`
* `historical_features`
* current/model/evaluation/output/log roots

Local config should currently include:

```yaml
role_history: docs/win/football/nfl/prop_engine/data/historical/features/player_role_history.parquet
player_form: docs/win/football/nfl/prop_engine/data/historical/features/player_form.parquet
```

Do not assume the GitHub version contains the latest local additions if they have not been committed.

Rolling config currently includes:

```yaml
rolling_windows:
  games: [1, 3, 5, 8]

ewm:
  spans: [3, 5, 8]
  adjust: false
```

Position-allowed config includes:

```yaml
positions:
  position_allowed:
    supported_groups: [QB, RB, WR, TE]
    prior_sample_size: 20.0
```

Target yardage types are signed continuous values. Negative passing/rushing/receiving yardage must be preserved.

---

# 6. Shared utilities

File:

`docs/win/football/nfl/prop_engine/scripts/common.py`

Important functions:

* `repo_root`
* `nfl_root`
* `prop_root`
* `load_config`
* `read_csv_required`
* `read_parquet_required`
* `write_parquet_atomic`
* `write_csv_atomic`
* `normalize_team`
* `normalize_player_id`
* `normalize_name`
* `parse_game_id`
* `ensure_unique`
* `require_columns`
* `reject_forbidden_feature_columns`
* `safe_numeric`
* `kickoff_timestamp`
* `log_run`
* `season_week_sort`

`common.py` restricts writes to the Prop Engine directory.

Core team aliases there include:

* WAS → WSH
* LA → LAR
* JAC → JAX

Historical franchise relocation needs special care:

* SD ↔ LAC
* OAK ↔ LV
* STL ↔ LAR

Do not globally rewrite historical team identity unless the feature specifically needs franchise continuity.

---

# 7. Issue status

## COMPLETE

`1. [x] Create the Prop Engine directory structure`

`2. [x] Create the system contract and model configuration`

`3. [x] Create shared Prop Engine utilities`

`4. [x] Build canonical player identity crosswalk`

`5. [x] Build/refresh nflverse player data`

`6. [x] Build historical player-game universe`

`7. [x] Build all nine historical targets`

`8. [x] Build player-level weekly opportunity and efficiency data`

`9. [x] Build team and opponent weekly opportunity tables`

`10. [x] Build opponent position-allowed statistics`

`11. [x] Build historical role and availability features`

`12. [x] Build historical player rolling-form features`

Do not redo these unless a later issue exposes a verified defect.

Next task is **Issue 13**, but its specification has not been provided in this handoff. Ask for or inspect the exact Issue 13 requirements rather than inventing them.

---

# 8. Issue 4 — identity crosswalk

Builder:

`docs/win/football/nfl/prop_engine/scripts/build/build_player_identity.py`

Output:

`docs/win/football/nfl/prop_engine/data/identity/player_crosswalk.parquet`

Rows:

`26,526`

Known unresolved current GSIS identities:

* Al-Jay Henderson — ESPN 4599158 — RB NYJ
* Greg Desrosiers Jr. — ESPN 4691889 — RB LAC
* Khalil Dinkins — ESPN 4699678 — TE SF

Known PFR collision/anomaly:

`pfr_id=WillJo10`

Crosswalk maps it to GSIS `00-0035944`, LA DE Jonah Williams, while a 2023 Cincinnati snap row represents OT Jonah Williams.

For snap joins, Issue 8 protected against this by including team context.

Do not casually alter the identity crosswalk because of that anomaly.

---

# 9. Issue 6 — historical universe

Builder:

`docs/win/football/nfl/prop_engine/scripts/build/build_historical_universe.py`

Output:

`docs/win/football/nfl/prop_engine/data/historical/universe/player_game_universe.parquet`

Validated:

* 554,660 rows
* 10,729 players
* 3,663 games
* 310,604 played player-games
* 10,800 played rows recovered by snap/participation backstop despite roster absence

Important universe fields include:

* season
* week
* game_id
* kickoff_timestamp
* player_id
* team
* position
* position_group
* depth_present_flag
* depth_rank
* depth_starter_flag
* prior_offense_snap_pct
* prior_defense_snap_pct
* prior_offense_participation
* prior_defense_participation

Pregame depth is already leakage protected.

2025 timestamp depth uses the latest valid snapshot strictly before target cutoff.

Prior snap/participation values are from previous games, not the target game.

---

# 10. Issue 7 — targets

Builder:

`docs/win/football/nfl/prop_engine/scripts/build/build_targets.py`

Output:

`docs/win/football/nfl/prop_engine/data/historical/targets/player_game_targets.parquet`

Rows:

`554,660`

Targets:

* passing_yards
* passing_tds
* rushing_yards
* rushing_tds
* receiving_yards
* receiving_tds
* kicking points = `3 * field_goals_made + extra_points_made`
* tackles = solo tackles + assisted tackles
* sacks

Do **not** use `def_tackles_with_assist`.

Negative yardage is intentionally preserved.

---

# 11. Issue 8 — player opportunity

Builder:

`docs/win/football/nfl/prop_engine/scripts/build/build_player_opportunity.py`

Output:

`docs/win/football/nfl/prop_engine/data/historical/opportunity/player_week_opportunity.parquet`

Validated:

* 241,205 rows
* 7,001 canonical players
* 3,663 games

Core player stats:

`2012–2025`

Snap coverage:

`2013–2025`

Participation:

`2016–2025`

Rich PBP:

`2021–2025`

Important derivations:

* red zone = yardline_100 <= 20
* inside 10 = <= 10
* goal line = <= 5
* target_share = targets / team targets
* air_yards_share = receiving_air_yards / team receiving_air_yards

Carry share:

* 2021–2025: player carries / PBP team rush attempts
* 2012–2020: player carries / summed player-stat team carries

`qb_kneel` is intentionally **not excluded**.

Defensive rate denominator uses resolved defensive snaps with participation fallback.

Snap join uses:

`season + week + game_id + player_id + team`

to avoid cross-team PFR collisions.

Raw opportunity values are realized same-week values. They must be lagged before model use.

---

# 12. Issue 9 — team/opponent opportunity

Builder:

`docs/win/football/nfl/prop_engine/scripts/build/build_team_opportunity.py`

Outputs:

* `team_week_opportunity.parquet`
* `opponent_week_opportunity.parquet`

Both contain:

`7,326 rows`

Policy:

* 2012–2020: core opportunity from historical player stats
* 2021–2025: direct PBP aggregation
* advanced team metrics use team_stats
* same-week raw values must be lagged downstream

Historical aliases used there:

* SD → LAC
* OAK → LV
* STL → LAR

---

# 13. Issue 10 — position allowed

Builder:

`docs/win/football/nfl/prop_engine/scripts/build/build_position_allowed.py`

Output:

`docs/win/football/nfl/prop_engine/data/historical/opportunity/position_allowed_week.parquet`

Rows:

`29,304`

Exactly:

`7,326 team-weeks × 4 position groups`

Supported:

* QB
* RB
* WR
* TE

Required columns include:

* season
* week
* defense_team
* offense_position_group
* players_faced
* targets_allowed
* receptions_allowed
* receiving_yards_allowed
* receiving_tds_allowed
* carries_allowed
* rushing_yards_allowed
* rushing_tds_allowed
* passing_yards_allowed
* passing_tds_allowed
* tackles_generated
* raw_rate_sample_size
* league_rate
* shrunk_rate

Shrinkage prior:

`20.0`

QB rate:

`passing_yards_allowed / pass_attempts_faced`

RB/WR/TE rate:

`(receiving_yards_allowed + rushing_yards_allowed) / (targets_allowed + carries_allowed)`

Production models should use lagged/shrunk rates rather than same-week raw rates.

---

# 14. Issue 11 — role and availability

Status:

`11. [x] Build historical role and availability features`

Builder:

`docs/win/football/nfl/prop_engine/scripts/build/build_role_history.py`

Output:

`docs/win/football/nfl/prop_engine/data/historical/features/player_role_history.parquet`

Rows:

`554,660`

Required columns:

* season
* week
* game_id
* player_id
* team
* position
* depth_rank_pregame
* depth_starter_flag_pregame
* injury_status_pregame
* injury_out_flag
* injury_doubtful_flag
* injury_questionable_flag
* prior_offense_snap_pct
* prior_defense_snap_pct
* snap_pct_roll3
* snap_pct_roll5
* snap_pct_ewm3
* snap_pct_ewm5
* prior_offense_participation
* prior_defense_participation
* participation_roll3
* participation_roll5
* depth_rank_change
* snap_share_change
* participation_change
* team_change_flag
* games_with_current_team_before_game
* starter_promotion_flag
* starter_demotion_flag
* teammate_out_count_position
* teammate_unavailable_snap_share_position
* role_history_games
* role_missing_flag

### Locked semantics

Primary snap share:

`max(offense_snap_pct, defense_snap_pct)`

Primary participation:

`max(offense_participation, defense_participation)`

Rolling features use only observed values from games with kickoff strictly before the target kickoff.

EWM:

* spans 3 and 5
* `adjust=False`

`depth_rank_change`:

current pregame depth rank minus previous pregame depth rank.

Positive = moved lower on depth chart.

Negative = promotion.

`team_change_flag` compares current franchise to latest strictly prior realized source-game franchise.

Historical franchise identity comparison normalizes:

* SD → LAC
* OAK → LV
* STL → LAR

This normalization is for **team-history comparisons only**. Historical output team values remain consistent with the universe.

Unavailable teammate definition:

**confirmed Out only**

Doubtful and questionable are retained separately and are not automatically treated as unavailable.

`teammate_unavailable_snap_share_position` sums prior snap share for same-position-family teammates confirmed Out, excluding the player.

`role_missing_flag = 1` only if current depth rank and all four prior snap/participation values are missing.

### Acceptance

Issue 11 passed independent validation for:

* exact headers/order
* universe grain
* pregame depth
* pregame injuries
* strict prior-only rolling snap/participation
* EWM/change features
* team history
* starter changes
* teammate-out features
* role_missing_flag
* finite/range constraints

Important discovered edge case:

Player `00-0039472` had 9 legitimate 2020 opportunity rows before first appearing in the 2024 universe. Therefore “first universe row must have zero role history” is an invalid assumption. History comes from source history, not merely previous universe rows.

---

# 15. Issue 12 — player rolling form

Status:

`12. [x] Build historical player rolling-form features`

Builder:

`docs/win/football/nfl/prop_engine/scripts/build/build_player_form.py`

Output:

`docs/win/football/nfl/prop_engine/data/historical/features/player_form.parquet`

Rows:

`554,660`

Total columns:

`400`

There are:

* 39 base metrics
* 10 derived features per metric
* 390 rolling/form feature columns

### Required naming convention

For every base metric:

* `{metric}_lag1`
* `{metric}_roll3_mean`
* `{metric}_roll5_mean`
* `{metric}_roll8_mean`
* `{metric}_roll3_median`
* `{metric}_roll5_std`
* `{metric}_ewm3`
* `{metric}_ewm5`
* `{metric}_season_to_date`
* `{metric}_career_prior`

### 39 base metrics

* pass_attempts
* dropbacks
* completions
* passing_yards
* passing_tds
* yards_per_attempt
* passing_td_rate
* passing_air_yards
* carries
* rushing_yards
* rushing_tds
* yards_per_carry
* carry_share
* red_zone_carries
* goal_line_carries
* targets
* receptions
* receiving_yards
* receiving_tds
* yards_per_target
* catch_rate
* target_share
* air_yards_share
* red_zone_targets
* red_zone_target_share
* field_goal_attempts
* field_goals_made
* extra_point_attempts
* extra_points_made
* tackles
* sacks
* qb_hits
* tackle_rate_per_def_play
* sack_rate_per_def_play
* qb_hit_rate_per_def_play
* offense_snap_pct
* defense_snap_pct
* offense_participation
* defense_participation

### Missing-history policy

Rookie or metric-no-history rows use strictly prior **position priors**.

2010 and 2011 historical player-stat files are used as prehistory seeds where the metric can legitimately be reconstructed.

2010–2011 support core box-score data including:

* attempts/completions/passing stats
* passing_air_yards
* carries/rushing
* targets/receptions/receiving
* receiving_air_yards
* kicking
* tackles/sacks/qb_hits

Unsupported rich prehistory metrics remain unavailable until their real source begins.

Examples:

* PBP-derived rich red-zone/dropback features begin in 2021.
* participation history begins in 2016.
* snap history begins according to actual snap source availability.

Never fabricate unavailable earlier history.

### Team changes

Career efficiency/history survives team changes.

Team share metrics reset when franchise changes:

* carry_share
* target_share
* air_yards_share
* red_zone_target_share

Historical franchise identity uses:

* SD → LAC
* OAK → LV
* STL → LAR

for continuity.

### Required flags

* `no_nfl_history_flag`
* `new_team_flag`
* `history_games`

`history_games` counts all strictly prior realized/source player-games, including valid 2010–2011 prehistory.

`no_nfl_history_flag = 1` iff `history_games == 0`.

`new_team_flag = 1` when prior NFL/source history exists and the current franchise differs from the latest strictly prior source-game franchise.

### Rolling semantics

Rolling windows use the last N **observed prior values**, not calendar weeks.

`roll5_std` uses:

`ddof=0`

EWM uses:

`adjust=False`

No same-game realized metric may enter the target row.

No future value is ever forward-filled.

### Acceptance

Issue 12 passed an independent validator that reconstructed all **390 derived columns** from source data.

Acceptance output ended:

`ISSUE 12 ACCEPTANCE: PASS`

Validated:

* exact 400-column contract
* canonical universe grain
* 2010–2011 prehistory
* exact history flags
* all 390 derived form features
* strict prior-kickoff joins
* position-prior fallback
* team-share resets
* career continuity across teams
* PBP source boundaries
* participation source boundaries
* no infinities
* no same-game leakage

---

# 16. Issue 12 validator

A local validator was saved at:

`docs/win/football/nfl/prop_engine/validate_issue12.py`

It is a validation helper, not a production builder.

Do not confuse it with the production script.

---

# 17. Data-source availability by era

Use this as a general guide, but inspect source schemas before relying on it.

### 2010–2011

Available as prehistory player stats.

Useful for leakage-safe seed history for compatible core metrics.

### 2012

Historical modeling universe begins.

Player stats available.

No useful snap source in the 2012 snap file.

### 2013–2015

Snap data available.

Participation not yet available.

### 2016–2020

Snap + participation available.

No Prop Engine rich PBP-derived features yet.

### 2021–2025

Rich PBP, snaps, participation, player stats, team stats, etc.

---

# 18. Critical modeling conventions

## Signed yardage

Do not clamp player yardage targets or historical features to zero merely because they represent yards.

Negative values can be legitimate.

## Zero versus missing

A realized zero is data.

A missing source is not zero.

This matters heavily for:

* efficiency rates
* participation
* snaps
* PBP-only metrics
* prehistory

## Position priors

Position priors must themselves be leakage safe.

A target row may use only position observations strictly before target kickoff.

Do not calculate a full-season position average and apply it backward.

## Same-week tables

Several historical opportunity tables contain realized statistics for that game/week.

They are source measurements, not directly pregame-safe features.

Lag or otherwise restrict them before feature use.

## Team-share history

Team share is context-dependent.

Issue 12 resets these after team/franchise changes.

Do not silently merge old-team target/carry share into new-team share history.

## Career efficiency

Efficiency history is player-level and survives team changes unless a future issue explicitly specifies otherwise.

---

# 19. Acceptance philosophy

A builder printing `"status":"passed"` is **not enough**.

Each issue should receive independent acceptance validation.

Acceptance should generally test:

1. exact required headers and order
2. row count
3. canonical grain uniqueness
4. source reconciliation
5. formulas independently
6. temporal leakage
7. missing-data semantics
8. impossible/negative values where appropriate
9. infinities
10. forbidden feature tokens

Only mark an issue `[x]` after independent acceptance passes.

---

# 20. User interaction preferences

The user prefers concise operational responses.

Do not write long retrospectives after every issue.

When an issue completes, preferred format is approximately:

`13. [x] <issue name>`

Then one very short future note if needed.

When debugging:

* identify the failure
* give one focused audit or patch
* wait for result
* do not jump ahead

Do not repeatedly ask questions when the answer is already in project history.

---

# 21. Current handoff point

Issues 1–12 are complete.

The last completed milestone is:

`12. [x] Build historical player rolling-form features`

Issue 12 acceptance passed completely.

The next bot should:

1. Obtain the exact **Issue 13 specification**.
2. Inspect relevant existing repo code/data before implementing.
3. Preserve all contracts above.
4. Work only on Issue 13.
5. Build locally through PowerShell.
6. Run independent acceptance before marking it complete.

Do not skip ahead.

---

# 22. Non-negotiable safety checks for future features

Before any feature is allowed into modeling, verify:

* Was it knowable before kickoff?
* Does it accidentally contain the target game's realized outcome?
* Was a season-wide aggregate calculated using future games?
* Was a rolling metric shifted correctly?
* Did a current-game snap/participation value leak in?
* Did a position prior include target/future observations?
* Did historical franchise renaming create a false team-change signal?
* Was missing source data incorrectly converted to zero?
* Does the feature contain sportsbook/market/projection information?

If any answer is uncertain, audit before proceeding.

---

# 23. Canonical status line for continuation

At handoff, assume:

`Issues 1–12 COMPLETE. Current task: begin Issue 13 only after receiving its exact specification.`
