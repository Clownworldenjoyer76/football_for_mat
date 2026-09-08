# NFL Prop Engine Data Contract

This document defines the data interfaces, identity rules, temporal rules, missing-data behavior, and source restrictions for the NFL Prop Engine under `docs/win/football/nfl/prop_engine/`.

The Prop Engine is a football-data-only projection system. All generated model inputs and outputs must remain inside the Prop Engine directory. Repository data outside the Prop Engine may be read only when it is an approved football source described below.

## 1. Canonical Keys

The canonical player-game grain is:

```text
season + week + game_id + player_id
```

`player_id` is the canonical GSIS player identifier. No ESPN ID, PFR ID, display name, jersey number, roster slot, or row number may replace GSIS at the canonical player-game grain.

Canonical team codes are normalized before joins. Required aliases are:

```text
SD  -> LAC
OAK -> LV
STL -> LAR
WAS -> WSH
LA  -> LAR
JAC -> JAX
```

Relocated-franchise aliases (`SD/LAC`, `OAK/LV`, `STL/LAR`) are treated as the same historical franchise when a feature requires franchise continuity. Display/output team labels may preserve the source-era abbreviation when appropriate, but join identity must use the canonical franchise mapping.

Canonical historical modeled seasons are 2012 through 2025. The configured current season is 2026. Rich PBP-derived features begin only where validated PBP is available; the configured rich-feature start is 2021. Participation-dependent features may not begin before 2016.

Every table claiming canonical player-game grain must be unique on the four canonical keys. Duplicate canonical keys are a hard validation failure.

## 2. Existing Read-Only Repository Sources

The following repository football sources are approved read-only inputs. They remain owned by the existing repository and are not rewritten by Prop Engine builders.

| Source | Path / pattern | Primary use |
|---|---|---|
| Historical player stats | `docs/win/football/nfl/data/historic_data/player_stats/stats_player_week_{season}.parquet` | historical outcomes, player volume, player efficiency |
| Historical games | `docs/win/football/nfl/data/historic_data/games/games_2010_2025.csv` | schedule/game identity and historical game context |
| Historical players | `docs/win/football/nfl/data/historic_data/players/players.parquet` | authoritative player identity seed |
| Historical weekly rosters | `docs/win/football/nfl/data/historic_data/weekly_rosters/roster_weekly_{season}.parquet` | historical roster membership and aliases |
| Historical depth charts | `docs/win/football/nfl/data/historic_data/depth_charts/depth_charts_{season}.parquet` | pregame depth role |
| Historical injuries | `docs/win/football/nfl/data/historic_data/injuries/injuries_{season}.parquet` | pregame injury context |
| Historical snap counts | `docs/win/football/nfl/data/historic_data/snap_counts/snap_counts_{season}.parquet` | prior snap share |
| Historical participation | `docs/win/football/nfl/data/historic_data/participation/pbp_participation_{season}.parquet` | prior participation |
| PBP | `docs/win/football/nfl/00_intake/pbp/{season}_pbp.csv.gz` | play-derived opportunity and rich football features |
| Team stats | `docs/win/football/nfl/00_intake/team_stats/{season}_team_stats.csv` | prior team/opponent form |
| QB stats | `docs/win/football/nfl/00_intake/qb/{season}_qb_stats.csv` | football-only QB context when explicitly required |
| Current schedule | `docs/win/football/nfl/00_intake/schedule/{season}_schedule.csv` | current game identity and matchup |
| Current ESPN roster | `docs/win/football/nfl/data/master/roster_master.csv` | current roster and ESPN alias context |
| Current depth charts | `docs/win/football/nfl/data/master/depth_charts/` | current depth role |
| Current injuries | `docs/win/football/nfl/00_intake/injuries/{season}_injuries.csv` | current injury state |
| Current weather | `docs/win/football/nfl/data/weather/week_{week}_NFL_weekly_weather.csv` | target-game weather |
| Current travel | `docs/win/football/nfl/data/travel/{season}_week_{week}_travel.csv` | target-game travel |
| Team master | `docs/win/football/nfl/data/master/team_master.csv` | canonical team resolution |

These files are read-only from the Prop Engine perspective. Prop Engine scripts must not use their presence as permission to ingest unrelated columns. In particular, historical schedule files may contain fields that are not approved prediction inputs; builders must explicitly select football-only columns.

## 3. Prop Engine Generated Sources

Prop Engine-generated data lives only under `docs/win/football/nfl/prop_engine/`.

Primary generated contracts include:

| Generated source | Path |
|---|---|
| Player identity crosswalk | `data/identity/player_crosswalk.parquet` |
| Historical player-game universe | `data/historical/universe/player_game_universe.parquet` |
| Historical targets | `data/historical/targets/player_game_targets.parquet` |
| Player opportunity | `data/historical/opportunity/player_week_opportunity.parquet` |
| Team opportunity | `data/historical/opportunity/team_week_opportunity.parquet` |
| Opponent opportunity | `data/historical/opportunity/opponent_week_opportunity.parquet` |
| Position allowed | `data/historical/opportunity/position_allowed_week.parquet` |
| Role history | `data/historical/features/player_role_history.parquet` |
| Player form | `data/historical/features/player_form.parquet` |
| Team form | `data/historical/features/team_form.parquet` |
| Opponent form | `data/historical/features/opponent_form.parquet` |
| Environment history | `data/historical/features/environment.parquet` |
| Defensive features | `data/historical/features/defensive_features.parquet` |
| Kicking features | `data/historical/features/kicking_features.parquet` |
| Canonical historical feature table | `data/historical/features/player_game_features.parquet` |
| Current refreshed source copies | `data/current/source/` |
| Current universe | `data/current/{season}_week_{week}_universe.parquet` |
| Current roles | `data/current/{season}_week_{week}_roles.parquet` |
| Week 1 priors | `data/current/{season}_week_1_priors.parquet` |
| Current feature table | `data/current/features/{season}_week_{week}_features.parquet` |
| Component projections | `data/current/{season}_week_{week}_component_projections.parquet` |
| Allocated opportunity | `data/current/{season}_week_{week}_allocated_opportunity.parquet` |
| Direct projections | `data/current/{season}_week_{week}_direct_projections.parquet` |
| Long final projections | `output/{season}/week_{week}_player_projections.csv` |
| Active-only projections | `output/{season}/week_{week}_active_player_projections.csv` |
| Wide final projections | `output/{season}/week_{week}_player_projections_wide.csv` |
| Weekly validation | `output/{season}/week_{week}_validation.json` |
| Weekly run manifest | `output/{season}/week_{week}_run_manifest.json` |
| Models and manifests | `models/` |
| Evaluation artifacts | `evaluation/` |
| Structured run logs | `logs/` |

Generated intermediate sources are not automatically model features. Only columns explicitly selected by the canonical feature manifest or a persisted model feature manifest may enter a model.

## 4. Player Identity Resolution

GSIS is the authoritative player identity.

The canonical crosswalk is:

```text
data/identity/player_crosswalk.parquet
```

Identity resolution priority is:

1. Exact GSIS ID.
2. Unique authoritative ESPN alias mapped to one GSIS ID.
3. Unique PFR alias mapped to one GSIS ID.
4. Normalized-name match only when that normalized name resolves uniquely to one GSIS ID.

A display-name match must never break an ambiguous identity tie. Ambiguous names remain unresolved rather than being guessed.

Historical player metadata and historical roster data establish GSIS-centered identity history. Current ESPN roster and depth-chart data may add current aliases and current-team context, but they do not replace GSIS as the canonical ID.

A player without a valid canonical GSIS ID may not be emitted as a modeled player-game row.

Player career efficiency history survives a trade or franchise change. Team-role/share history does not: team-share features reset when the player changes team/franchise.

## 5. Game Identity Resolution

`game_id` is the canonical game identifier carried on every canonical player-game row.

Supported game-ID forms are:

- a numeric game/event identifier; or
- normalized nflverse form `YYYY_WW_AWAY_HOME`.

For nflverse-form IDs, season and week are validated and team abbreviations are normalized. The normalized representation zero-pads the week:

```text
2024_1_JAC_WAS -> 2024_01_JAX_WSH
```

For a target week:

- every projected team must appear in exactly one scheduled game;
- every projected `game_id` must exist in the target-week schedule;
- home and away teams must be distinct;
- the same team may not appear in more than one target-week game;
- player-game joins must not infer a game from player name or team alone when `game_id` is available.

Schedule context is authoritative for the current matchup. Historical opponent context must never be carried forward as if it were the current opponent.

## 6. Target Definitions

The Prop Engine models exactly these nine targets.

| Target | Definition | Type |
|---|---|---|
| `passing_yards` | source passing yards | continuous; historical signed values preserved |
| `passing_tds` | source passing touchdowns | nonnegative count |
| `rushing_yards` | source rushing yards | continuous; historical signed values preserved |
| `rushing_tds` | source rushing touchdowns | nonnegative count |
| `receiving_yards` | source receiving yards | continuous; historical signed values preserved |
| `receiving_tds` | source receiving touchdowns | nonnegative count |
| `kicking_points` | `3 * field_goals_made + extra_points_made` | derived nonnegative count |
| `tackles` | `solo_tackles + assisted_tackles` | nonnegative count |
| `sacks` | source sacks | nonnegative count |

Historical raw yardage outcomes may be negative. Production final projections are floored at zero at the production-output boundary.

Realized zero and missing are different states:

- A verified participant with no recorded target statistic is retained with a realized zero where the target contract defines zero.
- A nonparticipant with no source target row is not converted into a false zero; the target remains missing.
- Missing source data must not be silently converted into realized production.

## 7. Role Definitions

Current role selection operates only on eligible current-universe players.

Core role fields include:

```text
starter_flag
primary_qb_flag
primary_kicker_flag
primary_role_flag
committee_role_flag
role_confidence
role_reason
```

Role rules:

- Exactly one primary QB must be selected per scheduled team.
- An available depth-rank-1 QB is primary.
- If the higher-ranked QB is ineligible, the next eligible ranked QB is promoted.
- Genuine unresolved QB depth ambiguity is a hard failure.
- Exactly one primary kicker must be selected per scheduled team.
- Kicker selection uses current depth rank plus strictly prior realized kicking attempts.
- Ambiguous kicker situations remain deterministic but use reduced role confidence and wider uncertainty.
- RB/FB/WR/TE and defensive positions may be represented as committee roles where workload is shared.
- `Out` and otherwise ineligible players may exist in the audit universe but must not receive active-only projections.

Target eligibility uses these role requirements:

| Targets | Eligible positions | Current requirement |
|---|---|---|
| passing yards / passing TDs | QB | identified QB role |
| rushing yards / rushing TDs | QB, RB, FB, WR, TE | recent usage or plausible depth/participation role |
| receiving yards / receiving TDs | RB, FB, WR, TE | recent usage or plausible depth/participation role |
| kicking points | K, PK | identified primary kicker role |
| tackles / sacks | DL, DE, DT, NT, EDGE, LB, ILB, OLB, MLB, DB, CB, S, FS, SS, NB | recent defensive participation or current starter/promotion role |

## 8. Time/As-Of Definitions

Every model feature must be knowable before the target-game kickoff.

The core temporal rule is:

```text
source observation timestamp < target kickoff timestamp
```

Equal-time observations are not accepted as prior history.

Consequences:

- Week N player rolling features exclude Week N realized player statistics.
- Week N snap features exclude Week N snap counts.
- Week N participation features exclude Week N participation.
- Week N team/opponent form excludes the Week N result.
- Depth snapshots used for the target game must precede kickoff.
- Injury snapshots used for the target game must precede kickoff.
- Current-season realized sources for Week N use only source week `< N`.
- No same-game realized opportunity, team performance, opponent performance, or position-allowed result may enter that game's feature row.

Rolling features use prior observed values only. EWM state is updated only after the observation has become historical relative to the next target row.

`as_of` on a weekly run is run provenance and represents the information cutoff for that run. It must never be used to authorize post-kickoff information.

Historical architecture selection uses chronological separation. The established development validation season is 2024; 2025 is untouched/reporting-only and may not leak into architecture selection, fitting, early stopping, feature-schema choice, or categorical vocabularies.

## 9. Missing-Data Rules

Missingness is explicit and must not be disguised as realized zero.

General fallback priority for player efficiency is:

1. same-season player prior;
2. recent prior-season player prior;
3. career player prior;
4. position-group league prior.

General fallback priority for role/share is:

1. current-team recent role;
2. current depth rank plus career role;
3. position/depth prior.

Special cases:

- **Rookie/no NFL history:** use position/depth priors; do not invent NFL production; widen uncertainty.
- **New team/trade:** retain player efficiency history; reset team share; rebuild workload from current team/depth context; widen uncertainty for uncertain new role.
- **Backup promotion:** retain player efficiency but replace prior backup workload with promoted-role workload expectation; widen uncertainty.
- **Missing snap history:** use depth plus statistical usage and set a missing flag; widen uncertainty.
- **Missing participation:** use depth plus statistical usage and set a missing flag; widen uncertainty.
- **Missing depth:** use current-team recent role, then career role, then position prior; set a missing flag and widen uncertainty.
- **Missing injury:** do not assume healthy; set a missing flag and widen uncertainty.
- **Missing weather:** use roof/surface context plus explicit weather-missing flags; do not invent target-game weather.
- **Missing player statistics:** follow the efficiency and role/share fallback ladders; set a missing flag and widen uncertainty.
- **Defensive low volume:** use prior hierarchy, preserve a low-volume flag, and widen uncertainty.
- **Kicker change:** retain player efficiency when available but assign workload from the current primary-kicker role; do not transfer another kicker's prior team share.

A missing categorical value may be represented by the model's persisted missing/unseen category convention. Numeric missingness may remain null/NaN where the feature contract permits it. Missing target-game environment or role state must never be filled by copying an unrelated prior-game opponent, weather, travel, injury, or depth value.

## 10. Market Data Exclusion

Market-derived information is forbidden as model input, feature source, target definition aid, role-selection signal, fallback signal, calibration input, or projection adjustment.

The Prop Engine config must keep:

```text
market_data_allowed: false
```

Forbidden feature/source concepts include, at minimum:

```text
odds
moneyline
spread
total_line
over_odds
under_odds
market
sportsbook
prop_line
drat
epred
projected_pts
win_probability
cover_probability
```

Forbidden repository source families include historical odds/market directories, sportsbook enrichment scripts, and DRAT/EPRED prediction directories.

The market-exclusion audit is a required preflight in historical validation, training, current role selection, current feature construction, direct/component projection, and weekly validation workflows.

The Prop Engine must not use sportsbook/player-prop data even if such data exists elsewhere in the repository.

## 11. Current-Week Source Freshness

Current-week source quality is validated before projection.

Target-week pregame snapshot sources must represent the target week/current snapshot:

```text
weekly roster
current ESPN roster
depth charts
injuries
schedule
weather
travel
```

Strictly prior realized sources must be available through Week N-1 for a Week N projection:

```text
player stats
snap counts
participation
PBP
team stats
```

Freshness rules:

- For Week N > 1, lagged realized sources must contain the prior required week and current feature builders must filter them to source week `< N`.
- For Week 1, zero current-season rows are valid for strictly prior realized sources because no current-season game has yet been played.
- A missing or stale required snapshot/lagged source is a source-quality failure unless an explicit accepted missing-data rule applies.
- Zero-byte/headerless current-season realized placeholder files may be treated as empty monitored sources at Week 1 rather than crashing the monitor.
- Schedule, weather, and travel are expected at game grain for the target week.
- Source duplicate keys, stale freshness, and excessive required identity/team/game missingness are validation failures.

The weekly runner writes source file inventory and SHA-256 hashes to:

```text
output/{season}/week_{week}_run_manifest.json
```

The same manifest records the feature schema hash, model versions, validation status, and `market_data_used: false`.

## 12. Known Unavailable Features

The following inputs are not available as verified Prop Engine features and must not be fabricated, inferred under a misleading verified label, or treated as observed source truth:

- verified routes run
- verified yards per route run
- verified pass-rush snaps
- verified individual pressure rate
- verified independent official last-minute inactive feed
- sportsbook/player-prop data

Derived proxy features may be used only when their names and metadata clearly identify them as derived/proxy quantities and when they are built from approved football-only sources available before kickoff. A proxy must never be relabeled as one of the verified unavailable features above.
