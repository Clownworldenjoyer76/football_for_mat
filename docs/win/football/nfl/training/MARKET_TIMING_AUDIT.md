# NFL Market Timing Audit

Date: 2026-08-15
Status: **V4 AS-OF BACKTEST BLOCKED**
Production inference changed: **No**

## Finding

The historical training market and the live betting market are not the same point-in-time data source.

### Historical canonical market fields

`docs/win/football/nfl/scripts/training/step1.py` carries the following fields directly from:

`docs/win/football/nfl/data/historic_data/games/games_2010_2025.csv`

- `away_moneyline`
- `home_moneyline`
- `spread_line`
- `away_spread_odds`
- `home_spread_odds`
- `total_line`
- `under_odds`
- `over_odds`

The source schema is the nflverse/nflfastR schedule schema. Its documented `spread_line` and `total_line` fields are closing lines. The row contains one market value per game and does not contain an as-of timestamp for the canonical price fields.

Therefore these fields cannot be treated as the sportsbook state that was available when a live weekly prediction was actually made.

### Step 2 legacy odds are not an as-of replacement

Step 2 reads:

`docs/win/football/nfl/data/historic_data/odds/nfl_odds_<season>.csv`

and adds `hist_odds_total`, `hist_home_spread`, and `hist_away_spread` plus weather fields. Those legacy odds rows do not contain a bookmaker, capture timestamp, or market-update timestamp. They cannot reconstruct a Tuesday/daily/live betting snapshot either.

The leak-free v2/v3 feature schemas correctly excluded these duplicate `hist_*` market fields.

## Live market source

The current live intake uses `docs/win/football/nfl/scripts/00_intake/pull_odds.py` and Odds-API.io.

It selects DraftKings first, then FanDuel, and records current moneyline, spread, total, prices, market `last_update`, and a raw-payload `fetched_at` timestamp.

The weekly schedule then uses the most recent current odds file produced by that pull.

The live workflows are manually dispatched, so the effective betting snapshot is the market state at the time the intake workflow is run, not a fixed historical lead time such as exactly Tuesday at 08:00.

## Historical provider capability test

A minimal provider probe was run against the finished 2025 Dallas Cowboys at Philadelphia Eagles game.

### Historical event lookup

- Historical-events request: HTTP 200
- Target game found: yes
- Provider event ID: `60525453`

### Historical odds lookup

- Historical-odds request: HTTP 200
- DraftKings/FanDuel returned: none

### Direct finished-event movement lookup

Moneyline movement requests were then made directly for the resolved event ID so the test did not depend on a historical closing-odds payload.

- DraftKings: HTTP 404, no opening/latest/movements
- FanDuel: HTTP 404, no opening/latest/movements

Result: the current provider cannot backfill the required finished-game DraftKings/FanDuel point-in-time history for this 2025 test event.

Probe outputs:

- `docs/win/football/nfl/training/market_timing_provider_probe.json`
- `docs/win/football/nfl/training/market_timing_direct_movement_probe.json`

## V4 training gate

A v4 model/backtest must **not** be trained from the current 2021-2025 historical market fields while describing them as prediction-time or Tuesday-time odds.

V4 remains blocked until one of these conditions is met:

1. A historical source is obtained that provides timestamped pre-kickoff moneyline/spread/total snapshots for the desired bookmaker(s) and seasons; or
2. Enough point-in-time snapshots are accumulated prospectively from the live intake to support a real walk-forward evaluation.

No interpolation from closing lines, no synthetic reconstruction from open/close endpoints, and no timestamp inference from file creation or Git commit dates is permitted.

## Prospective capture fix

The v4 branch updates `pull_odds.py` without changing the existing current-file locations used downstream.

Existing compatibility outputs remain:

- `00_intake/odds/YYYY_MM_DD_NFL_odds.csv`
- `00_intake/odds/raw/YYYY_MM_DD_nfl_odds.json`

Every pull also writes immutable timestamped snapshots to:

- `00_intake/odds/snapshots/<snapshot_id>_NFL_odds.csv`
- `00_intake/odds/raw/snapshots/<snapshot_id>_nfl_odds.json`

The normalized snapshot rows include:

- `snapshot_id`
- `snapshot_fetched_at`
- `last_update`
- bookmaker
- market type and side
- line and price
- all normalized current market fields

This fixes the prior same-day overwrite problem: repeated pulls can continue updating the compatibility file while preserving every actual point-in-time observation for future as-of joins.

## Next valid modeling step

Once a sufficient timestamped sample exists, build an as-of dataset using only the latest sportsbook snapshot with `snapshot_fetched_at <= decision_time` for each game. Compare model probabilities to the no-vig probability from that same snapshot and evaluate CLV against a separately defined closing market.

Until then, the correct model result is **no claimed historical incremental edge at the live decision time**.
