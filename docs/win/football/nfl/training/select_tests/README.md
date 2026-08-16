# NFL Historical Selection Filter Tests

This folder is isolated from the live NFL selection configuration.

## Historical input

The script reads:

`docs/win/football/nfl/training/backtests/step14_market_independent_probabilities_v4.csv`

Default seasons:

- 2022
- 2023
- 2024
- 2025

## How to use it

1. Edit only `docs/win/football/nfl/training/select_tests/markets.yaml` with the filters you want to test.
2. Run `NFL Historical Selection Filter Test` manually.
3. The workflow runs the test and commits the new files under `docs/win/football/nfl/training/select_tests/results/`.

## Result files

### `graded_picks.csv`

One row per selected historical bet with:

- season / week / teams
- market
- HOME / AWAY / OVER / UNDER
- line
- American odds
- model probability
- no-vig market probability
- edge
- expected value
- full Kelly
- capped Kelly
- WIN / LOSS / PUSH
- profit or loss in units using 1 unit risk per bet

### `summary_by_season.csv`

Performance for every season and market, including:

- ALL
- HOME / AWAY for moneyline and spread
- OVER / UNDER for totals

### `summary_by_market.csv`

All seasons combined, with the same side splits.

### Drilldown reports

- `by_ev_band.csv`
- `by_kelly_band.csv`
- `by_probability_band.csv`
- `by_edge_band.csv`
- `by_odds_band.csv`
- `by_line_band.csv`

Every drilldown contains season, market, side, band, picks, wins, losses, pushes, win percentage, profit units, return percentage, and average model metrics.

Moneyline/spread drilldowns include `ALL`, `HOME`, and `AWAY`.
Totals drilldowns include `ALL`, `OVER`, and `UNDER`.

The drilldown band definitions are editable under `report_bands:` in the test `markets.yaml`. These reporting bands do not change which bets are selected.

## Important historical-market limitation

The V4 backtest file contains historical **closing** odds and lines. This tester therefore grades filters against closing markets. It does not prove that the same odds were available earlier in the week.
