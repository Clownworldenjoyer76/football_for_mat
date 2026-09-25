#!/usr/bin/env python3
from __future__ import annotations

ODDS_OUTPUT_COLUMNS = [
    "snapshot_id",
    "snapshot_fetched_at",
    "game_id",
    "commence_time",
    "home_team",
    "away_team",
    "bookmaker",
    "market_type",
    "bet_side",
    "line",
    "odds_american",
    "odds_decimal",
    "last_update",
    "home_moneyline_american",
    "away_moneyline_american",
    "home_spread",
    "away_spread",
    "home_spread_american",
    "away_spread_american",
    "total",
    "over_american",
    "under_american",
]

EXPECTED_MARKET_SIDES = {
    ("h2h", "home"),
    ("h2h", "away"),
    ("spreads", "home"),
    ("spreads", "away"),
    ("totals", "over"),
    ("totals", "under"),
}
