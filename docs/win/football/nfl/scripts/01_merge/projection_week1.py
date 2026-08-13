#!/usr/bin/env python3
"""
Week 1 NFL model projection.

Uses current-season pregame schedule/market/DRAT/EPRED/weather/travel/
injury/depth/QB1 information, with prior-season team/QB performance and
prior-season snap/participation fallback because no current-season games have
been completed before Week 1.

WRITES ONLY:
  docs/win/football/nfl/01_merge/week_1_NFL_enriched.csv
"""

from projection import run_projection


# ============================================================================
# CHANGE THIS EACH SEASON
# ============================================================================
SEASON = 2026
# ============================================================================


def main() -> None:
    run_projection(season=SEASON, week1_mode=True)


if __name__ == "__main__":
    main()
