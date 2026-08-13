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

from projection import apply_models, load_schema, nfl_root, prepare_week


# ============================================================================
# CHANGE THIS EACH SEASON
# ============================================================================
SEASON = 2026
# ============================================================================

WEEK = 1


def main() -> None:
    root = nfl_root()
    schema = load_schema(root)

    original, features = prepare_week(
        root,
        SEASON,
        WEEK,
        True,
        schema,
    )

    projected = apply_models(
        root,
        original,
        features,
    )

    output_dir = root / "01_merge"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "week_1_NFL_enriched.csv"

    projected.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"WROTE {output_path} | "
        f"games={len(projected)} | "
        f"columns={len(projected.columns)}"
    )


if __name__ == "__main__":
    main()
