ISSUE 56 — SACKS DIRECT FEATURE DENOMINATOR FIX

Confirmed bug
-------------
Historical player sack rate is defined as sacks / defensive_plays.

The direct sacks model includes a derived matchup feature that was:
  player_sack_rate_per_def_play_roll5_mean * expected_opponent_dropbacks

That is dimensionally inconsistent. This package replaces it with:
  player_sack_rate_per_def_play_roll5_mean * expected_opponent_plays

The derived feature is renamed:
  matchup_player_sack_rate_x_opp_dropbacks
to:
  matchup_player_sack_rate_x_opp_plays

Patched source
--------------
- scripts/build/build_historical_features.py
- scripts/project/build_current_features.py
- validate_issue17.py
- config/features/sacks.json

Generated artifacts are not hand-edited.

Runner rebuild
--------------
1. Patch and validate source.
2. Rebuild historical features and feature manifest.
3. Validate Issue 17 (and Issue 19 if present).
4. Retrain deterministic direct models through 2024.
5. Validate direct models if validator exists.
6. Rerun architecture selection using 2024 only.
7. Recheck the prior top-level sacks denominator fix.
8. Validate rebuilt sacks feature/model artifacts.
9. Recalibrate from 2024 only.
10. Run the local production approval evaluator on 2025.

No threshold changes. No registry approval changes. No market inputs.
