ISSUE 56 — RUSHING TDS READ-ONLY DIAGNOSTIC

No model, calibration, threshold, registry, feature, or production file is modified.

The diagnostic does two things:

1. Point-repair search
   - candidates are selected strictly on 2024
   - 2025 is reporting-only
   - tests raw component, raw direct, direct/component blends, existing
     expected-count calibration, component/calibration blends, multiplicative
     scales, and additive offsets
   - locked acceptance thresholds are read but never changed

2. Structural target decomposition
   - rebuilds exact rushing-TD play locations from local PBP for 2024/2025
   - confirms how many target rushing TDs occurred from inside the 5 versus
     outside the 5 on the exact evaluation cohorts
   - compares current component projection error against:
       a) the full rushing-TD target
       b) the inside-the-5 rushing-TD subset

Confirmed source contract before this diagnostic:
- player goal-line carry share denominator is team goal-line carries
- rushing_td_per_goal_line_carry uses exact rush TDs from yardline_100 <= 5
- production target rushing_tds is the ordinary all-rushing-TDs weekly stat

Purpose:
Determine whether a legitimate 2024-only point repair is enough. If not,
quantify the structural gap before any expensive architecture/model rebuild.
