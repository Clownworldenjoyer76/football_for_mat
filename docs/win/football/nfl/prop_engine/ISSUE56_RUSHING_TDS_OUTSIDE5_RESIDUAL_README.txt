ISSUE 56 — RUSHING TDS OUTSIDE-5 RESIDUAL DIAGNOSTIC

This package is diagnostic only. It does not modify production models,
calibration, selected_model.json, thresholds, registry, or manifests.

Confirmed structural mismatch:
- target_rushing_tds = all weekly rushing TDs
- existing component predicts only rushing TDs from yardline_100 <= 5
- 2024 evaluation cohort: 37.278% of rushing TDs occurred outside the 5
- 2025 reporting cohort: 40.909% occurred outside the 5

Candidate tested:
    full rushing TD projection
      = existing inside-the-5 component projection
      + outside-the-5 rushing-TD residual prediction

Residual model:
- one LightGBM Poisson model only
- uses the already-approved rushing_tds direct feature schema
- target is exact PBP rushing TDs outside the 5
- trains through 2023
- best iteration selected on 2024
- final in-memory refit through 2024
- 2025 scored only after iteration count is frozen
- structural weights are fixed at 1.0 + 1.0, not tuned

The package reports locked-gate metrics for 2024 and then 2025.
No long all-target retraining is performed.
