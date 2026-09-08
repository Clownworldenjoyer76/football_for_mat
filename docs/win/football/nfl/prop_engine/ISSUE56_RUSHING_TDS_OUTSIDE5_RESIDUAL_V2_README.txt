ISSUE 56 — RUSHING TDS OUTSIDE-5 RESIDUAL DIAGNOSTIC V2

Fixes the failed first diagnostic.

Canonical PBP contract:
- config seasons.rich_feature_start = 2021
- do NOT require PBP for 2012-2020
- do NOT convert missing pre-2021 PBP labels into zero residual TDs

Chronology used:
- residual selection training: 2021-2023
- residual validation / best iteration: 2024
- final in-memory residual fit: 2021-2024
- 2025: reporting only, never used for selection

Before LightGBM starts, the runner performs a cheap preflight that verifies:
- every PBP file 2021-2025 exists
- required PBP columns exist
- historical feature schema contains every required feature
- rushing_tds selected-model chronology is correct
- audit validation/test windows exist
- selected features pass the configured market exclusion policy

The residual label is exact:
    target_rushing_tds - inside5_rushing_tds
and the script asserts the canonical weekly target exactly reconciles to PBP
all-rushing-TD totals throughout the PBP-backed modeling window.

Candidate remains diagnostic only:
    existing inside-5 component + outside-5 residual
with fixed weights 1.0 + 1.0.

No production models, calibration, registry, thresholds or manifests are modified.
