ISSUE 56 — RUSHING TDS TAIL CALIBRATION DIAGNOSTIC

Problem isolated by prior diagnostics
-------------------------------------
The raw selected component is close enough on MAE/improvement but underpredicts
rushing TD mean. The generic 12-bin isotonic expected-count calibration fixes
bias by imposing a positive floor on even the lowest-risk rows, which damages
MAE on this sparse target.

This diagnostic tests POINT-ONLY monotone tail corrections:
- tail additive adjustment
- tail multiplicative scale
- continuous tail slope expansion

For every candidate:
- rows below the frozen risk cutoff are unchanged from the raw component
- the numeric cutoff is derived from a predefined 2024 raw-prediction quantile
- candidate parameters are selected using 2024 only
- the same numeric cutoff/parameter is frozen before 2025 reporting
- 2025 does not define its own quantile cutoff

Probability policy
------------------
This is a point-only repair. The existing production-calibrated P(1+) path is
left unchanged and its Brier score is measured unchanged. This matches the
existing Issue 56 point-calibration separation between displayed expected point
and calibrated count probabilities.

No retraining. No PBP reads. No feature rebuild. No production artifacts,
thresholds, registry, selected_model.json, or calibration files are modified.
