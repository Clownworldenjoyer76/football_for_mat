ISSUE 56 — SACKS READ-ONLY POINT DIAGNOSTIC

This package does not retrain or modify any model, calibration, threshold,
registry, selected-model metadata, or production artifact.

It reads:
- evaluation/model_selection_predictions.parquet
- models/sacks/selected_model.json
- models/calibration/sacks_calibration.json
- config/acceptance_thresholds.yaml

It evaluates existing/directly-derived point candidates.

Candidate choice is based only on 2024 validation.
2025 is reported only after the 2024 choice is frozen.

Candidate families:
- identity direct
- component
- direct/component convex blends
- existing 2024 expected-count calibration
- raw-direct / calibrated-point blends
- multiplicative scale of direct
- additive offset of direct

The locked approval thresholds are read, not changed.

Purpose:
Determine cheaply whether a legitimate 2024-selected point repair exists.
If none passes the 2024 point gates, do not launch a long retrain based on
post-processing guesses; sacks requires structural model work instead.
