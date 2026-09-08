ISSUE 56 — SAFE CONTINUATION AFTER ISSUE 24 STATIC-MARKER FAILURE

This package does NOT retrain direct models.

Before any architecture selection/calibration work it:
1. Parses the trainer and normalizes only these deterministic policy assignment lines:
   SEED = 24024
   MODEL_SELECTION_TRAIN_END = 2023
   DEVELOPMENT_VALIDATION_SEASON = 2024
   FINAL_TRAIN_END = 2024
   UNTOUCHED_TEST_SEASON = 2025
2. Extracts validate_issue24.py's own required_markers list and proves every marker
   is present in the trainer.
3. Proves PRIMARY_OBJECTIVE exactly matches the nine expected target objectives.
4. Proves every direct-model metadata random_seed and LightGBM seed field is 24024.
5. Runs the FULL validate_issue24.py immediately.

Only after all cheap/full Issue 24 validation passes does it continue with:
- sacks direct-feature post-rebuild validation
- prior sacks component denominator validation
- 2024-only architecture selection
- regenerated selected-model validation
- 2024-only calibration
- production approval evaluation

No direct-model retraining is performed by this package.
No thresholds or production registry state are changed.
