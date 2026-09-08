ISSUE 56 — ISSUE 24 SEED CONTRACT CONTINUATION

Purpose:
The prior sacks direct-feature rebuild completed successfully, but validate_issue24.py
stopped on metadata["random_seed"] == 24024.

This continuation:
- verifies train_direct_models.py uses SEED = 24024
- verifies metadata random_seed is written from SEED
- reports every target's metadata random_seed and LightGBM primary param seed
- retrains direct models only when trainer/metadata reconciliation requires it
- reruns validate_issue24.py and does not bypass it
- resumes 2024-only architecture selection
- validates both sacks denominator repairs
- recalibrates on 2024 only
- reruns the production approval evaluator

No approval thresholds or registry state are changed.
