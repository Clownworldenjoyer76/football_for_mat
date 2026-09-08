ISSUE 56 — RUSHING TDS HURDLE DIAGNOSTIC

Why this exists
---------------
Previous diagnostics established:

1. Current raw component on 2024:
   MAE 0.086474        PASS
   improvement 8.649%  PASS
   Poisson 0.424515    PASS
   Brier 0.046478      PASS
   bias -0.041089      FAIL

2. Generic expected-count isotonic calibration fixes bias by lifting even the
   lowest-risk rows, which destroys MAE/improvement.

3. Uniform scaling, additive correction, direct/component blends, all-carry
   formulas, outside-the-5 residuals, and high-risk tail-only scalar corrections
   do not satisfy every 2024 gate.

Therefore the remaining structural question is whether a model that explicitly
separates zero vs positive TD outcomes can concentrate expected count on likely
scorers without lifting thousands of zero rows.

Diagnostic architecture
-----------------------
Binary hurdle gate:
    P(rushing_tds >= 1)

Expected count:
    P(rushing_tds >= 1) * E[rushing_tds | rushing_tds >= 1]

Only two severity formulations are allowed:
- unit severity = 1.0
- empirical mean positive-game TD count from training data

No severity hyperparameter grid is used.

Chronology
----------
- binary model selection training: through 2023
- binary best iteration: selected on 2024
- 2024 selects between the two predefined severity formulations
- final binary model: fit through 2024 using the frozen iteration count
- final positive severity: estimated through 2024
- 2025: reporting only

Feature/model policy
--------------------
- exact existing rushing_tds direct feature schema
- deterministic Issue 24 LightGBM parameters inherited from train_direct_models
- objective changed to binary and metric to binary_logloss
- no class weighting, because raw probability calibration must be preserved
- no new features
- no market data
- no threshold changes

The package is diagnostic only. It modifies no production model, calibration,
registry, selected_model.json, threshold, or manifest.
