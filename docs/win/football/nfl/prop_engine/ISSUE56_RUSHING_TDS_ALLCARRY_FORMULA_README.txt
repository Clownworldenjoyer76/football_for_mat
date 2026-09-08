ISSUE 56 — RUSHING TDS ALL-CARRY FORMULA DIAGNOSTIC

Purpose
-------
Test a target-aligned structural family without model retraining:

    projected rushing TDs = projected player carries * all-carry rushing TD rate

The rate definition is exactly aligned to the canonical target:

    rushing_tds / carries

This avoids the current component mismatch where the target is all rushing TDs
but the efficiency component uses only inside-the-5 rushing TDs.

Candidate families
------------------
Projected carry volume:
- player carries roll3 / roll5 / ewm5 / season-to-date / career prior
- team rush volume x player carry share combinations
- deterministic coalesced team/share variants

All-carry TD rate:
- rushing TDs / carries over roll3 / roll5 / ewm5 / season-to-date / career prior
- career empirical-Bayes rate using a fixed 100-carry prior exposure
- deterministic pregame fallback ladder

The 100-carry prior exposure is inherited from the existing stronger rushing-TD
efficiency regularization constant. It is not tuned here.

Selection policy
----------------
- candidate selection uses 2024 only
- 2025 is reporting-only
- no 2025 outcome affects candidate selection
- league prior for 2024 uses seasons < 2024
- league prior for 2025 uses seasons < 2025
- no market data
- locked thresholds are read, never changed

This diagnostic does not modify models, calibration, registry, thresholds,
selected_model.json, or production manifests.

Preflight reads parquet schema metadata only; it does not scan the historical feature table.
