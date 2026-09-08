#!/usr/bin/env python3
"""Create Issue 55 execution-sequence documentation and requirements."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEQUENCE = HERE / "IMPLEMENTATION_SEQUENCE.md"
REQUIREMENTS = HERE / "requirements.txt"

SEQUENCE_TEXT = '# Prop Engine exact implementation execution sequence\n\nThis document is the authoritative implementation and acceptance order for Issue 55.\n\n## Rules\n\n1. Execute the numbered stages in ascending order.\n2. A later stage must not be used to bypass an earlier validation or approval gate.\n3. Historical feature building must use football-only verified sources and the Issue 54 forbidden-input contract.\n4. Training and model selection must remain chronological and leakage-safe.\n5. Development weekly projection may use `--allow-unapproved-models` only at stage 55.54.\n6. Production weekly projection at stage 55.57 must run without the unapproved-model override.\n7. Source-dependent 2026 nflverse families may remain explicitly unavailable until nflverse publishes them; no unavailable rows may be fabricated.\n\n## Exact implementation sequence\n\n- **55.1** Create directory structure.\n- **55.2** Create `config/prop_engine.yaml`.\n- **55.3** Create `scripts/common.py`.\n- **55.4** Create `config/target_eligibility.yaml`.\n- **55.5** Create `config/fallback_rules.yaml`.\n- **55.6** Create `config/acceptance_thresholds.yaml`.\n- **55.7** Create all nine `config/features/*.json`.\n- **55.8** Create `build/build_player_identity.py`.\n- **55.9** Create `build/refresh_nflverse_player_data.py`.\n- **55.10** Create `build/build_historical_universe.py`.\n- **55.11** Create `build/build_targets.py`.\n- **55.12** Create `build/build_player_opportunity.py`.\n- **55.13** Create `build/build_team_opportunity.py`.\n- **55.14** Create `build/build_position_allowed.py`.\n- **55.15** Create `build/build_role_history.py`.\n- **55.16** Create `build/build_player_form.py`.\n- **55.17** Create `build/build_team_form.py`.\n- **55.18** Create `build/build_environment_history.py`.\n- **55.19** Create `build/build_defensive_features.py`.\n- **55.20** Create `build/build_kicking_features.py`.\n- **55.21** Create `build/build_historical_features.py`.\n- **55.22** Create market/source/historical validators.\n- **55.23** Create `run_historical_build.py`.\n- **55.24** Run historical build and resolve every validation failure.\n- **55.25** Create backtest folds.\n- **55.26** Create baseline training.\n- **55.27** Create opportunity training.\n- **55.28** Create efficiency training.\n- **55.29** Create direct training.\n- **55.30** Create architecture selection.\n- **55.31** Create uncertainty calibration.\n- **55.32** Create model report.\n- **55.33** Create `run_training.py`.\n- **55.34** Run chronological backtests.\n- **55.35** Populate numerical acceptance thresholds.\n- **55.36** Approve only models passing baseline and calibration gates.\n- **55.37** Create production registry.\n- **55.38** Create Week 1 priors.\n- **55.39** Create current universe.\n- **55.40** Create role selection.\n- **55.41** Create current features.\n- **55.42** Create component projections.\n- **55.43** Create opportunity allocation.\n- **55.44** Create direct projections.\n- **55.45** Create final projection assembly.\n- **55.46** Create wide output.\n- **55.47** Create weekly validation.\n- **55.48** Create `run_weekly.py`.\n- **55.49** Create all tests.\n- **55.50** Create requirements and documentation.\n- **55.51** Run unit tests.\n- **55.52** Run end-to-end historical smoke test.\n- **55.53** Run market exclusion audit.\n- **55.54** Run one development weekly projection with `--allow-unapproved-models`.\n- **55.55** Inspect generated QB, kicker, offensive-role, and defensive-role tables.\n- **55.56** Approve individual targets only after thresholds pass.\n- **55.57** Run first production weekly projection without unapproved-model override.\n\n## Historical build runner order\n\n`docs/win/football/nfl/prop_engine/scripts/run_historical_build.py` must execute:\n\n1. `build/build_player_identity.py`\n2. `build/build_historical_universe.py`\n3. `build/build_targets.py`\n4. `build/build_player_opportunity.py`\n5. `build/build_team_opportunity.py`\n6. `build/build_position_allowed.py`\n7. `build/build_role_history.py`\n8. `build/build_player_form.py`\n9. `build/build_team_form.py`\n10. `build/build_environment_history.py`\n11. `build/build_defensive_features.py`\n12. `build/build_kicking_features.py`\n13. `build/build_historical_features.py`\n14. `validate/audit_market_exclusion.py`\n15. `validate/validate_historical_data.py`\n\n## Training runner order\n\n`docs/win/football/nfl/prop_engine/scripts/run_training.py` must execute:\n\n1. `validate/audit_market_exclusion.py`\n2. `validate/validate_historical_data.py`\n3. `train/build_backtest_folds.py`\n4. `train/train_baselines.py`\n5. `train/train_opportunity_models.py`\n6. `train/train_efficiency_models.py`\n7. `train/train_direct_models.py`\n8. `train/select_architecture.py`\n9. `train/calibrate_uncertainty.py`\n10. `train/build_model_report.py`\n\nTraining remains fail-closed on its configured deterministic-training requirements. Defining the sequence does not waive those requirements.\n\n## Weekly runner order\n\n`docs/win/football/nfl/prop_engine/scripts/run_weekly.py` must execute:\n\n1. `build/refresh_nflverse_player_data.py`\n2. `build/build_player_identity.py`\n3. `validate/audit_market_exclusion.py`\n4. `validate/validate_source_quality.py`\n5. `project/build_current_universe.py`\n6. `project/select_roles.py`\n7. `project/build_week1_priors.py`\n8. `project/build_current_features.py`\n9. `project/project_components.py`\n10. `project/allocate_team_opportunity.py`\n11. `project/project_direct.py`\n12. `project/project_week.py`\n13. `report/build_wide_output.py`\n14. `validate/validate_week.py`\n\n## Acceptance boundary\n\nStages 55.1–55.50 define and implement the system. Stages 55.51–55.57 are execution and acceptance gates and require real run evidence; file existence alone is not proof that those gates passed.\n'
REQUIREMENTS_TEXT = '# NFL Prop Engine direct Python dependencies\npandas\nnumpy\nPyYAML\npyarrow\nlightgbm\nnflreadpy\npytest\n'


def atomic_create_or_verify(path: Path, expected: str) -> str:
    if path.exists():
        actual = path.read_text(encoding="utf-8-sig")
        if actual != expected:
            raise ValueError(
                f"Existing file differs from Issue 55 contract and was not overwritten: {path}"
            )
        return "verified"

    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temp = Path(handle.name)
    try:
        with handle:
            handle.write(expected)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()
    return "created"


def main() -> int:
    sequence_status = atomic_create_or_verify(SEQUENCE, SEQUENCE_TEXT)
    requirements_status = atomic_create_or_verify(REQUIREMENTS, REQUIREMENTS_TEXT)

    print(f"implementation_sequence={sequence_status}")
    print(f"requirements={requirements_status}")
    print("sequence_steps=57")
    print("requirements_direct_dependencies=7")
    print("ISSUE 55 EXECUTION SEQUENCE PATCH: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
