# NFL Prop Engine Model Versioning

This document defines the immutable version identifier, metadata contract, promotion workflow, and rollback rules for production NFL Prop Engine models.

The production registry is:

```text
docs/win/football/nfl/prop_engine/models/production_registry.json
```

A registry entry points to an approved immutable version. A registry value of `version: null` means that no model version is currently approved for production for that target.

## Version Format

Every promoted target model version must use exactly:

```text
YYYYMMDD_HHMM_{target}_{architecture}
```

Example:

```text
20260903_1642_receiving_yards_blend
```

The timestamp is the model-version creation/promotion timestamp in UTC, formatted as:

```text
YYYYMMDD_HHMM
```

`target` must be one of:

```text
passing_yards
passing_tds
rushing_yards
rushing_tds
receiving_yards
receiving_tds
kicking_points
tackles
sacks
```

`architecture` must describe the selected production architecture for the target. Approved architecture labels are normalized, filesystem-safe identifiers such as:

```text
baseline
direct
component
blend
```

When the architecture-selection artifact uses `direct_component_blend`, the production version suffix is normalized to `blend`.

Version identifiers are immutable. Re-training, changing a feature set, changing blend weights, changing calibration, or changing any required metadata creates a new version identifier. An existing version identifier must never be silently overwritten.

## Required Metadata

Every promoted model version must persist metadata containing all of the following keys:

```text
training_start
training_end
target
feature_schema_hash
git_commit_sha
model_family
validation_metrics
final_test_metrics
blend_weights
uncertainty_calibration_version
```

### `training_start`

Earliest training-data boundary used for the promoted artifact.

Recommended representation:

```text
YYYY-MM-DD
```

or a clearly documented season/week boundary when the model's training contract is season-based.

The value must describe the actual persisted final training artifact, not the earlier architecture-selection subset.

### `training_end`

Latest training-data boundary used for the promoted artifact.

The value must be strictly before any untouched final-test period used only for reporting. For the accepted architecture-selection flow, the final persisted model training cutoff and the reporting-only test period must remain distinguishable.

### `target`

Exact Prop Engine target name. It must match both the version identifier and the production-registry target key.

### `feature_schema_hash`

Hash of the exact ordered feature schema consumed by the promoted model.

This must correspond to the persisted model feature manifest, including feature order and type/category contract where applicable. A model may not be promoted if its recorded feature-schema hash does not reproduce from the persisted feature manifest.

### `git_commit_sha`

Full Git commit SHA identifying the repository state from which the model was produced.

Do not use a branch name, tag alone, dirty-working-tree description, or shortened human label as a substitute. If the working tree contains uncommitted model-relevant changes, the version is not reproducible from `git_commit_sha` alone and must not be promoted until provenance is resolved.

### `model_family`

The selected production architecture/model family.

Examples:

```text
baseline
direct
component
blend
```

If a target is a blend, `model_family` must identify it as a blend and the component model identities must remain recoverable from the version metadata or referenced manifests.

### `validation_metrics`

Metrics from the development validation split used for architecture/model acceptance.

This object must record the metrics relevant to the target and acceptance policy. It is not interchangeable with final-test reporting metrics.

Examples include:

```text
mae
bias
brier_1plus
poisson_deviance
improvement_vs_baseline_pct
```

Only applicable metrics need values, but the metadata must preserve the complete metric set actually used to approve the model.

### `final_test_metrics`

Metrics from the untouched final reporting/test split.

These metrics are reporting evidence only and must not be used retroactively to choose the architecture, tune features, tune hyperparameters, tune blend weights, or calibrate uncertainty.

If final-test metrics are unavailable, promotion must explicitly record why rather than silently substituting validation metrics.

### `blend_weights`

Blend coefficients used by the promoted target model.

For a non-blended model, use an explicit empty mapping:

```json
{}
```

For a blend, persist exact deterministic weights, for example:

```json
{
  "direct": 0.6,
  "component": 0.4
}
```

Weights must sum to 1.0 within numerical tolerance and must match the selected-model artifact used by weekly projection.

### `uncertainty_calibration_version`

Immutable identifier or hash/version reference for the uncertainty calibration applied to this model.

Calibration artifacts are stored under:

```text
docs/win/football/nfl/prop_engine/models/calibration/
```

The reference must identify the exact calibration used for intervals/probabilities for the promoted model version. Recalibration requires a new model version or a separately versioned immutable calibration reference that is explicitly linked in new promotion metadata.

## Canonical Metadata Example

A promoted version metadata document should follow this structure:

```json
{
  "version": "20260903_1642_receiving_yards_blend",
  "training_start": "2012-01-01",
  "training_end": "2024-12-31",
  "target": "receiving_yards",
  "feature_schema_hash": "sha256-of-ordered-feature-schema",
  "git_commit_sha": "full-40-character-git-sha",
  "model_family": "blend",
  "validation_metrics": {
    "mae": 0.0,
    "bias": 0.0,
    "improvement_vs_baseline_pct": 0.0
  },
  "final_test_metrics": {
    "mae": 0.0,
    "bias": 0.0
  },
  "blend_weights": {
    "direct": 0.6,
    "component": 0.4
  },
  "uncertainty_calibration_version": "receiving_yards-calibration-immutable-id"
}
```

The numeric values above are structural examples only; they are not production model metrics.

## Artifact Layout

A promoted version must remain reproducible from immutable model artifacts and metadata. The version identifier should be retained alongside, or referenced by, the target's selected production artifact set.

Existing model families currently persist artifacts under:

```text
docs/win/football/nfl/prop_engine/models/components/{component_name}/
docs/win/football/nfl/prop_engine/models/efficiency/{model_name}/
docs/win/football/nfl/prop_engine/models/{target}/
docs/win/football/nfl/prop_engine/models/calibration/
```

The target-level promotion metadata must link the selected target architecture to the exact component/direct/efficiency/calibration artifacts used to produce it.

Model binaries, manifests, metadata, blend configuration, and calibration references associated with an approved version must not be mutated in place.

## Promotion Workflow

A production promotion is target-specific.

1. Run the historical build and validation successfully.
2. Run the complete training pipeline successfully.
3. Confirm market exclusion passes.
4. Confirm historical leakage validation passes.
5. Confirm source/model acceptance thresholds pass.
6. Confirm the selected architecture in:
   `docs/win/football/nfl/prop_engine/models/{target}/selected_model.json`.
7. Capture the exact persisted model feature schema hash.
8. Capture `training_start` and `training_end`.
9. Capture the full `git_commit_sha`.
10. Capture validation metrics from the development-validation evidence.
11. Capture final-test metrics from the reporting-only test evidence.
12. Capture exact blend weights, or `{}` for a non-blend.
13. Capture the immutable uncertainty-calibration reference.
14. Construct the new version ID using:
    `YYYYMMDD_HHMM_{target}_{architecture}`.
15. Persist the immutable version metadata and artifact references.
16. Only after all acceptance checks pass, update the target entry in:
    `models/production_registry.json`
    to the new version and set `production_approved: true`.

A failed or incomplete promotion must leave the existing production registry unchanged.

## Production Registry Rules

Each registry target has:

```json
{
  "production_approved": false,
  "version": null
}
```

until a version is explicitly approved.

Rules:

- `production_approved: true` requires a non-null valid version identifier.
- `production_approved: false` must not be interpreted as production-ready even if model files exist.
- A weekly production run must use the version referenced by the production registry.
- An override that permits unapproved models is for controlled validation/testing only and must not mutate the registry.
- Training must never silently overwrite the artifact set referenced by an approved production version.
- Promotion of one target must not implicitly promote any other target.

## Validation, Test, and Calibration Separation

The following evidence has different purposes and must remain separate in version metadata:

- **validation_metrics**: model/architecture acceptance evidence from the development-validation split.
- **final_test_metrics**: untouched reporting-only test evidence.
- **uncertainty_calibration_version**: calibration fitted only from allowed OOF validation residuals, never from the final reporting/test split.

The final test set must not be used for model selection, feature selection, blend-weight tuning, hyperparameter tuning, threshold tuning, or uncertainty calibration.

## Blend Versioning

A blend is a first-class production architecture.

If any of the following changes, create a new version:

- direct-model artifact;
- component-model artifact;
- efficiency-model artifact affecting a component;
- blend weight;
- feature schema;
- training cutoff;
- code commit;
- uncertainty calibration.

The metadata must make the blend reproducible without inferring weights from current code defaults.

## Rollback

Rollback means changing the production registry pointer to a previously approved immutable version.

Rollback must not:

- edit the old version's metadata;
- replace the old model binary;
- recompute old metrics;
- change old blend weights;
- replace the old calibration artifact.

Before rollback, verify that the referenced version metadata and artifacts still match their recorded hashes/schema/provenance.

## Reproducibility and Failure Rules

A candidate must not be promoted when any required metadata field is missing.

A candidate must not be promoted when:

- the feature-schema hash cannot be reproduced;
- the target in metadata does not match the target in the version ID;
- the model family does not match the architecture suffix;
- the Git SHA is unavailable;
- validation evidence is missing;
- final-test reporting is incorrectly mixed into model selection;
- blend weights do not reproduce the selected blend;
- calibration provenance is unavailable;
- market-feature exclusion fails;
- historical leakage validation fails.

Version metadata is an audit record. Unknown values must be recorded explicitly as unavailable and block production approval where the missing value prevents reproducibility; they must not be fabricated.
