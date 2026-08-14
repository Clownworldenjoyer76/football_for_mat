# NFL Annual Model Retraining Checklist

Use this checklist after the NFL season is complete and all final 2026 game data has been collected.

## 1. Extend the historical build to include the completed season

The current historical build scripts in:

`docs/win/football/nfl/scripts/training/`

were originally built around 2021–2025.

Before retraining after the 2026 season, review **Steps 1–10** and update any hardcoded season ranges, file lists, output names, or validation logic so they include **2026**.

The goal is for the historical build to produce:

```text
docs/win/football/nfl/training/historical_core_2021.csv
docs/win/football/nfl/training/historical_core_2022.csv
docs/win/football/nfl/training/historical_core_2023.csv
docs/win/football/nfl/training/historical_core_2024.csv
docs/win/football/nfl/training/historical_core_2025.csv
docs/win/football/nfl/training/historical_core_2026.csv
```

For 2027, extend the upstream build again so `historical_core_2027.csv` is produced, and so on.

## 2. Do not manually add seasons to Steps 11, 13, or 14

These scripts are now designed to use the exact yearly files found in the training directory:

```text
docs/win/football/nfl/scripts/training/step11.py
docs/win/football/nfl/scripts/training/step13.py
docs/win/football/nfl/scripts/training/step14_calibration.py
```

Once `historical_core_2026.csv` exists, Step 11 should automatically train on 2021–2026.

Step 13 reads the same training-season range from the Step 11 schema.

Step 14 builds probability calibration from the new Step 13 chronological backtest.

Only change these scripts if the **model design, feature list, preprocessing, targets, or backtest methodology** intentionally changes.

## 3. Run the annual workflow

Run manually:

```text
.github/workflows/nfl_annual_training_models.yml
```

GitHub Actions workflow name:

```text
NFL Annual Training Models
```

The workflow runs:

```text
Steps 1–10
    ↓
Step 11: train margin + total models
    ↓
Step 13: chronological backtest
    ↓
Step 14: probability calibration
    ↓
Commit training/model outputs
```

## 4. Verify the historical training files

Before accepting the retraining, confirm:

- every expected season file exists;
- seasons are contiguous from 2021 through the newly completed season;
- no yearly file is empty;
- Step 11 reports the expected training seasons;
- the Step 11 schema still has the intended feature count;
- no postgame/result columns entered the feature matrix.

Current model design uses **260 features**. If that number changes, it should be because the feature specification was intentionally changed and both live projection scripts were updated to match.

## 5. Verify the generated model artifacts

The annual workflow should rebuild:

```text
docs/win/football/nfl/models/step11_margin_model.cbm
docs/win/football/nfl/models/step11_total_points_model.cbm
docs/win/football/nfl/models/step11_feature_schema.json
docs/win/football/nfl/models/step14_probability_calibration.json
```

Also verify:

```text
docs/win/football/nfl/training/backtests/step13_chronological_backtest.csv
```

Check that the Step 13 backtest includes the newly completed season as held-out chronological predictions where appropriate.

## 6. Validate Step 13 before trusting the new calibration

Spot-check the backtest for:

- chronological ordering by season / week / gameday / gametime;
- no future games used to train an earlier held-out game;
- `predicted_margin`;
- `actual_margin`;
- `predicted_total`;
- `actual_total_points`;
- predicted ML winner;
- actual home win;
- predicted ATS side;
- actual ATS result;
- predicted O/U side;
- actual total result;
- predicted home/away score arithmetic.

Score formulas must remain:

```text
predicted_home_score = (predicted_total + predicted_margin) / 2
predicted_away_score = (predicted_total - predicted_margin) / 2
```

## 7. Validate the Step 14 calibration

Confirm the new:

```text
step14_probability_calibration.json
```

contains calibrations for:

- moneyline;
- spread;
- total.

The live projection scripts depend on this artifact for:

```text
home_win_probability
away_win_probability
home_cover_probability
away_cover_probability
over_probability
under_probability
```

## 8. Confirm live projection compatibility

After retraining, verify both live scripts still match the new Step 11 schema exactly:

```text
docs/win/football/nfl/scripts/01_merge/projection_week1.py
docs/win/football/nfl/scripts/01_merge/projection.py
```

They must load the newly generated models/schema/calibration and construct the exact same feature names and order used during Step 11 training.

Do not begin the next season with a newly trained model until this compatibility check passes.

## 9. Preserve the old artifacts before major model-design changes

If changing the feature specification, CatBoost parameters, training period, or calibration methodology, save the prior models/schema/backtest/calibration before overwriting them.

A normal annual retraining with the same methodology does not require a separate script version.

---

## Annual short version

```text
1. Extend training Steps 1–10 to include the completed season.
2. Confirm historical_core_<new season>.csv is produced.
3. Run NFL Annual Training Models.
4. Verify Step 11 models + 260-feature schema.
5. Verify Step 13 chronological backtest.
6. Verify Step 14 probability calibration.
7. Confirm projection_week1.py and projection.py match the new schema/models.
8. Only then use the new models for the next NFL season.
```
