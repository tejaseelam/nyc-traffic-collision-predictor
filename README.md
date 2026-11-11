# nyc-traffic-collision-predictor
Predicting NYC traffic collision risk using ML

## Recent data pipeline updates

- Fixed collision cleaning to preserve the string IDs returned by NYC Open Data.
- Added validation to prevent integrations from silently adding or removing crashes.
- Corrected traffic matching to use the crash cross-street field.
- Added project dependencies and automated tests for weather, traffic, and speed-limit merges.
- Added a 500-meter by six-hour model panel builder that includes zero-crash periods.

## Build the model-ready dataset

After running the earlier download, cleaning, and integration scripts, run:

```bash
python scripts/6build_model_panel.py
```

This creates:

- `data/processed/model_panel_v1.parquet`: one row per grid cell and six-hour period, including periods with zero crashes.
- `data/processed/nyc_grid_500m.parquet`: the grid with borough, land coverage, and speed-limit summaries.
- `data/processed/nyc_grid_500m.geojson`: a map-friendly copy of the grid.
- `data/processed/model_panel_v1_summary.json`: validation totals and the settings used for the build.

The panel covers January 2021 through December 2024. It includes calendar, daily weather, borough, land coverage, and area-level speed-limit information. Traffic is intentionally excluded from version 1 until typical traffic profiles replace the sparse exact-hour matches.

## Create the train validation and test files

After building the model panel, run:

```bash
python scripts/7split_model_panel.py
```

This creates three chronological files for the modeling handoff:

- `model_panel_train_2021_2022.parquet`: use to fit models.
- `model_panel_validation_2023.parquet`: use to compare features, models, and settings.
- `model_panel_test_2024.parquet`: use once for the final evaluation after all modeling decisions are complete.

The script keeps all grid cells from a six-hour period in the same split and validates that the files preserve the original row and crash totals. Its checks are saved in `model_panel_split_summary.json`.

Use the training file to fit candidate models and the validation file to make every modeling decision. After selecting the final approach, retrain it with the combined training and validation data, then evaluate it once on the 2024 test file. Do not use the test file to choose features, models, settings, or a decision threshold.

See `MODELING_HANDOFF.md` for the verified split totals, target columns, recommended modeling workflow, and known limitations.

## Baseline and logistic regression for `has_crash`

After creating the splits, run:

```bash
python scripts/8baseline_logreg.py
```

This reads only the training and validation files and compares, on 2023:

- `cell_rate` and `cell_hour_rate`: historical crash rates per cell (and per cell and six-hour slot) from 2021-2022.
- `cell_rate_rolling` and `cell_hour_rate_rolling`: the same rates updated with every earlier period, so they use the same history as the logistic regression. `cell_hour_rate_rolling` is the baseline every model should beat.
- Logistic regression with calendar, weather, land, speed-limit, and past-only crash-history features, for each value in `--c-values`. The best model is selected by validation average precision.

Crash-history features for a period use only earlier periods. Zero-crash rows are subsampled for fitting (`--negative-sample-rate`) and the predicted probabilities are corrected back to the full-data rate.

Outputs: `reports/baseline_logreg/validation_metrics.json`, `reports/baseline_logreg/coefficients.csv`, the 2023 predictions of every model in `data/processed/predictions/validation_2023_predictions.parquet`, and the fitted model in `data/processed/models/logreg_has_crash.joblib`. The 2024 test file is not used.

## Plot the validation results

After running the logistic regression script (it reads the saved predictions), run:

```bash
python scripts/9plot_logreg_results.py
```

This saves report figures in `reports/baseline_logreg/figures/`: a crash capture curve, precision-recall curves, a calibration plot, the largest coefficients, and a predicted vs observed map. The map needs `data/processed/nyc_grid_500m.parquet` from the panel build and is skipped when that file is missing.

## Tree-based comparison model for `has_crash`

The tree model uses LightGBM, which needs OpenMP. On macOS, install it first with `brew install libomp`. Then, after creating the splits, run:

```bash
python scripts/10tree_model.py
python scripts/11plot_tree_results.py
```

This fits gradient-boosted trees (LightGBM `LGBMClassifier`) with the same setup as the logistic regression: the same training and validation files, past-only crash-history features, burn-in, negative subsampling with probability correction, evaluation measurements, and selection by validation average precision. Early stopping uses a random tenth of the sampled training rows, never the 2023 file. Two feature sets are compared:

- `core`: exactly the logistic regression's features, for a like-for-like comparison.
- `core_plus_location`: the core features plus each cell's grid position, which a tree can split into neighborhoods.

Feature importance is the drop in validation average precision when a feature, or a group of related features, is shuffled. Run `scripts/8baseline_logreg.py` first to include the logistic regression in the importance analysis, the comparison table, and the figures.

Outputs: `reports/tree_model/validation_metrics.json` (results, settings, and runtime for every candidate), `reports/tree_model/model_comparison.csv`, `reports/tree_model/feature_importance.csv`, figures in `reports/tree_model/figures/`, and the fitted models in `data/processed/models/tree_has_crash.joblib`. The 2024 test file is not used.

### Why LightGBM

LightGBM was compared with scikit-learn's `HistGradientBoostingClassifier` and XGBoost under the same setup and tree sizes (commit `4413130`, which has the comparison script). Best 2023 validation results for each library:

| Library | Feature set | Average precision | Recall in top 5% | Fit time |
|---|---|---|---|---|
| scikit-learn | `core` | 0.0606 | 0.232 | 5s |
| LightGBM | `core` | 0.0606 | 0.232 | 3s |
| XGBoost | `core` | 0.0606 | 0.232 | 13s |
| scikit-learn | `core_plus_location` | 0.0608 | 0.234 | 13s |
| LightGBM | `core_plus_location` | 0.0608 | 0.233 | 10s |
| XGBoost | `core_plus_location` | 0.0607 | 0.233 | 25s |

All three are within 0.0002 average precision, so the choice of library does not change accuracy. LightGBM was the fastest to fit.

## Compare the models on 2023

After running `scripts/8baseline_logreg.py` and `scripts/10tree_model.py`, run:

```bash
python scripts/12compare_models.py
```

This reads both 2023 prediction files, checks that every model was scored on the same cell-periods with the same outcome, and measures the rolling cell + time-of-day baseline, the logistic regression, and LightGBM with one set of definitions: average precision, ROC AUC, top-5% capture, and calibration (Brier score, log loss, mean predicted vs observed rate, calibration intercept and slope). It also resamples 2023 by whole days (`--bootstrap-repeats`, default 1000) to give a 95% interval for each model's gain over the baseline.

Outputs go to `reports/model_comparison/`, which is not committed: `maria_model_comparison.csv` in the shared format of `reports/evaluation/evaluation_results_template.csv`, the full metrics and intervals, and three figures. The 2024 test file is not used.

## Run the interactive dashboard

The GitHub Pages site lives in `docs/`. It is a static, dependency-free dashboard that uses precomputed 2023 validation results and never loads the 2024 test split.

After the baseline/logistic and tree modeling scripts have produced their validation prediction files, rebuild the public data package with:

```bash
python scripts/export_dashboard_data.py
```

On macOS, LightGBM also requires the OpenMP runtime (`brew install libomp`).

Preview the site locally without an internet connection:

```bash
python -m http.server 8000 --directory docs
```

Then open `http://localhost:8000`. To publish with GitHub Pages, configure the repository to deploy from the `docs/` directory on the `main` branch. The exported data package is checked to stay below 15 MB, and all displayed model results are labeled as 2023 validation results.
