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
