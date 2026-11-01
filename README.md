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
