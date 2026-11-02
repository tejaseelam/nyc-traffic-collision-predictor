# Modeling handoff

## Current status

The cleaned and integrated data have been converted into a model panel with one row for every 500-meter NYC grid cell and six-hour period from 2021 through 2024. Periods without crashes are included.

Run these commands from the project root to rebuild the panel and its chronological splits:

```bash
python scripts/6build_model_panel.py
python scripts/7split_model_panel.py
```

## Files to use

| File | Dates | Use |
|---|---|---|
| `model_panel_train_2021_2022.parquet` | 2021-2022 | Fit candidate models |
| `model_panel_validation_2023.parquet` | 2023 | Compare features, models, settings, and thresholds |
| `model_panel_test_2024.parquet` | 2024 | Run once for the final evaluation |

The files are created in `data/processed/`. That directory is intentionally excluded from Git, so each team member must run the pipeline or receive the generated files separately.

## Verified split totals

| Split | Rows | Crashes | Six-hour periods | Grid cells |
|---|---:|---:|---:|---:|
| Training | 10,628,800 | 193,963 | 2,920 | 3,640 |
| Validation | 5,314,400 | 89,011 | 1,460 | 3,640 |
| Test | 5,328,960 | 83,643 | 1,464 | 3,640 |

Together, the files contain all 21,272,160 panel rows and all 366,617 assigned crashes.

## Targets and features

- Use `crash_count` for a count model.
- Use `has_crash` for a binary rare-event model.
- Never include `crash_count` or `has_crash` among the input features for the other target.
- Available inputs include grid location, borough, calendar fields, daily weather, land coverage, and area-level speed-limit summaries.
- Traffic is not included in version 1 because the exact-hour match coverage was too sparse.

## Recommended workflow

1. Establish a historical-average baseline using only the training file.
2. Fit candidate models with the training file.
3. Use the validation file for every modeling decision.
4. Select one final approach and decision threshold.
5. Retrain that approach with the combined training and validation files.
6. Evaluate it once on the 2024 test file and report those results.

Do not inspect test performance repeatedly or use the test file to select features, models, settings, or thresholds. Any historical-average or lag feature must be calculated only from data earlier than the period being predicted.

## Validation and known limitations

`model_panel_split_summary.json` records the split boundaries and validation totals. Automated tests confirm that the split preserves rows and crashes, keeps complete grid coverage, and places boundary timestamps in the correct year.

Known limitations include 35,679 crashes without valid coordinates, 73 valid coordinates outside the retained NYC grid, present-day speed-limit data that may not represent every historical year, and traffic data that is not yet suitable for the main model.
