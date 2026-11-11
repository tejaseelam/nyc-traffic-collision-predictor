# Evaluation Experiment Instructions

Use these directions for the feature-removal, reporting-delay, and subgroup
experiments. The goal is for everyone to return results in the same format so
they can be combined without manual reformatting.

## What is stored in Git

This folder contains only:

- these shared instructions; and
- `evaluation_results_template.csv`, which everyone should copy before starting.

Do **not** upload individual experiment result CSVs or unfinished charts to Git.
Send those files directly to Blakely. After the files have been checked and
combined, one official `evaluation_results.csv` can be added to the repository.

## Workflow for each team member

1. Copy `evaluation_results_template.csv` and rename the copy for your assignment.
2. Delete the `REPLACE_...` text and add one row for every model and group tested.
3. Run the assigned experiment using only the 2023 validation split.
4. Fill in every required metric, count, and note.
5. Send Blakely the completed CSV, chart files, two or three main findings, and
   any limitations or problems you encountered.

Suggested filenames:

- `blakely_feature_removal.csv`
- `yashwant_reporting_delay.csv`
- `maria_model_comparison.csv`
- `teja_borough.csv`
- `karan_season_weather.csv`
- `ashik_time_risk.csv`

## Required rules

- Use **2023 validation data only** for all current experiments.
- Do not open, inspect, tune on, or report results from the 2024 test data.
- Use the current full LightGBM model as the main model benchmark.
- Compare with the rolling cell-and-time baseline and logistic regression when
  the experiment applies to all three models.
- Average precision is the main performance metric.
- Also report ROC AUC and top-5% capture.
- Report `n_rows` and `n_positive` for every evaluated group.
- Enter metrics as decimals from 0 to 1, not as formatted percentages.
- Leave unfinished numeric results blank. Do not use zero as a placeholder.
- Use `pending`, `complete`, or `error` in the `status` column.
- Explain errors, unusual samples, or interpretation warnings in `notes`.

## Top-5% capture definition

`top5_capture` is the share of **positive cell-periods** captured in the
highest-risk 5% of evaluated cell-periods. A positive cell-period is a
six-hour grid-cell period containing at least one crash. It is not the raw
number of crashes.

Use the `top5_scope` column to record how the top 5% was selected:

- `global`: use the citywide top-5% selection or cutoff. Use this for overall
  model comparisons, feature removal, and reporting-delay experiments.
- `within_group`: rank only the rows inside that subgroup and select its top
  5%. Use this for fair comparisons among boroughs, time periods, seasons,
  wet/dry conditions, and historical-risk groups.

When practical, subgroup owners may report both definitions as two separate
rows. Never combine global and within-group results without labeling them.

## CSV columns

| Column | Required content |
| --- | --- |
| `experiment` | `feature_removal`, `reporting_delay`, `borough`, `season`, `weather`, `time_period`, `historical_risk`, or `model_comparison` |
| `model` | `baseline`, `logistic`, or `lightgbm` |
| `group` | Specific condition, such as `no_weather`, `Brooklyn`, `winter`, or `7_day_delay` |
| `average_precision` | Main rare-event ranking metric, from 0 to 1 |
| `roc_auc` | Secondary ranking metric, from 0 to 1 |
| `top5_capture` | Share of positive cell-periods captured, from 0 to 1 |
| `top5_scope` | `global` or `within_group` |
| `n_rows` | Number of evaluated cell-periods |
| `n_positive` | Number of evaluated cell-periods containing at least one crash |
| `split` | Always `2023_validation` for the current experiments |
| `status` | `pending`, `complete`, or `error` |
| `notes` | Short method detail, finding, warning, or error explanation |

Use consistent group names with underscores, for example:

- `no_crash_history`, `no_time`, `no_roads_speed`, `no_location`, `no_weather`
- `0_day_delay`, `3_day_delay`, `7_day_delay`, `14_day_delay`
- `Bronx`, `Brooklyn`, `Manhattan`, `Queens`, `Staten_Island`
- `winter`, `spring`, `summer`, `fall`, `wet_days`, `dry_days`
- `12_AM_to_6_AM`, `6_AM_to_12_PM`, `12_PM_to_6_PM`, `6_PM_to_12_AM`
- `lowest_risk_quartile` through `highest_risk_quartile`

## What to send to Blakely

- One completed CSV using the shared columns and naming rules
- At least one chart as a PNG or SVG
- Two or three short findings written in plain language
- Any limitations, errors, or decisions that could affect comparison
- The script or branch name used to produce the results, if applicable

Before sending, confirm that the CSV opens correctly, contains no 2024 results,
has no zero placeholders, and clearly labels every top-5% calculation.
