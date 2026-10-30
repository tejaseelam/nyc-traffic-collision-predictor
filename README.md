# nyc-traffic-collision-predictor
Predicting NYC traffic collision risk using ML

## Recent data pipeline updates

- Fixed collision cleaning to preserve the string IDs returned by NYC Open Data.
- Added validation to prevent integrations from silently adding or removing crashes.
- Corrected traffic matching to use the crash cross-street field.
- Added project dependencies and automated tests for weather, traffic, and speed-limit merges.
