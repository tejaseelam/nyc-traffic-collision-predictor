# Final model freeze

The canonical validation panel contains **86,527 positive cell-periods** and has SHA-256 `d0d9210d25a882d3aecb19bb2e5d5935bcbd7a2523a263c114dd0ba034e044a8`.

The selected model is **LightGBM (no weather)**. No-weather was not worse on validation and its 95% interval did not show a meaningful disadvantage; selected for simplicity.

Features, settings, and the citywide top-5% evaluation rule are locked in `reports/final_model/model_freeze.json`. The 2024 test was not opened during selection.
