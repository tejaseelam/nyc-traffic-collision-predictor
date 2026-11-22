# Final conclusions

The project’s final model is a LightGBM classifier that uses past crash history, time, borough, grid location, land share, and road/speed-limit context. Weather was removed before the test year was opened because the no-weather model was slightly better on 2023 validation (average precision 0.0610 versus 0.0608) and the 1,000-sample whole-day bootstrap showed the two versions were statistically tied.

On the one-time 2024 test, the final model achieved average precision 0.0562, ROC AUC 0.8079, and captured 23.1% of positive cell-periods in the citywide highest-risk 5%. It remained ahead of logistic regression (0.0542 average precision; 22.4% capture) and the rolling historical baseline (0.0539; 22.3%). The improvement is real but modest: most useful ranking information already exists in a cell’s earlier crash pattern.

Feature removal supports that interpretation. Removing crash history or calendar time caused the largest validation losses. Removing road/speed or location information changed performance only slightly, and removing weather did not hurt it. The subgroup results show the clearest LightGBM gain overnight, while estimates for the lowest historical-risk quartile are uncertain because that group had only 325 positive periods.

Late reports reduce performance gradually rather than changing the model ranking. At a 90-day delay, LightGBM average precision was 0.0602 on 2023 validation and top-5% capture fell by no more than 0.24 percentage points. From 30 days onward, the measured average-precision loss was consistently below zero. The unexpectedly small loss from applying delayed inputs to an unchanged model should be investigated before deployment.

The dashboard should therefore be presented as a screening and prioritization tool, not an exact crash predictor. Even the final model misses about 77% of positive periods outside the highest-risk 5%, and the results do not establish that any feature causes crashes. Future work should test reporting behavior, historical speed-limit accuracy, stronger traffic exposure data, and stability across additional years or cities.
