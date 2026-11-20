# Seed-held-out five-batch harm-forecast experiment

**Status:** outcome-informed synthetic forecast extension, protocol fixed before generating the reserved seeds. It is separate from the earlier persistence analysis.

## Target and split

Use the existing binary synthetic generator/model law, seven conditions, 24 batches, and frozen core monitor. Development seeds are 950000--950029; held-out evaluation seeds are 950100--950129. Train a single pooled logistic forecast model on development seeds only, then evaluate once on disjoint evaluation seeds. Do not include condition ID or seed ID as predictors. The seed, condition and delay columns are only grouping keys.

At forecast batch `t`, predict the binary target that mean realised accuracy over the next five complete batches (`t+1` through `t+5`) falls more than 0.05 below the same-stream historical reference accuracy. Only t=1..19 have a complete future target. This is a future-cohort target; it is not the current batch's harm flag.

## As-of predictor sets

Reproduce delayed-label availability at 0/5/20 batches with same-clock release after prediction. Build three fixed models: (1) `time_only` uses normalized current batch and delay; (2) `unlabeled_monitor` adds current calibrated estimate, input alarm counts/flags, smallest adjusted p-value and concern score; (3) `full_released` adds latest released accuracy, mean accuracy of up to five most recently released batches, their harm fraction, released-row count and age of latest released batch. All features must be available before the t forecast. Do not include synthetic condition ID, current evaluator truth or any target-window data.

Fit each `StandardScaler` and `LogisticRegression(C=1, max_iter=1000)` only on development-seed rows. No threshold tuning. Compare with development-prevalence constant, last released batch's harm flag (development prevalence before any release), and the current concern score as a nonforecasting comparator.

## Analysis

Report ROC-AUC, average precision, Brier score and log loss overall, by delay and condition. Report default 0.5 threshold measures only descriptively. Use paired bootstrap resampling of the 30 complete evaluation seeds (2,000 replicates) for probability-score contrasts; rows, conditions and delays are not independent replicates. Evaluate incremental value from time-only to unlabeled to released-label feature sets. The model learns within these seven known synthetic laws; it is not a test on unseen regimes or natural streams.

All online feature rows and evaluator target rows are separate artifacts. Validate every feature row joins exactly one target only in evaluation, the future window begins at t+1, release availability is strictly before forecast t, and no evaluation-seed target enters model fitting. No future interval or population-risk certificate is claimed. Outcomes may show that future harm remains unpredictable; do not alter the locked model after evaluation.

## Outcome and interpretation (held-out run)

The frozen run completed across all 60 seeds, seven conditions and three delays (416 seconds on the local Anaconda environment). It produced 11,970 held-out forecasts per fitted predictor. The held-out target prevalence was 0.418. Time-only achieved ROC AUC 0.640, average precision 0.483 and Brier 0.231. The unlabeled monitor set achieved 0.838, 0.760 and 0.166; adding available released-label history achieved 0.865, 0.818 and 0.149. The development-prevalence constant had Brier 0.243; current concern as a raw, uncalibrated score had Brier 0.327; the last available harm flag had Brier 0.258. The latter heuristic scores are not calibrated probabilities.

Paired bootstrap resampling of 30 whole held-out seeds gave Brier reductions of 0.0644 (95% percentile interval 0.0603 to 0.0674) for time-only versus unlabeled predictors, 0.0169 (0.0160 to 0.0179) for unlabeled versus released-label predictors, and 0.0813 (0.0776 to 0.0839) for time-only versus released-label predictors. Each resample retains all conditions and delays for a seed. These intervals quantify seed variation only under the fixed seven known generator laws; they do not account for uncertainty over new mechanisms.

The result supports a prospective forecast within this synthetic design and shows that both current unlabeled monitoring evidence and already released labels add predictive information. It does not show natural-stream transfer, causal attribution, or analyst benefit. The very strong prediction is expected to depend on the recurring known synthetic laws and is not a distribution-free risk certificate. See the editable report in `../../reports/phase3/future_harm_forecast_20261116_v10/` and the structural checks in `../../tests/phase3/test_future_harm_forecast.py`.
