# UCI 224 exact plug-in interval extension

**Status:** exploratory, outcome-informed extension. The original UCI cohort run and its normal-approximation results are preserved unchanged.

## Question

Does replacing the primary external study's continuity-corrected normal approximation with the exact plug-in Poisson-binomial predictive distribution materially change interval width or empirical cohort coverage when calibrated correctness scores and predictions are held fixed?

## Frozen comparison

- Use only the existing UCI 224 day-2 v2 run, its source archive and its prediction caches. Do not refit or tune the classifier, calibrator, thresholds or feature tests.
- Evaluate monitoring cohorts 2–10 (nine cohorts). For each row, use the cached frozen calibrated correctness probability `q_i`; calculate the exact distribution of the sum of independent Bernoulli(`q_i`) variables and take the discrete 2.5% and 97.5% quantiles divided by cohort size.
- Recompute the existing normal approximation with continuity correction from exactly the same `q_i`. Compare lower/upper endpoints, widths and whether each interval contains observed accuracy.
- Treat this as a sensitivity check, not nine independent replications or a calibrated coverage claim. Keep the original interval outcomes visible. Do not use accuracy to select a method.

## Limitations

Both intervals condition on the fitted calibrator and independent correctness indicators. Neither incorporates calibration-parameter uncertainty, within-cohort dependence, historical-reference uncertainty or arbitrary-shift uncertainty. Exact computation removes normal-approximation error only; it cannot repair miscalibration or the narrow one-domain external design.
