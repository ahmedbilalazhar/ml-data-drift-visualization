# Six-hour Phase 3 extension · 20 November 2026

## Question

Do the fixed unlabeled-monitor and released-label harm forecasts retain value across a wider range of conditional shifts, severity levels, and temporal patterns, when tested on fresh paired seeds? The earlier 80-seed study found positive conditional-shift Brier gains in two designed mechanisms, but the monitor forecasts exactly matched the stationary control because the input distributions were held fixed. This extension is a stress test of predictor performance under the specified mechanisms. It cannot show that feature monitoring detects those conditional shifts.

## Locked design

- Train the same time-only, unlabeled-monitor and full-released predictors once on the existing 30 development seeds. Freeze them before evaluation.
- Evaluate fresh, non-overlapping seed blocks beginning at 973300, with 1,200 seeds per block when the wall clock permits. The launcher allows up to ten blocks (12,000 seeds / 540,000 seed-law combinations); the six-hour wall limit may leave the final block partial. Only blocks whose protocol lock is `complete` enter the primary analysis. Seeds and onset batches cycle evenly within each block. All laws for one seed share the same innovations and frozen reference model.
- Laws include a stationary control, C7 posterior flattening at strength 0.35/0.70/1.0, regional label reversal at thresholds -0.75/0/0.75, sensor sign inversion, and a joint feature-boundary stressor. Inputs and labels retain the parent generator semantics. Conditional-only interventions preserve the exact feature arrays. Sensor inversion changes only the observed feature used by the monitor, while labels remain generated from the clean feature array.
- Use the existing 24-batch stream, strict availability rule source + delay < forecast, delays 0/5/20, and five-batch future accuracy target. Report Brier, AUROC, prevalence, paired seed-level gain and stationary-control false-alarm rate at batch level.
- Compare the monitor with time-only and full-released with monitor, within each law and delay (54 contrasts). Use a common seed-cluster bootstrap across those contrasts. Report 95% simultaneous intervals from 2,000 maximum-absolute-centered resamples. Do not pool mechanisms into one headline estimate.
- Verify complete Cartesian coverage within each included block, seed/onset balance, exact shared-input identity for conditional laws, strict release counts, and forecast/target separation. Reconstruct released-label features only from outcomes permitted by source batch + delay < forecast origin.
- Save progress receipts every 100 completed seeds. At six hours, stop cleanly and retain all completed cells with explicit pending counts. Do not alter the design based on results.

## Time budget

| Phase | Maximum |
| --- | ---: |
| Protocol, source lock and smoke validation | 30 minutes |
| Locked evaluation, up to the six-hour wall limit | 6 hours maximum |
| Paired analysis, report, artifact checks and publication | After compute, using remaining available time |

The smoke pilot (seeds 973000-973299) is separate from confirmatory blocks. A different machine or numerical environment is not an independent replication of the data generating process.
