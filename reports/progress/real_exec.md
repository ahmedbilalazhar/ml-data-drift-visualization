# Execution log — real data (2026-09-28, laptop CPU, 50 s)

## Elec2 (Tier B, burden only, 54 test windows)
- 54/54 windows alert at the synthetic S0 threshold (KS range 0.59–1.00).
- Verdict: CALIBRATION-TRANSFER FAILURE (predicted, roadmap Sec. 9.1). Synthetic
  thresholds do not transfer to autocorrelated natural streams. NO false-positive
  claim made (no event truth); reported as burden + meanF1 0.608 association.
- Consequence: natural-stream thresholds need natural null calibration (no verified
  stationary catalog exists) or a reference-refresh policy — open E6 ablation.

## Covertype + controlled shift, Elevation/Aspect/Slope ×1.5σ at test row 40% (Tier C)
- Detection: delay 0 (matched window 40). Localization: top-3 EXACTLY the
  intervened vars (P@3 = 1.0) — ranking works on realistic feature structure.
- BUT 40/40 pre-onset windows also alert (pre-onset KS max 0.74): the fixed
  distant train reference saturates on real data (natural train→test
  non-stationarity + 54-dim max aggregation). Persistent post-onset alerts are
  correct persistence, not false alarms — the 99 count mixes both; kept with
  this caveat, not silently redefined.
- Consequence: threshold/aggregation recalibration per data family + fixed-vs-refresh
  reference ablation are required before any real-data detection claim.

## Files
`results/runs/_real/` (2 evidence+truth) · `results/aggregates/real_burden.csv`.
Multiclass path verified: macro-F1 + tree/logreg run native 7-class, no crash.
