# Execution log — multi-seed pilot (2026-09-28, laptop CPU)

- Ran `scripts/run_multiseed.py`: 8 templates × 5 seeds = 40 streams, pilot rows
  (2k/2k/6k), window 500 → 33 s total (resumed once after interruption; checkpoint
  skip worked — 17/40 reused, no duplicates).
- Threshold recalibrated on pooled S0 (5 seeds): 0.0936 (≈ single-seed 0.097 — stable).
- Detection over seeds: S1–S6 rate 1.00, CI [1.00,1.00], delays 0–1w;
  S0/S7 rate 0.00, 0.2 mean false alerts.
  CAUTION: CIs are tight because pilot shifts are large (easy cases by design).
  Confirmatory must add near-threshold severities where variance is informative;
  do NOT cut seeds to <5 on the basis of this table.
- Localization over seeds: mean recall 1.0 / AP ≈1.0 all templates; P@3 0.33–0.67
  (proxy dilution stable across seeds).
- Outputs: `results/runs/_multiseed/` (40 evidence+truth+meta),
  `results/aggregates/multiseed_detection.csv`, `multiseed_localization.csv`.
- Chain verified: pytest 4 passed (re-run below).
