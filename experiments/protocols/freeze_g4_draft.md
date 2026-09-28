# G4 Protocol Freeze — DRAFT for supervisor sign-off (roadmap Sec. 8.5/16.4)

No confirmatory/Colab-heavy run may start until each line is signed.
Pilot values below are FROZEN PROPOSALS from E0–E6 pilot data, not results.

## Frozen settings
- Windows: 500 rows, stride 500, non-overlapping; sensitivity 250/1000 (E6).
- Splits (full): 10k train / 10k calib / 30k test; reference ≤5000 train rows, fixed seed.
- Detectors: KS+Holm (primary) + normalized Wasserstein (secondary); same universe,
  reference, windows, calibration for all comparators. Categorical chi-square reported.
- Operating point: ~10 raw alerts / 1000 stationary windows, calibrated on
  INDEPENDENT stationary streams (pilot thr 0.097 is illustrative only — recalibrate
  on full calibration set, validate on held-out seeds).
- Event matching: abrupt = first alarm within 5 windows of onset; gradual S2 =
  start through 5 windows after transition end; one-to-one; misses explicit;
  S7 scored as blind-spot control (feature recall N/A, diagnosis via labels).
- Localization: display top-3, full ranking on demand; P/R/F1@k + AP; empty-truth
  policy (S0/S7 flagged, excluded from mean precision with reason stated).
- Delays: 2 main; 0/5 sensitivity; common release-boundary (predict-then-release).
- Performance: macro-F1 primary; degradation = −0.05 absolute persisting 2 eligible
  windows (pilot rule — confirm against calibration noise before locking); single-class
  windows = NaN (never zero); F1 computed only on released labels.
- Tuning: ≤6 dev configs per method/model/family, dev seeds only; NO test tuning.
- Seeds: 5 short-synthetic per template (dev→confirm split); natural streams = 1
  chronology (no fake seeds); ablations capped at 24 added jobs.
- Exclusions: partial windows flagged; failed jobs keep status + reason, never silently
  replaced; aggregation uses inclusion manifest with denominators.

## Sign-off
| Item | Owner | Date | Signature |
|---|---|---|---|
| Matching + metrics | ___ | ___ | ___ |
| Threshold + calibration plan | ___ | ___ | ___ |
| Seed inventory + budget | ___ | ___ | ___ |
| Heuristic checklist reviewer | ___ | ___ | ___ |
