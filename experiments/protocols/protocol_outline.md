# Protocol outline (freeze before confirmatory; roadmap Sec. 8.5)

- Windows: 500 rows, stride 500 (non-overlapping); sensitivity 250/1000.
- Splits: 10k train / 10k calib / 30k test (E0 pilot scales down; same code path).
- Reference: <=5000 train rows, fixed seed; stability vs subsampling tested.
- Operating point: ~10 raw alerts / 1000 stationary windows (calibrated on
  independent stationary streams, validated on held-out seeds).
- Matching: abrupt = first alarm within 5 windows of onset; gradual = start
  through 5 windows after transition end; one-to-one; misses explicit.
- Localization: top-3 display, full ranking on demand; P/R/F1@k + AP; empty-truth policy.
- Delays: 2 windows main; 0/5 sensitivity; common release-boundary convention.
- Performance: macro-F1 primary; degradation = -0.05 abs persisting 2 eligible
  windows (pilot rule, lock after noise check); single-class = NaN (not zero).
- Tuning: <=6 dev configs per method/model/family; no test tuning.
- Seeds: 5 short-synthetic pilot confirmatory start; final N from precision needs.
