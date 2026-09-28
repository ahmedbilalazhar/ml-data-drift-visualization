# Phase-2 results chapter — PILOT DRAFT (not confirmatory)

All numbers pilot-scale (2k/2k/6k rows, 1 seed/template, window 500, S0 threshold
0.097). They validate the pipeline and shape the frozen protocol; confirmatory
values require G4 sign-off + Colab inventory (5 held-out seeds, independent
calibration, 1000-window stationary validation).

## E1 — Detection (RQ1)
S0-calibrated threshold 0.097 → S1–S6 detected at delay 0–1 windows
(S1 8 alerts d0; S2 7 alerts d1; S3 5 d0; S4 7 d1; S5 8 d0; S6 8 d0).
S0: 1/12 false alert (small-sample noise — motivates full calibration).
S7: 1 alert, scored NOT detected (blind-spot control, by design).
Table: `results/aggregates/e1_detection.csv`; Fig: `E1_tradeoff.png`.

## E2 — Localization (RQ2)
Raw KS ranking: recall 1.0 / AP 1.0 on every template with feature truth
(S1 x11; S2/S3/S4 x0,x1; S5 x10; S6 x0-group). Precision@3 0.33–0.67 —
correlated proxies (x2) co-rank, reported as proxy-selection alongside
variable scores. S0/S7 empty-truth flagged and excluded from means.
Table: `e2_localization.csv`.

## E3 — Harm + delay (RQ3)
Frozen tree, macro-F1, degradation −0.05×2w. Proxy rule: S0 monitor;
S1/S5 investigate (alert, no drop — correctly NOT retrain); S6 update@d0
(dF1 0.074) → investigate@d2 (0.043, delay hides harm); S7 request-labels@d0/d2
(dF1 0.19/0.15) but monitor@d5 (0.011 — FALSE reassurance miss).
Tables: `e3_proxy.csv`, `e3_delay.csv`.Lesson: detector outputs are
delay-invariant (frozen core); actionability is not.

## E6 — Sensitivity
S2 windows 250/500/1000 → KS_max 0.61/0.54/0.54 (500 locked). Table `e6_window.csv`.

## Resources (laptop pilot)
E0 25.5 s / 33 MB peak; E6 units ≤4.1 s. Full inventory is Colab-gated
(`notebooks/colab/01_heavy_inventory.ipynb`); no throughput claims made.

## Failure panels
`Panels_failure.png`: (a) S1 alert≠harm, (b) S7 harm≠alert, (c) S7@d5 hidden.
These bound every positive claim above and motivate the aligned evidence-status
indicators + heuristic checklist (`studies/rubrics/heuristic_checklist.md`).

## Real data (laptop, `real_burden.csv`, `e6_refpolicy*.csv`)
- Elec2 (54w): 54/54 alert at synthetic thr — calibration-transfer failure;
  burden + meanF1 0.608 only. Covertype shift: delay 0, P@3 = 1.0, but 40/40
  pre-onset windows alert (fixed distant reference saturates; persistent
  post-onset alerts are correct persistence).
- Reference ablation: refresh halves mean KS (elec2 0.85→0.37, covtype 0.66→0.35)
  yet alert counts barely move (54→53, 100→99) — score LEVELS are policy-dependent
  but the transferred threshold saturates under either. Thresholds must be
  calibrated per (stream, policy) on appropriate null data; none exists for natural
  streams, so threshold alerting there stays gated. scipy ks_2samp falls back to
  asymptotic method on large samples (warning only, valid).
