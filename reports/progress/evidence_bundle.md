# Phase 2 evidence bundle (auto-generated 2026-09-29)

Detectors live: ks, wasserstein, js, adwin-inspired (+ mmd-rbf, joint-logreg offline).
Runner evidence columns: ks_global_D, w_global_norm, js_global_JS, adwin_global, macro_f1.

## Aggregates
- e1_detection.csv — S0-S7 detection/delay/burden
- e1_compare.csv — KS vs Wasserstein rates + thresholds
- e2_localization.csv — KS top3 + JS top3_js/p@k_js/r@k_js vs truth
- e3_proxy.csv / e3_delay.csv — MLOps action gate (Gap G2)
- multiseed_detection/localization.csv — held-out seeds with CIs
- e6_window.csv / e6_refpolicy*.csv / e6_refstability.csv — sensitivity + reference policy
- real_burden.csv — elec2/covtype arms (burden + score/F1, no event claims)
- heatmap_S2_s0.csv — feature x time KS matrix (Gap G1/G3 layer)
- offline_compare.csv — JS/ADWIN/MMD/joint side-by-side S0/S2/S4/S6/S7
- dde_demo_S4.csv — DDE lift (1.0→5.0) + rest scores on correlated group
- budget.csv — short dry-run projection (2 jobs, ~1.1h/147 jobs +25%)

## Figures
- Dashboard_S2/S4_s0.png — unified 3-section view (scores + heatmap + F1)
- Aligned_REAL_elec2/covtype.png — 5-panel real-stream views (KS/W/JS/ADWIN/F1)
- Aligned_S2_adwin.png — 5-panel KS/W/JS/ADWIN/F1 (new)
- Aligned_S2_js.png, Aligned_S0/S6/S7.png, Aligned_REAL_elec2.png
- Heatmap_S2_s0.png, E1_tradeoff.png, Panels_failure.png

## Data
- processed/elec2_ordered, covtype_subset100k, tierA-S0..S7-s0 (400k rows)
- manifests: elec2, covtype, covtype_subset, tierA (+ weather_PENDING)
- configs/datasets/{elec2,covtype,weather,synthetic_tierA}.yaml
- configs/scenarios/{E1,E2,E3,E6,full}.yaml (E1/E2 now read by scripts); methods/baselines.yaml
- src/drift_monitoring/data/real.py shared loader (run_real refactored onto it, outputs identical)

## Scripts (new this session)
- scripts/materialize_tierA.py, scripts/export_heatmap.py
- scripts/compare_offline.py, scripts/demo_dde.py
- scripts/build_dashboard.py (unified 3-section Gap-G1/G2 view)
- scripts/probe_tierA_scale.py (full-scale single stream: 50k rows, 60 windows, ~32s/15MB)

## Tests
- pytest 13 passed (invariants + P0 JS/pipelines + P2 ADWIN/MMD/DDE/joint + calibration/matching)
