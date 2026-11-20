# Phase 3 literature-baseline fidelity audit — 17 November 2026

This is a source-to-implementation comparability check and a versioned baseline update. No published performance number is transferred into the IDM-Viz results. The original Phase 2 submission and its baseline choices remain unchanged.

## Methods named in the strengthening specification

| Reference method | What the source describes | Project implementation | Fidelity and allowed comparison |
|---|---|---|---|
| DriftLens | An unsupervised drift detector based on deep-learning representations with a real-time monitoring design. | Three-component PCA fitted on training features, then coordinate-wise KS tests. | **Not reproduced.** PCA is a fixed linear representation and lacks DriftLens' learned deep representation and complete detector protocol. Treat only as the project's PCA-anchor comparator; do not call it DriftLens or transfer published results. |
| Domino drift-effect approach | A high-dimensional feature-drift method that models a correlation-mediated “domino” effect and estimates feature-drift probabilities. | Maximum pairwise correlation change with a history-calibrated threshold, evaluated on a designed marginal-preserving dependence intervention. | **Not reproduced.** Pair scoring is a targeted dependence diagnostic, not the paper's probability model or end-to-end domino method. Compare only on the explicitly shared synthetic dependence target, and report our own outcomes. |
| ADWIN / ADWIN2 | An adaptive window whose size changes online; ADWIN2 uses compressed buckets and compares subwindow means. | The earlier `ReleasedErrorADWINStyle` remains an internal comparator. A separate River 0.27.0 `river.drift.ADWIN` baseline consumes row-level errors from selected, delayed 500-row cohorts under the same budgets and release clocks. A post-hoc sensitivity adds fixed, label-independent audits at batches 1 and 12. | **Canonical implementation evaluated under a narrow replay.** The original one-probe run detected in 212/5,040 policy cells. The corrected two-probe sensitivity detects in 273/630 strict-cap cells at delays 0 and 5 (versus none for one probe), but average strict-cap audits rise from 1.00 to 2.29/2.38; this is not an equal-spend detector comparison. Neither schedule detects at delays 20/100 within 24 batches. The sensitivity reuses the same cached streams and is exploratory, not independent confirmation or a general property of ADWIN. |
| KS, persistence, raw Average Confidence, ATC-style | Internal controls with explicitly specified formulas in this project. | Implemented in the frozen experimental pipeline and reports. | These are the defensible primary matched controls in the current evidence package. “ATC-style” remains qualified; it is not presented as a reproduction of every published ATC protocol. |

## Comparability decision

DriftLens and Domino remain unreproduced. Canonical ADWIN is now a named baseline using the River implementation, with the budget and delayed-label rules made explicit. The comparison reuses cached model predictions from the same synthetic evaluation; it is not a new dataset or a fresh model rerun. Its strict-cap arm cannot progress beyond one probe because ADWIN has no second sampled cohort on which to detect a change; its exact-quota arm has additional feedback because it must fill the quota. This scope does not make the baselines interchangeable.

The missing DriftLens and Domino benchmarks remain limitations and prevent claims of superiority over those systems. The ADWIN replay is constrained by one synthetic stream family set and one cached model. The two-probe schedule improves availability of supervised feedback only at short delays, at additional strict-cap audit use; it does not establish equal-spend superiority or general superiority/weakness of ADWIN. River's documented defaults and the `clock=1` choice are recorded in the experiment locks and reports. The first midpoint attempt is retained as superseded because a clock-versus-cohort-budget comparison skipped its second probe; only corrected v2 is analyzed.

## Primary sources

1. S. Greco, B. Vacchetti, D. Apiletti and T. Cerquitelli, “Unsupervised Concept Drift Detection from Deep Learning Representations in Real-time,” arXiv:2406.17813 (2024). [Paper](https://arxiv.org/abs/2406.17813).
2. “Domino drift effect approach for probability estimation of feature drift in high-dimensional data,” *Knowledge and Information Systems* (2025). [Publisher paper](https://doi.org/10.1007/s10115-025-02362-0).
3. A. Bifet and R. Gavaldà, “Learning from Time-Changing Data with Adaptive Windowing,” *SIAM International Conference on Data Mining* (2007). [Publisher record](https://doi.org/10.1137/1.9781611972771.42), [River ADWIN API](https://riverml.xyz/latest/api/drift/ADWIN/), [River implementation](https://github.com/online-ml/river/blob/main/river/drift/adwin.py).

No method is cited as evidence that IDM-Viz achieves the same result.
