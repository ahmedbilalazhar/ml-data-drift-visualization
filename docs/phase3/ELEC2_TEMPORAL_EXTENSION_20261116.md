# ELEC2 second-domain temporal stream extension

**Status:** exploratory external-domain extension, locked before this run. It supplements, and does not replace, the Gas Sensor study or the Phase 2-based synthetic core.

## Dataset and admission

Use OpenML dataset 151 / Electricity version 1 (`electricity.arff`, file 2419), source SHA-256 `2d86fbc74c69a5c04bcde0119bce280dd6d8efa0115987edc75b77ccf024d7f2`. OpenML marks the dataset public and describes 45,312 rows, normalized date/order fields, electricity-market predictors, and the UP/DOWN label. Original dataset references listed by OpenML: M. Harries, *SPLICE-2 Comparative Evaluation: Electricity Pricing* (1999); J. Gama et al., *Learning with Drift Detection* (2004). River's official dataset wrapper also describes ELEC2 as a binary stream task with 45,312 samples and eight fields.

The source file's row order is **not** monotone in its normalized date field. Therefore, do not treat file order as chronology. Reconstruct a coarse order by normalized date, then by source-order position of each `(date, weekday)` group, then period and original row index; verify date monotonicity and period order within each date/weekday group. Eleven normalized date values contain multiple weekday groups, so within-tie ordering relies on their source-order grouping and the supplied date/weekday fields; this is not an exact calendar timestamp reconstruction. Exclude normalized date from classifier and input-test features because it is used to establish order. Also exclude contemporaneous NSW price as a predictor because it is directly tied to the price-direction target. Use the remaining six predictors: day, period, NSW demand, Victoria price, Victoria demand, and scheduled transfer. Label is UP/DOWN.

## Fixed temporal split and model

After stable temporal sorting, use rows 0–10,799 for classifier training, 10,800–14,461 for correctness calibration, 14,462–18,123 for the labelled accuracy/input reference, and rows 18,124 onward for monitoring. Do not shuffle. Fit `StandardScaler`, the frozen `LogisticRegression(C=1, max_iter=1000, random_state=2242026)`, three-component PCA and three-cluster MiniBatchKMeans on training inputs only. Fit isotonic correctness calibration on calibration rows only, using maximum class probability and correctness under the fixed classifier. Use the last 1,000 reference rows as input-test reference and all reference labels for historical accuracy.

Monitor in chronological batches of 500; retain the final partial batch. At each batch, compute two-sided KS tests for the six inputs plus three frozen PCA coordinates; apply Holm within the nine-test family at alpha .05. Export feature rankings and cluster occupancy as descriptive evidence. For impact, report mean calibrated correctness and exact plug-in Poisson-binomial predictive intervals and concern score using the Phase 3 binary-correctness kernel. This is not a calibrated population-risk interval; serial dependence, calibration uncertainty and arbitrary shift remain unaccounted for.

## Delayed labels and outcomes

Persist prediction probabilities, correctness scores, feature evidence and impact estimates before label release. Replay delays of 0, 5 and 20 monitoring batches from these same caches; same-clock delay-zero release follows persistence, and drain each queue after final input. Online exports contain no current labels, accuracy, F1 or correctness. Evaluator-only outcomes include accuracy, fixed two-class macro-F1, per-class precision/recall/F1/support, Brier score for correctness probabilities, 10-bin ECE, interval coverage/width and alarm status. Report alarms as input-distribution events only. No drift onset, causal feature, or detector recall is known for this real stream.

This adds a second dataset and natural ordered stream but remains one fixed domain/model/history split. It does not address analyst benefit, causal attribution, drift ground truth, multi-domain generality, adaptive recalibration or the general identifiability of accuracy under conditional shift.
