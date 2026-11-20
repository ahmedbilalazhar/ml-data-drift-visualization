# UCI 224 split/model sensitivity extension

**Status:** exploratory, outcome-informed extension; protocol frozen before its execution. It reuses the verified public archive and does not alter the original external evaluation.

## Question

Does the external result (and correctness-estimation failure) depend strongly on one random historical split or one classifier? Compare six stratified 60/20/20 historical splits, logistic regression and a fixed depth-six decision tree, and two acquisition-history windows. This remains one dataset; split repetitions are sensitivity analyses, not independent external replications.

## Frozen design

- Dataset: UCI Gas Sensor Array Drift, archived source SHA-256 `91e8f466f202e7a093d657673ce47311c3e90416f7df3057966058961c351fe4`; batch order is coarse acquisition order, not within-batch timestamps.
- History windows: batch 1 only (monitor batches 2–10), and batches 1–2 pooled (monitor batches 3–10). Compare models on shared batches 3–10 only. The second window has a larger label/data budget and is not a fair direct win over the first.
- Six predeclared random states: 420001–420006. Stratified 60% training, 20% correctness calibration, 20% labelled reference, using the same split code for each history window.
- Models: multinomial `LogisticRegression(C=1, max_iter=2000, random_state=2242026)` with training-only `StandardScaler`; `DecisionTreeClassifier(max_depth=6, min_samples_leaf=25, random_state=2242026)` on raw features. Class predictions use each model's `classes_[argmax(probability)]`.
- Correctness estimator: logistic calibration of maximum class probability, fit only on its designated calibration partition against fixed-model correctness. Also report raw mean maximum probability and labelled-reference accuracy as comparators. No monitored labels enter any fit.
- Outcomes by model/split/cohort: accuracy, fixed-six-class macro-F1, classwise precision/recall/F1/support, raw and calibrated estimate, each estimate's absolute error and signed error, calibrated correctness Brier score, and runtime. Predictions and truth remain separate in outputs.
- Report per-configuration outcomes plus descriptive mean and SD over six splits. Do not bootstrap/p-value split seeds as though they were independent datasets; emphasize shared-data and one-domain limits.
- No tuning, seed replacement or exclusion. A fit or calibration failure fails the run visibly.

This addresses robustness to historical partitioning and model choice only. It cannot solve the fundamental identifiability of accuracy under arbitrary conditional shift, certify intervals or establish broader domain transfer.
