# Released-label adaptation add-on

**Status:** Exploratory, outcome-informed add-on locked on 15 November 2026 at 17:38 Asia/Karachi. It follows the completed model-sensitivity and XOR experiments. It is not a preregistration made before the earlier results.

## Question and rationale

Does refitting a frozen logistic model on labels only after their scheduled release recover batch accuracy under a conditional concept change, and how does recovery differ for a linear model under an XOR concept complement?

This question was selected after version 1 showed that the input-only detector could not identify conditional-only changes and after the extension study established a new synthetic XOR task. The two cases are C3 from the original synthetic family and X3 from the XOR family. This is a narrow mechanism study, not evidence of general online adaptation.

## Locked design

- **Conditions:** `v1_synthetic/C3` (post-onset label-probability complement, features unchanged) and `xor_synthetic/X3` (post-onset XOR concept complement, features unchanged). The shared generators, 12,000-row streams and 24 batches of 500 are unchanged.
- **Seeds:** development seeds 290001–290005; final seeds 300–329 for both conditions. These do not collide with previous development or final seeds. All final seeds remain in analysis; no replacements.
- **Partitions:** retain 4,000 training, 2,000 calibration and 2,000 reference rows from each task's baseline law. Scaling and the initial model are fit on the 4,000 training rows only. Calibration/reference labels are not used for adaptation. Monitoring labels enter the learner only through the release broker.
- **Frozen arm:** one `LogisticRegression(C=1, max_iter=500, random_state=seed)` fit on the initial training partition and never updated.
- **Adaptation arm:** the same initial logistic fit and training-only scaler. After any monitoring labels are released, refit the same logistic specification from scratch on the original training rows plus every unique monitoring example released strictly before the next batch prediction. No unreleased label, calibration row, reference row, hyperparameter search or adaptive preprocessing is permitted. The update is applied only after the current prediction is persisted.
- **Label clocks:** paired delays 0, 5 and 20. For forecast batch `t` and delay `d`, the logged monitoring-label budget available before prediction is exactly `500 * min(24, max(0, t-d-1))`. A same-clock release follows prediction persistence, including delay zero. Drain through clock `24+d` without new input. Each delay eventually releases the same 12,000 stream labels; at a common input clock the available budget can differ and is reported explicitly.
- **Pairing:** frozen and adapted predictions use the same seed, condition, batch IDs and feature rows. The broker issues the same scheduled label IDs for each paired arm; only the adapting model receives released examples for fitting. Delay comparisons are described at their exact budget and evidence age; no unequal-budget adaptive-method leaderboard is formed.
- **Primary outcome:** paired seed-level difference in accuracy between adaptation and frozen logistic across post-onset batches 9–24. Secondary summaries: pre-onset batches 1–8, late post-onset batches 17–24, macro F1, update counts and exact released-label counts by prediction clock. No confidence calibration, harm-probability or detector-success claim is added.
- **Uncertainty:** 2,000 paired bootstrap replicates over complete final-seed vectors. Seeds, not batches, delays or method rows, are the resampling unit. Report all per-seed differences and denominators.
- **Separation:** online predictions and training-label budgets contain no current labels. Released labels are stored separately with `available_at`. Evaluator-only accuracy is joined after predictions are persisted. The adaptation arm may access only the broker's due release event.
- **Stop rules:** preserve failures and stop interpretation on ID overlap, future-label access, same-clock leakage, missing release records, unequal paired release IDs, queue non-drainage, seed omissions, convergence failures or non-finite predictions. Do not alter the seed set, stream size, thresholds or conditions after results are inspected.

## Output schemas frozen before final seeds

- `online_batches.csv`: task, seed, condition, delay, method, cohort, batch size, number of monitoring labels available before that prediction, training rows, fit count, released-monitoring-ID digest, latest allowed `available_at`, persistence marker, and prediction/probability means. It contains no label, accuracy, F1, correctness or harm column.
- `predictions/<original|xor>/<seed>/<condition>/delay<d>/batch<bb>.npz`: IDs and frozen/adapted predictions and probabilities for the batch, written before release. `features/` and `evaluator_truth/` use the same short task directory names.
- `features/` and `evaluator_truth/`: separate ID-aligned feature and label caches. These are offline reproducibility/evaluator artifacts, not the online predictor interface.
- `release_log.csv`: one row per scheduled batch release, with source batch, availability/release clock, last persisted forecast, drain marker, row count and ordered ID digest. `released_labels.csv` contains the released IDs and labels with their `available_at` clock.
- `fit_audit.csv`: per-method forecast batch, number of released examples used, training row count, fit count, released-ID digest and latest label availability used. No fit may include `available_at >= forecast_batch`.
- Evaluator outputs: `evaluator_batches.csv` contains accuracy, macro F1 and class denominators after exact ID joins; `paired_seed_differences.csv` contains per-seed paired deltas by pre/post regime; `paired_bootstrap.csv` contains means and percentile bounds from 2,000 seed-vector replicates; `budget_by_clock.csv` records exact release budget by delay and batch; `validation.json` records join, budget, order and drainage checks.

The five development seeds completed before this output inventory was written. Their outputs already conform to these schemas; the clarification does not change the locked estimand, data, seeds, model, timing or uncertainty method. The final run is not started until this schema inventory and its implementation pass targeted tests.

## Interpretation limits

The experiment uses one linear model, two selected synthetic conditional-change cases, and a cumulative refit rule. The frozen baseline receives no labels for fitting; this is a label-use versus no-update contrast, not a comparison of adaptation algorithms under identical consumed-label counts. The exact broker availability budget is matched across paired runs and logged at every prediction. Results do not establish deployment safety, external-domain generality, calibration repair or human benefit.
