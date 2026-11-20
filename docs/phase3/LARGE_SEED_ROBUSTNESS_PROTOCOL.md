# Large-seed robustness extension

**Status:** Outcome-informed, prospectively locked 15 November 2026, 22:35 Asia/Karachi. This is a post-v1/v2 robustness study, not a preregistration preceding those results.

## Weakness being tested

The current model sensitivity and XOR task evidence uses 30 final seeds per task. Its conclusions may be unstable to the particular random streams. This extension asks how estimates, calibration/interval behavior, harm concern, and input alarms vary across a much larger number of independent generator seeds under the already fixed protocols. It tests seed sensitivity only; it does not establish natural-domain generality.

## Frozen design

- Tasks: existing `v1_synthetic` and `xor_synthetic` generator families, unchanged.
- Development seeds: 990001â€“990005, checked disjoint from all previous development and final seeds.
- Final seeds: 1000â€“2499 inclusive for each task (1,500 seeds per task). No replacements, early stopping based on outcomes, or exclusions. Any technical failure is retained and causes the run to be marked failed rather than silently skipped.
- Partitions, models, preprocessing, calibration, threshold and condition definitions: exactly those in `V2_EXTENSION_MANIFEST.md` and `src/drift_monitoring/phase3/v2.py`. The logistic model and fixed depth-6 tree use the same observations; each has its own frozen calibrator and reference target. No tuning is permitted.
- Streams: all 7 original-task and 5 XOR conditions; 12,000 rows, 24 batches of 500. Original feature KS, PCA KS, Holm correction, raw/calibrated confidence, exact plug-in intervals, ATC-style threshold and concern score are unchanged.
- Online/evaluator separation: online batch records include only features/IDs-derived monitor output and no current accuracy, labels, F1 or harm. Predictions are scored and written before evaluator-only truth is joined into a separate file. This scale study stores batch summaries rather than per-instance caches; all streams are deterministically regenerated from the locked seeds. It is a separate evidence area from the cache-complete v2 run.
- Primary summaries: seed-level calibrated-AC MAE and bias, raw-AC MAE, interval empirical coverage and width, model-paired accuracy/MAE differences on matched cohorts, input alarm onset match, null alerts, and concern TP/FP/FN/TN/missed-harm counts. Report per-seed vectors and 2,000-replicate seed bootstrap intervals. Resampling unit is the complete seed, not batches, models or conditions. Include denominators and undefined metrics.
- Stop rules: abort interpretation on any omitted/duplicated seed, ID collision, model pairing mismatch, calibration identity failure, online truth field, failed exact harm boundary, non-finite output, convergence warning, or source/protocol hash mismatch. No seed replacement.
- Storage/runtime: aggregate-only summaries to bound storage. Thread pools fixed to one. A five-seed development run must validate contracts and measure elapsed time before the 3,000 task-seed final run. The final run is intended to fit within the current eight-hour work window; if measured timing contradicts this, preserve development diagnostics and mark final evidence incomplete rather than changing the fixed seed count after outcomes.

## Outcome-informed hypotheses

The prior 30-seed study motivates uncertainty quantification for (a) tree/logistic task sensitivity, (b) conditional-only confidence failures, (c) dependence-event detection misses, and (d) interval coverage below nominal. No direction of effect is assumed. Repeating the same synthetic laws at new seeds is replication within those laws, not a new task family or external validation.

## Outputs

`results/phase3/large_seed_robustness_development` and `results/phase3/large_seed_robustness_final` contain locks, environment, online and evaluator batch CSVs, per-seed condition summaries, paired model contrasts and validation. Report and figures are stored separately in `reports/phase3/large_seed_robustness_final`. Existing v1, v2 and adaptation results and packages are read-only.

A contract-development run was completed on seeds 990001-990005 per task; after correcting a reporting denominator check and adding the exact-rational boundary test, the final frozen source will be re-run on the same reserved development seed set in a distinct directory (`results/phase3/large_seed_robustness_development_finalcheck`). This is a code-contract check, not final-seed selection.

The final-check evaluator found a merge-suffix bug in the exact-harm audit; it did not affect the locked runner's generated online or evaluator data. The evaluator was corrected to use the evaluator-side reference columns. A fresh development directory will verify the corrected full runner/evaluator pair before any large-seed run; previous development outputs remain preserved as diagnostics.

