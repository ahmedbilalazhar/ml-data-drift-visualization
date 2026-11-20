# UCI 224 delayed-label replay extension

**Version:** `uci224-delayed-replay-v1`  
**Status:** exploratory extension, locked before this replay; based on the already reported Day 2/Day 3 outcomes.  
**Scope:** one frozen multiclass classifier, one UCI Gas Sensor acquisition sequence, coarse whole-batch release clocks.

## Motivation

The validated external evaluation showed severe transfer failure: batch-1 calibration substantially overestimated correctness in later acquisition batches, and its plug-in intervals covered 0/9 evaluated cohorts. The report also identified that the external result was retrospective rather than an as-of-time delayed-label replay. This extension addresses only the latter implementation/evidence gap. It does not tune a new model to improve the known failure or turn this one stream into independent replications.

## Frozen inputs and model

- Dataset source: the already admitted UCI Gas Sensor Array Drift archive, SHA-256 `91e8f466f202e7a093d657673ce47311c3e90416f7df3057966058961c351fe4`.
- Keep the original batch-1 stratified train/calibration/reference split, classifier, preprocessing, and predictions from the validated Day 2 study unchanged.
- Use cached class predictions and maximum class probabilities from batches 2 through 10. Do not refit or adapt the classifier.
- Initial correctness calibrator: the same logistic mapping from maximum class probability to binary correctness, fitted on batch-1 calibration labels only.
- Online recalibrator: cumulative logistic regression (`C=1`, `max_iter=2000`, fixed random state) on the same scalar confidence, using batch-1 calibration observations plus only released monitoring outcomes. Its form and settings are fixed; it is an exploratory online baseline, not a presumed improvement.

## Release-clock contract

Forecast batch `b` before incorporating any labels released at or after clock `b`. For delays `d in {0,1,2}` acquisition-batch steps, source batch `j` is eligible for a forecast only when `j + d < b`. Thus delay zero means labels from the immediately preceding cohort may be used; labels from the current cohort are always unavailable for its forecast. Batches are unequal in row count, so delay is explicitly in cohort steps, not elapsed time or number of records. After forecasting batch 10, pending labels may be drained for audit but cannot affect any reported forecast.

## Comparators and outcomes

For each target batch and delay, compare:

1. frozen calibration from batch 1;
2. cumulative released-label logistic recalibration;
3. raw mean maximum class probability;
4. fixed batch-1 reference accuracy.

Report per-batch signed/absolute accuracy-estimation error, mean Brier score on correctness, ECE-10, and release/calibration sample counts. Report equal-batch and row-weighted summaries separately. Preserve per-batch results; do not treat nine batches as independent replications and do not bootstrap them as such. Confidence intervals or significance claims are not warranted from one ordered sequence.

## Integrity checks

- The frozen class predictions and row order are identical for every delay and recalibration method.
- Every forecast artifact is label-free; evaluator truth is stored separately.
- At forecast `b`, all training outcomes have source batch `j` satisfying `j+d < b`; no current/future outcome is eligible.
- Perturbing labels in a cohort that is not yet released must leave all earlier/current forecasts unchanged.
- Label-release and calibration counts are explicit; no label is silently lost or duplicated.
- New outputs use a fresh versioned directory; the Day 2/Day 3 archives and original Phase 2 submission are unchanged.

## Claim boundary

This is a simulated cohort-level release schedule over known historical labels, not actual timestamped label arrivals. It can demonstrate code-level label availability and diagnose whether this one sequence's past released labels improve later correctness estimates. It cannot establish general delayed-label performance, population coverage, causal effects, transfer to another sensor domain, or human benefit. Results are outcome-informed because the original transfer failure motivated the extension; report all comparators, including negative results.

## Reproduction

From the repository root, use the project Anaconda interpreter and a new output path:

```powershell
& C:/Users/hp/anaconda3/python.exe -m pytest tests/phase3/test_uci224_delayed_replay.py -q
& C:/Users/hp/anaconda3/python.exe scripts/run_uci224_delayed_replay.py --source-run results/phase3/external_uci224_day2_v2 --out results/phase3/uci224_delayed_replay_20261116_v1
& C:/Users/hp/anaconda3/python.exe scripts/evaluate_uci224_delayed_replay.py --run results/phase3/uci224_delayed_replay_20261116_v1
```
