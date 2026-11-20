# Randomized-onset follow-up for the unseen conditional-randomization family

**Status:** predeclared exploratory follow-up created after the first unseen-family run showed a time-only AUC of 0.998 when its fixed onset coincided with the known training onsets. This follow-up addresses that specific temporal-alignment confound; it does not replace or conceal the first result.

## Protocol

- Keep the C7 conditional-randomization law exactly the same (`eta=0.70`, only `P(y|x)` changes, X generated identically); keep the same fixed IDM-Viz classifier, calibration/reference artifacts, forecasting models, feature sets, target, seed split, delays and metrics.
- Assign onset batches 5, 9, 13 and 17 in a balanced deterministic schedule across held-out evaluation seeds 950100-950129 (seed order cycles through the four values). The onset assignment is used only to generate the evaluation stream and is not passed as a predictor. No test outcomes, condition or seed ID enter the forecast model.
- The development forecasters remain trained on seeds 950000-950029 under the original seven C0-C6 laws. No C7 outcomes are used for fitting, threshold selection, calibration or feature selection.
- Evaluate 30 held-out seeds, 19 forecast origins, delays 0/5/20, the same five-batch future-harm target, three trained predictor sets and the same prevalence/last-released/current-concern comparators. Use paired whole-seed bootstrap.

## Rationale and limits

The first run's near-perfect time-only score indicates that synchronized onset timing alone can make the future target predictable. The balanced-onset follow-up asks whether the monitor/released-label predictors retain information when onset time is not fixed at the training-law transition point. It is still one designed synthetic family with a small set of known onset times; it is not a natural-stream validation or proof of broad regime generalization. Report both runs and all baselines.

## Observed follow-up result

The run passed the seed-disjoint, target-window, feature/outcome-separation and release-clock checks on 30 seeds, four balanced onset positions, 19 origins and three delays (1,710 forecasts per predictor). Target prevalence was 0.693. Overall ROC AUC / Brier were: time-only 0.875 / 0.238; unlabeled monitor 0.864 / 0.188; released-label model 0.830 / 0.194; development-prevalence baseline 0.500 / 0.290. Raw current-concern and last-released-harm scores had Brier 0.660 and 0.333, respectively; they are not calibrated future-risk probabilities.

The paired 30-seed bootstrap for time-only minus unlabeled-monitor Brier was +0.0506 [0.0408, 0.0619], favoring the unlabeled predictor. For unlabeled-monitor minus released-label Brier it was -0.0067 [-0.0229, 0.0086], which includes zero and does not establish an advantage from released labels in this held-out family. The original aligned-onset pilot's time-only AUC was 0.998, so timing remains influential even after this follow-up. This result supports limited transfer within one manually designed synthetic law only.

The evaluator result is `../../results/phase3/future_harm_unseen_family_balanced_20261116_v1/validation.json`; editable TeX and figure are in `../../reports/phase3/future_harm_unseen_family_balanced_20261116_v1/`; invariant tests are in `../../tests/phase3/test_future_harm_unseen_family.py`.

## Reproduction

```powershell
& C:/Users/hp/anaconda3/python.exe scripts/run_future_harm_unseen_family.py --onset-policy balanced_onsets --out results/phase3/future_harm_unseen_family_balanced_NEW
& C:/Users/hp/anaconda3/python.exe scripts/evaluate_future_harm_unseen_family.py --run results/phase3/future_harm_unseen_family_balanced_NEW
```
