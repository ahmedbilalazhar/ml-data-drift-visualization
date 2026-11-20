# Matched-budget calibration-family sensitivity on ELEC2

**Status:** exploratory, outcome-informed extension. This tests a predeclared calibration-family alternative on the existing fixed ELEC2 stream; it is not an independent confirmation.

## Question

Does a cumulative Platt-style logistic correctness calibrator behave differently from cumulative isotonic recalibration when both receive precisely the same released labels and frozen predictions?

## Locked comparison

- Source: the completed OpenML 151/ELEC2 run and its exact source/data hash, time ordering, preprocessing, 10,800/3,662/3,662 historical train/calibration/reference split, fixed classifier, 55 monitoring batches and cached predictions.
- Methods: (1) historical frozen isotonic correctness map; (2) cumulative isotonic refit from the original calibration examples and monitoring correctness pairs released strictly before the forecast; (3) cumulative logistic regression on logit predicted-class confidence, fit on exactly those same historical and already-released correctness pairs. Platt's model has `C=1`, `lbfgs`, fixed configuration; no hyperparameter search.
- Delays: 0, 5 and 20 batches. A monitoring cohort `b` is eligible only if `b + delay < forecast_batch`. The current batch's labels, including delay-zero same-clock labels, are appended after its forecast is persisted.
- The classifier, scaler, probability predictions, batch membership and label truth do not change across calibrators. Both updated calibrators use the same eligible row IDs at every forecast. The evaluator checks equality of release budgets, input IDs and predictions.
- Outcomes: per-batch equal-weight estimation MAE and signed bias, correctness Brier, fixed 10-bin ECE, plug-in interval coverage and width. Include every batch, including the final short cohort. Report the raw single-sequence results descriptively; do not treat its 55 serial cohorts as independent replicates.
- No calibration family or delay is selected based on these monitoring outcomes. This is a sensitivity analysis, not a confirmatory test or proof of deployment benefit.

## Interpretation boundary

Platt calibration is a parametric alternative and can have higher bias if the confidence-to-correctness relation is not sigmoid-shaped. Isotonic is more flexible but can overfit sparse or nonstationary history. Either can look better on this one stream by adaptation to this sequence. An observed gain would not demonstrate transfer to other domains; no gain would not establish that no calibration method can work. Plug-in interval coverage still omits calibrator and reference uncertainty and is not certified.

## Observed result

The evaluator passed for 55 batches at each delay, with exact prediction/ID equality and matched released-label counts between updated methods. Frozen isotonic MAE was 0.05857, correctness Brier 0.21597 and ECE$_{10}$ 0.08012 at each delay. Cumulative isotonic MAE was 0.06281, 0.06372 and 0.05913 for delays 0/5/20; cumulative Platt MAE was 0.06527, 0.06664 and 0.06154. Both updates reduced signed bias toward zero, but both had worse MAE and ECE than frozen isotonic at all delays. Plug-in coverage was 23/55 frozen, versus 18/55, 21/55 and 24/55 for cumulative isotonic, and 21/55 at all delays for cumulative Platt. Neither update consistently repaired interval coverage. This is a negative result for these two update rules on this one ELEC2 sequence, not a conclusion about all calibration methods or domains.

The report is `../../reports/phase3/elec2_calibration_family_20261116_v1/ELEC2_CALIBRATION_FAMILY.tex` and PDF. The evaluator receipt is `../../results/phase3/elec2_calibration_family_20261116_v1/validation.json`; structural checks are in `../../tests/phase3/test_elec2_platt_recalibration.py`.

## Reproduction

```powershell
& C:/Users/hp/anaconda3/python.exe scripts/run_elec2_platt_recalibration.py --run results/phase3/elec2_temporal_20261116_v2 --out results/phase3/elec2_platt_recalibration_NEW
& C:/Users/hp/anaconda3/python.exe scripts/evaluate_elec2_platt_recalibration.py --run results/phase3/elec2_temporal_20261116_v2 --platt results/phase3/elec2_platt_recalibration_NEW --isotonic results/phase3/elec2_recalibration_20261116_v1 --out results/phase3/elec2_calibration_family_NEW
```
