# Future-harm forecast on an unseen synthetic intervention family

**Status:** exploratory held-out family-transfer extension. Protocol fixed before generating/scoring the new condition. This is not a natural-stream evaluation and does not update the already evaluated forecast model.

## Transfer question

Does the forecast model trained on the seven existing C0-C6 synthetic laws retain useful five-batch future-harm prediction when evaluated on a conditional-randomization mechanism not used to fit that forecast model?

## New evaluation law

- Preserve each historical model/scaler/calibrator/reference fit exactly as in the core protocol. Use 30 held-out evaluation seeds 950100-950129 and the existing frozen base logistic classifier.
- Generate the baseline feature matrix with the same eight-dimensional baseline Gaussian law. Keep every X marginal and covariance unchanged over all 12,000 monitor rows.
- At zero-based row 4,000 (first changed row is one-based 4,001), alter only the label mechanism: `p_new(y=1|x) = (1-eta)*p_base(y=1|x) + eta*0.5`, eta=0.70. Use common uniform innovations and the exact predeclared label draw. This flattens the conditional class relation toward random labeling without changing X or the mean class prior under the symmetric base law.
- This is a new generator intervention, but still synthetic and related to the original logistic law. It does not represent arbitrary concept drift or an independent real domain.

## Forecast protocol

- Train the three pooled forecasters only on the saved 30 development seeds and seven original conditions from `future_harm_forecast_20261116_v1`; same feature sets, `StandardScaler`, `LogisticRegression(C=1, max_iter=1000)`, target and no-threshold-tuning rule. Refit those models deterministically from saved training rows; do not tune them on C7.
- Evaluate only on seeds 950100-950129 for this new C7 law. The target at forecast batch t is whether mean realized accuracy across batches t+1..t+5 is more than five percentage points below the same stream's historical reference. Forecast origins are t=1..19.
- Delays are 0/5/20. A monitoring label from batch b enters predictors only if `b+delay < t`; same-clock labels are appended after the forecast. For delay 20, no monitoring labels are available at any registered origin because t<=19.
- Compare time-only, unlabeled-monitor and released-label predictors with development prevalence, last released harm and current concern score. Keep the same metrics and whole-seed paired bootstrap. Do not compare repeated rows as independent samples.
- Persist label-free online predictors/probabilities separately from evaluator future targets. Audit source labels, target windows, exact IDs, seed disjointness, release clocks, and unchanged feature marginals before evaluation.

## Interpretation boundary

This tests transfer to one withheld conditional-randomization intervention, not to future unseen generator families generally, not to an external dataset and not to an operational population. The development/evaluation seeds are disjoint, but development includes seven known conditions and the intervention is manually designed. Report failures and undefined metrics. No causal attribution, human benefit, or risk certificate is claimed.

## First aligned-onset pilot outcome

The first locked C7 run used the shared training onset (one-based row 4001) and produced target prevalence 0.788 and time-only ROC AUC 0.9975. This demonstrates that the fixed onset timing was a strong predictor of the target and makes that pilot inadequate as evidence that monitoring features transfer. Its complete metrics are in `../../results/phase3/future_harm_unseen_family_20261116_v1/validation.json`. The balanced-onset follow-up and rationale are documented separately in `FUTURE_HARM_ONSET_RANDOMIZED_FOLLOWUP_20261116.md`; both runs are retained.

## Reproduction

```powershell
& C:/Users/hp/anaconda3/python.exe scripts/run_future_harm_unseen_family.py --out results/phase3/future_harm_unseen_family_NEW
& C:/Users/hp/anaconda3/python.exe scripts/evaluate_future_harm_unseen_family.py --run results/phase3/future_harm_unseen_family_NEW
& C:/Users/hp/anaconda3/python.exe scripts/report_future_harm_unseen_family.py --run results/phase3/future_harm_unseen_family_NEW --out reports/phase3/future_harm_unseen_family_NEW
```
