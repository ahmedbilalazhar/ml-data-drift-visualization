# IDM-Viz strengthening v3 protocol

**Status:** Locked computational extension, 16 November 2026. The submitted Phase 2 DOCX remains the authoritative basis; this protocol narrows claims and adds tests rather than replacing it. No instructor feedback was received.

## Objective and estimands

Evaluate whether a frozen, label-safe IDM-Viz pipeline can detect programmed input change, rank known changed marginal features, estimate current batch accuracy, express plug-in concern about a defined historical-reference drop, and verify cached predictions after scheduled label releases. These are separate estimands. A feature alarm is not a harm alarm, an h=0 estimate is not a future forecast, and label availability is not an intervention on model quality.

## Locked experiment matrix

- Three synthetic laws: linear probabilistic, axis-aligned nonlinear XOR, and five-region conditional behavior.
- Eight mechanisms per law: stationary control; abrupt nuisance shift; gradual predictive-coordinate shift; repeated small nuisance shifts; abrupt conditional reversal; recurrent conditional reversal; dependence change with unchanged marginal scales; and large covariate displacement.
- Each task/mechanism/seed stream has 96 consecutive batches of 500 rows. The first 24 batches share the same generated features and label uniforms across mechanisms for a given task and seed. Mechanism-specific row IDs remain unique.
- Historical data are independently generated in disjoint partitions: 4,000 training, 2,000 correctness-calibration and 2,000 fixed-reference rows. The primary run uses eight features; the runner also accepts the locked targeted dimensions 32 and 128.
- The exact held-out seed bank is listed in `configs/phase3_strengthening_v3.json` (30 evaluation seeds). Five development and ten validation seeds have separate, nonoverlapping banks. No evaluation seed is used to tune the protocol.

## Generator truth and terminology

Mechanisms are deterministic transforms of common Gaussian innovations and common Bernoulli label uniforms. Feature changes are applied before the conditional label law is computed. Conditional reversal complements both the realized binary outcomes and the evaluator-only conditional probability on active rows. The five-region task stores generating-region IDs in evaluator artifacts only.

The intervention named `M7_out_of_support` in the machine lock adds 2.5 units to two covariates after batch 24. Since a Gaussian source has unbounded mathematical support, this is **not** a literal zero-probability support violation. Interpret it as a finite-sample low-density covariate-extrapolation stress only. This name is retained in the locked config for stable IDs, while the scientific claim uses the qualified definition.

For the dependence mechanism, the first two coordinates retain unit marginal variance while their correlation changes from 0.3 to 0.9. A maximum pairwise-correlation discrepancy is calibrated using 100 pseudo-cohorts sampled from the historical calibration inputs against the fixed reference inputs. The detector is descriptive and project-specific; it is not a faithful reproduction of a named detector.

## Frozen model and online boundary

One `LogisticRegression(C=1, max_iter=2000)` is fit on training data and never refit in this matrix. Scaling, three-component PCA (capped to valid rank), three-cluster MiniBatchKMeans and isotonic correctness calibration use only their permitted historical partitions. The live scoring API accepts only current features and stable IDs; it has no label argument. At each clock, feature evidence and predictions are persisted before same-clock label releases. The broker verifies exact ordered IDs and drains through the last due label without creating new input.

The current accuracy point estimate is the mean isotonic probability of correctness. Comparators are raw Average Confidence, an ATC-style threshold learned on calibration data, and a constant historical-accuracy estimate. The exact binary Poisson-binomial interval is a plug-in calculation conditional on the supplied correctness probabilities. The concern score applies the declared 0.05 accuracy-drop criterion relative to the frozen historical reference; neither is a calibrated population guarantee.

## Evaluation and statistics

- Input alarms: Holm-adjusted original-variable and frozen-PCA KS tests, plus the separate historical-calibrated pairwise-correlation detector.
- Released-error comparison: after each scheduled cohort release, an exact small-window scan tests eligible error-rate split points using Hoeffding bounds, trims the older window after an alarm, and logs the release clock. This is explicitly an ADWIN-style comparator without exponential bucket compression; it is not an official/River ADWIN reproduction. Delay-100 alarms after clock 96 are post-stream verification, not online detections.
- Localization: known changed-feature precision@1 and recall@3 only for mechanisms with changed marginals. For unchanged marginals, these metrics are undefined. No causal responsibility is inferred.
- Estimation: seed-level MAE, signed bias, Brier score of correctness predictions, plug-in interval coverage and width, and concern versus verified harm.
- Feedback: delays 0, 5, 20 and 100 batches. Prediction arrays are computed once, then the same cached predictions are verified at each delay. Delay 100 has no releases during the 96 input batches and is evaluated only after queue drainage.
- Summaries use the complete seed block as the sampling unit, with 2,000 percentile-bootstrap replicates. Paired estimator contrasts use seed-level randomization tests and Holm correction. Batches, estimators, or repeated delay records are not independent replicates.
- Runtime measurements are implementation diagnostics only, not scientific outcomes.

## Reproducibility and claim boundary

Run state, expected cells, model/config, Phase 2 checksum and code hashes are locked before any cell; code, configuration and the submitted Phase 2 document are copied into a source snapshot. Each cell is committed atomically with a checksum. Resume is refused if the source, configuration or cell list differs. The analysis refuses incomplete runs or checksum mismatches.

This computational extension can support results for the declared synthetic laws and model only. It does not resolve arbitrary conditional-shift identifiability, broad natural-domain transfer, causal attribution, general future-harm forecasting, production safety, or decision benefit. The coordinated view is reported as an implementation prototype; no claim of improved analyst performance is made.
