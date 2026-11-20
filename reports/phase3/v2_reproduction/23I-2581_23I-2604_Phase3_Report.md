# Phase 3: Research Implementation and Experimental Evaluation

**Ahmed Bilal (23I-2581) and Umar Behram (23I-2604)**  
Department of AI and Data Science, FAST-NUCES Islamabad  
Data Analysis and Visualization

**Protocol status:** Outcome-informed extension; final evidence generated from the locked extension protocol.

## Summary

Version 2 adds a fixed shallow tree comparison on the original controlled task and a new XOR synthetic task. The final run has 360 feature streams and 17,280 model-batch records. It evaluates both learners with their own correctness calibrator, ATC-style threshold and historical reference. The XOR task is synthetic and is not external-domain validation.

On stationary XOR, tree minus logistic accuracy was 0.2168 (seed-bootstrap 95% interval 0.1567 to 0.2751). On v1 stationary data the paired accuracy difference was -0.0323 (-0.0343 to -0.0301).

Exact extension check passed for 210 v1-task streams, 1680 cached arrays and 25,210,080 array values. V1 logistic predictions, confidence arrays and original condition records are preserved as the matched comparison; no v1 seed or outcome is replaced.

## What the extension tests

The shallow tree is fixed at Gini criterion, depth 6 and minimum leaf size 25. Logistic regression retains `C=1`, `max_iter=500` and a 0.5 class cutoff. Both models use training-only scaling and independent isotonic calibrators fitted on the same correctness-calibration rows. No hyperparameter search is performed.

The new XOR task uses eight independent standard-normal features and the noisy concept `1{(X0 > 0) XOR (X1 > 0)}` with 10% baseline label noise. Its locked interventions are stationary, irrelevant mean shift, relevant mean shift, concept complement and increased label noise. The last two preserve inputs and alter only the label law. Synthetic event times are known from construction; no onset labels are fabricated for a natural dataset.

The local ELEC2 candidate was rejected: the raw ARFF’s `date` is non-monotone, 528 `(date, period)` pairs repeat, and 11 date values have 96 rows instead of 48. The date/period fields therefore do not uniquely reconstruct row chronology. The processed parquet also fails to read. OpenML’s local metadata reports its license field as `Public`; this report does not turn that label into a specific permissive license. The source record and original study are listed at the end.

## Overall results

| Task | Model | Calibrated AC MAE, mean [seed-bootstrap 95% interval] | Coverage | Mean interval width |
|---|---|---:|---:|---:|
| Original synthetic task | Logistic | 0.114189 [0.113086, 0.115365] | 0.695238 [0.675397, 0.712103] | 0.073760 [0.073481, 0.074024] |
| Original synthetic task | Shallow tree | 0.105031 [0.103582, 0.106560] | 0.622817 [0.605754, 0.639484] | 0.077718 [0.077465, 0.077968] |
| XOR synthetic task | Logistic | 0.023147 [0.021421, 0.024964] | 0.876667 [0.852771, 0.900833] | 0.087413 [0.087353, 0.087474] |
| XOR synthetic task | Shallow tree | 0.089146 [0.069485, 0.110378] | 0.750278 [0.705826, 0.798618] | 0.069385 [0.063868, 0.074830] |

For context, the earlier study reported calibrated AC MAE 0.114189 versus raw AC MAE 0.113387, and calibrated coverage 0.695238. Its retained baseline tables give the corresponding seed rows and denominators. These cohorts are not pooled with the two new synthetic tasks.

Per-condition summaries, complete seed rows, paired seed contrasts, pooled harm counts, undefined metrics and bootstrap intervals are provided in the CSV files. The tree’s model-specific historical reference changes its realised-harm label; differences in harm counts alone do not rank the models. Matched-cohort accuracy and estimation error are reported separately.

## Failure analysis

In v1 C3-C5, the calibrated confidence channel averaged 0.723 while evaluator-only accuracy averaged 0.488; 1,205 batches met the v1 realised-harm rule and the calibrated concern rule flagged 0. This is measured. Those conditions reuse identical input arrays while changing the conditional label law, so the input detector has no distributional input evidence by construction. The model receives familiar inputs and retains its source confidence even when its predictions become wrong. The frozen source calibrator cannot infer a label-law reversal from confidence alone.

Version 1 calibrated AC MAE (0.114189) exceeded raw AC MAE (0.113387) by 0.000802; calibrated interval coverage was 0.695. The plausible explanation that an isotonic map learned source reliability but did not transfer to changed regimes is consistent with the conditional-only results, but it is not a unique causal explanation. V2 adds pre/onset/post/recovery reliability summaries and reports calibration error and Brier score by model and regime.

For C6, v1 matched the dependence onset in 8/30 seeds with full KS/PCA and 4/30 with original-feature KS. Individual frozen PCA-coordinate adjusted-p detections in the onset window were: PC0 6/30, PC1 1/30, PC2 0/30. PCA therefore supplied some additional task-specific marginal evidence, but did not solve the missed-event problem; the PCA basis is descriptive and fixed from training.

Version 2 reports interval width alongside empirical coverage and concern TP/FP/FN/TN counts for each task, model, condition and regime. A no-flag outcome is not a safety verdict. The plug-in Poisson-binomial interval and concern score are not calibrated harm probabilities or arbitrary-shift coverage guarantees.

At the end of 24 input batches, delayed verification leaves 0 batches unresolved for delay 0, 5 for delay 5, and 20 for delay 20; after drainage all 24 are verified at ages 0, 5 and 20 batches respectively. Delay changes evidence availability and age, not the frozen estimator’s accuracy.

## Figures

**Paired model sensitivity:** estimation error by task, condition and learner.

![Paired model sensitivity](figures/model_sensitivity_mae.png)

**Interval behavior:** empirical coverage and width; the nominal line is a target.

![Coverage and interval width](figures/interval_coverage_width.png)

**Concern-rule outcomes:** harmful batches left unflagged are shown directly.

![Harm flags and misses](figures/harm_concern_outcomes.png)

**Reliability:** calibrated and raw scores across stream regimes.

![Calibration by regime](figures/calibration_by_regime.png)

**V1 failure analysis:** conditional-only shift with evaluator-only truth.

![Conditional-only failure](figures/v1_conditional_failure.png)

**V1 dependence analysis:** adjusted-p detections by frozen PCA coordinate.

![Dependence coordinate detection](figures/v1_c6_pca_components.png)


## Version comparison and research questions

- **Established beyond v1:** model sensitivity on matched original-task observations; an independently seeded synthetic task with an axis-aligned nonlinear boundary; model-specific calibration, harm targets, intervals and paired uncertainty; failure reliability split by task, model and regime.
- **Contradicts a universal model claim:** neither learner dominates every task/regime on accuracy, estimation error, coverage and harm flags. Interpret comparisons only within matched cohorts and state the model-specific reference target.
- **Unchanged:** v1’s logistic method, original condition evidence, thresholds, delays and negative findings remain intact. V2 does not retrofit v1 outcomes or claim a natural-domain replication.
- **RQ1:** current-batch realised deterioration is evaluated in controlled synthetic streams; future forecasting and certified population harm remain untested.
- **RQ2:** feature evidence remains descriptive marginal ranking; causal attribution and cluster localization accuracy remain unanswered.
- **RQ3:** analyst benefit remains unanswered; there is no human-subject evaluation.
- **RQ4:** robustness evidence now spans two synthetic task families, not external or natural-domain validation.

## Limitations and provenance

The evidence uses one linear probabilistic learner and one fixed shallow tree on two synthetic families. It does not include external-domain validation, neural autoencoders, causal discovery, online adaptation, selective-label analysis or a human study. Calibration is frozen, confidence-based, and dependent on the source law. Interval width and empirical coverage are descriptive. Bootstrap intervals resample complete seed vectors; batches, delay copies, models and ablations are not treated as independent datasets. Version 2 hypotheses were informed by observed v1 outcomes, so the extension is not described as an untouched preregistration.

The submitted Phase 2 DOCX remains the methodological basis; the older TeX is only supporting material, and no instructor feedback was received. The original two-page implementation brief was consulted and preserved. The existing offline replay is a prototype, contains retrospective release information and is not a security boundary or human-evaluation result.

**Sources consulted for the rejected ELEC2 candidate:** [OpenML dataset 151](https://www.openml.org/d/151); Gama, Medas, Castillo and Rodrigues, *Learning with Drift Detection* (2004), [primary paper](https://sweet.ua.pt/gladys/Papers/GamaMedasCastilloRodriguesSBIA04.pdf). The published task results are not treated as comparable benchmarks.

## Reproduction and artifact semantics

**Independent reproduction:** a clean run matched 741 CSVs (6,768,744 cells) and 1,140 compressed arrays (6,660 arrays; 95,554,560 values) exactly. Timing columns and protocol metadata were excluded from scientific equality checks.
The first comparator attempt differed only in bootstrap summaries of wall-clock `input_seconds`. That engineering timing was removed from uncertainty summaries, and both saved runs were reevaluated; no inputs, seeds, predictions, model settings or scientific thresholds changed.

Use `Reproduce.md` for the package contents and run separation; the project workspace guide `docs/phase3/23I-2581_23I-2604_Phase3_Reproduction.md` gives the exact full-rerun commands. The final run has a protocol lock and exact environment versions. `predictions/` contain persisted online scores; `released_verification.csv` contains scheduled label results; `evaluator_truth/` and `evaluator_batches.csv` are retrospective. Historical source documents and run archives are preserved.
