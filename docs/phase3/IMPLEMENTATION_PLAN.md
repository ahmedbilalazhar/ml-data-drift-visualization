# Phase 3 implementation plan for IDM Viz

Prepared 15 November 2026 for Ahmed Bilal and Umar Behram. Working submission deadline: **5 pm today, Asia/Karachi**. No instructor feedback on Phase 2 was received, as confirmed by the team. This is an implementation plan, not a completed experimental report.

## Recommended decision

Build a small, complete **IDM-Viz monitoring prototype** around your friend's submitted Phase 2: input alarms, feature evidence, calibrated accuracy estimates, delayed verification and one coordinated evidence view. Make the computational study the assessed contribution. Record the reduced scope in the Phase 3 report; do not claim the entire original proposal was implemented.

Use one frozen logistic classifier, seven controlled tabular conditions, 500-row batches and **30 final seeds**. Keep a predeclared 10-seed deadline pilot as the fallback if implementation or validation slips. Three development timing runs suggest that the numerical kernels are inexpensive on this laptop. Spend the time saved on correct label timing, genuine baselines, ablations, clear figures and reproducibility.

Do not implement NOTEARS, a neural autoencoder, many adaptation policies, selective labels or a 20-person analyst study before this deadline. Replace the autoencoder with a frozen PCA anchor and describe that substitution accurately. A working visualization can be demonstrated today; an improvement in human decisions cannot be claimed without the planned human study.

**Start with the label-safe runner and generator tests. Do not start with dashboard styling or downloading more datasets.**

## Sources and constraints

- Submitted methodological basis: [the friend's Phase 2 document](../23i-2604_23i-2581DAV-phase2.docx), SHA256 `786d4f23d54d0d09edd5f94740198f43ca3b406eb38626ae754fe04fe26c5ef4`.
- Assessment requirements: [Phase 03 Implementation](../Phase%2003%20-%20Implementation.pdf), SHA256 `9a94654bdb772b03d5228993f2cc7eaf1b8552f41c462ea84a4649bb0a828115`. Both pages were inspected. It requires implementation, documented data/setup, baselines, figures, comparison with research, ablations and discussion. It also requires documenting methodological changes.
- The earlier Phase 2 LaTeX proposal is supporting design material, not a replacement for what was actually submitted. Borrow its separation of monitoring from evaluator information, cohort identifiers, explicit clocks, paired seeds and qualified claims. Do not import its very large oracle simulation budget.
- Laptop inspected: Intel i7-10610U, four physical/eight logical cores, approximately 32 GB RAM. The existing Anaconda environment has NumPy 2.1.3, SciPy 1.17.0, scikit-learn 1.8.0, pandas 2.3.3 and matplotlib 3.10.8. No GPU or new package installation is required for the recommended core.

## Preserve the submitted proposal while correcting its weaknesses

| Submitted element | Phase 3 treatment | Reason and claim boundary |
|---|---|---|
| Five connected IDM-Viz layers | Keep detection, descriptive localization, impact, visualization and evidence-based status | Retains the submitted research identity |
| Frozen primary model and Average Confidence | Keep; fit confidence calibration on separate historical data | Calibration under source conditions does not guarantee calibration after shift |
| Autoencoder latent anchor | Substitute three-component PCA fitted once on training inputs | CPU-friendly latent comparator; not a reproduction of CADD or DriftLens |
| Causal feature graphs | Defer; rank marginal KS distances | Changed-feature evidence is not a causal explanation |
| Cluster view | Keep three frozen MiniBatchKMeans clusters for descriptive occupancy and confidence summaries | Without generating-region truth, do not claim cluster-localization accuracy or causal responsibility |
| Future harm at horizon h | Set h = 0 for the current named batch and verify it when labels arrive | Current-batch estimation before feedback is not forecasting future batches |
| Harm probability and its upper bound | Define a plug-in Poisson-binomial concern score explicitly | It is not a calibrated probability certificate under arbitrary shifts |
| Online recalibration and retraining | Keep model and calibrator frozen in the core | Preserves matched comparisons and exposes calibration failure instead of concealing it |
| 20-person within-subject study | Defer; demonstrate the interface and validate evidence consistency | No fabricated participants, usability scores or analyst-benefit conclusion |
| Many datasets and statistical tests | Start with one transparent generator and seven mechanisms; use paired seed summaries | Avoid a large, weakly controlled experiment grid |

The scope amendment should state: “No Phase 2 class feedback was provided. We retained the integrated monitoring objective and narrowed implementation to a reproducible frozen-model study. PCA replaces the autoencoder, marginal screening replaces causal discovery, and human/adaptation evaluation is deferred.” These are project choices, not instructor-approved exemptions. Deferred experiments mean the corresponding original research questions remain unanswered.

## Experimental specification to freeze before final runs

### Data roles and model

For each seed generate **disjoint** partitions: 4,000 training rows, 2,000 correctness-calibration rows, 2,000 historical risk-reference rows and 12,000 monitoring rows per condition. Use eight numeric features. Fit imputation if needed, StandardScaler, the classifier, PCA and clustering on training data only. Fit isotonic correctness calibration only on its designated partition. The final 1,000 historical reference inputs supply the fixed input comparison; the full 2,000 labelled reference rows supply historical accuracy.

Baseline inputs: independent standard-normal Z coordinates, X0 = Z0 and X1 = 0.3 Z0 + sqrt(1 - 0.3²) Z1, with the remaining coordinates unchanged. Generate labels independently from probability sigmoid(1.2 X0 - 0.9 X1 + 0.6 X2). The classifier is LogisticRegression(C=1, max_iter=500), fitted once per seed. Record convergence failures; never quietly keep a failed fit or drop its paired conditions.

Generate the modified inputs **before** sampling labels. Use shared Gaussian innovations and label uniforms across paired conditions where appropriate, with distinct stage namespaces. These shared draws create pairing, not extra independent replications.

| Condition | Intervention after monitoring row 4,000 | What it tests |
|---|---|---|
| C0 Stationary | None | Input-alert burden, estimation error and interval coverage |
| C1 Nuisance mean shift | Add 1.5 to X7; unchanged label rule | Detectable input movement with no change in the generating label rule; fitted-model harm must still be measured |
| C2 Gradual predictive covariate shift | Increase the X0 mean from 0 to 1.5 over rows 4,001–6,000, then retain it; unchanged label rule | Gradual input detection; a relevant input shift need not worsen accuracy |
| C3 Abrupt conditional change | Replace p(Y=1 given X) by 1-p | Harm that input-only monitoring cannot identify |
| C4 Gradual conditional change | Mix p and 1-p with rowwise weights rising from zero to one over rows 4,001–6,000 | Gradual deterioration without input change |
| C5 Recurring conditional change | Use 1-p for rows 4,001–8,000, then return to p | Deterioration and recovery; no native input-change event |
| C6 Dependence change | Replace correlation 0.3 by 0.9 using X1 = 0.9 Z0 + sqrt(1-0.9²) Z1 | Joint input change with unchanged individual Gaussian marginals |

The C6 direction is chosen from the declared opposite-sign label coefficients, before final outcomes are examined: greater correlation can concentrate the predictive score nearer its boundary. It is not selected because a fitted method performed well. Neither C1 nor C2 is assigned a guaranteed harmless label; all realised error changes are measured.

Use 24 nonoverlapping batches of 500 rows per monitoring stream. Seven conditions × 30 seeds give **210 streams, 2.52 million monitoring observations and 5,040 batch records per method before delay expansion**. This is the entire main grid; baselines and ablations reuse its observations and prediction logs.

Development seeds: 90001–90005. Final seeds: 100–129. The fallback pilot is seeds 100–109 and must be labelled a 10-seed pilot. Do not replace unsuccessful seeds or choose the seed count from favourable results. If the 30-seed run cannot be completed and repaired, report that status and the predetermined pilot separately.

### Evidence and label timing

At batch completion, predict from features, persist predictions and compute input/proxy evidence. Only then release labels scheduled for that timestamp. Use delays **0, 5 and 20 batches**; the original 100-batch delay is an optional clock-only extension. After the last input, drain the release queue without inventing more input observations.

IDs must join predictions and labels one to one. Keep the evaluator's unreleased labels inaccessible to monitoring code. Reuse predictions, KS scores, PCA coordinates, cluster assignments and Poisson-binomial distributions across delay schedules. The delays change evidence availability, not a frozen batch's eventual accuracy or proxy value. They do not multiply the number of independent streams.

## Methods and mathematical targets

### Input and descriptive evidence

Use two-sided KS tests on eight original variables and three frozen PCA coordinates. Apply Holm correction at family level 0.05 within each batch; an input alarm means at least one rejection. With these sample sizes SciPy's `method="auto"` attempts its exact calculation. Record warnings/fallbacks. The continuous independent synthetic null supports the per-batch interpretation; there is no lifetime false-alarm guarantee. Report achieved stationary alert burden rather than assuming identical realised burden between variants.

Rank original-variable KS distances for localization. A feature is flagged only if its original-variable test is rejected within the declared family; a top-three rank alone does not flag it. Use known changed marginals only in C1 and C2. C6 has an empty changed-marginal set: do not mark X0/X1 as changed marginal features merely because their correlation changed. Report joint/latent detection separately; no feature-pair or causal localization claim follows. Show cluster occupancy and mean estimated correctness descriptively, including cluster sample counts; suppress unstable per-cluster rates below 20 observations without deleting the batch.

### Accuracy estimates and intervals

Keep class predictions fixed: yhat = 1{p_model >= 0.5}, raw confidence s = max(p_model, 1-p_model). Fit an isotonic map g(s) to the correctness indicator 1{yhat=y} on the separate calibration partition. Set q = g(s), with `out_of_bounds="clip"`. Do not flip predicted classes when g changes; otherwise the calibration ablation changes two components.

For batch s, estimated accuracy is Ahat_s = mean(q_i). For the plug-in model K_s = sum Bernoulli(q_i), compute its Poisson-binomial CDF by the dynamic-programming recurrence:

    P_0(0) = 1
    P_j(k) = (1-q_j) P_(j-1)(k) + q_j P_(j-1)(k-1)

The 2.5th and 97.5th quantiles divided by batch size give a **model-based prediction interval for realised batch accuracy**. They are not a distribution-free confidence interval for population accuracy. Independence and valid target correctness probabilities are required; estimated calibration uncertainty is omitted by this plug-in construction. Conditional change is an intentional failure case. Measure empirical coverage and width, rather than assuming 95% coverage.

Use historical realised accuracy A_ref from the separate 2,000-row reference. Primary verification target:

    A_s = mean(1{yhat_i = y_i})
    H_s = 1{A_ref - A_s > 0.05}

This is realised batch deterioration relative to one observed historical reference, **not independently certified expected-population harm**. All methods use the same target and cohorts. This choice preserves the submitted batch-loss formulation while avoiding millions of oracle evaluations.

Define the pre-label proxy concern score as P_plugin(K_s / n < A_ref - 0.05). Compute it from the same CDF; a score at least 0.95 produces a “high proxy concern” flag. For strict inequality use k_max = ceil(n(A_ref-0.05))-1 and handle values outside 0…n. Compute threshold comparisons from integer correct counts and an exact rational margin to avoid floating-point boundary errors. The score is not 95% frequentist confidence that harm is present.

Do not require an input alarm before computing or displaying the impact estimate. Equally, a low proxy score or absent input alarm does not prove that the model is unharmed. Once labels are released, show the observed batch result and its evidence age. No automatic retraining or proven-optimal action is claimed.

## Baselines and ablations

Run these on the same frozen model, streams and cohorts:

| Comparator | Outcome evaluated | Important qualification |
|---|---|---|
| Raw Average Confidence | Accuracy-estimation MAE | Also the calibration-removal ablation |
| Calibrated Average Confidence alone | Accuracy-estimation MAE, coverage and harm concern | Must equal IDM-Viz's impact channel exactly; integration does not invent a better accuracy estimator |
| ATC-style estimator | Accuracy-estimation MAE | Fit a raw-confidence threshold on the calibration partition to match its empirical accuracy; disclose this adaptation |
| Constant historical accuracy | Accuracy-estimation MAE | An essential no-change baseline |
| Original-feature KS only | Native input detection; separately its use as a harm proxy | A distributional detector was not designed to certify harm |
| Released-label verification | Evidence availability and observed accuracy | An identity verifier of H_s after release, not a predictive method with magically perfect early accuracy |

For ATC-style estimation select a threshold from unique calibration scores plus endpoints minimizing abs(mean(s >= threshold)-calibration_accuracy), with ties to the larger threshold. Freeze it before monitoring. Do not label an approximate DoC-Feat, DriftLens or ADWIN-inspired implementation a faithful reproduction.

Three required component experiments:

1. **Remove calibration:** replace g(s) by raw confidence while retaining predictions, input scores and historical target. Compare MAE, realised coverage, interval width and concern errors.
2. **Remove latent evidence:** remove PCA tests and recompute the original-feature family correction at the same 0.05 level. Compare native detection, stationary burden and runtime; explain that changing the number of tested dimensions also changes power allocation. Impact estimates must remain identical.
3. **Remove interval handling:** compare the point rule A_ref-Ahat_s > 0.05 with the declared 0.95 plug-in concern rule. Report fewer/more flags together with correct and incorrect flags; reduced warning count alone is not improvement.

Compute these from shared caches. Persist prediction arrays in compressed NumPy files and batch summaries in CSV; regenerate synthetic inputs from the recorded seeds. Do not duplicate full feature/prediction logs for every delay or render a PNG for every batch. Removing a display panel is not evidence of an effect on analyst decisions. Removing a descriptive cluster view is not a meaningful accuracy ablation if it never feeds the estimator.

## Evaluation and required figures

Keep all complete batches in the main denominator. Provide per-condition and per-seed results before any equally weighted summary.

- Input task: pre-onset/stationary alerts per 1,000 batches; first-alarm delay and misses for C1/C2/C6 only. For an onset, match the first alarm within four batches; report duplicate alerts as alert burden. Conditional-only episodes are excluded from input-event recall.
- Feature task: precision at 1 and recall at 3 on C1/C2 changed regimes, with changed-set size. Empty marginal truth has undefined localization recall; show false feature flags separately.
- Impact task: MAE against realised A_s; mean interval width; empirical coverage; harm-flag precision, recall, false flags and the fraction of harmful batches left unflagged. Report the flagged fraction and unresolved status separately. A missing flag is not an explicit below-margin verdict. Only a rule that actually declares below-margin status can make a false-reassurance claim; do not reward deferral as accuracy. Give numerator/denominator counts. Undefined precision, recall or AUROC must remain undefined when the required class/flag is absent.
- Model description: reference and batch accuracy, macro-F1 and class counts. F1 is descriptive; it does not replace the error margin defining H_s.
- Cost: separate fitting, input scoring, impact computation, rendering and total elapsed time. Record peak process memory if measured. Do not substitute Python allocation tracking for total RAM.
- Uncertainty: retain complete seed vectors; use paired seed differences and a 2,000-replicate seed bootstrap for approximate intervals. Do not count batches, delay copies or ablations as independent replications. Do not run Friedman/Nemenyi merely by treating seeds as different datasets. Keep the analysis descriptive rather than manufacture a significance ranking across many outputs.

Minimum figure set:

1. A coordinated example: input alarm/score, feature heatmap, estimated versus verified accuracy, prediction interval and a labelled release-time track. Plot the minimum Holm-adjusted p-value against its 0.05 cutoff; use KS distances in the separate feature heatmap, rather than mixing a p-value threshold with distance units. An as-of-time view must hide unreleased losses. An offline “eventual truth” line must be explicitly labelled evaluator-only.
2. Estimation-MAE comparison by condition with all seed points or seed uncertainty.
3. Empirical interval coverage **and width**, split by condition; a 0.95 reference line is a target, not a certified achievement.
4. Ablation table or plot containing error, coverage/flag rates and runtime.

An offline HTML with a seed/scenario selector and shared batch selection is a useful prototype if straightforward. Static coordinated figures are the fallback for the 5 pm package; call them figures, not a completed interactive dashboard. Do not spend the report-writing reserve on a new UI framework.

## Existing code audit and implementation order

The repository contains useful scaffolding, but its old results are not ready-made Phase 3 evidence.

- `src/drift_monitoring/orchestration/runner.py` computes `macro_f1` directly from current `seg["label"]` while marking records delayed. Rebuild this boundary so unreleased truth is evaluator-only.
- `src/drift_monitoring/windows/replay.py` returns a segment containing labels and does not drain all delayed labels after the final input. Do not rely on its docstring alone for the evidence contract.
- `simulation/generator.py` S6 rotates correlated coordinates while claiming invariant marginals. With rho=0.7, that rotation gives variances approximately 0.3183 and 1.6817, not both one. Use the correlation construction specified above.
- `scripts/e3_proxy.py` derives decisions from retrospective loss summaries; it is not proof of prospective, label-safe operation.
- `scripts/build_dashboard.py` plots those old F1 values at their original batch positions. Adapt its visual structure only after fixing which evidence was available when. Its existing figure is not evidence of improved human decisions.
- Cached natural-data admission flags, old split ratios and old output CSVs must not be silently reused under the revised protocol.

Suggested new implementation boundary: `src/drift_monitoring/phase3/core.py` for generation/fitting/scoring, `evaluation.py` for evaluator-only aggregation, `scripts/run_phase3.py` for the runner and `scripts/report_phase3.py` for tables/figures. These are **planned files, not currently available commands**. Isolating the new protocol is faster and safer than repairing every historical experiment script.

Build in this order:

1. One stationary stream, deterministic splits, fitted artifacts and ID-joined prediction/label logs.
2. Replay and assertions for delays 0/5/20, including final queue drainage.
3. KS/PCA, frozen correctness calibration and tested Poisson-binomial calculations.
4. Seven conditions, baseline outputs and the three cached ablations.
5. Seed aggregation, four figures and evidence-view output.
6. A single command that regenerates the declared outputs and records source/config hashes and versions.

Essential checks: future-label tampering does not alter earlier monitor records; removing latent tests does not alter impact estimates; calibrated-AC-alone equals the full impact channel; delay schedules preserve predictions/proxies; C6 has analytically unit marginal variances; stationary and returned regimes follow their declared laws; Poisson-binomial all-zero/all-one/constant-probability cases match known distributions; every displayed record joins to one real cohort; all held-out outputs stay outside fitting/calibration.

## Schedule until 5 pm today

The following is a working allocation, not a guarantee. Move to the next stage as soon as its checks pass.

| Karachi time | Task | Required exit artifact |
|---|---|---|
| 2:10–2:20 pm | Freeze scope, config, seeds and amendment; confirm environment | Locked manifest and short methodology-change table |
| 2:20–3:10 pm | Implement the label-safe core and baselines | One valid smoke run, prediction/label logs and passing contract checks |
| 3:10–3:30 pm | Finish conditions/ablations; run development feasibility check | Projected runtime and no unresolved integrity failure |
| 3:30–3:50 pm | Execute the locked final run and aggregate | Complete CSVs, runtime records and reproducible seed summary |
| 3:50–4:25 pm | Generate figures and write the empirical report | Methods, actual results, ablations and critical discussion |
| 4:25–4:40 pm | Reproduce from a clean output directory; inspect report pages | Reproduction check and readable final report |
| 4:40–4:50 pm | Package and verify required files | Submission archive and separately accessible report |
| 4:50–5:00 pm | Submit and check upload | Upload confirmation; preserve the submitted snapshot |

**Hard decisions:** If core integrity checks are still failing at 3:30, stop adding features. Preserve the smallest honest verified study rather than submit leaked results. If the validated end-to-end 30-seed projection exceeds the allocated compute window, use the predetermined 10-seed pilot and label its limitation. At 4:25, freeze analysis changes and prioritize packaging. An incomplete study must be described as incomplete; no fallback can guarantee full rubric compliance.

If two people are available, one owns the runner/tests while the other prepares the report skeleton, citation/comparison table, figures and packaging from the common CSV schema. If working alone, keep the same priority order and use static coordinated figures first.

## What the 5 pm package should contain

| Rubric item | Concrete evidence to submit |
|---|---|
| Implement proposed methodology | Working reduced IDM-Viz core, updated architecture and an explicit deviation table |
| Data and preprocessing | Generator equations, split/seed manifest, training-only transformations and feature/label contract |
| Baselines | Matched AC, ATC-style, no-change and KS results with counts and relevant metrics |
| Graphical results | Four measured-result figures with interpretable axes and release-time semantics |
| Compare with existing research | Method/evidence/settings comparison with verified central papers; no incomparable published-score leaderboard |
| Ablations | Calibration, latent-evidence and interval-handling experiments on the same observations |
| Discussion | Which questions are partly answered, negative/failure cases, calibration limitations, no human/adaptation claim and next steps |

Package the report PDF, its editable source, source code, exact environment versions, locked config, tests, result CSVs, figures, optional offline HTML and a README with working reproduction commands. Keep the friend’s submitted Phase 2 unchanged. Do not present the current timing probe or historical E0 output as the final experimental study.

Suggested report structure: objective and scope changes; dataset/setup; method and baselines; detection/localization; accuracy and uncertainty; ablations; research comparison; limitations and reproducibility. Use only measured values in result tables. If a feature is deferred, say so once clearly and connect that omission to the unanswered question.

## Improvements during the following day

Keep the 5 pm submission snapshot immutable. Improve a new version in this order:

1. Repair any reproducibility failures and complete the predetermined 30-seed study if only the pilot was submitted. Report the extension prospectively; do not replace a pilot because its findings are unfavourable.
2. Add a frozen shallow tree as model sensitivity, with the same streams and separate correctness calibration. Then add one held-out generator/task family or a prospectively admitted natural dataset. Independent seeds from the same generator are not evidence of transfer to another real domain.
3. Finish genuine offline dashboard interaction and check all panels use the same cohort and as-of-time evidence. This establishes implementation consistency, not human benefit.
4. Add **one** separate released-label adaptation experiment only if required and time remains, comparing frozen versus periodic updates using identical released-label budgets. Keep it separate from the frozen-model results.
5. Extend delay 100 through the release log if useful; do not recompute identical features and predictions. Add selective labels only with a new, justified target-population analysis.

Do not promise to finish a valid 20-participant study, broad causal discovery benchmark or deployment validation within one day. Original RQ3 remains deferred. RQ2 covers descriptive marginal localization only; causal/cluster accuracy remains untested. RQ4 covers controlled regime robustness until a genuinely held-out source or task is added.

## Measured feasibility and research attribution

The development-only timing probe processed seven input scenarios × 24 batches for each of three seeds using the main classifier, correctness calibration, PCA, MiniBatchKMeans, original/latent KS tests and exact dynamic-programming Poisson-binomial CDFs. With KS `method="auto"`, its latest aligned kernels took **4.94–5.80 seconds per seed**, single-threaded. Thirty times the slowest measurement is approximately **2.9 minutes of these kernels**. Allow **15–25 minutes for the completed final run, aggregation and output overhead**, subject to a new end-to-end development check. Implementation, tests, result interpretation, figure generation and packaging are excluded from the probe. Its conditional-change rows exercise the same input/proxy calculations; it does not establish correct final generation, label timing, monitoring accuracy or empirical interval coverage.

The retained probe uses the planned +0.9 dependence endpoint and gradual covariate mean schedule. Recheck end-to-end timing after the actual label-safe runner, evaluation and output writer are complete. Timing seeds remain outside final evidence. The probe and its JSON record are in `tmp/phase3_planning_20261115/`.

Use [Kivimäki et al., JAIR 2025, DOI 10.1613/jair.1.16709](https://doi.org/10.1613/jair.1.16709) and the [reviewed author full text](https://arxiv.org/html/2407.08649v2) as the central AC/Poisson-binomial comparison. Attribute AC and disclose the fixed-prediction correctness-calibration and ATC-style adaptations. Match comparisons within this experiment; differences in data generation, models and protocol prevent importing its published rankings as ours.

The implementation choices are also documented in the official [SciPy KS reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html) and [scikit-learn calibration guidance](https://scikit-learn.org/stable/modules/calibration.html). The installed versions, rather than the latest online release, govern execution. Verify any additional Phase 2 reference before copying its numerical claims into the Phase 3 discussion. Its reference [15] is a systematic review: use it for taxonomy and context, not as an experimentally reproduced baseline. Do not repeat the submitted document's absolute assertion that no prior study integrates these ideas; the contribution here is a scoped implementation and controlled evaluation.
