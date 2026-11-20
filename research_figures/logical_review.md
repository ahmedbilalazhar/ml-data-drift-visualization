# Methodology audit and corrections

Reviewed 22 October 2026. The existing Phase 2 LaTeX source was revised in place. The four required deliverables, eight additional papers, three-paper comparison, three gaps and twelve prescribed methodology steps are retained. Word files and research implementation code are unchanged. Figure assets describe proposed work only.

## Substantive problems corrected in this audit

| Severity | Weakness | Correction in the submission |
|---|---|---|
| Major | Primary ratios excluded cohorts whose risk truth was near the harm margin. This can select easier cases and alter prevalence. | Promote all-cohort correct/wrong added-declaration frequencies and their identification bounds to the primary estimands. Keep uncertain truth in a fixed denominator. Certified-subset ratios are secondary diagnostics. |
| Major | Oracle uncertainty and seed uncertainty were combined through an approximate bootstrap while resembling a rigorous joint guarantee. | Derive an independent-seed bounded-mean envelope with four one-sided sampling tails totaling 0.04 and simultaneous oracle failure at most 0.01. The two primary population means have at least 95% joint coverage under the independent synthetic assumptions. Approximate bootstrap descriptions remain separate. |
| Major | A sparse or all-zero percentile bootstrap can falsely imply certainty. The previous zero-event exception covered only one special case. | Always report the finite-sample envelope, including degenerate bootstrap cases. Explicitly label a zero bootstrap distribution degenerate. Do not infer impossibility of an event. |
| Major | Thirty seeds could be interpreted as providing precise confirmation. | Show that the distribution-free half-width is approximately 0.277 at thirty seeds. Retain approximate precision planning but require an inconclusive interpretation when the budget cannot support the desired precision. |
| Major | A pooled reference average could obscure the actual fixed-versus-rolling research question. | Require reference-by-mechanism all-cohort bounds and paired fixed-minus-rolling contrast bounds. An overall average cannot establish uniform reference superiority. |
| Major | Significance tests on truth-selected ratios could change the evaluated population; native drift truth also differs by reference. | Define a directly observable disagreement frequency against the same released-label verdict on the same cohorts. Six Holm-adjusted sign tests assess directional consistency of this disagreement, not population error or equality of mean effects. Native-target and certified-harm scores remain descriptive. |
| Major | The study could claim new error control that already follows from the existing interval. | Derive the unconditional 0.025 wrong-positive probability bound under the independent bounded-loss model. Explain that it also bounds expected all-cohort wrong added declarations, but not selected-positive error or lifetime error. The empirical contribution is primarily the yield and failure mechanisms. |
| Major | A below-margin label could sound like a statement that a model is healthy or safe. | Define harm narrowly as excess classification error. Require absolute errors alongside excess risk. Exclude claims about adequate absolute accuracy, fairness, safety or business value. |
| Major | Prediction-label alignment had no operational identity contract. | Specify immutable prediction IDs, one-to-one joins, model/preprocessing versions, event and release times, duplicate-ID rejection and conflict codes. Equal feature rows are not automatically duplicates. |
| Major | Operational failure could become an undocumented exclusion or retrospective repair. | Require an outcome-blind correction log and complete paired-seed rerun with the same configuration. Preserve append-only checkpoint records; later or corrected labels cannot rewrite past availability. |
| Moderate | Preprocessing did not define all-missing columns, nonfinite values or future-information hazards. | Specify zero/unit fallback for all-missing numeric fitting columns, retain missingness, flag schema/nonfinite failures, prohibit future interpolation and outcome-dependent deletion, and audit duplication before splitting. |
| Major | Input drift truth could be confused with what a lossy encoded representation can observe. | Explicitly monitor the frozen representation. Explain imputation/category-merging limitations. Core synthetic numeric transforms are invertible; natural missing/unknown rates remain separate descriptive evidence. |
| Major | A maximum-score threshold under the complete null could look like valid individual-feature discovery under partial alternatives. | Rename the output screened features and disclaim strong family-wise/FDR control under partial alternatives. Distinguish ranking, screening and causal explanation. |
| Moderate | Dependent coverage validation could reuse a reference and overcount independent trials; its target was unnecessarily uncertain. | Require independent reference/cohort trajectory pairs. Stationary marginal excess risk is exactly zero. Report binomial coverage uncertainty, no validation retuning, and no transfer to changing regimes. |
| Moderate | Natural pseudo-window generation was not reproducible. | Specify contiguous length-20 block resampling within each disjoint calibration half, forming reference-plus-cohort pseudo-trajectories with the declared update schedule. No rare-tail or independence certification is implied. |
| Major | Industrial quality was an aspiration without an evidence basis. | Add primary NIST AI RMF documentation/validity guidance and operational ground-truth matching documentation. Treat thresholds as research choices and alignment as a design objective, never certification or production validation. |
| Moderate | Resource limits could produce outcome-dependent omission of difficult conditions. | Require development timing and a fixed allocation before final evaluation. If core work is unaffordable, revise and freeze the protocol before opening final seeds. Separate infrastructure failure from scientific null findings. |

## Retained safeguards from earlier revisions

The primary gate suppresses positive harm declarations only; below-margin verdicts are identical. The loss-only comparator is an identity check. Input normalization is affine rather than a saturated empirical CDF. Classifiers, preprocessing, thresholds and prediction rules are frozen before final evaluation. Model development, calibration, final streams and oracle evaluation use separate random-number namespaces. Input reference truth is reference-specific; predictive-risk reference truth remains fixed. True-class uncertainty requires released labels. Pointwise monitoring intervals are not anytime guarantees. Label latency, permanent interval deferral and alarm-gate deferral remain distinct. Natural cases have no certified population-harm truth, and no adaptation, causal explanation, analyst benefit or label-cost saving is claimed.

## Remaining empirical limits

The protocol is a controlled benchmark of frozen classifiers, not production validation. Its six mechanisms, two model families and complete deterministic label releases constrain external validity. Five development seeds provide uncertain variance estimates. Thirty final seeds give a wide finite-sample envelope for correct-yield estimation; more precise approximate summaries will require qualification. Natural-data block-bootstrap intervals remain exploratory. An application-specific loss, acceptable error/yield trade-off, realistic missing-label process and operational validation would be required before a deployment recommendation. These limits are explicit rather than hidden by stronger wording.

## Verification and provenance

Source environment/brace balance, first-citation IEEE bibliography order, unique/resolved labels, all required deliverable counts and included figure paths were checked. Twelve finite policy cases verify coverage and added-declaration identities; uncertain-truth assignments and paired contrast bounds were checked. The sampling/oracle error-budget arithmetic and the one-sided wrong-positive bound were verified analytically. These are document and equation audits, not implemented research experiments.

Both vector figure sources were updated to match the all-cohort analysis and prediction identifiers; SVG/vector PDF/300 dpi PNG outputs were regenerated and previews visually inspected. Word SHA256 hashes still match the protected originals. Detailed machine-readable checks are in `submission_validation.json`.

The built-in `compile_latex_document` tool is unavailable, and Document Control reports no connected sessions. Therefore the full submission has not been compiled or checked for page-layout/compiler errors in this session. No replacement submission PDF or editor tab was created.

## Authoritative guidance consulted

- NIST, *Artificial Intelligence Risk Management Framework (AI RMF 1.0)*, 2023, particularly MEASURE 2.1 and 2.5: https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf
- AWS, *Model quality*: https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality.html
- AWS, *Ingest Ground Truth labels and merge them with predictions*: https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-merge.html

These support the cited documentation and matching practices, not a recommendation to adopt a service or a claim that the research is certified.
