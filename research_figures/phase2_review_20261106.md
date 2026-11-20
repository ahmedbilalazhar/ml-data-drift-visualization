# Phase 2 research review — 6 November 2026

The existing submission source was revised in place. No model-training
experiment, measured project result, replacement submission, separate
submission PDF or Word regeneration was produced.

## Substantive weaknesses addressed

1. **No simultaneous input/conditional-change case.** Replaced the former
   gradual pure input-mean shift with a gradual mixed shift. Its post-regime
   has X1 shifted by one and label probability
   pi_eta(x) = eta + (1 - 2 eta) pi_0(x), eta = 1/4.
   The core still has six mechanisms, two classifiers and two input references.
   A matched eta=0 control shares inputs, predictions, mixture indicators and
   label uniforms. Input alarms must agree; released-loss evidence can differ.
   This identity is a design check, not an empirical discovery.
2. **The mixed-shift truth needed a justified oracle.** For any frozen binary
   classifier, its mixed post-risk is eta + (1 - 2 eta) times the canonical
   shifted-input risk under pi_0. The latter is already in the oracle register.
   The full register remains thirteen canonical input laws per classifier,
   twenty-six risk functionals per seed, with 239,153 draws per functional.
   The existing core and full oracle workload calculations therefore remain
   valid. Transition risks use mixtures of joint regimes, with the posterior
   of the selected regime; conditional probabilities are not naively averaged
   given an input whose density differs between regimes.
3. **Counterfactual labels could contaminate joins or interval caches.**
   Label-variant namespaces and join keys now distinguish paired label logs.
   Interval cache keys include stream, label variant, cohort, classifier and
   interval configuration. Only input scores can be reused across identical
   inputs; each label variant recomputes its loss evidence.
4. **Truth grouping could be misread as grouping by input law.** Shared harm
   truth now explicitly requires the same joint law of inputs and labels.
   Equal input laws alone do not justify pooling harm status.
5. **A degenerate bootstrap could conceal weak false-alarm validation.**
   Added a finite-sample bounded-mean KL interval for the twenty independent
   synthetic null trajectories, alongside the approximate bootstrap interval.
   Its guarantee is pointwise by configuration and targets the specified
   hundred-window validation horizon. An all-zero trajectory sample still has
   upper limit about 0.1684 under these cluster-level assumptions, so it
   cannot certify a one-percent population burden.
6. **Coverage-diagnostic uncertainty was underspecified.** Added explicit
   pointwise Hoeffding intervals for the 2,000 validation-pair mean and
   400-pair model-specific means. Confidence-level sensitivities explicitly
   recompute the applicable one-sided wrong-positive bound.
7. **Objectives and inference could appear misaligned.** The objectives now
   distinguish the two pooled primary means from exploratory reference
   contrasts and mechanism-specific analyses. The joint primary guarantee
   cannot be used to assert a confirmatory reference ranking.
8. **Precision@3 has a structural ceiling in sparse feature truth.** The
   report now includes changed-set size and the attainable ceiling. Added
   full-ranking average precision and its exact tied-order minimum/maximum.
   These ranges describe ranking ambiguity, not sampling confidence.
   Native reference summaries retain their eligibility counts; paired ranking
   contrasts use a common nonempty marginal-truth set.
9. **Display and numerical conventions needed completion.** Resampling
   percentiles follow the declared type-7 convention. Display panels identify
   prediction-time span, label scenario and interval confidence level.
   The reversed conditional case is described as reversing Bayes class
   assignment, rather than moving its geometric boundary.

## Verification

- Source structure, citation order and completeness, cross-references,
  selected-paper count, table widths, twelve steps and four deliverables pass.
- The ten selected papers and twenty-six references are retained.
- Mixed-posterior/risk identities: 48 deterministic cases pass.
- Transformed mixed-risk bounds: 96 cases pass.
- Canonical input-law register: five core and thirteen full laws per classifier.
- Null-validation KL envelope: exact convolution checks pass for twelve bounded
  distributions, including non-Bernoulli trajectory fractions.
- AP tied-order extrema: exhaustive checks pass for 124 small ranking/truth cases.
- Built-in LaTeX compilation passes for the saved source.
- Protected DOC/DOCX hashes are unchanged.

These are specification checks, not project experiment outcomes.
The primary confidence guarantee remains conditional on its stated independent
design assumptions. Finite-sample precision, runtime feasibility and empirical
usefulness still require the planned experiments. Full submission page layout
has not been inferred from compiler success.

Detailed records are under tmp/phase2_review/2026-11-06/.
The standard AP interpretation was checked against the official scikit-learn
documentation; the submission defines its own explicit ranking and tie rules:
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
