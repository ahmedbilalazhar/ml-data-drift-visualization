# Problem statement (one page, roadmap Sec. 16.1)

> A researcher monitoring a deployed tabular classifier receives feature
> windows immediately and labels later. They must distinguish distribution
> change from demonstrated predictive harm, locate affected variables, and
> decide: continue observing, investigate data quality, request labels, or
> evaluate a model update.

Key distinctions: P(X) change (unlabeled windows) vs P(Y|X) change (needs
labels/generator truth) vs predictive degradation (logged preds + released
labels) vs drift explanation (statistical, not causal) vs proactive warning
(pre-registered horizon, tested separately). Pure conditional change (S7) is
invisible to feature-only detectors BY DESIGN — the interface must show
"no measured input change / performance unknown" rather than false reassurance.
