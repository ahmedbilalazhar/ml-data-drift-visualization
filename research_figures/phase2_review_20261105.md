# Phase 2 review — 5 November 2026

This record concerns the existing submissions/phase2/DAV_Phase2_Submission.tex.
It documents source review, reference verification and mathematical specification
checks. No proposed model experiment was run and no measured project result was
added to the submission.

## Material corrections

1. **The closest recent literature was missing.** Added the published ICML
   Prediction-Powered Risk Monitoring (2026) and Suitability Filter (2025)
   studies. Both publisher records, linked published PDFs and author versions
   were reviewed. The selected corpus is now ten publications, within the
   brief's eight-to-ten range. All ten receive the nine-column comparison.
   The gap matrix now uses window interpretation, predictive-risk monitoring
   and margin-based suitability as its closest reference points.
2. **The contribution needed narrower attribution.** Existing work already
   uses risk tolerances and can make decisions without input alarms.
   The proposal now focuses on a paired audit of reference maintenance and a
   specified positive gate on the same named-cohort interval. It does not
   present that gate as standard practice or claim algorithmic superiority.
3. **The finite-sample precision statement was unnecessarily weak.** The
   primary envelope now inverts the bounded-mean Chernoff–Hoeffding KL bound,
   retaining the 0.04 sampling and 0.01 oracle allocations. The known correct
   yield range is at most 0.5 under the fixed benchmark mixture. This is valid
   for bounded fractional seed frequencies; it does not pretend that those
   frequencies are Bernoulli observations or turn the bootstrap into an exact
   confidence procedure.
4. **Sensitivity truth estimates could exceed the stated oracle error budget.**
   The union includes 26 directly estimated regime–classifier risks per seed,
   not just the ten core risks. At the declared precision and error budget,
   each uses 239,153 oracle draws. The full prospective workload is
   186,539,340 classifier–risk evaluations; the core subset uses 71,745,900.
   Smaller core-only calculations are labelled as a different prospective
   protocol, not reused for the full union.
5. **Unnecessary oracle label noise was avoidable.** The synthetic evaluator
   now averages conditional expected error using the known generator
   probability. The bounded-loss guarantee remains applicable.
   Monitoring continues to use observed labels after their declared release.
6. **Reference comparisons discarded some exact truth dependence.** Identical
   cohort laws and frozen classifiers share a harm status. Signed differences
   are aggregated within those groups before bounding uncertain truth.
   Identical/opposite contributions can cancel, while denominators and
   unresolved truth remain intact.
7. **All-cohort error alone can hide poor selected reliability.** Added a
   bounded false-claim proportion among the additional declarations.
   Zero declarations produce an undefined proportion, not perfect reliability.
8. **Natural validation preprocessing was ambiguous.** Candidate transforms now
   fit only the internal 80% fitting prefix and remain frozen on the 20%
   validation suffix. Selected settings refit on the full training prefix.
   Feature availability, classification-versus-forecasting scope and
   observation-index latency units are explicit.
9. **Dependent-data validation lacked a model and sample allocation.**
   Block-length selection uses 100 common selection pairs for each of five
   frozen development models; fresh validation uses 400 per model. Coverage
   is assessed against exact stationary excess risk zero, conditional on the
   development models, with bounded-mean uncertainty. It is a diagnostic,
   not a transferable finite-sample coverage guarantee.
10. **Sensitivity and failure interpretations needed more specificity.**
    Mean-shift multipliers and correlation variants are now explicit.
    The condition table explains each mechanism's scientific role.
    The Bayes-rule check shows why input change can improve error and why
    increased error under an unchanged conditional law need not be recoverable
    by retraining. Component scores and fitted artifacts are retained for
    reconstruction.
    The supplementary stationary dependent control has an explicit budget of
    thirty 60,000-row test trajectories, separate 10,000-row risk references
    and the primary cohort/checkpoint setting. Its 1.8 million test rows are
    outside the independent core count and confidence guarantee.
11. **Presentation and notation had small inconsistencies.** Widened the
    study-ID column for S10, used available longtable caption width, separated
    the proxy-error contrast symbol from top-k sets, and removed repeated
    process descriptions. The existing title page and two vector figures are
    retained.

## Reference evidence

Publisher records verify all ten selected studies as 2025–2026 conference
research/proceedings chapters. The selection uses the brief's conference or
chapter route; it does not assert journal quartiles for proceedings.
The SHAP forecasting chapter has a 2025 first-online date and a 2026 published
volume citation. Earlier methodological papers, operational documentation and
the Solozobov preprint are clearly outside the selected count.

The DOI metadata audit recovered all sixteen DOI records in the bibliography.
PMLR, JMLR and arXiv records were checked separately for non-DOI entries.
OpenReview displayed a browser-verification challenge; no bypass was attempted.
Published/source versions are distinguished where author copies supply the
reviewed methods.

Supporting records:

- references/papers/phase2_additional/publication_audit.json
- references/papers/phase2_additional/source_manifest.json
- references/papers/phase2_additional/search_followup_20261105.json
- tmp/phase2_review/2026-11-05/check_crossref_metadata.json

## Verification and limits

The saved source is checked for four deliverables, twelve ordered methodology
steps, ten comparison rows, nine comparison columns, citation order, reference
completeness, cross-references, environment/brace balance and table widths.
The gate algebra, shared-truth reference contrasts, partial-identification
bounds, release clocks and oracle workload arithmetic are checked independently.
KL inversion is checked over 601 grid points and exact finite distributions
with bounded non-Bernoulli observations. These are specification checks, not
project experiment results.

The two existing figure previews were visually inspected. Full submission
page layout is not inferred from source checks: the built-in compiler exposes
success diagnostics but no rendered pages to these tools. No separate
submission PDF, replacement document or new editor tab was created.
Compiler status is recorded against the final saved source hash in the current
validation record. The original DOC and DOCX hashes remain unchanged.

The protocol still requires future feasibility measurements, actual final
experiments and separately qualified natural-case analysis. Thirty seeds do
not assure narrow confidence limits or practical benefit.
