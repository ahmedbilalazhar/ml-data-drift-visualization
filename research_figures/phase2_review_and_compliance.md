# Phase 2 review against the supplied official brief

Reviewed and revised: 22 October 2026. This audit concerns proposal quality and submission scope; no research implementation or experiment was performed.

## Material weaknesses addressed

1. **Additional-paper requirement was unmet.** The previous three focal papers were already in the retained Phase 1 bibliography. The revision selects eight different 2025–2026 original conference papers/research chapters, archives their complete accessible texts and records publication/source links and checksums under `references/papers/phase2_additional/`. Older foundations and the delayed-label survey are not counted as the eight selections.
2. **Focal studies were distant from the final question.** The new nine-column comparison uses the KDD benchmark, Window Dilemma and PUDD papers. Together they address experimental truth, reference/window meaning and label-based evidence. The remaining five new papers are critically synthesized rather than merely listed.
3. **Prediction uncertainty could be mistaken for label-free evidence.** PUDD Equation 7 uses the probability assigned to the true class. The review explicitly identifies its label requirement and does not replace it with predicted-class confidence.
4. **Input and localization truth ignored reference meaning.** The revision defines the generating distributions for each current/reference comparison, reference-specific input-state truth and feature truth, and one common predictive-harm target. A fixed reference after a return to baseline is no longer automatically penalized for failing to raise a positive change alarm. Event scores with different eligible target sets are not ranked as though their denominators were identical.
5. **The power calculation lacked an appropriate primary hypothesis.** The two added-decision components have nonnegative directions by construction. The revision makes the primary question estimation-based and plans interval precision rather than claiming power to discover that ordering.
6. **Constant label delay was liable to be overstated.** Eligibility time is now explicitly determined by the release schedule. Once all cohort labels are available, later checkpoints use the same loss estimate. Availability, interval deferral and alarm-gating deferral are separated.
7. **Dependent-data resampling was underspecified.** The revision specifies overlapping block starts, separate reference/cohort sampling, truncation, replicate quantiles and development/validation roles. It does not mix different regimes into a fictitious stationary bootstrap sequence or extend stationary validation to nonstationary transitions.
8. **Easy pre-change decisions could dominate the headline.** Mandatory onset and harm-stratum breakdowns now accompany the all-cohort primary summary.
9. **Deliverable 4 was absent.** The two programmatically drawn vector diagrams are now included in the LaTeX source and exported as SVG, vector PDF and PNG, with editable Python source.

## Brief-to-deliverable mapping

| Requirement | Revision |
|---|---|
| 8–10 additional papers, published 2025–2026 | Eight selections S1–S8, none present in the retained Phase 1 bibliography |
| Original research; eligible publication route | Published conference papers and Springer research chapters; each has a method and empirical/analytical evaluation |
| Journal Q1–Q2 evidence where journal route is used | The eight counted additions use the conference/chapter routes; no journal quartiles are invented for proceedings |
| Detailed three-paper comparison | Nine required columns; focal studies S1–S3; synthesis separates tasks and evaluation assumptions |
| At least three research gaps | Three-row evidence matrix and six-part formal justification |
| Twelve prescribed methodology steps | All twelve retained with explicit methods, mathematical targets and metrics |
| Methodology architecture diagram | Overview and framework architecture included as vector PDFs |
| Proposed work only | Published findings are attributed; project methods/results remain prospective |
| Word files protected | DOCX and DOC remain unchanged; hashes are checked separately |

## Limits that proposal wording cannot eliminate

The study is a bounded evaluation contribution, not a new detector or a proven improvement over loss-only monitoring. Actual interval precision, computational feasibility, empirical findings and cross-domain transfer remain untested. The constant complete-release schedule does not cover selective labels, variable delays or partial-evidence accumulation. Natural harm truth is unknown and its bootstrap uncertainty remains exploratory. Faithful PUDD and conformal-adaptation reproduction is outside the core frozen-model comparison; no superiority over those published systems is claimed.

The official brief permits conference papers and book chapters. The corpus records this eligibility route explicitly rather than implying journal rankings for proceedings. Publisher metadata establishes the publication year; an earlier arXiv posting does not determine that year. Any instructor-specific venue-list rule beyond the supplied brief would require an additional administrative eligibility check.

## Verification status

The figure masters are rendered and visually inspected, and the generator verifies editable text, text widths and raster-free vector PDFs. The submission receives structural, citation, step-count, figure-path and corpus checks. The built-in LaTeX compiler is not exposed in this session, so the full submission PDF layout is **not verified**. No replacement submission PDF or different editor tab is created.
