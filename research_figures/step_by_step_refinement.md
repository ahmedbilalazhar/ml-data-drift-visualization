# Twelve-step methodology refinement

Reviewed and saved 23 October 2026. The open LaTeX source was edited in place. The study remains a Phase 2 proposal. No experiments, research implementation or Word conversion were performed.

| Step | Improvement | Reason |
|---|---|---|
| 1 — Research input | Separate cohort analysis from seed replication; distinguish observation, completion, release and assessment clocks; define cohort-evidence age. | Prevents the reader from interpreting old cohort evidence as current performance or counting reused condition cells as independent replications. |
| 2 — Data collection | Separate fitting, calibration, test and evaluator draws; preserve cancellation of the common baseline risk; give the baseline-to-post mixture risk identity. | Makes evaluation truth more reproducible and avoids unnecessary interval width from treating the same risk term as two unrelated quantities. |
| 3 — Preprocessing | Define exact cohort membership and a common trailing-partial-cohort exclusion; retain fitted transforms and original-to-encoded mapping. | Prevents policy-specific exclusions, loss-dependent partitioning and accidental reference overlap. |
| 4 — Representation | Specify normalized Wasserstein and categorical Jensen–Shannon distances, their units and zero conventions; distinguish original variables from encoded coordinates. | Removes ambiguity over double scaling and inflated localization denominators. |
| 5 — Proposed method | Define released-label coverage and record/join validity; give mutually exclusive eligibility/verdict cases; place these before dependent-data extensions. | Invalid or incomplete evidence cannot produce a below-margin or harmful verdict. Invalid records trigger recorded repair rather than silent exclusion. |
| 6 — Baselines | State the roles of detector comparators, matched gate policies, uncertainty ablations and constant controls; require a common cohort record/checkpoint. | Prevents a change in evidence access or a calibration target from being mistaken for a fair comparison of power, cost or verdict quality. |
| 7 — Experimental design | Count the planned independent streams and reused condition cells; specify exact gradual weights; require all affected conditions to rerun after a shared-routine correction. | Makes workload and dependence transparent while limiting selective repairs. The planned counts are 24 primary cells per seed, 72 scheduled cells per seed, 180 final mechanism streams and 10.8 million test observations. |
| 8 — Metrics | Lead with the two latent all-cohort yield components and their observable sum; require each table to name its denominator, eligibility, truth source and uncertainty type. | Makes the primary estimand visible before certified-subset diagnostics and keeps unavailable labels distinct from uncertain evaluation truth. |
| 9 — Statistical analysis | Distinguish oracle precision, independent-seed variation and bootstrap Monte Carlo error; record an independent analysis-resampling seed. | Prevents precision conditional on one calibration/development configuration from becoming a general deployment claim. Existing finite-sample and approximate reports remain separately qualified. |
| 10 — Ablation | State invariant quantities for each component removal; use one-factor sensitivities outside explicitly crossed core factors. | Supports interpretable component attribution and prevents an unintended full Cartesian sensitivity study. |
| 11 — Failure analysis | Enumerate all cases first; use deterministic chronological examples and explicit unknown diagnoses. | Reduces cherry-picking and avoids calling an observed association a confirmed root cause. |
| 12 — Research output | Specify the versioned research package and evidence-to-display reproducibility criteria; separate templates from future measured results. | Keeps the hand-in within Phase 2 while making later implementation and evaluation reviewable. |

## Diagram redesign

The overview now has twelve numbered nodes, grouped into preparation, method specification/experiments, complementary analyses and synthesis. Solid connectors express dependencies; the branch from metrics feeds statistical, ablation and failure analysis. The dashed panel groups crossed reference and label-delay settings. The figure is a protocol map, not a claim that every numbered heading is a serial runtime operation.

The detailed architecture shows explicit offline dependencies, parallel input and released-label channels, a common cohort evidence ledger, two ledger outputs (display and policy evaluation), and an evaluator-only synthetic risk source. It removes the earlier implication that the display supplies the evaluation evidence. No oracle arrow feeds monitoring. Natural-case uncertainty remains exploratory.

Both figures use editable SVG and programmatically generated vector PDF with embedded TrueType fonts, and include 300-dpi PNG previews. Following the literature-informed design revision, both figures are 180 × 166.667 mm; smallest labels are 9.69 pt at the intended 180-mm width. The current style rationale and paper examples are recorded in `diagram_design_research.md`. They are intended to span two paper columns. Color is supplemented by text, node numbers, grouping and line styles.

Publisher production guidance supports editable vector elements and consistent text, while NIST AI RMF supports documented test/evaluation methods. Neither source defines a universal diagram certification or validates the proposed research outcomes:

- [Nature research figure guide](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/)
- [NIST AI RMF Core](https://airc.nist.gov/airmf-resources/airmf/5-sec-core/)

## Verification

Checked the saved source for environment and brace balance, all twelve prescribed headings, four deliverables, eight additional selections, citation order (25 cited references), resolved labels and included figure paths. Twelve eligibility cases and twelve policy-algebra cases were checked, along with identification bounds, mixture-risk arithmetic, design counts and error-budget identities. This is a document/equation audit, not a research experiment.

Both exported PDFs contain vector elements without raster images and embed regular/bold TrueType fonts. SVGs retain editable text. Every overview step number occurs exactly once. The latest rendered previews were visually inspected after connector and spacing refinements. The figure manifest matches the current LaTeX source checksum. Protected Word hashes are unchanged.

The built-in LaTeX compilation tool remains unavailable, and Document Control reports no connected sessions. The complete submission PDF has not been compiled or visually validated in this session. No separate submission PDF or editor tab was created.
