# Phase 2 revision and remaining weaknesses

This note is for revision planning and is separate from the academic submission. The submission contains the core research question, extended literature comparison, gap synthesis, 12-step proposed methodology and academic references. All project actions are written as proposed work. Deliverable 4 is intentionally deferred until the content is finalized.

## Corrections made

| Weakness in the earlier response | Correction in the submission |
|---|---|
| Implementation audits, file paths, execution logs and existing pilot results made the proposal look like completed work. | Removed every project implementation or execution claim and every internal-project reference. Results in the comparison table are explicitly attributed to published studies. |
| The research question combined detection, localization, action validity, multiple interactions and transfer as competing primary targets. | Made harm-verdict reliability at comparable decision coverage the primary endpoint. Detection and localization remain secondary. |
| A policy could appear perfect by marking every case unresolved. | Added decision coverage, undefined error at zero coverage, comparable-coverage analysis and unresolved duration. |
| Delayed historical labels could be presented as evidence of current model health. | Made verdicts cohort-specific, retained evidence age and restricted conclusions about future observations. |
| Refreshing the input reference could silently redefine predictive harm. | Separated the input reference policy from a fixed predictive-risk reference. |
| Dependence-only evaluation needs invariant marginal distributions. | Proposed an explicit normal construction that varies correlation while preserving the marginals. No existing implementation is discussed. |
| Localization truth confused manipulated variables, marginal changes and dependency changes. | Required separate variable and dependency-group truth. |
| Exact replication language was not justified for tabular adaptations or simplified detectors. | Described comparator families and declared a bounded DDE extension rather than claiming exact paper reproduction. |
| Null alert budgets were liable to be interpreted as sequential lifetime guarantees. | Restricted the claim to held-out empirical stationary alert rates. |
| An aligned display could be taken as evidence of analyst improvement. | Kept the visualization as a proposed prototype and required a separate user study for such claims. |
| Label delay had no explicit deadline, allowing eventual full labels to erase the experimental distinction. | Specified a primary cohort assessment two windows after completion, with zero- and five-window sensitivities; later labels cannot rewrite the earlier verdict. |
| The harm channel could be indistinguishable from a loss-only rule. | Made identical interval-based loss monitoring an explicit control. Diagnostic context is not presumed to improve harm accuracy. |

## Remaining weaknesses and decisions before experiments

1. **Novelty remains provisional.** Evidence sufficiency, delayed labels and unified benchmarking already exist. The defensible contribution is the specific controlled interaction evaluation. Do not claim a new detector or that no similar workflow has ever been proposed. Relevant prior work: https://arxiv.org/abs/2604.15740 and https://arxiv.org/html/2606.07789v1.
2. **There is no completed power analysis.** Ten evaluation seeds are a feasible provisional count, not proof of adequate statistical power. Before final testing, specify the minimum meaningful difference, estimate variance using separate development runs and freeze the replication count. If it stays small, label interaction analyses exploratory.
3. **The 0.05 harm margin needs justification.** It is a provisional absolute error-risk change. Check its application relevance and reference uncertainty before freezing it; report sensitivity to reasonable alternative margins. It is not equivalent to a 0.05 macro-F1 decrease.
4. **A harm-verdict benchmark does not establish useful action.** A correct harm assessment cannot identify the best retraining, investigation or data-quality intervention. Recommendation costs stay secondary proxies. Actual adaptation benefit needs a separate matched-budget intervention experiment.
5. **Input-only harm verdicts are proxy baselines.** Scoring a distributional detector as if it directly predicted harm changes its task. Explicitly define the proxy mapping and include released-loss-only and simple combined baselines so the proposed policy is not rewarded against a straw-man baseline.
6. **Natural-stream calibration cannot manufacture a null.** An early natural-data segment is not certified stationary merely because it was used for calibration. Report natural alert burden and predictive association unless independent event or stationarity evidence exists.
7. **Risk intervals are not automatically valid under dependence or repeated inspection.** Block length, cohort eligibility, interval coverage and repeated comparisons need explicit rules. An ordinary bootstrap interval is not an anytime-valid risk certificate. Verify empirical coverage on controlled dependent streams.
8. **Long label delay can eliminate useful decision coverage.** Compare policies only over overlap in achievable coverage, and show evidence-waiting time. Do not extrapolate zero-delay performance into long-delay conditions.
9. **External validity is modest.** Electricity supplies one chronology; Covertype replay is constructed. Neither provides multiple independent deployments. Additional verified natural streams would strengthen transfer, but should not be added if they make the course study infeasible.
10. **Selective labels are outside the core design.** Fixed delays with eventual full coverage avoid selective-observation bias but do not cover permanent missingness or active label acquisition. Keep those as extensions instead of implying that the proposal resolves them.
11. **Feature explanation remains statistical.** Correctly ranking changed marginals does not establish a causal root cause, and a joint alarm does not automatically localize its source. Report group-level limits and ambiguous examples.
12. **Course eligibility still needs a separate check.** Supporting older studies and 2026 preprints establish prior art but do not automatically count as eligible recent Q1 journal or ranked-conference research papers. Do not infer journal quartiles or conference acceptance from reputation. The Phase 2 submission uses three primary comparison papers as requested; it does not certify Phase 1 corpus compliance.
13. **The student identifiers conflict in the retained material.** The Phase 1 cover uses Saneedullah 23I-2568 and Ahmed Bilal 23I-2581, while the supplied filename contains 23I-2604. The new document uses names only. Confirm IDs before adding a cover sheet.
14. **Deliverable 4 remains outstanding by request.** Finalize methodology wording before generating and inserting the diagram. Deliverables 1-3 are available in the Word files; the whole Phase 2 submission should not be treated as complete until the diagram is added.
15. **The proposed rule is not a new detector and may add no harm-verdict accuracy over loss-only monitoring.** The two interval rules are deliberately identical. The defensible experiment compares proxy and gated verdict failures under operating constraints; the added input and feature evidence is diagnostic context. Demonstrating extra analyst value would require a user study, and demonstrating better predictive decisions would require a distinct policy and evaluation. Do not frame this as guaranteed algorithmic novelty.
16. **Coverage matching may be impossible for some policies.** A deterministic input proxy always decides, while long-delay verification may decide nothing at a checkpoint. Comparisons must show both coverage and error, restrict direct error comparisons to shared operating points, and refrain from implying that low-coverage error alone establishes superiority.
17. **Synthetic risk truth has its own uncertainty.** Independent generator-based estimates must use disjoint samples, sufficient precision and a declared boundary band around the harm margin. Boundary-uncertain cohorts need their own count; they cannot be silently removed to improve primary results.

## Automated methodology diagram recommendation

Use a text-defined flowchart rendered by Graphviz DOT after the methodology is finalized. This is an automated authoring route rather than hand drawing. Graphviz's dot engine automatically lays out hierarchical directed flows; its command line exports SVG and PDF. Mermaid is a suitable alternative when easy Markdown preview matters more than precise layout. Neither tool alone makes a diagram compliant with a standard.

Use a top-to-bottom layout with concise process labels, named input/output artifacts, directed arrows and decision diamonds only for real branching conditions. Group the 12 steps into data preparation, monitoring, experimental evaluation and research output. Show the delayed-label channel separately from immediately available inputs, and distinguish calibration from test evaluation. Keep the diagram at methodology level rather than depicting completed software modules. The research pipeline should not gain adaptive feedback loops that the proposed frozen-model study does not contain.

Retain editable DOT source plus SVG/PDF masters. For dependable embedding in both modern and legacy Word formats, use a high-resolution PNG derived from the vector master and visually verify it at its printed size. The eventual diagram should fit one page, use readable labels and reproduce the final written method exactly.

Sources checked on 8 October 2026:

- Graphviz command line and export: https://graphviz.org/doc/info/command.html
- Graphviz hierarchical layout: https://graphviz.org/docs/layouts/dot/
- Mermaid CLI SVG PNG PDF generation: https://github.com/mermaid-js/mermaid-cli
- ISO 5807 flowchart documentation conventions: https://www.iso.org/standard/11955.html

ISO 5807 is a general flowchart documentation standard. It does not impose one universally required format or software package for academic research methodology diagrams. This recommendation uses its broad documentation purpose; no full standard-compliance claim is made.
