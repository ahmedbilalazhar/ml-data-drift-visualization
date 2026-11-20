# Research basis for the Phase 2 diagram style

Reviewed 23 October 2026. The existing LaTeX source and both figure definitions were revised in place. Word files and research implementation code remain unchanged.

**6 November review note.** This is the historical eight-paper diagram-design review. Study numbers below refer to that earlier selection: SHAP forecasting and performative drift are now supporting context, while current S6 and S8 are PRODEM and risk-violation monitoring. SHAP forecasting's publisher-recommended volume citation is 2026, with first online publication in 2025. The current selection has ten studies; its eligibility and comparison are recorded in the submission and current publication register. The diagram semantics remain applicable. The built-in compiler is now available; current saved-source compilation status is recorded in `submission_validation.json`, separately from full submission page-layout inspection.

## Selection

The selected pair is a **numbered research workflow** and a **swimlane-style information/data-flow architecture**. This is a judgement about the present research question and corpus, not an assertion that one notation is universally best. The workflow maps the prescribed twelve steps. The architecture explains the two evidence channels, the eligibility decision, the common record and the evaluator-only risk source. A neural-network layer diagram would misrepresent this frozen-classifier evaluation, while causal arrows would imply assumptions the proposed monitor does not establish.

## Examination of the user's papers

The eight Phase 2 texts plus three retained Phase 1 sources were inspected for figure captions and diagram roles. Representative workflow, framework, window, causal and coordinated-view pages were rendered and visually inspected. The automatic inventory records figure mentions, not a reliable count of unique diagrams; in-text mentions repeat and some algorithm listings are not figures.

| Source | Observed form and locator | Application to this submission |
|---|---|---|
| S1, KDD benchmark (2026) | Fig. 1: event-evaluation timing schematic, PDF p.3; Fig. 2: inference workflow, PDF p.6. | Separate a workflow's control structure from event metrics; do not import the source's model-update path into a frozen-classifier experiment. |
| S2, Window Dilemma (2026) | Fig. 1, PDF p.4: small-multiple distribution/window illustration. | Preserve reference-specific comparison meaning; do not invent distributions or label a non-detection proof of equality. |
| S3, PUDD (2025) | Fig. 2, PDF p.4: component/framework schematic for windows, bucketing and testing. | Use bounded functional groups and a clear information-flow direction; retain the fact that true-class uncertainty needs labels. |
| S4, Flexible and Efficient Drift Detection (2025) | Fig. 1, PDF p.3: batch-distance algorithm visualization next to Algorithm 1. | Make data objects and computations distinguishable; the proposed figure need not reproduce batching or the source algorithm. |
| S5, Causal Explanation (2026) | Fig. 1, PDF p.5: directed causal graph and intervention example. | Do not borrow causal-edge meaning. Arrows in the proposed architecture mean information dependencies. |
| S6, SHAP forecasting (2025) | Fig. 1, PDF p.4: hierarchical retraining workflow with historical-data store; Figs. 2–3: results heatmaps/time series. | Use explicit preparation and data records, but omit retraining. Keep numerical results plots out of a proposal that has no measured project outcomes. |
| S7, Conformal recurrence (2025) | Algorithm listings and evaluation tables; the automated caption scan found no numbered figure captions. | Formal method detail can live in equations/algorithm specifications; a figure need not pretend every study uses the same architecture style. |
| S8, Performative drift (2025) | Fig. 1, PDF p.3: checkerboard mechanism; Figs. 2–9: parameter/detection-rate results. | Retain the boundary between observational evidence and deliberate prediction interventions. Do not import a feedback mechanism into the proposed protocol. |
| DriftLens (2025), retained Phase 1 | Fig. 2, PDF p.4: offline/online framework; Fig. 3: data modelling. | Make fitted historical artifacts and runtime evidence separate and explicit. |
| DDE (2025), retained Phase 1 | Fig. 1, PDF p.9: feature-distance/time curves; Fig. 2, PDF p.12: lift curve. | Treat result curves as evidence-specific outputs, not decorative components of a methodology diagram. |
| DriftVis (2020), retained Phase 1 | Fig. 2, PDF p.4: data, stream-level and prediction-level views; Fig. 3: analysis workflow. | Let a common evidence record feed a display, without inferring demonstrated analyst usefulness from a diagram. |

Published identities and the core-paper full texts remain in the existing corpus register. No reviewed image is pasted into the submission; both revised figures are original programmatic drawings.

## External research and production guidance

- [Marshall, Freitas and Jay (2025), PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0318800), DOI 10.1371/journal.pone.0318800. Empirical diagram-guidance research, based on interviews, a diagram-revision evaluation and corpus analysis. Its neural-network scope supports cautious adaptation, not a proven usability result for this monitoring diagram. Publisher and Crossref metadata were checked.
- [Rougier, Droettboom and Bourne (2014), PLOS Computational Biology](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003833), DOI 10.1371/journal.pcbi.1003833. Figure-design guidance prioritizing the message, appropriate visual form and explanatory caption. This is supporting guidance, not one of the eight eligible new topic papers.
- [IEEE Author Center: File Formatting](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-graphics-for-your-article/file-formatting/). Supports PDF output, embedded fonts and approximately 9–10-point figure text at full size.
- [IEEE Author Center: Resolution and Size](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-graphics-for-your-article/resolution-and-size/). Distinguishes vector masters from raster previews and gives column widths.
- [Nature Research Figure Guide](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/). Supports editable vector elements and text.
- [OMG UML 2.5.1](https://www.omg.org/spec/UML/2.5.1) and [BPMN 2.0.2](https://www.omg.org/spec/BPMN/2.0.2) were considered as formal notation families. The present figures are research schematics and do not claim formal conformance to either specification.

## Implemented visual choices

The phase bands are labelled rather than decorated with large opaque banners. All information-dependency edges use the same dark arrow style. Colour reinforces named roles: preparation, input evidence, released-label evidence and assessment. The overview preserves twelve exact step numbers. The architecture uses rounded computation boxes, record cylinders and an eligibility diamond. A dashed boundary means read-only reporting/assessment only; alternative experimental settings use a plain labelled panel. The W0 dependency is routed between nodes, with a small underpass at an unconnected crossing. Invalid records stop the run for repair; an incomplete-label branch yields unresolved status. Evaluation truth has an edge to policy assessment only, never to the monitor.

The distinction between a design specification and a runtime data flow is stated in the manuscript. No synthetic output curve, causal effect, deployed service, neural architecture, feedback-learning loop or completed experiment has been invented.

## Exports and checks

Both masters are 180 × 166.667 mm. Minimum text is 9.69 pt at native size and approximately 9.18 pt at the width used by the current LaTeX source. PDF fonts are embedded; SVG text remains editable. PNG previews are 300 dpi, while vector PDF is the publication master. All labels fit their defined widths. The rendered figures were inspected for node/text collisions, arrow meanings, record boundaries and scientific agreement with the protocol. Source, citation, diagram and protected-Word checks are recorded in `submission_validation.json`.

At the historical design review, the built-in LaTeX compiler was unavailable and Document Control reported no connected sessions. That historical limitation is superseded for compilation by the 6 November compiler checks; full submission page-layout verification is not inferred from those checks. No replacement submission PDF or editor tab was created.
