# Source audit and revision record

Audit date: 4 October 2026. Target: Phase1_Drift_Report.tex.

## Verdict

The revision corrects source-version attribution, narrows unsupported gap headings, strengthens the evaluation critique, and separates recent research candidates from background documents. It remains provisional. The supplied collection does not establish eight to ten eligible studies, and Q1 evidence for four journal candidates is outstanding. No full-marks guarantee is justified.

## Findings and disposition

1. **Critical: corpus eligibility.** The original draft discussed ten documents, but five were contextual rather than confirmed eligible studies. Five original 2025/2026 candidates are available; four journal candidates still require ranking evidence. The revision separates the comparison into five candidate rows and five B-labelled contextual rows. It requires at least three additions if all candidates qualify, and more if they do not.
2. **Major: medical-imaging version mismatch.** The supplied arXiv v3 includes RSNA and reports earlier aggregate numbers. The accepted MICCAI paper's Section 3.2 describes PadChest, EMBED, and RETINA, the latter combining Kaggle DR, Aptos and Messidor-v2. Section 4.2 reports 91% average prevalence identification with 500 test images, 87% covariate identification, and 75% mixed-shift identification under its stated image/exam conditions. Earlier numbers were explicitly labelled as supplied-version results, so they were not fabricated, but reconciliation was unfinished. The revision now consistently uses the accepted paper for datasets and results and removes unconfirmed final-version preprocessing details. A PDF and extracted verification copy are cached in qa_sources.
   The accepted source itself has inconsistent count wording: its abstract says five public datasets, while Section 3.2 says four but enumerates PadChest, EMBED and composite RETINA. The revision uses the named sources: two individual sources plus three retinal sources, grouped into three collections. It does not silently repeat the inconsistent count.
3. **Major: research-gap scope.** The original headings imply absent integration or cross-regime evaluation. DriftVis already integrates diagnosis/adaptation; ConceptExplorer coordinates concept analysis; proactive trees include natural streams, rapid/abrupt stress tests and other learners. The revision retains all three main themes but defines missing controlled workflow benefit, prospective score-to-loss calibration, and joint analyst-task/transfer validation. No claim of globally unestablished novelty is made.
4. **Moderate: mathematical and empirical limitations.** FDD information loss, unit sensitivity, non-rejection/equality distinctions and selective-label bias are deductions or risks to test, not author-reported empirical failures. The revision labels these correctly. Posterior change with invariant inputs is a known observability limit, not a proposed dashboard innovation.
5. **Moderate: BAR interpretation.** Assis and Souza's Equation 5 uses ordinary accuracy and label economy. The previous absolute statement about zero at full label use needs the positive-accuracy condition; zero accuracy plus full label use gives 0/0 without a convention. The revision corrects this and requests separate component reporting.
6. **Moderate: comparative evaluation.** DriftLens timing excludes embedding extraction. Its baseline reference sizes differ, with the 20 Newsgroups shared smaller-reference exception. DDE's savings/quality trade-off depends on the detector. Proactive-tree synthetic improvements do not establish natural-stream superiority. CV4CDD's retrospective positional tolerance is not online delay. The revision retains these distinctions and adds matched-budget evaluation and the appropriate uncertainty unit.
7. **Minor: locators and layout.** The proactive paper's limitations are Section 7, not Section 6.5. Table cross-references are automatic. Long slash-separated metric strings were replaced with breakable text. The final summary occupies exactly two pages; a stranded list item was eliminated.

## Numerical evidence checked

| Study | Source locator | Verification |
|---|---|---|
| DriftLens | Sections V-B/V-C/V-D/V-E; Tables III–IV | 15/17 headline; after-embedding speed claim; severity correlation rather than loss; prototype purity 0.73–0.92 on the reported selected cases |
| DDE | Table 4 | Heartbeats ADWIN 8/280: recall 0.8436, precision 0.2951, F1 0.4373; JS 17/280: F1 0.8548 |
| Proactive trees | Tables 7, 13, 14; Section 7 | PHT-MC/VFDT synthetic accuracy 0.9574/0.9406; natural accuracy 0.7646/0.7654; runtime 16.73/7.28 s; existing stress/sensitivity/generalization experiments credited |
| ADWIN-U | Equation 5; Tables 8–9; Section 6.3 | StarLightCurves kurtosis accuracy 89.63%, BAR 86.70%; KS accuracy 91.62%, BAR 86.43%; donor change positions exist for selected streams |
| MICCAI shift identification | Accepted paper Sections 3.2, 4.2, 5 | Accepted datasets/results supersede supplied-version ones; image/exam units preserved; population/acquisition subtyping limitation retained |
| CV4CDD-4D | Table 3; conclusion | CDRIFT F1 0.56 at 1%, 0.92 at 2.5%; positional tolerances; offline whole-log analysis and generator/localization limits |
| DriftVis | Table 1; discussion | Mean stream: 97.9 promptly detected and 0.8 false alarms; variance stream: 77.5 and 6.4; integrated workflow already exists |
| ConceptExplorer | Case studies; Section 7 | Three cases and three expert interviews; label/error dependence and 15-attribute display; no controlled task effect size established |
| Lukats et al. | Publisher/source metadata; metric analysis | 2024 first online, 2025 volume, Review label with original benchmarking; opposing metric/alarm-frequency incentives |
| Bayram et al. | Supplied title page and overview | Firas Bayram (local Markdown header incorrectly says Fatma); 2022 overview establishes prior performance-aware detection |

## Remaining submission work

1. Complete exactly eight to ten eligible research papers using accessible full texts and documented venues. Do not count survey/older/background documents as substitutions.
2. Archive Q1 database, category, ranking year, quartile, URL/snapshot and retrieval date; apply one course-approved convention consistently. CORE2023 A is documented for MICCAI; journal quartiles are not inferred from publisher reputation.
3. Re-synthesize the critical summary and candidate table after the corpus is complete. Reassess the gaps against the expanded literature; the present scoped opportunities are not a systematic novelty proof.

## Validation

The built-in editor compiler returns the platform error 'Unable to find standard directories for platform'. The existing local Tectonic 0.17.0 executable successfully compiles the revised source. Citation keys and braces were checked; the source contains three Main Research Gaps and four Supporting Research Gaps. PDF pagination and rendered pages were inspected. The editable source and its pre-QA backup are preserved.
