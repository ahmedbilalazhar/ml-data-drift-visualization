# IDM-Viz Phase 3 strengthening supplement — 17 November 2026

This versioned report builds on the submitted Phase 2 methodology and earlier Phase 3 results. It does not replace the submitted methodology or earlier reports. The Phase 2 DOCX remains unchanged (SHA-256 786d4f23d54d0d09edd5f94740198f43ca3b406eb38626ae754fe04fe26c5ef4). No instructor feedback is attributed.

## Deliverables

- REPORT.tex: editable, generated TeX report source.
- REPORT.pdf: compiled 10-page report.
- 	ables.tex and igures/: generated numerical tables and five figures.
- 
eport_build_receipt.json: input run, source and figure checksums.
- pdf_build_receipt.json: PDF build and rendered-page checksums.
- qa_render/page-01.png … page-10.png: rendered pages used for visual inspection.

## Evidence included

The primary eight-feature evaluation completed 720/720 task–mechanism–seed cells. The separate dimensionality sensitivity completed 1,440/1,440 cells across 32 and 128 features. Each run has a locked protocol and checksum-verified cell artifacts; analyses include 14,400 and 28,800 seed rows respectively. The sensitivity results are explicitly descriptive: the generator and signal-to-noise ratio change with dimensionality, so this is not an isolated causal dimensionality test. Isotonic calibration does not uniformly beat its matched internal comparators.

## Reproduction

Run from the project root in the configured Anaconda environment, using new output directories for new runs:

`powershell
python scripts/run_strengthening_v3.py --profile evaluation 
  --output results/phase3/reproduction_evaluation_d8
python scripts/analyze_strengthening_v3.py 
  results/phase3/reproduction_evaluation_d8
python scripts/build_strengthening_report.py 
  results/phase3/reproduction_evaluation_d8 
  --output reports/phase3/reproduction_evaluation_d8
`

For the sensitivity extension, add --dimension 32 --dimension 128 to the runner and use a distinct output directory. Analyze that run, then build the combined report with:

`powershell
python scripts/build_strengthening_report.py 
  results/phase3/strengthening_v3_evaluation_d8 
  --sensitivity-analysis results/phase3/strengthening_v3_evaluation_sensitivity_d32_d128/analysis 
  --output reports/phase3/reproduction_evaluation_with_sensitivity
`

Compile the generated report with 	ectonic reports/phase3/reproduction_evaluation_with_sensitivity/REPORT.tex. Resume only an interrupted run using the same output directory and locked source snapshot.

## Interpretation and limitations

RQ3's decision-effect question remains open; the coordinated view is a prototype, and no claim of improved analyst decisions is made. There is no human-subject completion requirement. The results remain conditional on synthetic laws and the separately reported external-data exercises. They do not establish causal attribution, broad natural-domain transfer, a calibrated population-risk guarantee, or production safety. See docs/phase3/PHASE3_REMAINING_WEAKNESSES_20261116.md for the active unresolved-weakness register.
