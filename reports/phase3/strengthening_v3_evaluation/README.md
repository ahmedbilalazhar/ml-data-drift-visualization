# IDM-Viz Phase 3 strengthening report

This versioned supplement reports the completed, checksum-verified eight-feature evaluation. It builds on the submitted Phase 2 design; it does not replace that submission or earlier project reports. The Phase 2 DOCX is preserved with SHA-256 `786d4f23d54d0d09edd5f94740198f43ca3b406eb38626ae754fe04fe26c5ef4`.

## Files

- `REPORT.tex`: editable report source.
- `REPORT.pdf`: compiled report.
- `tables.tex`: generated result tables.
- `figures/`: four publication-style result plots.
- `report_build_receipt.json`: run, source, figure, and TeX checksums.
- `../STRENGTHENING_V3_PROTOCOL_20261116.md`: protocol details and claim boundaries.
- `../../../results/phase3/strengthening_v3_evaluation_d8/`: locked 720-cell run and seed-level CSV analysis.

The primary run completed all 720 task–mechanism–seed cells at eight features. The report's estimator comparisons show that isotonic calibration did not uniformly outperform its baselines; paired differences and Holm-adjusted stratum counts are included.

## Reproduce the primary report

Run from the project root in the Anaconda environment:

```powershell
python scripts/run_strengthening_v3.py --profile evaluation `
  --output results/phase3/reproduction_evaluation_d8
python scripts/analyze_strengthening_v3.py `
  results/phase3/reproduction_evaluation_d8
python scripts/build_strengthening_report.py `
  results/phase3/reproduction_evaluation_d8 `
  --output reports/phase3/reproduction_evaluation_d8
tectonic reports/phase3/reproduction_evaluation_d8/REPORT.tex
```

Use a new output directory for a fresh run. Resume an interrupted run only with its original lock and source snapshot.

## Optional dimensionality sensitivity

The separate 32/128-feature evaluation has completed 1,440/1,440 cells at 30 seeds per dimension. Its checksummed analysis and combined report are in `results/phase3/strengthening_v3_evaluation_sensitivity_d32_d128/analysis/` and `reports/phase3/strengthening_v3_evaluation_with_sensitivity_20261117/`. The commands below reproduce that sensitivity run and appendix in a new output directory:

```powershell
python scripts/run_strengthening_v3.py --profile evaluation `
  --dimension 32 --dimension 128 `
  --output results/phase3/strengthening_v3_evaluation_sensitivity_d32_d128
python scripts/analyze_strengthening_v3.py `
  results/phase3/strengthening_v3_evaluation_sensitivity_d32_d128
python scripts/build_strengthening_report.py `
  results/phase3/strengthening_v3_evaluation_d8 `
  --sensitivity-analysis results/phase3/strengthening_v3_evaluation_sensitivity_d32_d128/analysis `
  --output reports/phase3/strengthening_v3_evaluation_with_sensitivity
```

The reported sensitivity appendix is based on the completed run lock and checksum-gated analysis. The findings are descriptive because changing dimension also changes the synthetic feature space and signal-to-noise ratio.

## Scope

RQ3's decision-effect question remains open; the coordinated view is reported as a prototype, with no claim about improved analyst decisions. No instructor feedback is attributed. Results are conditional on the declared synthetic laws and do not establish causal attribution, general natural-domain transfer, a calibrated population-risk guarantee, or production safety.
