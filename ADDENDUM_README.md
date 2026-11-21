# IDM-Viz Phase 3 post-deadline strengthening addendum

This is an additive update layered over `IDM_Viz_Phase3_Final_Strengthening_20261116_v14.zip`. It does not replace the original Phase 2 submission, the pre-deadline ZIP, or the v14 improvement package. The Phase 2 DOCX and implementation brief are included unchanged; their SHA-256 values are in `docs/phase3/PHASE3_IMPLEMENTATION_SPEC_AUDIT_20261117.json`.

The addendum adds the NOAA GSOD one-station same-day classification exercise, audited 3/5/10 cluster-region and dependence-pair localization, verified v2 Elec2/Covertype processed tables, tests, TeX source and figures. It does not claim forecasting, broad transfer, causal attribution, analyst benefit or human-subject evidence. Read the two 17 November audit reports for complete result limits.

## Environment and clean-extraction verification

The scripts use the project Anaconda environment at `C:\Users\hp\anaconda3\python.exe`, pandas, numpy, scikit-learn, scipy, pyarrow, matplotlib, pytest, and Tectonic. Tectonic is required to compile the NOAA reproduction report.

From the extracted package root in PowerShell:

```powershell
C:\Users\hp\anaconda3\python.exe scripts\rebuild_processed_datasets_v2.py
C:\Users\hp\anaconda3\python.exe scripts\audit_datasets.py --output data\interim\audit_all_datasets_20261117.json
C:\Users\hp\anaconda3\python.exe -m pytest -q
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\reproduce_noaa_weather.ps1
C:\Users\hp\anaconda3\python.exe scripts\run_localization_audit.py --output results\phase3\localization_audit_20261117_cleancheck
C:\Users\hp\anaconda3\python.exe scripts\summarize_localization_audit.py --run results\phase3\localization_audit_20261117_cleancheck --output reports\phase3\localization_audit_20261117_cleancheck
```

The first build command verifies included v2 files by checksum and read-back without overwriting them. The localization commands use fresh, isolated output names. All downloads used for the NOAA and UCI exercises are included with source manifests. The NOAA run is same-day retrospective classification and is not a forecast.

## Known limitations

The five-region cluster audit found weak region recovery. The dependence-pair test is one synthetic positive control. Research Question 3 is unanswered and no human study was conducted. Named-literature baseline fidelity, natural-stream calibration under shift, and some matched adaptation and uncertainty studies remain open. No instructor feedback was received.
