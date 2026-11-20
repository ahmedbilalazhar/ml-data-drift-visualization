# Day 2 external evaluation

This is a separately versioned extension. The UCI source ZIP, admission audit, locked protocol, runner, frozen results, figures, editable report and PDF are included in the Day 2 archive. The Day 1 validated package is nested intact. The first Day 2 execution with incorrect split allocation remains in `results/phase3/external_uci224_day2_v1/` as superseded evidence; the corrected run is `external_uci224_day2_v2/`. The clean-extracted final package is versioned `IDM_Viz_Phase3_Day2_External_Evaluation_v1.3_Validated.zip`.

Run in the project root:

```powershell
& C:/Users/hp/anaconda3/python.exe scripts/evaluate_uci224_day2.py
& C:/Users/hp/anaconda3/python.exe scripts/analyze_uci224_day2_sensitivity.py
& C:/Users/hp/anaconda3/python.exe scripts/build_uci224_day2_figures.py
tectonic docs/phase3/research_improvement_day2/REPORT.tex --outdir docs/phase3/research_improvement_day2 --keep-logs
```

The sensitivity analysis is outcome-informed and diagnostic. Do not present it as part of the locked primary protocol. See the source page for terms and provenance; the dataset description says research use only.
