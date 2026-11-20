# Phase 3 Reproduction Instructions

**Ahmed Bilal (23I-2581) and Umar Behram (23I-2604)**  
Data Analysis and Visualization

## Environment

The validated scientific environment is `C:\Users\hp\anaconda3\python.exe`. Exact interpreter and package versions are recorded in `environment.json` for each run. Numerical threads are limited to one. No dataset download is needed; both tasks are generated locally.

## Complete reproduction

From the workspace root, choose new, unused output directories and run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\reproduce_phase3_v2.ps1 -RunDir results/phase3/v2_reproduction -ReportDir reports/phase3/v2_reproduction
```

The wrapper runs all locked final seeds, evaluates data contracts and metrics, builds the offline evidence replay, creates the Markdown report and PDF, and compares the reproduction with the authoritative v2 run. Existing run directories are refused.

## Individual commands

```powershell
& C:/Users/hp/anaconda3/python.exe scripts/run_phase3_v2.py --mode final --out results/phase3/v2_reproduction
& C:/Users/hp/anaconda3/python.exe scripts/evaluate_phase3_v2.py --run results/phase3/v2_reproduction
& C:/Users/hp/anaconda3/python.exe scripts/build_phase3_v2_view.py --run results/phase3/v2_reproduction --out reports/phase3/v2_reproduction/23I-2581_23I-2604_Phase3_Evidence_Replay.html
& C:/Users/hp/anaconda3/python.exe scripts/report_phase3_v2.py --run results/phase3/v2_reproduction --out reports/phase3/v2_reproduction
& C:/Users/hp/anaconda3/python.exe scripts/compare_phase3_v2.py --first results/phase3/v2_final --second results/phase3/v2_reproduction --out reports/phase3/v2_reproduction/reproduction_comparison.json
```

Render the PDF with the installed artifact runtime:

```powershell
& 'C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1112-737-0_26.1112.737.0_x64__3k8sg7r9htsxt/dependencies/python/python.exe' scripts/render_phase3_v2_pdf.py reports/phase3/v2_reproduction/23I-2581_23I-2604_Phase3_Report.md
```

Development-only smoke run:

```powershell
& C:/Users/hp/anaconda3/python.exe scripts/run_phase3_v2.py --mode development --out results/phase3/v2_smoke
& C:/Users/hp/anaconda3/python.exe scripts/evaluate_phase3_v2.py --run results/phase3/v2_smoke
```

Development seeds never enter the final result tables. Do not use a final run's existing directory as a new output path.

## Evidence separation

`online_batches.csv` contains predictions and feature-only monitoring evidence without current labels, accuracy, F1 or harm. `released_verification.csv` contains outcomes at their scheduled availability clocks. `evaluator_truth/` and `evaluator_batches.csv` are retrospective evaluation artifacts. Prediction and label IDs are joined exactly; delays 0, 5 and 20 use the same saved predictions and drain through clock 24+d.

The compact submission archive includes the report, code, result tables, figures, replay and validation. The separate full reproducibility archive also includes compressed instance-level caches. Neither archive is an upload receipt.

## 16 November strengthening extensions

Editable TeX sources and vector figures are provided in the report directories linked by `PHASE2_TRACEABILITY.md`. The final future-harm supplement is `reports/phase3/future_harm_forecast_20261116_v10/FUTURE_HARM_FORECAST.tex`; its evaluated outcome, split, and limits are recorded in `FUTURE_HARM_FORECAST_EXTENSION_20261116.md`.

From a clean workspace with the stated Anaconda environment:

```powershell
& C:/Users/hp/anaconda3/python.exe scripts/run_elec2_temporal_validation.py --out results/elec2_NEW
& C:/Users/hp/anaconda3/python.exe scripts/evaluate_elec2_temporal_validation.py --run results/elec2_NEW
& C:/Users/hp/anaconda3/python.exe scripts/run_elec2_delayed_recalibration.py --run results/elec2_NEW --out results/recal_NEW
& C:/Users/hp/anaconda3/python.exe scripts/evaluate_elec2_delayed_recalibration.py --run results/elec2_NEW --recalibration results/recal_NEW
& C:/Users/hp/anaconda3/python.exe scripts/run_future_harm_forecast.py --out results/forecast_NEW
& C:/Users/hp/anaconda3/python.exe scripts/evaluate_future_harm_forecast.py results/forecast_NEW
& C:/Users/hp/anaconda3/python.exe scripts/report_future_harm_forecast.py --run results/forecast_NEW --out reports/forecast_NEW
& C:/Users/hp/anaconda3/python.exe scripts/run_elec2_platt_recalibration.py --run results/phase3/elec2_temporal_20261116_v2 --out results/phase3/elec2_platt_recalibration_NEW
& C:/Users/hp/anaconda3/python.exe scripts/evaluate_elec2_platt_recalibration.py --run results/phase3/elec2_temporal_20261116_v2 --platt results/phase3/elec2_platt_recalibration_NEW --isotonic results/phase3/elec2_recalibration_20261116_v1 --out results/phase3/elec2_calibration_family_NEW
& C:/Users/hp/anaconda3/python.exe scripts/report_elec2_calibration_family.py --run results/phase3/elec2_temporal_20261116_v2 --platt results/phase3/elec2_platt_recalibration_NEW --isotonic results/phase3/elec2_recalibration_20261116_v1 --results results/phase3/elec2_calibration_family_NEW --out reports/phase3/elec2_calibration_family_NEW
& C:/Users/hp/anaconda3/python.exe scripts/run_future_harm_unseen_family.py --onset-policy balanced_onsets --out results/phase3/future_harm_unseen_family_NEW
& C:/Users/hp/anaconda3/python.exe scripts/evaluate_future_harm_unseen_family.py --run results/phase3/future_harm_unseen_family_NEW
& C:/Users/hp/anaconda3/python.exe scripts/report_future_harm_unseen_family.py --run results/phase3/future_harm_unseen_family_NEW --out reports/phase3/future_harm_unseen_family_NEW
& C:/Users/hp/anaconda3/python.exe -m pytest tests/phase3 -q
```

## UCI external metadata and delayed-label follow-up

The following commands produce the corrected denominator metadata and replay released-label recalibration on the cached, frozen batch-1 external study. The report addendum is editable TeX and includes PDF/SVG/PNG visualizations.

```powershell
& C:/Users/hp/anaconda3/python.exe -m pytest tests/phase3/test_uci224_delayed_replay.py tests/phase3/test_uci224_metadata_semantics.py -q
& C:/Users/hp/anaconda3/python.exe scripts/evaluate_uci224_day2.py --out results/phase3/external_uci224_day2_metadata_NEW
& C:/Users/hp/anaconda3/python.exe scripts/run_uci224_delayed_replay.py --source-run results/phase3/external_uci224_day2_v2 --out results/phase3/uci224_delayed_replay_NEW
& C:/Users/hp/anaconda3/python.exe scripts/evaluate_uci224_delayed_replay.py --run results/phase3/uci224_delayed_replay_NEW
& C:/Users/hp/anaconda3/python.exe scripts/report_uci224_delayed_replay.py --run results/phase3/uci224_delayed_replay_NEW --out reports/phase3/uci224_delayed_replay_NEW
tectonic reports/phase3/uci224_delayed_replay_NEW/REPORT.tex --outdir reports/phase3/uci224_delayed_replay_NEW --keep-logs
```

Do not reuse the recorded `*_v2`/`*_v3` output paths. The actual evaluated run is `results/phase3/uci224_delayed_replay_20261116_v2`; report source/PDF and figures are in `reports/phase3/uci224_delayed_replay_20261116_v3`. The previous Day 2/Day 3 records remain preserved.

The forecast run took 416 seconds in the working checkout; a clean-extraction rerun took 664.7 seconds. It uses only known synthetic laws and should not be represented as general or natural-domain validation. The report builder accepts both `--run` and `--out` so it can target a fresh experiment folder. Package v13 records clean-extraction experiment and report regeneration checks. Package v11 passed clean extraction before the report-builder output-path fix; the earlier v10 archive is retained without a validation receipt because its 20-minute clean-run timeout expired.
