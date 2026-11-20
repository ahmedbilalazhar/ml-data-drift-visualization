# IDM-Viz Phase 3 reproduction

Run in `C:\Users\hp\Desktop\University\2. DAV\3. Project` using PowerShell.
The authoritative implementation is `src/drift_monitoring/phase3/`. The submitted
Phase 2 DOCX, existing plan and manifest are preserved. Old E0 results, legacy action
scripts and their dashboard are not Phase 3 evidence. Source hashes in each
`protocol_lock.json` identify the numerical implementation and tests at execution.

## One command

```powershell
Set-Location 'C:\Users\hp\Desktop\University\2. DAV\3. Project'
& ./scripts/reproduce_phase3.ps1 -RunDir results/phase3/my_reproduction -ReportDir reports/phase3/my_reproduction
```

Existing output directories are refused. Choose a new name for every independent
run. The script uses the existing Anaconda environment for science and the bundled
artifact environment for the PDF. No environment installation is needed on this
laptop. On another machine, pass `-ScientificPython` and `-ArtifactPython` to Python
environments with the listed dependencies. `requirements-phase3.txt` records exact
scientific package versions. PDF authoring also needs reportlab, Pillow and
pypdfium2; the bundle's versions are recorded with the PDF validation evidence.

## Individual commands

```powershell
& C:/Users/hp/anaconda3/python.exe -m pytest tests -q
& C:/Users/hp/anaconda3/python.exe scripts/run_phase3.py --mode final --out results/phase3/my_reproduction
& C:/Users/hp/anaconda3/python.exe scripts/evaluate_phase3.py --run results/phase3/my_reproduction
& C:/Users/hp/anaconda3/python.exe scripts/report_phase3.py --run results/phase3/my_reproduction --out reports/phase3/my_reproduction
& C:/Users/hp/anaconda3/python.exe scripts/build_phase3_view.py --run results/phase3/my_reproduction --out reports/phase3/my_reproduction/evidence_view.html
& 'C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1112-737-0_26.1112.737.0_x64__3k8sg7r9htsxt/dependencies/python/python.exe' scripts/render_phase3_pdf.py reports/phase3/my_reproduction/IDM_Viz_Phase3_Report.md
& C:/Users/hp/anaconda3/python.exe scripts/compare_phase3_runs.py --first results/phase3/final_v1 --second results/phase3/my_reproduction --out reports/phase3/my_reproduction/comparison.json
```

For a small development check (not final evidence):

```powershell
& C:/Users/hp/anaconda3/python.exe scripts/run_phase3.py --mode development --seeds 90001 --out results/phase3/my_development
```

The final command always uses seeds 100-129. The runner refuses final seed
overrides and never silently drops a failed fit. It catches and records KS warnings,
then computes all matched ablations from the same cached probabilities.

## Data dictionary and evidence semantics

| File | Meaning |
| --- | --- |
| `protocol_lock.json` | Original manifest, hashes, seeds, clock and status; only complete runs are evaluated |
| `environment.json` | Actual interpreter/library versions and process memory |
| `online_batches.csv` | 5,040 feature-only batch records; no current truth, F1 or realised accuracy |
| `predictions/*_online.csv` | Per-stream online records persisted before release replay |
| `predictions/*.npz` | Exact instance IDs, predictions, raw/calibrated confidence, PCA/cluster arrays and two PMFs |
| `evaluator_truth/*.npz` | Explicitly evaluator-only labels and IDs; never passed to scoring |
| `clock_events.csv` | Phase 0 prediction and phase 1 verification; drainage to 24+d |
| `released_verification.csv` | 15,120 release records with availability clock, truth and ID hash |
| `evaluator_batches.csv` | Retrospective one-to-one online/truth join, delay zero only, not live evidence |
| `impact_per_seed.csv` | Condition/regime/seed/method MAE, coverage, width, counts, flags and AUROC |
| `impact_summary.csv` | Seed-bootstrap mean/interval and number of defined seeds per metric |
| `harm_counts.csv` | Pooled TP/FP/FN/TN, denominators and precision/recall |
| `detection_per_seed.csv` | Four-batch onset matching, misses, duplicate alert burden, native task only |
| `localization_batches.csv` | Changed marginal truth, precision@1, recall@3 and false flags |
| `paired_seed_differences.csv` | Matched MAE differences before resampling |
| `paired_bootstrap.csv` | 2,000-replicate paired seed intervals |
| `calibration_error_per_seed.csv` | Evaluator-only 20-bin adaptive correctness-calibration error |
| `delay_availability.csv` | Verified/unresolved counts at last input and complete drainage |
| `runtime_per_seed.csv` | Measured component times, distinct from runner total |
| `validation.json` | Actual end-to-end join, count and timing validation result |

Batch numbers start at 1; input onset is batch 9. C2/C4 transition through batch
12; C5 returns at batch 17. The same frozen model and predictions are used for all
delays. An absent flag is not a safety certificate. Accuracy interval coverage is
empirical; the interval and concern score are plug-in model quantities. Conditional
change deliberately violates the source-calibration assumption. Undefined metrics
are blank in CSVs and displayed as undefined in the report, never replaced by zero.
The ATC-style point harm rule and constant point harm rule are disclosed extensions
for comparison against the common realised-harm target, not calibrated probabilities.

The standalone HTML opens directly in a browser without a server. It contains
retrospective release data but filters displayed truth by its `available_at` clock.
Changing clock, delay or cohort must update every panel consistently. It is not a
security boundary and does not establish improved analyst decisions.

## Preservation and packaging

No upload is performed or claimed. The versioned archive is a prepared submission
snapshot. Its checksum inventory distinguishes it from later improvements. After
5 pm on 15 November 2026 Asia/Karachi, preserve this snapshot and create `v2` or a
new version for improvements. Do not overwrite the Phase 2 DOCX or requirements PDF.
The compact submission archive omits regenerable instance caches; the full
reproducibility archive includes them. Both include source, full numerical CSVs,
the report, four figures, interactive replay and actual validation records.
