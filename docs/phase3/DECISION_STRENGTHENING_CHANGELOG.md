# Decision evaluation strengthening — 16 November 2026

The submitted Phase 2 remains authoritative and unchanged. No instructor feedback was received. This work is an outcome-informed extension of completed synthetic results.

## Implemented

- Exactly equal two/four-cohort label budgets across uniform, input, concern, combined and released-label audit policies, at delays 0/5/20; predecision label timing and explicit quota tail filling.
- Thresholds from all five original development seeds; final 30 seeds remain separate. `decision_strengthening_v1` used only the first development seed and is superseded by `decision_strengthening_v2`, retained intact.
- Paired 2,000-replicate seed bootstrap contrasts and complete per-seed/condition/decision CSVs. This measures audit allocation, not classifier adaptation, human benefit or episode-level utility.
- Consolidation of existing changed-marginal localization evidence. Known feature targets only for C1/C2. No new claim of causal or dependence localization.
- Exploratory persistence forecasts at h=0/1/5, with issue and target clocks and four comparators. These do not establish a general future-loss predictor or forecast intervals.
- Correct external historical source/reference class-count denominators in separate metadata; old monitoring results unchanged.
- Revised evidence view clearly marks unresolved source-calibrated estimates and horizon-zero scope, retaining release filters.
- Consolidated editable LaTeX/PDF report with new plots and preserved model/task and adaptation evidence. Large-seed findings are complete and reported in the v5 manuscript; the 1,500-seed study is outcome-informed robustness replication, not external validation.

## Validation performed

- 81 whole-workspace tests passed in 10.56 seconds following the decision-policy changes, including three new information-timing/quota checks.
- Revised evidence view: 75,600 pure replay-logic release-boundary checks passed. This is not a browser usability study or full browser layout check.
- Report compiled using Tectonic 0.17.0. Rendered page contact sheets inspected; no overfull boxes or missing glyph warnings in corrected edition. Minor underfull paragraph/table spacing remains.
- Clean extracted scientific reruns and exact CSV comparisons are recorded in the supplementary archive's adjacent receipt, if completed. Do not infer success from this plan sentence.

## Unresolved limitations

Calibration transfer failure is exposed and communicated, not solved for arbitrary concept shift. Human evaluation and causal attribution remain deferred. Label-budget policy gains can depend on quota filling and the location/persistence of synthetic drift. All new comparisons were designed with knowledge of earlier outcomes. The external study is one dataset. The completed 1,500-seed study has its own lock and continuation record. Its failures under conditional-only change, weak C6 alarm sensitivity, and stationary XOR logistic underfit remain limitations.

## Reproduce

From workspace root with Anaconda scientific Python, use fresh directories:

```powershell
& C:/Users/hp/anaconda3/python.exe scripts/strengthen_phase3_decisions.py --out results/audit_NEW
& C:/Users/hp/anaconda3/python.exe scripts/evaluate_phase3_horizon_extension.py --out results/horizon_NEW
& C:/Users/hp/anaconda3/python.exe -m pytest tests -q
& ./tmp/tectonic/tectonic.exe reports/phase3/decision_strengthening_v5/REPORT.tex --outdir reports/phase3/decision_strengthening_v3 --keep-logs
```


## Large-seed completion and final report edition — 16 November 2026

- Completed the locked 1,500-seed-per-task robustness replication: 36,000 model streams, 864,000 online batch records and 864,000 evaluator records; no failed seeds. The seed bootstrap uses 2,000 whole-seed replicates.
- Reproduced a predeclared five-seed sample per task. Discrete fields match exactly; floating fields match at rtol=1e-12 / atol=1e-14 (maximum absolute difference 1.11e-16). The initial bitwise comparison failure remains recorded; it was numerical roundoff, not a changed scientific result.
- Consolidated the findings into `reports/phase3/decision_strengthening_v5/REPORT.tex` and PDF. The 14-page PDF was rendered with pypdfium2 and visually reviewed page by page: zero overfull boxes, zero missing characters, and 42 underfull spacing warnings. Detailed estimation and interval-coverage plots remain in the standalone robustness report.
- Full workspace validation: 81 tests passed in 19.67 seconds. The v2 supplementary archive passed CRC/member hashes, 5 clean-extraction tests, fresh audit and horizon reruns, and exact matches for all 10 generated CSVs. The versioned supplementary archive v4 includes this report and changelog and passed CRC and member-hash checks, clean-extraction reruns, and exact comparison of all 10 generated result CSVs. Its adjacent receipt is the release validation record.
- A first archive validation attempt failed because the clean extraction omitted the external UCI protocol/results input. The corrected v2 archive includes the evaluator, protocol results and dataset source; it passes clean extraction. The incomplete first archive is retained as a clearly named failed artifact, not a release candidate.
- Scope remains the frozen IDM-Viz project: this robustness run does not add real-domain data, resolve external calibration transfer, establish human benefit, or validate a future-risk predictor.

- Fixed an orphan heading in the standalone robustness PDF and shortened its five-seed sample-check note; underlying metrics and CSVs are unchanged. Re-rendered and visually checked the new page count.

## Additional external-domain and prospective-prediction extensions — 16 November 2026

- Added an ELEC2 temporal stream experiment with documented date-order ambiguity, fixed feature exclusions, delayed-label replay, and evaluator-only retrospective metrics. This broadens the pipeline exercise to a second external stream but does not establish drift recall because the stream has no known onset labels.
- Added a released-label-only isotonic recalibration comparison on ELEC2. It moved signed bias closer to zero but did not lower batch MAE for delays 0/5/20 and did not consistently improve plug-in interval coverage. Retain this as a negative mitigation result.
- Added a matched-budget Platt-style logistic correctness-calibration comparator using the exact same historical and strictly released monitoring labels as cumulative isotonic. Frozen isotonic remained best on equal-batch MAE and ECE at all delays; both updates reduced bias without fixing calibration transfer. Full outcomes and the report are in `ELEC2_CALIBRATION_FAMILY_EXTENSION_20261116.md` and `reports/phase3/elec2_calibration_family_20261116_v1/`.
- Added a matched-budget Platt-style logistic correctness-calibration comparator using the exact same historical and strictly released monitoring labels as cumulative isotonic. Frozen isotonic remained best on equal-batch MAE and ECE at all delays; both updates reduced bias without fixing calibration transfer. Full outcomes and the report are in `ELEC2_CALIBRATION_FAMILY_EXTENSION_20261116.md` and `reports/phase3/elec2_calibration_family_20261116_v1/`.
- Added a predeclared synthetic prospective forecast of five-batch future harm with disjoint development and held-out seeds, three predictor sets, strict delayed-label availability, and seed-cluster bootstrap. On 11,970 held-out forecasts, Brier was 0.231 time-only, 0.166 with unlabeled monitor evidence, and 0.149 with available released labels; 30-seed paired bootstrap favored both evidence additions. Its editable TeX, figure, protocol and outputs are in `FUTURE_HARM_FORECAST_EXTENSION_20261116.md` and `reports/phase3/future_harm_forecast_20261116_v10/`. The result is limited to the known synthetic laws and does not support human benefit or broad transfer.
- Package v11 (`output/phase3/IDM_Viz_Phase3_Final_Strengthening_20261116_v11.zip`) passed CRC, member hashes, clean extraction, 18 extracted tests, fresh evaluators, and exact CSV comparisons including a full forecast rerun. SHA-256: `54feda54c33588795be35f54d15d44e7d9d9336da7907f69fdd30bad6c9e3a9f`. Report-builder usability was improved afterward: v12 added an output path and v13 adds explicit run selection plus clean-extraction report regeneration checks. The v10 archive remains preserved as a failed validation attempt: its clean-extraction forecast exceeded the 20-minute timeout.
- Main package v13 (`output/phase3/IDM_Viz_Phase3_Final_Strengthening_20261116_v13.zip`) passed CRC, member SHA-256, clean extraction, 18 focused tests, future-harm runner/evaluator rerun, exact forecast CSV comparisons, and report TeX/figure regeneration comparisons. SHA-256: `4010ca6baf3cd67ed7516e16fcffff31558c97e17cfc92409f08ef25e0d33c2f`.
- Calibration addendum v2 (`output/phase3/IDM_Viz_ELEC2_Calibration_Family_Addendum_20261116_v2.zip`) passed CRC/member hashes, clean extraction, two focused tests, fresh Platt/evaluator/report commands, and exact comparisons for fit audits, per-batch/summary results, TeX and figure pixels. SHA-256: `e9a48f4708fdd932f9e5243a5ed941e1e6b60f0e98bdc47737aea6e47bcfb1fb`. Its preservation manifest references main package v13 without embedding a duplicate 401 MB archive.
- Added a held-out conditional-randomization law for prospective future-harm prediction. The first synchronized-onset run exposed a time-only AUC of 0.998; the balanced-onset follow-up (batches 5/9/13/17) yielded time/unlabeled/released-label Brier 0.238/0.188/0.194, with no clear released-label gain. Both the confounded pilot and the corrective follow-up are retained and documented, not conflated.
- Main package v13 was independently clean-extraction validated, and the matched-budget calibration addendum v2 passed clean extraction and exact result/report checks. The held-out future-harm addendum v1 also passed clean extraction, invariant tests, exact pilot/follow-up evaluator reruns, and exact balanced-onset TeX/figure regeneration. Its SHA-256 is `285977953af227e924aea91b95a8953eae225c58d6b8241ccc9c43fafd296a50`.
- Main package v13 (`output/phase3/IDM_Viz_Phase3_Final_Strengthening_20261116_v13.zip`) passed CRC, member SHA-256, clean extraction, 18 focused tests, future-harm runner/evaluator rerun, exact forecast CSV comparisons, and report TeX/figure regeneration comparisons. SHA-256: `4010ca6baf3cd67ed7516e16fcffff31558c97e17cfc92409f08ef25e0d33c2f`.
- Calibration addendum v2 (`output/phase3/IDM_Viz_ELEC2_Calibration_Family_Addendum_20261116_v2.zip`) passed CRC/member hashes, clean extraction, two focused tests, fresh Platt/evaluator/report commands, and exact comparisons for fit audits, per-batch/summary results, TeX and figure pixels. SHA-256: `e9a48f4708fdd932f9e5243a5ed941e1e6b60f0e98bdc47737aea6e47bcfb1fb`. Its preservation manifest references main package v13 without embedding a duplicate 401 MB archive.
# UCI external replay and metadata correction - 16 November 2026

- Followed the Day 3 external-improvement report's unresolved retrospective-label limitation with a new, separately versioned UCI 224 cohort-clock replay. The frozen classifier, predictions, confidence and initial calibration remain unchanged. Only the correctness calibrator is refit on historically available and strictly released outcomes; the rule is source batch + delay < forecast batch.
- Evaluated delays 0/1/2 cohorts against frozen calibration, raw Average Confidence and batch-1 reference accuracy over batches 2-10. Released-label logistic equal-cohort MAE was 0.169/0.227/0.283; frozen calibration was 0.545, raw confidence 0.411, reference baseline 0.512. These values are descriptive for one ordered sequence; online updates do not eliminate positive bias, and performance worsens with delay.
- Fixed the previously ambiguous batch-1 metadata in the current runner. Full source cohort counts/proportions and reference-partition evaluation counts/proportions now have explicit separate fields and denominators. The new output `results/phase3/external_uci224_day2_metadata_v3/` reproduces the prior accuracy, macro-F1, estimates, intervals and input alarms exactly.
- Added protocol, runner, evaluator, regression tests, report builder, three-page TeX/PDF addendum and vector/raster figures. Focused tests: 5 passed; full workspace suite: 107 passed. PDF was rendered and visually inspected page by page; the TeX log has no overfull/underfull box or undefined-reference warnings. The online score artifacts contain no truth columns; evaluator truth is in a separate directory.
- Limitations: batch order is coarse, release clocks are simulated, cohort sizes vary, only one external sequence is studied, recalibration results are outcome-informed and no inferential intervals are claimed. No causal, safety, human-benefit or broad transfer claim follows.
