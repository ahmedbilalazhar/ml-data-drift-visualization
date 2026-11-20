# IDM-Viz: verified completion audit and next implementation plan

Prepared 15 November 2026, approximately 20:35 Asia/Karachi. This is a prospective implementation plan informed by the existing outcomes. New experiments below are outcome-informed extensions, not retroactively preregistered results. Budget: up to 12 further hours of local experiment execution; report and packaging work also require wall time. Check the actual remaining handoff time at execution start.

## 1. What is actually complete

| Requirement | Evidence found | Assessment |
|---|---|---|
| Submitted Phase 2 preserved | Fresh SHA-256 matches recorded DOCX hash | Verified unchanged |
| Submitted Phase 3 preserved | Fresh SHA-256 matches recorded submitted ZIP | Verified unchanged |
| Original controlled methodology | Frozen logistic, seven conditions, 30 seeds, feature/PCA evidence, calibration, intervals, delayed releases | Implemented; historical full reproduction receipts exist |
| Baselines and ablations | Original/v2 evaluation and historical reports | Implemented; consolidate their numerical comparisons in final report |
| Additional task/model | XOR and fixed tree study | Implemented, explicitly outcome-informed |
| Adaptation | Separate released-label cumulative-refit protocol and results | Implemented, selected scenarios and limited recovery |
| External evaluation | UCI 224, corrected 267/89/89 split from batch 1, monitoring batches 2–10 | Implemented; fresh rerun exactly reproduces three CSVs and 40 prediction arrays |
| External diagnosis | Class-mix and confidence-transfer diagnostics | Implemented exploratory analysis |
| Scientific tests | Fresh Phase 3 suite: 63 passed in 15.10 s; whole suite: 76 passed in 28.47 s | Passed this audit; old receipts reporting 35 are stale |
| LaTeX | Day 1, Day 2 and Day 3 editable sources and PDFs | Present, but short layered reports rather than a complete integrated manuscript |
| Visualization | Existing coordinated HTML view and result figures | Present; current browser behavior not freshly checked in this audit |
| Latest archive | Day3 v1.4, 24 entries, CRC passed | Structurally readable; not synchronized with current hardened scripts |

Fresh audit artifacts: `results/phase3/completion_audit_20261115/`.

Fresh external results: MAE 0.544657889410022; interval coverage 0/9; mean width 0.025802992731058707; input alarms 9/9. The negative result is evidence, not a reason to tune until performance improves.

### Material remaining gaps

1. No single consolidated LaTeX manuscript presents the original methodology, all essential baseline and ablation results, external evaluation, adaptation, related research comparison, and reproduction details. The latest Day 3 PDF is only three pages and focuses on the external extension.
2. Two current scripts differ from the latest hash inventory: `scripts/evaluate_uci224_day2.py` and `scripts/analyze_uci224_day2_sensitivity.py`. Diff inspection shows parser validation, CLI/output-path support, and overwrite protection additions. The current external primary results still reproduce exactly. Package a new version; never edit an old validation receipt to conceal this drift.
3. External study differs from the earlier suggested design: one historical batch rather than two, logistic calibration rather than isotonic, one model rather than two, approximate rather than exact Poisson-binomial intervals, and no delayed-release replay. These are recorded choices, but they do not fulfill every optional experiment in the prompt.
4. README commands omit fresh output arguments and encounter existing directories. The final package needs an executable reproduction entry point tested after extraction.
5. External batch-1 `cohort_metrics.csv` mixes full-batch class counts with reference-only evaluation size (89) when calculating class proportions. Monitoring metrics are unaffected, but historical metadata must distinguish source counts from evaluated counts. Fix in a new study version and test proportions sum to one within the named population.
6. Current external implementation computes retrospective metrics directly in its evaluation loop. It is a retrospective cohort study, not evidence of deployed delayed-label behavior. Extend through a real broker if making that claim.

## 2. Preservation and scope rules

- Read submitted Phase 2 and map each implemented component to its objectives. Keep the DOCX and every submitted archive byte-for-byte unchanged. Older TeX is supporting material only. No instructor feedback was received.
- Preserve all original, v2, adaptation, Day 1/2/3 outputs. Write new code/results under a new version such as `strengthening_v4`; refuse existing output directories.
- Preserve the original UCI primary protocol and negative result. All expanded comparisons below are a separately locked, outcome-informed extension.
- Do not add neural models, causal discovery, participant studies, or a new dataset to fill time.
- Use local Anaconda CPU execution. Limit numerical library threads initially to two per worker, use at most two heavy workers, and measure memory and throughput before increasing concurrency. Avoid nesting process pools and unrestricted BLAS threads.

## 3. Execution order and checkpoints

### Stage A — stabilize implementation and manifest (approximately 0–2 wall hours)

1. Recheck clock, hashes, available storage, interpreter and package versions. Record a source-file hash inventory and Git status without resetting anything.
2. Create a machine-readable extension manifest specifying model parameters, history windows, split seeds, calibration variants, estimator formulas, Holm families, metric denominators, delay ordering, and intended comparisons.
3. Add the corrected historical/evaluation class-count metadata. Explicitly store dataset class IDs and use `classes_[argmax(probabilities)]` throughout.
4. Extract reusable external scoring from retrospective evaluation. Cache label-free predictions, confidence, feature tests, class ordering and cohort IDs before any monitoring labels are released. Truth belongs in evaluator-only storage.
5. Add an end-to-end CLI accepting source, manifest and fresh output paths. Make the sensitivity stage consume the selected run rather than a hard-coded old directory. Reject mismatched IDs and cache hashes.
6. Meaningful checks: split exclusivity; training-only transformations; non-contiguous class IDs; finite features; variable cohort sizes; class-proportion denominators; future-label perturbation leaves all earlier scores unchanged; release-before/after phase ordering; package imports.

Checkpoint A: tests pass, manifest is locked, old hashes unchanged, one tiny multiclass replay works, and the report skeleton compiles. Do not launch the full matrix before this checkpoint.

### Stage B — independent reproduction of original evidence (allow 1–3 compute hours)

1. Rerun the original locked 30-seed/7-condition study into fresh directories, including evaluation. Compare all scientific CSVs and cached prediction/score arrays to original final results. Exclude timestamps and measured timings explicitly.
2. Rerun the locked v2 and adaptation studies without altering seeds, generators, metrics or hyperparameters. Reuse validated feature/prediction caches only where doing so preserves the protocol.
3. Store machine-readable comparisons, mismatch locations, maximum numeric differences, runtime and versions. Require exact equality in the same supported environment when the protocol promises it; investigate mismatches rather than selecting a tolerance after looking at them.
4. If a defect is found, preserve the earlier run and quantify which tables/conclusions change. Fix only the affected pipeline and rerun affected outputs.

Checkpoint B: distinguish freshly reproduced studies from historically documented ones. A failed reproduction is an unresolved finding, not a passing receipt.

### Stage C — stronger external evaluation (allow 2–5 compute hours, including diagnostics)

Use the already admitted UCI 224 source with recorded SHA-256. No additional download is required unless source verification fails. Do not invent within-batch timestamps or true drift onsets.

Keep the original batch-1 study as the preserved primary external result. Add this explicit extension matrix:

- Historical windows: batch 1 only; pooled batches 1–2. Evaluate batches 3–10 as the common comparison set. Show batch 2 separately for the batch-1-only configuration. Additional historical labels make the pooled configuration a different budget, not a fair direct performance victory.
- Historical partitions: stratified 60/20/20 with seeds 420001 through 420006. Use seed 420001 as the designated extension configuration and the others as split sensitivities. Log actual class counts and indices, including rounding.
- Models: logistic `C=1`, `max_iter=2000`; decision tree `max_depth=6`, `min_samples_leaf=25`, otherwise fixed defaults recorded in the manifest. Apply training-only standardization to logistic and PCA. Do not tune on monitored cohorts.
- Correctness estimators: historical reference accuracy; raw Average Confidence; original fixed logistic correctness map; isotonic correctness map; ATC using the existing implementation's documented threshold convention. Fit correctness maps on calibration only; fit ATC on reference only if that is the established convention. Record every estimator's data budget.
- Core matrix: 2 windows × 6 splits × 2 models = 24 fitted configurations. Cache classifier probabilities once. Estimator and delay variants reuse them. This matrix is a plan, not a claim that results already exist.
- For isotonic and logistic calibrated correctness, compute exact plug-in Poisson-binomial intervals using the validated kernel, and compare with the normal approximation at the same probabilities. Include raw-confidence intervals as a baseline where defined. At up to 3,613 rows per cohort, time a representative case first; use a validated convolution method if runtime is material. Never silently substitute approximation for exact computation.
- Concern score: apply the existing documented reference-relative harm boundary without retuning its threshold. Evaluate retrospectively against that model's own reference accuracy, and explain that reference uncertainty is omitted. A concern probability is a proxy, not certified risk.
- Ablations: original features only versus original features plus PCA; calibrated versus raw confidence; existing point-estimate concern rule versus interval/distribution concern rule with exactly declared formulas. Keep all unaffected components fixed. Report these as extension ablations separately from historical ones.

For every acquisition cohort and configuration, export accuracy, fixed-six-class macro-F1, per-class support/precision/recall/F1, confusion counts, signed estimation error, MAE, interval coverage/width, alarm counts, concern decisions, and runtime. Summarize equal-cohort and row-weighted metrics separately. Report estimator baselines quantitatively, not just as columns of raw inputs.

Uncertainty: use paired seed-level intervals for independent synthetic streams. For the external dataset, show paired per-cohort differences and descriptive variation across six historical splits. Do not bootstrap eight cohorts as independent replications or turn six overlapping splits into six independent datasets. Retain class-composition diagnosis with its absent-class limitations.

Checkpoint C: complete external results and provenance; all configurations accounted for; failed configurations explicit; no monitoring-label fitting; unchanged original external result.

### Stage D — label-safe external replay and evidence view (allow up to 1 compute hour plus implementation)

1. Replay cached external predictions for delays 0, 5 and 20 acquisition-cohort steps. Persist each forecast before release at that step, including delay zero. Drain pending labels after the final acquisition cohort.
2. Variable cohort size must not change cohort-delay accounting. At delay 20, no monitored outcome is available during the short acquisition sequence; the view must show pending verification rather than apparent online accuracy.
3. Keep retrospective accuracy, class diagnosis, interval coverage and harm outcomes outside online scoring. Add explicit as-of filtering to the view.
4. Recheck the existing original/v2 UI: selected cohort, feature panel, as-of clock, release boundaries, axis meanings and stale controls. Include one hidden-label conditional failure example and one external calibration failure example selected by a stated rule.
5. Record browser checks actually performed. Provide a five-minute demonstration guide with exact launch commands and observable expected behavior.

Checkpoint D: future-label perturbation tests and UI timing checks pass; external replay is clearly a simulation of release schedules, not observed natural label timestamps.

## 4. Compute control and useful long runs

Track cumulative experiment wall time in a ledger. Pilot one expensive configuration before launching the full matrix and estimate total cost. The 12-hour allowance is available for real work; do not add sleeping, redundant reruns, huge seed counts, or post-hoc tuning to make it long.

Priority allocation: original/v2/adaptation reproduction 1–3 h; external matrix and exact intervals 2–5 h; replay and invariants up to 1 h; clean-package reproduction and contingency 2–3 h. These are allowances, not measured predictions. Existing evidence suggests many kernels may finish sooner.

If projected cost exceeds 12 h, keep the two models and estimator comparisons, reduce extension split sensitivity from six to three predeclared seeds BEFORE running the reduced matrix, and record the reason. Do not reduce according to which results look best. Preserve time for manuscript and packaging. If all required runs finish early, spend remaining wall time on analysis and presentation.

Optional only after mandatory work: one fixed synthetic sample-size sensitivity using 250, 500 and 1,000-row batches, 30 fresh paired seeds 500001–500030, original frozen logistic and seven conditions, with onset and delay specified in both rows and batches. Do not start this if it displaces report/package acceptance; larger batches change monitoring resolution and cannot be called a simple accuracy improvement. Lock the complete protocol before running.

## 5. Consolidated LaTeX report and figures

Start the manuscript during Stage A. Deliver one editable, self-contained report with `.tex`, `.bib`, figures and compiled PDF, approximately 14–18 substantive main pages plus appendices as needed. Length is subordinate to complete evidence and readability. Use A4 single column, restrained teal/orange/slate, readable serif typography, booktabs tables, consistent axes and vector PDF/SVG plot assets. Do not use generated decorative images as scientific evidence.

Required narrative: submitted Phase 2 objectives and deviations; dataset/splits and chronological availability; algorithms and concern/interval definitions; all essential baselines; original and v2 synthetic results; component ablations; external negative result and extension; adaptation with label budgets; literature comparison and fairness limits; discussion of research questions; limitations and reproduction.

Required visual set: architecture/information clocks; seven-condition overview; coordinated evidence example; paired estimator MAE comparisons; coverage versus width; reliability/conditional failure; PCA/feature ablation; external cohort accuracy and estimates with class composition; adaptation effect versus label budget. Add external interval-method comparison if run. Every plotted number must trace to a versioned table.

Verify literature through primary sources when writing new claims. Compare numerical results only under compatible settings; otherwise use a methods/settings comparison with explicit fairness limitations. The current minimal bibliography is insufficient for the full related-research section.

Compile with the available pinned Tectonic installation, record its exact path/version, check references, and render/inspect every page. Fix undefined citations, clipping, unreadable labels and inconsistent figure sizing. State clearly that no instructor feedback or human evaluation exists.

## 6. Final packaging and acceptance

- Produce a new flat, understandable package with report, source, configuration, compact result tables, tests, reproduction entry point, environment versions and receipts. Put large caches in a separate full archive; avoid requiring users to recursively unpack multiple obsolete archives.
- Test the exact README commands from a fresh extracted directory, with no project-installed import fallback. Use fresh output paths. Package the CURRENT hardened scripts and regenerate hash inventories for the NEW package only.
- Compare clean-run scientific outputs with locked final outputs. Document allowable nondeterministic metadata separately. CRC and file presence alone do not constitute experimental reproduction.
- Recheck submitted DOCX SHA-256 `786d4f23d54d0d09edd5f94740198f43ca3b406eb38626ae754fe04fe26c5ef4` and submitted archive SHA-256 `279d9867d1c75720a51df12caae2ac0b4d490641b83752e32642dcb87e1c8877`.
- Update completion status to show exactly which checkpoints passed. End with paths, real commands, real test results, experiment runtime, substantive findings and remaining limitations. Do not mark the entire prompt complete just because Day 1/2/3 filenames exist.

## 7. Immediate verified reproduction commands

From the project root (choose a NEW output directory for each run):

```powershell
& C:/Users/hp/anaconda3/python.exe -m pytest tests -q
& C:/Users/hp/anaconda3/python.exe scripts/evaluate_uci224_day2.py --out results/phase3/external_recheck_NEW
& C:/Users/hp/anaconda3/python.exe scripts/analyze_uci224_day2_sensitivity.py --run results/phase3/external_recheck_NEW --out results/phase3/external_sensitivity_NEW
```

The strengthening-v4 runner does not yet exist. Implement it according to the stages above before publishing commands claiming to run that study.
