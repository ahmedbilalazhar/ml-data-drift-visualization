# IDM-Viz: autonomous overnight execution and full strengthening plan

## Time and objective

Inspected at 23:45 Asia/Karachi on 15 November 2026. The attached handoff defines an existing eight-hour window, 22:29–06:29, not eight new hours from this inspection. Preserve the running locked experiment and use the remaining approximately 6 h 44 min for completion, analysis, reproduction and packaging. It is impossible to guarantee completion if Windows sleeps, power fails, the scientific run fails, or computation exceeds the window. Save explicit status and partial evidence in all cases.

The user authorized autonomous execution while asleep and continuation independent of Codex availability. A separate local Python continuation process is therefore used. It requires no model calls or API credits. It does not schedule future Codex activity or create a recurring watcher. It performs one finite dependency-ordered workflow.

## Preserved foundation

The submitted Phase 2 DOCX is authoritative, with SHA-256 `786d4f23d54d0d09edd5f94740198f43ca3b406eb38626ae754fe04fe26c5ef4`. No instructor feedback exists. Submitted Phase 3 archive SHA-256 is `279d9867d1c75720a51df12caae2ac0b4d490641b83752e32642dcb87e1c8877`. Verify both before and after. Preserve all historical reports, archives, manifests and results. New packages are prepared, not uploaded.

## Current experiments: do not restart

1. Locked robustness study, existing PID 29092 when inspected: `scripts/run_phase3_large_seed.py --mode final --out results/phase3/large_seed_robustness_final`. Original-task and XOR task, 1,500 fresh seeds 1000–2499 per task, paired logistic/tree, 7+5 conditions, 24×500 rows. Expected totals: 18,000 physical streams, 36,000 model streams, 216 million monitoring rows, 864,000 model-batch records. These are expected denominators, not completed results.
2. An independent external-method training pilot was also running, PID 24500 when inspected. Do not stop, modify, duplicate, or claim its outcome. Its completion must be assessed from its own protocol and receipts in a separate follow-up.
3. Last direct count was 585 original-task seeds with at least some summary output, through seed 1584; that last seed was partial. Never equate a seed appearing in a growing file with all its conditions completing.

## Mandatory workflow, in order

### A. Preserve and observe the existing locked run

- Record original-document and archive hashes. Leave numerical code and protocol unchanged.
- The continuation waits for `protocol_lock.json` status `complete`; it never launches a second runner. Save a bounded tail of summary progress every two minutes.
- If the run marks itself failed, stop downstream interpretation and save the error. No seed replacements, silent exclusions, shortening of the seed list, or rerun of favorable subsets.
- Cutoff for waiting is 05:49 Karachi, reserving 40 minutes before 06:29. If still running, preserve partial evidence and mark the automatic chain blocked. Do not kill the independent numerical process or mislabel partial data as complete.

### B. Evaluate the completed study

Run `evaluate_phase3_large_seed.py` against the existing final directory. Validate exact task/seed/condition/model denominators, joins, online truth exclusion, finite fields, harm boundaries and model pairing. Use the locked 2,000 complete-seed bootstrap replicates. Preserve and report the analyzer output-label amendment with locked and executed hashes.

Primary questions:

- Do the original 30-seed model/task contrasts persist over fresh streams?
- How stable are calibrated and raw accuracy-estimation error, bias and interval coverage/width?
- How often does the concern score miss harmful batches, including conditional-only changes?
- How often do original-feature/PCA alarms match controlled input-change events, and how frequent are null alarms?

Report all conditions, both models, regime definitions, counts and undefined metrics. Do not describe large synthetic sample size as external validation or an additional model family. More seeds reduce Monte Carlo uncertainty; they do not resolve the absence of a human study or real-world generalization.

### C. Independent reproduction and tests

Regenerate the predeclared sample seeds 1000, 1375, 1750, 2125, 2499 for both tasks and all conditions using the separate comparison script. Require exact parsed scientific CSV equality; exclude explicitly named measured timings. This is ten task-seed reproductions, not a full independent rerun of 3,000 task-seeds.

Run the broad test suite once after completion. Record command, exit code, elapsed time and output. If either sample reproduction or tests fail, save logs and stop the package-success claim. No automated modification of science to force passing results.

### D. Reports and graphical evidence

Run the existing robustness report generator with its actual sample-reproduction receipt. Produce paired model accuracy contrasts and calibration-error/coverage figures. Include effect sizes and paired uncertainty, concern confusion counts, missed harms and detector denominators.

Generate a rendered PDF plus page PNGs with the existing renderer. Also produce editable single-column A4 LaTeX from the same report text/tables and compile using local Tectonic. Preserve a ReportLab fallback PDF if TeX fails. Record compilation failure honestly; do not claim an attractive consolidated manuscript has been completed by automatic format conversion.

Detached execution cannot inspect page images as an agent. Set visual review to pending. Keep rendered review assets for morning inspection; do not fabricate `visual_review: passed`.

### E. Self-contained supplementary package

The older package builder incorrectly requires the optional large adaptation replication. Do not use it to fabricate a two-study package. Build a NEW robustness-only supplement with code, manifests, environment, full aggregate output, sample reproduction, tests/logs, figures, report sources and PDFs.

Preserve recognizable original module names rather than rewriting scientific source imports for cosmetic archive names. Include SHA inventories and CRC verification. Extract outside the workspace and run included Phase 3 tests there. The extracted test result belongs in an adjacent final receipt. Save original-archive preservation hashes again. Mark prepared, not uploaded, and visual review pending.

## Resource controls and contingency

- Keep the current main study at its locked one-thread setting. Do not add a competing large run. The other pilot is user work and remains undisturbed.
- Keep Windows awake for computation using a process-scoped execution-state request; release it when the continuation exits. This does not protect against shutdown, reboot, closing the lid under sleep policies, or disconnected power.
- Logs and receipts are durable on disk; no tool-session identifier is needed to inspect them later.
- Every subprocess has a timeout bounded by the remaining deadline. Mandatory stage failure stops the chain; LaTeX failure preserves the alternative PDF and source.
- The original final run has no new resume mechanism. Do not promise crash recovery or edit it mid-run to add one. Preserve failed/partial output and plan a separately versioned rerun if necessary.
- Do not start the optional 1,500-seed adaptation study overnight. Earlier adaptation evidence exists; an additional long study would compete with mandatory validation/reporting.

## Morning acceptance and remaining full-project strengthening

Overnight completion is a robustness supplement, not completion of every earlier prompt. Use this decision table next:

| Gate | Required action | Evidence needed |
|---|---|---|
| Scientific integrity | Check controller status, numerical lock, evaluator, sample comparison and tests | All required stages passed, no omitted seeds |
| Rendering | Inspect every generated page and figure; fix layout and recompile | Actual visual review record |
| Full methodology coverage | Reconcile original, XOR/tree, adaptation, external and robustness evidence against Phase 2 objectives | Updated traceability/deviation table |
| External comparisons | Read independent pilot protocol/results; complete fixed matched estimator/model comparisons only if still required | Locked configuration, fitted-data budgets, cohort-level and class metrics |
| External timing | Keep retrospective cohort results separate; implement real cached delayed-label replay only when claiming it | Variable-cohort broker and future-label perturbation checks |
| Calibration/interval explanation | Quantify transfer failure and uncertainty assumptions; compare exact and approximate intervals if planned | Matched probabilities and interval tables |
| Ablations | Consolidate feature/PCA, calibration and concern-rule results with model-matched harm targets | Explicit paired denominators and unchanged components |
| Literature | Verify primary sources and compare compatible settings; qualify incompatible numbers | Real citations and comparison table |
| Full report | Integrate all studies into one polished editable LaTeX manuscript, bibliography, figures and PDF | Approximately 14–18 substantive pages as justified, no padding |
| Evidence view/demo | Recheck as-of filtering, linked selections, delays and conditional failure example | Browser checks and five-minute demo guide |
| Final package | Synchronize current hardened code and hashes; clean extracted reproduction; simplify nested historical archives | New version, tested commands, preserved submitted snapshot |

The detailed technical experiment matrix and implementation tasks remain in `docs/phase3/NEXT_IMPLEMENTATION_PLAN_20261115.md`. Treat them as prospective work and reconcile them with the newer locked robustness protocol and external pilot before running anything. Never rerun completed expensive work merely because a plan lists it.

## Files and commands

Continuation implementation: `scripts/continue_phase3_overnight.py`.

Live status/logs: `results/phase3/overnight_continuation_20261115/`.

Expected reports: `reports/phase3/overnight_robustness_20261115/`.

Expected supplementary archive: `output/phase3/IDM_Viz_Robustness_Overnight_20261116.zip`, plus receipt.

Check `status.json` first. Only `completed_automated_checks` means all mandatory automatic stages succeeded; visual review still remains pending. The controller refuses duplicate launch via an exclusive lock file. Do not remove that lock or start another controller while it is alive.
