# Autonomous IDM-Viz Phase 3: 24-hour strengthening harness prompt

Copy everything from **BEGIN PROMPT** through **END PROMPT** into a fresh Codex task. It is designed to resume from the actual workspace state, preserve the submitted Phase 2 methodology, use at most about 12 hours of laptop compute within a 24-hour work window, and leave a reproducible, honestly scoped research package.

---

## BEGIN PROMPT

You are continuing an in-progress research implementation. Do the work in the existing project; do not respond with another plan instead of implementing it.

### Project and non-negotiable context

- Workspace: `C:\Users\hp\Desktop\University\2. DAV\3. Project`
- Project: IDM-Viz, impact-aware drift monitoring that connects input alarms, descriptive feature evidence, estimated accuracy, delayed-label verification, and coordinated visualization.
- Current date at handoff: 16 November 2026, Asia/Karachi. The original 15 November 2026, 5:00 pm deadline has passed. Treat this as a post-deadline, clearly versioned improvement. Do not replace, edit, or imply that the prior submission was changed.
- The actual Phase 2 methodological basis is `docs\23i-2604_23i-2581DAV-phase2.docx`. Keep it byte-for-byte intact. The older `DAV_Phase2_Submission.tex` is supporting material only; never substitute it for the submitted DOCX. Do not invent instructor feedback; none was received.
- Keep `docs\Phase 03 - Implementation.pdf` and the original submitted archive `output\phase3\23I-2581_23I-2604_Phase3.zip` byte-for-byte intact as well. Check and record their current SHA-256 before and after work.
- Read, in order, `docs\phase3\IMPLEMENTATION_PLAN.md`, `docs\phase3\experiment_manifest.json`, the submitted Phase 2 DOCX, and `docs\Phase 03 - Implementation.pdf`. Then inspect current source, tests, results, reports, receipts, archives and git/worktree state. Do not infer current completion from this prompt; other Codex sessions may have changed files.
- Do not reuse legacy runner outputs as valid evidence if they contain current-batch F1/accuracy before label release, if their dependence-only generator changes marginal distributions, or if retrospective losses are displayed as online evidence.
- Existing environment: `C:\Users\hp\anaconda3\python.exe`; laptop is an i7-10610U with approximately 32 GB RAM. Plan for at most 12 cumulative hours of heavy laptop compute and no more than 24 hours of wall-clock work. Use the remaining compute only where it changes evidence quality; do not run a large job just to fill the time. No Kaggle account is required unless a verified, accessible dataset genuinely needs it.
- The user is unavailable and explicitly wants autonomous execution. Do not ask questions or wait for approval for ordinary, reversible research implementation. Never send messages, upload, submit, publish, overwrite protected snapshots, or claim human evaluation without authorization/evidence.

### Resume immediately and maintain durable checkpoints

1. Read the newest versioned archive and adjacent `.receipt.json`. At handoff, the likely latest candidate is `output\phase3\IDM_Viz_Phase3_Final_Strengthening_20261116_v13.zip`, but verify the file and receipt. A package is validated only if its receipt says pass and names the exact archive hash. Earlier v10 timed out; v11 and v12 passed particular validation stages but were superseded by v13. Never state a version passed without checking its receipt.
2. Inspect current `docs\phase3\PHASE3_REMAINING_WEAKNESSES_20261116.md`, `PHASE2_TRACEABILITY.md`, `DECISION_STRENGTHENING_CHANGELOG.md`, the future-harm protocol/report, and all experiment execution receipts. Preserve prior output directories; use a new suffix for every new run.
3. Create or update a durable run ledger at `results\phase3\autonomous_24h_YYYYMMDD\STATUS.json` and human-readable `STATUS.md`. Record start time, elapsed wall time, compute time, current stage, commands, exit codes, result paths, hashes, failed attempts, completed tests, and exact next command. Flush progress after every stage and at least every 15 minutes. Keep long work resumable and never let a context reset erase the experiment state.
4. Check the system clock and remaining 24-hour window. If less than 24 hours remain, scale the plan by preserving the stage order and shortening exploratory breadth; never skip preservation, evaluation, or honest reporting. Give concise user-facing progress updates at meaningful milestones, but continue working.

### Current evidence to verify, not blindly trust

The workspace may already contain these additions. Inspect their code, protocol, results and test receipts first; do not repeat a complete experiment unless the implementation has a material defect:

- Frozen logistic classifier; label-safe online/evaluator separation; 500-row synthetic batches; seven controlled conditions; original-feature KS tests plus frozen three-component PCA; descriptive feature rankings and three clusters; isotonic correctness calibration and Average Confidence; plug-in Poisson-binomial interval and proxy concern score; delays 0/5/20; matched baselines and ablations.
- 1,500-seed-per-task synthetic robustness run and its sample reproduction comparison.
- Corrected unit-marginal C6 dependence intervention and exploratory correlation detector. The reported 30/30 vs 16/30 detection and increased stationary-alarm burden are a tradeoff, not a general result.
- UCI Gas Sensor split/model/history sensitivity; exact Poisson-binomial versus normal interval comparison (both had poor coverage); delay-100 availability-only audit.
- ELEC2/OpenML 151 second-stream exercise. It has 55 batches, no known drift-onset labels, and approximate ordering due normalized-date/weekday ties. Do not score alarms as recall or call it verified chronological ground truth.
- ELEC2 released-label isotonic recalibration comparison: bias moved toward zero, but batch MAE did not improve at delays 0/5/20 and coverage did not consistently improve.
- A subsequent matched-budget calibration-family extension compared frozen isotonic, cumulative isotonic, and cumulative Platt-style logistic calibration on the same ELEC2 predictions/strictly released labels. Both updates reduced signed bias but had worse MAE and ECE than frozen isotonic at all three delays. The focused addendum is `output/phase3/IDM_Viz_ELEC2_Calibration_Family_Addendum_20261116_v2.zip`, with a passing adjacent receipt; verify its SHA and contents.
- Five-batch prospective future-harm synthetic forecast using 30 development and 30 held-out seeds, seven known conditions and delays 0/5/20. Previously observed held-out Brier scores were 0.231 time-only, 0.166 with unlabeled-monitor features and 0.149 with available released labels; 30-seed bootstrap favored both evidence additions. This is interpolation under the same seven known generator laws, not unseen-regime or natural-stream forecasting.
- The current future-harm report is editable TeX with PDF/vector visualizations; verify its `--run`/`--out` reproduction instructions and compiled layout.

These evidence points are provisional until independently confirmed by on-disk files and receipts. If numbers differ, use the actual verified values and explain why.

### Goal and claim discipline

Strengthen the Phase 3 implementation and experimental evaluation as much as the available day permits. Address correctable weaknesses in code, data lineage, calibration, evaluation, baseline fidelity, sensitivity, visualization and reproducibility. Acknowledge that a 24-hour sprint cannot ethically or scientifically manufacture causal identification, arbitrary-shift identifiability, an approved human-subject result, many independent natural domains, or production operating costs. Do not claim “all weaknesses fixed.” Close weaknesses only when the relevant evidence exists; otherwise specify the exact evidence still needed.

### Detailed 24-hour execution schedule and checkpoints

The times below are elapsed from this harness start. Heavy laptop compute is capped at about 12 hours cumulative. Use the remainder for audit, implementation, interpretation, figure/report quality, test/package validation and durable handoff.

#### Hours 0–1.5 — forensic state audit and locked priorities

- Read required Phase 2 and Phase 3 files. Extract the Phase 2 objectives, constructs, proposed architecture and intended evaluation into a traceability table without rewriting the DOCX.
- Inventory relevant files with `rg --files`; inspect git status before editing. Do not “clean up” unrelated files or deletions.
- Verify current archive receipts, source hashes, protected document hashes, data hashes, experiment locks, incomplete/failed runs and any still-running research process. Do not stop unrelated processes.
- Map every outstanding weakness to: fix in code today; test today; new experiment today; requires more data/ethics/causal assumptions; or out of scope. Rank by scientific validity, not visual polish.
- Deliverable checkpoint: a dated audit note, current status ledger and prioritized execution queue. Begin implementation immediately; this is not a plan-only phase.

#### Hours 1.5–4 — correctness, leakage and reproducibility hardening

- Audit every path from predictions to released labels, evaluator truth, online views, summaries and reports. Confirm forecasts precede same-clock label releases and unavailable labels are missing rather than zero-filled/backdated.
- Audit fit boundaries: all scalers, PCA, clusters, calibrators, thresholds and predictive models must use only permitted development/historical data. Confirm no held-out seed, future cohort target or condition identifier leaks into model predictors.
- Add/repair scientific-invariant tests for schedule boundaries, IDs, exact cohort joins, no future features, seed disjointness, marginal-preserving dependence changes, interval implementation, output schema and clean-run determinism. Tests must fail on a true invariant violation, not just check a saved file exists.
- Audit dashboard and report data tables separately from online serving outputs. Ensure retrospective accuracy/F1/harm never appears with an online/as-of timestamp before its release.
- Fix any material bug before spending compute. Preserve failed outputs; write a diagnosis and create a new versioned run.
- Checkpoint: all targeted tests pass; no known leakage bug remains unaddressed or undocumented.

#### Hours 4–6 — external-data/protocol audit and strongest feasible new stream

- Audit ELEC2 and Gas Sensor source hashes, licenses/attribution, preprocessing, split windows, feature exclusions, target definition, timestamp order, duplicate/tied times, missingness, class balance, and source-to-result joins.
- Select at most one additional independently sourced dataset only if it has defensible ordering and a target/model interface relevant to this question. Prefer a recognized public benchmark with a primary/official source and transparent provenance. Verify its documentation, license, timestamp semantics and download integrity before use. Reject a dataset if temporal ordering is fabricated or unrecoverable.
- Freeze train/calibration/reference/monitor windows, model, features, normalization, seed, release delays, baselines and metrics before seeing the new monitoring outcomes. Keep a small development preview separate from held-out evaluation. If no suitable third stream is feasible, spend this block on a second defensible split/window of an already sourced stream and label it sensitivity, not an independent domain.
- Do not compare numbers to papers unless datasets, preprocessing, labels, metrics and protocol are genuinely comparable. Name “style/inspired” baselines accurately.
- Checkpoint: provenance report, frozen protocol, ingestion tests and a go/no-go decision written to the ledger.

#### Hours 6–12 — bounded, predeclared compute block (maximum six hours here)

Choose only the branches that address the highest-ranked open weaknesses and fit within the remaining compute budget. Run one foreground job at a time; checkpoint output every seed/split. Keep raw predictions/evaluator truth separate. Use seeds as the independent unit where appropriate.

**A. Calibration transfer and delayed-label sensitivity (preferred if no new suitable dataset):**

- On the existing Gas Sensor and ELEC2 streams, compare the frozen isotonic estimator with only predeclared, budget-matched alternatives (e.g. logistic/Platt and a fixed rolling isotonic window), all refit only from labels strictly available before each forecast.
- Match label budgets, prediction rows, release clocks and classifier; report MAE, signed bias, Brier/ECE or reliability summaries, and interval coverage/width. State that one serial stream is not an independent-replicate confidence interval. Use multiple preregistered split windows where defensible.
- Include an explicit “no update” baseline and the cumulative-isotonic result already observed. Do not tune a method on held-out outcomes and report the same holdout as confirmatory.

**B. Dependence-drift detector stability:**

- Use independent development and held-out seeds and an explicitly fixed false-alarm budget. Evaluate multiple correlation strengths, affected coordinate pairs, sample sizes, stationary nulls and marginal-preserving interventions. Compare against KS/PCA and a computationally fair multivariate dependence comparator.
- Report detection delay, missed detections, false alarms per 24 batches, and paired seed-level uncertainty. If a threshold is selected on development seeds, freeze it before held-out evaluation. Do not say one correlation detector solves dependence drift generally.

**C. Label selection and action policy:**

- Verify equal-budget comparisons under the exact same eligible labels, acquisition cost and release schedule. Add an inverse-probability or known inclusion-probability reference only if positivity and the sampling mechanism are explicit. Report policy regret/utility only with a predeclared synthetic utility function and label it simulation, not human/financial benefit.

**D. Future-harm forecasting:**

- Do not rerun the existing 30+30-seed forecast unless its source audit fails. If extending it, add a genuinely unseen intervention family with separate training and test generator families, no test-time tuning, and a baseline/model comparison fixed before test outcomes. If that cannot be done scientifically within the budget, stop and state that the held-out-seed result only interpolates among known laws.

During this block, terminate only a process you started if it is provably stuck or invalid; do not kill unrelated processes. On interruption, preserve partial status and resume without overwriting. Stop new compute by hour 12 unless an already-started high-value job has a credible finish time and remains under the 12-hour total compute cap.

#### Hours 12–15 — independent replication and statistical analysis

- Re-run the selected evaluator from saved predictions and truth rather than regenerate data unnecessarily.
- Use paired, seed/stream-level bootstrap for synthetic estimates; preserve pairing across methods. For a single external temporal stream, do not bootstrap batches as if independent; use descriptive metrics and carefully justified block/sensitivity analyses.
- Report point estimates, denominators, missing/undefined metrics, uncertainty intervals, class prevalence and all unsuccessful/negative outcomes. Use ROC AUC only when both classes occur; report undefined explicitly otherwise. Compare average precision with prevalence.
- Add checks for multiplicity where multiple conditions/detectors are searched. Distinguish exploratory outcome-informed work from locked confirmatory protocol.
- Verify exact seed lists, model/feature/threshold locks, label availability and reproduction against the execution receipts.

#### Hours 15–18 — beautiful but truthful figures and connected evidence view

- Produce publication-quality vector figures (PDF/SVG) plus PNG previews using a consistent, restrained palette, readable typography, legible labels, uncertainty/denominators and accessible contrast. Figures must encode the actual result and include sample sizes and metric definitions where helpful.
- Prioritize: calibration transfer/reliability and error by stream/split; detector sensitivity vs false-alarm tradeoff; forecast skill vs delay and baseline; a method/evidence timeline showing input alarms, estimates, released labels and retrospective outcomes at their valid availability clocks.
- Regenerate the coordinated evidence view from label-safe online artifacts. Provide separate retrospective layers with explicit verification timestamps. Run schema and behavior checks; do not represent automated checks as a human usability evaluation.
- Render the TeX report PDFs to images and inspect every page at readable size. Fix clipping, overflow, tiny text, awkward page breaks and misleading axis scales. Preserve editable `.tex` and source data/figure scripts.

#### Hours 18–21 — integrate the empirical report and traceability

- Update the current main report or create a clearly identified, versioned addendum tied to the prior v5 report. Include the Phase 2 traceability table, implementation changes, dataset provenance/preprocessing, exact experimental setup, baselines, ablations, all results, negative results, limitations, implementation/dependency details, reproduction commands, compute time and the truthful scope changes.
- Preserve the submitted Phase 2 DOCX unchanged; refer to it as the methodological basis. Do not imply instructor comments. Clearly distinguish the original methodology, implemented/justified amendments, and deferred work: causal discovery, neural autoencoders, extensive adaptation and human-subject evaluation remain deferred unless they are actually implemented and supported.
- Compare to literature only with verifiable primary sources and fair settings; otherwise state why numerical comparison is not fair and provide a conceptual comparison. Do not fabricate citations, participants, results or benefits.
- Update `PHASE2_TRACEABILITY.md`, the weakness disposition register, changelog and reproduction guide so they agree with the report and receipts.

#### Hours 21–23 — release candidate and clean-room reproduction

- Create a new, clearly versioned archive; never overwrite prior submitted or validated archives. Include source, tests, frozen protocols, data source/license notes, relevant cached outputs, report `.tex`/PDF/figures, and hash manifests.
- Run the focused and full test suites. Run clean extraction in a temporary directory; execute the exact reproduction commands required for the latest new analyses, evaluators and report builder. Compare deterministic CSVs and report TeX/figure pixels where appropriate. Record nondeterministic PDF metadata separately from actual figure/source equality.
- Verify archive CRC, member SHA-256 values, source/protocol/result hashes, and protected Phase 2/submission hashes before and after packaging. If any validation stage fails, retain the failed version with a failure receipt and create a new version; do not label it final.
- Do not upload or submit anything.

#### Hours 23–24 — final audit and handoff

- Review every claim against source data and results. Ensure no retrospective label, impossible denominator, ambiguous “accuracy estimate,” missing negative finding or unsupported causal/human-benefit language remains.
- Confirm final archive has a passing receipt, full report source and PDF, reproduction commands, hashes, test counts and no missing referenced files. Check that the latest weakness register is candid about unresolved issues.
- Write a concise final handoff with what changed, actual measurements, all commands, exact absolute artifact paths, validation receipts, preserved-file hashes and the next evidence needed for each remaining weakness. Mention any error/timeout and its recovery transparently.
- Leave `STATUS.json` and `STATUS.md` complete so another harness can resume without repeating work.

### Required scientific acceptance gates

Do not mark the 24-hour work complete unless all applicable gates below pass or are explicitly documented as blocked by external scientific requirements:

1. Protected Phase 2 and submitted snapshot SHA-256 hashes match the initial audit.
2. No current/future label leakage exists in online prediction, monitoring features, forecast inputs, dashboard state or policy selection.
3. Development, validation and held-out units are disjoint at the level claimed (seed, stream, split, or domain); uncertainty resampling uses the independent unit.
4. All thresholds, windows, models, feature lists and outcomes were locked before held-out evaluation, or the work is labeled exploratory.
5. Dataset source, license, preprocessing, ordering limitations, target and feature exclusions are documented and reproducible.
6. Baseline labels accurately describe implementations; no unsupported paper-number comparison is made.
7. All results include denominators and uncertainty appropriate to the design; negative and undefined results are retained.
8. Editable TeX, generated PDF and figure sources are included and layout-checked.
9. Unit/scientific tests, clean extraction, hashes, CRC, and fresh evaluator/report commands have a machine-readable pass receipt.
10. The final scientific claim is bounded to actual evidence. Do not claim causal attribution, arbitrary-shift identification, human benefit, distribution-free risk certification, production cost savings, or unseen-domain generalization without the corresponding evidence.

### Final-response format

Give a self-contained, concise but evidence-rich handoff. Include:

- work actually completed and material improvements over the on-disk starting state;
- new experiment outcomes, negative results, denominators and seed/stream-level uncertainty;
- exact full-suite and clean-extraction validation results;
- absolute links to the editable TeX report, compiled PDF, figures, code/protocols, weakness register, final archive and validation receipt;
- exact reproduction commands and environment;
- hashes proving preservation of the submitted Phase 2 DOCX and original submission snapshot;
- remaining weaknesses and the specific evidence needed to close them;
- actual elapsed time and compute used, including failed/aborted runs.

Never say all weaknesses have been solved if any of these remain open. Finish only when the best feasible evidence, report and reproducible package are delivered and the ledger says `complete` or lists an external blocker honestly.

## END PROMPT
