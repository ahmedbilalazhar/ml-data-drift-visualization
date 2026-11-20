# Project Handoff Prompt for an AI Agent

You are continuing an existing university Data Analytics and Visualization project. Treat the repository as the source of truth. Do not assume every experiment or report is final: identify status from the evidence and mark contradictions, superseded runs, and missing artifacts explicitly.

## Project identity and goal

The project is **Explainable Visualization and Monitoring of ML Data Drift**. It builds a reproducible framework to detect, localize, explain, and evaluate data/concept drift and its impact on ML systems. The repository README describes an E0 pilot and a reproducible harness, with later E1–E6 work and a substantial Phase 3 research/evaluation trail.

## Start here

1. `README.md` — quickstart and high-level repository map.
2. `plan.md` and `docs/research_notes/project-plan.md` — overall plan and project intent.
3. `docs/architecture/research-project-architecture-and-roadmap.md` — architecture and roadmap (planning, not results).
4. `experiments/protocols/research_charter.md`, `experiments/protocols/protocol_outline.md`, and `experiments/protocols/experiment_cards/` — research questions and experiment protocols.
5. `experiments/registries/` — run registry/logs; reconcile it with actual results.
6. `reports/progress/claim_to_evidence.md`, `evidence_bundle.md`, `failure_catalog.md`, `real_exec.md`, and `multiseed_exec.md` — progress, evidence, execution, and known failure context.
7. Latest relevant Phase 3 decision/protocol docs under `docs/phase3/`, then their matching reports under `reports/phase3/` and result/status artifacts under `results/phase3/`.
8. `src/drift_monitoring/`, `scripts/`, `configs/`, and `tests/` — implementation, commands, config, and scientific invariants.

## Repository structure

- `src/drift_monitoring/`: core package. README says it includes simulation, replay/windows, detectors (KS, Wasserstein, JS, ADWIN-inspired, RBF-MMD, joint logistic regression), explanation/ranking and DDE scores, frozen models, evaluation, visualization, orchestration, and real-data loading. Inspect code to confirm current behavior.
- `configs/`: datasets, scenarios, methods, and study YAML files used by runners.
- `scripts/`: smoke, multi-seed, real-data, E1 detection, E2 localization, E3 impact/proxy/delay, E6 reference/sensitivity, heatmap/dashboard export, demos, and dataset audit/download utilities. Use `python scripts\... --help` where supported before running expensive commands.
- `tests/`: unit and phase-specific tests, including leakage, label timing, and resume invariants. Read tests to understand intended behavior; run only targeted tests when asked or needed for a change.
- `experiments/protocols/`, `experiments/registries/`: protocol cards and run records.
- `data/`: raw (intended immutable and gitignored), processed, manifests, and interim data. Never overwrite raw inputs. Check manifests and dataset cards for provenance/licensing.
- `docs/`: charter, problem statement, data cards, architecture, decisions, protocols, and Phase 3 extensions/audits.
- `reports/`: progress, literature, manuscript, and many Phase 3 reports. Multiple versioned reports exist; do not treat every version as current. Follow decision logs and explicit validation/status documents to identify the latest valid result.
- `references/`: literature notes and supplied paper texts/assets. Respect source/license restrictions.
- `notebooks/`: interactive/Colab workflows.
- `research_figures/`: figures and plotting support.
- `studies/`: course requirements/tasks and rubrics.
- `prompts/`: previous research/review agent prompts.
- `output/`, `results/`, `tmp/`, `artifacts/`: generated outputs, data, packages, and temporary checks. These contain many large archives and versioned experiments. Do not commit additional bulky generated data by default; prefer tracked source, compact summaries, manifests, and reproducible commands. Preserve submitted/final packages unless explicitly asked to clean them.

## Current known state at handoff

The GitHub main branch contains the curated project snapshot. Use the [Phase 3 run index](PHASE3_RUN_INDEX.md) for exact paths to status, protocols and output tables for each local run. The original laptop copy contains the versioned ZIP packages, third-party literature copies and large generated traces. A retained manifest may mention an artifact that was left on the laptop. Record those dependencies when assessing reproducibility.

The latest visible documentation includes dated Phase 3 extensions and audits through 2026-11-18. Use actual `STATUS.md`, validation docs, decision logs, and run manifests to state which results are valid, which are exploratory/superseded, and what remains. Avoid asserting all phases are complete simply because many reports exist.

## What to do next

1. Build a concise status map: completed implementation, validated findings, exploratory/superseded work, and open tasks. Cite exact files and sections/commands. Start with the source files listed above and search for `TODO`, `FIXME`, `STATUS`, `validation`, `superseded`, and `limitations`.
2. Verify which result/report is canonical for each claim using protocol, run registry, status/validation, and evidence-to-claim files. Flag inconsistent sample sizes, seeds, data provenance, metrics, or conclusions instead of silently reconciling them.
3. Identify missing reproducibility elements: environment/dependency pins, dataset acquisition instructions, checksums/manifests, random seeds, configs, command lines, expected outputs, and license/provenance notes.
4. Propose a prioritized next-work list with rationale, acceptance criteria, and the exact files to edit. Keep scientific claims scoped to observed evidence.
5. For any code changes, make minimal changes, preserve raw data and existing final artifacts, document commands, and report what was and was not verified. Do not run costly full experiments unless explicitly authorized.

## Working style and output

Be evidence-led. Prefer links/paths and executable commands over vague summaries. Distinguish implementation status from scientific validation. When files disagree, report the disagreement and the evidence needed to resolve it. Deliver:

- a one-paragraph project summary;
- a folder map;
- a status table (done / in progress / missing or uncertain, each with file evidence);
- a prioritized next-steps list;
- reproducibility and publication risks;
- exact commands to reproduce the smallest relevant checks.

## Experiment-by-experiment result map (inspect canonical artifacts)

The repository has a large, evolving experiment matrix, so this map is a guide to the source-of-truth files, not a claim that every run is publication-ready. For each experiment, read its protocol first, then the matching `results/phase3/<run>/` manifest/status/summary and the report folder. Compare versions and explicitly report the final disposition and headline numeric results from the actual artifacts.

- E0 integrity/pilot: `experiments/protocols/experiment_cards/E0_integrity.md`; run IDs/config hashes/statuses in `experiments/registries/runs.csv`; smoke command in `README.md` (`python scripts\smoke_e0.py`).
- E1 detection: `experiments/protocols/experiment_cards/E1_detection.md`; inspect `scripts/` E1 runners, `configs/`, matching result tables, and Phase 2/3 traceability docs.
- E2 localization: `experiments/protocols/experiment_cards/E2_localization.md`; audit and reports under `results/phase3/localization_audit_*` and `reports/phase3/localization_audit_*`; consult `docs/phase3/LOCALIZATION_AUDIT_20261117.md` for current adjudication.
- E3 impact, delayed labels, and proxy/future harm: `experiments/protocols/experiment_cards/E3_impact.md`; related result families include `delay100_availability_*`, `future_harm_forecast_*`, `future_harm_unseen_family_*`, `uci224_delayed_replay_*`, and `noaa_selective_label_replay_*`. Find the canonical protocols in `docs/phase3/` (notably `DELAY100_AVAILABILITY_EXTENSION_20261116.md`, `FUTURE_HARM_FORECAST_EXTENSION_20261116.md`, `FUTURE_HARM_ONSET_RANDOMIZED_FOLLOWUP_20261116.md`, `FUTURE_HARM_UNSEEN_FAMILY_PROTOCOL_20261116.md`, and `UCI224_DELAYED_REPLAY_PROTOCOL_20261116.md`).
- E6 ablations/reference/sensitivity: `experiments/protocols/experiment_cards/E6_ablations.md`; inspect `scripts/` E6 runners and matching output tables; do not equate exploratory smoke runs with full evaluations.
- Phase 3 reproduction/versioning: `docs/phase3/V2_DECISION_LOG.md`, `V2_EXTENSION_MANIFEST.md`, `REPRODUCTION.md`, `reports/phase3/v2_final/`, `reports/phase3/v2_reproduction/`, and matching `results/phase3/v2_*` folders.
- Large-seed robustness: `docs/phase3/LARGE_SEED_ROBUSTNESS_PROTOCOL.md`, `results/phase3/large_seed_robustness_final/`, and `reports/phase3/large_seed_robustness_final/` (also a reprowording report version exists; establish whether it changes only prose or evidence).
- Strengthening v3 and high-dimensional sensitivity: `docs/phase3/STRENGTHENING_V3_PROTOCOL_20261116.md`; `results/phase3/strengthening_v3_evaluation*`; report variants in `reports/phase3/strengthening_v3_evaluation*`. Reconcile expected/completed cells and checksums against `docs/phase3/SCIENTIFIC_EXTENSION_RECONCILIATION_20261118.json`; that reconciliation says the d32/d128 run has all 1,440 expected cells hash-verified, while retaining a historical lock replacement PermissionError in metadata.
- Dependence extension/stability: `docs/phase3/DEPENDENCE_EXTENSION_PROTOCOL_20261116.md`, `DEPENDENCE_STABILITY_PROTOCOL_20261116_v1.md`, result run folders `dependence_extension_*` and `dependence_stability_*`, and corresponding reports.
- ELEC2 calibration/recalibration/temporal extensions: `ELEC2_CALIBRATION_FAMILY_EXTENSION_20261116.md`, `ELEC2_DELAYED_RECALIBRATION_20261116.md`, `ELEC2_TEMPORAL_EXTENSION_20261116.md`; match result folders `elec2_*` and report folders with same stem.
- NOAA/weather provenance and reproduction: `WEATHER_PROVENANCE_AUDIT_20261117.md`, `WEATHER_PROVENANCE_AUDIT_20261116.md`, `weather_noaa_1949_1999_v2/README.md`, `weather_reproduction_v2/README.md`, results under `noaa_*` and `weather_*`, and `reports/phase3/weather_*`. Track source provenance and whether each dataset is real, synthetic, or reconstructed.
- Selective-label and calibration uncertainty audits: protocols/audits in `docs/phase3/`, results in `selective_label_audit_*`, `noaa_reference_calibration_uncertainty_*`, and `noaa_training_uncertainty_*`, with corresponding report directories. Check seed-collision audit JSONs before interpreting independent replication.
- Adaptation policies/capacity/schedule/diversity: `docs/phase3/ADAPTATION_PROTOCOL.md`, `ADAPTATION_SEED_REPLICATION_PROTOCOL.md`; results/reports under `adaptation_*`, `matched_adaptation_*`, `labelbroker_*`, and `feature_truth_*`. Many similarly named smoke, development, audit, and confirmation runs exist. Read `STATUS.md`, protocol locks, seed-collision audits, and canonical confirmation outputs; do not select a version by date/name alone.
- Seasonal stress extension: `results/phase3/seasonal_stress_evaluation_20261118_v1/`, `results/phase3/seasonal_stress_reproduction_20261118_v1/`, `reports/phase3/seasonal_stress_20261118_v1/`, and `docs/phase3/SCIENTIFIC_EXTENSION_RECONCILIATION_20261118.json`. The reconciliation records both 90/90-cell runs complete and exact equality of their seed-level outcome data/figures; it reports three synthetic task families with input alarm excess around 0.744–0.768, feature-7 alarm excess around 0.791, and paired accuracy deltas near zero. Scope this claim to the stated synthetic families and verify report/table details before reuse.
- Automated/prolonged work: `results/phase3/autonomous_24h_20261116/STATUS.md`, `autonomous_extension_20261116/STATUS.md`, and `docs/phase3/AUTONOMOUS_24H_AUDIT_20261116.md`. Treat these as audit trails, not as a substitute for validating each contained experiment.

For every run being summarized, make a row with: experiment question; protocol path; exact run directory; state and expected/completed cells; dataset/provenance; seeds/sample size; primary metric and numeric result with uncertainty; report path; validation/reproduction status; known limitation or supersession. If a run lacks one of these, state `not found` and continue the audit. The CSV registry is a run index, not a scientific results table; obtain outcomes from result summaries/tables and reports.

## Reported outcomes to anchor the handoff

| Experiment | Reported outcome and scope | Evidence |
| --- | --- | --- |
| E0 integrity | Pilot took 25.5 seconds and peaked at 33 MB. Confirm the execution environment before comparing runtime. | [Pilot claim map](../reports/progress/claim_to_evidence.md), [run registry](../experiments/registries/runs.csv) |
| E1 detection | Controlled S1–S6 input changes detected at delay 0–1 at the pilot threshold of 0.097. S0 alarmed in 1 of 12 windows. Conditional-only S7 remained silent with a 0.19 F1 drop. | [Pilot claim map](../reports/progress/claim_to_evidence.md), [E1 protocol](../experiments/protocols/experiment_cards/E1_detection.md) |
| E2 localization | Pilot recall and average precision were both 1.0. Correlated proxies diluted precision at 3. S6 ground truth is defined at the group level. Review later localization audits for finer distinctions. | [Pilot claim map](../reports/progress/claim_to_evidence.md), [localization audit](../docs/phase3/LOCALIZATION_AUDIT_20261117.md) |
| E3 delayed action | Pilot S6 changed from update at delay 0 to investigate at delay 2. At delay 5 the policy continued monitoring during hidden S7 harm. The separate delay-100 experiment measured availability (0 of 5,040 labels by input end, all released by clock 124). | [Pilot claim map](../reports/progress/claim_to_evidence.md), [delay-100 extension](../docs/phase3/DELAY100_AVAILABILITY_EXTENSION_20261116.md) |
| E6 window sensitivity | On S2, windows of 250, 500 and 1,000 all alarmed, with KS max of 0.61, 0.54 and 0.54. The primary window remained fixed at 500. | [Pilot claim map](../reports/progress/claim_to_evidence.md), [E6 protocol](../experiments/protocols/experiment_cards/E6_ablations.md) |
| C6 dependence intervention | Targeted max-correlation detector alarmed in 30 of 30 held-out seeds versus 16 of 30 for KS/PCA. Stationary alarms rose from 0.83 to 1.70 per 24 batches. | [Weakness register](../docs/phase3/PHASE3_REMAINING_WEAKNESSES_20261116.md), [protocol](../docs/phase3/DEPENDENCE_EXTENSION_PROTOCOL_20261116.md) |
| UCI224 interval coverage | Mean widths were 0.0234 exact versus 0.0258 normal, with 0 of 9 cohorts covered by either on the evaluated replay. | [Interval extension](../docs/phase3/UCI224_EXACT_INTERVAL_EXTENSION_20261116.md), [weakness register](../docs/phase3/PHASE3_REMAINING_WEAKNESSES_20261116.md) |
| UCI224 delayed replay | On one ordered stream, equal-cohort MAE was 0.545 frozen versus 0.169, 0.227 and 0.283 with cumulative logistic recalibration at delays 0, 1 and 2. This is a simulated release clock across nine unequal target cohorts. | [Replay protocol](../docs/phase3/UCI224_DELAYED_REPLAY_PROTOCOL_20261116.md), [weakness register](../docs/phase3/PHASE3_REMAINING_WEAKNESSES_20261116.md) |
| ELEC2 calibration family | Matched cumulative isotonic and Platt updates reduced signed bias. Neither lowered MAE or ECE at the evaluated delays 0, 5 and 20. | [Calibration family protocol](../docs/phase3/ELEC2_CALIBRATION_FAMILY_EXTENSION_20261116.md), [weakness register](../docs/phase3/PHASE3_REMAINING_WEAKNESSES_20261116.md) |
| Unseen-family harm forecasting | Time-only AUC was 0.998 in the aligned-onset pilot and 0.875 in the balanced-onset follow-up. Monitor features improved Brier score over time alone in the follow-up. Released labels had no clear Brier improvement. | [Unseen-family protocol](../docs/phase3/FUTURE_HARM_UNSEEN_FAMILY_PROTOCOL_20261116.md), [weakness register](../docs/phase3/PHASE3_REMAINING_WEAKNESSES_20261116.md) |
| Seasonal stress | 90 of 90 cells completed in both the canonical evaluation and independent rerun, with exact equality of compared outcome fields. In three synthetic families, excess input alarm rates were 0.744–0.768 and paired accuracy deltas stayed within 0.000025 of zero. | [Aggregate results](../results/phase3/seasonal_stress_evaluation_20261118_v1/analysis/seasonal_summary.csv), [reconciliation](../docs/phase3/SCIENTIFIC_EXTENSION_RECONCILIATION_20261118.json) |

## Open work to assess first

1. Trace Phase 2 requirements to implemented and evaluated features. Start with [Phase 2 traceability](../docs/phase3/PHASE2_TRACEABILITY.md) and the [implementation audit](../docs/phase3/PHASE3_IMPLEMENTATION_SPEC_AUDIT_20261117.md). Check later experiments against the dated [weakness register](../docs/phase3/PHASE3_REMAINING_WEAKNESSES_20261116.md) before calling a weakness open or resolved.
2. Reconcile newest adaptation capacity, matched schedules, diversity and DDE subset runs against their seed audits and confirmation outputs. Report policy comparisons alongside their label, retraining and information budgets.
3. Specify the evidence that would justify claims about human decisions, operational cost, cross-domain performance, or calibrated probabilities. The current scope supports descriptive localization and controlled replay findings. Causal attribution, decision benefits and broad transfer still require independent evidence.
4. Reproduce the smallest relevant result using its documented data, seeds and runner. Quote the observed number separately from the original report, and flag absent run artifacts or differences in source versions explicitly.
