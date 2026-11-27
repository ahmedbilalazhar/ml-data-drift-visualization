# Authoritative Project Pointer Registry & Location Map for AI Agent / Collaborator

Start with the [Phase 4 demo](../docs/DEMO.md), [four coursework submissions](../docs/submissions/), [editable paper](../paper/), and [Phase 4 evidence map](../EXPERIMENTS.md). The map traces the published claims back to experiment code and evidence. The material below records Phase 3 context and open research work.

**Project Title**: Explainable Visualization and Monitoring of ML Data Drift (IDM-Viz)  
**Authors**: Ahmed Bilal Azhar (23I-2581) & Umar Behram (23I-2604)  
**Repository Remote**: `https://github.com/ahmedbilalazhar/ml-data-drift-visualization.git` (`main` branch)  
**Type**: Pure File Pointer Map & Location Registry (Contains zero procedural workflow directives).

---

## 1. Primary Entrypoints & Core Registers

- **Repository Overview & Architecture Map**: [`README.md`](../README.md)
- **Primary Experiment Registry (E0–E6 Specification & Output Pointers)**: [`EXPERIMENTS.md`](../EXPERIMENTS.md)
- **Phase 3 Experimental Run Index**: [`prompts/PHASE3_RUN_INDEX.md`](PHASE3_RUN_INDEX.md)
- **Project Roadmap & Research Scope**: [`plan.md`](../plan.md) and [`docs/research_notes/project-plan.md`](../docs/research_notes/project-plan.md)
- **System Architecture & Technical Design**: [`docs/architecture/research-project-architecture-and-roadmap.md`](../docs/architecture/research-project-architecture-and-roadmap.md)
- **Scientific Research Charter & Invariants**: [`experiments/protocols/research_charter.md`](../experiments/protocols/research_charter.md)
- **Historical Run Log**: [`experiments/registries/runs.csv`](../experiments/registries/runs.csv)

---

## 2. Scientific Implementation & Source Code Map

All production research code is under `src/drift_monitoring/`:

- **Statistical Drift Detectors** (Univariate KS, Wasserstein, JS Divergence, ADWIN-style, RBF-MMD, Joint Logistic Drift):
  - Implementation: [`src/drift_monitoring/detectors/drift_detectors.py`](../src/drift_monitoring/detectors/drift_detectors.py)
  - Unit Tests: [`tests/unit/test_p0_detectors.py`](../tests/unit/test_p0_detectors.py), [`tests/unit/test_p2_detectors.py`](../tests/unit/test_p2_detectors.py)
- **Drift Localization & DDE Scoring** (Data Drift Explanation via Shapley/ranking):
  - Implementation: [`src/drift_monitoring/explanations/localization.py`](../src/drift_monitoring/explanations/localization.py)
  - DDE Demo Script: [`scripts/demo_dde.py`](../scripts/demo_dde.py)
- **Frozen Models & Recalibrators** (Isotonic regression, Platt scaling, Logistic recalibration):
  - Implementation: [`src/drift_monitoring/models/frozen_models.py`](../src/drift_monitoring/models/frozen_models.py)
  - Calibration Unit Tests: [`tests/unit/test_t10_calibration.py`](../tests/unit/test_t10_calibration.py)
- **Stream Orchestration, Timing Windows & Evaluation Harness**:
  - Implementation: [`src/drift_monitoring/orchestration/runner.py`](../src/drift_monitoring/orchestration/runner.py)
  - Real Data Loader: [`src/drift_monitoring/data/real.py`](../src/drift_monitoring/data/real.py)
- **Aligned Multi-View Visualizations & Heatmaps**:
  - Implementation: [`src/drift_monitoring/visualization/aligned_views.py`](../src/drift_monitoring/visualization/aligned_views.py)
  - Heatmap Export Script: [`scripts/export_heatmap.py`](../scripts/export_heatmap.py)
- **Phase 3 Simulation Engine, Delay Queues & Adaptation Broker**:
  - Core Modules: [`src/drift_monitoring/phase3/core.py`](../src/drift_monitoring/phase3/core.py), [`src/drift_monitoring/phase3/evaluation.py`](../src/drift_monitoring/phase3/evaluation.py), [`src/drift_monitoring/phase3/adaptation.py`](../src/drift_monitoring/phase3/adaptation.py), [`src/drift_monitoring/phase3/replay.py`](../src/drift_monitoring/phase3/replay.py)
  - Scientific Contract Tests: [`tests/phase3/test_scientific_contract.py`](../tests/phase3/test_scientific_contract.py), [`tests/phase3/`](../tests/phase3/)

---

## 3. Findings, Reports & Academic Manuscripts

All 148+ compiled academic research PDFs, LaTeX sources, and standalone reports are tracked in Git. Only canonical versions are listed below (older draft iterations v1–v4 are superseded):

### A. Core Academic Manuscripts & Submission Documents
- **Phase 1 Literature & Drift Survey**:
  - LaTeX Manuscript: [`reports/manuscript/phase1/Phase1_Drift_Report.tex`](../reports/manuscript/phase1/Phase1_Drift_Report.tex)
  - Compiled PDF: [`reports/manuscript/phase1/Phase1_Drift_Report.pdf`](../reports/manuscript/phase1/Phase1_Drift_Report.pdf)
  - Literature Review: [`reports/literature/deep-research-report.md`](../reports/literature/deep-research-report.md) & [`reports/literature/deep-research-report-phase1.pdf`](../reports/literature/deep-research-report-phase1.pdf)
- **Phase 2 Research Report & Feedback Review**:
  - Review Notes: [`reports/manuscript/phase2_review_notes.md`](../reports/manuscript/phase2_review_notes.md)
  - Submission Document: [`docs/submissions/phase2/DAV_Phase2_Revised_Report.md`](../docs/submissions/phase2/DAV_Phase2_Revised_Report.md) & [`docs/23i-2604_23i-2581DAV-phase2.docx`](../docs/23i-2604_23i-2581DAV-phase2.docx)
- **Phase 3 Core Comprehensive Submission Documents**:
  - Core Research Report: [`reports/phase3/v2_archive_extracted/Report.md`](../reports/phase3/v2_archive_extracted/Report.md) & [`reports/phase3/v2_archive_extracted/Report.pdf`](../reports/phase3/v2_archive_extracted/Report.pdf)
  - Adaptation & Retraining Report: [`reports/phase3/v2_archive_extracted/Adaptation.md`](../reports/phase3/v2_archive_extracted/Adaptation.md) & [`reports/phase3/v2_archive_extracted/Adaptation.pdf`](../reports/phase3/v2_archive_extracted/Adaptation.pdf)
  - Architectural Decisions Log: [`reports/phase3/v2_archive_extracted/Decisions.md`](../reports/phase3/v2_archive_extracted/Decisions.md)
  - Rigorous Protocol Specification: [`reports/phase3/v2_archive_extracted/Protocol.md`](../reports/phase3/v2_archive_extracted/Protocol.md)
  - Clean Reproduction Guide: [`reports/phase3/v2_archive_extracted/Reproduce.md`](../reports/phase3/v2_archive_extracted/Reproduce.md)
  - Interactive Replay Dashboard: [`reports/phase3/v2_final/23I-2581_23I-2604_Phase3_Evidence_Replay.html`](../reports/phase3/v2_final/23I-2581_23I-2604_Phase3_Evidence_Replay.html)

### B. Canonical Phase 3 Empirical Findings (Latest Final Versions Only)
- **Decision Strengthening (Canonical: v5)**:
  - Report: [`reports/phase3/decision_strengthening_v5/REPORT.pdf`](../reports/phase3/decision_strengthening_v5/REPORT.pdf)
  - Metrics: [`results/phase3/decision_strengthening_v5/`](../results/phase3/decision_strengthening_v5/)
- **Large-Seed Robustness & Wording (Canonical: final / reprowording_v1)**:
  - Report: [`reports/phase3/large_seed_robustness_final/23I-2581_23I-2604_Phase3_Robustness_Report.pdf`](../reports/phase3/large_seed_robustness_final/23I-2581_23I-2604_Phase3_Robustness_Report.pdf)
  - Refined Worded Report: [`reports/phase3/large_seed_robustness_final_reprowording_v1/23I-2581_23I-2604_Phase3_Robustness_Report.pdf`](../reports/phase3/large_seed_robustness_final_reprowording_v1/23I-2581_23I-2604_Phase3_Robustness_Report.pdf)
  - Summary Metrics: [`results/phase3/large_seed_robustness_final/summary_metrics.csv`](../results/phase3/large_seed_robustness_final/summary_metrics.csv)
- **Adaptation Schedule Audit (Canonical: v19)**:
  - Audit Report: [`reports/phase3/adaptation_schedule_audit_20261118_v19/adaptation_schedule_audit.pdf`](../reports/phase3/adaptation_schedule_audit_20261118_v19/adaptation_schedule_audit.pdf)
- **Capacity Control Under Delayed Labels (Canonical: v9)**:
  - Audit Report: [`reports/phase3/adaptation_capacity_control_20261118_v9/adaptation_capacity_control.pdf`](../reports/phase3/adaptation_capacity_control_20261118_v9/adaptation_capacity_control.pdf)
- **Seasonal Stress & Regime Shifts (Canonical: 20261118_v1)**:
  - Report: [`reports/phase3/seasonal_stress_20261118_v1/REPORT.pdf`](../reports/phase3/seasonal_stress_20261118_v1/REPORT.pdf)
  - Verification Metrics: [`results/phase3/seasonal_stress_evaluation_20261118_v1/`](../results/phase3/seasonal_stress_evaluation_20261118_v1/)
- **Future Harm Prospective Forecasting (Canonical: v10 & Balanced Unseen)**:
  - Forecast Evaluation: [`reports/phase3/future_harm_forecast_20261116_v10/FUTURE_HARM_FORECAST.pdf`](../reports/phase3/future_harm_forecast_20261116_v10/FUTURE_HARM_FORECAST.pdf)
  - Balanced Unseen Family: [`reports/phase3/future_harm_unseen_family_balanced_20261116_v1/FUTURE_HARM_UNSEEN_FAMILY.pdf`](../reports/phase3/future_harm_unseen_family_balanced_20261116_v1/FUTURE_HARM_UNSEEN_FAMILY.pdf)
- **ELEC2 Calibration Family & Recalibration**:
  - Calibration Report: [`reports/phase3/elec2_calibration_family_20261116_v1/ELEC2_CALIBRATION_FAMILY.pdf`](../reports/phase3/elec2_calibration_family_20261116_v1/ELEC2_CALIBRATION_FAMILY.pdf)
  - Delayed Recalibration: [`reports/phase3/elec2_recalibration_20261116/ELEC2_RECALIBRATION.pdf`](../reports/phase3/elec2_recalibration_20261116/ELEC2_RECALIBRATION.pdf)
- **UCI Gas Sensor Drift Benchmark (Exact Intervals & Delayed Replay)**:
  - Exact Intervals: [`reports/phase3/uci224_exact_intervals_20261116/UCI224_EXACT_INTERVALS.pdf`](../reports/phase3/uci224_exact_intervals_20261116/UCI224_EXACT_INTERVALS.pdf)
  - Delayed Replay (Canonical: v5): [`reports/phase3/uci224_delayed_replay_20261116_v5/REPORT.pdf`](../reports/phase3/uci224_delayed_replay_20261116_v5/REPORT.pdf)
- **NOAA Weather Experiments (Training Uncertainty & Selective Label Replay)**:
  - Training Uncertainty (Canonical: v3): [`reports/phase3/noaa_training_uncertainty_20261117_v3/NOAA_TRAINING_UNCERTAINTY.pdf`](../reports/phase3/noaa_training_uncertainty_20261117_v3/NOAA_TRAINING_UNCERTAINTY.pdf)
  - Selective Label Replay (Canonical: v3): [`reports/phase3/noaa_selective_label_replay_20261117_v3/NOAA_SELECTIVE_LABEL_REPLAY.pdf`](../reports/phase3/noaa_selective_label_replay_20261117_v3/NOAA_SELECTIVE_LABEL_REPLAY.pdf)
  - Holdout Timeline: [`reports/phase3/weather_noaa_1949_1999_v2/REPORT.pdf`](../reports/phase3/weather_noaa_1949_1999_v2/REPORT.pdf)

---

## 4. Experimental Metrics, Run Checkpoints & Figures

- **E0–E6 Run Execution Checkpoints**:
  - Directory: [`results/runs/`](../results/runs/) (includes parameter hashes, seed manifests, and execution timestamps for all experiments E0 through E6).
- **Aggregated Metric Tables**:
  - Directory: [`results/aggregates/`](../results/aggregates/) (includes CSV aggregations for detection latency, localization accuracy, calibration ECE, and harm counts across cohorts).
- **Experimental Visual Figures & Heatmaps**:
  - Directory: [`results/figures/`](../results/figures/) (pilot phase detection curves, temporal calibration diagrams, and feature localization charts).
- **Canonical Phase 3 Verification Runs**:
  - Final Verification Cohort: [`results/phase3/final_v1/`](../results/phase3/final_v1/) (tracked metrics: calibration error, delay availability, detection latency, fit records, harm counts, localization scores, runtime).
  - Clean Reproduction Cohort: [`results/phase3/reproduction_v1/`](../results/phase3/reproduction_v1/) (independent seed replay verification matching `final_v1`).

---

## 5. Empirical Weaknesses & Methodological Boundaries

The project's empirical boundaries and investigation logs are cataloged in [`docs/phase3/PHASE3_REMAINING_WEAKNESSES_20261116.md`](../docs/phase3/PHASE3_REMAINING_WEAKNESSES_20261116.md):

| Weakness Investigation Area | Evidence & Implementation Pointer | Methodological Finding / Disposition |
| :--- | :--- | :--- |
| **Label Leakage Prior to Release** | [`docs/phase3/E3_DELAY_AUDIT.md`](../docs/phase3/E3_DELAY_AUDIT.md), [`tests/phase3/test_scientific_contract.py`](../tests/phase3/test_scientific_contract.py) | **Resolved in code**: Evaluator and online loops strictly isolated via asynchronous release queue broker. |
| **Marginal Shift Confound in C6** | [`docs/phase3/DEPENDENCE_EXTENSION_PROTOCOL_20261116.md`](../docs/phase3/DEPENDENCE_EXTENSION_PROTOCOL_20261116.md), [`reports/phase3/dependence_extension_20261116/`](../reports/phase3/dependence_extension_20261116/) | **Characterized**: Targeted max-correlation detector achieved 30/30 detection; stationary false alarms increased from 0.83 to 1.70 per 24 batches. |
| **Normal Interval Coverage Failure** | [`docs/phase3/UCI224_EXACT_INTERVAL_EXTENSION_20261116.md`](../docs/phase3/UCI224_EXACT_INTERVAL_EXTENSION_20261116.md) | **Evaluated**: Exact Poisson-binomial intervals are narrower (0.0234 vs 0.0258), but both failed to cover 0/9 cohorts due to severe miscalibration under unannounced drift. |
| **Delayed Recalibration Transfer** | [`docs/phase3/ELEC2_CALIBRATION_FAMILY_EXTENSION_20261116.md`](../docs/phase3/ELEC2_CALIBRATION_FAMILY_EXTENSION_20261116.md) | **Characterized Failure**: Isotonic and Platt updates reduced signed bias but failed to reduce MAE or ECE on ELEC2 stream. |
| **Prospective Harm Prediction** | [`docs/phase3/FUTURE_HARM_UNSEEN_FAMILY_PROTOCOL_20261116.md`](../docs/phase3/FUTURE_HARM_UNSEEN_FAMILY_PROTOCOL_20261116.md) | **Bounded**: Time-only AUC dropped from 0.998 to 0.875 when onset times were balanced; monitor features improved Brier score. |
| **Causal Attribution & Identifiability** | [`docs/phase3/PHASE3_REMAINING_WEAKNESSES_20261116.md`](../docs/phase3/PHASE3_REMAINING_WEAKNESSES_20261116.md) | **Fundamental Boundary**: Feature localization reflects descriptive statistical correlation/KS rankings, not causal attribution. Unlabeled accuracy under arbitrary shift remains non-identifiable. |

---

## 6. Coursework submissions and historical packages

- [All four phases](../docs/submissions/) lists artifacts and submission status.
- [Phase 3 package and receipt](../docs/submissions/phase3/) includes the reproduction link.
- [Phase 4 submission](../docs/submissions/phase4/) holds the submitted PDF and source ZIP. Work on the editable copy in [paper](../paper/).
- [Historical package retrieval](../docs/HISTORICAL_ARTIFACTS.md) covers the packages formerly in output/phase3.

---

## 7. Dataset Provenance & Locked Manifests

- **ELEC2 (Australian New South Wales Electricity Market)**:
  - Config: [`configs/datasets/elec2.yaml`](../configs/datasets/elec2.yaml)
  - Verified Manifest: [`data/manifests/elec2_verified_v2_manifest.json`](../data/manifests/elec2_verified_v2_manifest.json)
- **Covertype (Forest Cover Type / USGS)**:
  - Config: [`configs/datasets/covtype.yaml`](../configs/datasets/covtype.yaml)
  - Verified Manifest: [`data/manifests/covtype_subset_verified_v2_manifest.json`](../data/manifests/covtype_subset_verified_v2_manifest.json)
- **NOAA GSOD Weather (1949–1999)**:
  - Config: [`configs/datasets/weather.yaml`](../configs/datasets/weather.yaml)
  - Manifest: [`data/processed/weather_noaa_1949_1999_v1/weather_manifest.json`](../data/processed/weather_noaa_1949_1999_v1/weather_manifest.json)
  - Provenance Audit: [`docs/phase3/WEATHER_PROVENANCE_AUDIT_20261116.md`](../docs/phase3/WEATHER_PROVENANCE_AUDIT_20261116.md)
- **UCI Gas Sensor Drift (Array 224)**:
  - Source Readme: [`data/external/uci224_source/README.md`](../data/external/uci224_source/README.md)
  - Compressed Source Data: [`data/external/uci224_source/uci224.zip`](../data/external/uci224_source/uci224.zip)

---

## 8. Excluded Intermediate Caches (Local-Only Artifacts)

The following intermediate bulky objects are deliberately omitted from the Git repository (they are not needed for research replication, as all code, configs, manifests, summary metrics, and final compiled reports are tracked):
- Monolithic Archive Caches (`> 100 MB`, e.g. `23I-2581_23I-2604_Phase3_Research_Full.zip`, 1.14 GB): All internal academic reports, PDFs, markdown logs, protocols, and unique figures have been extracted and tracked under [`reports/phase3/v2_archive_extracted/`](../reports/phase3/v2_archive_extracted/).
- Raw Prediction Arrays (`.npz`): Intermediate array predictions generated by simulation runs.
- High-Frequency Trace Logs (`online_batches.csv`, `selected_label_log.csv` > 100 MB): Intermediate per-sample streaming dumps; aggregate statistical metrics and hypothesis tests are preserved in their respective experiment folders.
