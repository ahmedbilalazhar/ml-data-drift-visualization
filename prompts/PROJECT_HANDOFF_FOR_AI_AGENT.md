# Project Location Map & Pointer Registry for AI Agent / Collaborator

This document is the authoritative **pointer registry** for the research project **Explainable Visualization and Monitoring of ML Data Drift (IDM-Viz)** (Ahmed Bilal 23I-2581 & Umar Behram 23I-2604).
It maps the exact file locations of all scientific findings, implementation code, protocols, experimental registries, reports, weakness dispositions, and dataset artifacts.

---

## 1. Quick Navigation & Source of Truth

- **Primary Repository Entrypoint & Layout**: [`README.md`](../README.md)
- **Experiment Pointers & Metric Locations**: [`EXPERIMENTS.md`](../EXPERIMENTS.md)
- **Phase 3 Run Index & Registry**: [`PHASE3_RUN_INDEX.md`](PHASE3_RUN_INDEX.md)
- **Project Goals & Evolution**: [`plan.md`](../plan.md) and [`docs/research_notes/project-plan.md`](../docs/research_notes/project-plan.md)
- **Architecture & Technical Design**: [`docs/architecture/research-project-architecture-and-roadmap.md`](../docs/architecture/research-project-architecture-and-roadmap.md)
- **Research Charter & Questions**: [`experiments/protocols/research_charter.md`](../experiments/protocols/research_charter.md)
- **Run Execution Log**: [`experiments/registries/runs.csv`](../experiments/registries/runs.csv)

---

## 2. Codebase & Scientific Implementation Map

The Python implementation package is located under `src/drift_monitoring/`:

- **Drift Detectors (Univariate KS, Wasserstein, JS, ADWIN-style, RBF-MMD, Joint Logistic)**:
  - Implementation: [`src/drift_monitoring/detectors/drift_detectors.py`](../src/drift_monitoring/detectors/drift_detectors.py)
  - Unit tests: [`tests/unit/test_p0_detectors.py`](../tests/unit/test_p0_detectors.py), [`tests/unit/test_p2_detectors.py`](../tests/unit/test_p2_detectors.py)
- **Explanations, Localization & DDE Scoring**:
  - Implementation: [`src/drift_monitoring/explanations/localization.py`](../src/drift_monitoring/explanations/localization.py)
  - Synthetic Demo: [`scripts/demo_dde.py`](../scripts/demo_dde.py)
- **Frozen Models & Calibrators (Isotonic, Platt, Logistic Recalibration)**:
  - Implementation: [`src/drift_monitoring/models/frozen_models.py`](../src/drift_monitoring/models/frozen_models.py)
  - Unit tests: [`tests/unit/test_t10_calibration.py`](../tests/unit/test_t10_calibration.py)
- **Orchestration, Timing Windows & Evaluation Harness**:
  - Implementation: [`src/drift_monitoring/orchestration/runner.py`](../src/drift_monitoring/orchestration/runner.py)
  - Real Data Loader: [`src/drift_monitoring/data/real.py`](../src/drift_monitoring/data/real.py)
- **Aligned Views & Heatmap Visualization**:
  - Implementation: [`src/drift_monitoring/visualization/aligned_views.py`](../src/drift_monitoring/visualization/aligned_views.py)
  - Export Utility: [`scripts/export_heatmap.py`](../scripts/export_heatmap.py)
- **Phase 3 Core Simulation, Replay & Evaluation**:
  - Core Modules: [`src/drift_monitoring/phase3/core.py`](../src/drift_monitoring/phase3/core.py), [`src/drift_monitoring/phase3/evaluation.py`](../src/drift_monitoring/phase3/evaluation.py), [`src/drift_monitoring/phase3/adaptation.py`](../src/drift_monitoring/phase3/adaptation.py), [`src/drift_monitoring/phase3/replay.py`](../src/drift_monitoring/phase3/replay.py)
  - Invariant Contracts & Tests: [`tests/phase3/`](../tests/phase3/)

---

## 3. Findings, Reports & Manuscripts Directory

Over 148 compiled research PDFs, LaTeX sources, and standalone reports are tracked in Git:

### A. Academic Manuscripts & Submission Reports
- **Phase 1 Literature & Drift Report**:
  - LaTeX Manuscript: [`reports/manuscript/phase1/Phase1_Drift_Report.tex`](../reports/manuscript/phase1/Phase1_Drift_Report.tex)
  - Compiled PDF: [`reports/manuscript/phase1/Phase1_Drift_Report.pdf`](../reports/manuscript/phase1/Phase1_Drift_Report.pdf)
  - Literature Review: [`reports/literature/deep-research-report.md`](../reports/literature/deep-research-report.md) & [`deep-research-report-phase1.pdf`](../reports/literature/deep-research-report-phase1.pdf)
- **Phase 2 Research Report & Followup**:
  - Review Notes: [`reports/manuscript/phase2_review_notes.md`](../reports/manuscript/phase2_review_notes.md)
  - Submission Document: [`docs/submissions/DAV_Phase2_Revised_Report.md`](../docs/submissions/DAV_Phase2_Revised_Report.md) & [`docs/23i-2604_23i-2581DAV-phase2.docx`](../docs/23i-2604_23i-2581DAV-phase2.docx)
- **Phase 3 Comprehensive Manuscript & Evidence**:
  - Core V2 Research Report: [`reports/phase3/v2_archive_extracted/Report.md`](../reports/phase3/v2_archive_extracted/Report.md) & [`Report.pdf`](../reports/phase3/v2_archive_extracted/Report.pdf)
  - Adaptation & Retraining Report: [`reports/phase3/v2_archive_extracted/Adaptation.md`](../reports/phase3/v2_archive_extracted/Adaptation.md) & [`Adaptation.pdf`](../reports/phase3/v2_archive_extracted/Adaptation.pdf)
  - Decisions Log: [`reports/phase3/v2_archive_extracted/Decisions.md`](../reports/phase3/v2_archive_extracted/Decisions.md)
  - Protocol Specification: [`reports/phase3/v2_archive_extracted/Protocol.md`](../reports/phase3/v2_archive_extracted/Protocol.md)
  - Clean Reproduction Instructions: [`reports/phase3/v2_archive_extracted/Reproduce.md`](../reports/phase3/v2_archive_extracted/Reproduce.md)
  - Interactive Replay Dashboard: [`reports/phase3/v2_final/23I-2581_23I-2604_Phase3_Evidence_Replay.html`](../reports/phase3/v2_final/23I-2581_23I-2604_Phase3_Evidence_Replay.html)

### B. Detailed Phase 3 Experiment Reports
- **Decision Strengthening (v2–v5)**: [`reports/phase3/decision_strengthening_v5/REPORT.pdf`](../reports/phase3/decision_strengthening_v5/REPORT.pdf)
- **UCI Gas Sensor (Split Model Sensitivity & Exact Intervals)**:
  - Exact Intervals: [`reports/phase3/uci224_exact_intervals_20261116/UCI224_EXACT_INTERVALS.pdf`](../reports/phase3/uci224_exact_intervals_20261116/UCI224_EXACT_INTERVALS.pdf)
  - Split Sensitivity: [`reports/phase3/uci224_split_model_sensitivity_20261116_v2/UCI224_SPLIT_MODEL_SENSITIVITY.pdf`](../reports/phase3/uci224_split_model_sensitivity_20261116_v2/UCI224_SPLIT_MODEL_SENSITIVITY.pdf)
  - Delayed Replay: [`reports/phase3/uci224_delayed_replay_20261116_v5/REPORT.pdf`](../reports/phase3/uci224_delayed_replay_20261116_v5/REPORT.pdf)
- **ELEC2 Calibration Family & Recalibration**:
  - Calibration Family: [`reports/phase3/elec2_calibration_family_20261116_v1/ELEC2_CALIBRATION_FAMILY.pdf`](../reports/phase3/elec2_calibration_family_20261116_v1/ELEC2_CALIBRATION_FAMILY.pdf)
  - Delayed Recalibration: [`reports/phase3/elec2_recalibration_20261116/ELEC2_RECALIBRATION.pdf`](../reports/phase3/elec2_recalibration_20261116/ELEC2_RECALIBRATION.pdf)
- **NOAA Weather Experiments**:
  - Holdout & Transfer Timeline: [`reports/phase3/weather_noaa_1949_1999_v2/REPORT.pdf`](../reports/phase3/weather_noaa_1949_1999_v2/REPORT.pdf)
  - Selective Label Replay: [`reports/phase3/noaa_selective_label_replay_20261117_v3/NOAA_SELECTIVE_LABEL_REPLAY.pdf`](../reports/phase3/noaa_selective_label_replay_20261117_v3/NOAA_SELECTIVE_LABEL_REPLAY.pdf)
  - Training Uncertainty: [`reports/phase3/noaa_training_uncertainty_20261117_v3/NOAA_TRAINING_UNCERTAINTY.pdf`](../reports/phase3/noaa_training_uncertainty_20261117_v3/NOAA_TRAINING_UNCERTAINTY.pdf)
- **Future Harm Forecasting & Unseen Families**:
  - Forecast Evaluation: [`reports/phase3/future_harm_forecast_20261116_v10/FUTURE_HARM_FORECAST.pdf`](../reports/phase3/future_harm_forecast_20261116_v10/FUTURE_HARM_FORECAST.pdf)
  - Balanced Onset Unseen Family: [`reports/phase3/future_harm_unseen_family_balanced_20261116_v1/FUTURE_HARM_UNSEEN_FAMILY.pdf`](../reports/phase3/future_harm_unseen_family_balanced_20261116_v1/FUTURE_HARM_UNSEEN_FAMILY.pdf)
- **Large-Seed Robustness & Adaptation Schedule**:
  - Robustness Report: [`reports/phase3/large_seed_robustness_final/23I-2581_23I-2604_Phase3_Robustness_Report.pdf`](../reports/phase3/large_seed_robustness_final/23I-2581_23I-2604_Phase3_Robustness_Report.pdf)
  - Adaptation Schedule Audit: [`reports/phase3/adaptation_schedule_audit_20261118_v19/adaptation_schedule_audit.pdf`](../reports/phase3/adaptation_schedule_audit_20261118_v19/adaptation_schedule_audit.pdf)
  - Capacity Control: [`reports/phase3/adaptation_capacity_control_20261118_v9/adaptation_capacity_control.pdf`](../reports/phase3/adaptation_capacity_control_20261118_v9/adaptation_capacity_control.pdf)
  - Seasonal Stress: [`reports/phase3/seasonal_stress_20261118_v1/REPORT.pdf`](../reports/phase3/seasonal_stress_20261118_v1/REPORT.pdf)

---

## 4. Empirical Weaknesses & Methodological Boundaries

The project maintains an explicit weakness disposition log at [`docs/phase3/PHASE3_REMAINING_WEAKNESSES_20261116.md`](../docs/phase3/PHASE3_REMAINING_WEAKNESSES_20261116.md). Here is where each investigated weakness is located:

| Weakness Area | Pointer to Evidence & Implementation | Observed Empirical Status |
| :--- | :--- | :--- |
| **Label Leakage Prior to Release** | [`docs/phase3/E3_DELAY_AUDIT.md`](../docs/phase3/E3_DELAY_AUDIT.md), [`tests/phase3/test_scientific_contract.py`](../tests/phase3/test_scientific_contract.py) | **Resolved in code**: Evaluator and online loops strictly separated via release queues. |
| **Marginal Shift Confound in C6** | [`docs/phase3/DEPENDENCE_EXTENSION_PROTOCOL_20261116.md`](../docs/phase3/DEPENDENCE_EXTENSION_PROTOCOL_20261116.md), [`reports/phase3/dependence_extension_20261116/`](../reports/phase3/dependence_extension_20261116/) | **Characterized**: Targeted max-correlation detector achieved 30/30 detection; stationary false alarms increased from 0.83 to 1.70 per 24 batches. |
| **Normal Interval Approximation** | [`docs/phase3/UCI224_EXACT_INTERVAL_EXTENSION_20261116.md`](../docs/phase3/UCI224_EXACT_INTERVAL_EXTENSION_20261116.md) | **Evaluated**: Exact Poisson-binomial intervals are narrower (0.0234 vs 0.0258), but both failed to cover 0/9 cohorts due to miscalibration under shift. |
| **Delayed Recalibration Transfer** | [`docs/phase3/ELEC2_CALIBRATION_FAMILY_EXTENSION_20261116.md`](../docs/phase3/ELEC2_CALIBRATION_FAMILY_EXTENSION_20261116.md) | **Characterized Failure**: Isotonic and Platt updates reduced signed bias but failed to reduce MAE or ECE on ELEC2 stream. |
| **Prospective Harm Prediction** | [`docs/phase3/FUTURE_HARM_UNSEEN_FAMILY_PROTOCOL_20261116.md`](../docs/phase3/FUTURE_HARM_UNSEEN_FAMILY_PROTOCOL_20261116.md) | **Bounded**: Time-only AUC dropped from 0.998 to 0.875 when onset times were balanced; monitor features improved Brier score. |
| **Causal Attribution & Identifiability** | [`docs/phase3/PHASE3_REMAINING_WEAKNESSES_20261116.md`](../docs/phase3/PHASE3_REMAINING_WEAKNESSES_20261116.md) | **Open Limit**: Drift localization remains descriptive (correlation/KS rankings), not causal. Unlabeled accuracy under arbitrary shift is mathematically non-identifiable. |

---

## 5. Artifacts, Submission Bundles & Archives

- **Official Verified Phase 3 Submission Package**:
  - File: [`output/phase3/23I-2581_23I-2604_Phase3.zip`](../output/phase3/23I-2581_23I-2604_Phase3.zip) (10.35 MB)
  - Receipt: [`output/phase3/23I-2581_23I-2604_Phase3.receipt.json`](../output/phase3/23I-2581_23I-2604_Phase3.receipt.json)
- **Research Complete Package**:
  - File: [`output/phase3/23I-2581_23I-2604_Phase3_Research_Complete.zip`](../output/phase3/23I-2581_23I-2604_Phase3_Research_Complete.zip) (17.59 MB)
  - Receipt: [`output/phase3/23I-2581_23I-2604_Phase3_Research_Complete.receipt.json`](../output/phase3/23I-2581_23I-2604_Phase3_Research_Complete.receipt.json)
- **Clean Submission Baseline Archive**:
  - File: [`output/phase3/IDM_Viz_Phase3_v1_submission.zip`](../output/phase3/IDM_Viz_Phase3_v1_submission.zip) (4.90 MB)
- **Day-Wise Validated Packages (Days 1–3)**:
  - Day 1: [`output/phase3/IDM_Viz_Phase3_Day1_v1.1_Validated.zip`](../output/phase3/IDM_Viz_Phase3_Day1_v1.1_Validated.zip)
  - Day 2: [`output/phase3/IDM_Viz_Phase3_Day2_External_Evaluation_v1.3_Validated.zip`](../output/phase3/IDM_Viz_Phase3_Day2_External_Evaluation_v1.3_Validated.zip)
  - Day 3: [`output/phase3/IDM_Viz_Phase3_Day3_Final_Research_Improvement_v1.4_Validated.zip`](../output/phase3/IDM_Viz_Phase3_Day3_Final_Research_Improvement_v1.4_Validated.zip)
- **Addenda Manifest & Audit Files**:
  - Addendum Manifest: [`ADDENDUM_FILE_MANIFEST.json`](../ADDENDUM_FILE_MANIFEST.json)
  - Addendum Readme: [`ADDENDUM_README.md`](../ADDENDUM_README.md)
  - Status Register: [`output/phase3/IDM_Viz_Phase3_Addendum_Package_Status_20261117.json`](../output/phase3/IDM_Viz_Phase3_Addendum_Package_Status_20261117.json)
- **Monolithic Caches (> 100 MB, retained locally on laptop)**:
  - `output/phase3/23I-2581_23I-2604_Phase3_Research_Full.zip` (1.14 GB): Raw `.npz` arrays. All constituent reports, PDFs, decisions, and protocol files are extracted and tracked under [`reports/phase3/v2_archive_extracted/`](../reports/phase3/v2_archive_extracted/).

---

## 6. Dataset Provenance & Locked Manifests

- **ELEC2 (Australian New South Wales Electricity Market)**:
  - Config: [`configs/datasets/elec2.yaml`](../configs/datasets/elec2.yaml)
  - Manifest: [`data/manifests/elec2_verified_v2_manifest.json`](../data/manifests/elec2_verified_v2_manifest.json)
- **Covertype (Forest Cover Type / USGS)**:
  - Config: [`configs/datasets/covtype.yaml`](../configs/datasets/covtype.yaml)
  - Manifest: [`data/manifests/covtype_subset_verified_v2_manifest.json`](../data/manifests/covtype_subset_verified_v2_manifest.json)
- **NOAA GSOD Weather (1949–1999)**:
  - Config: [`configs/datasets/weather.yaml`](../configs/datasets/weather.yaml)
  - Audit Notes: [`docs/phase3/WEATHER_PROVENANCE_AUDIT_20261116.md`](../docs/phase3/WEATHER_PROVENANCE_AUDIT_20261116.md)
- **UCI Gas Sensor Drift (Array 224)**:
  - Source Readme: [`data/external/uci224_source/README.md`](../data/external/uci224_source/README.md)
  - Compressed Inputs: [`data/external/uci224_source/uci224_curl.zip`](../data/external/uci224_source/uci224_curl.zip)
