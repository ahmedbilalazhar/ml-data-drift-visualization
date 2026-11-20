# Experiment results and reproduction

Start with this page to find an experiment's measured results. The pilot section maps the original course experiments. The Phase 3 section covers the retained evaluation, replication, extension, and audit directories. Use the linked protocol and report to interpret the numbers.

CSV previews show the range of recorded values within a named table, across its conditions and methods. They are navigation aids and should not be cited as pooled estimates, uncertainty intervals, or comparisons between runs. Open the table to select the correct rows and denominators.

## Pilot experiments

| Experiment | Reported result | Evidence |
| --- | --- | --- |
| E0 integrity | Pilot execution took about 25.5 s and used 33 MB. See the registry for recorded completion and per-scenario evidence for S0–S7. | [runs.csv](experiments/registries/runs.csv) · [S0 evidence](results/runs/E0-S0-s0.evidence.csv) · [S7 evidence](results/runs/E0-S7-s0.evidence.csv) |
| E1 detection | S1–S6 detected at delay 0–1 on the pilot threshold. S0 produced 1 false alert in 12 windows. S7 was not detected under conditional shift and produced one alert, the same count as S0. | [e1_detection.csv](results/aggregates/e1_detection.csv) · [e1_compare.csv](results/aggregates/e1_compare.csv) |
| E2 localization | Pilot recall and AP were 1.0 in scenarios with a nonempty ground-truth feature set. Precision at 3 ranged from 0.333 to 0.667. | [e2_localization.csv](results/aggregates/e2_localization.csv) |
| E3 impact and release delays | S6 switched from update at delay 0 to investigate at delay 2. At delay 5 the S7 harm was hidden from the decision policy. | [e3_delay.csv](results/aggregates/e3_delay.csv) · [e3_proxy.csv](results/aggregates/e3_proxy.csv) |
| E6 sensitivity | S2 alerted with 250, 500 and 1,000 row windows. KS maxima were 0.61, 0.54 and 0.54. | [e6_window.csv](results/aggregates/e6_window.csv) · [e6_refpolicy.csv](results/aggregates/e6_refpolicy.csv) |

Protocol cards are in [experiments/protocols/experiment_cards](experiments/protocols/experiment_cards/). The [pilot claim map](reports/progress/claim_to_evidence.md) explains limits. Supporting results include the [multiseed detection table](results/aggregates/multiseed_detection.csv), [multiseed localization table](results/aggregates/multiseed_localization.csv), and [real data burden table](results/aggregates/real_burden.csv).

## Phase 3 runs

Use the [agent handoff](prompts/PROJECT_HANDOFF_FOR_AI_AGENT.md) for the scientific questions and headline findings. The [run index](prompts/PHASE3_RUN_INDEX.md) lists preserved artifacts. The [Phase 3 protocols](docs/phase3/) and [Phase 3 reports](reports/phase3/) provide context and scope.

### adaptation_capacity_control_20261118_v2

[Result directory](results/phase3/adaptation_capacity_control_20261118_v2/) · [audit_receipt.json](results/phase3/adaptation_capacity_control_20261118_v2/audit_receipt.json)

- [seed_metrics.csv](results/phase3/adaptation_capacity_control_20261118_v2/seed_metrics.csv). 4,320 rows; grouped by available fields `task`, `delay`, `method`. Observed column values: `accuracy` 0.46337 to 0.81473, `post_onset_accuracy` 0.41986 to 0.89417, `macro_f1` 0.34289 to 0.73565.

### adaptation_capacity_control_confirmation_20261118_v1

[Result directory](results/phase3/adaptation_capacity_control_confirmation_20261118_v1/) · [audit_receipt.json](results/phase3/adaptation_capacity_control_confirmation_20261118_v1/audit_receipt.json)

- [seed_metrics.csv](results/phase3/adaptation_capacity_control_confirmation_20261118_v1/seed_metrics.csv). 1,440 rows; grouped by available fields `task`, `delay`, `method`. Observed column values: `accuracy` 0.47056 to 0.81527, `post_onset_accuracy` 0.42 to 0.89256, `macro_f1` 0.34356 to 0.7345.

### adaptation_final

[Result directory](results/phase3/adaptation_final/) · [validation.json](results/phase3/adaptation_final/validation.json) · [protocol_lock.json](results/phase3/adaptation_final/protocol_lock.json)

- [paired_bootstrap.csv](results/phase3/adaptation_final/paired_bootstrap.csv). 108 rows; grouped by available fields `task`, `condition`, `delay`, `regime`, `metric`. Observed column values: `mean` -0.0025917 to 0.72566.

### adaptation_reproduction

[Result directory](results/phase3/adaptation_reproduction/) · [validation.json](results/phase3/adaptation_reproduction/validation.json) · [protocol_lock.json](results/phase3/adaptation_reproduction/protocol_lock.json)

- [paired_bootstrap.csv](results/phase3/adaptation_reproduction/paired_bootstrap.csv). 108 rows; grouped by available fields `task`, `condition`, `delay`, `regime`, `metric`. Observed column values: `mean` -0.0025917 to 0.72566.

### adaptation_schedule_audit_20261118_v4

[Result directory](results/phase3/adaptation_schedule_audit_20261118_v4/) · [audit_receipt.json](results/phase3/adaptation_schedule_audit_20261118_v4/audit_receipt.json)

The accounting audit covered 720 seed-policy cells. Label availability at the final forecast was 47,500, 45,000 and 37,500 at delays 0, 5 and 20. See the [documented scope](docs/phase3/).

- [seed_bootstrap_summary.csv](results/phase3/adaptation_schedule_audit_20261118_v4/seed_bootstrap_summary.csv). 144 rows; grouped by available fields `delay`, `method`, `metric`. Observed column values: `mean` 0 to 22000.
- [seed_metrics.csv](results/phase3/adaptation_schedule_audit_20261118_v4/seed_metrics.csv). 720 rows; grouped by available fields `task`, `delay`, `method`. Observed column values: `accuracy` 0.38383 to 0.72852, `post_onset_accuracy` 0.27156 to 0.7285, `macro_f1` 0.38302 to 0.72567.

### adaptation_seed_replication_development

[Result directory](results/phase3/adaptation_seed_replication_development/) · [validation.json](results/phase3/adaptation_seed_replication_development/validation.json) · [protocol_lock.json](results/phase3/adaptation_seed_replication_development/protocol_lock.json)

- [batch24_label_budget_per_seed.csv](results/phase3/adaptation_seed_replication_development/batch24_label_budget_per_seed.csv). 30 rows; grouped by available fields `task`, `condition`, `delay`. See the linked table for comparison values.
- [paired_bootstrap.csv](results/phase3/adaptation_seed_replication_development/paired_bootstrap.csv). 36 rows; grouped by available fields `task`, `condition`, `delay`, `regime`, `metric`. Observed column values: `mean` -0.0066 to 0.02855.

### adwin_audit_baseline_20261117_v1

[Result directory](results/phase3/adwin_audit_baseline_20261117_v1/) · [protocol_lock.json](results/phase3/adwin_audit_baseline_20261117_v1/protocol_lock.json)

- [adwin_summary.csv](results/phase3/adwin_audit_baseline_20261117_v1/adwin_summary.csv). 24 rows; grouped by available fields `delay`. Observed column values: `budget_cohorts` 2 to 8, `budget_labels` 1000 to 4000, `unspent_budget_cohorts` 0 to 7.
- [adwin_per_seed.csv](results/phase3/adwin_audit_baseline_20261117_v1/adwin_per_seed.csv). 5,040 rows; grouped by available fields `condition`, `delay`, `policy`. Observed column values: `budget_cohorts` 2 to 8, `budget_labels` 1000 to 4000, `unspent_budget_cohorts` 0 to 7.

### adwin_midpoint_probe_sensitivity_20261118_v2

[Result directory](results/phase3/adwin_midpoint_probe_sensitivity_20261118_v2/) · [protocol_lock.json](results/phase3/adwin_midpoint_probe_sensitivity_20261118_v2/protocol_lock.json)

- [two_probe_summary.csv](results/phase3/adwin_midpoint_probe_sensitivity_20261118_v2/two_probe_summary.csv). 24 rows; grouped by available fields `delay`. Observed column values: `budget_cohorts` 2 to 8, `budget_labels` 1000 to 4000, `unspent_budget_cohorts` 0 to 6.
- [two_probe_per_seed.csv](results/phase3/adwin_midpoint_probe_sensitivity_20261118_v2/two_probe_per_seed.csv). 5,040 rows; grouped by available fields `condition`, `delay`, `policy`. Observed column values: `budget_cohorts` 2 to 8, `budget_labels` 1000 to 4000, `unspent_budget_cohorts` 0 to 6.

### adwin_two_probe_sensitivity_20261118_v1

[Result directory](results/phase3/adwin_two_probe_sensitivity_20261118_v1/) · [protocol_lock.json](results/phase3/adwin_two_probe_sensitivity_20261118_v1/protocol_lock.json)

- [two_probe_summary.csv](results/phase3/adwin_two_probe_sensitivity_20261118_v1/two_probe_summary.csv). 24 rows; grouped by available fields `delay`. Observed column values: `budget_cohorts` 2 to 8, `budget_labels` 1000 to 4000, `unspent_budget_cohorts` 0 to 6.
- [two_probe_per_seed.csv](results/phase3/adwin_two_probe_sensitivity_20261118_v1/two_probe_per_seed.csv). 5,040 rows; grouped by available fields `condition`, `delay`, `policy`. Observed column values: `budget_cohorts` 2 to 8, `budget_labels` 1000 to 4000, `unspent_budget_cohorts` 0 to 6.

### audit_policy_matrix_20261117_v1

[Result directory](results/phase3/audit_policy_matrix_20261117_v1/) · [protocol_lock.json](results/phase3/audit_policy_matrix_20261117_v1/protocol_lock.json)

- [policy_summary.csv](results/phase3/audit_policy_matrix_20261117_v1/policy_summary.csv). 1,008 rows; grouped by available fields `condition`, `delay`, `policy`. Observed column values: `budget_cohorts` 2 to 8, `budget_labels` 1000 to 4000, `unspent_budget_cohorts` 0 to 7.2667.
- [policy_per_seed.csv](results/phase3/audit_policy_matrix_20261117_v1/policy_per_seed.csv). 30,240 rows; grouped by available fields `condition`, `delay`, `policy`. Observed column values: `budget_cohorts` 2 to 8, `budget_labels` 1000 to 4000, `unspent_budget_cohorts` 0 to 8.

### autonomous_24h_20261116

[Result directory](results/phase3/autonomous_24h_20261116/) · [STATUS.md](results/phase3/autonomous_24h_20261116/STATUS.md)

Review the linked status or receipt for what was executed and which evidence is available.

### autonomous_extension_20261116

[Result directory](results/phase3/autonomous_extension_20261116/) · [STATUS.md](results/phase3/autonomous_extension_20261116/STATUS.md)

Review the linked status or receipt for what was executed and which evidence is available.

### cluster_mapping_sensitivity_20261117_v2

[Result directory](results/phase3/cluster_mapping_sensitivity_20261117_v2/) · [audit_receipt.json](results/phase3/cluster_mapping_sensitivity_20261117_v2/audit_receipt.json)

- [cluster_mapping_summary.csv](results/phase3/cluster_mapping_sensitivity_20261117_v2/cluster_mapping_summary.csv). 9 rows. Observed column values: `ari_mean` 0.066718 to 0.098465, `macro_f1_mean` 0.22552 to 0.3698, `region2_ap_mean` 0.20733 to 0.24409.

### completion_audit_20261115

[Result directory](results/phase3/completion_audit_20261115/) · [audit_receipt.json](results/phase3/completion_audit_20261115/audit_receipt.json) · [summary.json](results/phase3/completion_audit_20261115/external_reproduction/summary.json) · [protocol_lock.json](results/phase3/completion_audit_20261115/external_reproduction/protocol_lock.json)

- [class_metrics.csv](results/phase3/completion_audit_20261115/external_reproduction/class_metrics.csv). 60 rows; grouped by available fields `batch`. Observed column values: `count` 0 to 744, `precision` 0 to 1, `recall` 0 to 1.
- [cohort_metrics.csv](results/phase3/completion_audit_20261115/external_reproduction/cohort_metrics.csv). 10 rows; grouped by available fields `batch`. Observed column values: `accuracy` 0.19388 to 0.94382, `macro_f1_fixed_six_classes` 0.16564 to 0.94867, `estimated_accuracy` 0.97529 to 0.97713.

### dde_subset_audit_20261118_v4

[Result directory](results/phase3/dde_subset_audit_20261118_v4/)

- [evaluation_seed_8732201_five_region_d128.csv](results/phase3/dde_subset_audit_20261118_v4/evaluation_seed_8732201_five_region_d128.csv). 280 rows; grouped by available fields `task`, `batch`, `method`. Observed column values: `hidden_feature_count` 96, `detected_anchor_count` 0, `target_feature_count` 0 to 2.

### decision_strengthening_v2

[Result directory](results/phase3/decision_strengthening_v2/) · [validation.json](results/phase3/decision_strengthening_v2/validation.json) · [protocol_lock.json](results/phase3/decision_strengthening_v2/protocol_lock.json)

- [localization_summary.csv](results/phase3/decision_strengthening_v2/localization_summary.csv). 2 rows; grouped by available fields `condition`. Observed column values: `precision_at_1` 0.9875 to 1, `recall_at_3` 0.99375 to 1, `feature_flag_recall` 0.97292 to 1.
- [policy_summary.csv](results/phase3/decision_strengthening_v2/policy_summary.csv). 210 rows; grouped by available fields `condition`, `delay`, `policy`. Observed column values: `budget` 2 to 4, `harmful_batches` 0.2 to 16.167, `harmful_audits` 0 to 4.

### delay100_availability_20261116_v1

[Result directory](results/phase3/delay100_availability_20261116_v1/) · [validation.json](results/phase3/delay100_availability_20261116_v1/validation.json)

- [per_stream_availability.csv](results/phase3/delay100_availability_20261116_v1/per_stream_availability.csv). 210 rows; grouped by available fields `condition`. See the linked table for comparison values.

### dependence_extension_20261116_final_v2

[Result directory](results/phase3/dependence_extension_20261116_final_v2/) · [validation.json](results/phase3/dependence_extension_20261116_final_v2/validation.json)

- [condition_summary.csv](results/phase3/dependence_extension_20261116_final_v2/condition_summary.csv). 7 rows; grouped by available fields `condition`. Observed column values: `onset_detection_rate` 0.4 to 1, `any_time_detection_rate` 0.83333 to 1, `pre_onset_false_alarm_rate` 0.4.
- [bootstrap_summary.csv](results/phase3/dependence_extension_20261116_final_v2/bootstrap_summary.csv). 1 rows; grouped by available fields `condition`, `metric`. Observed column values: `mean_difference` 0.46667.

### elec2_calibration_family_20261116_v1

[Result directory](results/phase3/elec2_calibration_family_20261116_v1/) · [validation.json](results/phase3/elec2_calibration_family_20261116_v1/validation.json)

- [summary.csv](results/phase3/elec2_calibration_family_20261116_v1/summary.csv). 9 rows; grouped by available fields `delay`, `method`. Observed column values: `equal_batch_mae` 0.058571 to 0.066639, `equal_batch_bias` -0.010622 to -0.00047451, `mean_correctness_brier` 0.21518 to 0.21842.

### elec2_platt_recalibration_20261116_v1

[Result directory](results/phase3/elec2_platt_recalibration_20261116_v1/) · [protocol_lock.json](results/phase3/elec2_platt_recalibration_20261116_v1/protocol_lock.json)

- [delay_0_fit_audit.csv](results/phase3/elec2_platt_recalibration_20261116_v1/delay_0_fit_audit.csv). 55 rows; grouped by available fields `delay`, `batch`. See the linked table for comparison values.

### elec2_recalibration_20261116_v1

[Result directory](results/phase3/elec2_recalibration_20261116_v1/) · [validation.json](results/phase3/elec2_recalibration_20261116_v1/evaluation/validation.json) · [protocol_lock.json](results/phase3/elec2_recalibration_20261116_v1/protocol_lock.json)

- [summary.csv](results/phase3/elec2_recalibration_20261116_v1/evaluation/summary.csv). 6 rows; grouped by available fields `delay`, `method`. Observed column values: `equal_batch_mae` 0.058571 to 0.063719, `equal_batch_bias` -0.010622 to -0.00047451, `row_weighted_brier` 0.21518 to 0.2164.

### elec2_temporal_20261116_v2

[Result directory](results/phase3/elec2_temporal_20261116_v2/) · [summary.json](results/phase3/elec2_temporal_20261116_v2/evaluation/summary.json) · [protocol_lock.json](results/phase3/elec2_temporal_20261116_v2/protocol_lock.json)

- [delay_summary.csv](results/phase3/elec2_temporal_20261116_v2/evaluation/delay_summary.csv). 3 rows; grouped by available fields `delay`. See the linked table for comparison values.
- [per_class_metrics.csv](results/phase3/elec2_temporal_20261116_v2/evaluation/per_class_metrics.csv). 110 rows; grouped by available fields `batch`. Observed column values: `precision` 0.3224 to 0.9375, `recall` 0.13253 to 0.97544, `f1` 0.22148 to 0.85673.

### external_uci224_day2_metadata_v3

[Result directory](results/phase3/external_uci224_day2_metadata_v3/) · [summary.json](results/phase3/external_uci224_day2_metadata_v3/summary.json) · [protocol_lock.json](results/phase3/external_uci224_day2_metadata_v3/protocol_lock.json)

- [class_metrics.csv](results/phase3/external_uci224_day2_metadata_v3/class_metrics.csv). 60 rows; grouped by available fields `batch`. Observed column values: `count` 0 to 744, `precision` 0 to 1, `recall` 0 to 1.
- [cohort_metrics.csv](results/phase3/external_uci224_day2_metadata_v3/cohort_metrics.csv). 10 rows; grouped by available fields `batch`. Observed column values: `accuracy` 0.19388 to 0.94382, `macro_f1_fixed_six_classes` 0.16564 to 0.94867, `estimated_accuracy` 0.97529 to 0.97713.

### external_uci224_day2_sensitivity_v2

[Result directory](results/phase3/external_uci224_day2_sensitivity_v2/)

- [class_mix_sensitivity.csv](results/phase3/external_uci224_day2_sensitivity_v2/class_mix_sensitivity.csv). 72 rows; grouped by available fields `batch`. Observed column values: `accuracy` 0 to 1.

### external_uci224_day2_v2

[Result directory](results/phase3/external_uci224_day2_v2/) · [summary.json](results/phase3/external_uci224_day2_v2/summary.json) · [protocol_lock.json](results/phase3/external_uci224_day2_v2/protocol_lock.json)

- [class_metrics.csv](results/phase3/external_uci224_day2_v2/class_metrics.csv). 60 rows; grouped by available fields `batch`. Observed column values: `count` 0 to 744, `precision` 0 to 1, `recall` 0 to 1.
- [cohort_metrics.csv](results/phase3/external_uci224_day2_v2/cohort_metrics.csv). 10 rows; grouped by available fields `batch`. Observed column values: `accuracy` 0.19388 to 0.94382, `macro_f1_fixed_six_classes` 0.16564 to 0.94867, `estimated_accuracy` 0.97529 to 0.97713.

### external_uci224_exact_interval_20261116_v1

[Result directory](results/phase3/external_uci224_exact_interval_20261116_v1/) · [summary.json](results/phase3/external_uci224_exact_interval_20261116_v1/summary.json)

Exact and normal-approximation intervals each covered zero of nine cohorts. Their mean widths were 0.0234 and 0.0258 respectively. See the [documented scope](docs/phase3/).

- [cohort_interval_comparison.csv](results/phase3/external_uci224_exact_interval_20261116_v1/cohort_interval_comparison.csv). 9 rows; grouped by available fields `batch`. Observed column values: `accuracy` 0.19388 to 0.65839, `estimate` 0.97529 to 0.97687, `exact_width` 0.0096872 to 0.043478.

### future_harm_forecast_20261116_v1

[Result directory](results/phase3/future_harm_forecast_20261116_v1/) · [protocol_lock.json](results/phase3/future_harm_forecast_20261116_v1/protocol_lock.json)

- [descriptive_threshold_metrics.csv](results/phase3/future_harm_forecast_20261116_v1/descriptive_threshold_metrics.csv). 6 rows; grouped by available fields `model`. Observed column values: `accuracy` 0.56416 to 0.77552, `precision` 0 to 1, `recall` 0 to 0.67307.

### future_harm_unseen_family_20261116_v1

[Result directory](results/phase3/future_harm_unseen_family_20261116_v1/) · [validation.json](results/phase3/future_harm_unseen_family_20261116_v1/validation.json) · [protocol_lock.json](results/phase3/future_harm_unseen_family_20261116_v1/protocol_lock.json)

- [future_targets.csv](results/phase3/future_harm_unseen_family_20261116_v1/evaluator_truth/future_targets.csv). 1,710 rows; grouped by available fields `condition`, `delay`, `batch`. Observed column values: `future_accuracy` 0.5308 to 0.7396, `reference_accuracy` 0.695 to 0.7435, `future_harm` 0 to 1.

### future_harm_unseen_family_balanced_20261116_v1

[Result directory](results/phase3/future_harm_unseen_family_balanced_20261116_v1/) · [validation.json](results/phase3/future_harm_unseen_family_balanced_20261116_v1/validation.json) · [protocol_lock.json](results/phase3/future_harm_unseen_family_balanced_20261116_v1/protocol_lock.json)

- [future_targets.csv](results/phase3/future_harm_unseen_family_balanced_20261116_v1/evaluator_truth/future_targets.csv). 1,710 rows; grouped by available fields `condition`, `delay`, `batch`. Observed column values: `future_accuracy` 0.5308 to 0.7464, `reference_accuracy` 0.695 to 0.7435, `future_harm` 0 to 1.

### horizon_metadata_strengthening_v2

[Result directory](results/phase3/horizon_metadata_strengthening_v2/)

- [horizon_summary.csv](results/phase3/horizon_metadata_strengthening_v2/horizon_summary.csv). 84 rows; grouped by available fields `condition`, `horizon`, `method`. Observed column values: `mae` 0.014799 to 0.3792, `bias` -0.070465 to 0.37649.
- [horizon_per_seed.csv](results/phase3/horizon_metadata_strengthening_v2/horizon_per_seed.csv). 2,520 rows; grouped by available fields `condition`, `horizon`, `method`. Observed column values: `mae` 0.0090617 to 0.40505, `bias` -0.10558 to 0.40505.

### large_seed_robustness_final

[Result directory](results/phase3/large_seed_robustness_final/) · [validation.json](results/phase3/large_seed_robustness_final/validation.json) · [protocol_lock.json](results/phase3/large_seed_robustness_final/protocol_lock.json)

- [seed_bootstrap_summary.csv](results/phase3/large_seed_robustness_final/seed_bootstrap_summary.csv). 1,666 rows; grouped by available fields `task`, `condition`, `regime`, `model`, `metric`. Observed column values: `mean` -0.04246 to 23.901.

### localization_audit_20261117_v3

[Result directory](results/phase3/localization_audit_20261117_v3/) · [audit_receipt.json](results/phase3/localization_audit_20261117_v3/audit_receipt.json)

- [dependence_pair_summary.csv](results/phase3/localization_audit_20261117_v3/dependence_pair_summary.csv). 9 rows; grouped by available fields `task`. Observed column values: `alarm_rate` 0.045833 to 1, `top_pair_accuracy` 0.013889 to 1.
- [region_localization_summary.csv](results/phase3/localization_audit_20261117_v3/region_localization_summary.csv). 18 rows. Observed column values: `region2_recall` 0.046536 to 0.21579, `region2_precision` 0.21532 to 0.2327, `coverage` 0.51518 to 1.

### matched_adaptation_diversity_20261118_v1

[Result directory](results/phase3/matched_adaptation_diversity_20261118_v1/) · [audit_receipt.json](results/phase3/matched_adaptation_diversity_20261118_v1/audit_receipt.json)

- [seed_metrics.csv](results/phase3/matched_adaptation_diversity_20261118_v1/seed_metrics.csv). 3,240 rows; grouped by available fields `task`, `delay`, `method`. Observed column values: `accuracy` 0.46337 to 0.81473, `post_onset_accuracy` 0.41986 to 0.89417, `macro_f1` 0.34289 to 0.73565.

### matched_adaptation_schedule_20261117_v2

[Result directory](results/phase3/matched_adaptation_schedule_20261117_v2/) · [audit_receipt.json](results/phase3/matched_adaptation_schedule_20261117_v2/audit_receipt.json)

- [seed_metrics.csv](results/phase3/matched_adaptation_schedule_20261117_v2/seed_metrics.csv). 540 rows; grouped by available fields `task`, `delay`, `method`. Observed column values: `accuracy` 0.38383 to 0.72852, `post_onset_accuracy` 0.27156 to 0.72847, `macro_f1` 0.38302 to 0.72567.

### noaa_reference_calibration_uncertainty_20261117_v2

[Result directory](results/phase3/noaa_reference_calibration_uncertainty_20261117_v2/) · [audit_receipt.json](results/phase3/noaa_reference_calibration_uncertainty_20261117_v2/audit_receipt.json)

- [uncertainty_summary.csv](results/phase3/noaa_reference_calibration_uncertainty_20261117_v2/uncertainty_summary.csv). 63 rows; grouped by available fields `batch`. Observed column values: `estimate_mean` 0.67052 to 0.86794, `estimate_upper` 0.67272 to 0.8823, `concern_mean` 3.4203e-15 to 0.99534.

### noaa_reference_calibration_uncertainty_block7_20261117_v1

[Result directory](results/phase3/noaa_reference_calibration_uncertainty_block7_20261117_v1/) · [audit_receipt.json](results/phase3/noaa_reference_calibration_uncertainty_block7_20261117_v1/audit_receipt.json)

- [uncertainty_summary.csv](results/phase3/noaa_reference_calibration_uncertainty_block7_20261117_v1/uncertainty_summary.csv). 63 rows; grouped by available fields `batch`. Observed column values: `estimate_mean` 0.66986 to 0.86878, `estimate_upper` 0.67272 to 0.88151, `concern_mean` 2.0207e-14 to 0.99678.

### noaa_reference_calibration_uncertainty_block90_20261117_v1

[Result directory](results/phase3/noaa_reference_calibration_uncertainty_block90_20261117_v1/) · [audit_receipt.json](results/phase3/noaa_reference_calibration_uncertainty_block90_20261117_v1/audit_receipt.json)

- [uncertainty_summary.csv](results/phase3/noaa_reference_calibration_uncertainty_block90_20261117_v1/uncertainty_summary.csv). 63 rows; grouped by available fields `batch`. Observed column values: `estimate_mean` 0.66852 to 0.86794, `estimate_upper` 0.67272 to 0.88705, `concern_mean` 3.4707e-13 to 0.99087.

### noaa_selective_label_replay_20261117_v1

[Result directory](results/phase3/noaa_selective_label_replay_20261117_v1/) · [audit_receipt.json](results/phase3/noaa_selective_label_replay_20261117_v1/audit_receipt.json)

- [design_summary.csv](results/phase3/noaa_selective_label_replay_20261117_v1/design_summary.csv). 3 rows; grouped by available fields `policy`. Observed column values: `mean_selected_naive_accuracy` 0.54667 to 0.75211, `mean_estimation_error` -0.2039 to 0.0015362, `design_sd_selected_accuracy` 1.1108e-16 to 0.023978.

### noaa_training_uncertainty_20261117_v2

[Result directory](results/phase3/noaa_training_uncertainty_20261117_v2/) · [audit_receipt.json](results/phase3/noaa_training_uncertainty_20261117_v2/audit_receipt.json)

- [training_uncertainty_summary.csv](results/phase3/noaa_training_uncertainty_20261117_v2/training_uncertainty_summary.csv). 21 rows; grouped by available fields `batch`. Observed column values: `accuracy_mean` 0.67647 to 0.86517, `accuracy_upper` 0.69379 to 0.87278, `concern_mean` 2.7765e-14 to 0.99895.

### noaa_weather_1949_1999_v1

[Result directory](results/phase3/noaa_weather_1949_1999_v1/) · [run_contract.json](results/phase3/noaa_weather_1949_1999_v1/run_contract.json) · [run_summary.json](results/phase3/noaa_weather_1949_1999_v1/run_summary.json) · [protocol_lock.json](results/phase3/noaa_weather_1949_1999_v1/protocol_lock.json)

- [batch_metrics.csv](results/phase3/noaa_weather_1949_1999_v1/evaluator/batch_metrics.csv). 21 rows; grouped by available fields `batch`. Observed column values: `accuracy` 0.678 to 0.826, `macro_f1` 0.58908 to 0.72422, `balanced_accuracy` 0.5919 to 0.71575.

### seasonal_stress_evaluation_20261118_v1

[Result directory](results/phase3/seasonal_stress_evaluation_20261118_v1/) · [analysis_manifest.json](results/phase3/seasonal_stress_evaluation_20261118_v1/analysis/analysis_manifest.json) · [protocol_lock.json](results/phase3/seasonal_stress_evaluation_20261118_v1/protocol_lock.json)

Nuisance-only seasonality raised input alarm rates by roughly 0.744–0.768 across three synthetic families while paired mean accuracy deltas stayed within 0.000025 of zero. See the [documented scope](docs/phase3/).

- [seasonal_summary.csv](results/phase3/seasonal_stress_evaluation_20261118_v1/analysis/seasonal_summary.csv). 27 rows; grouped by available fields `task`, `metric`. Observed column values: `mean` -2.5e-05 to 0.81146.
- [seasonal_paired_per_seed.csv](results/phase3/seasonal_stress_evaluation_20261118_v1/analysis/seasonal_paired_per_seed.csv). 90 rows; grouped by available fields `task`. Observed column values: `accuracy_seasonal` 0.49996 to 0.72817, `accuracy_stationary` 0.4996 to 0.72852, `accuracy_delta` -0.0013958 to 0.00085417.

### seasonal_stress_reproduction_20261118_v1

[Result directory](results/phase3/seasonal_stress_reproduction_20261118_v1/) · [analysis_manifest.json](results/phase3/seasonal_stress_reproduction_20261118_v1/analysis/analysis_manifest.json) · [protocol_lock.json](results/phase3/seasonal_stress_reproduction_20261118_v1/protocol_lock.json)

Completed all 90 cells and exactly reproduced the compared seed-level outcome columns in the canonical seasonal stress run. See the [documented scope](docs/phase3/).

- [seasonal_summary.csv](results/phase3/seasonal_stress_reproduction_20261118_v1/analysis/seasonal_summary.csv). 27 rows; grouped by available fields `task`, `metric`. Observed column values: `mean` -2.5e-05 to 0.81146.
- [seasonal_paired_per_seed.csv](results/phase3/seasonal_stress_reproduction_20261118_v1/analysis/seasonal_paired_per_seed.csv). 90 rows; grouped by available fields `task`. Observed column values: `accuracy_seasonal` 0.49996 to 0.72817, `accuracy_stationary` 0.4996 to 0.72852, `accuracy_delta` -0.0013958 to 0.00085417.

### selective_label_audit_20261117_v2

[Result directory](results/phase3/selective_label_audit_20261117_v2/) · [audit_receipt.json](results/phase3/selective_label_audit_20261117_v2/audit_receipt.json)

- [seed_bootstrap_summary.csv](results/phase3/selective_label_audit_20261117_v2/seed_bootstrap_summary.csv). 52 rows; grouped by available fields `task`, `policy`, `metric`. Observed column values: `mean` 0.38695 to 95.297.

### strengthening_v3_evaluation_d8

[Result directory](results/phase3/strengthening_v3_evaluation_d8/) · [analysis_manifest.json](results/phase3/strengthening_v3_evaluation_d8/analysis/analysis_manifest.json) · [protocol_lock.json](results/phase3/strengthening_v3_evaluation_d8/protocol_lock.json)

- [delayed_verification_summary.csv](results/phase3/strengthening_v3_evaluation_d8/analysis/delayed_verification_summary.csv). 480 rows; grouped by available fields `task`, `delay`. Observed column values: `mean_accuracy_at_release` 0.27467 to 0.87897, `mean_evidence_age_batches` 0 to 100, `verified_harm_rate` 0 to 1.
- [detection_summary.csv](results/phase3/strengthening_v3_evaluation_d8/analysis/detection_summary.csv). 48 rows; grouped by available fields `task`. Observed column values: `stationary_false_alarm_rate` 0.034722 to 0.075, `post_intervention_detection_rate` 0.86667 to 1, `mean_detection_delay_batches` 0 to 26.931.

### strengthening_v3_evaluation_sensitivity_d32_d128

[Result directory](results/phase3/strengthening_v3_evaluation_sensitivity_d32_d128/) · [analysis_manifest.json](results/phase3/strengthening_v3_evaluation_sensitivity_d32_d128/analysis/analysis_manifest.json) · [protocol_lock.json](results/phase3/strengthening_v3_evaluation_sensitivity_d32_d128/protocol_lock.json)

All 1,440 scientific cells were present and hash-verified. A historical lock replacement PermissionError is documented in the reconciliation. See the [documented scope](docs/phase3/).

- [delayed_verification_summary.csv](results/phase3/strengthening_v3_evaluation_sensitivity_d32_d128/analysis/delayed_verification_summary.csv). 960 rows; grouped by available fields `task`, `delay`. Observed column values: `mean_accuracy_at_release` 0.2776 to 0.84606, `mean_evidence_age_batches` 0 to 100, `verified_harm_rate` 0 to 1.
- [detection_summary.csv](results/phase3/strengthening_v3_evaluation_sensitivity_d32_d128/analysis/detection_summary.csv). 96 rows; grouped by available fields `task`. Observed column values: `stationary_false_alarm_rate` 0.038889 to 0.054167, `post_intervention_detection_rate` 0.83333 to 1, `mean_detection_delay_batches` 0 to 24.963.

### uci224_delayed_replay_20261116_v2

[Result directory](results/phase3/uci224_delayed_replay_20261116_v2/) · [validation.json](results/phase3/uci224_delayed_replay_20261116_v2/evaluation/validation.json) · [protocol_lock.json](results/phase3/uci224_delayed_replay_20261116_v2/protocol_lock.json)

Across nine unequal cohorts, equal-cohort MAE was 0.545 frozen versus 0.169, 0.227 and 0.283 with delayed cumulative recalibration. See the [documented scope](docs/phase3/).

- [summary.csv](results/phase3/uci224_delayed_replay_20261116_v2/evaluation/summary.csv). 12 rows; grouped by available fields `method`. Observed column values: `delay_cohorts` 0 to 2, `equal_batch_mae` 0.1695 to 0.54466, `equal_batch_bias` 0.1695 to 0.54466.

### uci224_split_model_sensitivity_20261116_v2

[Result directory](results/phase3/uci224_split_model_sensitivity_20261116_v2/) · [validation.json](results/phase3/uci224_split_model_sensitivity_20261116_v2/evaluation/validation.json) · [protocol_lock.json](results/phase3/uci224_split_model_sensitivity_20261116_v2/protocol_lock.json)

- [per_class_metrics.csv](results/phase3/uci224_split_model_sensitivity_20261116_v2/evaluation/per_class_metrics.csv). 1,224 rows; grouped by available fields `model`, `batch`. Observed column values: `precision` 0 to 1, `recall` 0 to 1, `f1` 0 to 1.
- [per_cohort_metrics.csv](results/phase3/uci224_split_model_sensitivity_20261116_v2/evaluation/per_cohort_metrics.csv). 204 rows; grouped by available fields `model`, `batch`. Observed column values: `accuracy` 0.10544 to 0.91371, `macro_f1_fixed_six` 0.040404 to 0.74178, `raw_mean_confidence` 0.45143 to 0.99553.

### v2_final

[Result directory](results/phase3/v2_final/) · [validation.json](results/phase3/v2_final/validation.json) · [protocol_lock.json](results/phase3/v2_final/protocol_lock.json)

- [impact_summary.csv](results/phase3/v2_final/impact_summary.csv). 8,460 rows; grouped by available fields `task`, `condition`, `regime`, `model`, `method`, `metric`. Observed column values: `mean` -0.10405 to 1.
- [overall_summary.csv](results/phase3/v2_final/overall_summary.csv). 40 rows; grouped by available fields `task`, `model`, `metric`. Observed column values: `mean` -0.0016914 to 1.

### v2_reproduction

[Result directory](results/phase3/v2_reproduction/) · [validation.json](results/phase3/v2_reproduction/validation.json) · [protocol_lock.json](results/phase3/v2_reproduction/protocol_lock.json)

- [impact_summary.csv](results/phase3/v2_reproduction/impact_summary.csv). 8,460 rows; grouped by available fields `task`, `condition`, `regime`, `model`, `method`, `metric`. Observed column values: `mean` -0.10405 to 1.
- [overall_summary.csv](results/phase3/v2_reproduction/overall_summary.csv). 40 rows; grouped by available fields `task`, `model`, `metric`. Observed column values: `mean` -0.0016914 to 1.

### weather_reproduction_v2

[Result directory](results/phase3/weather_reproduction_v2/) · [run_contract.json](results/phase3/weather_reproduction_v2/run_contract.json) · [run_summary.json](results/phase3/weather_reproduction_v2/run_summary.json) · [protocol_lock.json](results/phase3/weather_reproduction_v2/protocol_lock.json)

- [batch_metrics.csv](results/phase3/weather_reproduction_v2/evaluator/batch_metrics.csv). 21 rows; grouped by available fields `batch`. Observed column values: `accuracy` 0.678 to 0.826, `macro_f1` 0.58908 to 0.72422, `balanced_accuracy` 0.5919 to 0.71575.

## Reproduction inputs and storage

- **Core Phase 3 v2.** The final and reproduction summaries, protocol locks and [historical comparison receipt](reports/phase3/v2_reproduction/reproduction_comparison.json) are published. Use the [v2 guide](docs/phase3/23I-2581_23I-2604_Phase3_Reproduction.md) and [runner](scripts/run_phase3_v2.py) to regenerate two full run directories before using the exact comparator. The comparator expects all original CSVs and NPZ arrays on both sides. The GitHub snapshot is a compact reference for examining published results.
- **UCI224.** The ten [frozen prediction batches](results/phase3/external_uci224_day2_v2/) allow the [delayed replay](scripts/run_uci224_delayed_replay.py) and [exact interval analysis](scripts/evaluate_uci224_exact_interval_extension.py) to use the same model outputs. For a fresh source evaluation, place the original dataset archive at the path declared in [the evaluator](scripts/evaluate_uci224_day2.py) and verify the SHA-256 recorded in the [replay protocol](docs/phase3/UCI224_DELAYED_REPLAY_PROTOCOL_20261116.md).
- **ELEC2.** The published [temporal run](results/phase3/elec2_temporal_20261116_v2/) includes the frozen predictions and evaluator truth required by the isotonic and Platt-style recalibration runners. For original stream preparation see the [dataset downloader](scripts/download_datasets.py) and [data manifests](data/manifests/).
- **Earlier Phase 3 snapshots.** Historical protocols can reference local-only original archives and full traces. If comparing with a historical run, obtain that exact input set or regenerate it from its own locked protocol. Do not substitute differently scoped results and call the comparison an exact reproduction.
