# Localization audit: 30 evaluation seeds

This post-hoc diagnostic evaluates the dimension/cluster weakness in the strengthening specification. It uses the locked 30-seed evaluation bank but does not retune any model or threshold using monitoring outcomes. Raw results, evaluator mappings, seed-bootstrap tables, figures and report TeX are retained in the linked result/report directories. Result and report receipts bind source, configuration, input and output hashes.

## Methods

For the five-region task, MiniBatchKMeans models with 3, 5 and 10 clusters were fitted using the same historical training partition as the frozen predictor. A one-to-one Hungarian cluster-to-region mapping was fitted from historical training feature assignments and generating-region IDs only. This fixed map was applied to each monitoring stream. Outcomes include adjusted Rand index, mapped accuracy/macro-F1, coverage when cluster counts exceed true regions, and precision/recall for the intervened region 2. The monitor's generating-region truth was not used to select or remap clusters.

For dependence localization, a maximum absolute pairwise-correlation-change score selects one top pair per 500-row cohort. Its alarm threshold is a per-seed 95th percentile calibrated from historical reference data against pseudo-current cohorts sampled from the distinct historical calibration partition. The held-out intervention changes pair (0,1) while preserving the designed marginal distributions. We also include stationary controls and pre-intervention cohorts. The seed, not the serial batch, is the bootstrap unit (2,000 resamples).

## Findings

- On the 30-seed linear task, dependence alarms fire on all active intervention batches (mean 1.000, 95% seed bootstrap interval 1.000–1.000); top-pair accuracy is also 1.000. Nonlinear XOR likewise alarms on all active batches with pair accuracy 1.000. The five-region task alarms on 0.955 of active batches and identifies the pair on 0.991 of active batches. Stationary alarm rates range from 0.054 to 0.066 across the tasks.
- Region recovery is weak. With five clusters, mean adjusted Rand index is 0.088 (95% CI 0.080–0.097), mapped accuracy is 0.356, and macro-F1 is 0.354. Region-2 recall during abrupt reversal is 0.216 (CI 0.207–0.225), approximately its 20% prevalence; precision is 0.223. The 3-cluster setting maps only three of five regions; the 10-cluster one-to-one mapping leaves roughly 48% of monitoring assignments unmapped. These results reject treating the configured cluster ID as a trustworthy generating-region label.

## Limits and implications

The dependence result is a targeted synthetic positive control, not evidence that all dependence changes are localizable. The regional task shows that clusters do not recover the known intervention region with useful accuracy under this representation and k-means design. A coordinated dashboard must display region occupancy and ambiguity as descriptive evidence and must not imply causal responsibility or reliable region identity. No claim about natural-domain localization, RQ3 decision benefit or analyst performance is supported.

## Reproduction

From the project root:

```powershell
C:\Users\hp\anaconda3\python.exe scripts\run_localization_audit.py --output results\phase3\localization_audit_20261117_reproduction_v2
C:\Users\hp\anaconda3\python.exe scripts\summarize_localization_audit.py --run results\phase3\localization_audit_20261117_reproduction_v2 --output reports\phase3\localization_audit_20261117_reproduction_v2
```

The scripts refuse to overwrite versioned output directories. Cluster mappings are fitted on historical training rows; pair thresholds use only historical inputs.
