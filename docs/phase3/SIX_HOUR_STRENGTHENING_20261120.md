# Phase 3 strengthening plan · 20 November 2026

The prior five-batch harm forecast used one previously unseen conditional law. This extension checks whether its gains carry across three different designed mechanisms, with a stationary negative control and the original C7 design as a benchmark on fresh seeds. All decisions below are fixed before examining the extension’s outcomes.

| Work | Maximum wall time | Deliverable |
| --- | ---: | --- |
| Specify protocol and check inputs | 30 min | This plan, input and source hashes, and a two-seed end-to-end smoke run |
| Locked synthetic evaluation | 4 h | 80 fresh seeds, five paired conditions, three release delays and 19 forecast origins |
| Independent reproduction checks on UCI224 and ELEC2 | 45 min | Compare regenerated replay outputs with the published evidence, recording any mismatch |
| Audit and report | 45 min | Run status, seed-level contrasts, simultaneous intervals and a concise result report |

Stop when these deliverables are complete. Use fresh output directories and at most three worker processes, each limited to one numerical thread. Preserve unfinished outputs if the wall budget is exhausted. Take no additional looks to tune the design or add seeds based on the outcomes.

## Locked forecast design

- Use the existing forecast training data, feature sets and three logistic models from the C7 protocol. Fit on seeds 950000–950029 only. Compare the time-only model, unlabeled monitor model, and full released-label model. Report development prevalence and last released harm as secondary baselines. Use seeds 972100–972179 for evaluation, paired across conditions, with exactly 20 onsets each at batches 5, 9, 13 and 17. Smoke seeds 972000–972001 do not enter the final analysis.
- Use 24 cohorts of 500 observations per stream with the original reference, scaler, classifier and calibrator training procedure. Predict whether average accuracy over batches t+1 through t+5 is more than five percentage points below that seed’s reference accuracy. Evaluate t=1 through 19 at delays 0, 5 and 20 using the strict rule `source_batch + delay < t`.
- Conditions share innovations and reference within seed. Stationary has no change. C7 flattens the posterior toward 0.5 with mixing weight 0.70. Regional reversal replaces p with 1−p for x0 ≥ 0 from onset, keeping input data unchanged. Boundary concentration multiplies x0, x1 and x2 by 0.25 from onset before generating labels from the original posterior law. Sensor sign inversion negates the observed x0 from onset after the labels are generated from clean inputs. These mechanisms are designed counterexamples and do not constitute natural-domain validation.
- Primary outcome is paired seed mean Brier gain, positive when the enhanced model has lower loss. Compare monitor versus time-only and full versus monitor per condition and delay. Use a common seed resample across all 30 contrasts and the 95th percentile of the maximum absolute centered bootstrap deviation for simultaneous intervals (5,000 replicates). Report empirical prevalence, Brier score and AUROC by condition, delay and model. AUROC is undefined for single-class groups. Report all strata, with equal seed weighting and no pooling across scenarios.
- Audit complete combinations, disjoint seeds, release availability, and separation of evaluator targets from online features. Recompute a subset after mutating all truth unavailable at an origin and verify online features are identical. Publish a protocol lock, validation receipt, compact aggregate and seed tables. Keep raw forecast tables in the run directory for replication.

## Replay checks

Rebuild the UCI224 delayed-label replay from the archived source and the ten frozen prediction batches. Rebuild ELEC2 Platt recalibration from the published temporal predictions and evaluator truth. Compare common scientific tables byte for byte where possible and report exact file matches separately from any differing output. These checks test repeatability of the existing studies. They do not add independent data.
