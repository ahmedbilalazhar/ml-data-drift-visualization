# Phase 3 forecast stress study and replay audit

This extension tests the existing five-batch harm forecast on three designed mechanisms beyond the original C7 experiment. It also checks reproducibility on the existing UCI224 and ELEC2 external replays. See the [locked plan](../../../docs/phase3/SIX_HOUR_STRENGTHENING_20261120.md).

## What the new experiment showed

The evaluation completed 80 fresh seeds, five paired conditions, three simulated label delays, and 19 origins per stream. It generated 22,800 forecasts per model and used the seed as the resampling unit. The release clock and all 22,800 unavailable-outcome mutation checks passed. The full evaluation took 178 seconds on this laptop. A second complete run on the same seeds and environment reproduced all six scientific CSVs byte for byte. See the [comparison receipt](../../../results/phase3/six_hour_forecast_20261120/reproduction_comparison.json).

| Evaluation condition | Time-only Brier at delay 0 | Unlabeled monitor Brier at delay 0 | Gain with simultaneous 95% interval |
| --- | ---: | ---: | --- |
| Stationary | 0.1869 | 0.2755 | −0.0886 [−0.1223, −0.0549] |
| C7 conditional flattening | 0.2347 | 0.1900 | +0.0447 [+0.0110, +0.0785] |
| Regional label reversal | 0.2404 | 0.1950 | +0.0453 [+0.0116, +0.0791] |
| Boundary concentration | 0.2346 | 0.5683 | −0.3337 [−0.3674, −0.2999] |
| Sensor sign inversion | 0.2414 | 0.2643 | −0.0229 [−0.0567, +0.0108] |

Positive gain means lower Brier loss for the enhanced model. The monitor versus time-only effects round to the same figures at delays 5 and 20. The full model improved on the unlabeled monitor in the stationary control at delays 0 and 5. Across the four shifted conditions, all 12 full-versus-monitor intervals included zero.

The two conditional-shift scenarios used the exact same observed inputs as the stationary control and produced identical unlabeled forecast probabilities at all 4,560 matched origins per condition. Their lower Brier scores therefore reflect forecast loss against different realized outcomes. They provide no evidence that the monitor separated conditional shift from stationarity. Stationary harm prevalence was zero in this sample, so stationary AUROC is undefined. Time-only AUROC was approximately 0.87 on the shifted designs, consistent with onset timing remaining informative even when balanced across four onset batches. The boundary-concentration failure shows why the trained monitor's probabilities cannot be transferred across arbitrary mechanisms.

## External replay audit

- **UCI224.** The one-thread run stopped at the strict confidence equality guard. Across 13,465 monitoring rows, the refit produced zero class mismatches and eight confidence differences, with maximum magnitude `9.99e-16`. With the interpreter's default thread configuration the existing runner completed and its evaluator passed. The published aggregate summary and release audits matched exactly. One per-batch table differed numerically by at most `1.94e-16`, and six prediction arrays differed by at most `3.33e-16`. There were 45 matching value sets across the 52 compared scientific files. The original strict guard was preserved.
- **ELEC2.** The Platt replay completed. Of 168 fit-audit CSVs and probability archives, 12 matched byte for byte and in values. The remaining probability arrays differed by at most `3.33e-16`. Both replays use previously studied streams and do not add new external validation data.

## Reproduce and inspect

From the repository root, use the interpreter and package versions in the run lock. This run used Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, pandas 2.3.3 and scikit-learn 1.8.0. Write to unused output paths.

```powershell
python scripts/run_phase3_six_hour_forecast.py --workers 3 --max-seconds 14400 --out results/phase3/new_forecast_run
python scripts/run_uci224_delayed_replay.py --source-run results/phase3/external_uci224_day2_v2 --out results/phase3/new_uci_replay
python scripts/evaluate_uci224_delayed_replay.py --run results/phase3/new_uci_replay
python scripts/run_elec2_platt_recalibration.py --run results/phase3/elec2_temporal_20261116_v2 --out results/phase3/new_elec2_replay
```

- [All forecast metrics](../../../results/phase3/six_hour_forecast_20261120/metrics_by_family_delay.csv)
- [All 30 simultaneous contrasts](../../../results/phase3/six_hour_forecast_20261120/simultaneous_contrasts.csv)
- [Paired seed gains](../../../results/phase3/six_hour_forecast_20261120/paired_seed_gains.csv)
- [Protocol lock](../../../results/phase3/six_hour_forecast_20261120/protocol_lock.json) and [validation receipt](../../../results/phase3/six_hour_forecast_20261120/validation.json)
- [External replay audit](../../../results/phase3/six_hour_replay_audit_20261120/replay_audit.json) and [UCI refit differences](../../../results/phase3/six_hour_replay_audit_20261120/uci224_refit_differences.csv)

The published snapshot includes the compact result tables. The runner regenerates the row-level online features, evaluator targets and forecast probabilities listed in the validation receipt. The protocol lock records bytes from the execution environment. Git line-ending conversion can change file hashes without changing the contents of text inputs.
