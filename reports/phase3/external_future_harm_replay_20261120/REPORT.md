# External future accuracy and harm replay

This follow-up asks whether existing frozen estimates or already released labels provide a useful forecast of upcoming accuracy on three previously studied streams. The [locked design](../../../docs/phase3/EXTERNAL_FUTURE_HARM_REPLAY_20261120.md) fixes four rules before execution, with no fit to any of the external outcomes.

## Five-cohort accuracy MAE

Lower values are better. The table uses all valid forecast origins. When release-based methods had no labels, they used the historical reference. Delays count acquisitions or batches, so their elapsed time and number of rows differ by stream.

| Stream | Delay | Origins with labels / all | Historical | Current monitor | Latest release | Mean of last three |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ELEC2 | 0 | 49 / 50 | 0.0943 | 0.0435 | 0.0599 | 0.0614 |
| ELEC2 | 2 | 47 / 50 | 0.0943 | 0.0435 | 0.0710 | 0.0692 |
| ELEC2 | 5 | 44 / 50 | 0.0943 | 0.0435 | 0.0788 | 0.0757 |
| NOAA | 0 | 15 / 16 | 0.0329 | 0.0527 | 0.0410 | 0.0337 |
| NOAA | 2 | 13 / 16 | 0.0329 | 0.0527 | 0.0362 | 0.0311 |
| NOAA | 5 | 10 / 16 | 0.0329 | 0.0527 | 0.0239 | 0.0216 |
| UCI224 | 0 | 3 / 4 | 0.5529 | 0.5854 | 0.3161 | 0.3060 |
| UCI224 | 2 | 1 / 4 | 0.5529 | 0.5854 | 0.4677 | 0.4677 |
| UCI224 | 5 | 0 / 4 | 0.5529 | 0.5854 | 0.5529 | 0.5529 |

- **ELEC2.** The current monitor delivered the lowest MAE at each delay. It flagged 39 of 40 realized harm origins, and it also flagged all ten origins without realized harm. Low accuracy error here does not establish a discriminating decision rule.
- **NOAA.** Released-label methods had their lowest MAE at delay 5 in this replay. The three-cohort released mean still missed all four realized five-cohort harm origins at that delay. Higher delay paired the predictions with different past cohorts, so the lower observed error does not imply that delaying labels improves the method.
- **UCI224.** The three-cohort released mean had the lowest MAE at delay 0 and missed one of four realized harm origins. Five-cohort outcomes covered only four forecast origins in total. At delay 5 the strict release clock supplied zero labels at all four origins, making released-label methods equivalent to their historical fallback.

Restricting evaluation to origins with released labels even at the longest delay leaves 44 five-cohort origins on ELEC2, ten on NOAA and zero on UCI224. On this subset at delay 5, the monitor had MAE 0.0453 on ELEC2 and the three-cohort released mean had MAE 0.0265 on NOAA. The empty UCI224 subset is an availability result.

These are descriptive results from three already studied streams with overlapping targets. No independent domain validation or confidence interval is implied. This experiment compares point forecasts and deterministic harm decisions, rather than the synthetic study's probability forecasts.

## Reproduce and inspect

From the repository root, run the standard-library script using a new path:

    python scripts/run_external_future_harm_replay.py --out results/phase3/new_external_forecast

The same-environment rerun reproduced all three scientific CSVs byte for byte. All 456 unavailable-outcome perturbation checks passed. A separate comparison checked the release clock for all 1,824 forecast rows.

- [All metrics, including one-cohort horizons and the common subset](../../../results/phase3/external_future_harm_replay_20261120/summary.csv)
- [Forecasts](../../../results/phase3/external_future_harm_replay_20261120/online_forecasts.csv) and [separate evaluator targets](../../../results/phase3/external_future_harm_replay_20261120/evaluator_targets.csv)
- [Protocol and source hashes](../../../results/phase3/external_future_harm_replay_20261120/protocol_lock.json) and [reproduction receipt](../../../results/phase3/external_future_harm_replay_20261120/reproduction_comparison.json)
