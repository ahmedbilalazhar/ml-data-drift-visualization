# NOAA Weather provenance, contract and evaluation audit

**Version:** addendum v2, 17 November 2026  
**Dataset:** NOAA GSOD station 72554014949 (Offutt AFB), 1949–1999  
**Research use:** chronological same-day rain/drizzle classification and feature monitoring; not rain forecasting.

## Source and target

The stream was rebuilt from 51 official NOAA/NCEI annual station CSV files. NOAA identifies GSOD as a daily global summary product ([NCEI metadata record](https://www.ncei.noaa.gov/metadata/geoportal/rest/metadata/item/gov.noaa.ncdc:C00516/html)). The station, annual inventory, file digests, date endpoints, and processing choices are recorded in `data/processed/weather_noaa_1949_1999_v2/weather_manifest.json`.

Eight same-day numeric predictors are retained: temperature, dew point, sea-level pressure, visibility, wind speed, maximum wind speed, daily maximum temperature, and daily minimum temperature. The outcome is the second digit of NOAA's six-position FRSHTT event code, indicating rain/drizzle observed on that summary date. Precipitation amount, snow depth and all FRSHTT event fields are excluded from predictors. Rows with missing selected predictors or invalid flags are dropped. A complete chronological sort precedes stable zero-based row-ID assignment.

This is a retrospective classification exercise because the features and rain flag describe the same daily summary. It is not a prospective forecast. No independently verified drift-onset labels exist, so alarm fraction is not reported as a false-alarm rate.

## Legacy benchmark reconciliation

The common un-timestamped scikit-multiflow benchmark mirror has 18,159 records. A greedy monotone match on eight one-decimal feature values finds 18,072 matches against the official NOAA rows; 87 are unmatched and 45 matched rows disagree on the rain label. This diagnostic does not prove unique row correspondence. We therefore use the verified timestamped NOAA data and labels as the evaluation source, and do not claim exact reconstruction of the un-timestamped mirror. The comparison is recorded in `data/external/noaa_gsod_weather_1949_1999_v1/benchmark_comparison_20261117.json`.

## Split and run

The admitted data contain 18,210 rows, including 5,746 positive rain labels. Chronological partitions contain 3,642 train, 1,821 calibration, 1,821 reference, and 10,926 monitoring rows. Twenty-one complete 500-row monitoring batches (10,500 rows) are evaluated; the final 426 monitoring rows are disclosed and excluded from complete-batch metrics.

The classifier and feature artifacts remain frozen. Online scoring receives features, IDs and timestamps; scores are persisted before the label broker releases labels at delay zero. The batch score file contains no truth or outcome metrics. Labels are joined by stable IDs for evaluator-only verification. This transfer run has one station and one chronological holdout; its 21 batches are serial temporal blocks, not independent replications.

The clean reproduction in `data/processed/weather_reproduction_v2/`, `results/phase3/weather_reproduction_v2/`, and `reports/phase3/weather_reproduction_v2/` regenerates the processed data with the same SHA-256 and completes preprocessing, scoring, verification, figure generation and TeX/PDF compilation.

## Results and limitations

The 21 monitoring batches have mean accuracy 0.751, macro-F1 0.661 and balanced accuracy 0.654. The frozen isotonic estimate has mean absolute error 0.044 and signed bias +0.042; historical-constant MAE is 0.044 and slightly better (0.044 rounded; exact metrics are in evaluator CSVs). Plug-in interval coverage is 9/21 (0.429), which is weak and descriptive at this sample size. The input alarm fraction is 1.0 and cannot be interpreted as false-alarm rate without onset truth. The stream does not establish general transfer, causal attribution, decision benefit, natural future-risk calibration or weather forecasting.

The first v1 preprocessing intermediate is retained but rejected: it produced only 272 records due to mishandling sentinel values. It is explicitly marked in `data/processed/weather_noaa_1949_1999_v1/REJECTED_INTERMEDIATE.md` and is not consumed by this evaluation. The corrected v2 parser only treats exactly 999.9 (or non-finite values) as missing, preserving valid pressure readings above 999.9.

The broader dataset audit initially encountered Parquet read errors (`Repetition level histogram size mismatch`) in the older Elec2 and Covertype processed files. They were rebuilt into a separate verified v2 namespace from the same checksummed raw sources, read back, and loaded successfully. The old files remain preserved and unused by the current loaders. The all-dataset audit now passes; see `data/interim/audit_all_datasets_20261117.json`.

## Reproduction

From the project root in PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/reproduce_noaa_weather.ps1
```

The script refuses to overwrite its versioned outputs. The successful isolated outputs for this run are under the `*_reproduction_v2` directories. Main evaluation output, report source and PDF are under `results/phase3/noaa_weather_1949_1999_v1/` and `reports/phase3/weather_noaa_1949_1999_v2/`.
